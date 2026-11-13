#include "tpuforge/pjrt_driver.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tpuforge {

namespace {

#define TPUF_API_HAS(api_ptr, field) \
    ((api_ptr) != nullptr && \
     (api_ptr)->struct_size >= (offsetof(PJRT_Api, field) + sizeof((api_ptr)->field)) && \
     (api_ptr)->field != nullptr)

std::string consume_pjrt_error(const PJRT_Api* api, PJRT_Error* err) {
    if (!err) {
        return {};
    }
    std::string message = "Unknown PJRT error";
    if (TPUF_API_HAS(api, PJRT_Error_Message)) {
        PJRT_Error_Message_Args margs{};
        margs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        margs.error = err;
        api->PJRT_Error_Message(&margs);
        if (margs.message && margs.message_size > 0) {
            message.assign(margs.message, margs.message_size);
        }
    }
    if (TPUF_API_HAS(api, PJRT_Error_Destroy)) {
        PJRT_Error_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        dargs.error = err;
        api->PJRT_Error_Destroy(&dargs);
    }
    return message;
}

bool await_and_destroy_event(const PJRT_Api* api, PJRT_Event* event, std::string* error_out) {
    if (!event) {
        return true;
    }
    std::string await_err;
    if (TPUF_API_HAS(api, PJRT_Event_Await)) {
        PJRT_Event_Await_Args wargs{};
        wargs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        wargs.event = event;
        if (PJRT_Error* err = api->PJRT_Event_Await(&wargs)) {
            await_err = consume_pjrt_error(api, err);
        }
    }
    if (TPUF_API_HAS(api, PJRT_Event_Destroy)) {
        PJRT_Event_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        dargs.event = event;
        if (PJRT_Error* err = api->PJRT_Event_Destroy(&dargs)) {
            std::string derr = consume_pjrt_error(api, err);
            if (await_err.empty()) {
                await_err = std::move(derr);
            }
        }
    }
    if (!await_err.empty()) {
        if (error_out) {
            *error_out = std::move(await_err);
        }
        return false;
    }
    return true;
}

struct BufferGuard {
    const PJRT_Api* api{nullptr};
    PJRT_Buffer* buf{nullptr};

    BufferGuard() = default;
    BufferGuard(const PJRT_Api* a, PJRT_Buffer* b) noexcept : api(a), buf(b) {}
    BufferGuard(const BufferGuard&) = delete;
    BufferGuard& operator=(const BufferGuard&) = delete;
    BufferGuard(BufferGuard&& other) noexcept
        : api(other.api), buf(std::exchange(other.buf, nullptr)) {}
    BufferGuard& operator=(BufferGuard&& other) noexcept {
        if (this != &other) {
            reset();
            api = other.api;
            buf = std::exchange(other.buf, nullptr);
        }
        return *this;
    }

    void reset() noexcept {
        if (buf && TPUF_API_HAS(api, PJRT_Buffer_Destroy)) {
            PJRT_Buffer_Destroy_Args args{};
            args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            args.buffer = buf;
            if (PJRT_Error* err = api->PJRT_Buffer_Destroy(&args)) {
                (void)consume_pjrt_error(api, err);
            }
        }
        buf = nullptr;
    }

    ~BufferGuard() { reset(); }
};

struct EventGuard {
    const PJRT_Api* api{nullptr};
    PJRT_Event* event{nullptr};

    EventGuard() = default;
    EventGuard(const PJRT_Api* a, PJRT_Event* e) noexcept : api(a), event(e) {}
    EventGuard(const EventGuard&) = delete;
    EventGuard& operator=(const EventGuard&) = delete;

    ~EventGuard() {
        if (event && TPUF_API_HAS(api, PJRT_Event_Destroy)) {
            PJRT_Event_Destroy_Args args{};
            args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
            args.event = event;
            if (PJRT_Error* err = api->PJRT_Event_Destroy(&args)) {
                (void)consume_pjrt_error(api, err);
            }
        }
    }

    PJRT_Event* release() {
        PJRT_Event* e = event;
        event = nullptr;
        return e;
    }
};

#if defined(_WIN32)
std::string win32_error_message(DWORD code) {
    wchar_t* buf = nullptr;
    DWORD flags = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                  FORMAT_MESSAGE_IGNORE_INSERTS;
    DWORD chars = FormatMessageW(
        flags, nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPWSTR>(&buf), 0, nullptr
    );
    if (chars == 0 || !buf) {
        return "Win32 error code " + std::to_string(code);
    }
    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(chars), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(utf8_len), '\0');
    if (utf8_len > 0) {
        WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(chars), out.data(), utf8_len, nullptr, nullptr);
    }
    LocalFree(buf);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

bool utf8_to_wide(const std::string& utf8, std::wstring* wide_out) {
    if (utf8.empty()) {
        wide_out->clear();
        return true;
    }
    if (utf8.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    int needed = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0
    );
    if (needed <= 0) {
        return false;
    }
    wide_out->assign(static_cast<size_t>(needed), L'\0');
    int converted = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), static_cast<int>(utf8.size()),
        wide_out->data(), needed
    );
    wide_out->resize(static_cast<size_t>(converted));
    return true;
}
#endif

void close_native_library(void* handle) {
    if (!handle) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

bool validate_required_api_slots(const PJRT_Api* api, std::string* error_out) {
    if (!api) {
        if (error_out) *error_out = "GetPjrtApi() returned null PJRT_Api pointer";
        return false;
    }
    if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
        if (error_out) {
            *error_out = "Incompatible PJRT API major version: plugin=" +
                         std::to_string(api->pjrt_api_version.major_version) +
                         " host=" + std::to_string(PJRT_API_MAJOR);
        }
        return false;
    }

    const bool has_required =
        TPUF_API_HAS(api, PJRT_Error_Destroy) &&
        TPUF_API_HAS(api, PJRT_Error_Message) &&
        TPUF_API_HAS(api, PJRT_Plugin_Initialize) &&
        TPUF_API_HAS(api, PJRT_Event_Destroy) &&
        TPUF_API_HAS(api, PJRT_Event_Await) &&
        TPUF_API_HAS(api, PJRT_Client_Create) &&
        TPUF_API_HAS(api, PJRT_Client_Destroy) &&
        TPUF_API_HAS(api, PJRT_Client_PlatformName) &&
        TPUF_API_HAS(api, PJRT_Client_Devices) &&
        TPUF_API_HAS(api, PJRT_Client_Compile) &&
        TPUF_API_HAS(api, PJRT_Client_BufferFromHostBuffer) &&
        TPUF_API_HAS(api, PJRT_Device_GetDescription) &&
        TPUF_API_HAS(api, PJRT_DeviceDescription_Id) &&
        TPUF_API_HAS(api, PJRT_DeviceDescription_Kind) &&
        TPUF_API_HAS(api, PJRT_LoadedExecutable_Destroy) &&
        TPUF_API_HAS(api, PJRT_LoadedExecutable_Execute) &&
        TPUF_API_HAS(api, PJRT_Buffer_Destroy) &&
        TPUF_API_HAS(api, PJRT_Buffer_ToHostBuffer);

    if (!has_required) {
        if (error_out) {
            *error_out = "PJRT plugin table is missing one or more required C API functions or struct_size is too small";
        }
        return false;
    }
    return true;
}

bool upload_host_buffer(
    const PJRT_Api* api,
    PJRT_Client* client,
    PJRT_Device* device,
    const void* data,
    PJRT_Buffer_Type dtype,
    const int64_t* dims,
    size_t num_dims,
    PJRT_Buffer** out_buffer,
    std::string* error_out
) {
    PJRT_Client_BufferFromHostBuffer_Args bargs{};
    bargs.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    bargs.client = client;
    bargs.data = data;
    bargs.type = dtype;
    bargs.dims = dims;
    bargs.num_dims = num_dims;
    bargs.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    bargs.device = device;

    if (PJRT_Error* err = api->PJRT_Client_BufferFromHostBuffer(&bargs)) {
        if (error_out) {
            *error_out = "PJRT_Client_BufferFromHostBuffer failed: " + consume_pjrt_error(api, err);
        } else {
            (void)consume_pjrt_error(api, err);
        }
        return false;
    }
    if (!await_and_destroy_event(api, bargs.done_with_host_buffer, error_out)) {
        if (bargs.buffer && TPUF_API_HAS(api, PJRT_Buffer_Destroy)) {
            PJRT_Buffer_Destroy_Args dargs{};
            dargs.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
            dargs.buffer = bargs.buffer;
            if (PJRT_Error* derr = api->PJRT_Buffer_Destroy(&dargs)) {
                (void)consume_pjrt_error(api, derr);
            }
        }
        return false;
    }
    *out_buffer = bargs.buffer;
    if (!*out_buffer) {
        if (error_out) {
            *error_out = "PJRT_Client_BufferFromHostBuffer succeeded but returned null buffer";
        }
        return false;
    }
    return true;
}

bool download_f32_2d_buffer(
    const PJRT_Api* api,
    PJRT_Buffer* src,
    size_t expected_rows,
    size_t expected_cols,
    float* dst,
    const char* label,
    std::string* error_out
) {
    if (!src) {
        if (error_out) {
            *error_out = std::string("PJRT_LoadedExecutable_Execute returned null buffer for ") + label;
        }
        return false;
    }
    if (TPUF_API_HAS(api, PJRT_Buffer_ElementType)) {
        PJRT_Buffer_ElementType_Args targs{};
        targs.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
        targs.buffer = src;
        if (PJRT_Error* err = api->PJRT_Buffer_ElementType(&targs)) {
            if (error_out) *error_out = std::string("PJRT_Buffer_ElementType failed (") + label + "): " + consume_pjrt_error(api, err);
            else (void)consume_pjrt_error(api, err);
            return false;
        }
        if (targs.type != PJRT_Buffer_Type_F32) {
            if (error_out) {
                *error_out = std::string("Output ") + label + " buffer dtype mismatch: expected F32";
            }
            return false;
        }
    }
    if (TPUF_API_HAS(api, PJRT_Buffer_Dimensions)) {
        PJRT_Buffer_Dimensions_Args dargs{};
        dargs.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
        dargs.buffer = src;
        if (PJRT_Error* err = api->PJRT_Buffer_Dimensions(&dargs)) {
            if (error_out) *error_out = std::string("PJRT_Buffer_Dimensions failed (") + label + "): " + consume_pjrt_error(api, err);
            else (void)consume_pjrt_error(api, err);
            return false;
        }
        if (dargs.num_dims != 2 || !dargs.dims ||
            dargs.dims[0] != static_cast<int64_t>(expected_rows) ||
            dargs.dims[1] != static_cast<int64_t>(expected_cols)) {
            if (error_out) {
                *error_out = std::string("Output ") + label + " buffer shape mismatch";
            }
            return false;
        }
    }
    if (expected_cols > 0 && expected_rows > std::numeric_limits<size_t>::max() / expected_cols) {
        if (error_out) *error_out = std::string("Output ") + label + " element count overflow";
        return false;
    }
    const size_t expected_elems = expected_rows * expected_cols;
    if (expected_elems > std::numeric_limits<size_t>::max() / sizeof(float)) {
        if (error_out) *error_out = std::string("Output ") + label + " byte size overflow";
        return false;
    }
    const size_t expected_bytes = expected_elems * sizeof(float);

    PJRT_Buffer_ToHostBuffer_Args qargs{};
    qargs.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    qargs.src = src;
    qargs.dst = nullptr;
    qargs.dst_size = 0;
    if (PJRT_Error* err = api->PJRT_Buffer_ToHostBuffer(&qargs)) {
        if (error_out) *error_out = std::string("PJRT_Buffer_ToHostBuffer(size query, ") + label + ") failed: " + consume_pjrt_error(api, err);
        else (void)consume_pjrt_error(api, err);
        return false;
    }
    if (!await_and_destroy_event(api, qargs.event, error_out)) {
        return false;
    }
    if (qargs.dst_size != expected_bytes) {
        if (error_out) {
            *error_out = std::string("Output ") + label + " byte size mismatch: expected " +
                         std::to_string(expected_bytes) + ", got " + std::to_string(qargs.dst_size);
        }
        return false;
    }

    PJRT_Buffer_ToHostBuffer_Args hargs{};
    hargs.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    hargs.src = src;
    hargs.dst = dst;
    hargs.dst_size = expected_bytes;
    if (PJRT_Error* err = api->PJRT_Buffer_ToHostBuffer(&hargs)) {
        if (error_out) *error_out = std::string("PJRT_Buffer_ToHostBuffer(") + label + ") failed: " + consume_pjrt_error(api, err);
        else (void)consume_pjrt_error(api, err);
        return false;
    }
    return await_and_destroy_event(api, hargs.event, error_out);
}

struct StagedPluginGuard {
    const PJRT_Api* api{nullptr};
    PJRT_Client* client{nullptr};
    void* lib_handle{nullptr};

    StagedPluginGuard() = default;
    StagedPluginGuard(const PJRT_Api* a, PJRT_Client* c, void* h) noexcept
        : api(a), client(c), lib_handle(h) {}
    StagedPluginGuard(const StagedPluginGuard&) = delete;
    StagedPluginGuard& operator=(const StagedPluginGuard&) = delete;

    void dismiss() noexcept {
        api = nullptr;
        client = nullptr;
        lib_handle = nullptr;
    }

    ~StagedPluginGuard() {
        if (client && TPUF_API_HAS(api, PJRT_Client_Destroy)) {
            PJRT_Client_Destroy_Args dargs{};
            dargs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
            dargs.client = client;
            if (PJRT_Error* err = api->PJRT_Client_Destroy(&dargs)) {
                (void)consume_pjrt_error(api, err);
            }
        }
        close_native_library(lib_handle);
    }
};

bool query_client_devices(
    const PJRT_Api* api,
    PJRT_Client* client,
    PJRT_Device* const** out_devices,
    size_t* out_count,
    std::string* error_out
) {
    if (TPUF_API_HAS(api, PJRT_Client_AddressableDevices)) {
        PJRT_Client_AddressableDevices_Args aargs{};
        aargs.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        aargs.client = client;
        if (PJRT_Error* err = api->PJRT_Client_AddressableDevices(&aargs)) {
            if (error_out) {
                *error_out = "PJRT_Client_AddressableDevices failed: " + consume_pjrt_error(api, err);
            } else {
                (void)consume_pjrt_error(api, err);
            }
            return false;
        }
        if (aargs.num_addressable_devices > 0 && aargs.addressable_devices) {
            *out_devices = aargs.addressable_devices;
            *out_count = aargs.num_addressable_devices;
            return true;
        }
    }

    PJRT_Client_Devices_Args dargs{};
    dargs.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    dargs.client = client;
    if (PJRT_Error* err = api->PJRT_Client_Devices(&dargs)) {
        if (error_out) {
            *error_out = "PJRT_Client_Devices failed: " + consume_pjrt_error(api, err);
        } else {
            (void)consume_pjrt_error(api, err);
        }
        return false;
    }
    *out_devices = dargs.devices;
    *out_count = dargs.num_devices;
    return true;
}

} // namespace

PjrtDriver::~PjrtDriver() {
    std::unique_lock lk(mu_);
    {
        std::scoped_lock reg_lk(exec_reg_mu_);
        for (ZeroRecompileExecutor* exec : executors_) {
            if (exec) {
                exec->invalidate_with_api(api_, /*detach_driver=*/true);
            }
        }
        executors_.clear();
    }
    if (client_ && TPUF_API_HAS(api_, PJRT_Client_Destroy)) {
        PJRT_Client_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        dargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_Destroy(&dargs)) {
            (void)consume_pjrt_error(api_, err);
        }
    }
    client_ = nullptr;
    api_ = nullptr;
    loaded_path_.clear();
    devices_.clear();
    platform_name_.clear();
    close_native_library(lib_handle_);
    lib_handle_ = nullptr;
}

bool PjrtDriver::load_plugin(const std::string& library_path, std::string* error_out) {
    if (library_path.empty()) {
        if (error_out) {
            *error_out = "Empty PJRT plugin library path";
        }
        return false;
    }

    // Resolve target library path first so same-plugin reloads can release exclusive TPU ownership.
    std::string resolved_path = library_path;
#if defined(_WIN32)
    std::wstring wide_path;
    if (!utf8_to_wide(library_path, &wide_path)) {
        if (error_out) {
            *error_out = "Invalid UTF-8 in PJRT plugin path";
        }
        return false;
    }
    DWORD abs_len = GetFullPathNameW(wide_path.c_str(), 0, nullptr, nullptr);
    if (abs_len == 0) {
        if (error_out) {
            *error_out = "Failed to resolve PJRT plugin path '" + library_path +
                         "': " + win32_error_message(GetLastError());
        }
        return false;
    }
    std::wstring abs_wide_path(static_cast<size_t>(abs_len), L'\0');
    DWORD written = GetFullPathNameW(wide_path.c_str(), abs_len, abs_wide_path.data(), nullptr);
    if (written == 0 || written >= abs_len) {
        if (error_out) {
            *error_out = "Failed to resolve full PJRT plugin path '" + library_path + "'";
        }
        return false;
    }
    abs_wide_path.resize(static_cast<size_t>(written));
    int r_utf8_len = WideCharToMultiByte(
        CP_UTF8, 0, abs_wide_path.data(), static_cast<int>(abs_wide_path.size()), nullptr, 0, nullptr, nullptr
    );
    if (r_utf8_len > 0) {
        resolved_path.assign(static_cast<size_t>(r_utf8_len), '\0');
        WideCharToMultiByte(
            CP_UTF8, 0, abs_wide_path.data(), static_cast<int>(abs_wide_path.size()),
            resolved_path.data(), r_utf8_len, nullptr, nullptr
        );
    }
#endif

    {
        bool same_loaded = false;
        {
            std::shared_lock lk(mu_);
            same_loaded = (lib_handle_ != nullptr && api_ != nullptr && client_ != nullptr &&
                           !loaded_path_.empty() && loaded_path_ == resolved_path);
        }
        if (same_loaded) {
            unload();
        }
    }

    // Load and validate into local staging variables so failures keep prior state intact.
    void* next_handle = nullptr;
    const PJRT_Api* (*get_api_fn)(void) = nullptr;

#if defined(_WIN32)
    HMODULE mod = LoadLibraryExW(abs_wide_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!mod) {
        if (error_out) {
            *error_out = "LoadLibraryExW failed for '" + library_path + "': " +
                         win32_error_message(GetLastError());
        }
        return false;
    }
    next_handle = static_cast<void*>(mod);
    FARPROC sym = GetProcAddress(mod, "GetPjrtApi");
    if (!sym) {
        std::string err_msg = "Symbol GetPjrtApi not found in '" + library_path + "': " +
                              win32_error_message(GetLastError());
        close_native_library(next_handle);
        if (error_out) *error_out = std::move(err_msg);
        return false;
    }
    get_api_fn = reinterpret_cast<const PJRT_Api* (*)(void)>(sym);
#else
    int dl_flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
    dl_flags |= RTLD_NODELETE;
#endif
    next_handle = dlopen(library_path.c_str(), dl_flags);
    if (!next_handle) {
        if (error_out) {
            const char* dl_err = dlerror();
            *error_out = "dlopen failed for '" + library_path + "': " +
                         (dl_err ? dl_err : "unknown dlopen error");
        }
        return false;
    }
    dlerror();
    void* sym = dlsym(next_handle, "GetPjrtApi");
    const char* sym_err = dlerror();
    if (!sym || sym_err != nullptr) {
        std::string err_msg = "Symbol GetPjrtApi not found in '" + library_path + "': " +
                              (sym_err ? sym_err : "null symbol");
        close_native_library(next_handle);
        if (error_out) *error_out = std::move(err_msg);
        return false;
    }
    get_api_fn = reinterpret_cast<const PJRT_Api* (*)(void)>(sym);
#endif

    const PJRT_Api* next_api = get_api_fn();
    std::string val_err;
    if (!validate_required_api_slots(next_api, &val_err)) {
        close_native_library(next_handle);
        if (error_out) *error_out = std::move(val_err);
        return false;
    }

    PJRT_Plugin_Initialize_Args init_args{};
    init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    if (PJRT_Error* err = next_api->PJRT_Plugin_Initialize(&init_args)) {
        std::string msg = "PJRT_Plugin_Initialize failed: " + consume_pjrt_error(next_api, err);
        close_native_library(next_handle);
        if (error_out) *error_out = std::move(msg);
        return false;
    }

    PJRT_Client_Create_Args create_args{};
    create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    if (PJRT_Error* err = next_api->PJRT_Client_Create(&create_args)) {
        std::string msg = "PJRT_Client_Create failed: " + consume_pjrt_error(next_api, err);
        close_native_library(next_handle);
        if (error_out) *error_out = std::move(msg);
        return false;
    }
    PJRT_Client* next_client = create_args.client;
    if (!next_client) {
        close_native_library(next_handle);
        if (error_out) *error_out = "PJRT_Client_Create succeeded but returned null client";
        return false;
    }

    StagedPluginGuard staged{next_api, next_client, next_handle};

    PJRT_Client_PlatformName_Args pname_args{};
    pname_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
    pname_args.client = next_client;
    if (PJRT_Error* err = next_api->PJRT_Client_PlatformName(&pname_args)) {
        if (error_out) *error_out = "PJRT_Client_PlatformName failed: " + consume_pjrt_error(next_api, err);
        else (void)consume_pjrt_error(next_api, err);
        return false;
    }
    std::string next_platform;
    if (pname_args.platform_name && pname_args.platform_name_size > 0) {
        next_platform.assign(pname_args.platform_name, pname_args.platform_name_size);
    }

    PJRT_Device* const* raw_devices = nullptr;
    size_t raw_count = 0;
    std::string dev_err;
    if (!query_client_devices(next_api, next_client, &raw_devices, &raw_count, &dev_err)) {
        if (error_out) *error_out = std::move(dev_err);
        return false;
    }

    std::vector<DeviceInfo> next_devices;
    next_devices.reserve(raw_count);
    for (size_t i = 0; i < raw_count; ++i) {
        PJRT_Device* dev = raw_devices[i];
        if (!dev) {
            continue;
        }
        PJRT_Device_GetDescription_Args gdargs{};
        gdargs.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
        gdargs.device = dev;
        if (PJRT_Error* err = next_api->PJRT_Device_GetDescription(&gdargs)) {
            if (error_out) *error_out = "PJRT_Device_GetDescription failed: " + consume_pjrt_error(next_api, err);
            else (void)consume_pjrt_error(next_api, err);
            return false;
        }
        PJRT_DeviceDescription* desc = gdargs.device_description;
        if (!desc) {
            continue;
        }

        PJRT_DeviceDescription_Id_Args id_args{};
        id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
        id_args.device_description = desc;
        if (PJRT_Error* err = next_api->PJRT_DeviceDescription_Id(&id_args)) {
            if (error_out) *error_out = "PJRT_DeviceDescription_Id failed: " + consume_pjrt_error(next_api, err);
            else (void)consume_pjrt_error(next_api, err);
            return false;
        }

        PJRT_DeviceDescription_Kind_Args kind_args{};
        kind_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
        kind_args.device_description = desc;
        if (PJRT_Error* err = next_api->PJRT_DeviceDescription_Kind(&kind_args)) {
            if (error_out) *error_out = "PJRT_DeviceDescription_Kind failed: " + consume_pjrt_error(next_api, err);
            else (void)consume_pjrt_error(next_api, err);
            return false;
        }

        DeviceInfo info{};
        info.id = static_cast<int32_t>(id_args.id);
        if (kind_args.device_kind && kind_args.device_kind_size > 0) {
            info.kind.assign(kind_args.device_kind, kind_args.device_kind_size);
        }
        info.memory_bytes = 0;
        if (TPUF_API_HAS(next_api, PJRT_Device_MemoryStats)) {
            PJRT_Device_MemoryStats_Args mstats{};
            mstats.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
            mstats.device = dev;
            if (PJRT_Error* err = next_api->PJRT_Device_MemoryStats(&mstats)) {
                (void)consume_pjrt_error(next_api, err);
            } else if (mstats.bytes_limit_is_set && mstats.bytes_limit > 0) {
                info.memory_bytes = static_cast<size_t>(mstats.bytes_limit);
            }
        }
        info.is_tpu = (info.kind.find("TPU") != std::string::npos);
        const int32_t mxu = infer_mxu_dim(info.kind);
        info.mxu_dim_x = mxu;
        info.mxu_dim_y = mxu;
        next_devices.push_back(std::move(info));
    }

    // All staging checks succeeded: invalidate old executables & tear down old plugin, then commit.
    std::unique_lock lk(mu_);
    {
        std::scoped_lock reg_lk(exec_reg_mu_);
        for (ZeroRecompileExecutor* exec : executors_) {
            if (exec) {
                exec->invalidate_with_api(api_, /*detach_driver=*/false);
            }
        }
    }
    if (client_ && TPUF_API_HAS(api_, PJRT_Client_Destroy)) {
        PJRT_Client_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        dargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_Destroy(&dargs)) {
            (void)consume_pjrt_error(api_, err);
        }
    }
    close_native_library(lib_handle_);

    lib_handle_ = next_handle;
    api_ = next_api;
    client_ = next_client;
    loaded_path_ = std::move(resolved_path);
    platform_name_ = std::move(next_platform);
    devices_ = std::move(next_devices);
    staged.dismiss();
    return true;
}

void PjrtDriver::unload() {
    std::unique_lock lk(mu_);
    {
        std::scoped_lock reg_lk(exec_reg_mu_);
        for (ZeroRecompileExecutor* exec : executors_) {
            if (exec) {
                exec->invalidate_with_api(api_, /*detach_driver=*/false);
            }
        }
    }
    if (client_ && TPUF_API_HAS(api_, PJRT_Client_Destroy)) {
        PJRT_Client_Destroy_Args dargs{};
        dargs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        dargs.client = client_;
        if (PJRT_Error* err = api_->PJRT_Client_Destroy(&dargs)) {
            (void)consume_pjrt_error(api_, err);
        }
    }
    client_ = nullptr;
    api_ = nullptr;
    loaded_path_.clear();
    devices_.clear();
    platform_name_.clear();
    close_native_library(lib_handle_);
    lib_handle_ = nullptr;
}

bool PjrtDriver::is_loaded() const {
    std::shared_lock lk(mu_);
    return lib_handle_ != nullptr && api_ != nullptr && client_ != nullptr;
}

std::string PjrtDriver::platform_name() const {
    std::shared_lock lk(mu_);
    return platform_name_;
}

std::vector<DeviceInfo> PjrtDriver::query_devices() const {
    std::shared_lock lk(mu_);
    return devices_;
}

const PJRT_Api* PjrtDriver::api() const {
    std::shared_lock lk(mu_);
    return api_;
}

PJRT_Client* PjrtDriver::client() const {
    std::shared_lock lk(mu_);
    return client_;
}

void PjrtDriver::register_executor(ZeroRecompileExecutor* exec) {
    if (!exec) {
        return;
    }
    std::scoped_lock reg_lk(exec_reg_mu_);
    executors_.push_back(exec);
}

void PjrtDriver::unregister_executor(ZeroRecompileExecutor* exec) {
    if (!exec) {
        return;
    }
    std::scoped_lock reg_lk(exec_reg_mu_);
    executors_.erase(std::remove(executors_.begin(), executors_.end(), exec), executors_.end());
}

void PjrtDriver::unregister_and_destroy_executable(ZeroRecompileExecutor* exec) {
    if (!exec) {
        return;
    }
    std::shared_lock lk(mu_);
    {
        std::scoped_lock reg_lk(exec_reg_mu_);
        executors_.erase(std::remove(executors_.begin(), executors_.end(), exec), executors_.end());
    }
    exec->invalidate_with_api(api_, /*detach_driver=*/true);
}

ZeroRecompileExecutor::ZeroRecompileExecutor(PjrtDriver& driver)
    : driver_(&driver) {
    driver.register_executor(this);
}

ZeroRecompileExecutor::~ZeroRecompileExecutor() {
    PjrtDriver* drv = driver_.exchange(nullptr);
    if (drv) {
        drv->unregister_and_destroy_executable(this);
    }
}

void ZeroRecompileExecutor::invalidate_with_api(const PJRT_Api* api, bool detach_driver) {
    if (detach_driver) {
        driver_.store(nullptr);
    }
    std::scoped_lock exec_lk(mu_);
    if (executable_ && TPUF_API_HAS(api, PJRT_LoadedExecutable_Destroy)) {
        PJRT_LoadedExecutable_Destroy_Args dargs{};
        dargs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        dargs.executable = executable_;
        if (PJRT_Error* err = api->PJRT_LoadedExecutable_Destroy(&dargs)) {
            (void)consume_pjrt_error(api, err);
        }
    }
    executable_ = nullptr;
    num_outputs_ = 1;
    compiled_ = false;
    shape_key_ = StaticShapeKey{};
}

bool ZeroRecompileExecutor::compile_once(
    const StaticArenaConfig& config,
    const std::string& program_code,
    const std::string& format,
    std::string* error_out
) {
    if (!SystolicConstants::is_valid_tile_size(config.tile_size) ||
        config.vocab_size == 0 || config.max_seq_len < config.tile_size ||
        (config.max_seq_len % config.tile_size) != 0 ||
        config.hidden_dim == 0 || config.num_kv_heads == 0 || config.head_dim == 0) {
        if (error_out) {
            *error_out = "Invalid StaticArenaConfig dimensions for systolic compilation";
        }
        return false;
    }

    PjrtDriver* drv = driver_.load();
    if (!drv) {
        if (error_out) {
            *error_out = "ZeroRecompileExecutor has been detached from its PjrtDriver";
        }
        return false;
    }

    std::shared_lock drv_lk(drv->mu_);
    if (!drv->lib_handle_ || !drv->api_ || !drv->client_) {
        if (error_out) {
            *error_out = "PjrtDriver is not loaded";
        }
        return false;
    }

    // Normalize "stablehlo" alias to OpenXLA PJRT's canonical "mlir" format string.
    std::string effective_format = (format == "stablehlo") ? std::string("mlir") : format;

    StaticShapeKey next_key{
        config.tile_size,
        config.max_seq_len,
        config.hidden_dim,
        config.num_kv_heads,
        config.head_dim,
        config.vocab_size,
        effective_format,
        program_code,
    };

    std::scoped_lock exec_lk(mu_);
    if (compiled_ && executable_ != nullptr && shape_key_ == next_key) {
        return true;
    }

    const PJRT_Api* api = drv->api_;
    std::string mutable_code = program_code;
    PJRT_Program program{};
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.code = mutable_code.empty() ? nullptr : mutable_code.data();
    program.code_size = mutable_code.size();
    program.format = effective_format.empty() ? nullptr : effective_format.c_str();
    program.format_size = effective_format.size();

    PJRT_Client_Compile_Args cargs{};
    cargs.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    cargs.client = drv->client_;
    cargs.program = &program;

    if (PJRT_Error* err = api->PJRT_Client_Compile(&cargs)) {
        if (error_out) {
            *error_out = "PJRT_Client_Compile failed: " + consume_pjrt_error(api, err);
        } else {
            (void)consume_pjrt_error(api, err);
        }
        return false;
    }
    if (!cargs.executable) {
        if (error_out) {
            *error_out = "PJRT_Client_Compile succeeded but returned null executable";
        }
        return false;
    }

    PJRT_LoadedExecutable* next_exec = cargs.executable;
    auto destroy_next_exec = [&]() {
        if (next_exec && TPUF_API_HAS(api, PJRT_LoadedExecutable_Destroy)) {
            PJRT_LoadedExecutable_Destroy_Args dargs{};
            dargs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
            dargs.executable = next_exec;
            if (PJRT_Error* derr = api->PJRT_LoadedExecutable_Destroy(&dargs)) {
                (void)consume_pjrt_error(api, derr);
            }
        }
    };

    size_t next_num_outputs = 1;
    if (TPUF_API_HAS(api, PJRT_LoadedExecutable_GetExecutable) &&
        TPUF_API_HAS(api, PJRT_Executable_Destroy) &&
        TPUF_API_HAS(api, PJRT_Executable_NumOutputs)) {
        PJRT_LoadedExecutable_GetExecutable_Args gargs{};
        gargs.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
        gargs.loaded_executable = next_exec;
        if (PJRT_Error* err = api->PJRT_LoadedExecutable_GetExecutable(&gargs)) {
            if (error_out) *error_out = "PJRT_LoadedExecutable_GetExecutable failed: " + consume_pjrt_error(api, err);
            else (void)consume_pjrt_error(api, err);
            destroy_next_exec();
            return false;
        }
        PJRT_Executable* raw_exec = gargs.executable;
        if (raw_exec) {
            PJRT_Executable_NumOutputs_Args nargs{};
            nargs.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
            nargs.executable = raw_exec;
            PJRT_Error* nerr = api->PJRT_Executable_NumOutputs(&nargs);
            {
                PJRT_Executable_Destroy_Args dargs{};
                dargs.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
                dargs.executable = raw_exec;
                if (PJRT_Error* derr = api->PJRT_Executable_Destroy(&dargs)) {
                    (void)consume_pjrt_error(api, derr);
                }
            }
            if (nerr) {
                if (error_out) *error_out = "PJRT_Executable_NumOutputs failed: " + consume_pjrt_error(api, nerr);
                else (void)consume_pjrt_error(api, nerr);
                destroy_next_exec();
                return false;
            }
            next_num_outputs = nargs.num_outputs;
        }
    }
    if (next_num_outputs == 0) {
        if (error_out) {
            *error_out = "Compiled PJRT executable reports 0 outputs (expected >= 1 logits output)";
        }
        destroy_next_exec();
        return false;
    }

    if (executable_ && TPUF_API_HAS(api, PJRT_LoadedExecutable_Destroy)) {
        PJRT_LoadedExecutable_Destroy_Args dargs{};
        dargs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        dargs.executable = executable_;
        if (PJRT_Error* err = api->PJRT_LoadedExecutable_Destroy(&dargs)) {
            (void)consume_pjrt_error(api, err);
        }
    }

    executable_ = next_exec;
    num_outputs_ = next_num_outputs;
    shape_key_ = std::move(next_key);
    compiled_ = true;
    compile_count_ += 1;
    return true;
}

bool ZeroRecompileExecutor::execute_step(
    StaticArena& arena,
    size_t tile_rows,
    std::string* error_out
) {
    return execute_step_impl(arena, nullptr, nullptr, tile_rows, error_out);
}

bool ZeroRecompileExecutor::execute_step_with_kv(
    StaticArena& arena,
    std::vector<float>* out_step_k,
    std::vector<float>* out_step_v,
    size_t tile_rows,
    std::string* error_out
) {
    return execute_step_impl(arena, out_step_k, out_step_v, tile_rows, error_out);
}

bool ZeroRecompileExecutor::execute_step_impl(
    StaticArena& arena,
    std::vector<float>* out_step_k,
    std::vector<float>* out_step_v,
    size_t tile_rows,
    std::string* error_out
) {
    const size_t effective_rows = (tile_rows == 0) ? arena.tile_size() : tile_rows;
    if (!SystolicConstants::is_valid_tile_size(effective_rows)) {
        throw std::invalid_argument(
            "Systolic array execution requires static tile_rows in {128, 256} to prevent XLA recompilation"
        );
    }

    PjrtDriver* drv = driver_.load();
    if (!drv) {
        if (error_out) {
            *error_out = "ZeroRecompileExecutor has been detached from its PjrtDriver";
        }
        return false;
    }

    std::shared_lock drv_lk(drv->mu_);
    if (!drv->lib_handle_ || !drv->api_ || !drv->client_) {
        if (error_out) {
            *error_out = "PjrtDriver is not loaded";
        }
        return false;
    }

    std::scoped_lock exec_lk(mu_);
    if (!compiled_ || !executable_) {
        if (error_out) {
            *error_out = "ZeroRecompileExecutor has not compiled an executable yet";
        }
        return false;
    }

    const bool wants_kv_out = (out_step_k != nullptr || out_step_v != nullptr);
    const bool use_stateful_kv = (num_outputs_ >= 3);
    if (wants_kv_out && !use_stateful_kv) {
        if (error_out) {
            *error_out = "Compiled executable has fewer than 3 outputs and does not emit step K/V tensors";
        }
        return false;
    }

    const StaticArenaConfig& cfg = arena.config();
    if (!arena.token_tile() || !arena.position_tile() ||
        !arena.causal_mask() || !arena.logits_tile() ||
        (use_stateful_kv && (!arena.kv_prefix_k() || !arena.kv_prefix_v()))) {
        if (error_out) {
            *error_out = "StaticArena has been moved from";
        }
        return false;
    }
    if (cfg.tile_size != shape_key_.tile_rows ||
        effective_rows != shape_key_.tile_rows ||
        cfg.max_seq_len != shape_key_.max_seq_len ||
        cfg.hidden_dim != shape_key_.hidden_dim ||
        cfg.num_kv_heads != shape_key_.num_kv_heads ||
        cfg.head_dim != shape_key_.head_dim ||
        cfg.vocab_size != shape_key_.vocab_size) {
        if (error_out) {
            *error_out = "StaticArena config does not match compiled StaticShapeKey";
        }
        return false;
    }

    const PJRT_Api* api = drv->api_;
    PJRT_Client* client = drv->client_;

    PJRT_Device* const* raw_devices = nullptr;
    size_t num_devices = 0;
    if (!query_client_devices(api, client, &raw_devices, &num_devices, error_out)) {
        return false;
    }
    if (num_devices == 0 || !raw_devices || !raw_devices[0]) {
        if (error_out) {
            *error_out = "PJRT client has no target devices for BufferFromHostBuffer";
        }
        return false;
    }
    PJRT_Device* target_device = raw_devices[0];

    std::vector<BufferGuard> in_guards;
    const size_t num_args = use_stateful_kv ? 6 : 4;
    in_guards.reserve(num_args);
    for (size_t i = 0; i < num_args; ++i) {
        in_guards.emplace_back(api, nullptr);
    }

    const int64_t dims_1d[1] = {static_cast<int64_t>(effective_rows)};
    const int64_t dims_mask[2] = {
        static_cast<int64_t>(effective_rows),
        static_cast<int64_t>(effective_rows),
    };
    const int32_t plen_val = arena.valid_prefix_len();

    if (!upload_host_buffer(
            api, client, target_device, arena.token_tile(), PJRT_Buffer_Type_S32,
            dims_1d, 1, &in_guards[0].buf, error_out) ||
        !upload_host_buffer(
            api, client, target_device, arena.position_tile(), PJRT_Buffer_Type_S32,
            dims_1d, 1, &in_guards[1].buf, error_out) ||
        !upload_host_buffer(
            api, client, target_device, arena.causal_mask(), PJRT_Buffer_Type_F32,
            dims_mask, 2, &in_guards[2].buf, error_out) ||
        !upload_host_buffer(
            api, client, target_device, &plen_val, PJRT_Buffer_Type_S32,
            nullptr, 0, &in_guards[3].buf, error_out)) {
        return false;
    }

    if (use_stateful_kv) {
        const int64_t dims_kv_prefix[2] = {
            static_cast<int64_t>(cfg.max_seq_len),
            static_cast<int64_t>(arena.kv_row_elems()),
        };
        if (!upload_host_buffer(
                api, client, target_device, arena.kv_prefix_k(), PJRT_Buffer_Type_F32,
                dims_kv_prefix, 2, &in_guards[4].buf, error_out) ||
            !upload_host_buffer(
                api, client, target_device, arena.kv_prefix_v(), PJRT_Buffer_Type_F32,
                dims_kv_prefix, 2, &in_guards[5].buf, error_out)) {
            return false;
        }
    }

    std::vector<PJRT_Buffer*> arg_ptrs(num_args, nullptr);
    for (size_t i = 0; i < num_args; ++i) {
        arg_ptrs[i] = in_guards[i].buf;
    }
    PJRT_Buffer* const* arg_list[1] = {arg_ptrs.data()};

    std::vector<PJRT_Buffer*> out_ptrs(num_outputs_, nullptr);
    PJRT_Buffer** out_list[1] = {out_ptrs.data()};
    PJRT_Event* complete_event = nullptr;

    PJRT_ExecuteOptions opts{};
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_LoadedExecutable_Execute_Args eargs{};
    eargs.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    eargs.executable = executable_;
    eargs.options = &opts;
    eargs.argument_lists = arg_list;
    eargs.num_devices = 1;
    eargs.num_args = num_args;
    eargs.output_lists = out_list;
    eargs.device_complete_events = &complete_event;
    eargs.execute_device = nullptr;

    PJRT_Error* exec_err = api->PJRT_LoadedExecutable_Execute(&eargs);
    EventGuard complete_guard{api, complete_event};
    std::vector<BufferGuard> out_guards;
    out_guards.reserve(num_outputs_);
    for (PJRT_Buffer* b : out_ptrs) {
        out_guards.push_back(BufferGuard{api, b});
    }
    if (exec_err) {
        if (error_out) {
            *error_out = "PJRT_LoadedExecutable_Execute failed: " + consume_pjrt_error(api, exec_err);
        } else {
            (void)consume_pjrt_error(api, exec_err);
        }
        return false;
    }
    if (!await_and_destroy_event(api, complete_guard.release(), error_out)) {
        return false;
    }

    if (!download_f32_2d_buffer(
            api, out_ptrs[0], effective_rows, cfg.vocab_size, arena.logits_tile(), "logits", error_out)) {
        return false;
    }

    if (use_stateful_kv && (out_step_k || out_step_v)) {
        const size_t kv_elems = arena.kv_row_elems();
        const size_t step_elems = effective_rows * kv_elems;
        if (out_step_k) {
            out_step_k->resize(step_elems);
            if (!download_f32_2d_buffer(
                    api, out_ptrs[1], effective_rows, kv_elems, out_step_k->data(), "step_k", error_out)) {
                return false;
            }
        }
        if (out_step_v) {
            out_step_v->resize(step_elems);
            if (!download_f32_2d_buffer(
                    api, out_ptrs[2], effective_rows, kv_elems, out_step_v->data(), "step_v", error_out)) {
                return false;
            }
        }
    }

    execute_count_ += 1;
    return true;
}

bool ZeroRecompileExecutor::is_compiled() const {
    std::scoped_lock exec_lk(mu_);
    return compiled_;
}

size_t ZeroRecompileExecutor::compile_count() const {
    std::scoped_lock exec_lk(mu_);
    return compile_count_;
}

size_t ZeroRecompileExecutor::execute_count() const {
    std::scoped_lock exec_lk(mu_);
    return execute_count_;
}

size_t ZeroRecompileExecutor::num_outputs() const {
    std::scoped_lock exec_lk(mu_);
    return num_outputs_;
}

StaticShapeKey ZeroRecompileExecutor::shape_key() const {
    std::scoped_lock exec_lk(mu_);
    return shape_key_;
}

} // namespace tpuforge
