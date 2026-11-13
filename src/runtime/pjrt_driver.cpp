#include "tpuforge/pjrt_driver.h"

#include <cstring>
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

PjrtDriver::~PjrtDriver() {
    unload();
}

std::string PjrtDriver::format_and_destroy_error(PJRT_Error* err) const {
    if (!err) {
        return "";
    }
    std::string msg = "PJRT error";
    if (api_ && api_->PJRT_Error_Message) {
        PJRT_Error_Message_Args args{};
        args.struct_size = sizeof(args);
        args.error = err;
        api_->PJRT_Error_Message(&args);
        if (args.message && args.message_size > 0) {
            msg.assign(args.message, args.message_size);
        }
    }
    if (api_ && api_->PJRT_Error_Destroy) {
        PJRT_Error_Destroy_Args d_args{};
        d_args.struct_size = sizeof(d_args);
        d_args.error = err;
        api_->PJRT_Error_Destroy(&d_args);
    }
    return msg;
}

bool PjrtDriver::load_plugin(const std::string& path, std::string* error_out) {
    unload();
    if (path.empty()) {
        if (error_out) {
            *error_out = "PJRT plugin path cannot be empty";
        }
        return false;
    }

#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(path.c_str());
    if (!handle) {
        DWORD code = GetLastError();
        if (error_out) {
            *error_out = "Failed to LoadLibraryA('" + path + "'), Win32 error " + std::to_string(code);
        }
        return false;
    }
    FARPROC raw_sym = GetProcAddress(handle, "GetPjrtApi");
    if (!raw_sym) {
        if (error_out) {
            *error_out = "Symbol 'GetPjrtApi' not found in PJRT plugin: " + path;
        }
        FreeLibrary(handle);
        return false;
    }
    lib_handle_ = reinterpret_cast<void*>(handle);
    auto get_api_fn = reinterpret_cast<PJRT_GetApiFn>(raw_sym);
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* dl_err = dlerror();
        if (error_out) {
            *error_out = std::string("Failed to dlopen('") + path + "'): " + (dl_err ? dl_err : "unknown");
        }
        return false;
    }
    void* raw_sym = dlsym(handle, "GetPjrtApi");
    if (!raw_sym) {
        if (error_out) {
            *error_out = "Symbol 'GetPjrtApi' not found in PJRT plugin: " + path;
        }
        dlclose(handle);
        return false;
    }
    lib_handle_ = handle;
    auto get_api_fn = reinterpret_cast<PJRT_GetApiFn>(raw_sym);
#endif

    const PJRT_Api* api = get_api_fn();
    if (!api) {
        if (error_out) {
            *error_out = "GetPjrtApi returned null pointer for: " + path;
        }
        unload();
        return false;
    }
    if (api->pjrt_api_version.major_version < 0) {
        if (error_out) {
            *error_out = "Invalid PJRT API major_version";
        }
        unload();
        return false;
    }

    api_ = api;

    if (api_->PJRT_Client_Create) {
        PJRT_Client_Create_Args args{};
        args.struct_size = sizeof(args);
        PJRT_Error* err = api_->PJRT_Client_Create(&args);
        if (err) {
            if (error_out) {
                *error_out = format_and_destroy_error(err);
            }
            unload();
            return false;
        }
        client_ = args.client;
    }

    if (!client_) {
        if (error_out) {
            *error_out = "PJRT_Client_Create did not return a valid client";
        }
        unload();
        return false;
    }

    return true;
}

void PjrtDriver::unload() {
    if (client_ && api_ && api_->PJRT_Client_Destroy) {
        PJRT_Client_Destroy_Args args{};
        args.struct_size = sizeof(args);
        args.client = client_;
        PJRT_Error* err = api_->PJRT_Client_Destroy(&args);
        if (err) {
            format_and_destroy_error(err);
        }
    }
    client_ = nullptr;
    api_ = nullptr;

    if (lib_handle_) {
#if defined(_WIN32)
        FreeLibrary(reinterpret_cast<HMODULE>(lib_handle_));
#else
        dlclose(lib_handle_);
#endif
        lib_handle_ = nullptr;
    }
}

std::string PjrtDriver::platform_name() const {
    if (!is_loaded() || !api_->PJRT_Client_PlatformName) {
        return "";
    }
    PJRT_Client_PlatformName_Args args{};
    args.struct_size = sizeof(args);
    args.client = client_;
    PJRT_Error* err = api_->PJRT_Client_PlatformName(&args);
    if (err) {
        format_and_destroy_error(err);
        return "";
    }
    if (!args.platform_name || args.platform_name_size == 0) {
        return "";
    }
    return std::string(args.platform_name, args.platform_name_size);
}

std::vector<DeviceInfo> PjrtDriver::query_devices() const {
    std::vector<DeviceInfo> out;
    if (!is_loaded() || !api_->PJRT_Client_Devices || !api_->PJRT_Device_GetDescription) {
        return out;
    }

    PJRT_Client_Devices_Args dev_args{};
    dev_args.struct_size = sizeof(dev_args);
    dev_args.client = client_;
    PJRT_Error* err = api_->PJRT_Client_Devices(&dev_args);
    if (err) {
        format_and_destroy_error(err);
        return out;
    }

    for (size_t i = 0; i < dev_args.num_devices; ++i) {
        PJRT_Device* dev = dev_args.devices[i];
        if (!dev) {
            continue;
        }
        PJRT_Device_GetDescription_Args desc_args{};
        desc_args.struct_size = sizeof(desc_args);
        desc_args.device = dev;
        PJRT_Error* d_err = api_->PJRT_Device_GetDescription(&desc_args);
        if (d_err) {
            format_and_destroy_error(d_err);
            continue;
        }

        DeviceInfo info;
        info.id = desc_args.description.device_id;
        info.kind = desc_args.description.device_kind ? desc_args.description.device_kind : "Unknown";
        info.memory_bytes = desc_args.description.memory_bytes;
        info.is_tpu = (info.kind.find("TPU") != std::string::npos);
        info.mxu_dim_x = static_cast<int32_t>(SystolicConstants::kSystolicTileSize);
        info.mxu_dim_y = static_cast<int32_t>(SystolicConstants::kSystolicTileSize);
        out.push_back(std::move(info));
    }

    return out;
}

ZeroRecompileExecutor::ZeroRecompileExecutor(PjrtDriver& driver) : driver_(driver) {}

ZeroRecompileExecutor::~ZeroRecompileExecutor() {
    release_executable();
}

void ZeroRecompileExecutor::release_executable() {
    if (executable_ && driver_.api() && driver_.api()->PJRT_LoadedExecutable_Destroy) {
        PJRT_LoadedExecutable_Destroy_Args args{};
        args.struct_size = sizeof(args);
        args.executable = executable_;
        PJRT_Error* err = driver_.api()->PJRT_LoadedExecutable_Destroy(&args);
        if (err && driver_.api()->PJRT_Error_Destroy) {
            PJRT_Error_Destroy_Args d_args{};
            d_args.struct_size = sizeof(d_args);
            d_args.error = err;
            driver_.api()->PJRT_Error_Destroy(&d_args);
        }
    }
    executable_ = nullptr;
}

bool ZeroRecompileExecutor::compile_once(
    const StaticArena::Config& config,
    const std::string& program_code,
    const std::string& program_format,
    std::string* error_out
) {
    if (!driver_.is_loaded() || !driver_.api() || !driver_.api()->PJRT_Client_Compile) {
        if (error_out) {
            *error_out = "PJRT driver is not loaded or lacks PJRT_Client_Compile";
        }
        return false;
    }

    StaticShapeKey target_key{
        SystolicConstants::kSystolicTileSize,
        config.max_seq_len,
        config.hidden_dim,
        config.vocab_size
    };

    if (executable_ != nullptr && compiled_key_ == target_key) {
        return true;
    }

    release_executable();

    PJRT_Client_Compile_Args args{};
    args.struct_size = sizeof(args);
    args.client = driver_.client();
    args.program_code = program_code.data();
    args.program_code_size = program_code.size();
    args.program_format = program_format.c_str();
    args.tile_rows = target_key.tile_rows;
    args.max_seq_len = target_key.max_seq_len;
    args.hidden_dim = target_key.hidden_dim;
    args.vocab_size = target_key.vocab_size;

    PJRT_Error* err = driver_.api()->PJRT_Client_Compile(&args);
    if (err) {
        if (error_out) {
            if (driver_.api()->PJRT_Error_Message) {
                PJRT_Error_Message_Args m_args{};
                m_args.struct_size = sizeof(m_args);
                m_args.error = err;
                driver_.api()->PJRT_Error_Message(&m_args);
                *error_out = (m_args.message && m_args.message_size > 0)
                                 ? std::string(m_args.message, m_args.message_size)
                                 : "PJRT_Client_Compile failed";
            }
            if (driver_.api()->PJRT_Error_Destroy) {
                PJRT_Error_Destroy_Args d_args{};
                d_args.struct_size = sizeof(d_args);
                d_args.error = err;
                driver_.api()->PJRT_Error_Destroy(&d_args);
            }
        }
        return false;
    }

    if (!args.executable) {
        if (error_out) {
            *error_out = "PJRT_Client_Compile returned null executable";
        }
        return false;
    }

    executable_ = args.executable;
    compiled_key_ = target_key;
    ++compile_count_;
    return true;
}

bool ZeroRecompileExecutor::execute_step(
    StaticArena& arena,
    size_t tile_rows,
    std::string* error_out
) {
    if (tile_rows != SystolicConstants::kSystolicTileSize) {
        throw std::invalid_argument(
            "ZeroRecompileExecutor requires fixed systolic tile_rows == 128 to prevent XLA recompilation"
        );
    }
    if (!executable_ || !driver_.api() || !driver_.api()->PJRT_LoadedExecutable_Execute) {
        if (error_out) {
            *error_out = "No compiled PJRT executable loaded";
        }
        return false;
    }

    StaticShapeKey arena_key{
        SystolicConstants::kSystolicTileSize,
        arena.config().max_seq_len,
        arena.config().hidden_dim,
        arena.config().vocab_size
    };
    if (!(arena_key == compiled_key_)) {
        if (error_out) {
            *error_out = "StaticArena config dimensions do not match compiled static executable";
        }
        return false;
    }

    PJRT_LoadedExecutable_Execute_Args args{};
    args.struct_size = sizeof(args);
    args.executable = executable_;
    args.token_tile = arena.token_tile_128();
    args.position_tile = arena.position_tile_128();
    args.causal_mask = arena.causal_mask_128x128();
    args.valid_prefix_len = arena.valid_prefix_len();
    args.tile_rows = tile_rows;
    args.vocab_size = arena.config().vocab_size;
    args.out_logits = arena.logits_tile_128();

    PJRT_Error* err = driver_.api()->PJRT_LoadedExecutable_Execute(&args);
    if (err) {
        if (error_out) {
            if (driver_.api()->PJRT_Error_Message) {
                PJRT_Error_Message_Args m_args{};
                m_args.struct_size = sizeof(m_args);
                m_args.error = err;
                driver_.api()->PJRT_Error_Message(&m_args);
                *error_out = (m_args.message && m_args.message_size > 0)
                                 ? std::string(m_args.message, m_args.message_size)
                                 : "PJRT_LoadedExecutable_Execute failed";
            }
            if (driver_.api()->PJRT_Error_Destroy) {
                PJRT_Error_Destroy_Args d_args{};
                d_args.struct_size = sizeof(d_args);
                d_args.error = err;
                driver_.api()->PJRT_Error_Destroy(&d_args);
            }
        }
        return false;
    }

    ++execute_count_;
    return true;
}

} // namespace tpuforge
