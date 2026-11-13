#include "tpuforge/pjrt_c_api.h"

#include <ctype.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define PJRT_EXPORT __declspec(dllexport)
#else
#define PJRT_EXPORT __attribute__((visibility("default")))
#endif

#define PJRT_ARGS_HAVE(Type, args_ptr, field) \
    ((args_ptr) != NULL && (args_ptr)->struct_size >= (offsetof(Type, field) + sizeof(((Type*)0)->field)))

struct TestPjrtError {
    PJRT_Error base;
    PJRT_Error_Code code;
    char message[192];
};

static void test_err_vtable_destroy(PJRT_Error* error);
static void test_err_vtable_message(const PJRT_Error* error, const char** message, size_t* message_size);
static PJRT_Error_Code test_err_vtable_code(const PJRT_Error* error);
static void test_err_vtable_payload(
    const PJRT_Error* error, PJRT_Error_PayloadVisitor visitor, void* user_arg
);

static const PJRT_Error_FunctionTable kErrorVtable = {
    .struct_size = PJRT_Error_FunctionTable_STRUCT_SIZE,
    .instance_size = sizeof(struct TestPjrtError),
    .extension_start = NULL,
    .destroy = test_err_vtable_destroy,
    .message = test_err_vtable_message,
    .get_code = test_err_vtable_code,
    .for_each_payload = test_err_vtable_payload,
};

static const struct TestPjrtError kOomError = {
    .base = {&kErrorVtable},
    .code = PJRT_Error_Code_RESOURCE_EXHAUSTED,
    .message = "Out of host memory allocating PJRT_Error in test plugin",
};

static void test_err_vtable_destroy(PJRT_Error* error) {
    if (!error || error == &kOomError.base) {
        return;
    }
    free(error);
}

static void test_err_vtable_message(const PJRT_Error* error, const char** message, size_t* message_size) {
    if (!error || !message || !message_size) {
        return;
    }
    const struct TestPjrtError* e = (const struct TestPjrtError*)error;
    *message = e->message;
    *message_size = strlen(e->message);
}

static PJRT_Error_Code test_err_vtable_code(const PJRT_Error* error) {
    if (!error) {
        return PJRT_Error_Code_OK;
    }
    const struct TestPjrtError* e = (const struct TestPjrtError*)error;
    return e->code;
}

static void test_err_vtable_payload(
    const PJRT_Error* error, PJRT_Error_PayloadVisitor visitor, void* user_arg
) {
    (void)error;
    (void)visitor;
    (void)user_arg;
}

static PJRT_Error* make_err(PJRT_Error_Code code, const char* msg) {
    struct TestPjrtError* e = (struct TestPjrtError*)calloc(1, sizeof(struct TestPjrtError));
    if (!e) {
        return (PJRT_Error*)&kOomError.base;
    }
    e->base.vtable = &kErrorVtable;
    e->code = code;
    snprintf(e->message, sizeof(e->message), "%s", msg ? msg : "Unknown error");
    return &e->base;
}

struct PJRT_Event {
    bool ready;
};

struct PJRT_DeviceDescription {
    int id;
    int process_index;
    const char* kind;
    const char* debug_str;
    const char* short_str;
};

struct PJRT_Device {
    PJRT_DeviceDescription desc;
    int64_t bytes_limit;
};

struct PJRT_Client {
    struct PJRT_Device devices[2];
    PJRT_Device* device_ptrs[2];
    atomic_size_t compile_calls;
};

struct PJRT_LoadedExecutable {
    struct PJRT_Client* client;
    size_t tile_rows;
    size_t vocab_size;
    size_t num_outputs;
    size_t kv_elems;
};

struct PJRT_Executable {
    size_t tile_rows;
    size_t vocab_size;
    size_t num_outputs;
    size_t kv_elems;
};

struct PJRT_Buffer {
    PJRT_Buffer_Type type;
    int64_t dims[4];
    size_t num_dims;
    size_t byte_size;
    void* data;
};

static PJRT_Event* make_ready_event(void) {
    PJRT_Event* ev = (PJRT_Event*)calloc(1, sizeof(PJRT_Event));
    if (ev) {
        ev->ready = true;
    }
    return ev;
}

static size_t dtype_bytes(PJRT_Buffer_Type t) {
    switch (t) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
        case PJRT_Buffer_Type_F16:
        case PJRT_Buffer_Type_BF16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
        case PJRT_Buffer_Type_C64:
            return 8;
        case PJRT_Buffer_Type_C128:
            return 16;
        default:
            return 0;
    }
}

static bool compute_byte_size(
    PJRT_Buffer_Type type, const int64_t* dims, size_t num_dims, size_t* out_bytes
) {
    size_t elem_sz = dtype_bytes(type);
    if (elem_sz == 0 || num_dims > 4) {
        return false;
    }
    size_t elems = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        if (dims[i] < 0) {
            return false;
        }
        size_t d = (size_t)dims[i];
        if (d > 0 && elems > SIZE_MAX / d) {
            return false;
        }
        elems *= d;
    }
    if (elems > SIZE_MAX / elem_sz) {
        return false;
    }
    *out_bytes = elems * elem_sz;
    return true;
}

static void test_error_destroy(PJRT_Error_Destroy_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_Error_Destroy_Args, args, error) && args->error) {
        test_err_vtable_destroy(args->error);
    }
}

static void test_error_message(PJRT_Error_Message_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_Error_Message_Args, args, message_size) && args->error) {
        test_err_vtable_message(args->error, &args->message, &args->message_size);
    }
}

static PJRT_Error* test_error_get_code(PJRT_Error_GetCode_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Error_GetCode_Args, args, code) || !args->error) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Error_GetCode");
    }
    args->code = test_err_vtable_code(args->error);
    return NULL;
}

static PJRT_Error* test_plugin_initialize(PJRT_Plugin_Initialize_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Plugin_Initialize_Args, args, extension_start)) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid struct_size in PJRT_Plugin_Initialize");
    }
    return NULL;
}

static PJRT_Error* test_plugin_attributes(PJRT_Plugin_Attributes_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Plugin_Attributes_Args, args, num_attributes)) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Plugin_Attributes");
    }
    args->attributes = NULL;
    args->num_attributes = 0;
    return NULL;
}

static PJRT_Error* test_event_destroy(PJRT_Event_Destroy_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_Event_Destroy_Args, args, event) && args->event) {
        free(args->event);
    }
    return NULL;
}

static PJRT_Error* test_event_is_ready(PJRT_Event_IsReady_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Event_IsReady_Args, args, is_ready) || !args->event) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Event_IsReady");
    }
    args->is_ready = args->event->ready;
    return NULL;
}

static PJRT_Error* test_event_error(PJRT_Event_Error_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Event_Error_Args, args, event) || !args->event) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Event_Error");
    }
    return NULL;
}

static PJRT_Error* test_event_await(PJRT_Event_Await_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Event_Await_Args, args, event) || !args->event) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Event_Await");
    }
    return NULL;
}

static PJRT_Error* test_client_create(PJRT_Client_Create_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Client_Create_Args, args, client)) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid struct_size in PJRT_Client_Create");
    }
    struct PJRT_Client* c = (struct PJRT_Client*)calloc(1, sizeof(struct PJRT_Client));
    if (!c) {
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed in PJRT_Client_Create");
    }
    const int64_t hbm_32gb = 32LL * 1024LL * 1024LL * 1024LL;

    c->devices[0].desc.id = 0;
    c->devices[0].desc.process_index = 0;
    c->devices[0].desc.kind = "Google TPU v6e";
    c->devices[0].desc.debug_str = "TPU_0(process=0,trillium=v6e)";
    c->devices[0].desc.short_str = "TPU_0";
    c->devices[0].bytes_limit = hbm_32gb;

    c->devices[1].desc.id = 1;
    c->devices[1].desc.process_index = 0;
    c->devices[1].desc.kind = "Google TPU v6e";
    c->devices[1].desc.debug_str = "TPU_1(process=0,trillium=v6e)";
    c->devices[1].desc.short_str = "TPU_1";
    c->devices[1].bytes_limit = hbm_32gb;

    c->device_ptrs[0] = &c->devices[0];
    c->device_ptrs[1] = &c->devices[1];
    atomic_init(&c->compile_calls, 0);
    args->client = c;
    return NULL;
}

static PJRT_Error* test_client_destroy(PJRT_Client_Destroy_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_Client_Destroy_Args, args, client) && args->client) {
        free(args->client);
    }
    return NULL;
}

static PJRT_Error* test_client_platform_name(PJRT_Client_PlatformName_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Client_PlatformName_Args, args, platform_name_size) || !args->client) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Client_PlatformName");
    }
    static const char kPlatform[] = "tpu";
    args->platform_name = kPlatform;
    args->platform_name_size = sizeof(kPlatform) - 1;
    return NULL;
}

static PJRT_Error* test_client_devices(PJRT_Client_Devices_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Client_Devices_Args, args, num_devices) || !args->client) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid client in PJRT_Client_Devices");
    }
    args->devices = args->client->device_ptrs;
    args->num_devices = 2;
    return NULL;
}

static PJRT_Error* test_client_addressable_devices(PJRT_Client_AddressableDevices_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Client_AddressableDevices_Args, args, num_addressable_devices) ||
        !args->client) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid client in PJRT_Client_AddressableDevices"
        );
    }
    args->addressable_devices = args->client->device_ptrs;
    args->num_addressable_devices = 2;
    return NULL;
}

static PJRT_Error* test_device_description_id(PJRT_DeviceDescription_Id_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_DeviceDescription_Id_Args, args, id) || !args->device_description) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_DeviceDescription_Id");
    }
    args->id = args->device_description->id;
    return NULL;
}

static PJRT_Error* test_device_description_process_index(
    PJRT_DeviceDescription_ProcessIndex_Args* args
) {
    if (!PJRT_ARGS_HAVE(PJRT_DeviceDescription_ProcessIndex_Args, args, process_index) ||
        !args->device_description) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_DeviceDescription_ProcessIndex"
        );
    }
    args->process_index = args->device_description->process_index;
    return NULL;
}

static PJRT_Error* test_device_description_attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_DeviceDescription_Attributes_Args, args, attributes) ||
        !args->device_description) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_DeviceDescription_Attributes"
        );
    }
    args->attributes = NULL;
    args->num_attributes = 0;
    return NULL;
}

static PJRT_Error* test_device_description_kind(PJRT_DeviceDescription_Kind_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_DeviceDescription_Kind_Args, args, device_kind_size) ||
        !args->device_description) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_DeviceDescription_Kind"
        );
    }
    args->device_kind = args->device_description->kind;
    args->device_kind_size = strlen(args->device_description->kind);
    return NULL;
}

static PJRT_Error* test_device_description_debug_string(
    PJRT_DeviceDescription_DebugString_Args* args
) {
    if (!PJRT_ARGS_HAVE(PJRT_DeviceDescription_DebugString_Args, args, debug_string_size) ||
        !args->device_description) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_DeviceDescription_DebugString"
        );
    }
    args->debug_string = args->device_description->debug_str;
    args->debug_string_size = strlen(args->device_description->debug_str);
    return NULL;
}

static PJRT_Error* test_device_description_to_string(PJRT_DeviceDescription_ToString_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_DeviceDescription_ToString_Args, args, to_string_size) ||
        !args->device_description) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_DeviceDescription_ToString"
        );
    }
    args->to_string = args->device_description->short_str;
    args->to_string_size = strlen(args->device_description->short_str);
    return NULL;
}

static PJRT_Error* test_device_get_description(PJRT_Device_GetDescription_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Device_GetDescription_Args, args, device_description) ||
        !args->device) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid device in PJRT_Device_GetDescription"
        );
    }
    args->device_description = &args->device->desc;
    return NULL;
}

static PJRT_Error* test_device_is_addressable(PJRT_Device_IsAddressable_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Device_IsAddressable_Args, args, is_addressable) || !args->device) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Device_IsAddressable");
    }
    args->is_addressable = true;
    return NULL;
}

static PJRT_Error* test_device_memory_stats(PJRT_Device_MemoryStats_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Device_MemoryStats_Args, args, bytes_limit_is_set) || !args->device) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Device_MemoryStats");
    }
    memset(
        ((char*)args) + offsetof(PJRT_Device_MemoryStats_Args, bytes_in_use),
        0,
        args->struct_size - offsetof(PJRT_Device_MemoryStats_Args, bytes_in_use)
    );
    args->bytes_in_use = 0;
    args->bytes_limit = args->device->bytes_limit;
    args->bytes_limit_is_set = true;
    return NULL;
}

static const char* find_substr_bounded(const char* hay, size_t hay_len, const char* needle) {
    size_t nlen = strlen(needle);
    if (!hay || nlen == 0 || hay_len < nlen) {
        return NULL;
    }
    for (size_t i = 0; i <= hay_len - nlen; ++i) {
        if (memcmp(hay + i, needle, nlen) == 0) {
            return hay + i;
        }
    }
    return NULL;
}

static bool parse_uint_bounded(const char* p, const char* end, size_t* out_val) {
    if (!p || p >= end || !isdigit((unsigned char)*p)) {
        return false;
    }
    size_t val = 0;
    while (p < end && isdigit((unsigned char)*p)) {
        size_t digit = (size_t)(*p - '0');
        if (val > (SIZE_MAX - digit) / 10) {
            return false;
        }
        val = val * 10 + digit;
        ++p;
    }
    *out_val = val;
    return true;
}

static PJRT_Error* test_client_compile(PJRT_Client_Compile_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Client_Compile_Args, args, executable) ||
        !args->client || !args->program) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid client/program in PJRT_Client_Compile");
    }
    if (!PJRT_ARGS_HAVE(PJRT_Program, args->program, format_size)) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid PJRT_Program struct_size");
    }
    const char* fmt = args->program->format;
    const size_t fmt_len = args->program->format_size;
    const bool valid_fmt = fmt != NULL && (
        (fmt_len == 4 && memcmp(fmt, "mlir", 4) == 0) ||
        (fmt_len == 3 && memcmp(fmt, "hlo", 3) == 0) ||
        (fmt_len == 15 && memcmp(fmt, "hlo_with_config", 15) == 0)
    );
    if (!valid_fmt) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT,
            "Unknown program format in PJRT_Client_Compile (expected mlir, hlo, or hlo_with_config)"
        );
    }

    size_t tile_rows = 128;
    size_t vocab_size = 256;
    size_t num_outputs = 1;
    size_t kv_elems = 0;
    const char* code = args->program->code;
    const size_t code_len = args->program->code_size;
    if (code && code_len > 0) {
        const char* end = code + code_len;
        const char* tensor_pos = find_substr_bounded(code, code_len, "tensor<");
        if (tensor_pos) {
            const char* num_start = tensor_pos + strlen("tensor<");
            size_t parsed_rows = 0;
            if (parse_uint_bounded(num_start, end, &parsed_rows)) {
                tile_rows = parsed_rows;
                const char* x_pos = num_start;
                while (x_pos < end && isdigit((unsigned char)*x_pos)) {
                    ++x_pos;
                }
                if (x_pos < end && *x_pos == 'x') {
                    size_t parsed_vocab = 0;
                    if (parse_uint_bounded(x_pos + 1, end, &parsed_vocab) && parsed_vocab > 0) {
                        vocab_size = parsed_vocab;
                    }
                }
            }
        } else {
            const char* vpos = find_substr_bounded(code, code_len, "vocab=");
            if (vpos) {
                size_t parsed_vocab = 0;
                if (parse_uint_bounded(vpos + strlen("vocab="), end, &parsed_vocab) && parsed_vocab > 0) {
                    vocab_size = parsed_vocab;
                }
            }
        }

        const char* opos = find_substr_bounded(code, code_len, "outputs=");
        if (opos) {
            size_t parsed_outs = 0;
            if (!parse_uint_bounded(opos + strlen("outputs="), end, &parsed_outs) ||
                (parsed_outs != 1 && parsed_outs != 3)) {
                return make_err(
                    PJRT_Error_Code_INVALID_ARGUMENT,
                    "Unsupported outputs= count in test plugin (expected 1 or 3)"
                );
            }
            num_outputs = parsed_outs;
        }
        const char* kpos = find_substr_bounded(code, code_len, "kv_elems=");
        if (kpos) {
            size_t parsed_kv = 0;
            if (!parse_uint_bounded(kpos + strlen("kv_elems="), end, &parsed_kv) || parsed_kv == 0 ||
                parsed_kv > 8192) {
                return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid kv_elems= in program");
            }
            kv_elems = parsed_kv;
        }
    }

    if (tile_rows != 128) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "TPU systolic compiler requires tile_rows == 128");
    }
    if (vocab_size == 0 || vocab_size > 1048576) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Unsupported vocab_size in PJRT_Client_Compile");
    }
    if (num_outputs == 3 && kv_elems == 0) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT,
            "Stateful 3-output program requires kv_elems=<n> in program metadata"
        );
    }

    struct PJRT_LoadedExecutable* exec =
        (struct PJRT_LoadedExecutable*)calloc(1, sizeof(struct PJRT_LoadedExecutable));
    if (!exec) {
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed in PJRT_Client_Compile");
    }
    exec->client = args->client;
    exec->tile_rows = tile_rows;
    exec->vocab_size = vocab_size;
    exec->num_outputs = num_outputs;
    exec->kv_elems = kv_elems;
    atomic_fetch_add(&args->client->compile_calls, 1);
    args->executable = exec;
    return NULL;
}

static PJRT_Error* test_client_buffer_from_host_buffer(
    PJRT_Client_BufferFromHostBuffer_Args* args
) {
    if (!PJRT_ARGS_HAVE(PJRT_Client_BufferFromHostBuffer_Args, args, buffer) ||
        !args->client || !args->data) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Client_BufferFromHostBuffer"
        );
    }
    if (args->num_byte_strides != 0) {
        return make_err(
            PJRT_Error_Code_UNIMPLEMENTED, "Strided host buffers are not supported in test plugin"
        );
    }
    size_t byte_size = 0;
    if (!compute_byte_size(args->type, args->dims, args->num_dims, &byte_size)) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid shape or dtype in BufferFromHostBuffer");
    }

    struct PJRT_Buffer* buf = (struct PJRT_Buffer*)calloc(1, sizeof(struct PJRT_Buffer));
    if (!buf) {
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for PJRT_Buffer");
    }
    buf->type = args->type;
    buf->num_dims = args->num_dims;
    for (size_t i = 0; i < args->num_dims; ++i) {
        buf->dims[i] = args->dims[i];
    }
    buf->byte_size = byte_size;
    if (byte_size > 0) {
        buf->data = malloc(byte_size);
        if (!buf->data) {
            free(buf);
            return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for buffer storage");
        }
        memcpy(buf->data, args->data, byte_size);
    }
    args->done_with_host_buffer = make_ready_event();
    args->buffer = buf;
    return NULL;
}

static PJRT_Error* test_executable_destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_LoadedExecutable_Destroy_Args, args, executable) && args->executable) {
        free(args->executable);
    }
    return NULL;
}

static PJRT_Error* test_raw_executable_destroy(PJRT_Executable_Destroy_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_Executable_Destroy_Args, args, executable) && args->executable) {
        free(args->executable);
    }
    return NULL;
}

static PJRT_Error* test_loaded_executable_get_executable(
    PJRT_LoadedExecutable_GetExecutable_Args* args
) {
    if (!PJRT_ARGS_HAVE(PJRT_LoadedExecutable_GetExecutable_Args, args, executable) ||
        !args->loaded_executable) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_LoadedExecutable_GetExecutable"
        );
    }
    struct PJRT_Executable* raw = (struct PJRT_Executable*)calloc(1, sizeof(struct PJRT_Executable));
    if (!raw) {
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for PJRT_Executable");
    }
    raw->tile_rows = args->loaded_executable->tile_rows;
    raw->vocab_size = args->loaded_executable->vocab_size;
    raw->num_outputs = args->loaded_executable->num_outputs;
    raw->kv_elems = args->loaded_executable->kv_elems;
    args->executable = raw;
    return NULL;
}

static PJRT_Error* test_executable_num_outputs(PJRT_Executable_NumOutputs_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Executable_NumOutputs_Args, args, num_outputs) || !args->executable) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Executable_NumOutputs");
    }
    args->num_outputs = args->executable->num_outputs;
    return NULL;
}

static PJRT_Error* test_executable_execute(PJRT_LoadedExecutable_Execute_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_LoadedExecutable_Execute_Args, args, execute_device) ||
        !args->executable || !args->argument_lists || !args->output_lists) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid arguments in PJRT_LoadedExecutable_Execute");
    }
    const size_t n_outs = args->executable->num_outputs;
    const size_t expected_args = (n_outs == 3) ? 6 : 4;
    if (args->num_devices != 1 || args->num_args != expected_args) {
        return make_err(
            PJRT_Error_Code_INVALID_ARGUMENT, "Unexpected num_devices or num_args in Execute"
        );
    }
    PJRT_Buffer* const* in = args->argument_lists[0];
    PJRT_Buffer** out = args->output_lists[0];
    if (!in || !out || !in[0] || !in[1] || !in[2] || !in[3]) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Missing input buffers in Execute");
    }
    if (n_outs == 3 && (!in[4] || !in[5])) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Missing KV prefix buffers in stateful Execute");
    }

    const struct PJRT_Buffer* tok_buf = in[0];
    const struct PJRT_Buffer* pos_buf = in[1];
    const struct PJRT_Buffer* mask_buf = in[2];
    const struct PJRT_Buffer* plen_buf = in[3];

    if (tok_buf->type != PJRT_Buffer_Type_S32 || tok_buf->num_dims != 1 || tok_buf->dims[0] != 128 ||
        pos_buf->type != PJRT_Buffer_Type_S32 || pos_buf->num_dims != 1 || pos_buf->dims[0] != 128 ||
        mask_buf->type != PJRT_Buffer_Type_F32 || mask_buf->num_dims != 2 ||
        mask_buf->dims[0] != 128 || mask_buf->dims[1] != 128 ||
        plen_buf->type != PJRT_Buffer_Type_S32 ||
        !(plen_buf->num_dims == 0 || (plen_buf->num_dims == 1 && plen_buf->dims[0] == 1))) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Input tensor shape or dtype mismatch");
    }

    const int32_t valid_prefix_len = *(const int32_t*)plen_buf->data;
    const size_t kv_elems = args->executable->kv_elems;
    const float* kv_k_prefix = NULL;
    const float* kv_v_prefix = NULL;
    if (n_outs == 3) {
        const struct PJRT_Buffer* kp = in[4];
        const struct PJRT_Buffer* vp = in[5];
        if (kp->type != PJRT_Buffer_Type_F32 || kp->num_dims != 2 ||
            vp->type != PJRT_Buffer_Type_F32 || vp->num_dims != 2 ||
            kp->dims[0] != vp->dims[0] || kp->dims[1] != vp->dims[1] ||
            (size_t)kp->dims[1] != kv_elems || valid_prefix_len < 0 ||
            valid_prefix_len > kp->dims[0]) {
            return make_err(
                PJRT_Error_Code_INVALID_ARGUMENT,
                "KV prefix tensor shape/dtype or valid_prefix_len out of bounds"
            );
        }
        kv_k_prefix = (const float*)kp->data;
        kv_v_prefix = (const float*)vp->data;
    }

    const size_t rows = args->executable->tile_rows;
    const size_t vocab = args->executable->vocab_size;
    if (rows > SIZE_MAX / vocab || (rows * vocab) > SIZE_MAX / sizeof(float)) {
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Output buffer size overflow");
    }

    struct PJRT_Buffer* out_buf = (struct PJRT_Buffer*)calloc(1, sizeof(struct PJRT_Buffer));
    if (!out_buf) {
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for output PJRT_Buffer");
    }
    out_buf->type = PJRT_Buffer_Type_F32;
    out_buf->num_dims = 2;
    out_buf->dims[0] = (int64_t)rows;
    out_buf->dims[1] = (int64_t)vocab;
    out_buf->byte_size = rows * vocab * sizeof(float);
    out_buf->data = malloc(out_buf->byte_size);
    if (!out_buf->data) {
        free(out_buf);
        return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for logits output data");
    }

    struct PJRT_Buffer* out_k = NULL;
    struct PJRT_Buffer* out_v = NULL;
    if (n_outs == 3) {
        if (kv_elems > SIZE_MAX / rows || (rows * kv_elems) > SIZE_MAX / sizeof(float)) {
            free(out_buf->data);
            free(out_buf);
            return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Output KV size overflow");
        }
        size_t kv_bytes = rows * kv_elems * sizeof(float);
        out_k = (struct PJRT_Buffer*)calloc(1, sizeof(struct PJRT_Buffer));
        out_v = (struct PJRT_Buffer*)calloc(1, sizeof(struct PJRT_Buffer));
        if (!out_k || !out_v) {
            free(out_k);
            free(out_v);
            free(out_buf->data);
            free(out_buf);
            return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for step KV buffers");
        }
        out_k->type = PJRT_Buffer_Type_F32;
        out_k->num_dims = 2;
        out_k->dims[0] = (int64_t)rows;
        out_k->dims[1] = (int64_t)kv_elems;
        out_k->byte_size = kv_bytes;
        out_k->data = calloc(1, kv_bytes);

        out_v->type = PJRT_Buffer_Type_F32;
        out_v->num_dims = 2;
        out_v->dims[0] = (int64_t)rows;
        out_v->dims[1] = (int64_t)kv_elems;
        out_v->byte_size = kv_bytes;
        out_v->data = calloc(1, kv_bytes);
        if (!out_k->data || !out_v->data) {
            free(out_k->data);
            free(out_v->data);
            free(out_k);
            free(out_v);
            free(out_buf->data);
            free(out_buf);
            return make_err(PJRT_Error_Code_RESOURCE_EXHAUSTED, "Allocation failed for step KV data");
        }
    }

    const int32_t* token_tile = (const int32_t*)tok_buf->data;
    const int32_t* pos_tile = (const int32_t*)pos_buf->data;
    const float* causal_mask = (const float*)mask_buf->data;
    float* logits = (float*)out_buf->data;

    if (n_outs == 1) {
        for (size_t r = 0; r < rows; ++r) {
            int32_t tok = token_tile[r];
            int32_t pos = pos_tile[r];
            float mask_diag = causal_mask[r * rows + r];
            float* row_out = logits + (r * vocab);

            int64_t sum = (int64_t)tok + (int64_t)pos + 1;
            int64_t target = sum % (int64_t)vocab;
            if (target < 0) {
                target += (int64_t)vocab;
            }
            float base = 0.01f * (float)((int64_t)valid_prefix_len + (int64_t)pos) + mask_diag * 0.0f;
            for (size_t v = 0; v < vocab; ++v) {
                float diff = (float)((int64_t)v - target);
                row_out[v] = base - 0.001f * fabsf(diff);
            }
        }
        out[0] = out_buf;
    } else {
        float* step_k_out = (float*)out_k->data;
        float* step_v_out = (float*)out_v->data;

        int64_t prefix_accum = 0;
        for (int32_t p = 0; p < valid_prefix_len; ++p) {
            prefix_accum += llroundf(kv_k_prefix[(size_t)p * kv_elems]);
            prefix_accum += llroundf(kv_v_prefix[(size_t)p * kv_elems]);
        }

        for (size_t r = 0; r < rows; ++r) {
            step_k_out[r * kv_elems] = (float)token_tile[r];
            step_v_out[r * kv_elems] = (float)pos_tile[r];
            for (size_t d = 1; d < kv_elems; ++d) {
                step_k_out[r * kv_elems + d] = 0.01f * (float)((token_tile[r] + (int32_t)d) % 31);
                step_v_out[r * kv_elems + d] = 0.01f * (float)((pos_tile[r] + (int32_t)d) % 29);
            }

            int64_t tree_accum = 0;
            for (size_t c = 0; c < rows; ++c) {
                if (causal_mask[r * rows + c] > -1e4f) {
                    tree_accum += (int64_t)token_tile[c] + (int64_t)pos_tile[c];
                }
            }

            int64_t ctx = prefix_accum + tree_accum + (int64_t)token_tile[r];
            int64_t mod = (vocab > 85) ? 16 : (int64_t)vocab;
            int64_t offset = ((ctx % mod) + mod) % mod;
            size_t target = (vocab > 85) ? (size_t)(69 + offset) : (size_t)offset;

            float* row_out = logits + (r * vocab);
            for (size_t v = 0; v < vocab; ++v) {
                float diff = (float)((int64_t)v - (int64_t)target);
                row_out[v] = -0.001f * fabsf(diff);
            }
        }
        out[0] = out_buf;
        out[1] = out_k;
        out[2] = out_v;
    }

    if (args->device_complete_events) {
        args->device_complete_events[0] = make_ready_event();
    }
    return NULL;
}

static PJRT_Error* test_buffer_destroy(PJRT_Buffer_Destroy_Args* args) {
    if (PJRT_ARGS_HAVE(PJRT_Buffer_Destroy_Args, args, buffer) && args->buffer) {
        free(args->buffer->data);
        free(args->buffer);
    }
    return NULL;
}

static PJRT_Error* test_buffer_element_type(PJRT_Buffer_ElementType_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Buffer_ElementType_Args, args, type) || !args->buffer) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Buffer_ElementType");
    }
    args->type = args->buffer->type;
    return NULL;
}

static PJRT_Error* test_buffer_dimensions(PJRT_Buffer_Dimensions_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Buffer_Dimensions_Args, args, num_dims) || !args->buffer) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Buffer_Dimensions");
    }
    args->dims = args->buffer->dims;
    args->num_dims = args->buffer->num_dims;
    return NULL;
}

static PJRT_Error* test_buffer_to_host_buffer(PJRT_Buffer_ToHostBuffer_Args* args) {
    if (!PJRT_ARGS_HAVE(PJRT_Buffer_ToHostBuffer_Args, args, event) || !args->src) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Invalid args in PJRT_Buffer_ToHostBuffer");
    }
    if (!args->dst) {
        args->dst_size = args->src->byte_size;
        args->event = make_ready_event();
        return NULL;
    }
    if (args->dst_size < args->src->byte_size) {
        return make_err(PJRT_Error_Code_INVALID_ARGUMENT, "Destination host buffer too small");
    }
    if (args->src->byte_size > 0) {
        memcpy(args->dst, args->src->data, args->src->byte_size);
    }
    args->event = make_ready_event();
    return NULL;
}

static const PJRT_Api kTestPjrtApi = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .extension_start = NULL,
    .pjrt_api_version = {
        .struct_size = PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = NULL,
        .major_version = PJRT_API_MAJOR,
        .minor_version = PJRT_API_MINOR,
    },
    .PJRT_Error_Destroy = test_error_destroy,
    .PJRT_Error_Message = test_error_message,
    .PJRT_Error_GetCode = test_error_get_code,
    .PJRT_Plugin_Initialize = test_plugin_initialize,
    .PJRT_Plugin_Attributes = test_plugin_attributes,
    .PJRT_Event_Destroy = test_event_destroy,
    .PJRT_Event_IsReady = test_event_is_ready,
    .PJRT_Event_Error = test_event_error,
    .PJRT_Event_Await = test_event_await,
    .PJRT_Client_Create = test_client_create,
    .PJRT_Client_Destroy = test_client_destroy,
    .PJRT_Client_PlatformName = test_client_platform_name,
    .PJRT_Client_Devices = test_client_devices,
    .PJRT_Client_AddressableDevices = test_client_addressable_devices,
    .PJRT_Client_Compile = test_client_compile,
    .PJRT_Client_BufferFromHostBuffer = test_client_buffer_from_host_buffer,
    .PJRT_DeviceDescription_Id = test_device_description_id,
    .PJRT_DeviceDescription_ProcessIndex = test_device_description_process_index,
    .PJRT_DeviceDescription_Attributes = test_device_description_attributes,
    .PJRT_DeviceDescription_Kind = test_device_description_kind,
    .PJRT_DeviceDescription_DebugString = test_device_description_debug_string,
    .PJRT_DeviceDescription_ToString = test_device_description_to_string,
    .PJRT_Device_GetDescription = test_device_get_description,
    .PJRT_Device_IsAddressable = test_device_is_addressable,
    .PJRT_Device_MemoryStats = test_device_memory_stats,
    .PJRT_Executable_Destroy = test_raw_executable_destroy,
    .PJRT_Executable_NumOutputs = test_executable_num_outputs,
    .PJRT_LoadedExecutable_Destroy = test_executable_destroy,
    .PJRT_LoadedExecutable_GetExecutable = test_loaded_executable_get_executable,
    .PJRT_LoadedExecutable_Execute = test_executable_execute,
    .PJRT_Buffer_Destroy = test_buffer_destroy,
    .PJRT_Buffer_ElementType = test_buffer_element_type,
    .PJRT_Buffer_Dimensions = test_buffer_dimensions,
    .PJRT_Buffer_ToHostBuffer = test_buffer_to_host_buffer,
};

PJRT_EXPORT const PJRT_Api* GetPjrtApi(void) {
    return &kTestPjrtApi;
}
