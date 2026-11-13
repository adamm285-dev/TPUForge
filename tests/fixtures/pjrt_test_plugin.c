#include "tpuforge/pjrt_c_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define PJRT_EXPORT __declspec(dllexport)
#else
#define PJRT_EXPORT __attribute__((visibility("default")))
#endif

struct PJRT_Device {
    PJRT_DeviceDescription desc;
};

struct PJRT_Client {
    struct PJRT_Device devices[2];
    PJRT_Device* device_ptrs[2];
    size_t compile_calls;
};

struct PJRT_LoadedExecutable {
    struct PJRT_Client* client;
    size_t tile_rows;
    size_t max_seq_len;
    size_t hidden_dim;
    size_t vocab_size;
};

struct PJRT_Error {
    char message[128];
};

static PJRT_Error* make_err(const char* msg) {
    PJRT_Error* e = (PJRT_Error*)malloc(sizeof(PJRT_Error));
    if (e) {
        snprintf(e->message, sizeof(e->message), "%s", msg);
    }
    return e;
}

static PJRT_Error* test_error_destroy(PJRT_Error_Destroy_Args* args) {
    if (args && args->error) {
        free(args->error);
    }
    return NULL;
}

static void test_error_message(PJRT_Error_Message_Args* args) {
    if (args && args->error) {
        args->message = args->error->message;
        args->message_size = strlen(args->error->message);
    }
}

static PJRT_Error* test_client_create(PJRT_Client_Create_Args* args) {
    if (!args) {
        return make_err("Null args in PJRT_Client_Create");
    }
    struct PJRT_Client* c = (struct PJRT_Client*)calloc(1, sizeof(struct PJRT_Client));
    if (!c) {
        return make_err("Allocation failed in PJRT_Client_Create");
    }
    const size_t hbm_32gb = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    c->devices[0].desc.device_id = 0;
    c->devices[0].desc.process_index = 0;
    c->devices[0].desc.device_kind = "Google TPU v6e";
    c->devices[0].desc.debug_string = "TPU_0(process=0,trillium=v6e)";
    c->devices[0].desc.memory_bytes = hbm_32gb;

    c->devices[1].desc.device_id = 1;
    c->devices[1].desc.process_index = 0;
    c->devices[1].desc.device_kind = "Google TPU v6e";
    c->devices[1].desc.debug_string = "TPU_1(process=0,trillium=v6e)";
    c->devices[1].desc.memory_bytes = hbm_32gb;

    c->device_ptrs[0] = &c->devices[0];
    c->device_ptrs[1] = &c->devices[1];
    args->client = c;
    return NULL;
}

static PJRT_Error* test_client_destroy(PJRT_Client_Destroy_Args* args) {
    if (args && args->client) {
        free(args->client);
    }
    return NULL;
}

static PJRT_Error* test_client_platform_name(PJRT_Client_PlatformName_Args* args) {
    if (!args) {
        return make_err("Null args in PJRT_Client_PlatformName");
    }
    static const char kPlatform[] = "tpu";
    args->platform_name = kPlatform;
    args->platform_name_size = sizeof(kPlatform) - 1;
    return NULL;
}

static PJRT_Error* test_client_devices(PJRT_Client_Devices_Args* args) {
    if (!args || !args->client) {
        return make_err("Invalid client in PJRT_Client_Devices");
    }
    args->devices = args->client->device_ptrs;
    args->num_devices = 2;
    return NULL;
}

static PJRT_Error* test_device_get_description(PJRT_Device_GetDescription_Args* args) {
    if (!args || !args->device) {
        return make_err("Invalid device in PJRT_Device_GetDescription");
    }
    args->description = args->device->desc;
    return NULL;
}

static PJRT_Error* test_client_compile(PJRT_Client_Compile_Args* args) {
    if (!args || !args->client) {
        return make_err("Invalid client in PJRT_Client_Compile");
    }
    if (args->tile_rows != 128) {
        return make_err("TPU systolic compiler requires tile_rows == 128");
    }
    struct PJRT_LoadedExecutable* exec =
        (struct PJRT_LoadedExecutable*)calloc(1, sizeof(struct PJRT_LoadedExecutable));
    if (!exec) {
        return make_err("Allocation failed in PJRT_Client_Compile");
    }
    exec->client = args->client;
    exec->tile_rows = args->tile_rows;
    exec->max_seq_len = args->max_seq_len;
    exec->hidden_dim = args->hidden_dim;
    exec->vocab_size = args->vocab_size;
    args->client->compile_calls += 1;
    args->executable = exec;
    return NULL;
}

static PJRT_Error* test_executable_destroy(PJRT_LoadedExecutable_Destroy_Args* args) {
    if (args && args->executable) {
        free(args->executable);
    }
    return NULL;
}

static PJRT_Error* test_executable_execute(PJRT_LoadedExecutable_Execute_Args* args) {
    if (!args || !args->executable || !args->token_tile || !args->position_tile ||
        !args->causal_mask || !args->out_logits) {
        return make_err("Invalid arguments in PJRT_LoadedExecutable_Execute");
    }
    if (args->tile_rows != args->executable->tile_rows ||
        args->vocab_size != args->executable->vocab_size) {
        return make_err("Shape mismatch in PJRT_LoadedExecutable_Execute");
    }

    const size_t rows = args->tile_rows;
    const size_t vocab = args->vocab_size;
    for (size_t r = 0; r < rows; ++r) {
        int32_t tok = args->token_tile[r];
        int32_t pos = args->position_tile[r];
        float mask_diag = args->causal_mask[r * rows + r];
        float* row_out = args->out_logits + (r * vocab);

        float base = 0.01f * (float)(args->valid_prefix_len + pos) + mask_diag * 0.0f;
        for (size_t v = 0; v < vocab; ++v) {
            float diff = (float)((int32_t)v - ((tok + pos + 1) % (int32_t)vocab));
            row_out[v] = base - 0.001f * fabsf(diff);
        }
    }
    return NULL;
}

static const PJRT_Api kTestPjrtApi = {
    sizeof(PJRT_Api),
    {sizeof(PJRT_Api_Version), 0, 54},
    test_error_destroy,
    test_error_message,
    test_client_create,
    test_client_destroy,
    test_client_platform_name,
    test_client_devices,
    test_device_get_description,
    test_client_compile,
    test_executable_destroy,
    test_executable_execute,
};

PJRT_EXPORT const PJRT_Api* GetPjrtApi(void) {
    return &kTestPjrtApi;
}
