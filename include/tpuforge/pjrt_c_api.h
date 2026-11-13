#ifndef TPUFORGE_PJRT_C_API_H_
#define TPUFORGE_PJRT_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes for PJRT C API
typedef enum {
    PJRT_SUCCESS = 0,
    PJRT_ERROR_CANCELLED = 1,
    PJRT_ERROR_UNKNOWN = 2,
    PJRT_ERROR_INVALID_ARGUMENT = 3,
    PJRT_ERROR_NOT_FOUND = 4,
    PJRT_ERROR_ALREADY_EXISTS = 5,
    PJRT_ERROR_RESOURCE_EXHAUSTED = 6,
    PJRT_ERROR_UNAVAILABLE = 7,
    PJRT_ERROR_UNIMPLEMENTED = 8,
    PJRT_ERROR_INTERNAL = 9
} PJRT_ErrorCode;

// Opaque handles
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Error PJRT_Error;

// Primitive types for tensor layouts
typedef enum {
    PJRT_DTYPE_INVALID = 0,
    PJRT_DTYPE_PRED = 1,
    PJRT_DTYPE_S8 = 2,
    PJRT_DTYPE_S16 = 3,
    PJRT_DTYPE_S32 = 4,
    PJRT_DTYPE_S64 = 5,
    PJRT_DTYPE_U8 = 6,
    PJRT_DTYPE_U16 = 7,
    PJRT_DTYPE_U32 = 8,
    PJRT_DTYPE_U64 = 9,
    PJRT_DTYPE_F16 = 10,
    PJRT_DTYPE_F32 = 11,
    PJRT_DTYPE_F64 = 12,
    PJRT_DTYPE_BF16 = 13 // Primary datatype for Google TPU MXUs
} PJRT_Buffer_Type;

// Device description
typedef struct {
    int32_t device_id;
    int32_t process_index;
    const char* device_kind; // e.g. "TPU v4", "TPU v5e", "TPU v6e", "CPU"
    const char* debug_string;
    size_t memory_bytes;
} PJRT_DeviceDescription;

typedef struct {
    size_t struct_size;
    int32_t major_version;
    int32_t minor_version;
} PJRT_Api_Version;

typedef struct {
    size_t struct_size;
    PJRT_Error* error;
} PJRT_Error_Destroy_Args;

typedef struct {
    size_t struct_size;
    const PJRT_Error* error;
    const char* message;
    size_t message_size;
} PJRT_Error_Message_Args;

typedef struct {
    size_t struct_size;
    PJRT_Client* client;
} PJRT_Client_Create_Args;

typedef struct {
    size_t struct_size;
    PJRT_Client* client;
} PJRT_Client_Destroy_Args;

typedef struct {
    size_t struct_size;
    PJRT_Client* client;
    const char* platform_name;
    size_t platform_name_size;
} PJRT_Client_PlatformName_Args;

typedef struct {
    size_t struct_size;
    PJRT_Client* client;
    PJRT_Device* const* devices;
    size_t num_devices;
} PJRT_Client_Devices_Args;

typedef struct {
    size_t struct_size;
    PJRT_Device* device;
    PJRT_DeviceDescription description;
} PJRT_Device_GetDescription_Args;

typedef struct {
    size_t struct_size;
    PJRT_Client* client;
    const char* program_code;
    size_t program_code_size;
    const char* program_format;
    size_t tile_rows;
    size_t max_seq_len;
    size_t hidden_dim;
    size_t vocab_size;
    PJRT_LoadedExecutable* executable;
} PJRT_Client_Compile_Args;

typedef struct {
    size_t struct_size;
    PJRT_LoadedExecutable* executable;
} PJRT_LoadedExecutable_Destroy_Args;

typedef struct {
    size_t struct_size;
    PJRT_LoadedExecutable* executable;
    const int32_t* token_tile;
    const int32_t* position_tile;
    const float* causal_mask;
    int32_t valid_prefix_len;
    size_t tile_rows;
    size_t vocab_size;
    float* out_logits;
} PJRT_LoadedExecutable_Execute_Args;

typedef PJRT_Error* (*PJRT_Error_Destroy_Fn)(PJRT_Error_Destroy_Args* args);
typedef void (*PJRT_Error_Message_Fn)(PJRT_Error_Message_Args* args);
typedef PJRT_Error* (*PJRT_Client_Create_Fn)(PJRT_Client_Create_Args* args);
typedef PJRT_Error* (*PJRT_Client_Destroy_Fn)(PJRT_Client_Destroy_Args* args);
typedef PJRT_Error* (*PJRT_Client_PlatformName_Fn)(PJRT_Client_PlatformName_Args* args);
typedef PJRT_Error* (*PJRT_Client_Devices_Fn)(PJRT_Client_Devices_Args* args);
typedef PJRT_Error* (*PJRT_Device_GetDescription_Fn)(PJRT_Device_GetDescription_Args* args);
typedef PJRT_Error* (*PJRT_Client_Compile_Fn)(PJRT_Client_Compile_Args* args);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Destroy_Fn)(PJRT_LoadedExecutable_Destroy_Args* args);
typedef PJRT_Error* (*PJRT_LoadedExecutable_Execute_Fn)(PJRT_LoadedExecutable_Execute_Args* args);

typedef struct PJRT_Api {
    size_t struct_size;
    PJRT_Api_Version pjrt_api_version;
    PJRT_Error_Destroy_Fn PJRT_Error_Destroy;
    PJRT_Error_Message_Fn PJRT_Error_Message;
    PJRT_Client_Create_Fn PJRT_Client_Create;
    PJRT_Client_Destroy_Fn PJRT_Client_Destroy;
    PJRT_Client_PlatformName_Fn PJRT_Client_PlatformName;
    PJRT_Client_Devices_Fn PJRT_Client_Devices;
    PJRT_Device_GetDescription_Fn PJRT_Device_GetDescription;
    PJRT_Client_Compile_Fn PJRT_Client_Compile;
    PJRT_LoadedExecutable_Destroy_Fn PJRT_LoadedExecutable_Destroy;
    PJRT_LoadedExecutable_Execute_Fn PJRT_LoadedExecutable_Execute;
} PJRT_Api;

typedef const PJRT_Api* (*PJRT_GetApiFn)(void);

#ifdef __cplusplus
}
#endif

#endif // TPUFORGE_PJRT_C_API_H_
