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
} PJRT_DeviceDescription;

#ifdef __cplusplus
}
#endif

#endif // TPUFORGE_PJRT_C_API_H_
