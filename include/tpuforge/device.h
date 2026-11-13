#pragma once

#include "tpuforge/pjrt_c_api.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tpuforge {

struct DeviceInfo {
    int32_t id;
    std::string kind;
    size_t memory_bytes;
    bool is_tpu;
    int32_t mxu_dim_x{128}; // TPU standard systolic array dimension
    int32_t mxu_dim_y{128};
};

/// Infer systolic MXU array width (128 or 256) from a PJRT/XLA device_kind string.
int32_t infer_mxu_dim(const std::string& device_kind);

class Device {
public:
    Device(int32_t id, std::string kind, size_t memory_bytes);
    ~Device() = default;

    const DeviceInfo& info() const { return info_; }
    bool is_tpu() const { return info_.is_tpu; }

private:
    DeviceInfo info_;
};

class DeviceManager {
public:
    static DeviceManager& instance();

    // Discover host CPU device (use PjrtDriver directly for PJRT plugin initialization)
    bool initialize(const std::string& pjrt_plugin_path = "");

    std::vector<std::shared_ptr<Device>> devices() const;
    std::shared_ptr<Device> default_device();
    bool is_initialized() const;

private:
    DeviceManager() = default;
    ~DeviceManager();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    mutable std::mutex mu_;
    std::vector<std::shared_ptr<Device>> devices_;
    bool initialized_{false};
};

} // namespace tpuforge
