#pragma once

#include "tpuforge/pjrt_c_api.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace tpuforge {

struct DeviceInfo {
    int32_t id;
    std::string kind;
    size_t memory_bytes;
    bool is_tpu;
    int32_t mxu_dim_x{128}; // TPU standard systolic array dimension
    int32_t mxu_dim_y{128};
};

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
    
    // Discover available execution backends (TPU / CPU fallback)
    bool initialize(const std::string& pjrt_plugin_path = "");
    
    const std::vector<std::shared_ptr<Device>>& devices() const { return devices_; }
    std::shared_ptr<Device> default_device();

private:
    DeviceManager() = default;
    std::vector<std::shared_ptr<Device>> devices_;
    bool initialized_{false};
};

} // namespace tpuforge
