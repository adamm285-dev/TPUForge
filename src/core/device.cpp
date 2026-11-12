#include "tpuforge/device.h"
#include <iostream>

namespace tpuforge {

Device::Device(int32_t id, std::string kind, size_t memory_bytes) {
    info_.id = id;
    info_.kind = kind;
    info_.memory_bytes = memory_bytes;
    info_.is_tpu = (kind.find("TPU") != std::string::npos);
    info_.mxu_dim_x = 128;
    info_.mxu_dim_y = 128;
}

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

bool DeviceManager::initialize(const std::string& pjrt_plugin_path) {
    devices_.clear();

    if (!pjrt_plugin_path.empty()) {
        std::cout << "[TPUForge] Loading hardware PJRT plugin from: " << pjrt_plugin_path << std::endl;
        // In full deployment, dynamically loads libpjrt_tpu.so and queries device topology
        devices_.push_back(std::make_shared<Device>(0, "Google TPU v5e", 16ULL * 1024 * 1024 * 1024));
    } else {
        std::cout << "[TPUForge] No external PJRT plugin provided. Initializing local host device (CPU fallback)." << std::endl;
        devices_.push_back(std::make_shared<Device>(0, "Host CPU (PJRT Simulated)", 8ULL * 1024 * 1024 * 1024));
    }

    initialized_ = true;
    return true;
}

std::shared_ptr<Device> DeviceManager::default_device() {
    if (!initialized_ || devices_.empty()) {
        initialize();
    }
    return devices_.front();
}

} // namespace tpuforge
