#include "tpuforge/device.h"

#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace tpuforge {

namespace {

size_t query_host_physical_memory_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX mem_status{};
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        return static_cast<size_t>(mem_status.ullTotalPhys);
    }
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
    return 0;
#endif
}

} // namespace

Device::Device(int32_t id, std::string kind, size_t memory_bytes) {
    info_.id = id;
    info_.kind = std::move(kind);
    info_.memory_bytes = memory_bytes;
    info_.is_tpu = (info_.kind.find("TPU") != std::string::npos);
    info_.mxu_dim_x = 128;
    info_.mxu_dim_y = 128;
}

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::~DeviceManager() = default;

bool DeviceManager::initialize(const std::string& pjrt_plugin_path) {
    std::scoped_lock lk(mu_);

    if (!pjrt_plugin_path.empty()) {
        std::cerr << "[TPUForge] DeviceManager::initialize() only discovers host devices; "
                  << "use PjrtDriver::load_plugin() for PJRT plugins (" << pjrt_plugin_path << ")."
                  << std::endl;
        return false;
    }

    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) {
        cores = 1;
    }
    size_t ram_bytes = query_host_physical_memory_bytes();
    if (ram_bytes == 0) {
        std::cerr << "[TPUForge] Warning: OS physical memory probe returned 0 bytes." << std::endl;
    }
    std::string cpu_kind = "Host CPU (" + std::to_string(cores) + " threads)";

    std::vector<std::shared_ptr<Device>> next_devices;
    next_devices.push_back(std::make_shared<Device>(0, cpu_kind, ram_bytes));
    std::cout << "[TPUForge] Initialized local host device: " << cpu_kind << std::endl;

    devices_ = std::move(next_devices);
    initialized_ = true;
    return true;
}

std::vector<std::shared_ptr<Device>> DeviceManager::devices() const {
    std::scoped_lock lk(mu_);
    return devices_;
}

std::shared_ptr<Device> DeviceManager::default_device() {
    std::scoped_lock lk(mu_);
    if (!initialized_ || devices_.empty()) {
        throw std::logic_error("DeviceManager is not initialized; call initialize() first");
    }
    return devices_.front();
}

bool DeviceManager::is_initialized() const {
    std::scoped_lock lk(mu_);
    return initialized_;
}

} // namespace tpuforge
