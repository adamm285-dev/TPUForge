#include "tpuforge/device.h"
#include "tpuforge/pjrt_driver.h"

#include <iostream>
#include <thread>

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

bool DeviceManager::initialize(const std::string& pjrt_plugin_path) {
    devices_.clear();
    initialized_ = false;

    if (!pjrt_plugin_path.empty()) {
        std::cout << "[TPUForge] Loading hardware PJRT plugin from: " << pjrt_plugin_path << std::endl;
        PjrtDriver driver;
        std::string err;
        if (!driver.load_plugin(pjrt_plugin_path, &err)) {
            std::cerr << "[TPUForge] Failed to load PJRT plugin: " << err << std::endl;
            return false;
        }
        auto queried = driver.query_devices();
        if (queried.empty()) {
            std::cerr << "[TPUForge] PJRT plugin reported zero devices." << std::endl;
            return false;
        }
        for (const auto& d : queried) {
            devices_.push_back(std::make_shared<Device>(d.id, d.kind, d.memory_bytes));
        }
        initialized_ = true;
        return true;
    }

    unsigned int cores = std::thread::hardware_concurrency();
    if (cores == 0) {
        cores = 1;
    }
    size_t ram_bytes = query_host_physical_memory_bytes();
    std::string cpu_kind = "Host CPU (" + std::to_string(cores) + " threads)";

    std::cout << "[TPUForge] Initialized local host device: " << cpu_kind << std::endl;
    devices_.push_back(std::make_shared<Device>(0, cpu_kind, ram_bytes));
    initialized_ = true;
    return true;
}

std::shared_ptr<Device> DeviceManager::default_device() {
    if (!initialized_ || devices_.empty()) {
        initialize();
    }
    if (devices_.empty()) {
        return nullptr;
    }
    return devices_.front();
}

} // namespace tpuforge
