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

int32_t infer_mxu_dim(const std::string& device_kind) {
    std::string lower;
    lower.reserve(device_kind.size());
    for (char ch : device_kind) {
        if (ch >= 'A' && ch <= 'Z') {
            lower.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            lower.push_back(ch);
        }
    }
    const char* patterns_256[] = {
        "v6", "trillium", "v7", "tpu7", "ironwood", "v8", "tpu 8", "tpu8", "8i", "8t"
    };
    for (const char* pat : patterns_256) {
        if (lower.find(pat) != std::string::npos) {
            return 256;
        }
    }
    return 128;
}

Device::Device(int32_t id, std::string kind, size_t memory_bytes) {
    info_.id = id;
    info_.kind = std::move(kind);
    info_.memory_bytes = memory_bytes;
    info_.is_tpu = (info_.kind.find("TPU") != std::string::npos);
    const int32_t mxu = infer_mxu_dim(info_.kind);
    info_.mxu_dim_x = mxu;
    info_.mxu_dim_y = mxu;
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
