#include "tpuforge/device.h"
#include <iostream>

namespace tpuforge {

// PJRT dynamic loader stub for linking with libpjrt_tpu.so or libpjrt_cpu.so
class PjrtDriver {
public:
    static bool load_plugin(const std::string& path) {
        std::cout << "[TPUForge::PJRT] Hooking OpenXLA PJRT C API via: " << path << std::endl;
        return true;
    }
};

} // namespace tpuforge
