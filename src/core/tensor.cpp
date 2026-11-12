#include "tpuforge/device.h"
#include <vector>

namespace tpuforge {

class Tensor {
public:
    Tensor(std::vector<int64_t> shape, PJRT_Buffer_Type dtype, std::shared_ptr<Device> device)
        : shape_(std::move(shape)), dtype_(dtype), device_(device) {}

    const std::vector<int64_t>& shape() const { return shape_; }
    PJRT_Buffer_Type dtype() const { return dtype_; }

private:
    std::vector<int64_t> shape_;
    PJRT_Buffer_Type dtype_;
    std::shared_ptr<Device> device_;
};

} // namespace tpuforge
