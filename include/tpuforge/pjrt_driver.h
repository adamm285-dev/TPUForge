#pragma once

#include "tpuforge/device.h"
#include "tpuforge/pjrt_c_api.h"
#include "tpuforge/static_arena.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tpuforge {

class PjrtDriver {
public:
    PjrtDriver() = default;
    ~PjrtDriver();

    PjrtDriver(const PjrtDriver&) = delete;
    PjrtDriver& operator=(const PjrtDriver&) = delete;

    bool load_plugin(const std::string& path, std::string* error_out = nullptr);
    void unload();

    bool is_loaded() const { return api_ != nullptr && client_ != nullptr; }
    const PJRT_Api* api() const { return api_; }
    PJRT_Client* client() const { return client_; }
    std::string platform_name() const;
    std::vector<DeviceInfo> query_devices() const;

private:
    std::string format_and_destroy_error(PJRT_Error* err) const;

    void* lib_handle_{nullptr};
    const PJRT_Api* api_{nullptr};
    PJRT_Client* client_{nullptr};
};

class ZeroRecompileExecutor {
public:
    struct StaticShapeKey {
        size_t tile_rows{SystolicConstants::kSystolicTileSize};
        size_t max_seq_len{0};
        size_t hidden_dim{0};
        size_t vocab_size{0};

        bool operator==(const StaticShapeKey& other) const {
            return tile_rows == other.tile_rows &&
                   max_seq_len == other.max_seq_len &&
                   hidden_dim == other.hidden_dim &&
                   vocab_size == other.vocab_size;
        }
    };

    explicit ZeroRecompileExecutor(PjrtDriver& driver);
    ~ZeroRecompileExecutor();

    ZeroRecompileExecutor(const ZeroRecompileExecutor&) = delete;
    ZeroRecompileExecutor& operator=(const ZeroRecompileExecutor&) = delete;

    bool compile_once(
        const StaticArena::Config& config,
        const std::string& program_code,
        const std::string& program_format = "stablehlo",
        std::string* error_out = nullptr
    );

    bool execute_step(
        StaticArena& arena,
        size_t tile_rows = SystolicConstants::kSystolicTileSize,
        std::string* error_out = nullptr
    );

    size_t compile_count() const { return compile_count_; }
    size_t execute_count() const { return execute_count_; }
    bool is_compiled() const { return executable_ != nullptr; }

private:
    void release_executable();

    PjrtDriver& driver_;
    PJRT_LoadedExecutable* executable_{nullptr};
    StaticShapeKey compiled_key_{};
    size_t compile_count_{0};
    size_t execute_count_{0};
};

} // namespace tpuforge
