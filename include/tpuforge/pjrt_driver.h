#pragma once

#include "tpuforge/device.h"
#include "tpuforge/pjrt_c_api.h"
#include "tpuforge/static_arena.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

namespace tpuforge {

class ZeroRecompileExecutor;

struct StaticShapeKey {
    size_t tile_rows{SystolicConstants::kSystolicTileSize};
    size_t max_seq_len{0};
    size_t hidden_dim{0};
    size_t num_kv_heads{0};
    size_t head_dim{0};
    size_t vocab_size{0};
    std::string program_format;
    std::string program_code;

    bool operator==(const StaticShapeKey& other) const = default;
};

class PjrtDriver {
public:
    PjrtDriver() = default;
    ~PjrtDriver();

    PjrtDriver(const PjrtDriver&) = delete;
    PjrtDriver& operator=(const PjrtDriver&) = delete;
    PjrtDriver(PjrtDriver&&) = delete;
    PjrtDriver& operator=(PjrtDriver&&) = delete;

    // Transactional dynamic load of a PJRT C API plugin (.dll / .so).
    // Preserves existing loaded state if the new plugin fails to load or initialize.
    bool load_plugin(const std::string& library_path, std::string* error_out = nullptr);
    void unload();

    bool is_loaded() const;
    std::string platform_name() const;
    std::vector<DeviceInfo> query_devices() const;

    const PJRT_Api* api() const;
    PJRT_Client* client() const;

private:
    friend class ZeroRecompileExecutor;

    void register_executor(ZeroRecompileExecutor* exec);
    void unregister_executor(ZeroRecompileExecutor* exec);
    void unregister_and_destroy_executable(ZeroRecompileExecutor* exec);

    mutable std::shared_mutex mu_;
    std::mutex exec_reg_mu_;
    void* lib_handle_{nullptr};
    const PJRT_Api* api_{nullptr};
    PJRT_Client* client_{nullptr};
    std::string loaded_path_;
    std::string platform_name_;
    std::vector<DeviceInfo> devices_;
    std::vector<ZeroRecompileExecutor*> executors_;
};

class ZeroRecompileExecutor {
public:
    explicit ZeroRecompileExecutor(PjrtDriver& driver);
    ~ZeroRecompileExecutor();

    ZeroRecompileExecutor(const ZeroRecompileExecutor&) = delete;
    ZeroRecompileExecutor& operator=(const ZeroRecompileExecutor&) = delete;
    ZeroRecompileExecutor(ZeroRecompileExecutor&&) = delete;
    ZeroRecompileExecutor& operator=(ZeroRecompileExecutor&&) = delete;

    // Compiles the static systolic XLA / StableHLO module ONCE per StaticShapeKey.
    // Subsequent calls with an identical StaticShapeKey reuse the compiled executable.
    bool compile_once(
        const StaticArenaConfig& config,
        const std::string& program_code,
        const std::string& format = "mlir",
        std::string* error_out = nullptr
    );

    // Executes one systolic step via official PJRT host->device buffers and execution.
    // Throws std::invalid_argument if tile_rows != 128 to prevent shape polymorphism.
    bool execute_step(
        StaticArena& arena,
        size_t tile_rows = SystolicConstants::kSystolicTileSize,
        std::string* error_out = nullptr
    );

    // Executes one stateful systolic step with prefix KV buffers uploaded [max_seq_len, kv_row_elems]
    // and step KV outputs [128, kv_row_elems] downloaded alongside logits [128, vocab_size].
    bool execute_step_with_kv(
        StaticArena& arena,
        std::vector<float>* out_step_k,
        std::vector<float>* out_step_v,
        size_t tile_rows = SystolicConstants::kSystolicTileSize,
        std::string* error_out = nullptr
    );

    bool is_compiled() const;
    size_t compile_count() const;
    size_t execute_count() const;
    size_t num_outputs() const;
    StaticShapeKey shape_key() const;

private:
    friend class PjrtDriver;

    bool execute_step_impl(
        StaticArena& arena,
        std::vector<float>* out_step_k,
        std::vector<float>* out_step_v,
        size_t tile_rows,
        std::string* error_out
    );

    // Caller must hold PjrtDriver::mu_ (unique) and PjrtDriver::exec_reg_mu_.
    void invalidate_with_api(const PJRT_Api* api, bool detach_driver);

    std::atomic<PjrtDriver*> driver_{nullptr};
    mutable std::mutex mu_;
    PJRT_LoadedExecutable* executable_{nullptr};
    size_t num_outputs_{1};
    StaticShapeKey shape_key_;
    bool compiled_{false};
    size_t compile_count_{0};
    size_t execute_count_{0};
};

} // namespace tpuforge
