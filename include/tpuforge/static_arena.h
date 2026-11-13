#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tpuforge {

struct SystolicConstants {
    static constexpr size_t kSystolicTileSize = 128;
    static constexpr size_t kArenaAlignment = 128;
};

class StaticArena {
public:
    struct Config {
        size_t max_seq_len{8192};
        size_t hidden_dim{128};
        size_t num_kv_heads{8};
        size_t head_dim{128};
        size_t vocab_size{32000};
    };

    explicit StaticArena(const Config& config);
    ~StaticArena();

    StaticArena(const StaticArena&) = delete;
    StaticArena& operator=(const StaticArena&) = delete;
    StaticArena(StaticArena&& other) noexcept;
    StaticArena& operator=(StaticArena&& other) noexcept;

    const Config& config() const { return config_; }
    size_t kv_row_elems() const { return config_.num_kv_heads * config_.head_dim; }

    int32_t* token_tile_128() { return token_tile_128_; }
    const int32_t* token_tile_128() const { return token_tile_128_; }

    int32_t* position_tile_128() { return position_tile_128_; }
    const int32_t* position_tile_128() const { return position_tile_128_; }

    float* causal_mask_128x128() { return causal_mask_128x128_; }
    const float* causal_mask_128x128() const { return causal_mask_128x128_; }

    float* kv_prefix_k() { return kv_prefix_k_; }
    const float* kv_prefix_k() const { return kv_prefix_k_; }

    float* kv_prefix_v() { return kv_prefix_v_; }
    const float* kv_prefix_v() const { return kv_prefix_v_; }

    float* logits_tile_128() { return logits_tile_128_; }
    const float* logits_tile_128() const { return logits_tile_128_; }

    int32_t valid_prefix_len() const { return valid_prefix_len_; }
    void set_valid_prefix_len(int32_t len);

    // Copies accepted speculative KV rows from a 128-row step tile into the static prefix arena.
    void commit_accepted_kv_rows(
        const float* step_k_tile,
        const float* step_v_tile,
        const std::vector<int32_t>& accepted_row_indices
    );

    bool is_aligned_128() const;

private:
    static void* allocate_aligned(size_t bytes);
    static void free_aligned(void* ptr);
    void release_all();

    Config config_{};
    int32_t* token_tile_128_{nullptr};
    int32_t* position_tile_128_{nullptr};
    float* causal_mask_128x128_{nullptr};
    float* kv_prefix_k_{nullptr};
    float* kv_prefix_v_{nullptr};
    float* logits_tile_128_{nullptr};
    int32_t valid_prefix_len_{0};
};

} // namespace tpuforge
