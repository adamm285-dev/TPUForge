#pragma once

#include "tpuforge/device.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace tpuforge {

class KVCacheManager {
public:
    struct Config {
        size_t num_layers{32};
        size_t num_kv_heads{8};
        size_t head_dim{128};     // Aligned to 128 for TPU MXU systolic matrix layout
        size_t max_seq_len{8192};
        size_t page_size{16};     // Number of tokens per page
        PJRT_Buffer_Type dtype{PJRT_Buffer_Type_BF16};
    };

    explicit KVCacheManager(const Config& config, std::shared_ptr<Device> device);
    ~KVCacheManager();

    // Initialize logical host-side page table entries for HBM pool tracking
    // (does not allocate device HBM buffers directly).
    bool allocate_pool(size_t total_pages);

    // Reserve pages for a sequence (throws on overflow or insufficient free pages).
    std::vector<int32_t> allocate_sequence_pages(size_t num_tokens);

    // Free sequence pages back to the pool with bounds and double-free validation.
    void release_pages(const std::vector<int32_t>& page_ids);

    size_t free_page_count() const { return free_pages_.size(); }
    size_t total_page_count() const { return total_pages_; }
    size_t bytes_per_page() const;

private:
    Config config_;
    std::shared_ptr<Device> device_;
    size_t total_pages_{0};
    std::vector<int32_t> free_pages_;
    std::vector<uint8_t> page_in_use_;
};

struct IciRingPlan {
    size_t ring_size{1};
    size_t total_seq_len{0};
    size_t chunk_seq_len{0};
    size_t pages_per_rank{0};
    size_t bytes_per_rank{0};
    size_t total_cluster_bytes{0};
    std::vector<std::pair<int32_t, int32_t>> send_perm;
    std::vector<int32_t> chunk_start_positions;
};

class IciRingPlanner {
public:
    static IciRingPlan plan(const KVCacheManager::Config& kv_config, size_t ring_size);
    static size_t source_rank_at_step(size_t dst_rank, size_t step, size_t ring_size);
    static std::vector<uint8_t> causal_active_hops(
        size_t query_rank,
        size_t ring_size,
        size_t chunk_seq_len,
        size_t valid_seq_len
    );
};

} // namespace tpuforge
