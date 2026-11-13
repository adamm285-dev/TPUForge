#include "tpuforge/kv_cache.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace tpuforge {

namespace {

size_t checked_mul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string("KVCacheManager overflow in ") + what);
    }
    return a * b;
}

size_t dtype_size_bytes(PJRT_Buffer_Type dtype) {
    switch (dtype) {
        case PJRT_Buffer_Type_PRED:
        case PJRT_Buffer_Type_S8:
        case PJRT_Buffer_Type_U8:
        case PJRT_Buffer_Type_F8E5M2:
        case PJRT_Buffer_Type_F8E4M3FN:
        case PJRT_Buffer_Type_F8E4M3B11FNUZ:
        case PJRT_Buffer_Type_F8E5M2FNUZ:
        case PJRT_Buffer_Type_F8E4M3FNUZ:
        case PJRT_Buffer_Type_F8E4M3:
        case PJRT_Buffer_Type_F8E3M4:
        case PJRT_Buffer_Type_F8E8M0FNU:
            return 1;
        case PJRT_Buffer_Type_S16:
        case PJRT_Buffer_Type_U16:
        case PJRT_Buffer_Type_F16:
        case PJRT_Buffer_Type_BF16:
            return 2;
        case PJRT_Buffer_Type_S32:
        case PJRT_Buffer_Type_U32:
        case PJRT_Buffer_Type_F32:
            return 4;
        case PJRT_Buffer_Type_S64:
        case PJRT_Buffer_Type_U64:
        case PJRT_Buffer_Type_F64:
        case PJRT_Buffer_Type_C64:
            return 8;
        case PJRT_Buffer_Type_C128:
            return 16;
        default:
            throw std::invalid_argument("Unsupported or invalid PJRT_Buffer_Type for KV cache");
    }
}

} // namespace

KVCacheManager::KVCacheManager(const Config& config, std::shared_ptr<Device> device)
    : config_(config), device_(std::move(device)) {
    if (!device_) {
        throw std::invalid_argument("KVCacheManager requires a non-null device");
    }
    if (config_.page_size == 0 || config_.num_layers == 0 ||
        config_.num_kv_heads == 0 || config_.head_dim == 0) {
        throw std::invalid_argument("KVCacheManager dimensions and page_size must be non-zero");
    }
    (void)bytes_per_page();
}

KVCacheManager::~KVCacheManager() = default;

size_t KVCacheManager::bytes_per_page() const {
    const size_t elem_size = dtype_size_bytes(config_.dtype);
    size_t elems = checked_mul(2, config_.num_layers, "layers");
    elems = checked_mul(elems, config_.num_kv_heads, "kv_heads");
    elems = checked_mul(elems, config_.head_dim, "head_dim");
    elems = checked_mul(elems, config_.page_size, "page_size");
    return checked_mul(elems, elem_size, "bytes_per_page");
}

bool KVCacheManager::allocate_pool(size_t total_pages) {
    if (free_pages_.size() != total_pages_) {
        throw std::logic_error("Cannot re-initialize KV cache pool while pages are still in use");
    }
    if (total_pages > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("total_pages exceeds int32 page identifier range");
    }

    const size_t total_bytes = checked_mul(total_pages, bytes_per_page(), "total_pool_bytes");
    std::vector<int32_t> next_free(total_pages);
    std::iota(next_free.begin(), next_free.end(), int32_t{0});
    std::vector<uint8_t> next_in_use(total_pages, 0);
    free_pages_ = std::move(next_free);
    page_in_use_ = std::move(next_in_use);
    total_pages_ = total_pages;

    const double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "[TPUForge::KVCache] Initialized host page-index pool: " << total_pages
              << " logical pages (" << total_gb << " GB tracked capacity) for "
              << device_->info().kind << std::endl;
    return true;
}

std::vector<int32_t> KVCacheManager::allocate_sequence_pages(size_t num_tokens) {
    if (num_tokens == 0) {
        return {};
    }
    const size_t pages_needed = (num_tokens / config_.page_size) +
                                ((num_tokens % config_.page_size) != 0 ? 1 : 0);
    if (free_pages_.size() < pages_needed) {
        throw std::runtime_error(
            "KVCacheManager out of pages: requested " + std::to_string(pages_needed) +
            ", available " + std::to_string(free_pages_.size())
        );
    }

    std::vector<int32_t> allocated;
    allocated.reserve(pages_needed);
    for (size_t i = 0; i < pages_needed; ++i) {
        int32_t id = free_pages_.back();
        free_pages_.pop_back();
        page_in_use_[static_cast<size_t>(id)] = 1;
        allocated.push_back(id);
    }
    return allocated;
}

void KVCacheManager::release_pages(const std::vector<int32_t>& page_ids) {
    if (page_ids.empty()) {
        return;
    }
    std::vector<int32_t> sorted_ids(page_ids);
    std::sort(sorted_ids.begin(), sorted_ids.end());
    if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end()) {
        throw std::invalid_argument("KVCacheManager::release_pages received duplicate page_id");
    }
    for (int32_t id : sorted_ids) {
        if (id < 0 || static_cast<size_t>(id) >= total_pages_) {
            throw std::out_of_range("KVCacheManager::release_pages page_id out of bounds");
        }
        if (page_in_use_[static_cast<size_t>(id)] == 0) {
            throw std::logic_error("KVCacheManager::release_pages double-free detected");
        }
    }
    for (int32_t id : page_ids) {
        page_in_use_[static_cast<size_t>(id)] = 0;
        free_pages_.push_back(id);
    }
}

IciRingPlan IciRingPlanner::plan(const KVCacheManager::Config& kv_config, size_t ring_size) {
    if (ring_size == 0 || ring_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("IciRingPlanner::plan requires 1 <= ring_size <= INT32_MAX");
    }
    if (kv_config.num_layers == 0 || kv_config.num_kv_heads == 0 ||
        kv_config.head_dim == 0 || kv_config.page_size == 0 || kv_config.max_seq_len == 0) {
        throw std::invalid_argument("IciRingPlanner::plan requires all dimensions and max_seq_len > 0");
    }
    if (kv_config.max_seq_len > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::length_error("IciRingPlanner::plan max_seq_len exceeds INT32_MAX");
    }
    if (kv_config.max_seq_len % ring_size != 0) {
        throw std::invalid_argument("IciRingPlanner::plan max_seq_len must be divisible by ring_size");
    }
    const size_t chunk_seq_len = kv_config.max_seq_len / ring_size;
    if (chunk_seq_len % kv_config.page_size != 0) {
        throw std::invalid_argument("IciRingPlanner::plan chunk_seq_len must be divisible by page_size");
    }

    const size_t elem_size = dtype_size_bytes(kv_config.dtype);
    size_t bytes_per_rank = checked_mul(2, kv_config.num_layers, "ring_layers");
    bytes_per_rank = checked_mul(bytes_per_rank, kv_config.num_kv_heads, "ring_kv_heads");
    bytes_per_rank = checked_mul(bytes_per_rank, kv_config.head_dim, "ring_head_dim");
    bytes_per_rank = checked_mul(bytes_per_rank, chunk_seq_len, "ring_chunk_seq_len");
    bytes_per_rank = checked_mul(bytes_per_rank, elem_size, "bytes_per_rank");
    const size_t total_cluster_bytes = checked_mul(bytes_per_rank, ring_size, "total_cluster_bytes");

    IciRingPlan p;
    p.ring_size = ring_size;
    p.total_seq_len = kv_config.max_seq_len;
    p.chunk_seq_len = chunk_seq_len;
    p.pages_per_rank = chunk_seq_len / kv_config.page_size;
    p.bytes_per_rank = bytes_per_rank;
    p.total_cluster_bytes = total_cluster_bytes;
    p.send_perm.reserve(ring_size);
    p.chunk_start_positions.reserve(ring_size);

    for (size_t r = 0; r < ring_size; ++r) {
        p.send_perm.emplace_back(
            static_cast<int32_t>(r),
            static_cast<int32_t>((r + 1) % ring_size)
        );
        p.chunk_start_positions.push_back(static_cast<int32_t>(r * chunk_seq_len));
    }
    return p;
}

size_t IciRingPlanner::source_rank_at_step(size_t dst_rank, size_t step, size_t ring_size) {
    if (ring_size == 0) {
        throw std::invalid_argument("IciRingPlanner::source_rank_at_step requires ring_size >= 1");
    }
    if (dst_rank >= ring_size) {
        throw std::out_of_range("IciRingPlanner::source_rank_at_step dst_rank >= ring_size");
    }
    return (dst_rank + ring_size - (step % ring_size)) % ring_size;
}

std::vector<uint8_t> IciRingPlanner::causal_active_hops(
    size_t query_rank,
    size_t ring_size,
    size_t chunk_seq_len,
    size_t valid_seq_len
) {
    if (ring_size == 0 || chunk_seq_len == 0) {
        throw std::invalid_argument("IciRingPlanner::causal_active_hops requires ring_size >= 1 and chunk_seq_len >= 1");
    }
    if (query_rank >= ring_size) {
        throw std::out_of_range("IciRingPlanner::causal_active_hops query_rank >= ring_size");
    }

    std::vector<uint8_t> active(ring_size, 0);
    const size_t q_start = checked_mul(query_rank, chunk_seq_len, "q_start");
    if (valid_seq_len == 0 || q_start >= valid_seq_len) {
        return active;
    }

    const size_t q_end = std::min(checked_mul(query_rank + 1, chunk_seq_len, "q_end"), valid_seq_len);
    const size_t q_max_pos = q_end - 1;
    for (size_t step = 0; step < ring_size; ++step) {
        const size_t kv_rank = source_rank_at_step(query_rank, step, ring_size);
        const size_t kv_min_pos = checked_mul(kv_rank, chunk_seq_len, "kv_min_pos");
        active[step] = (kv_min_pos <= q_max_pos && kv_min_pos < valid_seq_len) ? 1 : 0;
    }
    return active;
}

} // namespace tpuforge
