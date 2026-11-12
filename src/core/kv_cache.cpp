#include "tpuforge/kv_cache.h"
#include <iostream>
#include <numeric>

namespace tpuforge {

KVCacheManager::KVCacheManager(const Config& config, std::shared_ptr<Device> device)
    : config_(config), device_(device) {}

KVCacheManager::~KVCacheManager() = default;

size_t KVCacheManager::bytes_per_page() const {
    // 2 (for K and V) * num_layers * num_kv_heads * head_dim * page_size * sizeof(bfloat16)
    size_t element_size = (config_.dtype == PJRT_DTYPE_BF16 || config_.dtype == PJRT_DTYPE_F16) ? 2 : 4;
    return 2 * config_.num_layers * config_.num_kv_heads * config_.head_dim * config_.page_size * element_size;
}

bool KVCacheManager::allocate_pool(size_t total_pages) {
    total_pages_ = total_pages;
    free_pages_.resize(total_pages);
    std::iota(free_pages_.begin(), free_pages_.end(), 0);

    size_t total_bytes = total_pages * bytes_per_page();
    double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "[TPUForge::KVCache] Allocated " << total_pages 
              << " pages (" << total_gb << " GB) on " 
              << device_->info().kind << std::endl;
    return true;
}

std::vector<int32_t> KVCacheManager::allocate_sequence_pages(size_t num_tokens) {
    size_t pages_needed = (num_tokens + config_.page_size - 1) / config_.page_size;
    std::vector<int32_t> allocated;

    if (free_pages_.size() < pages_needed) {
        std::cerr << "[TPUForge::KVCache] Out of memory: requested " << pages_needed 
                  << " pages, but only " << free_pages_.size() << " available." << std::endl;
        return allocated;
    }

    for (size_t i = 0; i < pages_needed; ++i) {
        allocated.push_back(free_pages_.back());
        free_pages_.pop_back();
    }

    return allocated;
}

void KVCacheManager::release_pages(const std::vector<int32_t>& page_ids) {
    for (int32_t id : page_ids) {
        free_pages_.push_back(id);
    }
}

} // namespace tpuforge
