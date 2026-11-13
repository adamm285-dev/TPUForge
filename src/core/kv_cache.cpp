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

} // namespace tpuforge
