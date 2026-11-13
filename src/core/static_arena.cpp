#include "tpuforge/static_arena.h"

#include <array>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace tpuforge {

namespace {

size_t checked_mul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::length_error(std::string("StaticArena size overflow in ") + what);
    }
    return a * b;
}

} // namespace

size_t StaticArena::kv_row_elems() const {
    return checked_mul(config_.num_kv_heads, config_.head_dim, "kv_row_elems");
}

void* StaticArena::allocate_aligned(size_t bytes) {
    constexpr size_t align = SystolicConstants::kArenaAlignment;
    const size_t safe_bytes = (bytes == 0) ? align : bytes;
    if (safe_bytes > std::numeric_limits<size_t>::max() - (align - 1)) {
        throw std::bad_alloc();
    }
    const size_t padded = ((safe_bytes + align - 1) / align) * align;

#if defined(_WIN32)
    void* ptr = _aligned_malloc(padded, align);
#else
    void* ptr = std::aligned_alloc(align, padded);
#endif
    if (!ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, padded);
    return ptr;
}

void StaticArena::free_aligned(void* ptr) {
    if (!ptr) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

StaticArena::StaticArena(const Config& config) : config_(config) {
    if (!SystolicConstants::is_valid_tile_size(config_.tile_size)) {
        throw std::invalid_argument("StaticArena tile_size must be 128 or 256");
    }
    const size_t tile = config_.tile_size;
    if (config_.max_seq_len < tile ||
        config_.max_seq_len > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("StaticArena max_seq_len must be in [tile_size, INT32_MAX]");
    }
    if (config_.hidden_dim == 0 || config_.num_kv_heads == 0 ||
        config_.head_dim == 0 || config_.vocab_size == 0 ||
        config_.vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("StaticArena dimensions must be non-zero and vocab_size <= INT32_MAX");
    }

    const size_t row_elems = checked_mul(config_.num_kv_heads, config_.head_dim, "kv_row_elems");
    const size_t kv_elems = checked_mul(config_.max_seq_len, row_elems, "kv_elems");
    const size_t kv_bytes = checked_mul(kv_elems, sizeof(float), "kv_bytes");
    const size_t logits_elems = checked_mul(tile, config_.vocab_size, "logits_elems");
    const size_t logits_bytes = checked_mul(logits_elems, sizeof(float), "logits_bytes");
    const size_t mask_bytes = checked_mul(tile * tile, sizeof(float), "mask_bytes");

    try {
        token_tile_128_ = static_cast<int32_t*>(allocate_aligned(tile * sizeof(int32_t)));
        position_tile_128_ = static_cast<int32_t*>(allocate_aligned(tile * sizeof(int32_t)));
        causal_mask_128x128_ = static_cast<float*>(allocate_aligned(mask_bytes));
        kv_prefix_k_ = static_cast<float*>(allocate_aligned(kv_bytes));
        kv_prefix_v_ = static_cast<float*>(allocate_aligned(kv_bytes));
        logits_tile_128_ = static_cast<float*>(allocate_aligned(logits_bytes));
    } catch (...) {
        release_all();
        throw;
    }
}

StaticArena::~StaticArena() {
    release_all();
}

void StaticArena::release_all() {
    free_aligned(token_tile_128_);
    free_aligned(position_tile_128_);
    free_aligned(causal_mask_128x128_);
    free_aligned(kv_prefix_k_);
    free_aligned(kv_prefix_v_);
    free_aligned(logits_tile_128_);

    token_tile_128_ = nullptr;
    position_tile_128_ = nullptr;
    causal_mask_128x128_ = nullptr;
    kv_prefix_k_ = nullptr;
    kv_prefix_v_ = nullptr;
    logits_tile_128_ = nullptr;
}

StaticArena::StaticArena(StaticArena&& other) noexcept
    : config_(other.config_),
      token_tile_128_(std::exchange(other.token_tile_128_, nullptr)),
      position_tile_128_(std::exchange(other.position_tile_128_, nullptr)),
      causal_mask_128x128_(std::exchange(other.causal_mask_128x128_, nullptr)),
      kv_prefix_k_(std::exchange(other.kv_prefix_k_, nullptr)),
      kv_prefix_v_(std::exchange(other.kv_prefix_v_, nullptr)),
      logits_tile_128_(std::exchange(other.logits_tile_128_, nullptr)),
      valid_prefix_len_(std::exchange(other.valid_prefix_len_, 0)) {}

StaticArena& StaticArena::operator=(StaticArena&& other) noexcept {
    if (this != &other) {
        release_all();
        config_ = other.config_;
        token_tile_128_ = std::exchange(other.token_tile_128_, nullptr);
        position_tile_128_ = std::exchange(other.position_tile_128_, nullptr);
        causal_mask_128x128_ = std::exchange(other.causal_mask_128x128_, nullptr);
        kv_prefix_k_ = std::exchange(other.kv_prefix_k_, nullptr);
        kv_prefix_v_ = std::exchange(other.kv_prefix_v_, nullptr);
        logits_tile_128_ = std::exchange(other.logits_tile_128_, nullptr);
        valid_prefix_len_ = std::exchange(other.valid_prefix_len_, 0);
    }
    return *this;
}

void StaticArena::set_valid_prefix_len(int32_t len) {
    const size_t max_allowed = config_.max_seq_len - config_.tile_size;
    if (len < 0 || static_cast<size_t>(len) > max_allowed) {
        throw std::out_of_range("valid_prefix_len out of range [0, max_seq_len - tile_size]");
    }
    valid_prefix_len_ = len;
}

void StaticArena::commit_accepted_kv_rows(
    const float* step_k_tile,
    const float* step_v_tile,
    const std::vector<int32_t>& accepted_row_indices
) {
    if (!kv_prefix_k_ || !kv_prefix_v_) {
        throw std::logic_error("StaticArena has been moved from");
    }
    if (!step_k_tile || !step_v_tile) {
        throw std::invalid_argument("step_k_tile and step_v_tile must not be null");
    }
    const size_t row_elems = kv_row_elems();
    const size_t row_bytes = row_elems * sizeof(float);
    const size_t count = accepted_row_indices.size();

    const size_t tile = config_.tile_size;
    if (count > tile) {
        throw std::out_of_range("accepted_row_indices size exceeds systolic tile rows");
    }
    const size_t max_allowed = config_.max_seq_len - tile;
    if (static_cast<size_t>(valid_prefix_len_) + count > max_allowed) {
        throw std::out_of_range("StaticArena KV prefix capacity exceeded (tile headroom required)");
    }

    // Pass 1: validate all indices and reject duplicates before mutating prefix buffers.
    std::array<bool, SystolicConstants::kMaxSystolicTileSize> seen{};
    for (size_t idx = 0; idx < count; ++idx) {
        const int32_t r = accepted_row_indices[idx];
        if (r < 0 || static_cast<size_t>(r) >= tile) {
            throw std::out_of_range("accepted_row_indices entry out of [0, tile_size - 1]");
        }
        if (seen[static_cast<size_t>(r)]) {
            throw std::invalid_argument("accepted_row_indices contains duplicate row index");
        }
        seen[static_cast<size_t>(r)] = true;
    }

    // Pass 2: copy rows using memmove to tolerate any buffer aliasing.
    for (size_t idx = 0; idx < count; ++idx) {
        const int32_t r = accepted_row_indices[idx];
        const size_t src_offset = static_cast<size_t>(r) * row_elems;
        const size_t dst_offset = (static_cast<size_t>(valid_prefix_len_) + idx) * row_elems;

        std::memmove(kv_prefix_k_ + dst_offset, step_k_tile + src_offset, row_bytes);
        std::memmove(kv_prefix_v_ + dst_offset, step_v_tile + src_offset, row_bytes);
    }

    valid_prefix_len_ += static_cast<int32_t>(count);
}

bool StaticArena::is_aligned_128() const {
    auto aligned = [](const void* ptr) -> bool {
        return ptr != nullptr &&
               (reinterpret_cast<uintptr_t>(ptr) % SystolicConstants::kArenaAlignment) == 0;
    };
    return aligned(token_tile_128_) &&
           aligned(position_tile_128_) &&
           aligned(causal_mask_128x128_) &&
           aligned(kv_prefix_k_) &&
           aligned(kv_prefix_v_) &&
           aligned(logits_tile_128_);
}

} // namespace tpuforge
