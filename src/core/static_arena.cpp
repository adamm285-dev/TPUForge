#include "tpuforge/static_arena.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace tpuforge {

void* StaticArena::allocate_aligned(size_t bytes) {
    constexpr size_t align = SystolicConstants::kArenaAlignment;
    size_t safe_bytes = (bytes == 0) ? align : bytes;
    size_t padded = ((safe_bytes + align - 1) / align) * align;

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
    if (config_.max_seq_len < SystolicConstants::kSystolicTileSize) {
        throw std::invalid_argument("StaticArena max_seq_len must be >= 128");
    }
    if (config_.num_kv_heads == 0 || config_.head_dim == 0 || config_.vocab_size == 0) {
        throw std::invalid_argument("StaticArena dimensions must be non-zero");
    }

    constexpr size_t tile = SystolicConstants::kSystolicTileSize;
    const size_t kv_elems = config_.max_seq_len * kv_row_elems();

    token_tile_128_ = static_cast<int32_t*>(allocate_aligned(tile * sizeof(int32_t)));
    position_tile_128_ = static_cast<int32_t*>(allocate_aligned(tile * sizeof(int32_t)));
    causal_mask_128x128_ = static_cast<float*>(allocate_aligned(tile * tile * sizeof(float)));
    kv_prefix_k_ = static_cast<float*>(allocate_aligned(kv_elems * sizeof(float)));
    kv_prefix_v_ = static_cast<float*>(allocate_aligned(kv_elems * sizeof(float)));
    logits_tile_128_ = static_cast<float*>(allocate_aligned(tile * config_.vocab_size * sizeof(float)));
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
    const int32_t max_allowed =
        static_cast<int32_t>(config_.max_seq_len - SystolicConstants::kSystolicTileSize);
    if (len < 0 || len > max_allowed) {
        throw std::out_of_range("valid_prefix_len out of range [0, max_seq_len - 128]");
    }
    valid_prefix_len_ = len;
}

void StaticArena::commit_accepted_kv_rows(
    const float* step_k_tile,
    const float* step_v_tile,
    const std::vector<int32_t>& accepted_row_indices
) {
    if (!step_k_tile || !step_v_tile) {
        throw std::invalid_argument("step_k_tile and step_v_tile must not be null");
    }
    const size_t row_elems = kv_row_elems();
    const size_t row_bytes = row_elems * sizeof(float);
    const size_t count = accepted_row_indices.size();

    if (static_cast<size_t>(valid_prefix_len_) + count > config_.max_seq_len) {
        throw std::out_of_range("StaticArena KV prefix capacity exceeded");
    }

    for (size_t idx = 0; idx < count; ++idx) {
        int32_t r = accepted_row_indices[idx];
        if (r < 0 || static_cast<size_t>(r) >= SystolicConstants::kSystolicTileSize) {
            throw std::out_of_range("accepted_row_indices entry out of [0, 127]");
        }
        const size_t src_offset = static_cast<size_t>(r) * row_elems;
        const size_t dst_offset = (static_cast<size_t>(valid_prefix_len_) + idx) * row_elems;

        std::memcpy(kv_prefix_k_ + dst_offset, step_k_tile + src_offset, row_bytes);
        std::memcpy(kv_prefix_v_ + dst_offset, step_v_tile + src_offset, row_bytes);
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
