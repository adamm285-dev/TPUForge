#pragma once

#include "tpuforge/static_arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace tpuforge {

struct TreeNode {
    int32_t token_id{0};
    int32_t parent_idx{-1};
    int32_t depth{0};
    float draft_prob{1.0f};
    float cumulative_logprob{0.0f};
    std::vector<int32_t> children;
};

struct PackedSystolicTile {
    size_t tile_size{SystolicConstants::kSystolicTileSize};
    size_t active_nodes{0};
    std::vector<int32_t> token_ids;
    std::vector<int32_t> position_offsets;
    std::vector<uint8_t> mask_u8;
    std::vector<float> additive_mask_f32;
};

class SystolicTree {
public:
    static constexpr size_t kMaxNodes = SystolicConstants::kSystolicTileSize;
    static constexpr size_t kMaxNodesV6 = SystolicConstants::kSystolicTileSizeV6;

    SystolicTree() { reset(0, SystolicConstants::kSystolicTileSize); }
    explicit SystolicTree(int32_t root_last_token, size_t tile_size = SystolicConstants::kSystolicTileSize) {
        reset(root_last_token, tile_size);
    }

    // Resets the tree. tile_size == 0 preserves the current tile_size_.
    void reset(int32_t root_last_token, size_t tile_size = 0);

    size_t tile_size() const noexcept { return tile_size_; }
    size_t max_nodes() const noexcept { return tile_size_; }

    // Adds a child under parent_idx, returning its node index or -1 if the systolic tile is full.
    // Deduplicates if parent_idx already has a child with the same token_id.
    int32_t add_child(int32_t parent_idx, int32_t token_id, float draft_prob = 1.0f);

    size_t num_nodes() const { return nodes_.size(); }
    const std::vector<TreeNode>& nodes() const { return nodes_; }

    // Packs the tree into a fixed tile_size-row systolic tile with 2D causal ancestor attention mask.
    PackedSystolicTile pack(int32_t valid_prefix_len) const;

private:
    size_t tile_size_{SystolicConstants::kSystolicTileSize};
    std::vector<TreeNode> nodes_;
};

class SuffixTreeDrafter {
public:
    // Builds an up-to-tile_size speculative tree from n-gram suffix matches in context_tokens.
    // When tile_size == 0, automatically chooses 256 if max_nodes > 128, else 128.
    static SystolicTree build_tree(
        const std::vector<int32_t>& context_tokens,
        size_t max_nodes = SystolicTree::kMaxNodes,
        size_t max_depth = 16,
        size_t tile_size = 0
    );
};

class TreeBuilder {
public:
    struct BranchCandidate {
        int32_t token_id{0};
        float prob{1.0f};
    };

    using BranchCallback = std::function<std::vector<BranchCandidate>(const std::vector<int32_t>& path_tokens)>;

    static SystolicTree build_from_branching_callback(
        int32_t root_last_token,
        const BranchCallback& callback,
        size_t max_nodes = SystolicTree::kMaxNodes,
        size_t max_depth = 16,
        size_t tile_size = 0
    );
};

struct TreeVerificationResult {
    std::vector<int32_t> emitted_tokens;
    std::vector<int32_t> accepted_draft_tokens;
    std::vector<int32_t> accepted_node_indices;
    int32_t bonus_token{0};
    size_t acceptance_count{0};
    int32_t terminal_depth{0};
};

class SystolicTreeVerifier {
public:
    static TreeVerificationResult verify_greedy(
        const SystolicTree& tree,
        const float* target_logits_128,
        size_t vocab_size
    );

    static TreeVerificationResult verify_stochastic(
        const SystolicTree& tree,
        const float* target_logits_128,
        size_t vocab_size,
        float temperature,
        uint64_t seed
    );
};

} // namespace tpuforge
