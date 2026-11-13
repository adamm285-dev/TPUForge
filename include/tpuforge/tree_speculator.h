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
    size_t active_nodes{0};
    std::array<int32_t, SystolicConstants::kSystolicTileSize> token_ids{};
    std::array<int32_t, SystolicConstants::kSystolicTileSize> position_offsets{};
    std::array<uint8_t, SystolicConstants::kSystolicTileSize * SystolicConstants::kSystolicTileSize> mask_u8{};
    std::array<float, SystolicConstants::kSystolicTileSize * SystolicConstants::kSystolicTileSize> additive_mask_f32{};
};

class SystolicTree {
public:
    static constexpr size_t kMaxNodes = SystolicConstants::kSystolicTileSize;

    SystolicTree() { reset(0); }
    explicit SystolicTree(int32_t root_last_token) { reset(root_last_token); }

    void reset(int32_t root_last_token);

    // Adds a child under parent_idx, returning its node index or -1 if the 128-node tile is full.
    // Deduplicates if parent_idx already has a child with the same token_id.
    int32_t add_child(int32_t parent_idx, int32_t token_id, float draft_prob = 1.0f);

    size_t num_nodes() const { return nodes_.size(); }
    const std::vector<TreeNode>& nodes() const { return nodes_; }

    // Packs the tree into a fixed 128-row systolic tile with 2D causal ancestor attention mask.
    PackedSystolicTile pack(int32_t valid_prefix_len) const;

private:
    std::vector<TreeNode> nodes_;
};

class SuffixTreeDrafter {
public:
    // Builds an up-to-128-node speculative tree from n-gram suffix matches in context_tokens.
    static SystolicTree build_tree(
        const std::vector<int32_t>& context_tokens,
        size_t max_nodes = SystolicTree::kMaxNodes,
        size_t max_depth = 16
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
        size_t max_depth = 16
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
