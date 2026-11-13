#include "tpuforge/tree_speculator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace tpuforge {

namespace {

constexpr float kMaskBlockValue = -1e9f;

double canonical_u01(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

std::vector<int32_t> extract_node_path(const SystolicTree& tree, int32_t node_idx) {
    std::vector<int32_t> rev;
    const auto& nodes = tree.nodes();
    int32_t curr = node_idx;
    while (curr >= 0) {
        rev.push_back(nodes[static_cast<size_t>(curr)].token_id);
        curr = nodes[static_cast<size_t>(curr)].parent_idx;
    }
    std::reverse(rev.begin(), rev.end());
    return rev;
}

int32_t argmax_row(const float* row, size_t vocab_size) {
    if (std::isnan(row[0])) {
        throw std::domain_error("NaN in target logits");
    }
    int32_t best_idx = 0;
    float best_val = row[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (std::isnan(row[v])) {
            throw std::domain_error("NaN in target logits");
        }
        if (row[v] > best_val) {
            best_val = row[v];
            best_idx = static_cast<int32_t>(v);
        }
    }
    return best_idx;
}

void compute_softmax(const float* logits, size_t vocab_size, float temperature, std::vector<double>& out_probs) {
    out_probs.resize(vocab_size);
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t v = 0; v < vocab_size; ++v) {
        const float x = logits[v];
        if (std::isnan(x) || x == std::numeric_limits<float>::infinity()) {
            throw std::domain_error("NaN or +inf in target logits");
        }
        if (x != -std::numeric_limits<float>::infinity()) {
            const double scaled = static_cast<double>(x) / static_cast<double>(temperature);
            if (scaled > max_val) {
                max_val = scaled;
            }
        }
    }
    if (max_val == -std::numeric_limits<double>::infinity()) {
        throw std::domain_error("All target logits are -inf");
    }

    double sum_exp = 0.0;
    for (size_t v = 0; v < vocab_size; ++v) {
        const float x = logits[v];
        if (x == -std::numeric_limits<float>::infinity()) {
            out_probs[v] = 0.0;
        } else {
            const double scaled = static_cast<double>(x) / static_cast<double>(temperature);
            const double e = std::exp(scaled - max_val);
            out_probs[v] = e;
            sum_exp += e;
        }
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        throw std::domain_error("Softmax normalizer is non-finite or non-positive");
    }
    const double inv_sum = 1.0 / sum_exp;
    for (size_t v = 0; v < vocab_size; ++v) {
        out_probs[v] *= inv_sum;
    }
}

std::vector<TreeBuilder::BranchCandidate> query_suffix_candidates(
    const std::vector<int32_t>& context_tokens,
    const std::vector<int32_t>& path_after_root,
    size_t top_k = 3
) {
    // Reconstruct active virtual suffix ending at the candidate node.
    std::vector<int32_t> virtual_seq;
    const size_t max_order = 4;
    const size_t ctx_len = context_tokens.size();
    const size_t ctx_tail_keep = std::min(ctx_len, max_order);
    virtual_seq.reserve(ctx_tail_keep + path_after_root.size());
    for (size_t i = ctx_len - ctx_tail_keep; i < ctx_len; ++i) {
        virtual_seq.push_back(context_tokens[i]);
    }
    for (int32_t t : path_after_root) {
        virtual_seq.push_back(t);
    }

    std::unordered_map<int32_t, double> scores;
    const size_t vlen = virtual_seq.size();
    const size_t limit_order = std::min(max_order, vlen);

    // Match n-grams of order 4 down to 1 against historical context_tokens.
    for (size_t order = limit_order; order >= 1; --order) {
        if (ctx_len <= order) {
            continue;
        }
        const size_t v_start = vlen - order;
        double order_weight = std::pow(10.0, static_cast<double>(order));
        for (size_t pos = order - 1; pos + 1 < ctx_len; ++pos) {
            bool match = true;
            for (size_t k = 0; k < order; ++k) {
                if (context_tokens[pos - order + 1 + k] != virtual_seq[v_start + k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                int32_t next_tok = context_tokens[pos + 1];
                double recency = 1.0 + static_cast<double>(pos) / static_cast<double>(ctx_len);
                scores[next_tok] += order_weight * recency;
            }
        }
    }

    if (scores.empty()) {
        return {};
    }

    std::vector<std::pair<int32_t, double>> ranked(scores.begin(), scores.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });

    if (ranked.size() > top_k) {
        ranked.resize(top_k);
    }

    double total = 0.0;
    for (const auto& kv : ranked) {
        total += kv.second;
    }

    std::vector<TreeBuilder::BranchCandidate> out;
    out.reserve(ranked.size());
    for (const auto& kv : ranked) {
        out.push_back(TreeBuilder::BranchCandidate{
            kv.first,
            static_cast<float>(kv.second / total)
        });
    }
    return out;
}

} // namespace

void SystolicTree::reset(int32_t root_last_token, size_t tile_size) {
    if (root_last_token < 0) {
        throw std::invalid_argument("SystolicTree::reset root_last_token must be non-negative");
    }
    if (tile_size != 0) {
        if (!SystolicConstants::is_valid_tile_size(tile_size)) {
            throw std::invalid_argument("SystolicTree tile_size must be 128 or 256");
        }
        tile_size_ = tile_size;
    }
    nodes_.clear();
    nodes_.reserve(tile_size_);
    TreeNode root;
    root.token_id = root_last_token;
    root.parent_idx = -1;
    root.depth = 0;
    root.draft_prob = 1.0f;
    root.cumulative_logprob = 0.0f;
    nodes_.push_back(std::move(root));
}

int32_t SystolicTree::add_child(int32_t parent_idx, int32_t token_id, float draft_prob) {
    if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= nodes_.size()) {
        throw std::out_of_range("SystolicTree::add_child parent_idx out of range");
    }
    if (token_id < 0) {
        throw std::invalid_argument("SystolicTree::add_child token_id must be non-negative");
    }
    if (!std::isfinite(draft_prob) || draft_prob < 0.0f) {
        throw std::invalid_argument("SystolicTree::add_child draft_prob must be finite and non-negative");
    }

    // Deduplicate existing child with same token_id under parent_idx.
    for (int32_t existing_idx : nodes_[static_cast<size_t>(parent_idx)].children) {
        if (nodes_[static_cast<size_t>(existing_idx)].token_id == token_id) {
            return existing_idx;
        }
    }

    if (nodes_.size() >= tile_size_) {
        return -1;
    }

    const float clamped_prob = std::clamp(draft_prob, 1e-12f, 1.0f);
    TreeNode child;
    child.token_id = token_id;
    child.parent_idx = parent_idx;
    child.depth = nodes_[static_cast<size_t>(parent_idx)].depth + 1;
    child.draft_prob = clamped_prob;
    child.cumulative_logprob =
        nodes_[static_cast<size_t>(parent_idx)].cumulative_logprob + std::log(clamped_prob);

    int32_t new_idx = static_cast<int32_t>(nodes_.size());
    nodes_.push_back(std::move(child));
    nodes_[static_cast<size_t>(parent_idx)].children.push_back(new_idx);
    return new_idx;
}

PackedSystolicTile SystolicTree::pack(int32_t valid_prefix_len) const {
    const size_t T = tile_size_;
    if (nodes_.empty() || nodes_.size() > T) {
        throw std::logic_error("SystolicTree must contain between 1 and tile_size nodes");
    }
    if (valid_prefix_len < 0 ||
        valid_prefix_len > std::numeric_limits<int32_t>::max() - static_cast<int32_t>(T)) {
        throw std::out_of_range("SystolicTree::pack valid_prefix_len out of valid int32 range");
    }

    PackedSystolicTile tile;
    tile.tile_size = T;
    tile.active_nodes = nodes_.size();
    tile.token_ids.assign(T, 0);
    tile.position_offsets.assign(T, 0);
    tile.mask_u8.assign(T * T, 0);
    tile.additive_mask_f32.assign(T * T, kMaskBlockValue);

    const size_t n = nodes_.size();
    for (size_t i = 0; i < n; ++i) {
        tile.token_ids[i] = nodes_[i].token_id;
        tile.position_offsets[i] = valid_prefix_len + nodes_[i].depth;

        // Mark causal ancestor path from node i back to root 0.
        int32_t curr = static_cast<int32_t>(i);
        while (curr >= 0) {
            const size_t col = static_cast<size_t>(curr);
            tile.mask_u8[i * T + col] = 1;
            tile.additive_mask_f32[i * T + col] = 0.0f;
            curr = nodes_[col].parent_idx;
        }
    }

    // Padded rows [n .. T - 1] attend strictly to themselves so softmax stays finite.
    for (size_t i = n; i < T; ++i) {
        tile.mask_u8[i * T + i] = 1;
        tile.additive_mask_f32[i * T + i] = 0.0f;
    }

    return tile;
}

SystolicTree SuffixTreeDrafter::build_tree(
    const std::vector<int32_t>& context_tokens,
    size_t max_nodes,
    size_t max_depth,
    size_t tile_size
) {
    if (context_tokens.empty()) {
        throw std::invalid_argument("SuffixTreeDrafter requires non-empty context_tokens");
    }
    const size_t resolved_tile = (tile_size == 0)
        ? (max_nodes > SystolicTree::kMaxNodes
               ? SystolicConstants::kSystolicTileSizeV6
               : SystolicConstants::kSystolicTileSize)
        : tile_size;
    if (!SystolicConstants::is_valid_tile_size(resolved_tile)) {
        throw std::invalid_argument("SuffixTreeDrafter tile_size must be 128 or 256");
    }
    const size_t clamped_nodes = std::clamp<size_t>(max_nodes, 1, resolved_tile);

    SystolicTree tree(context_tokens.back(), resolved_tile);
    if (clamped_nodes == 1 || max_depth == 0) {
        return tree;
    }

    struct FrontierItem {
        float priority;
        int32_t node_idx;
        bool operator<(const FrontierItem& other) const {
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return node_idx > other.node_idx;
        }
    };

    std::priority_queue<FrontierItem> pq;
    pq.push(FrontierItem{0.0f, 0});

    while (!pq.empty() && tree.num_nodes() < clamped_nodes) {
        FrontierItem item = pq.top();
        pq.pop();

        const TreeNode& node = tree.nodes()[static_cast<size_t>(item.node_idx)];
        if (static_cast<size_t>(node.depth) >= max_depth) {
            continue;
        }

        // Extract path tokens after root (depth >= 1).
        std::vector<int32_t> full_path = extract_node_path(tree, item.node_idx);
        std::vector<int32_t> path_after_root;
        if (full_path.size() > 1) {
            path_after_root.assign(full_path.begin() + 1, full_path.end());
        }

        auto candidates = query_suffix_candidates(context_tokens, path_after_root, 3);
        for (const auto& cand : candidates) {
            if (tree.num_nodes() >= clamped_nodes) {
                break;
            }
            int32_t child_idx = tree.add_child(item.node_idx, cand.token_id, cand.prob);
            if (child_idx < 0) {
                break;
            }
            float child_logprob = tree.nodes()[static_cast<size_t>(child_idx)].cumulative_logprob;
            pq.push(FrontierItem{child_logprob, child_idx});
        }
    }

    return tree;
}

SystolicTree TreeBuilder::build_from_branching_callback(
    int32_t root_last_token,
    const BranchCallback& callback,
    size_t max_nodes,
    size_t max_depth,
    size_t tile_size
) {
    const size_t resolved_tile = (tile_size == 0)
        ? (max_nodes > SystolicTree::kMaxNodes
               ? SystolicConstants::kSystolicTileSizeV6
               : SystolicConstants::kSystolicTileSize)
        : tile_size;
    if (!SystolicConstants::is_valid_tile_size(resolved_tile)) {
        throw std::invalid_argument("TreeBuilder tile_size must be 128 or 256");
    }
    const size_t clamped_nodes = std::clamp<size_t>(max_nodes, 1, resolved_tile);
    SystolicTree tree(root_last_token, resolved_tile);
    if (clamped_nodes == 1 || max_depth == 0 || !callback) {
        return tree;
    }

    struct FrontierItem {
        float priority;
        int32_t node_idx;
        bool operator<(const FrontierItem& other) const {
            if (priority != other.priority) {
                return priority < other.priority;
            }
            return node_idx > other.node_idx;
        }
    };

    std::priority_queue<FrontierItem> pq;
    pq.push(FrontierItem{0.0f, 0});

    while (!pq.empty() && tree.num_nodes() < clamped_nodes) {
        FrontierItem item = pq.top();
        pq.pop();

        if (static_cast<size_t>(tree.nodes()[static_cast<size_t>(item.node_idx)].depth) >= max_depth) {
            continue;
        }

        std::vector<int32_t> path = extract_node_path(tree, item.node_idx);
        auto candidates = callback(path);
        for (const auto& cand : candidates) {
            if (tree.num_nodes() >= clamped_nodes) {
                break;
            }
            int32_t child_idx = tree.add_child(item.node_idx, cand.token_id, cand.prob);
            if (child_idx < 0) {
                break;
            }
            float child_logprob = tree.nodes()[static_cast<size_t>(child_idx)].cumulative_logprob;
            pq.push(FrontierItem{child_logprob, child_idx});
        }
    }

    return tree;
}

TreeVerificationResult SystolicTreeVerifier::verify_greedy(
    const SystolicTree& tree,
    const float* target_logits_128,
    size_t vocab_size
) {
    if (!target_logits_128 || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        tree.num_nodes() == 0 || tree.num_nodes() > tree.tile_size()) {
        throw std::invalid_argument("Invalid arguments to SystolicTreeVerifier::verify_greedy");
    }

    TreeVerificationResult result;
    const auto& nodes = tree.nodes();
    int32_t curr = 0;
    result.accepted_node_indices.push_back(0);

    while (true) {
        const float* row = target_logits_128 + (static_cast<size_t>(curr) * vocab_size);
        int32_t best_tok = argmax_row(row, vocab_size);

        int32_t matched_child = -1;
        for (int32_t c : nodes[static_cast<size_t>(curr)].children) {
            const int32_t c_tok = nodes[static_cast<size_t>(c)].token_id;
            if (c_tok < 0 || static_cast<size_t>(c_tok) >= vocab_size) {
                throw std::out_of_range("Draft child token_id out of [0, vocab_size)");
            }
            if (matched_child < 0 && c_tok == best_tok) {
                matched_child = c;
            }
        }

        if (matched_child >= 0) {
            result.accepted_draft_tokens.push_back(best_tok);
            result.emitted_tokens.push_back(best_tok);
            result.accepted_node_indices.push_back(matched_child);
            curr = matched_child;
        } else {
            result.bonus_token = best_tok;
            result.emitted_tokens.push_back(best_tok);
            break;
        }
    }

    result.acceptance_count = result.accepted_draft_tokens.size();
    result.terminal_depth = nodes[static_cast<size_t>(curr)].depth;
    return result;
}

TreeVerificationResult SystolicTreeVerifier::verify_stochastic(
    const SystolicTree& tree,
    const float* target_logits_128,
    size_t vocab_size,
    float temperature,
    uint64_t seed
) {
    if (!target_logits_128 || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        tree.num_nodes() == 0 || tree.num_nodes() > tree.tile_size()) {
        throw std::invalid_argument("Invalid arguments to SystolicTreeVerifier::verify_stochastic");
    }
    if (!std::isfinite(temperature)) {
        throw std::invalid_argument("temperature must be finite");
    }
    if (temperature < 1e-6f) {
        return verify_greedy(tree, target_logits_128, vocab_size);
    }

    TreeVerificationResult result;
    const auto& nodes = tree.nodes();
    std::mt19937_64 rng(seed);

    std::vector<double> probs;
    std::vector<double> residual;
    int32_t curr = 0;
    result.accepted_node_indices.push_back(0);

    while (true) {
        const float* row = target_logits_128 + (static_cast<size_t>(curr) * vocab_size);
        compute_softmax(row, vocab_size, temperature, probs);
        residual = probs;
        double mass = 0.0;
        for (double w : residual) {
            mass += w;
        }

        const auto& children = nodes[static_cast<size_t>(curr)].children;
        int32_t accepted_child = -1;

        for (int32_t c : children) {
            int32_t tok = nodes[static_cast<size_t>(c)].token_id;
            if (tok < 0 || static_cast<size_t>(tok) >= vocab_size) {
                throw std::out_of_range("Draft child token_id out of [0, vocab_size)");
            }
            if (accepted_child >= 0) {
                continue;
            }
            const double p_tok = residual[static_cast<size_t>(tok)];
            if (p_tok <= 0.0) {
                continue;
            }
            if (p_tok >= mass || (canonical_u01(rng) * mass) < p_tok) {
                accepted_child = c;
            } else {
                // Reject child c cleanly by zeroing its unnormalized residual weight without resetting.
                residual[static_cast<size_t>(tok)] = 0.0;
                mass = 0.0;
                for (double w : residual) {
                    mass += w;
                }
            }
        }

        if (accepted_child >= 0) {
            int32_t tok = nodes[static_cast<size_t>(accepted_child)].token_id;
            result.accepted_draft_tokens.push_back(tok);
            result.emitted_tokens.push_back(tok);
            result.accepted_node_indices.push_back(accepted_child);
            curr = accepted_child;
        } else {
            if (!(mass > 0.0) || !std::isfinite(mass)) {
                throw std::logic_error("Residual probability mass exhausted in verify_stochastic");
            }
            const double target = canonical_u01(rng) * mass;
            double cdf = 0.0;
            int32_t bonus = -1;
            int32_t last_nonzero = -1;
            for (size_t v = 0; v < vocab_size; ++v) {
                if (residual[v] > 0.0) {
                    last_nonzero = static_cast<int32_t>(v);
                    cdf += residual[v];
                    if (target < cdf) {
                        bonus = last_nonzero;
                        break;
                    }
                }
            }
            if (bonus < 0) {
                bonus = last_nonzero;
            }
            if (bonus < 0) {
                throw std::logic_error("No nonzero entry available in residual distribution");
            }
            result.bonus_token = bonus;
            result.emitted_tokens.push_back(bonus);
            break;
        }
    }

    result.acceptance_count = result.accepted_draft_tokens.size();
    result.terminal_depth = nodes[static_cast<size_t>(curr)].depth;
    return result;
}

} // namespace tpuforge
