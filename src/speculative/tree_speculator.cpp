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

template <typename BestTokenFn>
TreeVerificationResult walk_greedy_tree(
    const SystolicTree& tree,
    size_t vocab_size,
    BestTokenFn&& best_token_for_node
) {
    TreeVerificationResult result;
    const auto& nodes = tree.nodes();
    int32_t curr = 0;
    result.accepted_node_indices.push_back(0);

    while (true) {
        const int32_t best_tok = best_token_for_node(static_cast<size_t>(curr));

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

struct NGramKey {
    uint8_t order{0};
    int32_t t[4]{0, 0, 0, 0};

    bool operator==(const NGramKey& other) const noexcept {
        if (order != other.order) {
            return false;
        }
        for (uint8_t i = 0; i < order; ++i) {
            if (t[i] != other.t[i]) {
                return false;
            }
        }
        return true;
    }
};

struct NGramKeyHash {
    size_t operator()(const NGramKey& k) const noexcept {
        uint64_t h = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(k.order);
        for (uint8_t i = 0; i < k.order; ++i) {
            uint64_t x = static_cast<uint32_t>(k.t[i]);
            h ^= x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<size_t>(h ^ (h >> 31));
    }
};

struct TransitionStats {
    int32_t next_tok{0};
    uint32_t count{0};
    uint64_t pos_sum{0};
};

struct NGramBucket {
    static constexpr size_t kLinearThreshold = 16;
    static constexpr size_t kCacheTopThreshold = 128;
    static constexpr size_t kTopScanCap = 64;

    std::vector<TransitionStats> items;
    std::unordered_map<int32_t, size_t> pos_by_tok;
    mutable std::vector<size_t> cached_top;
    mutable bool dirty{false};

    void record(int32_t next_tok, uint64_t pos) {
        if (items.size() <= kLinearThreshold && pos_by_tok.empty()) {
            for (auto& e : items) {
                if (e.next_tok == next_tok) {
                    ++e.count;
                    e.pos_sum += pos;
                    dirty = true;
                    return;
                }
            }
            if (items.size() == kLinearThreshold) {
                for (size_t i = 0; i < items.size(); ++i) {
                    pos_by_tok.emplace(items[i].next_tok, i);
                }
            } else {
                items.push_back(TransitionStats{next_tok, 1u, pos});
                dirty = true;
                return;
            }
        }
        auto it = pos_by_tok.find(next_tok);
        if (it != pos_by_tok.end()) {
            auto& e = items[it->second];
            ++e.count;
            e.pos_sum += pos;
        } else {
            pos_by_tok.emplace(next_tok, items.size());
            items.push_back(TransitionStats{next_tok, 1u, pos});
        }
        dirty = true;
    }

    template <typename Visitor>
    void visit_top(Visitor&& fn) const {
        if (items.size() <= kCacheTopThreshold) {
            for (const auto& e : items) {
                fn(e);
            }
            return;
        }
        if (dirty) {
            cached_top.resize(items.size());
            for (size_t i = 0; i < items.size(); ++i) {
                cached_top[i] = i;
            }
            const size_t cap = std::min(kTopScanCap, cached_top.size());
            std::partial_sort(
                cached_top.begin(),
                cached_top.begin() + static_cast<std::ptrdiff_t>(cap),
                cached_top.end(),
                [&](size_t a, size_t b) {
                    const auto& ea = items[a];
                    const auto& eb = items[b];
                    if (ea.count != eb.count) {
                        return ea.count > eb.count;
                    }
                    if (ea.pos_sum != eb.pos_sum) {
                        return ea.pos_sum > eb.pos_sum;
                    }
                    return ea.next_tok < eb.next_tok;
                }
            );
            cached_top.resize(cap);
            dirty = false;
        }
        for (size_t idx : cached_top) {
            fn(items[idx]);
        }
    }
};

size_t resolve_drafter_tile_size(size_t tile_size, size_t max_nodes, const char* caller) {
    const size_t resolved = (tile_size == 0)
        ? (max_nodes > SystolicTree::kMaxNodes
               ? SystolicConstants::kSystolicTileSizeV6
               : SystolicConstants::kSystolicTileSize)
        : tile_size;
    if (!SystolicConstants::is_valid_tile_size(resolved)) {
        throw std::invalid_argument(std::string(caller) + " tile_size must be 128 or 256");
    }
    return resolved;
}

bool has_child_token(const SystolicTree& tree, int32_t parent_idx, int32_t token_id) {
    const auto& children = tree.nodes()[static_cast<size_t>(parent_idx)].children;
    for (int32_t c : children) {
        if (tree.nodes()[static_cast<size_t>(c)].token_id == token_id) {
            return true;
        }
    }
    return false;
}

std::vector<int32_t> extract_path_after_root(const SystolicTree& tree, int32_t node_idx) {
    std::vector<int32_t> full_path = extract_node_path(tree, node_idx);
    if (full_path.size() <= 1) {
        return {};
    }
    return std::vector<int32_t>(full_path.begin() + 1, full_path.end());
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
    SuffixIndex index(context_tokens);
    return index.build_tree(max_nodes, max_depth, tile_size);
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

struct SuffixIndex::Impl {
    std::vector<int32_t> context;
    std::unordered_map<NGramKey, NGramBucket, NGramKeyHash> buckets;
};

SuffixIndex::SuffixIndex() : impl_(new Impl()) {}

SuffixIndex::SuffixIndex(const std::vector<int32_t>& initial_tokens) : impl_(new Impl()) {
    reset(initial_tokens);
}

SuffixIndex::~SuffixIndex() {
    delete impl_;
    impl_ = nullptr;
}

SuffixIndex::SuffixIndex(const SuffixIndex& other)
    : impl_(other.impl_ ? new Impl(*other.impl_) : new Impl()) {}

SuffixIndex& SuffixIndex::operator=(const SuffixIndex& other) {
    if (this != &other) {
        Impl* next = other.impl_ ? new Impl(*other.impl_) : new Impl();
        delete impl_;
        impl_ = next;
    }
    return *this;
}

SuffixIndex::SuffixIndex(SuffixIndex&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

SuffixIndex& SuffixIndex::operator=(SuffixIndex&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void SuffixIndex::reset(const std::vector<int32_t>& initial_tokens) {
    if (!impl_) {
        impl_ = new Impl();
    }
    impl_->context.clear();
    impl_->buckets.clear();
    if (!initial_tokens.empty()) {
        append_tokens(initial_tokens.data(), initial_tokens.size());
    }
}

void SuffixIndex::append_token(int32_t token_id) {
    if (!impl_) {
        impl_ = new Impl();
    }
    const size_t old_len = impl_->context.size();
    if (old_len >= 1) {
        const size_t pos = old_len - 1;
        const size_t max_order = std::min<size_t>(4, old_len);
        NGramKey key;
        for (size_t order = 1; order <= max_order; ++order) {
            key.order = static_cast<uint8_t>(order);
            const size_t start = pos + 1 - order;
            for (size_t k = 0; k < order; ++k) {
                key.t[k] = impl_->context[start + k];
            }
            impl_->buckets[key].record(token_id, static_cast<uint64_t>(pos));
        }
    }
    impl_->context.push_back(token_id);
}

void SuffixIndex::append_tokens(const int32_t* tokens, size_t count) {
    if (count == 0) {
        return;
    }
    if (!tokens) {
        throw std::invalid_argument("SuffixIndex::append_tokens received null tokens pointer");
    }
    if (!impl_) {
        impl_ = new Impl();
    }
    impl_->context.reserve(impl_->context.size() + count);
    if (impl_->buckets.empty() && count > 256) {
        impl_->buckets.reserve(count * 3);
    }
    for (size_t i = 0; i < count; ++i) {
        append_token(tokens[i]);
    }
}

std::vector<TreeBuilder::BranchCandidate> SuffixIndex::query(
    const std::vector<int32_t>& path_after_root,
    size_t top_k
) const {
    if (!impl_ || impl_->context.size() < 2 || top_k == 0) {
        return {};
    }
    const size_t ctx_len = impl_->context.size();
    const size_t vlen = ctx_len + path_after_root.size();
    const size_t limit_order = std::min<size_t>(4, vlen);

    std::unordered_map<int32_t, double> scores;
    for (size_t order = limit_order; order >= 1; --order) {
        if (ctx_len <= order) {
            continue;
        }
        NGramKey key;
        key.order = static_cast<uint8_t>(order);
        const size_t v_start = vlen - order;
        for (size_t k = 0; k < order; ++k) {
            const size_t v_idx = v_start + k;
            key.t[k] = (v_idx < ctx_len)
                ? impl_->context[v_idx]
                : path_after_root[v_idx - ctx_len];
        }
        auto it = impl_->buckets.find(key);
        if (it == impl_->buckets.end()) {
            continue;
        }
        const double order_weight = std::pow(10.0, static_cast<double>(order));
        const double inv_ctx_len = 1.0 / static_cast<double>(ctx_len);
        it->second.visit_top([&](const TransitionStats& e) {
            const double sum_recency =
                static_cast<double>(e.count) + static_cast<double>(e.pos_sum) * inv_ctx_len;
            scores[e.next_tok] += order_weight * sum_recency;
        });
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

SystolicTree SuffixIndex::build_tree(
    size_t max_nodes,
    size_t max_depth,
    size_t tile_size
) const {
    if (!impl_ || impl_->context.empty()) {
        throw std::invalid_argument("SuffixTreeDrafter requires non-empty context_tokens");
    }
    const size_t resolved_tile = resolve_drafter_tile_size(tile_size, max_nodes, "SuffixTreeDrafter");
    const size_t clamped_nodes = std::clamp<size_t>(max_nodes, 1, resolved_tile);

    SystolicTree tree(impl_->context.back(), resolved_tile);
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

        std::vector<int32_t> path_after_root = extract_path_after_root(tree, item.node_idx);
        auto candidates = query(path_after_root, 3);
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

size_t SuffixIndex::size() const noexcept {
    return impl_ ? impl_->context.size() : 0;
}

const std::vector<int32_t>& SuffixIndex::context_tokens() const noexcept {
    static const std::vector<int32_t> kEmpty;
    return impl_ ? impl_->context : kEmpty;
}

SystolicTree TreeBuilder::build_from_block_top_k(
    int32_t root_last_token,
    const int32_t* draft_token_ids,
    const float* draft_probs,
    size_t block_steps,
    size_t top_k,
    const SuffixIndex* suffix_fallback,
    size_t max_nodes,
    size_t max_depth,
    size_t tile_size
) {
    if (block_steps > 0 && top_k > 0 && (!draft_token_ids || !draft_probs)) {
        throw std::invalid_argument(
            "TreeBuilder::build_from_block_top_k requires non-null draft buffers"
        );
    }

    const size_t resolved_tile = resolve_drafter_tile_size(
        tile_size, max_nodes, "TreeBuilder::build_from_block_top_k"
    );
    const size_t clamped_nodes = std::clamp<size_t>(max_nodes, 1, resolved_tile);

    SystolicTree tree(root_last_token, resolved_tile);
    if (clamped_nodes == 1 || max_depth == 0) {
        return tree;
    }

    // Reserve the full top-1 parallel block spine before spending slots on branches.
    std::vector<int32_t> spine_nodes;
    spine_nodes.reserve(std::min(block_steps, max_depth) + 1);
    spine_nodes.push_back(0);
    if (block_steps > 0 && top_k > 0) {
        const size_t spine_steps = std::min(block_steps, max_depth);
        for (size_t s = 0; s < spine_steps && tree.num_nodes() < clamped_nodes; ++s) {
            const int32_t tok = draft_token_ids[s * top_k];
            const float raw_p = draft_probs[s * top_k];
            if (tok < 0 || !std::isfinite(raw_p) || raw_p <= 0.0f) {
                break;
            }
            const float p = std::clamp(raw_p, 1e-6f, 1.0f);
            const int32_t child_idx = tree.add_child(spine_nodes.back(), tok, p);
            if (child_idx < 0) {
                break;
            }
            spine_nodes.push_back(child_idx);
        }
    }

    struct EdgeItem {
        float score_logprob{0.0f};
        int32_t parent_idx{0};
        int32_t token_id{0};
        float prob{0.0f};
        bool operator<(const EdgeItem& other) const noexcept {
            if (score_logprob != other.score_logprob) {
                return score_logprob < other.score_logprob;
            }
            if (parent_idx != other.parent_idx) {
                return parent_idx > other.parent_idx;
            }
            return token_id > other.token_id;
        }
    };

    std::priority_queue<EdgeItem> pq;

    auto push_block_edges = [&](int32_t parent_idx, size_t start_k) {
        const size_t depth = static_cast<size_t>(tree.nodes()[static_cast<size_t>(parent_idx)].depth);
        if (depth >= block_steps || depth >= max_depth || top_k == 0) {
            return;
        }
        const float parent_lp = tree.nodes()[static_cast<size_t>(parent_idx)].cumulative_logprob;
        for (size_t k = start_k; k < top_k; ++k) {
            const int32_t tok = draft_token_ids[depth * top_k + k];
            const float raw_p = draft_probs[depth * top_k + k];
            if (tok < 0 || !std::isfinite(raw_p) || raw_p <= 0.0f) {
                continue;
            }
            const float p = std::clamp(raw_p, 1e-6f, 1.0f);
            pq.push(EdgeItem{parent_lp + std::log(p), parent_idx, tok, p});
        }
    };

    auto push_suffix_edges = [&](int32_t parent_idx, float discount) {
        if (!suffix_fallback || suffix_fallback->size() < 2) {
            return;
        }
        const size_t depth = static_cast<size_t>(tree.nodes()[static_cast<size_t>(parent_idx)].depth);
        if (depth >= max_depth) {
            return;
        }
        std::vector<int32_t> path_after_root = extract_path_after_root(tree, parent_idx);
        auto cands = suffix_fallback->query(path_after_root, 3);
        const float parent_lp = tree.nodes()[static_cast<size_t>(parent_idx)].cumulative_logprob;
        for (const auto& cand : cands) {
            if (cand.token_id < 0 || !std::isfinite(cand.prob) || cand.prob <= 0.0f) {
                continue;
            }
            const float p = std::clamp(cand.prob * discount, 1e-6f, 1.0f);
            pq.push(EdgeItem{parent_lp + std::log(p), parent_idx, cand.token_id, p});
        }
    };

    for (int32_t spine_idx : spine_nodes) {
        const size_t d = static_cast<size_t>(tree.nodes()[static_cast<size_t>(spine_idx)].depth);
        if (d >= max_depth) {
            continue;
        }
        if (d < block_steps && top_k > 1) {
            push_block_edges(spine_idx, 1);
        }
        if (suffix_fallback != nullptr) {
            const float discount = (d >= block_steps) ? 0.5f : 0.05f;
            push_suffix_edges(spine_idx, discount);
        }
    }

    while (!pq.empty() && tree.num_nodes() < clamped_nodes) {
        EdgeItem item = pq.top();
        pq.pop();

        const TreeNode& parent = tree.nodes()[static_cast<size_t>(item.parent_idx)];
        if (static_cast<size_t>(parent.depth) >= max_depth) {
            continue;
        }
        if (has_child_token(tree, item.parent_idx, item.token_id)) {
            continue;
        }

        const int32_t child_idx = tree.add_child(item.parent_idx, item.token_id, item.prob);
        if (child_idx < 0) {
            break;
        }

        const size_t child_depth = static_cast<size_t>(tree.nodes()[static_cast<size_t>(child_idx)].depth);
        if (child_depth >= max_depth) {
            continue;
        }
        if (child_depth < block_steps && top_k > 0) {
            push_block_edges(child_idx, 0);
        } else if (suffix_fallback != nullptr) {
            push_suffix_edges(child_idx, 0.5f);
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

    return walk_greedy_tree(tree, vocab_size, [&](size_t node_idx) {
        return argmax_row(target_logits_128 + (node_idx * vocab_size), vocab_size);
    });
}

TreeVerificationResult SystolicTreeVerifier::verify_greedy_tokens(
    const SystolicTree& tree,
    const int32_t* target_argmax_tokens,
    size_t vocab_size
) {
    if (!target_argmax_tokens || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        tree.num_nodes() == 0 || tree.num_nodes() > tree.tile_size()) {
        throw std::invalid_argument("Invalid arguments to SystolicTreeVerifier::verify_greedy_tokens");
    }

    return walk_greedy_tree(tree, vocab_size, [&](size_t node_idx) {
        const int32_t best_tok = target_argmax_tokens[node_idx];
        if (best_tok < 0 || static_cast<size_t>(best_tok) >= vocab_size) {
            throw std::domain_error("NaN or out-of-bounds token in target_argmax_tokens");
        }
        return best_tok;
    });
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
