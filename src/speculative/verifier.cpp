#include "tpuforge/speculative.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tpuforge {

namespace {

int32_t argmax_checked(const float* logits, size_t vocab_size) {
    if (std::isnan(logits[0])) {
        throw std::domain_error("NaN in target_logits_matrix");
    }
    int32_t best_token = 0;
    float max_val = logits[0];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (std::isnan(logits[v])) {
            throw std::domain_error("NaN in target_logits_matrix");
        }
        if (logits[v] > max_val) {
            max_val = logits[v];
            best_token = static_cast<int32_t>(v);
        }
    }
    return best_token;
}

} // namespace

SystolicVerifier::SystolicVerifier(std::shared_ptr<Device> device)
    : device_(std::move(device)) {
    if (!device_) {
        throw std::invalid_argument("SystolicVerifier requires a non-null Device");
    }
}

VerificationResult SystolicVerifier::verify(
    const SpeculativeBatch& batch,
    const float* target_logits_matrix,
    size_t vocab_size
) {
    if (!target_logits_matrix || vocab_size == 0 ||
        vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("Invalid target_logits_matrix or vocab_size in SystolicVerifier::verify");
    }

    VerificationResult result;
    const size_t k = batch.draft_tokens.size();

    for (size_t i = 0; i < k; ++i) {
        const int32_t draft_tok = batch.draft_tokens[i];
        if (draft_tok < 0 || static_cast<size_t>(draft_tok) >= vocab_size) {
            throw std::out_of_range("Draft token out of [0, vocab_size) in SystolicVerifier::verify");
        }
        const float* logits = target_logits_matrix + (i * vocab_size);
        const int32_t target_token = argmax_checked(logits, vocab_size);
        if (target_token == draft_tok) {
            result.accepted_tokens.push_back(target_token);
        } else {
            result.next_token = target_token;
            break;
        }
    }

    if (result.accepted_tokens.size() == k) {
        const float* final_logits = target_logits_matrix + (k * vocab_size);
        result.next_token = argmax_checked(final_logits, vocab_size);
    }

    result.acceptance_count = result.accepted_tokens.size();
    result.acceptance_rate = (k > 0) ? (static_cast<double>(result.acceptance_count) / static_cast<double>(k)) : 0.0;
    return result;
}

} // namespace tpuforge
