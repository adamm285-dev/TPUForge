#include "tpuforge/speculative.h"
#include <algorithm>
#include <iostream>

namespace tpuforge {

SystolicVerifier::SystolicVerifier(std::shared_ptr<Device> device)
    : device_(device) {}

VerificationResult SystolicVerifier::verify(
    const SpeculativeBatch& batch,
    const float* target_logits_matrix,
    size_t vocab_size
) {
    VerificationResult result;
    size_t k = batch.draft_tokens.size();

    // In a systolic array, all K draft verification steps are computed in parallel
    // via a single batched forward pass: [K+1, Hidden] x [Hidden, Vocab]
    for (size_t i = 0; i < k; ++i) {
        const float* logits = target_logits_matrix + (i * vocab_size);
        
        // Find greedy target token (argmax)
        int32_t target_token = 0;
        float max_val = logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (logits[v] > max_val) {
                max_val = logits[v];
                target_token = static_cast<int32_t>(v);
            }
        }

        // Compare draft token against target model argmax
        if (target_token == batch.draft_tokens[i]) {
            result.accepted_tokens.push_back(target_token);
        } else {
            // Divergence detected. Reject current draft token and accept target model's correction
            result.next_token = target_token;
            break;
        }
    }

    // If all draft tokens were accepted, choose token from the (K+1)-th position
    if (result.accepted_tokens.size() == k) {
        const float* final_logits = target_logits_matrix + (k * vocab_size);
        int32_t final_token = 0;
        float max_val = final_logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (final_logits[v] > max_val) {
                max_val = final_logits[v];
                final_token = static_cast<int32_t>(v);
            }
        }
        result.next_token = final_token;
    }

    result.acceptance_count = result.accepted_tokens.size();
    result.acceptance_rate = (k > 0) ? (static_cast<double>(result.acceptance_count) / k) : 0.0;

    return result;
}

} // namespace tpuforge
