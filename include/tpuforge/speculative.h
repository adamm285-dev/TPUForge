#pragma once

#include "tpuforge/device.h"
#include <vector>
#include <cstdint>

namespace tpuforge {

struct SpeculativeBatch {
    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> draft_tokens; // Proposed tokens by draft model
    size_t num_speculative_steps{4};
};

struct VerificationResult {
    std::vector<int32_t> accepted_tokens;
    int32_t next_token{-1};
    size_t acceptance_count{0};
    double acceptance_rate{0.0};
};

class SystolicVerifier {
public:
    explicit SystolicVerifier(std::shared_ptr<Device> device);

    // Verify draft tokens in parallel using systolic matrix tiling
    // Verifies K tokens in a single forward pass over the target model
    VerificationResult verify(
        const SpeculativeBatch& batch,
        const float* target_logits_matrix, // [K+1, Vocab_size]
        size_t vocab_size
    );

private:
    std::shared_ptr<Device> device_;
};

} // namespace tpuforge
