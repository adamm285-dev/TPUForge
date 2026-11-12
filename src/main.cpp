#include "tpuforge/device.h"
#include "tpuforge/kv_cache.h"
#include "tpuforge/speculative.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "       TPUForge Native Engine v0.1.0       " << std::endl;
    std::cout << "===========================================" << std::endl;

    // 1. Initialize Device Manager
    auto& dev_mgr = tpuforge::DeviceManager::instance();
    dev_mgr.initialize();
    auto device = dev_mgr.default_device();

    std::cout << "[Main] Active device: " << device->info().kind 
              << " | MXU systolic layout: " << device->info().mxu_dim_x 
              << "x" << device->info().mxu_dim_y << std::endl;

    // 2. Setup KV-Cache Ring Pool
    tpuforge::KVCacheManager::Config kv_config;
    kv_config.num_layers = 28;
    kv_config.num_kv_heads = 16;
    kv_config.head_dim = 128; // Systolic tile aligned
    kv_config.page_size = 16;
    
    tpuforge::KVCacheManager kv_mgr(kv_config, device);
    kv_mgr.allocate_pool(4096); // 4096 physical pages

    // 3. Demonstrate Systolic Speculative Token Verification
    tpuforge::SystolicVerifier verifier(device);
    
    tpuforge::SpeculativeBatch batch;
    batch.prompt_tokens = {101, 2054, 2003};
    batch.draft_tokens = {1037, 3231, 2005, 1037}; // 4 speculative draft tokens

    size_t vocab_size = 32000;
    // Simulate target logits matrix [K+1, vocab_size] = [5, 32000]
    std::vector<float> mock_logits(5 * vocab_size, 0.0f);
    
    // Simulate target accepting first 3 tokens and rejecting the 4th
    mock_logits[0 * vocab_size + 1037] = 10.0f; // Accept 1037
    mock_logits[1 * vocab_size + 3231] = 10.0f; // Accept 3231
    mock_logits[2 * vocab_size + 2005] = 10.0f; // Accept 2005
    mock_logits[3 * vocab_size + 9999] = 10.0f; // Target predicts 9999 instead of draft's 1037

    auto result = verifier.verify(batch, mock_logits.data(), vocab_size);

    std::cout << "\n[Speculative Results]" << std::endl;
    std::cout << "Tokens Drafted: " << batch.draft_tokens.size() << std::endl;
    std::cout << "Tokens Accepted: " << result.acceptance_count << std::endl;
    std::cout << "Correction Token: " << result.next_token << std::endl;
    std::cout << "Acceptance Rate: " << (result.acceptance_rate * 100.0) << "%" << std::endl;
    std::cout << "Effective Speedup: " << (1.0 + result.acceptance_count) << "x tokens per target forward pass." << std::endl;

    std::cout << "\n[TPUForge] Engine initialized and verified successfully." << std::endl;
    return 0;
}
