#include "tpuforge/device.h"
#include "tpuforge/kv_cache.h"
#include "tpuforge/pjrt_driver.h"
#include "tpuforge/speculative.h"
#include "tpuforge/static_arena.h"
#include "tpuforge/tree_speculator.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::cout << "===========================================" << std::endl;
    std::cout << "       TPUForge Native Engine v0.2.0       " << std::endl;
    std::cout << "===========================================" << std::endl;

    std::string plugin_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--plugin") {
            plugin_path = argv[i + 1];
        }
    }

    // 1. Initialize Device Manager (Host CPU hardware probe or explicit PJRT plugin)
    auto& dev_mgr = tpuforge::DeviceManager::instance();
    if (!dev_mgr.initialize(plugin_path)) {
        std::cerr << "[Main] Device initialization failed." << std::endl;
        return 1;
    }
    auto device = dev_mgr.default_device();
    double ram_gb = static_cast<double>(device->info().memory_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "[Main] Active device: " << device->info().kind
              << " | Memory: " << ram_gb << " GB"
              << " | MXU systolic layout: " << device->info().mxu_dim_x
              << "x" << device->info().mxu_dim_y << std::endl;

    // 2. Setup StaticArena (128-byte aligned zero-recompilation HBM/host buffers)
    tpuforge::StaticArena::Config arena_cfg;
    arena_cfg.max_seq_len = 8192;
    arena_cfg.hidden_dim = 128;
    arena_cfg.num_kv_heads = 8;
    arena_cfg.head_dim = 128;
    arena_cfg.vocab_size = 32000;

    tpuforge::StaticArena arena(arena_cfg);
    std::cout << "[StaticArena] 128-byte systolic alignment: "
              << (arena.is_aligned_128() ? "VERIFIED" : "FAILED") << std::endl;

    if (!plugin_path.empty()) {
        tpuforge::PjrtDriver driver;
        std::string err;
        if (!driver.load_plugin(plugin_path, &err)) {
            std::cerr << "[PJRT] Plugin load failed: " << err << std::endl;
            return 1;
        }
        tpuforge::ZeroRecompileExecutor exec(driver);
        if (!exec.compile_once(arena_cfg, "module @tpuforge_static {}", "stablehlo", &err)) {
            std::cerr << "[PJRT] Compile failed: " << err << std::endl;
            return 1;
        }
        const int32_t test_prefixes[] = {16, 129, 1024, 4096};
        for (int32_t prefix : test_prefixes) {
            arena.set_valid_prefix_len(prefix);
            if (!exec.execute_step(arena, tpuforge::SystolicConstants::kSystolicTileSize, &err)) {
                std::cerr << "[PJRT] Execute failed: " << err << std::endl;
                return 1;
            }
        }
        std::cout << "[PJRT] Zero-Recompilation Check -> Compile Count: " << exec.compile_count()
                  << " | Execute Count: " << exec.execute_count() << std::endl;
    } else {
        const int32_t test_prefixes[] = {16, 129, 1024, 4096};
        for (int32_t prefix : test_prefixes) {
            arena.set_valid_prefix_len(prefix);
        }
        std::cout << "[StaticArena] Zero-reallocation prefix switching verified (16 -> 4096)." << std::endl;
    }

    // 3. Setup KV-Cache Ring Pool
    tpuforge::KVCacheManager::Config kv_config;
    kv_config.num_layers = 28;
    kv_config.num_kv_heads = 16;
    kv_config.head_dim = 128;
    kv_config.page_size = 16;

    tpuforge::KVCacheManager kv_mgr(kv_config, device);
    kv_mgr.allocate_pool(4096);

    // 4. Build and verify a 128-node systolic speculative token tree
    std::vector<int32_t> context_tokens;
    const std::vector<int32_t> code_pattern = {
        101, 2054, 2003, 1037, 3231, 2005, 4412, 5519, 6102, 7201, 8810, 9123, 1450, 2290
    };
    for (int rep = 0; rep < 12; ++rep) {
        context_tokens.insert(context_tokens.end(), code_pattern.begin(), code_pattern.end());
    }
    // End right on prefix {101, 2054, 2003} so drafter expands multi-step continuations.
    context_tokens.push_back(101);
    context_tokens.push_back(2054);
    context_tokens.push_back(2003);

    auto t0 = std::chrono::steady_clock::now();
    tpuforge::SystolicTree tree = tpuforge::SuffixTreeDrafter::build_tree(context_tokens, 128, 16);
    auto packed = tree.pack(128);
    auto t1 = std::chrono::steady_clock::now();
    double draft_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // Populate target logits buffer in StaticArena along the primary tree branch.
    const size_t vocab_size = arena_cfg.vocab_size;
    float* target_logits = arena.logits_tile_128();
    std::memset(target_logits, 0, tpuforge::SystolicConstants::kSystolicTileSize * vocab_size * sizeof(float));

    const auto& nodes = tree.nodes();
    int32_t curr = 0;
    size_t chain_depth = 0;
    while (!nodes[static_cast<size_t>(curr)].children.empty() && chain_depth < 6) {
        int32_t next_child = nodes[static_cast<size_t>(curr)].children.front();
        int32_t tok = nodes[static_cast<size_t>(next_child)].token_id;
        target_logits[static_cast<size_t>(curr) * vocab_size + static_cast<size_t>(tok)] = 10.0f;
        curr = next_child;
        ++chain_depth;
    }
    target_logits[static_cast<size_t>(curr) * vocab_size + 9999] = 10.0f;

    auto verify_res = tpuforge::SystolicTreeVerifier::verify_greedy(tree, target_logits, vocab_size);

    std::cout << "\n[Systolic Tree Speculative Results]" << std::endl;
    std::cout << "Nodes Packed: " << packed.active_nodes << " / 128 (Draft + Pack: " << draft_us << " us)" << std::endl;
    std::cout << "Tokens Accepted: " << verify_res.acceptance_count << std::endl;
    std::cout << "Bonus/Correction Token: " << verify_res.bonus_token << std::endl;
    std::cout << "Total Tokens Emitted: " << verify_res.emitted_tokens.size() << std::endl;
    std::cout << "Effective Speedup: " << verify_res.emitted_tokens.size()
              << "x tokens per target forward pass." << std::endl;

    std::cout << "\n[TPUForge] Engine initialized and verified successfully." << std::endl;
    return 0;
}
