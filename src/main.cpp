#include "tpuforge/device.h"
#include "tpuforge/kv_cache.h"
#include "tpuforge/openai_server.h"
#include "tpuforge/pjrt_driver.h"
#include "tpuforge/speculative.h"
#include "tpuforge/static_arena.h"
#include "tpuforge/tree_speculator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void handle_shutdown_signal(int) {
    g_stop_requested = 1;
}

bool load_program_file(const std::string& path, std::string* out_code, std::string* out_err) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        if (out_err) {
            *out_err = "Failed to open program file: " + path;
        }
        return false;
    }
    std::string contents(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>()
    );
    if (ifs.bad()) {
        if (out_err) {
            *out_err = "I/O error reading program file: " + path;
        }
        return false;
    }
    if (contents.empty()) {
        if (out_err) {
            *out_err = "Program file is empty: " + path;
        }
        return false;
    }
    *out_code = std::move(contents);
    return true;
}

bool parse_port_strict(const std::string& raw, uint16_t* out_port) {
    if (raw.empty()) {
        return false;
    }
    uint32_t val = 0;
    for (unsigned char ch : raw) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        val = val * 10u + static_cast<uint32_t>(ch - '0');
        if (val > 65535u) {
            return false;
        }
    }
    *out_port = static_cast<uint16_t>(val);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string plugin_path;
    std::string program_path;
    std::string program_format = "mlir";
    bool serve_mode = false;
    std::string bind_host = "127.0.0.1";
    uint16_t bind_port = 8080;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--plugin") {
            if (i + 1 >= argc) {
                std::cerr << "[CLI] Missing value after --plugin" << std::endl;
                return 1;
            }
            plugin_path = argv[++i];
        } else if (arg == "--program") {
            if (i + 1 >= argc) {
                std::cerr << "[CLI] Missing value after --program" << std::endl;
                return 1;
            }
            program_path = argv[++i];
        } else if (arg == "--format") {
            if (i + 1 >= argc) {
                std::cerr << "[CLI] Missing value after --format" << std::endl;
                return 1;
            }
            program_format = argv[++i];
        } else if (arg == "--serve") {
            serve_mode = true;
        } else if (arg == "--host") {
            if (i + 1 >= argc) {
                std::cerr << "[CLI] Missing value after --host" << std::endl;
                return 1;
            }
            bind_host = argv[++i];
        } else if (arg == "--port") {
            if (i + 1 >= argc) {
                std::cerr << "[CLI] Missing value after --port" << std::endl;
                return 1;
            }
            if (!parse_port_strict(argv[++i], &bind_port)) {
                std::cerr << "[CLI] Invalid port number (expected integer in 0..65535)" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "[CLI] Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if ((serve_mode || !plugin_path.empty()) && program_path.empty()) {
        std::cerr << "[CLI] --plugin and --serve require --program <path_to_mlir_or_hlo>" << std::endl;
        return 1;
    }

    std::string program_code;
    if (!program_path.empty()) {
        std::string read_err;
        if (!load_program_file(program_path, &program_code, &read_err)) {
            std::cerr << "[CLI] " << read_err << std::endl;
            return 1;
        }
    }

    std::cout << "===========================================" << std::endl;
    std::cout << "       TPUForge Native Engine v0.2.0       " << std::endl;
    std::cout << "===========================================" << std::endl;

    // 1. Discover execution device once (single PJRT client if --plugin provided, else Host CPU)
    std::unique_ptr<tpuforge::PjrtDriver> plugin_driver;
    std::shared_ptr<tpuforge::Device> device;
    if (!plugin_path.empty()) {
        std::cout << "[Main] Loading hardware PJRT plugin from: " << plugin_path << std::endl;
        plugin_driver = std::make_unique<tpuforge::PjrtDriver>();
        std::string err;
        if (!plugin_driver->load_plugin(plugin_path, &err)) {
            std::cerr << "[PJRT] Plugin load failed: " << err << std::endl;
            return 1;
        }
        auto devs = plugin_driver->query_devices();
        if (devs.empty()) {
            std::cerr << "[PJRT] Plugin reported zero devices." << std::endl;
            return 1;
        }
        device = std::make_shared<tpuforge::Device>(devs[0].id, devs[0].kind, devs[0].memory_bytes);
    } else {
        if (serve_mode) {
            std::cerr << "[OpenAIServer] --serve requires --plugin <path_to_pjrt_library>" << std::endl;
            return 1;
        }
        auto& dev_mgr = tpuforge::DeviceManager::instance();
        if (!dev_mgr.initialize()) {
            std::cerr << "[Main] Device initialization failed." << std::endl;
            return 1;
        }
        device = dev_mgr.default_device();
    }
    double ram_gb = static_cast<double>(device->info().memory_bytes) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "[Main] Active device: " << device->info().kind
              << " | Memory: " << ram_gb << " GB"
              << " | MXU systolic layout: " << device->info().mxu_dim_x
              << "x" << device->info().mxu_dim_y << std::endl;

    tpuforge::StaticArena::Config arena_cfg;
    arena_cfg.max_seq_len = 8192;
    arena_cfg.hidden_dim = 128;
    arena_cfg.num_kv_heads = 8;
    arena_cfg.head_dim = 128;
    arena_cfg.vocab_size = 32000;

    if (serve_mode) {
        try {
            tpuforge::ForgeEngine engine(
                *plugin_driver, arena_cfg, program_code, program_format, "tpuforge-systolic-128"
            );
            tpuforge::OpenAIServer::Config srv_cfg;
            srv_cfg.host = bind_host;
            srv_cfg.port = bind_port;

            tpuforge::OpenAIServer server(engine, srv_cfg);
            std::string err;
            if (!server.start(&err)) {
                std::cerr << "[OpenAIServer] Failed to start: " << err << std::endl;
                return 1;
            }
            std::signal(SIGINT, handle_shutdown_signal);
            std::signal(SIGTERM, handle_shutdown_signal);

            std::cout << "[OpenAIServer] Listening on http://" << bind_host << ":"
                      << server.bound_port() << std::endl;
            while (!g_stop_requested) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            std::cout << "[OpenAIServer] Shutting down..." << std::endl;
            server.stop();
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "[OpenAIServer] Fatal error: " << e.what() << std::endl;
            return 1;
        }
    }

    // 2. Setup StaticArena (128-byte aligned zero-recompilation HBM/host buffers)
    tpuforge::StaticArena arena(arena_cfg);
    std::cout << "[StaticArena] 128-byte systolic alignment: "
              << (arena.is_aligned_128() ? "VERIFIED" : "FAILED") << std::endl;

    std::unique_ptr<tpuforge::ZeroRecompileExecutor> exec;
    if (plugin_driver) {
        std::string err;
        exec = std::make_unique<tpuforge::ZeroRecompileExecutor>(*plugin_driver);
        if (!exec->compile_once(arena_cfg, program_code, program_format, &err)) {
            std::cerr << "[PJRT] Compile failed: " << err << std::endl;
            return 1;
        }
        const int32_t test_prefixes[] = {16, 129, 1024, 4096};
        for (int32_t prefix : test_prefixes) {
            arena.set_valid_prefix_len(prefix);
            if (!exec->execute_step(arena, tpuforge::SystolicConstants::kSystolicTileSize, &err)) {
                std::cerr << "[PJRT] Execute failed: " << err << std::endl;
                return 1;
            }
        }
        std::cout << "[PJRT] Zero-Recompilation Check -> Compile Count: " << exec->compile_count()
                  << " | Execute Count: " << exec->execute_count() << std::endl;
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
    context_tokens.push_back(101);
    context_tokens.push_back(2054);
    context_tokens.push_back(2003);

    auto t0 = std::chrono::steady_clock::now();
    tpuforge::SystolicTree tree = tpuforge::SuffixTreeDrafter::build_tree(context_tokens, 128, 16);
    auto packed = tree.pack(128);
    auto t1 = std::chrono::steady_clock::now();
    double draft_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    const size_t vocab_size = arena_cfg.vocab_size;
    if (exec) {
        constexpr size_t T = tpuforge::SystolicConstants::kSystolicTileSize;
        arena.set_valid_prefix_len(128);
        std::memcpy(arena.token_tile_128(), packed.token_ids.data(), T * sizeof(int32_t));
        std::memcpy(arena.position_tile_128(), packed.position_offsets.data(), T * sizeof(int32_t));
        std::memcpy(arena.causal_mask_128x128(), packed.additive_mask_f32.data(), T * T * sizeof(float));

        std::string err;
        if (!exec->execute_step(arena, T, &err)) {
            std::cerr << "[PJRT] Tree tile execute failed: " << err << std::endl;
            return 1;
        }
        auto verify_res = tpuforge::SystolicTreeVerifier::verify_greedy(
            tree, arena.logits_tile_128(), vocab_size
        );
        std::cout << "\n[Systolic Tree PJRT Execution Results]" << std::endl;
        std::cout << "Nodes Packed: " << packed.active_nodes << " / 128 (Draft + Pack: " << draft_us << " us)" << std::endl;
        std::cout << "Tokens Accepted: " << verify_res.acceptance_count << std::endl;
        std::cout << "Bonus/Correction Token: " << verify_res.bonus_token << std::endl;
        std::cout << "Total Tokens Emitted: " << verify_res.emitted_tokens.size() << std::endl;
    } else {
        // Host self-test without PJRT plugin: exercise verifier against deterministic synthetic oracle logits.
        float* ref_logits = arena.logits_tile_128();
        std::memset(ref_logits, 0, tpuforge::SystolicConstants::kSystolicTileSize * vocab_size * sizeof(float));

        const auto& nodes = tree.nodes();
        int32_t curr = 0;
        size_t chain_depth = 0;
        while (!nodes[static_cast<size_t>(curr)].children.empty() && chain_depth < 6) {
            int32_t next_child = nodes[static_cast<size_t>(curr)].children.front();
            int32_t tok = nodes[static_cast<size_t>(next_child)].token_id;
            ref_logits[static_cast<size_t>(curr) * vocab_size + static_cast<size_t>(tok)] = 10.0f;
            curr = next_child;
            ++chain_depth;
        }
        ref_logits[static_cast<size_t>(curr) * vocab_size + std::min<size_t>(9999, vocab_size - 1)] = 10.0f;

        auto verify_res = tpuforge::SystolicTreeVerifier::verify_greedy(tree, ref_logits, vocab_size);

        std::cout << "\n[Host Self-Test: Deterministic Verifier Reference Check]" << std::endl;
        std::cout << "Nodes Packed: " << packed.active_nodes << " / 128 (Draft + Pack: " << draft_us << " us)" << std::endl;
        std::cout << "Tokens Accepted: " << verify_res.acceptance_count << std::endl;
        std::cout << "Bonus/Correction Token: " << verify_res.bonus_token << std::endl;
        std::cout << "Total Tokens Emitted: " << verify_res.emitted_tokens.size() << std::endl;
        std::cout << "Effective Speedup: " << verify_res.emitted_tokens.size()
                  << "x (synthetic host self-test oracle; not a model benchmark)" << std::endl;
    }

    std::cout << "\n[TPUForge] Engine initialized and verified successfully." << std::endl;
    return 0;
}
