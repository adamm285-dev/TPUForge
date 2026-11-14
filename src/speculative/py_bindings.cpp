#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tpuforge/device.h"
#include "tpuforge/kv_cache.h"
#include "tpuforge/openai_server.h"
#include "tpuforge/pjrt_driver.h"
#include "tpuforge/static_arena.h"
#include "tpuforge/tree_speculator.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> make_1d_numpy(const T* src, size_t n) {
    if (n > 0 && !src) {
        throw std::logic_error("Null source buffer in make_1d_numpy");
    }
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::length_error("1D numpy allocation size overflow");
    }
    auto buf = std::make_unique<T[]>(n);
    if (n > 0) {
        std::memcpy(buf.get(), src, n * sizeof(T));
    }
    nb::capsule owner(buf.get(), [](void* p) noexcept { delete[] static_cast<T*>(p); });
    T* raw = buf.release();
    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(raw, 1, shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> make_2d_numpy(const T* src, size_t rows, size_t cols) {
    if (cols > 0 && rows > std::numeric_limits<size_t>::max() / cols) {
        throw std::length_error("2D numpy element count overflow");
    }
    const size_t elems = rows * cols;
    if (elems > 0 && !src) {
        throw std::logic_error("Null source buffer in make_2d_numpy");
    }
    if (elems > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::length_error("2D numpy byte size overflow");
    }
    auto buf = std::make_unique<T[]>(elems);
    if (elems > 0) {
        std::memcpy(buf.get(), src, elems * sizeof(T));
    }
    nb::capsule owner(buf.get(), [](void* p) noexcept { delete[] static_cast<T*>(p); });
    T* raw = buf.release();
    size_t shape[2] = {rows, cols};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(raw, 2, shape, owner);
}

nb::object invoke_py_tile_cb(
    const nb::callable& fn,
    int32_t valid_prefix_len,
    const int32_t* tok_ptr,
    const int32_t* pos_ptr,
    const float* mask_ptr,
    size_t T
) {
    auto tok_arr = make_1d_numpy<int32_t>(tok_ptr, T);
    auto pos_arr = make_1d_numpy<int32_t>(pos_ptr, T);
    auto mask_arr = make_2d_numpy<float>(mask_ptr, T, T);
    return fn(valid_prefix_len, tok_arr, pos_arr, mask_arr);
}

bool is_python_active() noexcept {
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsInitialized() != 0 && Py_IsFinalizing() == 0;
#else
    return Py_IsInitialized() != 0 && _Py_IsFinalizing() == 0;
#endif
}

std::mutex g_live_servers_mu;
std::vector<std::weak_ptr<tpuforge::OpenAIServer>> g_live_servers;

void register_live_server(const std::shared_ptr<tpuforge::OpenAIServer>& srv) {
    if (!srv) {
        return;
    }
    std::lock_guard lk(g_live_servers_mu);
    g_live_servers.erase(
        std::remove_if(
            g_live_servers.begin(),
            g_live_servers.end(),
            [](const std::weak_ptr<tpuforge::OpenAIServer>& wp) { return wp.expired(); }
        ),
        g_live_servers.end()
    );
    g_live_servers.push_back(srv);
}

void shutdown_all_live_servers() {
    std::vector<std::shared_ptr<tpuforge::OpenAIServer>> active;
    {
        std::lock_guard lk(g_live_servers_mu);
        for (const auto& wp : g_live_servers) {
            if (auto sp = wp.lock()) {
                active.push_back(std::move(sp));
            }
        }
        g_live_servers.clear();
    }
    if (is_python_active()) {
        nb::gil_scoped_release nogil;
        for (auto& srv : active) {
            srv->stop();
        }
        active.clear();
    } else {
        for (auto& srv : active) {
            srv->stop();
        }
        active.clear();
    }
}

struct PyOpenAIServer {
    std::shared_ptr<tpuforge::OpenAIServer> server;

    PyOpenAIServer(tpuforge::ForgeEngine& eng, const tpuforge::OpenAIServer::Config& cfg)
        : server(std::make_shared<tpuforge::OpenAIServer>(eng, cfg)) {
        register_live_server(server);
    }

    explicit PyOpenAIServer(tpuforge::ForgeEngine& eng)
        : server(std::make_shared<tpuforge::OpenAIServer>(eng)) {
        register_live_server(server);
    }

    ~PyOpenAIServer() {
        if (server) {
            if (is_python_active()) {
                nb::gil_scoped_release nogil;
                server->stop();
                server.reset();
            } else {
                server->stop();
                server.reset();
            }
        }
    }
};

} // namespace

NB_MODULE(_tpuforge_ext, m) {
    m.doc() = "TPUForge native C++20 systolic tree and static PJRT runtime bindings";
    m.def("_shutdown_all_servers", &shutdown_all_live_servers);
    nb::module_::import_("atexit").attr("register")(m.attr("_shutdown_all_servers"));
    m.def("infer_mxu_dim", &tpuforge::infer_mxu_dim, nb::arg("device_kind"),
          "Infer systolic MXU tile dimension (128 or 256) from a PJRT/XLA device_kind string.");

    nb::class_<tpuforge::DeviceInfo>(m, "DeviceInfo")
        .def_ro("id", &tpuforge::DeviceInfo::id)
        .def_ro("kind", &tpuforge::DeviceInfo::kind)
        .def_ro("memory_bytes", &tpuforge::DeviceInfo::memory_bytes)
        .def_ro("is_tpu", &tpuforge::DeviceInfo::is_tpu)
        .def_ro("mxu_dim_x", &tpuforge::DeviceInfo::mxu_dim_x)
        .def_ro("mxu_dim_y", &tpuforge::DeviceInfo::mxu_dim_y);

    nb::class_<tpuforge::KVCacheManager::Config>(m, "KVCacheConfig")
        .def(nb::init<>())
        .def_rw("num_layers", &tpuforge::KVCacheManager::Config::num_layers)
        .def_rw("num_kv_heads", &tpuforge::KVCacheManager::Config::num_kv_heads)
        .def_rw("head_dim", &tpuforge::KVCacheManager::Config::head_dim)
        .def_rw("max_seq_len", &tpuforge::KVCacheManager::Config::max_seq_len)
        .def_rw("page_size", &tpuforge::KVCacheManager::Config::page_size)
        .def("set_dtype", [](tpuforge::KVCacheManager::Config& cfg, const std::string& dtype_name) {
            std::string lower;
            lower.reserve(dtype_name.size());
            for (unsigned char ch : dtype_name) {
                lower.push_back(static_cast<char>(std::tolower(ch)));
            }
            if (lower == "bf16" || lower == "bfloat16") {
                cfg.dtype = PJRT_Buffer_Type_BF16;
            } else if (lower == "fp16" || lower == "float16") {
                cfg.dtype = PJRT_Buffer_Type_F16;
            } else if (lower == "fp32" || lower == "float32") {
                cfg.dtype = PJRT_Buffer_Type_F32;
            } else if (lower == "fp8" || lower == "f8e4m3fn") {
                cfg.dtype = PJRT_Buffer_Type_F8E4M3FN;
            } else {
                throw std::invalid_argument("Unsupported KVCacheConfig dtype: " + dtype_name);
            }
        }, nb::arg("dtype_name"));

    nb::class_<tpuforge::IciRingPlan>(m, "IciRingPlan")
        .def_ro("ring_size", &tpuforge::IciRingPlan::ring_size)
        .def_ro("total_seq_len", &tpuforge::IciRingPlan::total_seq_len)
        .def_ro("chunk_seq_len", &tpuforge::IciRingPlan::chunk_seq_len)
        .def_ro("pages_per_rank", &tpuforge::IciRingPlan::pages_per_rank)
        .def_ro("bytes_per_rank", &tpuforge::IciRingPlan::bytes_per_rank)
        .def_ro("total_cluster_bytes", &tpuforge::IciRingPlan::total_cluster_bytes)
        .def_ro("send_perm", &tpuforge::IciRingPlan::send_perm)
        .def_ro("chunk_start_positions", &tpuforge::IciRingPlan::chunk_start_positions);

    nb::class_<tpuforge::IciRingPlanner>(m, "IciRingPlanner")
        .def_static("plan", &tpuforge::IciRingPlanner::plan,
                    nb::arg("kv_config"), nb::arg("ring_size"))
        .def_static("source_rank_at_step", &tpuforge::IciRingPlanner::source_rank_at_step,
                    nb::arg("dst_rank"), nb::arg("step"), nb::arg("ring_size"))
        .def_static("causal_active_hops", &tpuforge::IciRingPlanner::causal_active_hops,
                    nb::arg("query_rank"), nb::arg("ring_size"),
                    nb::arg("chunk_seq_len"), nb::arg("valid_seq_len"));

    nb::class_<tpuforge::TreeNode>(m, "TreeNode")
        .def_ro("token_id", &tpuforge::TreeNode::token_id)
        .def_ro("parent_idx", &tpuforge::TreeNode::parent_idx)
        .def_ro("depth", &tpuforge::TreeNode::depth)
        .def_ro("draft_prob", &tpuforge::TreeNode::draft_prob)
        .def_ro("cumulative_logprob", &tpuforge::TreeNode::cumulative_logprob)
        .def_ro("children", &tpuforge::TreeNode::children);

    nb::class_<tpuforge::SystolicTree>(m, "SystolicTree")
        .def(nb::init<>())
        .def(nb::init<int32_t, size_t>(),
             nb::arg("root_last_token"),
             nb::arg("tile_size") = tpuforge::SystolicConstants::kSystolicTileSize)
        .def("reset", &tpuforge::SystolicTree::reset,
             nb::arg("root_last_token"), nb::arg("tile_size") = 0)
        .def("tile_size", &tpuforge::SystolicTree::tile_size)
        .def("add_child", &tpuforge::SystolicTree::add_child,
             nb::arg("parent_idx"), nb::arg("token_id"), nb::arg("draft_prob") = 1.0f)
        .def("num_nodes", &tpuforge::SystolicTree::num_nodes)
        .def("nodes", &tpuforge::SystolicTree::nodes)
        .def("pack", [](const tpuforge::SystolicTree& tree, int32_t valid_prefix_len) {
            auto packed = tree.pack(valid_prefix_len);
            const size_t T = packed.tile_size;
            nb::dict d;
            d["tile_size"] = T;
            d["active_nodes"] = packed.active_nodes;
            d["token_ids"] = make_1d_numpy<int32_t>(packed.token_ids.data(), T);
            d["position_offsets"] = make_1d_numpy<int32_t>(packed.position_offsets.data(), T);
            d["mask_u8"] = make_2d_numpy<uint8_t>(packed.mask_u8.data(), T, T);
            d["additive_mask_f32"] = make_2d_numpy<float>(packed.additive_mask_f32.data(), T, T);
            return d;
        }, nb::arg("valid_prefix_len") = 0);

    nb::class_<tpuforge::SuffixTreeDrafter>(m, "SuffixTreeDrafter")
        .def_static("build_tree", &tpuforge::SuffixTreeDrafter::build_tree,
                    nb::arg("context_tokens"),
                    nb::arg("max_nodes") = tpuforge::SystolicTree::kMaxNodes,
                    nb::arg("max_depth") = 16,
                    nb::arg("tile_size") = 0);

    nb::class_<tpuforge::TreeVerificationResult>(m, "TreeVerificationResult")
        .def_ro("emitted_tokens", &tpuforge::TreeVerificationResult::emitted_tokens)
        .def_ro("accepted_draft_tokens", &tpuforge::TreeVerificationResult::accepted_draft_tokens)
        .def_ro("accepted_node_indices", &tpuforge::TreeVerificationResult::accepted_node_indices)
        .def_ro("bonus_token", &tpuforge::TreeVerificationResult::bonus_token)
        .def_ro("acceptance_count", &tpuforge::TreeVerificationResult::acceptance_count)
        .def_ro("terminal_depth", &tpuforge::TreeVerificationResult::terminal_depth);

    nb::class_<tpuforge::SystolicTreeVerifier>(m, "SystolicTreeVerifier")
        .def_static("verify_greedy",
            [](const tpuforge::SystolicTree& tree,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits) {
                if (logits.shape(0) != tree.tile_size()) {
                    throw std::invalid_argument("Expected logits shape [tree.tile_size(), vocab_size]");
                }
                const size_t vocab_size = logits.shape(1);
                if (vocab_size == 0 ||
                    vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw std::invalid_argument("vocab_size out of valid [1, INT32_MAX] range");
                }
                return tpuforge::SystolicTreeVerifier::verify_greedy(
                    tree, logits.data(), vocab_size);
            },
            nb::arg("tree"), nb::arg("target_logits_128").noconvert())
        .def_static("verify_greedy_tokens",
            [](const tpuforge::SystolicTree& tree,
               nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> argmax_tokens,
               size_t vocab_size) {
                if (argmax_tokens.shape(0) != tree.tile_size()) {
                    throw std::invalid_argument("Expected target_argmax_tokens shape [tree.tile_size()]");
                }
                if (vocab_size == 0 ||
                    vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw std::invalid_argument("vocab_size out of valid [1, INT32_MAX] range");
                }
                return tpuforge::SystolicTreeVerifier::verify_greedy_tokens(
                    tree, argmax_tokens.data(), vocab_size);
            },
            nb::arg("tree"), nb::arg("target_argmax_tokens").noconvert(), nb::arg("vocab_size"))
        .def_static("verify_stochastic",
            [](const tpuforge::SystolicTree& tree,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits,
               float temperature,
               uint64_t seed) {
                if (logits.shape(0) != tree.tile_size()) {
                    throw std::invalid_argument("Expected logits shape [tree.tile_size(), vocab_size]");
                }
                const size_t vocab_size = logits.shape(1);
                if (vocab_size == 0 ||
                    vocab_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw std::invalid_argument("vocab_size out of valid [1, INT32_MAX] range");
                }
                return tpuforge::SystolicTreeVerifier::verify_stochastic(
                    tree, logits.data(), vocab_size, temperature, seed);
            },
            nb::arg("tree"), nb::arg("target_logits_128").noconvert(),
            nb::arg("temperature") = 1.0f, nb::arg("seed") = 0ULL);

    nb::class_<tpuforge::StaticArena::Config>(m, "StaticArenaConfig")
        .def(nb::init<>())
        .def_rw("max_seq_len", &tpuforge::StaticArena::Config::max_seq_len)
        .def_rw("hidden_dim", &tpuforge::StaticArena::Config::hidden_dim)
        .def_rw("num_kv_heads", &tpuforge::StaticArena::Config::num_kv_heads)
        .def_rw("head_dim", &tpuforge::StaticArena::Config::head_dim)
        .def_rw("vocab_size", &tpuforge::StaticArena::Config::vocab_size)
        .def_rw("tile_size", &tpuforge::StaticArena::Config::tile_size);

    nb::class_<tpuforge::StaticArena>(m, "StaticArena")
        .def(nb::init<const tpuforge::StaticArena::Config&>(), nb::arg("config"))
        .def("tile_size", &tpuforge::StaticArena::tile_size)
        .def("is_aligned_128", &tpuforge::StaticArena::is_aligned_128)
        .def("valid_prefix_len", &tpuforge::StaticArena::valid_prefix_len)
        .def("set_valid_prefix_len", &tpuforge::StaticArena::set_valid_prefix_len, nb::arg("len"))
        .def("kv_row_elems", &tpuforge::StaticArena::kv_row_elems)
        .def("set_tile_inputs",
            [](tpuforge::StaticArena& arena,
               nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> token_ids,
               nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> pos_offsets,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> additive_mask) {
                if (!arena.token_tile() || !arena.position_tile() || !arena.causal_mask()) {
                    throw std::logic_error("StaticArena has been moved from");
                }
                const size_t T = arena.tile_size();
                if (token_ids.shape(0) != T || pos_offsets.shape(0) != T ||
                    additive_mask.shape(0) != T || additive_mask.shape(1) != T) {
                    throw std::invalid_argument("Expected [tile_size] token/pos and [tile_size, tile_size] mask");
                }
                std::memcpy(arena.token_tile(), token_ids.data(), T * sizeof(int32_t));
                std::memcpy(arena.position_tile(), pos_offsets.data(), T * sizeof(int32_t));
                std::memcpy(arena.causal_mask(), additive_mask.data(), T * T * sizeof(float));
            },
            nb::arg("token_ids").noconvert(),
            nb::arg("position_offsets").noconvert(),
            nb::arg("additive_mask").noconvert())
        .def("commit_accepted_kv_rows",
            [](tpuforge::StaticArena& arena,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> step_k_tile,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> step_v_tile,
               const std::vector<int32_t>& accepted_row_indices) {
                const size_t T = arena.tile_size();
                if (step_k_tile.shape(0) != T || step_v_tile.shape(0) != T ||
                    step_k_tile.shape(1) != arena.kv_row_elems() ||
                    step_v_tile.shape(1) != arena.kv_row_elems()) {
                    throw std::invalid_argument("KV tile shape mismatch with StaticArena");
                }
                arena.commit_accepted_kv_rows(
                    step_k_tile.data(), step_v_tile.data(), accepted_row_indices);
            },
            nb::arg("step_k_tile").noconvert(),
            nb::arg("step_v_tile").noconvert(),
            nb::arg("accepted_row_indices"))
        .def("get_kv_prefix_k_slice",
            [](const tpuforge::StaticArena& arena, size_t start_row, size_t num_rows) {
                if (!arena.kv_prefix_k()) {
                    throw std::logic_error("StaticArena has been moved from");
                }
                const size_t max_rows = arena.config().max_seq_len;
                if (num_rows > max_rows || start_row > max_rows - num_rows) {
                    throw std::out_of_range("KV slice exceeds max_seq_len");
                }
                const size_t cols = arena.kv_row_elems();
                return make_2d_numpy<float>(
                    arena.kv_prefix_k() + (start_row * cols), num_rows, cols);
            },
            nb::arg("start_row"), nb::arg("num_rows"))
        .def("get_logits_tile",
            [](const tpuforge::StaticArena& arena) {
                if (!arena.logits_tile()) {
                    throw std::logic_error("StaticArena has been moved from");
                }
                return make_2d_numpy<float>(
                    arena.logits_tile(),
                    arena.tile_size(),
                    arena.config().vocab_size);
            });

    nb::class_<tpuforge::PjrtDriver>(m, "PjrtDriver")
        .def(nb::init<>())
        .def("load_plugin", [](tpuforge::PjrtDriver& drv, const std::string& path) {
            std::string err;
            bool ok = false;
            {
                nb::gil_scoped_release nogil;
                ok = drv.load_plugin(path, &err);
            }
            return nb::make_tuple(ok, err);
        }, nb::arg("path"))
        .def("unload", &tpuforge::PjrtDriver::unload, nb::call_guard<nb::gil_scoped_release>())
        .def("is_loaded", &tpuforge::PjrtDriver::is_loaded)
        .def("platform_name", &tpuforge::PjrtDriver::platform_name)
        .def("query_devices", &tpuforge::PjrtDriver::query_devices);

    nb::class_<tpuforge::ZeroRecompileExecutor>(m, "ZeroRecompileExecutor")
        .def(nb::init<tpuforge::PjrtDriver&>(), nb::arg("driver"), nb::keep_alive<1, 2>())
        .def("compile_once",
            [](tpuforge::ZeroRecompileExecutor& exec,
               const tpuforge::StaticArena::Config& config,
               const std::string& program_code,
               const std::string& program_format) {
                std::string err;
                bool ok = false;
                {
                    nb::gil_scoped_release nogil;
                    ok = exec.compile_once(config, program_code, program_format, &err);
                }
                return nb::make_tuple(ok, err);
            },
            nb::arg("config"), nb::arg("program_code"), nb::arg("program_format") = "mlir")
        .def("execute_step",
            [](tpuforge::ZeroRecompileExecutor& exec,
               tpuforge::StaticArena& arena,
               size_t tile_rows) {
                std::string err;
                bool ok = false;
                {
                    nb::gil_scoped_release nogil;
                    ok = exec.execute_step(arena, tile_rows, &err);
                }
                return nb::make_tuple(ok, err);
            },
            nb::arg("arena"), nb::arg("tile_rows") = 0)
        .def("compile_count", &tpuforge::ZeroRecompileExecutor::compile_count)
        .def("execute_count", &tpuforge::ZeroRecompileExecutor::execute_count)
        .def("num_outputs", &tpuforge::ZeroRecompileExecutor::num_outputs)
        .def("is_compiled", &tpuforge::ZeroRecompileExecutor::is_compiled);

    nb::class_<tpuforge::ChatMessage>(m, "ChatMessage")
        .def(nb::init<>())
        .def(nb::init<std::string, std::string>(), nb::arg("role"), nb::arg("content"))
        .def_rw("role", &tpuforge::ChatMessage::role)
        .def_rw("content", &tpuforge::ChatMessage::content);

    nb::class_<tpuforge::ByteTokenizer>(m, "ByteTokenizer")
        .def(nb::init<size_t>(), nb::arg("vocab_size") = tpuforge::ByteTokenizer::kMinVocabSize)
        .def("vocab_size", &tpuforge::ByteTokenizer::vocab_size)
        .def("encode", &tpuforge::ByteTokenizer::encode,
             nb::arg("text"), nb::arg("add_bos") = false, nb::arg("add_eos") = false)
        .def("encode_chat", &tpuforge::ByteTokenizer::encode_chat, nb::arg("messages"))
        .def("decode", &tpuforge::ByteTokenizer::decode,
             nb::arg("tokens"), nb::arg("skip_special") = true);

    nb::class_<tpuforge::GenerationRequest>(m, "GenerationRequest")
        .def(nb::init<>())
        .def_rw("messages", &tpuforge::GenerationRequest::messages)
        .def_rw("prompt_tokens", &tpuforge::GenerationRequest::prompt_tokens)
        .def_rw("eos_token_ids", &tpuforge::GenerationRequest::eos_token_ids)
        .def_rw("max_tokens", &tpuforge::GenerationRequest::max_tokens)
        .def_rw("temperature", &tpuforge::GenerationRequest::temperature)
        .def_rw("seed", &tpuforge::GenerationRequest::seed)
        .def_rw("ignore_eos", &tpuforge::GenerationRequest::ignore_eos);

    nb::class_<tpuforge::GenerationResult>(m, "GenerationResult")
        .def_ro("text", &tpuforge::GenerationResult::text)
        .def_ro("token_ids", &tpuforge::GenerationResult::token_ids)
        .def_ro("finish_reason", &tpuforge::GenerationResult::finish_reason)
        .def_ro("prompt_tokens_count", &tpuforge::GenerationResult::prompt_tokens_count)
        .def_ro("completion_tokens_count", &tpuforge::GenerationResult::completion_tokens_count)
        .def_ro("total_tokens_count", &tpuforge::GenerationResult::total_tokens_count)
        .def_ro("target_forward_passes", &tpuforge::GenerationResult::target_forward_passes)
        .def_ro("decode_forward_passes", &tpuforge::GenerationResult::decode_forward_passes)
        .def_ro("draft_tokens_accepted", &tpuforge::GenerationResult::draft_tokens_accepted)
        .def_ro("mean_tokens_per_decode_pass", &tpuforge::GenerationResult::mean_tokens_per_decode_pass)
        .def_ro("compile_count", &tpuforge::GenerationResult::compile_count);

    nb::class_<tpuforge::ForgeEngine>(m, "ForgeEngine")
        .def(nb::init<
                 tpuforge::PjrtDriver&,
                 const tpuforge::StaticArena::Config&,
                 const std::string&,
                 const std::string&,
                 std::string>(),
             nb::arg("driver"),
             nb::arg("config"),
             nb::arg("program_code"),
             nb::arg("program_format") = "mlir",
             nb::arg("model_id") = "tpuforge-systolic-128",
             nb::keep_alive<1, 2>())
        .def("__init__",
             [](tpuforge::ForgeEngine* self,
                const tpuforge::StaticArena::Config& config,
                nb::object target_step_obj,
                nb::object commit_kv_obj,
                std::string model_id,
                nb::object prefill_step_obj,
                nb::object greedy_step_obj) {
                 auto py_deleter = [](nb::callable* p) noexcept {
                     if (!p) {
                         return;
                     }
                     if (is_python_active()) {
                         nb::gil_scoped_acquire gil;
                         delete p;
                     }
                 };

                 auto wrap_opt_cb = [&py_deleter](const nb::object& obj) -> std::shared_ptr<nb::callable> {
                     if (obj.is_none()) {
                         return {};
                     }
                     return std::shared_ptr<nb::callable>(
                         new nb::callable(nb::cast<nb::callable>(obj)), py_deleter
                     );
                 };

                 auto step_cb = wrap_opt_cb(target_step_obj);
                 auto kv_cb = wrap_opt_cb(commit_kv_obj);
                 auto prefill_cb = wrap_opt_cb(prefill_step_obj);
                 auto greedy_cb = wrap_opt_cb(greedy_step_obj);

                 const size_t vocab_size = config.vocab_size;
                 const size_t T = config.tile_size;

                 tpuforge::ForgeEngine::TargetStepFn cpp_step;
                 if (step_cb) {
                     cpp_step = [step_cb, vocab_size, T](
                         int32_t valid_prefix_len,
                         const int32_t* tok_ptr,
                         const int32_t* pos_ptr,
                         const float* mask_ptr,
                         float* out_logits
                     ) {
                         if (!is_python_active()) {
                             throw std::runtime_error("Python interpreter is finalizing");
                         }
                         nb::gil_scoped_acquire gil;
                         try {
                             nb::object ret = invoke_py_tile_cb(
                                 *step_cb, valid_prefix_len, tok_ptr, pos_ptr, mask_ptr, T
                             );
                             nb::module_ np_mod = nb::module_::import_("numpy");
                             nb::object arr_obj = np_mod.attr("ascontiguousarray")(
                                 ret, np_mod.attr("float32")
                             );
                             auto out_arr = nb::cast<
                                 nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                             >(arr_obj);
                             if (out_arr.shape(0) != T || out_arr.shape(1) != vocab_size) {
                                 throw std::invalid_argument(
                                     "target_step_fn return logits shape mismatch: expected [tile_size, vocab_size]"
                                 );
                             }
                             std::memcpy(out_logits, out_arr.data(), T * vocab_size * sizeof(float));
                         } catch (nb::python_error& e) {
                             throw std::runtime_error(e.what());
                         }
                     };
                 }

                 tpuforge::ForgeEngine::CommitKvFn cpp_kv;
                 if (kv_cb) {
                     cpp_kv = [kv_cb](
                         int32_t prev_valid_prefix_len,
                         const std::vector<int32_t>& accepted_row_indices
                     ) {
                         if (!is_python_active()) {
                             throw std::runtime_error("Python interpreter is finalizing");
                         }
                         nb::gil_scoped_acquire gil;
                         try {
                             (*kv_cb)(prev_valid_prefix_len, accepted_row_indices);
                         } catch (nb::python_error& e) {
                             throw std::runtime_error(e.what());
                         }
                     };
                 }

                 tpuforge::ForgeEngine::PrefillStepFn cpp_prefill;
                 if (prefill_cb) {
                     cpp_prefill = [prefill_cb, T](
                         int32_t valid_prefix_len,
                         const int32_t* tok_ptr,
                         const int32_t* pos_ptr,
                         const float* mask_ptr
                     ) {
                         if (!is_python_active()) {
                             throw std::runtime_error("Python interpreter is finalizing");
                         }
                         nb::gil_scoped_acquire gil;
                         try {
                             (void)invoke_py_tile_cb(
                                 *prefill_cb, valid_prefix_len, tok_ptr, pos_ptr, mask_ptr, T
                             );
                         } catch (nb::python_error& e) {
                             throw std::runtime_error(e.what());
                         }
                     };
                 }

                 tpuforge::ForgeEngine::GreedyStepFn cpp_greedy;
                 if (greedy_cb) {
                     cpp_greedy = [greedy_cb, T](
                         int32_t valid_prefix_len,
                         const int32_t* tok_ptr,
                         const int32_t* pos_ptr,
                         const float* mask_ptr,
                         int32_t* out_argmax_tokens
                     ) {
                         if (!is_python_active()) {
                             throw std::runtime_error("Python interpreter is finalizing");
                         }
                         nb::gil_scoped_acquire gil;
                         try {
                             nb::object ret = invoke_py_tile_cb(
                                 *greedy_cb, valid_prefix_len, tok_ptr, pos_ptr, mask_ptr, T
                             );
                             using Int32Tile = nb::ndarray<
                                 const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu
                             >;
                             Int32Tile fast_arr;
                             if (nb::try_cast<Int32Tile>(ret, fast_arr)) {
                                 if (fast_arr.shape(0) != T) {
                                     throw std::invalid_argument(
                                         "greedy_step_fn return token shape mismatch: expected [tile_size]"
                                     );
                                 }
                                 std::memcpy(out_argmax_tokens, fast_arr.data(), T * sizeof(int32_t));
                             } else {
                                 nb::module_ np_mod = nb::module_::import_("numpy");
                                 nb::object arr_obj = np_mod.attr("ascontiguousarray")(
                                     ret, np_mod.attr("int32")
                                 );
                                 auto out_arr = nb::cast<Int32Tile>(arr_obj);
                                 if (out_arr.shape(0) != T) {
                                     throw std::invalid_argument(
                                         "greedy_step_fn return token shape mismatch: expected [tile_size]"
                                     );
                                 }
                                 std::memcpy(out_argmax_tokens, out_arr.data(), T * sizeof(int32_t));
                             }
                         } catch (nb::python_error& e) {
                             throw std::runtime_error(e.what());
                         }
                     };
                 }

                 new (self) tpuforge::ForgeEngine(
                     config,
                     std::move(cpp_step),
                     std::move(cpp_kv),
                     std::move(model_id),
                     std::move(cpp_prefill),
                     std::move(cpp_greedy)
                 );
             },
             nb::arg("config"),
             nb::arg("target_step_fn") = nb::none(),
             nb::arg("commit_kv_fn") = nb::none(),
             nb::arg("model_id") = "",
             nb::arg("prefill_step_fn") = nb::none(),
             nb::arg("greedy_step_fn") = nb::none())
        .def("generate",
             [](tpuforge::ForgeEngine& eng, const tpuforge::GenerationRequest& req) {
                 tpuforge::GenerationRequest req_copy = req;
                 nb::gil_scoped_release nogil;
                 return eng.generate(req_copy, nullptr);
             },
             nb::arg("request"))
        .def("model_id", &tpuforge::ForgeEngine::model_id)
        .def("config", &tpuforge::ForgeEngine::config)
        .def("tokenizer", &tpuforge::ForgeEngine::tokenizer)
        .def("compile_count", &tpuforge::ForgeEngine::compile_count);

    nb::class_<tpuforge::OpenAIServer::Config>(m, "OpenAIServerConfig")
        .def(nb::init<>())
        .def_rw("host", &tpuforge::OpenAIServer::Config::host)
        .def_rw("port", &tpuforge::OpenAIServer::Config::port)
        .def_rw("worker_threads", &tpuforge::OpenAIServer::Config::worker_threads)
        .def_rw("max_queued_connections", &tpuforge::OpenAIServer::Config::max_queued_connections)
        .def_rw("socket_timeout_ms", &tpuforge::OpenAIServer::Config::socket_timeout_ms);

    nb::class_<PyOpenAIServer>(m, "OpenAIServer")
        .def(nb::init<tpuforge::ForgeEngine&, const tpuforge::OpenAIServer::Config&>(),
             nb::arg("engine"), nb::arg("config"), nb::keep_alive<1, 2>())
        .def(nb::init<tpuforge::ForgeEngine&>(),
             nb::arg("engine"), nb::keep_alive<1, 2>())
        .def("start", [](PyOpenAIServer& srv) {
            std::string err;
            bool ok = false;
            {
                nb::gil_scoped_release nogil;
                ok = srv.server->start(&err);
            }
            return nb::make_tuple(ok, err);
        })
        .def("stop", [](PyOpenAIServer& srv) {
            nb::gil_scoped_release nogil;
            srv.server->stop();
        })
        .def("is_running", [](const PyOpenAIServer& srv) {
            nb::gil_scoped_release nogil;
            return srv.server->is_running();
        })
        .def("bound_port", [](const PyOpenAIServer& srv) {
            nb::gil_scoped_release nogil;
            return srv.server->bound_port();
        })
        .def("__enter__", [](PyOpenAIServer& srv) -> PyOpenAIServer& {
            std::string err;
            bool ok = false;
            {
                nb::gil_scoped_release nogil;
                ok = srv.server->start(&err);
            }
            if (!ok) {
                throw std::runtime_error("OpenAIServer.start() failed: " + err);
            }
            return srv;
        }, nb::rv_policy::reference_internal)
        .def("__exit__", [](PyOpenAIServer& srv, const nb::args&) {
            nb::gil_scoped_release nogil;
            srv.server->stop();
        });
}
