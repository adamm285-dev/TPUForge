#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tpuforge/device.h"
#include "tpuforge/openai_server.h"
#include "tpuforge/pjrt_driver.h"
#include "tpuforge/static_arena.h"
#include "tpuforge/tree_speculator.h"

#include <algorithm>
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

    nb::class_<tpuforge::DeviceInfo>(m, "DeviceInfo")
        .def_ro("id", &tpuforge::DeviceInfo::id)
        .def_ro("kind", &tpuforge::DeviceInfo::kind)
        .def_ro("memory_bytes", &tpuforge::DeviceInfo::memory_bytes)
        .def_ro("is_tpu", &tpuforge::DeviceInfo::is_tpu)
        .def_ro("mxu_dim_x", &tpuforge::DeviceInfo::mxu_dim_x)
        .def_ro("mxu_dim_y", &tpuforge::DeviceInfo::mxu_dim_y);

    nb::class_<tpuforge::TreeNode>(m, "TreeNode")
        .def_ro("token_id", &tpuforge::TreeNode::token_id)
        .def_ro("parent_idx", &tpuforge::TreeNode::parent_idx)
        .def_ro("depth", &tpuforge::TreeNode::depth)
        .def_ro("draft_prob", &tpuforge::TreeNode::draft_prob)
        .def_ro("cumulative_logprob", &tpuforge::TreeNode::cumulative_logprob)
        .def_ro("children", &tpuforge::TreeNode::children);

    nb::class_<tpuforge::SystolicTree>(m, "SystolicTree")
        .def(nb::init<>())
        .def(nb::init<int32_t>(), nb::arg("root_last_token"))
        .def("reset", &tpuforge::SystolicTree::reset, nb::arg("root_last_token"))
        .def("add_child", &tpuforge::SystolicTree::add_child,
             nb::arg("parent_idx"), nb::arg("token_id"), nb::arg("draft_prob") = 1.0f)
        .def("num_nodes", &tpuforge::SystolicTree::num_nodes)
        .def("nodes", &tpuforge::SystolicTree::nodes)
        .def("pack", [](const tpuforge::SystolicTree& tree, int32_t valid_prefix_len) {
            auto packed = tree.pack(valid_prefix_len);
            nb::dict d;
            d["active_nodes"] = packed.active_nodes;
            d["token_ids"] = make_1d_numpy<int32_t>(
                packed.token_ids.data(), tpuforge::SystolicConstants::kSystolicTileSize);
            d["position_offsets"] = make_1d_numpy<int32_t>(
                packed.position_offsets.data(), tpuforge::SystolicConstants::kSystolicTileSize);
            d["mask_u8"] = make_2d_numpy<uint8_t>(
                packed.mask_u8.data(),
                tpuforge::SystolicConstants::kSystolicTileSize,
                tpuforge::SystolicConstants::kSystolicTileSize);
            d["additive_mask_f32"] = make_2d_numpy<float>(
                packed.additive_mask_f32.data(),
                tpuforge::SystolicConstants::kSystolicTileSize,
                tpuforge::SystolicConstants::kSystolicTileSize);
            return d;
        }, nb::arg("valid_prefix_len") = 0);

    nb::class_<tpuforge::SuffixTreeDrafter>(m, "SuffixTreeDrafter")
        .def_static("build_tree", &tpuforge::SuffixTreeDrafter::build_tree,
                    nb::arg("context_tokens"),
                    nb::arg("max_nodes") = tpuforge::SystolicTree::kMaxNodes,
                    nb::arg("max_depth") = 16);

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
                if (logits.shape(0) != tpuforge::SystolicConstants::kSystolicTileSize) {
                    throw std::invalid_argument("Expected logits shape [128, vocab_size]");
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
        .def_static("verify_stochastic",
            [](const tpuforge::SystolicTree& tree,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits,
               float temperature,
               uint64_t seed) {
                if (logits.shape(0) != tpuforge::SystolicConstants::kSystolicTileSize) {
                    throw std::invalid_argument("Expected logits shape [128, vocab_size]");
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
        .def_rw("vocab_size", &tpuforge::StaticArena::Config::vocab_size);

    nb::class_<tpuforge::StaticArena>(m, "StaticArena")
        .def(nb::init<const tpuforge::StaticArena::Config&>(), nb::arg("config"))
        .def("is_aligned_128", &tpuforge::StaticArena::is_aligned_128)
        .def("valid_prefix_len", &tpuforge::StaticArena::valid_prefix_len)
        .def("set_valid_prefix_len", &tpuforge::StaticArena::set_valid_prefix_len, nb::arg("len"))
        .def("kv_row_elems", &tpuforge::StaticArena::kv_row_elems)
        .def("set_tile_inputs",
            [](tpuforge::StaticArena& arena,
               nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> token_ids,
               nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> pos_offsets,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> additive_mask) {
                if (!arena.token_tile_128() || !arena.position_tile_128() ||
                    !arena.causal_mask_128x128()) {
                    throw std::logic_error("StaticArena has been moved from");
                }
                constexpr size_t T = tpuforge::SystolicConstants::kSystolicTileSize;
                if (token_ids.shape(0) != T || pos_offsets.shape(0) != T ||
                    additive_mask.shape(0) != T || additive_mask.shape(1) != T) {
                    throw std::invalid_argument("Expected [128] token/pos and [128, 128] mask");
                }
                std::memcpy(arena.token_tile_128(), token_ids.data(), T * sizeof(int32_t));
                std::memcpy(arena.position_tile_128(), pos_offsets.data(), T * sizeof(int32_t));
                std::memcpy(arena.causal_mask_128x128(), additive_mask.data(), T * T * sizeof(float));
            },
            nb::arg("token_ids").noconvert(),
            nb::arg("position_offsets").noconvert(),
            nb::arg("additive_mask").noconvert())
        .def("commit_accepted_kv_rows",
            [](tpuforge::StaticArena& arena,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> step_k_tile,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> step_v_tile,
               const std::vector<int32_t>& accepted_row_indices) {
                constexpr size_t T = tpuforge::SystolicConstants::kSystolicTileSize;
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
                if (!arena.logits_tile_128()) {
                    throw std::logic_error("StaticArena has been moved from");
                }
                return make_2d_numpy<float>(
                    arena.logits_tile_128(),
                    tpuforge::SystolicConstants::kSystolicTileSize,
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
            nb::arg("arena"), nb::arg("tile_rows") = tpuforge::SystolicConstants::kSystolicTileSize)
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
                nb::callable target_step_fn,
                nb::object commit_kv_obj,
                std::string model_id) {
                 auto py_deleter = [](nb::callable* p) noexcept {
                     if (!p) {
                         return;
                     }
                     if (is_python_active()) {
                         nb::gil_scoped_acquire gil;
                         delete p;
                     }
                 };

                 auto step_cb = std::shared_ptr<nb::callable>(
                     new nb::callable(std::move(target_step_fn)), py_deleter
                 );

                 std::shared_ptr<nb::callable> kv_cb;
                 if (!commit_kv_obj.is_none()) {
                     kv_cb = std::shared_ptr<nb::callable>(
                         new nb::callable(nb::cast<nb::callable>(commit_kv_obj)), py_deleter
                     );
                 }

                 const size_t vocab_size = config.vocab_size;
                 tpuforge::ForgeEngine::TargetStepFn cpp_step =
                     [step_cb, vocab_size](
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
                             constexpr size_t T = tpuforge::SystolicConstants::kSystolicTileSize;
                             auto tok_arr = make_1d_numpy<int32_t>(tok_ptr, T);
                             auto pos_arr = make_1d_numpy<int32_t>(pos_ptr, T);
                             auto mask_arr = make_2d_numpy<float>(mask_ptr, T, T);

                             nb::object ret = (*step_cb)(valid_prefix_len, tok_arr, pos_arr, mask_arr);
                             nb::module_ np_mod = nb::module_::import_("numpy");
                             nb::object arr_obj = np_mod.attr("ascontiguousarray")(
                                 ret, np_mod.attr("float32")
                             );
                             auto out_arr = nb::cast<
                                 nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>
                             >(arr_obj);
                             if (out_arr.shape(0) != T || out_arr.shape(1) != vocab_size) {
                                 throw std::invalid_argument(
                                     "target_step_fn return logits shape mismatch: expected [128, vocab_size]"
                                 );
                             }
                             std::memcpy(out_logits, out_arr.data(), T * vocab_size * sizeof(float));
                         } catch (nb::python_error& e) {
                             throw std::runtime_error(e.what());
                         }
                     };

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

                 new (self) tpuforge::ForgeEngine(
                     config, std::move(cpp_step), std::move(cpp_kv), std::move(model_id)
                 );
             },
             nb::arg("config"),
             nb::arg("target_step_fn"),
             nb::arg("commit_kv_fn") = nb::none(),
             nb::arg("model_id") = "tpuforge-custom-128")
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
