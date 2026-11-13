#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tpuforge/device.h"
#include "tpuforge/pjrt_driver.h"
#include "tpuforge/static_arena.h"
#include "tpuforge/tree_speculator.h"

#include <cstring>
#include <stdexcept>

namespace nb = nanobind;

namespace {

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> make_1d_numpy(const T* src, size_t n) {
    T* buf = new T[n];
    std::memcpy(buf, src, n * sizeof(T));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(buf, 1, shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> make_2d_numpy(const T* src, size_t rows, size_t cols) {
    T* buf = new T[rows * cols];
    std::memcpy(buf, src, rows * cols * sizeof(T));
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    size_t shape[2] = {rows, cols};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(buf, 2, shape, owner);
}

} // namespace

NB_MODULE(_tpuforge_ext, m) {
    m.doc() = "TPUForge native C++20 systolic tree and static PJRT runtime bindings";

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
                return tpuforge::SystolicTreeVerifier::verify_greedy(
                    tree, logits.data(), logits.shape(1));
            },
            nb::arg("tree"), nb::arg("target_logits_128"))
        .def_static("verify_stochastic",
            [](const tpuforge::SystolicTree& tree,
               nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> logits,
               float temperature,
               uint64_t seed) {
                if (logits.shape(0) != tpuforge::SystolicConstants::kSystolicTileSize) {
                    throw std::invalid_argument("Expected logits shape [128, vocab_size]");
                }
                return tpuforge::SystolicTreeVerifier::verify_stochastic(
                    tree, logits.data(), logits.shape(1), temperature, seed);
            },
            nb::arg("tree"), nb::arg("target_logits_128"),
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
                constexpr size_t T = tpuforge::SystolicConstants::kSystolicTileSize;
                if (token_ids.shape(0) != T || pos_offsets.shape(0) != T ||
                    additive_mask.shape(0) != T || additive_mask.shape(1) != T) {
                    throw std::invalid_argument("Expected [128] token/pos and [128, 128] mask");
                }
                std::memcpy(arena.token_tile_128(), token_ids.data(), T * sizeof(int32_t));
                std::memcpy(arena.position_tile_128(), pos_offsets.data(), T * sizeof(int32_t));
                std::memcpy(arena.causal_mask_128x128(), additive_mask.data(), T * T * sizeof(float));
            },
            nb::arg("token_ids"), nb::arg("position_offsets"), nb::arg("additive_mask"))
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
            nb::arg("step_k_tile"), nb::arg("step_v_tile"), nb::arg("accepted_row_indices"))
        .def("get_kv_prefix_k_slice",
            [](const tpuforge::StaticArena& arena, size_t start_row, size_t num_rows) {
                if (start_row + num_rows > arena.config().max_seq_len) {
                    throw std::out_of_range("KV slice exceeds max_seq_len");
                }
                const size_t cols = arena.kv_row_elems();
                return make_2d_numpy<float>(
                    arena.kv_prefix_k() + (start_row * cols), num_rows, cols);
            },
            nb::arg("start_row"), nb::arg("num_rows"))
        .def("get_logits_tile",
            [](const tpuforge::StaticArena& arena) {
                return make_2d_numpy<float>(
                    arena.logits_tile_128(),
                    tpuforge::SystolicConstants::kSystolicTileSize,
                    arena.config().vocab_size);
            });

    nb::class_<tpuforge::PjrtDriver>(m, "PjrtDriver")
        .def(nb::init<>())
        .def("load_plugin", [](tpuforge::PjrtDriver& drv, const std::string& path) {
            std::string err;
            bool ok = drv.load_plugin(path, &err);
            return nb::make_tuple(ok, err);
        }, nb::arg("path"))
        .def("unload", &tpuforge::PjrtDriver::unload)
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
                bool ok = exec.compile_once(config, program_code, program_format, &err);
                return nb::make_tuple(ok, err);
            },
            nb::arg("config"), nb::arg("program_code"), nb::arg("program_format") = "stablehlo")
        .def("execute_step",
            [](tpuforge::ZeroRecompileExecutor& exec,
               tpuforge::StaticArena& arena,
               size_t tile_rows) {
                std::string err;
                bool ok = exec.execute_step(arena, tile_rows, &err);
                return nb::make_tuple(ok, err);
            },
            nb::arg("arena"), nb::arg("tile_rows") = tpuforge::SystolicConstants::kSystolicTileSize)
        .def("compile_count", &tpuforge::ZeroRecompileExecutor::compile_count)
        .def("execute_count", &tpuforge::ZeroRecompileExecutor::execute_count)
        .def("is_compiled", &tpuforge::ZeroRecompileExecutor::is_compiled);
}
