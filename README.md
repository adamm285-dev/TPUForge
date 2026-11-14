# TPUForge

<p align="center">
  <b>Native C++20 &amp; OpenXLA Inference Engine for Google Cloud TPUs</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-34A853" alt="Apache 2.0 License"></a>
  <a href="src/"><img src="https://img.shields.io/badge/Engine-C++20%20%2F%20PJRT%20C%20ABI-4285F4" alt="C++20 and PJRT C ABI"></a>
  <a href="benchmarks/"><img src="https://img.shields.io/badge/Kernels-OpenXLA%20%2F%20JAX-FBBC04" alt="OpenXLA and JAX Kernels"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Offline%20Tests-48%20Passing-EA4335" alt="48 Offline Tests Passing"></a>
</p>

<!-- SLIDE_01: assets/slides/slide_01.png -->

Most LLM serving stacks were designed for GPU warps, asynchronous CUDA streams, and dynamic shape graphs. On Google Cloud TPUs, running a standard batch-1 token decode step leaves `127` of `128` rows idle on a `v5e` systolic Matrix Multiply Unit (`255` of `256` on `v6e` Trillium), while varying prompt lengths trigger multi-second XLA recompilations.

**TPUForge** aligns speculative verification, static memory allocation, and inter-chip communication directly to TPU hardware geometry. Evaluated live on Cloud TPU `v5e-1` (`15.75 GiB` HBM, `JAX 0.11.2`, `bfloat16`), TPUForge reaches **`202.80 tok/s` decode (`4.12x` decode, `7.50x` end-to-end)** on `Qwen/Qwen2.5-Coder-7B-Instruct` with `64/64` exact greedy token agreement and **`328.79 tok/s` decode (`3.47x` decode, `5.68x` end-to-end)** on `Qwen/Qwen2.5-Coder-3B-Instruct` with `128/128` exact agreement.

---

## Quickstart (Verify in 30 Seconds)

| Mode | Needs Cloud TPU? | Command | What You Verify |
| :--- | :---: | :--- | :--- |
| **1. Build C++20 runtime & CLI** | **No** | `cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/tpuforge_bench --tile-size 128` | Native `pjrt_c_api.h` driver, `StaticArena`, and `128`/`256`-row systolic tree packing. |
| **2. Run offline verification suite** | **No** | `pytest -q` | All `48` offline C++/nanobind, PJRT arena, XLA sharding, `INT8`/`FP8`, MoE streaming, and AST-seeding checks. |
| **3. Run OpenXLA CPU parity bench** | **No** | `python -m benchmarks.run_qwen_bench --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tile-size auto --tp-size auto` | `1`-compile static arena (`M-002`), `128`-node tree parity (`M-003`), and exact match vs Vanilla JAX. |
| **4. Run on Cloud TPU (`v5e` / `v6e`)** | **Yes** | Open [`colab/TPUForge_Benchmark.ipynb`](colab/TPUForge_Benchmark.ipynb) or standalone [`benchmarks/results/`](benchmarks/results/) | Multi-chip Ring-Attention (`128k`), `Qwen` `3B`/`7B`/`32B`, `Qwen3-4B` block-tree, `14B` `INT8`, and `OLMoE` `mmap`. |

```bash
# Local native C++20 build + test suite (Clang/GCC/MSVC + CMake + Ninja)
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tpuforge_bench --tile-size 128
./build/tpuforge_bench --tile-size 256
pytest -q
```

---

## Contents

- [1. Verified Cloud TPU Results (`v5e-1`)](#1-verified-cloud-tpu-results-v5e-1)
- [2. Honest Cold-Prompt Accounting and Phase 2 Extensions](#2-honest-cold-prompt-accounting-and-phase-2-extensions)
- [3. Architecture and Subsystem Design](#3-architecture-and-subsystem-design)
- [4. Command Reference (Phase 1 and Phase 2)](#4-command-reference-phase-1-and-phase-2)
- [5. Hardware Matrix and Repository Layout](#5-hardware-matrix-and-repository-layout)
- [6. License](#6-license)

---

## 1. Verified Cloud TPU Results (`v5e-1`)

<!-- SLIDE_02: assets/slides/slide_02.png -->
<!-- SLIDE_03: assets/slides/slide_03.png -->
<!-- SLIDE_04: assets/slides/slide_04.png -->

Every number in this section comes directly from live Cloud TPU `v5e-1` runs (`TPU v5 lite`, `15.75 GiB` HBM, `JAX 0.11.2`, `bfloat16`) with on-device `jnp.argmax` (`verify_greedy_tokens`) and zero-constant-capture `_weights_tuple` lowering. Every speculative run matched Vanilla JAX greedy output **token for token (`100%`)**.

### Autoregressive Decode & End-to-End Speedup

| Model | Workload | Vanilla JAX (tok/s) | Static Arena (tok/s) | TPUForge Tree (tok/s) | Tokens / Pass | Decode Speedup | Total Speedup | Exact Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen2.5-Coder-7B-Instruct`** (`tile=128`, `gen=64`) | Code Refactor | `49.25` | `45.82` | **`202.80`** | `4.57` | **`4.12x`** | **`7.50x`** (`0.358s` vs `2.685s`) | `64 / 64` |
| **`Qwen2.5-Coder-7B-Instruct`** (`tile=128`, `gen=64`) | Cold Codegen | `49.30` | | **`62.38`** | | **`1.27x`** | **`1.84x`** | `64 / 64` |
| **`Qwen2.5-Coder-3B-Instruct`** (`tile=128`, `gen=128`) | Code Refactor | `94.88` | | **`328.79`** | `3.88` | **`3.47x`** | **`5.68x`** (`0.411s` vs `2.334s`) | `128 / 128` |
| **`Qwen2.5-Coder-3B-Instruct`** (`tile=128`, `gen=128`) | Cold Codegen | `95.13` | | **`119.80`** | | **`1.26x`** | **`1.79x`** | `128 / 128` |

### Zero-Recompilation Prompt Sweep (`[32, 64, 95, 195]` Unaligned Tokens)

| Model (`tile=128`, `TP=1`) | Vanilla Compiles | TPUForge Compiles | Full Sweep Time | Post-Warmup (`64..195`) | Post-Warmup Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen2.5-Coder-3B-Instruct`** | `5` | **`1`** | `1.215s` vs `3.342s` (`2.75x`) | `0.180s` vs `2.273s` | **`12.60x`** |
| **`Qwen2.5-Coder-7B-Instruct`** | `5` | **`1`** | `1.633s` vs `3.869s` (`2.37x`) | `0.347s` vs `2.679s` | **`7.72x`** |

### On-Device Argmax, Hardware-Matched MXU Tiles, & Flash-Ring (`M-005`)

* **Removing the PCIe Logits Bottleneck (`M-003`):** Returning raw `[128, vocab]` logits over PCIe (`~78 MB` at `tile=128`, `~156 MB` at `tile=256`) capped throughput to `97.22 tok/s` (`3B`) and `115.27 tok/s` (`7B`). Moving `jnp.argmax` onto TPU HBM compressed an entire `128`-node verification pass to `11.80 ms` vs `10.62 ms` for a `1`-token Vanilla step on `3B` (`22.54 ms` vs `20.63 ms` on `7B`), driving a `3.38x` jump on `3B` and `1.76x` on `7B`.
* **Hardware-Matched Systolic Tile Width (`infer_mxu_dim`):** On `v5e-1` (`128x128` physical MXU), matching `tile=128` (`328.79 tok/s`, `11.80 ms/pass`) beats an oversized `tile=256` (`268.39 tok/s`, `14.45 ms/pass`) by **`1.23x`** on pure MXU execution while keeping `256` ready for `v6e` Trillium / `v7`.
* **Blockwise Flash-Ring Attention (`M-005`, `ring=1`, `TP=1`, BF16):** Achieved **`2,680,640 tok/s`** prefill (`1.337 ms`) and **`0.353 ms`** tree step (`362,752` tree-nodes/s, `3.91e-03` prefill / `1.22e-04` tree BF16 max abs error) at `3584 / 4096`, and **`519,735 tok/s`** prefill (`55.167 ms`) with **`1.032 ms`** tree step (`124,047` tree-nodes/s, `1.75 GiB` KV, `collective-permute` HLO verified) at `28672 / 32768`.

---

## 2. Honest Cold-Prompt Accounting and Phase 2 Extensions

<!-- SLIDE_05: assets/slides/slide_05.png -->
<!-- SLIDE_06: assets/slides/slide_06.png -->
<!-- SLIDE_07: assets/slides/slide_07.png -->

Pure suffix drafting (`SuffixIndex`) accelerates repetitive code edits (`4.12x` on `7B`), where matching patterns already exist in the prompt context. On zero-overlap cold prompts, acceptance drops and decode speedup narrows to **`1.27x` (`7B`)** and **`1.26x` (`3B`)**. Rather than tuning benchmarks around warm prompts, TPUForge Phase 2 implements four architectural extensions to close cold-start drafting, single-chip capacity, and sparse MoE offloads:

| Extension | Implemented Mechanism | Offline Verification Status (`48/48` pytest) | Cloud TPU Status |
| :--- | :--- | :--- | :--- |
| **`M-006`: Hybrid Block-Diffusion Tree** (`tpuforge_block_tree`) | `benchmarks/block_diffusion_jax.py` (`QwenBlockTreeRunner`, `TreeBuilder::build_from_block_top_k`) fuses parallel block-drafter `top-k` predictions (compatible with HF `dflash_config` checkpoints) with `O(1)` `SuffixIndex` continuations (`Qwen3` `q/k_norm` + `head_dim`). | Verified (`test_block_tree_and_suffix.py`): sub-`15 ms` tree packing at `32K+`, cold parity & `TP=2` HLO green. | Staged on TPU `v5e` |
| **`M-009`: Out-of-Band Repository AST Seeding** | `benchmarks/ast_seed_jax.py`, `SuffixIndex::seed_sequences`, `ForgeEngine::seed_ast_text`, `--ast-seed-path`, and HTTP `"seed_snippets"` pre-warm cold codegen drafting with zero prompt or KV-cache token cost. | Verified (`test_ast_seed_drafter.py`): Python/C++ AST extraction, exact greedy parity, 1-token fallback. | Staged on TPU `v5e` |
| **`M-007`: Native `INT8` / `FP8` Systolic Kernels** | `benchmarks/quantization_jax.py` runs symmetric per-channel `int8` / `float8_e4m3fn` mixed-precision `jax.lax.dot_general` (`BF16` activations x `INT8`/`FP8` weights) inside constant-free static arenas. | Verified (`test_quantized_systolic.py`): `< 0.65x` linear HBM footprint vs `BF16`, `TP=2` parity & `all-reduce` HLO. | Staged on TPU `v5e` |
| **`M-008`: Host `mmap` / DRAM Sparse MoE Streaming** | `benchmarks/moe_streaming_jax.py` pins attention, routers, and static KV caches in HBM while streaming active experts (`OLMoE`, `Qwen3-MoE`) into fixed `E_stage` buffers amortized across `128`-node trees. | Verified (`test_moe_streaming.py`): `< 45%` HBM isolation, `0` steady-state recompiles, `>= 3.0` tok/pass amortization. | Staged on TPU `v5e` |

*(No simulated Cloud TPU numbers are reported anywhere for Phase 2 until live `v5e-8` / `v5e-1` runs complete.)*

---

## 3. Architecture and Subsystem Design

<!-- SLIDE_08: assets/slides/slide_08.png -->

```
        +-----------------------------------------------------------------+
        |   Client / OpenAI /v1/chat/completions (`seed_snippets` + SSE)  |
        +-----------------------------------------------------------------+
                                         |
                                         v
        +-----------------------------------------------------------------+
        |                 TPUForge Native C++20 Engine                    |
        |   `ForgeEngine` | `O(1)` `SuffixIndex` + Out-of-Band AST Pack   |
        |       Hybrid Block-Diffusion `TreeBuilder` (128 / 256 tile)     |
        +-----------------------------------------------------------------+
                     |                   |                    |
        +------------+                   |                    +------------+
        v                                v                                 v
+-----------------------+   +-------------------------+   +------------------------+
| 128/256-Row Systolic  |   | Zero-Recompile Static   |   | Host mmap/DRAM Sparse  |
| Tree Verifier &       |   | HBM Arena & Optical     |   | MoE Expert Stager      |
| INT8/FP8 MXU Kernels  |   | Torus ICI Ring-Attn     |   | (E_stage -> 128 nodes) |
+-----------------------+   +-------------------------+   +------------------------+
        |                                |                                 |
        +--------------------------------+---------------------------------+
                                         |
                                         v
        +-----------------------------------------------------------------+
        |          OpenXLA / PJRT C ABI (`pjrt_c_api.h` Driver)           |
        +-----------------------------------------------------------------+
                         |                               |
                         v                               v
             [Local PJRT CPU Plugin]         [Cloud TPU PJRT Plugin]
           (Development / Simulation)      (`libpjrt_tpu.so` / Silicon)
```

1. **Serving Ingress (`M-004`):** Native C++20 HTTP/1.1 `/v1/chat/completions` server (`OpenAIServer`) with `ByteTokenizer`, `Utf8StreamDecoder`, SSE streaming, `nanobind` `_tpuforge_ext` Python bindings, and out-of-band `"seed_snippets"` support.
2. **`O(1)` `SuffixIndex` & Hybrid Block-Diffusion Packing (`M-003`, `M-006`, `M-009`):** Incremental 1-to-4-gram suffix hash index merges out-of-band repository AST declarations and parallel block-drafter `top-k` candidates into `128`- or `256`-row fixed causal tiles via `TreeBuilder::build_from_block_top_k`.
3. **Mixed-Precision & Zero-Recompile Kernels (`M-002`, `M-007`):** `StaticArena` (`128`-byte alignment) and constant-free `TargetWeightsTuple` / `DraftWeightsTuple` eliminate dynamic-shape stalls and support `bfloat16` activations against `int8` or `float8_e4m3fn` per-channel weights.
4. **Host `mmap` Sparse MoE Expert Streaming (`M-008`):** `ExpertHostStore` streams active routed experts (`2D` `OLMoE` or `3D` stacked `Qwen3-MoE`) into fixed-shape `E_stage` HBM buffers so one PCIe transfer serves all `128` speculative candidates simultaneously.
5. **Optical Torus Ring-Attention & GSPMD Tensor Parallelism (`M-005`):** Native `IciRingPlanner` + `jax.shard_map` + `jax.lax.ppermute` ring kernels (`128K` multi-chip target) alongside column/row `P(..., "tp")` sharding (`TP=4` / `TP=8`) and persistent OpenXLA compilation caching (`TPUFORGE_XLA_CACHE_DIR`).
6. **PJRT C ABI Backend (`pjrt_c_api.h`):** Dispatches fixed-shape compiled programs directly across local `libpjrt_cpu` and Cloud TPU `libpjrt_tpu.so`.

---

## 4. Command Reference (Phase 1 and Phase 2)

```bash
# Baseline Vanilla JAX vs StaticArena (M-002) vs TPUForge Tree (M-003) on Qwen2.5 or Qwen3
python -m benchmarks.run_qwen_bench --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tile-size auto --tp-size auto

# Hybrid Block-Diffusion + O(1) Suffix Systolic Tree (tpuforge_block_tree, compatible with Z Lab DFlash HF checkpoints)
python -m benchmarks.run_qwen_bench --model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 --tile-size auto --tp-size auto

# Native INT8 Per-Channel Systolic Quantization + Out-of-Band Repository AST Seeding (M-007 + M-009)
python -m benchmarks.run_qwen_bench --model Qwen/Qwen2.5-Coder-14B-Instruct --weight-quant int8 --ast-seed-path . --tile-size auto --tp-size auto

# Host-DRAM / mmap Sparse MoE Expert Streaming + 128-Node Systolic Tree Amortization (M-008)
python -m benchmarks.moe_streaming_jax --model allenai/OLMoE-1B-7B-0924-Instruct --host-store mmap --stage-experts 16 --weight-quant int8

# M-005 Optical Torus Ring-Attention (ppermute prefill + stationary-KV systolic tree via IciRingPlanner)
python -m benchmarks.ring_attention_jax --max-seq-len 8192 --valid-seq-len 6144 --ring-size auto --tp-size 1

# Build standalone source zip for Kaggle / Colab TPU execution (includes persistent XLA cache + mmap reader)
python -m benchmarks.package_kaggle_bundle

# Execute native C++ CLI against a live TPU PJRT plugin
./build/tpuforge_bench --plugin /path/to/libpjrt_tpu.so --program model.mlir --tile-size 256
```

---

## 5. Hardware Matrix and Repository Layout

| Hardware Target | Status | Scope |
| :--- | :--- | :--- |
| **Local CPU (PJRT CPU)** | Verified (`48/48` pytest) | `128`/`256` tile simulation, `O(1)` `SuffixIndex` + AST seeding, block-tree packing, `INT8`/`FP8` systolic kernels, sparse MoE `mmap`/DRAM streaming, OpenXLA GSPMD TP parity |
| **Cloud TPU `v5e-1` (`15.75 GiB` HBM)** | Verified Silicon (Phase 1) / Staged (Phase 2) | `32K` Flash-Ring (`519,735 tok/s`), `3B` (`328.79 tok/s`), `7B` (`202.80 tok/s`), `tile=128` vs `256` (`1.23x`); Phase 2 (`Qwen3-4B`, `14B INT8`, `OLMoE mmap`) staged |
| **Kaggle Free TPU `v5e-8` (`128 GiB` HBM)** | Queued / Staged | `8x16 GiB` pooled HBM: `128K` ICI Ring-Attention (`M-005`), `7B TP=4`, `32B TP=8`, Phase 2 suite |
| **Cloud TPU `v6e` (Trillium) / `v7` (`8x`)** | Ready | Auto-inferred `256x256` MXU tile width via `tpu.infer_mxu_dim(device_kind)` |

* `include/`: Public C/C++ engine headers (`pjrt_c_api.h`, `kv_cache.h`, `tree_speculator.h`, `openai_server.h`, `engine.h`).
* `src/`: Core C++20 implementation (`core/`, `ops/`, `speculative/`, `runtime/`).
* `benchmarks/`: OpenXLA/JAX benchmark and kernel modules (`qwen_jax.py`, `quantization_jax.py`, `moe_streaming_jax.py`, `ast_seed_jax.py`, `block_diffusion_jax.py`, `ring_attention_jax.py`, `run_qwen_bench.py`, `package_kaggle_bundle.py`).
* `tests/`: `48` offline unit, C++/nanobind, XLA sharding, quantized systolic, sparse MoE streaming, AST-seeding, block-tree, and ring-attention tests (`pytest`).
* `docs/`: Architecture and TRC proposal notes ([`docs/TRC_APPLICATION.md`](docs/TRC_APPLICATION.md)).

---

## 6. License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for details.
