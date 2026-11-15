# TPUForge

<p align="center">
  <b>Native C++20 &amp; OpenXLA Inference Engine for Google Cloud TPUs (Gemma 4 + Qwen3)</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-34A853" alt="Apache 2.0 License"></a>
  <a href="src/"><img src="https://img.shields.io/badge/Engine-C++20%20%2F%20PJRT%20C%20ABI-4285F4" alt="C++20 and PJRT C ABI"></a>
  <a href="benchmarks/"><img src="https://img.shields.io/badge/Kernels-OpenXLA%20%2F%20JAX-FBBC04" alt="OpenXLA and JAX Kernels"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Offline%20Tests-55%20Passing-EA4335" alt="55 Offline Tests Passing"></a>
</p>

<!-- SLIDE_01: assets/slides/slide_01.png -->

Most LLM serving stacks were built around GPU warps, asynchronous CUDA streams, and dynamic-shape runtime graphs. On Google Cloud TPUs, standard batch-1 token decode leaves `127` of `128` rows idle on a `v5e` Matrix Multiply Unit (`255` of `256` on `v6e` Trillium and `v7`), while unaligned prompt shapes stall behind multi-second XLA recompilations.

**TPUForge** aligns static HBM allocation, `128`/`256`-row systolic tree verification, native `INT8`/`FP8` quantized `dot_general`, host-`mmap` MoE streaming, and optical torus Ring-Attention directly to TPU silicon geometry. Its primary model runtime natively targets late-2026 **Google Gemma 4** (`Apache 2.0`, ungated: `26B-A4B-it` MoE, `12B-it`, and `31B-it` with matching `DFlash` block-diffusion drafters), alongside a secondary `Qwen3-4B` cross-architecture check and verified live Cloud TPU `v5e-1` silicon baselines (`202.80 tok/s` / `4.12x` decode on `Qwen2.5-Coder-7B-Instruct`, `328.79 tok/s` / `3.47x` decode on `3B`).

---

## Quickstart (Verify in 30 Seconds)

| Mode | Needs Cloud TPU? | Command | What You Verify |
| :--- | :---: | :--- | :--- |
| **1. Build C++20 runtime & CLI** | **No** | `cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build && ./build/tpuforge_bench --tile-size 128` | Native `pjrt_c_api.h` driver, `StaticArena`, and `128`/`256`-row systolic tree packing. |
| **2. Run offline verification suite** | **No** | `pytest -q` | All `55` offline C++/nanobind, PJRT arena, Gemma 4 (`dense`/`MoE`/`DFlash`/`TP=2`), `INT8`/`FP8`, MoE `mmap`, and AST-seeding checks. |
| **3. Run OpenXLA CPU parity bench** | **No** | `python -m benchmarks.run_qwen_bench --model google/gemma-4-12B-it --draft-model z-lab/gemma4-12B-it-DFlash --weight-quant int8 --ast-seed-path .` | `1`-compile static arena (`M-002`), `128`-node hybrid DFlash + AST tree parity (`M-003`/`M-006`/`M-007`), and exact match vs Vanilla JAX. |
| **4. Run on Cloud TPU (`v5e` / `v6e`)** | **Yes** | Open [`colab/TPUForge_Benchmark.ipynb`](colab/TPUForge_Benchmark.ipynb) or standalone [`benchmarks/results/`](benchmarks/results/) | Multi-chip Ring-Attention (`128k`), Gemma 4 `12B`/`26B-A4B`/`31B`, `INT8` single-chip fit, `mmap` MoE streaming, and `Qwen3-4B` check. |

```bash
# Local native C++20 build + offline verification suite (Clang/GCC/MSVC + CMake + Ninja)
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/tpuforge_bench --tile-size 128
./build/tpuforge_bench --tile-size 256
pytest -q
```

---

## Contents

- [1. Target Lineup: Google Gemma 4 on Cloud TPUs](#1-target-lineup-google-gemma-4-on-cloud-tpus)
- [2. Verified Cloud TPU Silicon Baseline (`v5e-1`)](#2-verified-cloud-tpu-silicon-baseline-v5e-1)
- [3. Cold-Prompt Drafting, Quantization, & MoE Extensions (`M-006`..`M-008`)](#3-cold-prompt-drafting-quantization--moe-extensions-m-006m-008)
- [4. Architecture and Subsystem Design](#4-architecture-and-subsystem-design)
- [5. Command Reference and Hardware Matrix](#5-command-reference-and-hardware-matrix)
- [6. License](#6-license)

---

## 1. Target Lineup: Google Gemma 4 on Cloud TPUs

<!-- SLIDE_02: assets/slides/slide_02.png -->
<!-- SLIDE_03: assets/slides/slide_03.png -->

All flagship runs use ungated `Apache 2.0` Google Gemma 4 weights implemented natively in pure JAX/OpenXLA ([`benchmarks/gemma4_jax.py`](benchmarks/gemma4_jax.py)) without `transformers` or `torch` runtime overhead:

| Target Pair | Role & Architecture | Weight Footprint | Primary TPU Mapping & Hardware Physics | Cloud TPU Status |
| :--- | :--- | :--- | :--- | :--- |
| **`google/gemma-4-26B-A4B-it`** + `z-lab/gemma-4-26B-A4B-it-DFlash` | **Flagship Sparse MoE** (`128` experts, top-`8`, `4B` active) | `48.07 GiB` `BF16` -> **`25.08 GiB` total** at `INT8` (`0.80 GiB` DFlash) | Fits 100% inside `32 GiB` `v6e-1` HBM (`INT8`), runs `TP=8` on `v5e-8`, or streams via host `mmap` on `15.75 GiB` `v5e-1` (`3.81 GiB` HBM trunk + static KV). | Staged on `v5e-8` / `v6e-1` (Pending silicon JSON) |
| **`google/gemma-4-12B-it`** + `z-lab/gemma4-12B-it-DFlash` | **Single-Chip Dense** (`48` layers, `5` sliding : `1` full) | `22.28 GiB` `BF16` -> **`13.97 GiB` total** (`13.29` + `~0.68 GiB` DFlash `INT8`) | Fits together inside `15.75 GiB` `v5e-1` HBM via native `INT8` systolic `dot_general`; runs both `BF16` and `INT8` on `32 GiB` `v6e-1`. | Staged on `v5e` / `v6e` (Pending silicon JSON) |
| **`google/gemma-4-31B-it`** + `z-lab/gemma-4-31B-it-DFlash` | **Multi-Chip Flagship Dense** (`TP=8` Megatron GSPMD) | `58.25 GiB` `BF16` (`2.86 GiB` DFlash) -> **`9.80 GiB/chip` at `TP=8`** | Reuses tied `embed_tokens` directly as `lm_head` (`transpose_lm_head=True`), avoiding a duplicate `2.62 GiB` projection across `v5e-8`. | Staged on `v5e-8` (Pending silicon JSON) |
| **`Qwen/Qwen3-4B`** + `z-lab/Qwen3-4B-DFlash-b16` | **Secondary Cross-Arch Check** (`4B` dense) | `~7.5 GiB` `BF16` | Cross-architecture verification alongside `M-005` Optical Torus Ring-Attention. | Staged on `v5e` / `v6e` |

### Why Gemma 4 Amplifies TPUForge's Kernel Design

* **`262,144`-Vocabulary PCIe Avoidance:** With a `262,144`-token vocabulary, returning raw `[tile, vocab]` logits over PCIe transfers **`134 MB/step`** (`128` tile) or **`268 MB/step`** (`256` tile). Performing `jnp.argmax` directly on TPU HBM shrinks host-to-device synchronization to `512 B` (`tile=128`) or `1 KB` (`tile=256`).
* **Pure-JAX Periodic Scan & Mixed Attention Geometry:** Nested `jax.lax.scan` compiles Gemma 4's `5:1` `sliding_attention` / `full_attention` pattern (`attention_k_eq_v=True`, proportional RoPE with `partial_rotary_factor=0.25` on global layers, per-layer residual scalar, `30.0` softcapping, and `3D` stacked MoE banks with `router.scale` / `per_expert_scale`) down to a compact XLA HLO graph with automatic KV replication when `num_global_key_value_heads < tp_size`.
* **Explicit Architectural Scope Boundaries:** Edge variants with Per-Layer Embeddings or cross-layer KV sharing (`gemma-4-E2B` / `gemma-4-E4B`, where `hidden_size_per_layer_input != 0` or `num_kv_shared_layers != 0`) raise an explicit `ValueError` at config load. Hybrid recurrent/linear-attention models such as `Qwen 3.6` / `Qwen 3.8` (`linear_attention` DeltaNet state) are excluded because fixed-memory recurrent states cannot roll back zero-copy across `128`-node speculative tree branches.

---

## 2. Verified Cloud TPU Silicon Baseline (`v5e-1`)

<!-- SLIDE_04: assets/slides/slide_04.png -->
<!-- SLIDE_05: assets/slides/slide_05.png -->

All live silicon numbers currently reported here were measured on Google Cloud TPU `v5e-1` (`TPU v5 lite`, `15.75 GiB` HBM, `JAX 0.11.2`, `bfloat16`) using on-device `jnp.argmax` (`verify_greedy_tokens`) and zero-constant-capture `_weights_tuple` lowering on `Qwen2.5-Coder` (`M-002`..`M-005`). Every speculative run matched Vanilla JAX greedy output **token for token (`100%`)**. Live Gemma 4 Cloud TPU runs are staged and will populate a separate silicon table once executed.

### Autoregressive Decode & Zero-Recompilation Sweep (`Qwen2.5-Coder` `v5e-1` Baseline)

| Model (`v5e-1`, `BF16`) | Workload | Vanilla JAX (tok/s) | Static Arena (tok/s) | TPUForge Tree (tok/s) | Tokens / Pass | Decode Speedup | Total Speedup | Exact Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen2.5-Coder-7B-Instruct`** (`tile=128`, `gen=64`) | Code Refactor | `49.25` | `45.82` | **`202.80`** | `4.57` | **`4.12x`** | **`7.50x`** (`0.358s` vs `2.685s`) | `64 / 64` |
| **`Qwen2.5-Coder-7B-Instruct`** (`tile=128`, `gen=64`) | Cold Codegen | `49.30` | | **`62.38`** | | **`1.27x`** | **`1.84x`** | `64 / 64` |
| **`Qwen2.5-Coder-3B-Instruct`** (`tile=128`, `gen=128`) | Code Refactor | `94.88` | | **`328.79`** | `3.88` | **`3.47x`** | **`5.68x`** (`0.411s` vs `2.334s`) | `128 / 128` |
| **`Qwen2.5-Coder-3B-Instruct`** (`tile=128`, `gen=128`) | Cold Codegen | `95.13` | | **`119.80`** | | **`1.26x`** | **`1.79x`** | `128 / 128` |

| Prompt-Shape Sweep (`[32, 64, 95, 195]`, `tile=128`) | Vanilla Compiles | TPUForge Compiles | Full Sweep Time | Post-Warmup (`64..195`) | Post-Warmup Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Qwen2.5-Coder-3B-Instruct`** (`v5e-1`) | `5` | **`1`** | `1.215s` vs `3.342s` (`2.75x`) | `0.180s` vs `2.273s` | **`12.60x`** |
| **`Qwen2.5-Coder-7B-Instruct`** (`v5e-1`) | `5` | **`1`** | `1.633s` vs `3.869s` (`2.37x`) | `0.347s` vs `2.679s` | **`7.72x`** |

* **Eliminating Host Logits Stalls (`M-003`):** Moving `jnp.argmax` onto HBM removed `~78 MB` (`tile=128`) and `~156 MB` (`tile=256`) PCIe transfers on `Qwen2.5-Coder`, shrinking a full `128`-node verification pass to `11.80 ms` vs `10.62 ms` per `1`-token Vanilla step on `3B` (`22.54 ms` vs `20.63 ms` on `7B`). That gained `3.38x` over pre-argmax `97.22 tok/s` (`3B`) and `1.76x` over `115.27 tok/s` (`7B`).
* **Hardware-Matched MXU Tile Width (`infer_mxu_dim`):** On `v5e-1` (`128x128` physical MXU), `tile=128` (`328.79 tok/s`, `11.80 ms/pass`) beats `tile=256` (`268.39 tok/s`, `14.45 ms/pass`) by **`1.23x`** while `tile=256` targets `256x256` MXUs on `v6e` Trillium and `v7`.
* **Blockwise Flash-Ring Attention (`M-005`, `ring=1`, `TP=1`, `BF16`):** Reached **`2,680,640 tok/s`** prefill (`1.337 ms`) and **`0.353 ms`** tree step (`362,752` tree-nodes/s; `3.91e-03` prefill / `1.22e-04` tree `BF16` max abs error) at `3584 / 4096`, plus **`519,735 tok/s`** prefill (`55.167 ms`) and **`1.032 ms`** tree step (`124,047` tree-nodes/s, `1.75 GiB` KV, `collective-permute` HLO verified) at `28672 / 32768`.

---

## 3. Cold-Prompt Drafting, Quantization, & MoE Extensions (`M-006`..`M-008`)

<!-- SLIDE_06: assets/slides/slide_06.png -->
<!-- SLIDE_07: assets/slides/slide_07.png -->

Context-only suffix drafting drops to **`1.27x` (`7B`)** and **`1.26x` (`3B`)** when prompts share no prior n-grams. TPUForge implements four extensions to handle cold codegen, single-chip memory caps, and sparse MoE capacity across both Gemma 4 and Qwen3:

| Extension | Implemented Mechanism | Offline Verification Status (`55/55` pytest) | Cloud TPU Status |
| :--- | :--- | :--- | :--- |
| **`M-006`: Hybrid Block-Diffusion Tree** (`tpuforge_block_tree`) | `Gemma4BlockTreeRunner` (`benchmarks/gemma4_jax.py`) + `QwenBlockTreeRunner` (`benchmarks/block_diffusion_jax.py`) fuse parallel `DFlash` block-drafter `top-k` predictions with `O(1)` `SuffixIndex` continuations via `TreeBuilder::build_from_block_top_k`. | Verified (`test_block_tree_and_suffix.py` & `test_gemma4.py`): sub-`15 ms` tree packing at `32K+`, 4-mode parity, `transpose_lm_head` tied reuse, `TP=2` HLO green. | Staged on `v5e` / `v6e` |
| **`M-009`: Out-of-Band Repository AST Seeding** | `benchmarks/ast_seed_jax.py`, `SuffixIndex::seed_sequences`, `ForgeEngine::seed_ast_text`, `--ast-seed-path`, and HTTP `"seed_snippets"` pre-warm cold codegen drafting at zero prompt/KV cost. | Verified (`test_ast_seed_drafter.py`): Python/C++ AST extraction, exact greedy parity, 1-token fallback. | Staged on `v5e` / `v6e` |
| **`M-007`: Native `INT8` / `FP8` Systolic Kernels** | `benchmarks/quantization_jax.py` computes pre-sharding symmetric per-channel `int8` / `float8_e4m3fn` weights and scales (`>=2D` N-D periodic scan specs) executed via constant-free `jax.lax.dot_general`. | Verified (`test_quantized_systolic.py` & `test_gemma4.py`): `< 0.65x` linear HBM footprint vs `BF16`, Gemma 4 `INT8`/`FP8` 4-mode parity, `TP=2` `all-reduce` HLO. | Staged on `v5e` / `v6e` |
| **`M-008`: Host `mmap` Sparse MoE Streaming** | `benchmarks/moe_streaming_jax.py` + `Gemma4MoeStreamRunner` pin attention, routers, and KV caches in HBM while `ExpertHostStore` streams routed `3D` stacked or `2D` experts from `mmap` into `E_stage` slots amortized across `128` tree nodes. | Verified (`test_moe_streaming.py` & `test_gemma4.py`): `< 30%` Gemma 4 HBM trunk ratio (`< 45%` OLMoE), `0` steady-state recompiles, Windows `mmap` release verified. | Staged on `v5e` / `v6e` |

*(No simulated Cloud TPU numbers are reported anywhere for Phase 2 or Gemma 4 until live silicon JSON outputs complete.)*

---

## 4. Architecture and Subsystem Design

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
2. **`O(1)` `SuffixIndex` & Hybrid Block-Diffusion Packing (`M-003`, `M-006`, `M-009`):** Incremental 1-to-4-gram suffix hash index merges repository AST declarations and parallel `DFlash` `top-k` candidates into fixed `128`- or `256`-row causal tiles via `TreeBuilder::build_from_block_top_k`.
3. **Mixed-Precision & Tied-Embedding Kernels (`M-002`, `M-007`):** `StaticArena` (`128`-byte alignment) and constant-free weight tuples execute `bfloat16` activations against `int8`/`float8_e4m3fn` per-channel weights while reusing tied `embed_tokens` directly on Gemma 4 (`transpose_lm_head=True`).
4. **Host `mmap` Sparse MoE Expert Streaming (`M-008`):** `ExpertHostStore` streams active routed experts (`3D` stacked Gemma 4 / `Qwen3-MoE` or `2D` `OLMoE`) from `mmap` into fixed-shape `E_stage` HBM buffers amortized across all `128` speculative candidates simultaneously.
5. **Optical Torus Ring-Attention & GSPMD Tensor Parallelism (`M-005`):** Native `IciRingPlanner` + `jax.shard_map` + `jax.lax.ppermute` ring kernels alongside Megatron `P(..., "tp")` sharding (`TP=8` with replicated global KV when `num_global_key_value_heads < tp_size`) and persistent OpenXLA compilation caching.
6. **PJRT C ABI Backend (`pjrt_c_api.h`):** Dispatches fixed-shape compiled programs across local `libpjrt_cpu` and Cloud TPU `libpjrt_tpu.so`.

---

## 5. Command Reference and Hardware Matrix

```bash
# Google Gemma 4 26B-A4B Sparse MoE + DFlash + AST Seeding in INT8 (32 GiB v6e-1 HBM or TP=8 v5e-8)
python -m benchmarks.run_qwen_bench --model google/gemma-4-26B-A4B-it --draft-model z-lab/gemma-4-26B-A4B-it-DFlash --weight-quant int8 --ast-seed-path . --tile-size auto --tp-size auto

# Google Gemma 4 26B-A4B Sparse MoE via host mmap streaming on 16 GiB v5e-1
python -m benchmarks.run_qwen_bench --model google/gemma-4-26B-A4B-it --host-store mmap --stage-experts 16 --weight-quant int8 --ast-seed-path . --tile-size auto --tp-size 1

# Google Gemma 4 12B Dense + DFlash in INT8 (fits on single 15.75 GiB v5e-1 chip)
python -m benchmarks.run_qwen_bench --model google/gemma-4-12B-it --draft-model z-lab/gemma4-12B-it-DFlash --weight-quant int8 --ast-seed-path . --tile-size auto --tp-size auto

# Google Gemma 4 31B Dense + DFlash via Megatron TP=8 (pooled 128 GiB v5e-8)
python -m benchmarks.run_qwen_bench --model google/gemma-4-31B-it --draft-model z-lab/gemma-4-31B-it-DFlash --ast-seed-path . --tile-size auto --tp-size 8

# M-005 Optical Torus Ring-Attention & standalone Kaggle/Colab bundle builder
python -m benchmarks.ring_attention_jax --max-seq-len 8192 --valid-seq-len 6144 --ring-size auto --tp-size 1
python -m benchmarks.package_kaggle_bundle
```

| Hardware Target | Status | Scope |
| :--- | :--- | :--- |
| **Local CPU (PJRT CPU)** | Verified (`55/55` pytest) | `128`/`256` tile simulation, Gemma 4 periodic `5:1` scan + `3D` MoE + `DFlash` + `INT8`/`FP8` + `mmap` + `TP=2`, `O(1)` `SuffixIndex` + AST seeding, OpenXLA GSPMD TP parity |
| **Cloud TPU `v5e-1` (`15.75 GiB` HBM)** | Verified Silicon (`Qwen2.5`) / Staged (`Gemma 4`) | Verified: `32K` Flash-Ring (`519,735 tok/s`), `3B` (`328.79 tok/s`), `7B` (`202.80 tok/s`), `tile=128` vs `256` (`1.23x`). Staged: Gemma 4 `12B INT8` (`13.97 GiB`), `26B-A4B mmap INT8`, `Qwen3-4B` check |
| **Cloud TPU `v6e-1` (`32 GiB` HBM, Trillium)** | Ready / Staged | `256x256` MXU auto-inferred via `infer_mxu_dim`; `26B-A4B INT8 + DFlash` (`25.08 GiB` total) and `12B BF16/INT8 + DFlash` fit 100% inside single-chip HBM |
| **Kaggle Free TPU `v5e-8` (`128 GiB` HBM)** | Queued / Staged | `8x16 GiB` pooled HBM: `128K` ICI Ring-Attention (`M-005`), Gemma 4 `26B-A4B` `TP=8`, `31B` `TP=8` (`9.80 GiB/chip` tied `BF16`), and `12B` tile sweep |

* `include/`: Public C/C++ engine headers (`pjrt_c_api.h`, `kv_cache.h`, `tree_speculator.h`, `openai_server.h`, `engine.h`).
* `src/`: Core C++20 implementation (`core/`, `ops/`, `speculative/`, `runtime/`).
* `benchmarks/`: OpenXLA/JAX modules (`gemma4_jax.py`, `qwen_jax.py`, `quantization_jax.py`, `moe_streaming_jax.py`, `ast_seed_jax.py`, `block_diffusion_jax.py`, `ring_attention_jax.py`, `run_qwen_bench.py`, `package_kaggle_bundle.py`).
* `tests/`: `55` offline unit, C++/nanobind, XLA sharding, Gemma 4, quantized systolic, sparse MoE streaming, AST-seeding, block-tree, and ring-attention tests (`pytest`).
* `docs/`: Architecture and TRC proposal notes ([`docs/TRC_APPLICATION.md`](docs/TRC_APPLICATION.md)).

---

## 6. License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for details.
