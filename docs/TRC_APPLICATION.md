# TPU Research Cloud (TRC) Application Outline

**Project Title:** TPU Forge (TPUForge): High-Throughput Native Inference Engine for Google Cloud TPUs  
**Primary Contact:** Adam (floorsmithllc / CineClearAI)  
**Target Hardware:** Cloud TPU v5e / Cloud TPU v6e (Trillium)  
**Open Source Repository:** https://github.com/adamm285-dev/TPUForge  

---

## 1. Project Summary
TPU Forge is an open-source, high-performance LLM serving and inference engine built specifically for Google Cloud TPUs. While GPU-focused engines (such as vLLM, TensorRT-LLM, and SGLang) optimize for asynchronous CUDA streams and warp scheduling, TPU serving infrastructure remains heavily dependent on high-level Python frameworks that incur scheduling overhead, dynamic shape recompilation stalls, and sub-optimal systolic Matrix Multiply Unit (MXU) utilization.

TPU Forge targets the low-level OpenXLA PJRT C ABI directly alongside pure-JAX fixed-shape kernels, introducing:
1. **Systolic-Aligned Speculative Verification:** Aligning hybrid parallel block-diffusion (`DFlash`) and `O(1)` AST-seeded suffix candidate trees to fixed `128 x 128` (`v5e`) and `256 x 256` (`v6e`/`v7`) MXU tiles.
2. **Zero-Copy Static HBM Arenas & Tied-Weight Reuse:** Pre-allocated `128`-byte aligned KV-cache page tables, `INT8`/`FP8` mixed-precision `dot_general`, host-`mmap` MoE streaming, and tied `embed_tokens` reuse (`lm_head` deduplication).
3. **Inter-Chip Interconnect (ICI) Topology Exploitation:** Custom `("ring", "tp")` optical torus sharding and Megatron tensor parallelism across multi-chip TPU pods.

---

## 2. Target Models, Research Scope, & Open Source Impact
All benchmarks, kernel designs, and runtime code are open-sourced under the Apache 2.0 license (`55/55` offline pytest verification tests passing locally across two Python interpreters, `0` active mocks).

### Primary Evaluation Lineup: Google Gemma 4 (`Apache 2.0`, Ungated)
* **Flagship Sparse MoE (`google/gemma-4-26B-A4B-it` + `z-lab/gemma-4-26B-A4B-it-DFlash`):** `128` experts, top-`8` routing (`4B` active per token). Native per-channel `INT8` quantization shrinks total footprint from `48.07 GiB` `BF16` to **`25.08 GiB`**, fitting completely inside a single `32 GiB` Cloud TPU `v6e-1` HBM chip, scaling across `v5e-8` at `TP=8`, and streaming via host `mmap` (`3.81 GiB` HBM trunk + static KV) on `15.75 GiB` `v5e-1`.
* **Single-Chip Dense (`google/gemma-4-12B-it` + `z-lab/gemma4-12B-it-DFlash`):** Native `INT8` systolic quantization reduces footprint from `22.28 GiB` `BF16` to **`13.97 GiB` total** (`13.29 GiB` + `~0.68 GiB` DFlash), fitting cleanly inside a single `15.75 GiB` `v5e-1` chip while benchmarking both `BF16` and `INT8` on `v6e-1`.
* **Multi-Chip Dense (`google/gemma-4-31B-it` + `z-lab/gemma-4-31B-it-DFlash`):** Reuses tied `embed_tokens` directly (`transpose_lm_head=True`) to eliminate a duplicate `2.62 GiB` logits matrix, fitting across `v5e-8` at **`9.80 GiB/chip` (`TP=8`, `BF16`)** with replicated global KV handling when `num_global_key_value_heads < tp_size`.
* **Why Gemma 4 Amplifies TPUForge Optimizations:** Gemma 4's `262,144`-token vocabulary inflates naive host logits D2H transfers to **`134 MB/step`** (`128` tile) or **`268 MB/step`** (`256` tile); TPUForge's on-device `jnp.argmax` shrinks transfer to `512 B` / `1 KB`.
* **Explicit Scope Boundaries:** Edge Gemma 4 variants (`E2B`/`E4B` with Per-Layer Embeddings or cross-layer KV sharing) raise a configuration guard, and recurrent linear-attention models (`Qwen 3.6`/`3.8` DeltaNet state) are excluded because fixed-memory recurrent states cannot roll back zero-copy across `128`-node speculative tree branches. (`Qwen/Qwen3-4B` is retained strictly as a secondary cross-architecture check.)

### Verified Single-Chip Cloud TPU `v6e-1` (Trillium) Gemma 4 Silicon Results
All metrics below were measured live on Cloud TPU `v6e-1` (`TPU v6 lite`, `31.24 GiB` HBM, `tile=256`, `JAX 0.11.2`, archived in `benchmarks/results/colab_gemma4_results.json`):
* **`google/gemma-4-26B-A4B-it` Sparse MoE + `z-lab/gemma-4-26B-A4B-it-DFlash` (`INT8`, `24.95 GiB` target HBM):** Fitted 100% inside single-chip `31.24 GiB` `v6e-1` HBM (`1` vs `5` prompt-sweep compiles, `13.27s` vs `26.94s`), accelerating code refactoring from `5.53 tok/s` Vanilla JAX (`13.77 tok/s` Static Arena) to **`167.54 tok/s`** (`12.80` tok/pass, **`30.30x` decode / `32.78x` total vs Vanilla JAX**, **`12.17x` vs Static Arena**, `128/128` exact parity), cold codegen (`78,096` AST tokens) to **`125.83 tok/s`** (`9.85` tok/pass, `22.76x` decode / `25.80x` total, `128/128` exact parity), and cold chat to **`36.53 tok/s`** (`2.78` tok/pass, `6.62x` decode).
* **`google/gemma-4-12B-it` Dense + `z-lab/gemma4-12B-it-DFlash` (`BF16` vs `INT8`):** Native `INT8` systolic kernels reduced target HBM from `22.18 GiB` (`BF16`) to `12.97 GiB` (`INT8`, `0.585x`, fitting inside `15.75 GiB` `v5e-1` as well) while speeding Static Arena single-token decode `1.64x` (`28.73` vs `17.54 tok/s`). Code refactoring hit **`328.97 tok/s`** (`12.80` tok/pass, `12.07x` vs `INT8` Vanilla / `11.45x` vs Static / `18.00x` vs `BF16` Vanilla, `128/128` parity) in `INT8` (`244.28 tok/s` cold codegen, `92.61 tok/s` cold chat) and **`186.95 tok/s`** (`11.64` tok/pass, `10.23x` decode / `20.63x` total, `128/128` parity) in `BF16` (`154.19 tok/s` cold codegen, `57.30 tok/s` cold chat), with `1` vs `5` compiles (`4.46s` vs `39.30s` = `8.80x` `INT8`; `12.05s` vs `39.38s` = `3.27x` `BF16`).
* **Cross-Architecture `Qwen3-4B` (`BF16`, `8.22 GiB` HBM) & `v6e-1` Flash-Ring (`M-005`):** Reached **`900.55 tok/s`** on `Qwen3-4B` code refactor (`11.64` tok/pass, `7.06x` decode / `8.65x` vs Static, `128/128` parity) and `764.05 tok/s` on cold codegen (`114/114` parity), while `28,672/32,768` Flash-Ring reached **`638,908 tok/s`** prefill (`44.88 ms`) and **`1.116 ms`** per `256`-node tree step (**`229,305` tree-nodes/s**, `1.85x` `v5e-1` node throughput).

### Preliminary Single-Chip TPU `v5e-1` Verified Silicon Results (`Qwen2.5-Coder` Phase 1 Baseline)
* **Qwen2.5-Coder-7B-Instruct (`tile=128`, `max_seq=1024`, `gen=64`, `BF16`):** `202.80 tok/s` decode vs. `49.25 tok/s` Vanilla JAX on code refactoring (`4.12x` decode speedup, `7.50x` end-to-end speedup `0.358s` vs. `2.685s`, `4.57` tok/pass, `64/64` exact greedy parity) and `62.38 tok/s` vs. `49.30 tok/s` on cold codegen (`1.27x` decode, `1.84x` total, `64/64` parity). Moving `jnp.argmax` onto HBM shrinks a `128`-node verification step to `22.54 ms` vs. `20.63 ms` for `1`-token Vanilla (`~1.09x`).
* **Qwen2.5-Coder-3B-Instruct (`tile=128`, `max_seq=2048`, `gen=128`, `BF16`) & Zero-Recompilation (`M-002`):** `328.79 tok/s` decode vs. `94.88 tok/s` Vanilla JAX on code refactoring (`3.47x` decode speedup, `5.68x` end-to-end speedup `0.411s` vs. `2.334s`, `3.88` tok/pass, `128/128` parity; `11.80 ms` vs. `10.62 ms` per pass) and `119.80 tok/s` vs. `95.13 tok/s` on cold codegen (`1.26x` decode, `1.79x` total). Across `[32, 64, 95, 195]` unaligned prompt sweeps: `1` vs. `5` compiles (`2.75x` faster on `3B`, `12.60x` on lengths `64..195`; `2.37x` faster on `7B`, `7.72x` on lengths `64..195`).
* **MXU Hardware-Matched Tile Sizing (`tile=128` vs. `tile=256` on `v5e`):** `328.79 tok/s` (`11.80 ms/pass`) at hardware-matched `tile=128` vs. `268.39 tok/s` (`14.45 ms/pass`) at `tile=256` on `Qwen2.5-Coder-3B-Instruct` (`1.23x` faster pure MXU compute after removing the `156 MB` vs. `78 MB` PCIe logits bottleneck).
* **Blockwise Flash-Ring Attention (`M-005`):** `519,735 tok/s` prefill and `1.032 ms` tree verification step at `28,672/32,768` context (`1.75 GiB` KV), with `3.91e-03` prefill and `1.22e-04` tree `BF16` max abs error at `3,584/4,096`, motivating multi-chip `v5e-8` and `v6e-8` ICI pod scaling.

---

## 3. Resource Request
With single-chip `v5e-1` and `v6e-1` Trillium Gemma 4 silicon results already verified above, this TRC request specifically targets multi-chip ICI pod validation (`M-005` / `M-006` Pod Closure) and single-chip `v5e-1` host-`mmap` MoE profiling (`M-008`):
* **Initial Phase:** 1x Cloud TPU v5e-4 or v5e-8 slice for multi-chip Gemma 4 `31B` (`9.80 GiB/chip` `TP=8` `BF16`) and `26B-A4B` `TP=8` profiling alongside `128K` ICI ring-attention scaling.
* **Scaling Phase:** 1x Cloud TPU v6e-4 or v6e-8 Trillium pod slice to extend verified single-chip `256 x 256` MXU results (`328.97 tok/s` `12B INT8`, `167.54 tok/s` `26B-A4B INT8`) across pod-scale ICI tensor parallelism and `FP8` kernels.

---

## 4. Application Link
Submit this proposal via the official Google TPU Research Cloud portal:
https://sites.research.google/trc/about/
