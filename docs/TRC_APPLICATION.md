# TPU Research Cloud (TRC) Application Outline

**Project Title:** TPU Forge (TPUForge): High-Throughput Native Inference Engine for Google Cloud TPUs  
**Primary Contact:** Adam (floorsmithllc / CineClearAI)  
**Target Hardware:** Cloud TPU v5e / Cloud TPU v6e (Trillium)  
**Open Source Repository:** https://github.com/adamm285/TPUForge  

---

## 1. Project Summary
TPU Forge is an open-source, high-performance LLM serving and inference engine designed specifically for Google Cloud TPUs. While GPU-focused engines (such as vLLM, TensorRT-LLM, and TensorFold) have dramatically advanced inference throughput on NVIDIA hardware, TPU serving infrastructure remains heavily dependent on high-level Python frameworks that incur scheduling overhead, dynamic memory fragmentation, and sub-optimal systolic tile utilization.

TPU Forge targets the low-level OpenXLA PJRT C ABI directly, introducing:
1. **Systolic-Aligned Speculative Verification:** Aligning speculative multi-token draft batches to $128 \times 128$ MXU matrix multiply tiles.
2. **Zero-Copy HBM Buffer Pooling:** Pre-allocated page tables directly in TPU High Bandwidth Memory.
3. **Inter-Chip Interconnect (ICI) Topology Exploitation:** Custom pipeline and tensor sharding over TPU torus topologies for multi-chip pods.

---

## 2. Research & Open Source Impact
All benchmarks, kernel designs, and runtime code developed under TPU Forge will be open-sourced under the Apache 2.0 license. The research will produce open benchmarks comparing:
- TPU v5e/v6e native PJRT token latency vs. Python/JAX baseline token latency.
- Speculative acceptance rates and wall-clock token acceleration under systolic-aligned batching.

### Preliminary Single-Chip TPU `v5e-1` Silicon Results
- **Qwen2.5-Coder-7B-Instruct (`tile=128`, `max_seq=1024`, `gen=64`, BF16):** `202.80 tok/s` decode vs. `49.25 tok/s` Vanilla JAX on code refactoring (`4.12x` decode speedup, `7.50x` end-to-end speedup `0.358s` vs. `2.685s`, `4.57` tok/pass, `64/64` exact greedy parity) and `62.38 tok/s` vs. `49.30 tok/s` on cold codegen (`1.27x` decode, `1.84x` total, `64/64` parity). Eliminating host logits transfer via on-device `jnp.argmax` reduces a `128`-node systolic verification step to `22.54 ms` vs. `20.63 ms` for a `1`-token Vanilla step (`~1.09x`).
- **Qwen2.5-Coder-3B-Instruct (`tile=128`, `max_seq=2048`, `gen=128`, BF16) & Zero-Recompilation (`M-002`):** `328.79 tok/s` decode vs. `94.88 tok/s` Vanilla JAX on code refactoring (`3.47x` decode speedup, `5.68x` end-to-end speedup `0.411s` vs. `2.334s`, `3.88` tok/pass, `128/128` parity; `11.80 ms` vs. `10.62 ms` per pass = `~1.11x`) and `119.80 tok/s` vs. `95.13 tok/s` on cold codegen (`1.26x` decode, `1.79x` total). Across `[32, 64, 95, 195]` unaligned prompt sweeps: `1` vs. `5` compiles (`2.75x` faster on `3B`, `12.60x` on lengths `64..195`; `2.37x` faster on `7B`, `7.72x` on lengths `64..195`).
- **MXU Hardware-Matched Tile Sizing (`tile=128` vs. `tile=256` on `v5e`):** `328.79 tok/s` (`11.80 ms/pass`) at hardware-matched `tile=128` vs. `268.39 tok/s` (`14.45 ms/pass`) at `tile=256` on `Qwen2.5-Coder-3B-Instruct` (`1.23x` faster pure MXU execution after removing the `156 MB` vs. `78 MB` PCIe logits bottleneck via on-device argmax).
- **Blockwise Flash-Ring Attention (`M-005`):** `519,735 tok/s` prefill and `1.032 ms` tree verification step at `28,672/32,768` context (`1.75 GiB` KV), with `3.91e-03` prefill and `1.22e-04` tree BF16 max abs error at `3,584/4,096`, motivating multi-chip `v5e-8` and `v6e-8` ICI pod scaling.

---

## 3. Resource Request
- **Initial Phase:** 1x Cloud TPU v5e-4 or v5e-8 slice for single-node kernel profiling and KV-cache ring buffer tuning.
- **Scaling Phase:** 1x Cloud TPU v6e-8 or v4-16 pod slice for multi-chip ICI tensor-parallelism benchmarking.

---

## 4. Application Link
Submit this proposal via the official Google TPU Research Cloud portal:
https://sites.research.google/trc/about/
