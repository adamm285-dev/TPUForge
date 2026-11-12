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

---

## 3. Resource Request
- **Initial Phase:** 1x Cloud TPU v5e-4 or v5e-8 slice for single-node kernel profiling and KV-cache ring buffer tuning.
- **Scaling Phase:** 1x Cloud TPU v6e-8 or v4-16 pod slice for multi-chip ICI tensor-parallelism benchmarking.

---

## 4. Application Link
Submit this proposal via the official Google TPU Research Cloud portal:
https://sites.research.google/trc/about/
