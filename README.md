# TPU Forge (TPUForge)

**High-Throughput Native Inference Engine for Google Cloud TPUs**  
*Inspired by the zero-overhead, bare-metal philosophy of TensorFold (Ash Hart), tailored for TPU Systolic Matrix Units (MXUs) and Inter-Chip Interconnects (ICI).*

---

## 1. Vision & Philosophy

Modern LLM serving engines have become burdened by heavy framework abstractions: multi-gigabyte Python runtimes, dynamic dispatch graphs, GC pauses, and unaligned tensor allocations.

**TPU Forge** is designed to explore the frontier of low-level TPU efficiency:
1. **Low-Level Native Execution:** Target Google's OpenXLA and PJRT (Python/C JIT Runtime) C ABI directly, bypassing Python interpreter latency on high-throughput token loops.
2. **Systolic-Aligned Speculative Verification:** Traditional speculative decoding (draft -> verify) on GPUs optimizes for memory bandwidth. On TPUs, speculative verification must align with the $128 \times 128$ Matrix Multiply Unit (MXU) systolic array to prevent idle execution cycles.
3. **Inter-Chip Interconnect (ICI) Topology Exploitation:** TPUs are connected in ultra-low-latency 2D/3D torus topologies. TPU Forge designs tensor and pipeline sharding explicitly around ICI link bandwidth rather than generic all-reduce abstractions.
4. **Zero-Copy HBM Buffer Pooling:** Pre-allocated, pinned High Bandwidth Memory (HBM) ring buffers for Key-Value (KV) cache pages to ensure zero memory fragmentation during continuous token generation.

---

## 2. Architecture Overview

```
                      +---------------------------------------+
                      |       Client / OpenAI API Server      |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         TPUForge Native Engine         |
                      | (Token Dispatch, Continuous Batching) |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             [Speculative Decoder]                 [KV-Cache Manager]
         (Aligned MXU Draft-Verify)               (Pinned Ring-Buffer)
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         PJRT C API Abstraction        |
                      |  (pjrt_c_api.h / Hardware Driver ABI)  |
                      +---------------------------------------+
                                          |
                 +------------------------+------------------------+
                 |                                                 |
                 v                                                 v
      [Local PJRT CPU Plugin]                         [Cloud TPU PJRT Plugin]
      (Development / Simulation)                    (libpjrt_tpu.so / Hardware)
```

---

## 3. Directory Layout

- `include/`: Public C/C++ engine headers and the PJRT C API definitions (`pjrt_c_api.h`).
- `src/`: Core engine implementation:
  - `core/`: Tensor buffers, memory allocators, and hardware device abstraction.
  - `ops/`: Systolic-aligned matrix operations and attention verification kernels.
  - `speculative/`: Speculative decoding coordination and exact-token verification.
  - `runtime/`: PJRT loader and HLO execution pipeline.
- `benchmarks/`: Microbenchmarks for memory bandwidth, latency, and tokens/sec.
- `scripts/`: TPU provisioning, research cloud submission materials, and Colab runners.
- `colab/`: Ready-to-run Google Colab notebooks for testing on interactive TPU runtimes.

---

## 4. Hardware Roadmap

| Hardware Target | Status | Focus Area |
| :--- | :--- | :--- |
| **Local CPU (PJRT CPU)** | Active | 128/256 tile simulation, OpenXLA GSPMD TP parity, 28/28 pytest |
| **Kaggle Free TPU `v5e-8` (128 GB HBM)** | Ready | 8x16 GB pooled HBM, Megatron TP (`7B` auto, `32B` `TP=8`), streaming loader |
| **Cloud TPU `v5e` / `v6e` (Trillium)** | Target | Auto `128x128` (v5e) or `256x256` (v6e/v7) MXU via `infer_mxu_dim` |
| **Cloud TPU `v7` / `8x` Series** | Supported | Native `256x256` systolic tile parameterization |

---

## 5. Getting Started

### Local Build & Test
```bash
# Configure and compile with CMake + Ninja
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Verify native 128-row and 256-row systolic paths
./build/tpuforge_bench --tile-size 128
./build/tpuforge_bench --tile-size 256
pytest -q
```

### Real-Weight Qwen2.5-Coder Benchmark, ICI Ring-Attention (`M-005`), & Kaggle `v5e-8` Bundle
```bash
# Run Vanilla JAX vs StaticArena (M-002) vs TPUForge Tree (M-003)
python -m benchmarks.run_qwen_bench --model Qwen/Qwen2.5-Coder-0.5B-Instruct --tile-size auto --tp-size auto

# Run M-005 Optical Torus Ring-Attention (ppermute prefill + stationary-KV systolic tree decode via IciRingPlanner)
python -m benchmarks.ring_attention_jax --max-seq-len 8192 --valid-seq-len 6144 --ring-size auto --tp-size 1

# Build standalone source zip for Kaggle / Colab TPU upload
python -m benchmarks.package_kaggle_bundle
```

### Running Native CLI on TPU via PJRT
Set the environment variable pointing to the TPU PJRT plugin:
```bash
./build/tpuforge_bench --plugin /path/to/libpjrt_tpu.so --program model.mlir --tile-size 256
```

---

## 6. License
Apache 2.0 / MIT Dual License.
