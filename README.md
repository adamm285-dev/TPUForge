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
| **Local CPU (PJRT CPU)** | Active | 128/256 tile simulation, OpenXLA GSPMD TP parity, 31/31 pytest |
| **Kaggle Free TPU `v5e-8` (128 GB HBM)** | Ready | 8x16 GB pooled HBM, Megatron TP (`7B` auto, `32B` `TP=8`), streaming loader |
| **Cloud TPU `v5e` / `v6e` (Trillium)** | Verified (v5e-1) / Ready (v6e) | Auto `128x128` (v5e) or `256x256` (v6e/v7) MXU via `infer_mxu_dim` |
| **Cloud TPU `v7` / `8x` Series** | Supported | Native `256x256` systolic tile parameterization |

### Verified Cloud TPU Silicon Telemetry (`TPU v5 lite` / `v5e-1`, `15.75 GiB` HBM, `JAX 0.11.2`, BF16)

| Benchmark / Model | Configuration | Measured Silicon Results |
| :--- | :--- | :--- |
| **Qwen2.5-Coder-7B-Instruct** (`M-002`, `M-003`) | `tile=128`, `TP=1`, `max_seq=1024`, `gen=64` | Prompt sweep: `1` vs `5` compiles (`1.667s` vs `3.916s`, **`2.35x` faster**; `0.350s` vs `2.743s` = **`7.83x`** on lengths `64..195`). Code refactor: **`115.27 tok/s`** vs `49.18 tok/s` Vanilla JAX (**`2.34x` decode speedup**, **`4.05x` total speedup** `0.636s` vs `2.575s`, **`4.57` tok/pass**, `64/64` exact greedy parity). |
| **Qwen2.5-Coder-3B-Instruct** (`M-002`, `M-003`) | `tile=128` vs `tile=256`, `TP=1`, `max_seq=2048`, `gen=128` | Prompt sweep (`tile=128`): `1` vs `5` compiles (`1.304s` vs `3.269s`, **`2.51x` faster**). Code refactor: **`97.22 tok/s`** (`1.395s` vs `2.263s` total = **`1.62x`**, `3.88` tok/pass, `128/128` parity) at hardware-matched `tile=128` vs `48.81 tok/s` at `tile=256` (**`1.99x` faster** via `infer_mxu_dim`). |
| **M-005 Blockwise Flash-Ring** (`M-005`) | `tile=128`, `ring=1`, `TP=1`, BF16 | `3584/4096` parity: **`2,680,640 tok/s`** prefill (`1.337 ms`), **`0.353 ms`** tree step (`362,752` tree-nodes/s, `3.91e-03` / `1.22e-04` max abs err). `28672/32768` (`1.75 GiB` KV): **`519,735 tok/s`** prefill (`55.167 ms`), **`1.032 ms`** tree step (`124,047` tree-nodes/s, `collective-permute` HLO verified). |

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
