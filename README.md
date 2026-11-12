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
| **Local CPU (PJRT CPU)** | Prototyping | Architectural validation, token loops, tests |
| **Google Colab TPU (v2/v3/v4)** | Experimental | Interactive validation, latency baselines |
| **Cloud TPU v5e (Single-Chip)** | Target | Low-cost inference profiling (\$0.50/hr spot) |
| **Cloud TPU v6e (Trillium)** | Production Goal | 4.7x compute density, large-scale multi-slice ICI |

---

## 5. Getting Started

### Local Build & Test
```bash
# Configure and compile with CMake
cmake -B build -S .
cmake --build build --config Release

# Run test suite
./build/tpuforge_tests
```

### Running on TPU via PJRT
Set the environment variable pointing to the TPU PJRT plugin:
```bash
export PJRT_PLUGIN_PATH=/path/to/libpjrt_tpu.so
./build/tpuforge_bench --model gemma-2b
```

---

## 6. License
Apache 2.0 / MIT Dual License.
