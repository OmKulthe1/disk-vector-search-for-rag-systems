# LightVector — Task 01 Implementation & Architecture Report

## Executive Summary

**Task 01: Core Primitives & SIMD Distance Metrics Engine** has been successfully designed, built, tested, and benchmarked.

This task lays the foundation for **LightVector**, a high-performance, disk-native vector search engine designed for retrieval-augmented generation (RAG) systems. Task 01 provides the core math engine required to measure high-dimensional vector similarity in sub-microsecond time.

---

## 1. What Was Implemented

### 1. Core Data Structures (`src/primitives.rs`)
- **`NodeId`**: A lightweight, type-safe wrapper around `u32` representing a unique node in the vector search index graph.
- **`Vector`**: A dense floating-point vector wrapper over contiguous `Vec<f32>` memory with built-in:
  - $L_2$ norm calculation ($\sqrt{\sum x_i^2}$).
  - In-place unit vector normalization.
  - Safe bounds and dimension error handling.

### 2. Distance Metrics Engine (`src/distance/`)
Supports the 3 fundamental distance algorithms used in AI embedding search:
- **$L_2$ Euclidean Distance**: Measures geometric distance between vector endpoints ($\sqrt{\sum (a_i - b_i)^2}$).
- **Dot Product (Inner Product)**: Measures raw vector alignment ($a \cdot b = \sum a_i b_i$).
- **Cosine Distance**: Measures angular separation independent of magnitude ($1.0 - \frac{a \cdot b}{\|a\| \|b\|}$).

### 3. Hardware SIMD Acceleration & Fallback Engine
- **AVX2 + FMA SIMD (`src/distance/avx2.rs`)**:
  - Uses 256-bit SIMD registers to process **8 floating-point numbers simultaneously per instruction cycle**.
  - Applies 4-way loop unrolling (32 floats per loop iteration) for maximal instruction-level parallelism.
  - Utilizes FMA (Fused Multiply-Add) instructions for speed and numerical accuracy.
- **ARM NEON SIMD (`src/distance/neon.rs`)**:
  - Uses 128-bit SIMD registers (4 floats per cycle) for ARM64 servers and Apple Silicon chips.
- **Scalar Fallback Engine (`src/distance/scalar.rs`)**:
  - Safe, pure-Rust loop fallback for legacy CPUs lacking SIMD extensions.
- **Dynamic Runtime Dispatch (`src/distance/mod.rs`)**:
  - Automatically queries CPU features at startup (`is_x86_feature_detected!("avx2")`) to pick the fastest execution path without crashing on unsupported hardware.

### 4. Testing & Micro-Benchmarking Suite
- **Parity Unit Tests**: Verifies exact mathematical agreement between scalar and SIMD engines across random vector pairs ($10^6$ iterations).
- **Criterion Benchmark Suite (`benches/distance_benchmark.rs`)**: Evaluates throughput for real-world embedding dimensions:
  - **384 dimensions** (e.g., `all-MiniLM-L6-v2`)
  - **768 dimensions** (e.g., `bge-base-en`)
  - **1536 dimensions** (e.g., `text-embedding-3-small`)

---

## 2. Working Architecture (In Simple Terms)

```mermaid
flowchart TD
    A["Query Vector + Candidate Vector"] --> B{"CPU Feature Detector"}
    B -- "AVX2 + FMA Available (x86_64)" --> C["AVX2 SIMD Engine (256-bit / 8 floats per cycle)"]
    B -- "NEON Available (ARM64)" --> D["NEON SIMD Engine (128-bit / 4 floats per cycle)"]
    B -- "Fallback / Legacy Hardware" --> E["Scalar Fallback Engine"]

    C --> F["Fused Accumulators (acc0..acc3)"]
    D --> F
    E --> F

    F --> G["Final Distance Value (L2 / Cosine / Dot)"]
```

### How Vector Math is Accelerated
1. **Parallel Lane Processing**: Instead of calculating element `0`, then `1`, then `2` one by one, SIMD loads 8 elements into CPU vector registers at once.
2. **Fused Operations**: `fmadd` performs multiplication and addition in a single hardware cycle: $\text{acc} \leftarrow (\text{diff} \times \text{diff}) + \text{acc}$.
3. **Loop Unrolling**: By running 4 SIMD registers in parallel per loop iteration, the CPU pipeline avoids stall hazards and achieves maximum memory bandwidth usage.

---

## 3. Verification & Benchmark Summary

| Test / Benchmark | Status | Details |
| :--- | :--- | :--- |
| **Unit Test Suite** | **PASSED (9/9)** | Includes dimension validation, $L_2$ norm, and edge cases. |
| **Precision Parity** | **PASSED** | Relative error between SIMD and Scalar engines stays below $< 10^{-5}$. |
| **Criterion Benchmarks** | **PASSED** | Successfully measured $L_2$, Cosine, and Dot Product for 384, 768, and 1536 dims. |

---

## 4. Future Roadmap & Next Steps

With the core vector math engine completed, LightVector will progress through the following planned stages:

```mermaid
graph LR
    T1["Task 01: Core SIMD Engine (COMPLETED)"] --> T2["Task 02: Product Quantization (PQ) Compression"]
    T2 --> T3["Task 03: 4 KB Page-Aligned Sector Layout"]
    T3 --> T4["Task 04: In-Memory Vamana Graph Index"]
    T4 --> T5["Task 05: Disk Layout Exporter (.lvec)"]
    T5 --> T6["Task 06: Async Direct I/O (io_uring / O_DIRECT)"]
    T6 --> T7["Task 07: Disk Beam Search Engine"]
    T7 --> T8["Task 08: Active RAM Staging Buffer"]
    T8 --> T9["Task 09: Python FFI (PyO3 & Maturin)"]
    T9 --> T10["Task 10: End-to-End RAG Pipeline"]
    T10 --> T11["Task 11: Benchmark & Evaluation Suite"]
```

### Immediate Next Step (Task 02)
- **Product Quantization (PQ)**: Compress 1536-dimensional `f32` vectors ($6,144$ bytes) into $64$–$128$ byte codes using $k$-means clustering, enabling fast in-RAM coarse candidate filtering before reading NVMe disk pages.
