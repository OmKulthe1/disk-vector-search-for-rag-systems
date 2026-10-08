# LightVector: Rust Disk-Native Vector Search Engine Roadmap

A production-grade, 11-step development roadmap for an asynchronous, page-aligned disk-based vector search engine with Python PyO3 bindings.

---

## - [x] Task 01: Core Primitives & SIMD Distance Metrics Engine
* **Scope:** Implement foundational data structures (`Vector`, `NodeId`, `DistanceMetric`) and SIMD-accelerated distance algorithms ($L_2$ Euclidean, Cosine, Inner Product).
* **Technical Details:** Use `std::simd` or platform intrinsic crates (`packed_simd` / AVX2 / NEON) for 128/256/512-bit vector math on `f32` slices.
* **Acceptance Criteria:**
  - [x] Computes $L_2$ and Cosine distances for 1536-dimensional vectors in sub-microsecond time.
  - [x] Automatically falls back to scalar calculations if SIMD hardware features are absent.
* **Testing Guidelines:**
  - *Unit Tests:* Compare SIMD output precision against standard `f32::dot` calculations across $10^6$ random vector pairs (epsilon error $< 10^{-5}$).
  - *Benchmark:* Add `criterion` benchmarks measuring QPS for 384, 768, and 1536 dimension inputs.

---

## - [x] Task 02: Product Quantization (PQ) Compression Module
* **Scope:** Build an in-RAM vector compressor using $k$-means clustering to convert float vectors into short byte codes.
* **Technical Details:** Segment $D$-dimensional vectors into $M$ sub-vectors. Run $k$-means ($k=256$) to generate codebooks. Construct Asymmetric Distance Computation (ADC) lookup tables in memory.
* **Acceptance Criteria:**
  - [x] Compresses 1536-dimensional `f32` vectors ($6,144$ bytes) down to $64$–$128$ bytes ($16\times$–$96\times$ compression ratio).
  - [x] ADC table lookups run completely in RAM with zero heap allocations during search.
* **Testing Guidelines:**
  - *Unit Tests:* Verify codebook generation and assert exact byte encoding/decoding parity for known cluster centers.
  - *Integration Test:* Measure mean squared error (MSE) between full-precision distance and PQ estimated distance on a 10,000-vector dataset.

---

## - [x] Task 03: 4 KB Page-Aligned Sector Serialization Layout
* **Scope:** Design zero-copy binary disk sector structures aligned strictly to 4,096-byte boundaries for direct NVMe SSD I/O.
* **Technical Details:** Utilize `zerocopy` / `bytemuck` with `#[repr(C, align(4096))]`. Store metadata, uncompressed `f32` vector, and adjacency list within sector boundaries.
* **Acceptance Criteria:**
  - [x] Every graph node block maps directly to a 4 KB physical page boundary without byte padding bleed across pages.
  - [x] Zero heap allocations required when casting raw sector byte slices into node structs (`zerocopy::FromBytes`).
* **Testing Guidelines:**
  - *Unit Tests:* Assert `std::mem::align_of::<DiskNode>() == 4096` and `std::mem::size_of::<DiskNode>() % 4096 == 0`.
  - *Roundtrip Test:* Write synthetic nodes to file, read back raw bytes, and assert complete field identity.

---

## - [ ] Task 04: In-Memory Vamana Graph Construction Engine
* **Scope:** Implement the single-layer Vamana graph indexing algorithm using directional $\alpha$-pruning.
* **Technical Details:** Greedy graph traversal from a medoid node. Candidate pruning using $\alpha$-distance threshold ($\alpha \ge 1.0$) to establish long-range shortcut edges alongside local cluster edges.
* **Acceptance Criteria:**
  - [ ] Graph out-degree is strictly bounded by parameter $R_{max}$ (e.g., $R_{max} = 64$).
  - [ ] No orphan nodes or disconnected subgraphs exist; all nodes are reachable from the medoid entry point.
* **Testing Guidelines:**
  - *Graph Integrity Test:* Perform breadth-first search (BFS) from medoid to verify 100% node reachability on a 5,000-vector dataset.
  - *Property Tests:* Validate that increasing $\alpha$ decreases average graph path length (diameter).

---

## - [ ] Task 05: Disk Layout Exporter & Binary Index Builder
* **Scope:** Flush in-memory Vamana graph structures and full-precision vectors into the page-aligned binary disk index file format (`.lvec`).
* **Technical Details:** Write header block (dimensions, count, medoid ID, PQ codebook) followed by sequential sector-aligned node entries.
* **Acceptance Criteria:**
  - [ ] Generates a single, deterministic `.lvec` index file readable by random offset reads.
  - [ ] File size scales strictly as $N \times 4096 \text{ bytes} + \text{Header Size}$.
* **Testing Guidelines:**
  - *Sanity Test:* Parse exported binary file header and verify offset math points to valid node IDs.
  - *Corrupt Input Test:* Ensure engine gracefully returns `Err(IndexCorrupted)` when encountering unexpected file length or invalid magic bytes.

---

## - [ ] Task 06: Asynchronous Direct Disk I/O Driver (`io_uring` / `O_DIRECT`)
* **Scope:** Build a high-throughput, non-blocking disk reader bypassing the OS page cache for high-concurrency NVMe reads.
* **Technical Details:** Wrap `tokio-uring` or `rio` with Linux `O_DIRECT` flags. Implement async batch sector fetching for candidate node lists.
* **Acceptance Criteria:**
  - [ ] Executes parallel 4 KB disk block reads without blocking Tokio worker threads.
  - [ ] Sector buffer memory addresses are strictly aligned to system page boundaries.
* **Testing Guidelines:**
  - *Performance Test:* Execute 1,000 parallel random sector fetches; assert throughput exceeds standard `std::fs::File::seek` reads by $>3\times$.
  - *Platform Fallback Test:* Verify `mmap` / standard async fallback operates correctly on non-Linux operating systems.

---

## - [ ] Task 07: Disk-Native Beam Search & Candidate Re-Ranking Engine
* **Scope:** Combine in-RAM PQ navigation with async SSD vector reads to perform fast, accurate Top-$k$ nearest neighbor search.
* **Technical Details:** 
  1. Route search path in RAM using PQ lookup tables.
  2. Select top $W$ candidate node IDs.
  3. Issue async `io_uring` batch fetch for candidate sectors.
  4. Compute exact $L_2$/Cosine distance using SSD uncompressed vectors and re-rank.
* **Acceptance Criteria:**
  - [ ] Achieves $\ge 90\%$ Recall@10 on 100,000 vector datasets.
  - [ ] Average query latency stays under $35\text{ ms}$ on standard NVMe drives.
* **Testing Guidelines:**
  - *Accuracy Test:* Compare query results against exact brute-force linear scan ($L_2$ exact match).
  - *I/O Budget Test:* Verify that average disk reads per query stay below $\le 8$ sector fetches.

---

## - [ ] Task 08: Active RAM Staging Buffer for Dynamic Inserts
* **Scope:** Support live vector insertions without requiring an immediate, expensive full-disk index rebuild.
* **Technical Details:** Maintain a lock-free in-memory staging buffer (RAM graph) alongside the frozen disk index. Route queries concurrently across both tiers and merge candidate lists.
* **Acceptance Criteria:**
  - [ ] Newly inserted vectors are immediately searchable within $<5\text{ ms}$.
  - [ ] Thread-safe concurrent writes and reads without deadlocks or query latency spikes.
* **Testing Guidelines:**
  - *Concurrency Test:* Run continuous background writes while firing 100 concurrent search queries; assert zero data races or panic crashes.
  - *Merge Test:* Verify query results properly unite top candidates from both RAM buffer and disk index.

---

## - [ ] Task 09: Python FFI & Binding Layer (PyO3 & Maturin)
* **Scope:** Expose the Rust vector search engine as an importable Python module (`lightvector`).
* **Technical Details:** Implement PyO3 bindings for `IndexReader`, `IndexBuilder`, and `SearchConfig`. Release the Python GIL (`py.allow_threads`) during disk I/O operations.
* **Acceptance Criteria:**
  - [ ] Engine installs via `pip install .` via Maturin.
  - [ ] Accepts NumPy `float32` arrays directly as vector inputs without unnecessary memory copies.
* **Testing Guidelines:**
  - *Pytest Suite:* Write Python unit tests checking index build, save, load, and query methods.
  - *Memory Leak Test:* Run 100,000 Python search calls under Valgrind/AddressSanitizer to ensure zero memory leaks across FFI boundaries.

---

## - [ ] Task 10: End-to-End RAG Pipeline & Local LLM Adapter
* **Scope:** Integrate the Python engine into a complete local RAG system using local embedding models and Ollama.
* **Technical Details:** Ingest text corpus $\rightarrow$ chunk text $\rightarrow$ generate embeddings via HuggingFace `sentence-transformers` $\rightarrow$ store in LightVector $\rightarrow$ pass retrieved chunks into `Ollama` (Llama-3).
* **Acceptance Criteria:**
  - [ ] Successfully retrieves relevant document context and outputs correct answers for natural language queries.
  - [ ] End-to-end user query execution time under 2 seconds (including LLM generation).
* **Testing Guidelines:**
  - *E2E Test:* Ingest 10,000 Wikipedia text chunks, execute 50 factual queries, and assert context relevance.

---

## - [ ] Task 11: Automated Evaluation Suite (Benchmarking & Profiling)
* **Scope:** Build automated CI tools to track Recall@10, QPS, RAM footprint, and disk read counts across releases.
* **Technical Details:** Automated harness executing standard vector benchmarks (e.g., SIFT1M, GLOVE100).
* **Acceptance Criteria:**
  - [ ] Generates a structured Markdown benchmark report after every major build.
  - [ ] Confirms system RAM usage stays under $<15\%$ of total dataset size.
* **Testing Guidelines:**
  - *Regression Test:* Fail CI pipeline if Recall@10 drops below 90% or latency increases by $>15\%$.