# Long-Term Hardware-Accelerated Indexing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build capability-driven embedding acceleration while preserving CPU reliability and the project's local/offline-first behavior.

**Architecture:** Optimize and measure cross-file embedding batches before introducing accelerators. Keep query embeddings on CPU and run passage indexing in a disposable provider-specific worker selected by capability probes, with CPU fallback and locked installer environments.

**Tech Stack:** Python 3.12/3.13, FastEmbed, ONNX Runtime execution providers, WebGPU, CUDA, MIGraphX/ROCm, LanceDB, `uv`, pytest.

## Summary

- Optimize cross-file batching before adding GPU backends.
- Keep query embeddings on CPU; run passage indexing in a disposable accelerator worker.
- Default installation mode is automatic detection with explicit override.
- Never install packages at application runtime; the installer prepares locked provider environments.
- Promote a backend to automatic selection only after correctness and performance gates pass for the default Jina model.

## Public Interfaces

- Add installer option:

  ```text
  --accelerator auto|cpu|cuda|webgpu|migraphx|coreml
  ```

  `auto` is the default. Failed detection, installation, or probing falls back to CPU and reports the reason.

- Add runtime configuration:

  ```text
  INCODE_EMBED_ACCELERATOR=auto|cpu|cuda|webgpu|migraphx|coreml
  INCODE_EMBED_BATCH_SIZE=auto|1..256
  INCODE_EMBED_MEMORY_MB=<positive integer>
  INCODE_EMBED_STRICT=0|1
  ```

  `STRICT=1` disables CPU fallback.

- Add `code-indexing-mcp model status`, returning model ID, requested/resolved accelerator, device, execution provider, available providers, batch calibration, benchmark-cache state, and fallback reason.

- Extend `IndexReport` with `embedding_backend`, `embedding_batch_size`, `scan_ms`, `parse_ms`, `embed_ms`, `commit_ms`, `peak_memory_bytes`, and `fallback_count`. Existing fields remain unchanged.

## Phased Implementation

### Phase 1: Measurement and CPU Pipeline

- Add phase timing, throughput, batch-size, and peak-memory measurements.
- Add a reproducible benchmark command covering cold start, warm indexing, incremental indexing, and forced reindexing, with machine-readable JSON output.
- Refactor indexing into scan → parse/stage → batch-embed → per-file commit.
- Batch chunks across changed files rather than separately per file. Preserve the mapping from every vector to its file and chunk.
- Bucket inputs by approximate token length to reduce padding and memory waste.
- Preserve existing semantics: a failed file retains its previous chunks, unchanged files are not read, and only fully embedded files are committed.
- On batch failure, halve the batch twice; then bisect to identify a content-specific failure. Treat provider crashes and resource failures as run-level failures eligible for CPU fallback.

Release gate: CPU results are identical, all current tests pass, and warm CPU indexing is not slower than the current implementation by more than 5%.

### Phase 2: Backend Contract and Isolated Worker

- Split the embedding abstraction into query and passage roles while retaining a shared model identity and vector dimension.
- Introduce `BackendDescriptor` and `BackendSelection` types carrying provider, device, stability level, precision, and fallback diagnostics.
- Keep the CPU query model in the MCP process.
- Run passage embedding in a spawned worker using a small IPC protocol: initialize, probe, embed batch, report memory, stop.
- Transfer only text inputs and packed float vectors over IPC; parsing and storage remain in the main process.
- Terminate the worker after indexing to release VRAM or unified memory.
- If an automatic accelerator fails initialization, compilation, or minimum-batch inference, terminate it and retry remaining staged chunks through the CPU backend. Strict mode returns a dedicated backend/resource error instead.
- Cache successful probe and batch calibration data by model artifact, backend/runtime version, OS/architecture, device, and driver.

Release gate: worker crashes and OOMs cannot terminate the MCP server or corrupt an existing index.

### Phase 3: Locked Installation and CUDA

- Remove FastEmbed from unconditional dependencies and define mutually exclusive `cpu`, `cuda`, `webgpu`, and `migraphx` runtime extras. `uv` supports explicitly conflicting extras in one lockfile.
- Keep the main `.venv` on the `cpu` extra for query embeddings and fallback.
- Create a separate accelerator environment under the installation directory using exactly one accelerator extra. Share the existing model cache between environments.
- CUDA uses `fastembed-gpu`; CPU uses `fastembed`. They must remain isolated because the distributions and ONNX Runtime variants conflict.
- Extend installer detection with OS/architecture, NVIDIA driver/runtime checks, wheel availability, and a post-install inference probe. Hardware detection nominates a backend; only the runtime probe confirms it.
- Pin supported CUDA/cuDNN/ONNX Runtime combinations per release. Unsupported combinations fall back to the CPU installation without modifying system drivers.
- Ship CUDA as the first stable automatic accelerator.

Release gates:

- CPU/CUDA vector cosine similarity is at least 0.999 on the golden corpus.
- Search top-k overlap with CPU is at least 99%.
- Forced indexing of a corpus with at least 1,000 chunks is at least 1.25× faster end-to-end.
- Small incremental jobs remain on CPU whenever accelerator startup would make them slower.

### Phase 4: Metal, Vulkan, and AMD

- Implement a direct ONNX backend for providers FastEmbed cannot configure, reusing the same tokenizer, pooling, normalization, and model artifact.
- Add the ONNX Runtime WebGPU plugin backend. It provides the primary cross-platform path:
  - Metal on macOS.
  - Vulkan on Linux.
  - D3D12 or Vulkan on Windows.
- Keep Core ML manual-only initially. The current Jina model benchmark was substantially slower than CPU because only part of the graph was offloaded and dynamic shapes caused excessive partitioning.
- Add MIGraphX for a pinned ROCm/Linux compatibility matrix. Unsupported ROCm versions try WebGPU/Vulkan, then CPU.
- If WebGPU fails the Apple Silicon promotion gates, implement an MLX backend as the designated Metal fallback; do not add PyTorch/MPS unless MLX also fails model-parity requirements.
- Backends progress through `experimental → manual → automatic`. Automatic promotion requires the same correctness and performance gates as CUDA on dedicated hardware runners.

### Phase 5: Adaptive Selection and Hardening

- Calibrate candidate batch sizes and representative sequence lengths once per device/model/runtime combination.
- Record cold-load cost and warm throughput, then calculate a workload crossover threshold.
- After parsing reveals the pending token/chunk count, use CPU below that threshold and the accelerator above it.
- Bound batches by configured memory, item count, and token count. On OOM, shrink and retry; cache the lower safe limit.
- Add local-only structured diagnostics and documentation for selected provider, fallback reason, driver incompatibility, and recommended override.
- Never download accelerator packages or alter drivers while serving MCP requests.
- Provider changes do not invalidate existing indexes when parity gates pass; execution provider and precision are diagnostic metadata, not part of index compatibility.

## Test and Acceptance Plan

- Unit tests for provider ordering, explicit overrides, strict mode, calibration-cache invalidation, adaptive batching, OOM retries, and fallback.
- Indexing tests for cross-file batching, vector-to-chunk ordering, mixed successful/failed files, unchanged files, cancellation, and worker termination.
- Installer tests for CPU-only machines, compatible/incompatible NVIDIA systems, supported/unsupported ROCm, WebGPU-capable platforms, interrupted installs, and CPU rollback.
- Golden-vector tests comparing every backend with CPU for finite values, dimension, normalization, cosine similarity, and search ranking.
- Hardware CI matrix:
  - CPU: Linux, macOS Apple Silicon, Windows.
  - CUDA: supported NVIDIA Linux and Windows runners.
  - WebGPU: Apple Silicon and Linux Vulkan.
  - MIGraphX: supported AMD/ROCm Linux runner.
- Performance results are stored as CI artifacts. Performance regressions block automatic-provider promotion but do not block CPU-only fixes.

## Assumptions and Defaults

- Automatic installation with override is the chosen default because no preference was supplied.
- CPU remains universally supported and is always the failure fallback.
- Acceleration targets passage indexing; query embeddings remain CPU-first for predictable single-query latency.
- The default model, tokenizer, pooling, normalization, vector dimension, and stored embedding text remain unchanged, so acceleration alone requires no reindex or storage migration.
- Core ML remains opt-in until it beats CPU on the exact model and workload; WebGPU is the planned Metal/Vulkan route.
- All benchmarking, device information, and fallback diagnostics remain local with no telemetry.
