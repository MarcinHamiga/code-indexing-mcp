# Phase 4: Direct ONNX WebGPU and MIGraphX Design

## Scope

This Phase 4 direct-ONNX slice makes the existing `webgpu` and `migraphx` backend descriptors executable without
putting a conflicting ONNX Runtime distribution in the serving environment. It preserves the
Phase 2/3 architecture: query embeddings stay on CPU in the MCP process, passage embeddings run
in a disposable worker from a locked accelerator environment, and any failure either degrades to
CPU or raises the existing strict backend error.

This slice does not promote WebGPU or MIGraphX to automatic selection. Both remain experimental
until their correctness and performance gates pass on dedicated hardware. Core ML remains
manual-only. The conditional MLX implementation is explicitly outside this slice: these acceptance
gates decide whether that follow-up is required, and the long-term Phase 4 cannot be marked fully
complete until the contingency is either implemented or superseded by passing WebGPU evidence.

## Chosen approach

Add a small direct ONNX passage model owned by this project. It resolves the same
`jinaai/jina-embeddings-v2-base-code` Hugging Face snapshot FastEmbed uses, loads the same
`onnx/model.onnx`, configures the tokenizer from the same four JSON files, performs attention-mask
mean pooling, and L2-normalizes float32 output. The object exposes the narrow shape the existing
worker needs: `passage_embed`, a tokenizer with `encode`, and the providers that the created
session actually resolved.

The worker selects this direct implementation only for WebGPU and MIGraphX. CPU, CUDA, and Core
ML continue through FastEmbed. Imports become lazy so a direct accelerator environment does not
need FastEmbed and therefore does not accidentally install its CPU ONNX Runtime.

Alternatives rejected:

- Subclassing FastEmbed's internal pooled model would reduce copied preprocessing code but would
  retain the package/runtime conflict and depend on private class layouts.
- A Node/ONNX Runtime Web sidecar would add a second package manager and another IPC boundary while
  the native Python WebGPU plugin now supplies Metal, Vulkan, and D3D12 through the existing worker.

## Runtime and data flow

1. The installer builds one mutually exclusive accelerator extra in `.venv-accel`.
2. `code_indexing_mcp.accelerator_probe` runs in that interpreter.
3. The probe asks the direct backend to prepare the provider:
   - WebGPU registers `onnxruntime-ep-webgpu`, discovers matching EP devices, and attaches them to
     `SessionOptions`.
   - MIGraphX uses the provider built into AMD's pinned ONNX Runtime wheel.
4. The direct backend resolves or reuses the model snapshot in the shared model cache, loads the
   tokenizer configuration, and creates the inference session.
5. The probe embeds the existing minimum corpus and records the resolved provider only after
   vector dimension, finiteness, and normalization checks pass.
6. Runtime selection reads that record. An explicitly requested experimental backend runs through
   the external interpreter launcher and the existing `PassageBackendSession`.
7. Worker startup, provider verification, batch retry, memory accounting, CPU degradation, strict
   mode, telemetry, and worker teardown remain unchanged.

## Packaging and compatibility

The `webgpu` extra uses the Microsoft plugin package and a compatible CPU core runtime:

- `onnxruntime>=1.24.4,<1.25`
- `onnxruntime-ep-webgpu==0.1.0`
- direct model dependencies (`huggingface-hub`, `numpy`, and `tokenizers`)
- wheels on macOS 14+ universal2, Linux x86-64 with glibc 2.27+, and Windows x86-64

The `migraphx` extra pins AMD's ROCm 7.2.1 wheel:

- `onnxruntime-migraphx==1.23.2`
- Linux x86-64, CPython 3.12 only
- direct model dependencies as above

The project supports Python 3.12 and 3.13, but AMD publishes no matching MIGraphX Python 3.13
wheel in the pinned matrix. The installer therefore rejects that combination before syncing.
An explicit unsupported MIGraphX request tries WebGPU when WebGPU has a wheel for the platform,
then reports CPU if its real inference probe also fails. Automatic installation continues to
prepare only providers that are eligible for automatic runtime selection; without CUDA it remains
CPU until WebGPU or MIGraphX passes promotion gates.

All four runtime extras (`cpu`, `cuda`, `webgpu`, and `migraphx`) are mutually exclusive in the uv
lock. Accelerator packages are never installed at request time.

## Failure handling

- Missing/corrupt model artifacts become the existing model-unavailable failure in the worker.
- Plugin registration, device discovery, session creation, compilation, or minimum-batch
  inference failures fail the installer probe and remove the half-built environment.
- A session that silently omits the requested provider is rejected before content is indexed.
- A later provider crash follows the existing run-level CPU fallback. Strict mode forbids it.
- Unsupported platform, Python, or ROCm combinations are diagnosed before a large environment
  download starts.

## Testing and promotion

Unit tests use fake tokenizers and fake ONNX Runtime sessions to pin:

- input IDs, attention masks, and token type IDs;
- attention-mask mean pooling and L2 normalization;
- conventional MIGraphX provider ordering;
- WebGPU plugin registration and EP-device attachment;
- refusal when no requested device/provider resolves;
- worker dispatch without a FastEmbed import.

An opt-in model test compares the direct CPU path with FastEmbed on the golden corpus, requiring
finite normalized vectors, cosine similarity of at least 0.999, and at least 99% top-k overlap.
Provider-gated hardware tests reuse the same assertions for WebGPU or MIGraphX and exercise the
existing end-to-end benchmark. They gather promotion evidence but do not make either backend
automatic in this phase.
