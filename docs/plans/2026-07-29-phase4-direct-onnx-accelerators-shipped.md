# Phase 4A Direct ONNX Accelerators: Shipped

## Outcome

The direct-ONNX accelerator slice is implemented. WebGPU and MIGraphX now have locked,
mutually-exclusive installer environments and execute passage embeddings through the existing
external worker, probe, fallback, memory, and telemetry architecture. Both remain experimental and
explicit-only; CUDA remains the only automatic accelerator.

This records the narrowed Phase 4A slice, not completion of every conditional item in the long-term
Phase 4. In particular, the MLX contingency is a separate implementation slice whose trigger is
measured here.

No index migration is required. The default model ID, tokenizer, ONNX artifact, attention-mask
mean pooling, float32 L2 normalization, 768-dimensional vector shape, and stored embedding text are
unchanged.

## Delivered

- A project-owned direct ONNX passage model with lazy third-party imports and shared Hugging Face
  cache resolution.
- Native WebGPU plugin registration, EP-device discovery, device attachment, and resolved-provider
  verification.
- Conventional MIGraphX provider ordering through AMD's pinned ONNX Runtime wheel.
- Worker routing that keeps CPU, CUDA, and Core ML on FastEmbed while WebGPU/MIGraphX avoid a
  FastEmbed runtime conflict.
- Locked `webgpu` and `migraphx` extras alongside the existing mutually-exclusive `cpu` and `cuda`
  extras.
- Installer detection and fallback for the exact supported platform/runtime matrix.
- Real-model vector and ranking parity gates plus opt-in dedicated-hardware correctness and
  performance gates.

## Locked matrix

| Backend | Supported combination | Locked packages |
| --- | --- | --- |
| WebGPU | macOS 14+ Apple Silicon | ONNX Runtime 1.24.4, WebGPU plugin 0.1.0 |
| WebGPU | Linux x86-64, glibc 2.27+ | ONNX Runtime 1.24.4, WebGPU plugin 0.1.0 |
| WebGPU | Windows x86-64 | ONNX Runtime 1.24.4, WebGPU plugin 0.1.0 |
| MIGraphX | Linux x86-64, CPython 3.12, ROCm 7.2.1 | AMD ONNX Runtime/MIGraphX 1.23.2 |

The WebGPU plugin uses Metal on macOS, Vulkan on Linux, and D3D12 or Vulkan on Windows. An
unsupported MIGraphX request tries WebGPU when its complete locked pair is available, then CPU.
Automatic installation still considers only CUDA.

## Acceptance evidence

The real cached `jinaai/jina-embeddings-v2-base-code` model passed CPU/direct-ONNX parity locally:

- corresponding-row cosine similarity: at least 0.999;
- CPU-query top-5 overlap: at least 99%;
- finite, normalized float32 vectors with dimension 768.

A fresh installer-built WebGPU environment on an Apple M4 Pro running macOS 26.5.2 passed the real
two-passage probe and the same vector/ranking parity gate. Its 1,000-chunk forced-index result was:

| Backend | Wall time | Relative speed |
| --- | ---: | ---: |
| CPU | 19.498 s | 1.00× |
| WebGPU | 17.555 s | 1.11× |

The required promotion gate is 1.25×. WebGPU therefore remains experimental and is not eligible
for `auto`. This failed Apple Silicon performance gate activates the long-term plan's MLX
contingency as a follow-up slice; no incompatible model change or unmeasured automatic promotion is
made here. MIGraphX remains provider-gated until the acceptance suite runs on the pinned AMD/ROCm
hardware matrix.

## Verification commands

```bash
uv lock --check
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy src
uv run --extra cpu mypy scripts/benchmark_index_memory.py
git diff --check
```

Real-model and hardware gates:

```bash
INCODE_MODEL_TEST_CACHE=/path/to/cache uv run --extra cpu pytest -m model

INCODE_MODEL_TEST_CACHE=/path/to/cache \
INCODE_ACCEL_ENV=/path/to/accelerator.json \
INCODE_TEST_ACCELERATOR=webgpu \
  uv run --extra cpu pytest -m accelerator
```
