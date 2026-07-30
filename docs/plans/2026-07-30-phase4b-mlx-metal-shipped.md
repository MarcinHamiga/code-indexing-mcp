# Phase 4B MLX Metal Backend: Shipped

## Outcome

The MLX contingency the long-term plan designated for Apple Silicon is implemented, measured, and
promoted. Passage indexing on macOS 14+ arm64 now runs on Metal through MLX, from a locked
installer environment, using the existing external worker, probe, fallback, memory, and telemetry
architecture. MLX and CUDA are the two accelerators eligible for `auto`; WebGPU and MIGraphX remain
experimental and explicit-only, and Core ML remains manual-only.

This closes the conditional item Phase 4A left open. The plan's PyTorch/MPS escape hatch is not
needed: it applied only if MLX also failed model parity, and MLX matched the CPU model exactly.

No index migration is required. The model ID, tokenizer, attention-mask mean pooling, float32 L2
normalization, 768-dimensional vector shape, and stored embedding text are unchanged.

## Delivered

- A project-owned MLX passage model that reproduces the pinned JinaBERT v2 graph — ALiBi, post-norm
  query and key projections, GEGLU feed-forward — from the float32 initializers of the same ONNX
  artifact the CPU path loads.
- A one-time, revision-keyed weight conversion written atomically and memory-mapped thereafter,
  with configuration, tensor-name, and shape guards that refuse an artifact whose graph moved.
- A `Runtime` dimension on the backend contract, so a backend that is not an ONNX execution provider
  stops being special-cased by name at each site: no ONNX CPU provider is placed behind it, its
  target is not required to be published before loading, and its environment reports no execution
  providers it does not have.
- Worker routing that keeps CPU, CUDA, and Core ML on FastEmbed, WebGPU and MIGraphX on the direct
  ONNX model, and MLX on its own — without importing FastEmbed into an MLX environment.
- A locked `mlx` extra, mutually exclusive with `cpu`, `cuda`, `webgpu`, and `migraphx`, and
  installer detection restricted to the platform where MLX has both a wheel and a Metal device.
- Unit coverage of the forward pass against an independent NumPy implementation of the same graph,
  plus the opt-in dedicated-hardware correctness and performance gates.

## Locked matrix

| Backend | Supported combination | Locked packages | Selection |
| --- | --- | --- | --- |
| MLX | macOS 14+ Apple Silicon | MLX 0.32.0, ONNX 1.22 (conversion only) | automatic |

MLX also publishes CPU-only Linux and Windows wheels. They are excluded by marker: preparing them
would build a "Metal" environment with no Metal in it, which would pass its own probe and then lose
to the CPU it really is. An unsupported explicit `mlx` request reports CPU rather than degrading to
WebGPU.

## Acceptance evidence

An installer-shaped locked MLX environment on an Apple M4 Pro running macOS 26.5.2 passed the real
two-passage probe (`resolved_providers: ["MlxMetalBackend"]`, MLX 0.32.0, device `metal`) and both
gates against the corpus Phase 4A used:

- corresponding-row cosine similarity: 1.0, maximum absolute deviation 2.8e-7;
- CPU-query top-5 overlap: 100%;
- finite, normalized float32 vectors of dimension 768.

| Backend | Wall time (1,000 chunks) | Relative speed | Peak memory |
| --- | ---: | ---: | ---: |
| CPU | 19.874 s / 20.152 s / 20.192 s | 1.00× | 1.68 GB |
| WebGPU (Phase 4A) | 17.555 s | 1.11× | — |
| MLX | 13.116 s / 12.920 s / 12.943 s | 1.52–1.56× | 1.26 GB |

The promotion gate is 1.25×, so MLX was promoted to `AUTOMATIC` and `--accelerator auto` now
prepares it on a supported Mac. The evidence is one machine rather than a matrix of Apple Silicon
runners; reinstalling with `--accelerator cpu`, or setting `INCODE_EMBED_ACCELERATOR=cpu`, is the way
back on a Mac where it does not hold.

## Verification commands

```bash
uv lock --check
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy src
uv run --extra cpu mypy scripts/benchmark_index_memory.py
git diff --check

# MLX's own unit tests need MLX, which the serving environment does not have.
uv run --extra mlx pytest tests/test_mlx_backend.py
```

Real-model and hardware gates:

```bash
INCODE_MODEL_TEST_CACHE=/path/to/cache uv run --extra cpu pytest -m model

INCODE_MODEL_TEST_CACHE=/path/to/cache \
INCODE_ACCEL_ENV=/path/to/accelerator.json \
INCODE_TEST_ACCELERATOR=mlx \
  uv run --extra cpu pytest -m accelerator
```

## What is left in the long-term plan

- WebGPU and MIGraphX still need their own promotion evidence: WebGPU on Linux/Vulkan and Windows,
  MIGraphX on the pinned AMD/ROCm matrix.
- Phase 5, adaptive selection and hardening, is untouched. Its workload crossover threshold matters
  more now that a Mac prepares an accelerator by default: a small incremental job should stay on CPU
  whenever starting the MLX worker would make it slower, and today only the fixed batch bounds and
  the memory ceiling limit a run.
