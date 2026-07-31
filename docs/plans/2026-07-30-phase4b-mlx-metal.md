# Phase 4B MLX Metal Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the MLX Metal passage backend the long-term plan designated as the Apple Silicon
fallback after WebGPU missed its 1.25× promotion gate, and measure it against the same gates.

**Architecture:** Reproduce the pinned JinaBERT v2 graph in MLX from the float32 initializers of the
ONNX artifact the CPU path already uses, execute it in the existing disposable external worker, and
teach the descriptor contract that a backend's runtime is not always ONNX Runtime. Probe, fallback,
strict mode, memory accounting, telemetry, and teardown stay authoritative.

**Tech Stack:** Python 3.12/3.13, MLX, ONNX (weight extraction only), Hugging Face Hub, tokenizers,
NumPy, uv, pytest.

### Task 1: A runtime dimension on the backend contract

**Files:**

- Modify: `src/code_indexing_mcp/backends.py`
- Modify: `src/code_indexing_mcp/accelerator_probe.py`
- Modify: `src/code_indexing_mcp/application.py`
- Modify: `tests/test_backends.py`
- Modify: `tests/test_accelerator_probe.py`

**Step 1: Write failing tests**

Assert that an MLX descriptor exists, is experimental, is not eligible for `auto`, is honoured when
requested explicitly against a record that offers `MlxMetalBackend`, and that its `providers` tuple
does not append an ONNX CPU provider. Assert the probe does not require an MLX provider to appear in
`onnxruntime.get_available_providers()`, still rejects an MLX model that reports no providers, and
reports MLX's own runtime version.

**Step 2: Verify RED**

```bash
uv run --extra cpu pytest tests/test_backends.py tests/test_accelerator_probe.py -q
```

**Step 3: Implement**

- Add `Runtime` (`onnxruntime`, `onnxruntime-plugin`, `mlx`) and `BackendDescriptor.runtime`.
- Add `Accelerator.MLX`, `MLX_PROVIDER`, and the experimental descriptor.
- Replace the probe's WebGPU-by-name exemption with `descriptor.provider_is_preregistered`, and its
  `{WEBGPU, MIGRAPHX}` set with `descriptor.uses_direct_model`.
- Make `runtime_version` answer for a given runtime and let the probe key prefer the runtime version
  the prepared environment recorded.

**Step 4: Verify GREEN**

```bash
uv run --extra cpu pytest tests/test_backends.py tests/test_accelerator_probe.py \
  tests/test_application.py tests/test_passage_backend.py -q
uv run --extra cpu mypy src
```

### Task 2: The MLX passage model

**Files:**

- Create: `src/code_indexing_mcp/mlx_backend.py`
- Create: `tests/test_mlx_backend.py`
- Modify: `pyproject.toml`

**Step 1: Write failing tests**

Against a synthesized miniature ONNX graph with the same node and initializer names, assert that
extraction finds every weight and the ALiBi slopes, that a renamed node or a wrong shape is refused,
that a config missing one of the five architectural facts is refused, that conversion writes once
and is reused, and that the forward pass matches an independent NumPy implementation of the same
graph to float32 tolerance — including masked padding and mask-mean pooling.

**Step 2: Verify RED**

```bash
uv run --extra mlx pytest tests/test_mlx_backend.py -q
```

**Step 3: Implement**

`read_model_config`, `extract_weights`, `ensure_converted_weights` (atomic, revision-keyed),
`JinaBertMlx`, `mask_mean_normalize`, and `MlxEmbedding` exposing `passage_embed`, `tokenizer`, and
`resolved_providers`. Reuse `resolve_model_snapshot` and `load_tokenizer` from `direct_onnx`. Keep
MLX and ONNX imports inside construction paths.

**Step 4: Verify GREEN**

```bash
uv run --extra mlx pytest tests/test_mlx_backend.py -q
uv run --extra cpu pytest -q
uv run --extra cpu mypy src
```

### Task 3: Worker routing

**Files:**

- Modify: `src/code_indexing_mcp/embedding_worker.py`
- Modify: `tests/test_embedding_worker.py`

**Step 1: Write a failing routing test**

Extend the direct-accelerator parametrisation so `mlx` loads `MlxEmbedding` with the worker's cache
directory, offline flag, model id, and accelerator, and fails if FastEmbed is imported.

**Step 2-4: Verify RED, dispatch on the accelerator, verify GREEN**

```bash
uv run --extra cpu pytest tests/test_embedding_worker.py tests/test_indexing_backend.py -q
```

### Task 4: Locked installation

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `install.py`
- Modify: `tests/test_installer.py`

**Step 1: Write failing installer tests**

Cover nomination on macOS 14+ arm64, rejection on Intel macOS, older macOS, Linux, and Windows,
rejection reporting CPU rather than falling through to WebGPU, the `mlx` extra reaching locked `uv
sync`, the macOS version recorded as the driver version, and `auto` still ignoring MLX.

**Step 2-4: Verify RED, add the extra and detection, regenerate the lock, verify GREEN**

```bash
uv lock && uv lock --check
uv run --extra cpu pytest tests/test_installer.py tests/test_accelerator_env.py -q
```

### Task 5: Acceptance gates and real hardware

**Files:**

- Modify: `tests/test_accelerator_acceptance.py`

**Step 1: Accept `mlx` in the gate suite**

`CODE_INDEXING_TEST_ACCELERATOR=mlx` must exercise the same parity and 1,000-chunk performance gates
through a prepared environment.

**Step 2: Run them on Apple Silicon**

Build a locked MLX environment, probe it, then run the parity and performance gates and record the
numbers.

### Task 6: Documentation and full verification

**Files:**

- Modify: `README.md`
- Create: `docs/plans/2026-07-30-phase4b-mlx-metal-shipped.md`

```bash
uv lock --check
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy src
uv run --extra cpu mypy scripts/benchmark_index_memory.py
git diff --check
```

Record the measured gate outcome, and whether it clears the 1.25× promotion threshold, in the
shipped document rather than deciding promotion in advance.
