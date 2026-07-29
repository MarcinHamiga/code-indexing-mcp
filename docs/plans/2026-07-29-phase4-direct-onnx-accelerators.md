# Phase 4 Direct ONNX Accelerators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the experimental WebGPU and MIGraphX passage backends installable and executable
through the existing isolated-worker architecture while preserving exact CPU model semantics.

**Architecture:** Add a project-owned direct ONNX model for providers FastEmbed cannot configure.
Dispatch WebGPU and MIGraphX workers to it, keep every other backend on FastEmbed, and extend the
locked installer to prepare only supported platform/runtime combinations. Existing probe,
fallback, strict-mode, telemetry, and teardown machinery stays authoritative.

**Tech Stack:** Python 3.12/3.13, ONNX Runtime plugin EP API, ONNX Runtime MIGraphX, Hugging Face
Hub, tokenizers, NumPy, uv, pytest.

### Task 1: Direct ONNX model parity

**Files:**

- Create: `tests/test_direct_onnx.py`
- Create: `src/incode_mcp/direct_onnx.py`
- Modify: `pyproject.toml`

**Step 1: Write failing tests**

Add focused tests that construct the model around fakes and assert:

```python
vectors = list(model.passage_embed(["short", "a longer passage"]))
assert vectors[0].dtype == np.float32
assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)
assert session.last_inputs["input_ids"].dtype == np.int64
assert session.last_inputs["attention_mask"].shape == (2, padded_width)
```

Also test snapshot resolution in offline mode, tokenizer truncation/padding setup, optional
`token_type_ids`, provider reporting, and zero-norm-safe normalization.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_direct_onnx.py -q
```

Expected: collection fails because `incode_mcp.direct_onnx` does not exist.

**Step 3: Implement the minimum direct model**

Implement:

- constants for the one supported model and its `onnx/model.onnx` artifact;
- `resolve_model_snapshot(cache_directory, offline, model_id)`;
- JSON-driven tokenizer setup matching FastEmbed;
- `mean_pool_and_normalize(output, attention_mask)`;
- `DirectOnnxEmbedding` with `passage_embed`, `tokenizer`, and `resolved_providers`;
- injectable artifact/session/tokenizer seams for unit tests.

Keep third-party imports inside construction paths so the serving process does not load an
accelerator runtime.

**Step 4: Verify GREEN**

Run the direct-model tests, then:

```bash
uv run pytest tests/test_embedding.py tests/test_direct_onnx.py -q
uv run ruff check src/incode_mcp/direct_onnx.py tests/test_direct_onnx.py
uv run mypy src
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/incode_mcp/direct_onnx.py tests/test_direct_onnx.py pyproject.toml
git commit -m "feat: add direct ONNX passage model"
```

### Task 2: Worker, probe, and selection integration

**Files:**

- Modify: `src/incode_mcp/embedding.py`
- Modify: `src/incode_mcp/embedding_worker.py`
- Modify: `src/incode_mcp/accelerator_probe.py`
- Modify: `src/incode_mcp/backends.py`
- Modify: `src/incode_mcp/application.py`
- Modify: `tests/test_embedding_worker.py`
- Modify: `tests/test_backends.py`
- Modify: `tests/test_application.py`
- Create: `tests/test_accelerator_probe.py`

**Step 1: Write failing routing tests**

Assert that:

- CPU/CUDA/Core ML choose the lazy FastEmbed loader;
- WebGPU/MIGraphX choose `DirectOnnxEmbedding`;
- plugin provider discovery is recorded and verified;
- a prepared experimental environment is selected only when explicitly requested;
- `auto` still selects only CUDA;
- direct accelerator imports work when `fastembed` is unavailable.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_embedding_worker.py tests/test_accelerator_probe.py \
  tests/test_backends.py tests/test_application.py -q
```

Expected: new routing and plugin-discovery assertions fail.

**Step 3: Implement routing**

- Move FastEmbed imports behind `TYPE_CHECKING` or function scope.
- Make `_load_model` dispatch on `WorkerConfig.accelerator`.
- Teach provider resolution to read `resolved_providers` from the direct model.
- Let the accelerator probe prepare plugin providers before checking availability.
- Preserve `WorkerConfig`, IPC messages, packed-vector format, and fallback callbacks.
- Keep WebGPU/MIGraphX at `Stability.EXPERIMENTAL`.

**Step 4: Verify GREEN**

Run the targeted tests above plus:

```bash
uv run pytest tests/test_passage_backend.py tests/test_indexing_backend.py -q
uv run ruff check src tests
uv run mypy src
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/incode_mcp tests
git commit -m "feat: route experimental providers through direct ONNX"
```

### Task 3: Locked WebGPU and MIGraphX installation

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `install.py`
- Modify: `tests/test_installer.py`
- Modify: `tests/test_accelerator_env.py`

**Step 1: Write failing installer tests**

Cover:

- supported WebGPU wheel platforms;
- WebGPU rejection on unsupported platforms;
- MIGraphX acceptance only on Linux x86-64, Python 3.12, ROCm 7.2.1;
- unsupported MIGraphX falling through WebGPU to CPU;
- the selected extra passed to locked uv sync;
- record reuse invalidated by changed ROCm/runtime details;
- probe failure removes the environment and record.

**Step 2: Verify RED**

Run:

```bash
uv run pytest tests/test_installer.py tests/test_accelerator_env.py -q
```

Expected: WebGPU/MIGraphX plans still report that no locked installation exists.

**Step 3: Implement packaging and detection**

- Add mutually exclusive direct-runtime extras.
- Pin the Microsoft WebGPU plugin/core pair.
- Pin the AMD ROCm 7.2.1/MIGraphX wheel for CPython 3.12.
- Extend `AcceleratorPlan` with the internal extra when it differs from the public name.
- Add bounded stdlib-only ROCm detection.
- Prefer supported explicit MIGraphX, then supported WebGPU, then CPU.
- Preserve CUDA-first automatic detection and never install an experimental backend for `auto`.

**Step 4: Regenerate and verify the lock**

Run:

```bash
uv lock
uv lock --check
uv run pytest tests/test_installer.py tests/test_accelerator_env.py -q
```

Expected: the universal lock resolves all mutually exclusive environments and tests pass.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock install.py tests/test_installer.py tests/test_accelerator_env.py
git commit -m "feat: install locked WebGPU and MIGraphX runtimes"
```

### Task 4: Model and hardware acceptance gates

**Files:**

- Modify: `tests/test_model_integration.py`
- Create: `tests/test_accelerator_acceptance.py`
- Modify: `pyproject.toml`

**Step 1: Write parity tests**

Using the existing opt-in model cache:

```python
cpu = np.asarray(list(fastembed.passage_embed(GOLDEN_TEXTS)))
direct = np.asarray(list(direct_onnx.passage_embed(GOLDEN_TEXTS)))
assert min(cosine_rows(cpu, direct)) >= 0.999
assert top_k_overlap(cpu, direct) >= 0.99
```

Add provider-gated tests selected by `INCODE_TEST_ACCELERATOR` so WebGPU/MIGraphX runners exercise
the same vectors and a 1,000-chunk end-to-end benchmark without failing ordinary CPU CI.

**Step 2: Verify RED and GREEN**

Run first to observe missing acceptance helpers, implement them, then run:

```bash
INCODE_MODEL_TEST_CACHE=<shared-cache> uv run pytest -m model -q
uv run pytest tests/test_accelerator_acceptance.py -q
```

Expected locally: direct CPU parity passes; hardware cases skip with an explicit reason unless a
provider environment is configured.

**Step 3: Commit**

```bash
git add tests/test_model_integration.py tests/test_accelerator_acceptance.py pyproject.toml
git commit -m "test: add accelerator parity and promotion gates"
```

### Task 5: Documentation and full verification

**Files:**

- Modify: `README.md`
- Create: `docs/plans/2026-07-29-phase4-direct-onnx-accelerators-shipped.md`

**Step 1: Document installation and diagnostics**

Describe explicit WebGPU/MIGraphX installation, the supported platform/runtime matrix, Metal /
Vulkan / D3D12 mapping, experimental status, CPU fallback, strict mode, and promotion commands.

**Step 2: Run all verification**

```bash
uv lock --check
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mypy scripts/benchmark_index_memory.py
git diff --check
```

Expected: all commands pass; only documented opt-in hardware tests skip.

**Step 3: Real WebGPU smoke test**

On a supported host, build a temporary locked WebGPU environment, run
`python -m incode_mcp.accelerator_probe --accelerator webgpu`, and confirm the report names the
WebGPU provider and returns normalized 768-dimensional vectors. If the host does not expose a
WebGPU device, retain the diagnostic and verify CPU rollback.

**Step 4: Commit**

```bash
git add README.md docs/plans
git commit -m "docs: describe phase 4 accelerator support"
```

**Step 5: Final branch review**

Review `git diff main...HEAD`, rerun the verification suite after any cleanup, and use
`superpowers:requesting-code-review` followed by `superpowers:finishing-a-development-branch`.
