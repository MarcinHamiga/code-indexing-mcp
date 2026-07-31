# Phase 5 Adaptive Selection and Hardening Implementation Plan

**Goal:** Close the long-term plan's Phase 5 — calibrate each backend once per configuration, use
CPU for work too small to repay an accelerator's startup, bound batches by memory as well as count,
remember the limit an overrun found, and report all of it locally.

**Architecture:** Measure through the worker protocol that already exists, store in the probe cache
that already keys by configuration, and decide mid-run because the indexing pipeline streams. See
`2026-07-30-phase5-adaptive-selection-design.md`.

**Tech Stack:** Python 3.12/3.13, pytest, existing worker/probe/backend machinery.

### Task 1: Memory-bounded microbatches

**Files:** `src/code_indexing_mcp/token_batching.py`, `src/code_indexing_mcp/application.py`,
`tests/test_token_batching.py`, `tests/test_application.py`

Add `max_token_product_for(memory_bytes)`: the default product scaled by the ceiling it is measured
against, floored at one longest window and capped. Wire it into the `SegmentPlan` the application
builds. Assert the default ceiling reproduces today's 4,096 exactly, that a halved ceiling halves
the product, and that a 1 GiB floor still admits a single max-length sequence.

### Task 2: Calibration measurement

**Files:** create `src/code_indexing_mcp/calibration.py`, create `tests/test_calibration.py`

- `CalibrationResult(max_items, characters_per_second, load_ns, limited_by)`.
- `calibration_candidates(...)` — a deterministic, code-shaped synthetic corpus at two
  representative lengths.
- `calibrate(session, plan, *, batch_sizes, ...)` — time `plan_and_embed` at each candidate size
  through the session, stop early once throughput stops improving, treat a resource error at a size
  as that size being unsafe and keep the last safe one, and never let a calibration failure escape
  as anything but "not calibrated".
- `crossover_characters(load_ns, cpu_rate, accelerator_rate)` — the formula, `None` when the
  accelerator is not faster.

Tests drive a fake session with programmed per-size timings: the fastest size wins, a plateau stops
the sweep early, a size that raises `INDEX_RESOURCE_LIMIT` is refused and the previous one kept, a
session that fails outright yields no calibration rather than an exception, and the crossover
formula matches hand-computed values including the no-crossover case.

### Task 3: Persisting calibration

**Files:** `src/code_indexing_mcp/probe_cache.py`, `tests/test_probe_cache.py`

Extend `ProbeRecord` with `characters_per_second`, `load_ns`, and `limited_by`; bump
`CACHE_SCHEMA_VERSION` to 2. Assert a version-1 file is ignored, a round trip preserves the new
fields, and a record missing them is still readable within version 2 only if it validates.

### Task 4: Cold-load cost and the reduced safe limit

**Files:** `src/code_indexing_mcp/embedding_worker.py`, `tests/test_embedding_worker.py`

Record `load_duration_ns` across spawn plus `initialize`, and `safe_max_items` when a retry
succeeds at a reduced size. Assert both, including that a first-attempt success leaves
`safe_max_items` unset.

### Task 5: Calibrating and crossing over in the passage session

**Files:** `src/code_indexing_mcp/passage_backend.py`, `tests/test_passage_backend.py`

- Calibrate on a probe miss, store the result, and use a cached one on a hit.
- Accept `crossover_characters`; embed on CPU below it, switch to the accelerator on the request
  that crosses it, and keep the existing verification and degradation behaviour for that switch.
- Report the backend actually used, the characters embedded, and a selection reason; never count a
  deferral as a fallback.
- Write a reduced `safe_max_items` back to the cache.

Tests: a small run never spawns the accelerator; a run that crosses does, exactly once; strict mode
is unaffected by deferral; a deferred run reports no fallback; an accelerator slower than CPU is
never used; a reduced batch size reaches the cache.

### Task 6: Settings, wiring, and diagnostics

**Files:** `src/code_indexing_mcp/settings.py`, `src/code_indexing_mcp/application.py`,
`src/code_indexing_mcp/models.py`, `src/code_indexing_mcp/indexing.py`, and their tests

- `CODE_INDEXING_EMBED_CROSSOVER=auto|off|<characters>` and `CODE_INDEXING_EMBED_CALIBRATE=0|1`.
- `Application` computes the crossover from the CPU and accelerator records and passes it down.
- `ModelStatus` gains the rates, cold-load milliseconds, crossover, and a recommended override;
  `batch_calibration` gains `measured` and `reduced`.
- `IndexReport` gains `embedded_characters`, `embedding_crossover_characters`, and
  `embedding_selection_reason`.

### Task 7: Documentation and full verification

**Files:** `README.md`, create `docs/plans/2026-07-30-phase5-adaptive-selection-shipped.md`

```bash
uv lock --check
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu ruff format --check .
uv run --extra cpu mypy src
uv run --extra cpu mypy scripts/benchmark_index_memory.py
git diff --check
uv run --extra mlx pytest tests/test_mlx_backend.py
```

Record measured crossover and calibration numbers from this machine in the shipped document.
