# CPU Passage Thread Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve raw CPU indexing throughput by selecting a measured passage-worker thread count while preserving explicit settings and indexing correctness.

**Architecture:** Add a bounded CPU-only thread tuner and its own configuration-aware cache. Invoke it from the existing passage-worker lifecycle for sufficiently large CPU-only runs; apply compatible cached choices immediately. Preserve accelerator calibration, token batching, query threading, and storage behavior.

**Tech Stack:** Python 3.12/3.13, FastEmbed, ONNX Runtime, psutil, existing multiprocessing workers, pytest.

**Spec:** [CPU passage thread tuning design and evidence](../specs/2026-09-07-cpu-passage-thread-tuning-design.md). Read it first: it defines behavior, exclusions, experimental limits, and the rationale for this PR.

## Global Constraints

- Python >=3.12,<3.14; no new runtime dependencies.
- No model, tokenizer, prefix, window-boundary, vector-dimension, precision, storage-schema, or retrieval-contract change.
- Preserve bounded worker memory, disposal, cancellation, per-file failure retention, and transactional commit/recovery.
- Explicit thread counts 1..64 remain authoritative.
- Calibration never adds indexed chunks, reference rows, or user-file errors.
- At most one passage model is resident during tuning and indexing.
- Benchmark fresh indexes and forced rebuilds as well as no-op and small edits.
- Run the repository formatter, lint, typecheck, and full CI test gate before push.

---

## Scope, base, and success criteria

Proposed PR title: **perf: tune CPU passage inference threads for indexing**.

Base analyzed: `dfdc6d0fa0a7ee1daabde20c7912d261fefc92fe`. Worktree:
`.worktrees/indexing-speed-plan`, branch `plan/indexing-speed`. Only this plan,
its design, and analysis evidence have been added; no runtime implementation is
included. Recheck the base if targeting a newer or different main branch.

Preservation update: these artifacts are being committed directly to `main`
on top of `e9768f7`, as requested. The embedding, worker, token-batching,
calibration, settings, passage-backend, and backend-coordinator source files
are identical between that main revision and the analyzed revision. Main also
contains newer structural-reference work, so the historical benchmark numbers
remain attributed to the original analyzed revision rather than relabeled as
measurements of main. The original configure-wizard checkout is untouched.

Priority evidence: a recorded full CPU build spent 96.9% of wall time embedding.
The configuration experiment using four rather than two threads yielded 1.44×
end-to-end throughput on 44 real-source chunks. The automatic policy itself is
unmeasured. Release gates below determine whether it should ship.

Target release result on the measured x86 host: at least **1.25× median index
throughput** with a compatible tuning cache on a fresh >=1,000-chunk corpus and
a forced rebuild. A first run including calibration must improve wall time by
at least 10% on the large corpus. Other tested hosts must not regress by >5%;
no-op and small-edit median/p95 must not regress by >5% or 20 ms, whichever is
larger. These are proposed acceptance thresholds, not existing measured results.

## File map

| File | Responsibility |
| --- | --- |
| New `src/code_indexing_mcp/cpu_thread_tuning.py` | Pure trial selection, effective CPU allowance, bounded CPU tuning orchestration. |
| New `src/code_indexing_mcp/cpu_thread_cache.py` | Versioned tuning-key/record types, validated bounded persistence. |
| `src/code_indexing_mcp/settings.py:128`, `:176` | Auto-thread setting while retaining the resolved integer. |
| `src/code_indexing_mcp/installer/settings_spec.py:184` | Shared auto/int validation and explanatory text. |
| `src/code_indexing_mcp/embedding_worker.py:324`, `:395`, `:456` | Optional absolute deadline for tuning requests; existing calls retain defaults. |
| `src/code_indexing_mcp/backend_coordinator.py:298` | Per-run CPU-only tuner wiring and factories parameterized by thread count. |
| `src/code_indexing_mcp/passage_backend.py:241`, `:302`, `:349` | Trigger, close/reopen, same-run choice, separate tuning diagnostics. |
| `src/code_indexing_mcp/models.py`, `indexing.py:238`, `backend_coordinator.py:372` | Defaulted reporting fields and status output. |
| `scripts/benchmark_cpu_threads.py` | Reproducible real-worker and full-index benchmark, derived from analysis probes. |
| Corresponding new/existing tests; `README.md:835` | Policy/lifecycle/cache tests, release gates, user configuration documentation. |

## Task 1: Define auto-thread settings, choice policy, and cache contract

**Files:** Create `cpu_thread_tuning.py`, `cpu_thread_cache.py`,
`tests/test_cpu_thread_tuning.py`, `tests/test_cpu_thread_cache.py`; modify
`settings.py`, `installer/settings_spec.py`, `tests/test_settings.py`,
`tests/test_installer_settings_spec.py`.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class CpuThreadTrial:
    threads: int
    samples_ns: tuple[int, int, int]

@dataclass(frozen=True)
class CpuThreadChoice:
    threads: int
    measured_ns: int
    reason: str

def thread_candidates(effective_cpus: int, baseline_threads: int) -> tuple[int, ...]:
    allowed = max(1, effective_cpus)
    return tuple(sorted({min(baseline_threads, allowed),
                         *(n for n in (1, 2, 4, 8) if n <= allowed)}))

def select_threads(
    trials: Sequence[CpuThreadTrial], *, baseline_threads: int
) -> CpuThreadChoice:
    valid = [t for t in trials if all(n > 0 for n in t.samples_ns)]
    baseline = next((t for t in valid if t.threads == baseline_threads), None)
    if baseline is None:
        return CpuThreadChoice(baseline_threads, 0, "baseline_unmeasured")
    baseline_ns = int(statistics.median(baseline.samples_ns))
    best = min(valid, key=lambda t: (statistics.median(t.samples_ns), t.threads))
    best_ns = int(statistics.median(best.samples_ns))
    if best_ns > baseline_ns * 0.90:
        return CpuThreadChoice(baseline_threads, baseline_ns, "no_material_gain")
    return CpuThreadChoice(best.threads, best_ns, "measured_gain")
```

Import `dataclass`, `Sequence`, and `statistics` in the new policy module.
Run baseline first, then remaining candidates; sorted return order above is
the candidate set, not the execution order. Clamp only the automatic baseline
to effective allowance; explicit integers bypass this policy entirely.

- [ ] Add red settings tests using `IndexSettings.from_environment({})`,
  `{"CODE_INDEXING_EMBED_THREADS": "auto"}`, `"4"`, `"0"`, and `"65"`.
  Unset/auto set `embedding_threads_auto=True` and retain conservative integer
  resolution; explicit four gives `(4, False)`; invalid integers raise the
  existing configuration error. Verify direct dataclass construction retains
  the new flag's `False` default.
- [ ] Add policy tests, including a noisy single sample that must not win:

```python
def test_choice_requires_a_material_median_gain():
    trials = [CpuThreadTrial(2, (100, 100, 100)),
              CpuThreadTrial(4, (1, 98, 99))]
    assert select_threads(trials, baseline_threads=2).threads == 2

def test_choice_uses_a_repeatable_gain():
    trials = [CpuThreadTrial(2, (100, 101, 99)),
              CpuThreadTrial(4, (70, 69, 71)),
              CpuThreadTrial(8, (70, 71, 69))]
    assert select_threads(trials, baseline_threads=2).threads == 4

def test_thread_candidates_respect_cpu_allowance():
    assert thread_candidates(1, 2) == (1,)
    assert thread_candidates(3, 2) == (1, 2)
```

- [ ] Define `CpuThreadTuningKey` as a frozen dataclass with `model_id`,
  `model_artifact`, `runtime_version`, `fastembed_version`, `cpu_identity`,
  `effective_cpus`, `cpu_arena`, `memory_bytes`, `max_tokens`, `overlap_tokens`,
  `max_token_product`, `max_items`, and `policy_version=1`. Fingerprint canonical
  sorted JSON with SHA-256. Every field must affect the key.
- [ ] Define `CpuThreadTuningRecord(key_fingerprint: str, choice:
  CpuThreadChoice, recorded_at_ns: int)` and `CpuThreadTuningCache(path: Path)`
  with `load(key) -> CpuThreadTuningRecord | None` and `store(key, choice) ->
  None`. Reuse the atomic-temp-file/lock pattern from `probe_cache.py:226`,
  without changing `ProbeKey` or the accelerator-probe file.
- [ ] Add cache tests for round-trip, each key component changing, invalid
  thread values, nonfinite/negative timing values, malformed/truncated JSON,
  unknown schema, atomic replacement, and newest-64 retention. Treat unreadable
  or invalid records as misses. Persist a measured baseline choice too.
- [ ] Implement the setting parser and shared installer `auto_int` field;
  remove its old numeric dynamic default and display `auto`. Preserve explicit
  saved integers during configure/update. Cover installer normalization of
  `AUTO`, range validation, and preservation of a saved `2`.
- [ ] Implement effective CPU allowance from CPU count, supported affinity,
  and readable Linux cgroup quota, with tests using injected filesystem/CPU
  observations. Use the minimum positive allowance, floor at one. Account for
  cgroup v2 `cpu.max` and v1 quota/period; unavailable data is not fatal.
- [ ] Run the focused tests, format/lint the edits, and commit the independently
  testable policy/cache/settings change:

```bash
uv run pytest tests/test_cpu_thread_tuning.py tests/test_cpu_thread_cache.py tests/test_settings.py tests/test_installer_settings_spec.py
uv run ruff format .
uv run ruff check .
git add src/code_indexing_mcp/cpu_thread_tuning.py src/code_indexing_mcp/cpu_thread_cache.py src/code_indexing_mcp/settings.py src/code_indexing_mcp/installer/settings_spec.py tests/test_cpu_thread_tuning.py tests/test_cpu_thread_cache.py tests/test_settings.py tests/test_installer_settings_spec.py
git commit -m "feat: define automatic CPU passage thread tuning policy"
```

## Task 2: Implement bounded real-worker measurements

**Files:** Modify `cpu_thread_tuning.py`, `embedding_worker.py`,
`tests/test_cpu_thread_tuning.py`, `tests/test_embedding_worker.py`.

**Interfaces consumed:** Task 1 choice/trial types and candidate policy.
**Interface produced:** `tune_cpu_threads` returns `CpuThreadChoice`. It takes
`make_session: Callable[[int, float], EmbeddingWorkerSession]` (thread count and
absolute deadline) and keyword-only
`baseline_threads: int`, `effective_cpus: int`, `plan: SegmentPlan`,
`clock: Callable[[], float] = time.monotonic`, and
`budget_seconds: float = 15.0`. It owns/closes every trial session; the caller
closes its real-work session before entry. It never returns a live worker.

Implement the explicit sequence below. Extend
`EmbeddingWorkerSession.__init__` with a keyword-only `deadline: float | None =
None`; it is an absolute monotonic timestamp used by all requests in that
session. Omitted means the current behavior. Do not change the worker wire
payload just to add a parent-side time budget.

- [ ] Add a fake-clock/fake-session test that records `open`, `close`, warmup,
  and three timed calls. Assert never more than one session is live and that
  model load is excluded from sample time but included in the overall budget.
- [ ] Add deadline tests for initialization and embedding with a nonresponding
  child. Expiry must terminate/join the worker and raise the existing worker
  error with a specific deadline reason; a request started before the deadline
  must not be allowed to wait for the ordinary long watchdog afterward.
- [ ] In `_request`, check remaining time before send/start and before each
  existing receive/poll wait. Cap the poll timeout to remaining time; on expiry
  use the existing termination/error path. Keep memory sampling active.
- [ ] Build the sample as `list(calibration_candidates())[::2]` only if that
  retains both representative lengths; otherwise explicitly interleave four
  of each length from the deterministic corpus. Add an assertion/test for
  eight candidates with both shapes. Keep exactly this sample across trials.
- [ ] Open baseline first with the absolute deadline, initialize, run one
  warmup, then time three identical requests using `clock`. Convert to positive
  nanoseconds; discard incomplete trials. Close the worker in `finally` before
  the next candidate. Pass the identical `SegmentPlan` to all calls.
  Call `make_session(threads, deadline)` so initialization and every request
  receive the same overall deadline, rather than a fresh budget per worker.
- [ ] Feed complete trials to `select_threads`. On trial resource/backend
  errors, close that worker and retain previous safe results. Stop on deadline.
  Do not catch `BaseException`: cancellation/KeyboardInterrupt propagates.
  Exclude successful-but-shrunk batch trials (nonzero retries or reduced
  `safe_max_items`) from comparison, because they did not use the fixed plan.
- [ ] Test failure at baseline initialization, a later OOM, worker death,
  expiry during a request, cancellation, all slower candidates, and reduced
  batch retries. None may leak a process or fabricate a completed timing.
- [ ] Run tests and commit:

```bash
uv run pytest tests/test_cpu_thread_tuning.py tests/test_embedding_worker.py
uv run ruff format .
uv run ruff check .
git add src/code_indexing_mcp/cpu_thread_tuning.py src/code_indexing_mcp/embedding_worker.py tests/test_cpu_thread_tuning.py tests/test_embedding_worker.py
git commit -m "perf: measure CPU thread choices within a worker deadline"
```

## Task 3: Apply tuning to CPU-only indexing and expose actual selection

**Files:** Create `tests/test_models.py`; modify `backend_coordinator.py`, `passage_backend.py`, `models.py`,
`indexing.py`, `tests/test_passage_backend.py`, `tests/test_indexing_backend.py`,
`tests/test_indexing.py`, `tests/test_server.py`.

**Interfaces consumed:** `tune_cpu_threads`, tuning cache, auto-thread flag.
**Interfaces produced:** Optional/defaulted `embedding_threads: int | None`,
`embedding_thread_selection: str | None`, `embedding_tuning_duration_ms: int =
0` on `IndexReport`, with corresponding model-status and session telemetry
fields. `SessionTelemetry` changes live in `embedding_worker.py` too.

- [ ] Add integration tests around existing fake workers in
  `tests/test_passage_backend.py:669`. Parameterize factories by threads and
  record every WorkerConfig. Cover the following table before implementation:

| Input/run | Required result |
| --- | --- |
| CPU auto, short run, empty cache | Existing baseline worker; no tuning. |
| CPU auto, crosses 512 KiB, empty cache | One tuning attempt, selected threads used for remaining content. |
| CPU auto, valid cache | Selected threads on first worker; no tuning. |
| CPU explicit 3 | Every CPU worker uses exactly 3; no tuner/cache override. |
| Calibration disabled, cache miss | Baseline; no measurement. |
| Calibration disabled, compatible hit | Cached choice; no measurement. |
| No-op index | No inference worker or tuning record write. |
| Accelerator, deferred CPU, degraded CPU, in-process | Existing behavior; tuner not wired. |
| Tuning fails / times out | Resume with safe choice; content still succeeds. |
| Model/token/memory/batch/CPU allocation changes | Old tuning record ignored. |

- [ ] Create CPU-thread factory closures in `_passage_session_factory` using
  `dataclasses.replace(cpu_config, threads=chosen_threads)` and the existing
  launcher/ceiling. Trial factories accept `(threads, deadline)` and pass the
  deadline into `EmbeddingWorkerSession`; real-work sessions omit it. Opt into
  tuning only for a run whose initial selection is
  CPU-only, whose execution mode is worker, and whose setting is auto.
- [ ] Load cache per run rather than at coordinator construction. Build the
  key with the actual plan's effective batch size and CPU allowance. Keep
  `_cpu_probe_key`, `_measure_reference`, and accelerator crossover records
  separate; the new rate must not overwrite their reference configuration.
- [ ] At the CPU-only selection boundary, compute the trigger using successful
  run characters plus the incoming group. Before tuning, retire the real-work
  worker through the existing telemetry-preserving close path. Set the
  once-per-run attempted flag before starting trials. Measure total tuning time
  including closure/reopening and persist only a valid completed choice.
- [ ] Create a fresh selected worker and send the original pending request.
  Persisting and reusing baseline selection prevents repeated fruitless sweeps.
  Catch only tuning failures; failures in this real content request continue
  through the established run/file error policy.
- [ ] Track trial peak RSS separately and combine with real worker peak for
  the reported overall maximum. Trial segment/token/retry counts must not
  inflate indexed counts. Add a regression that compares exact stored chunks,
  text offsets, file records, and reference rows with tuning on/off using
  deterministic fake vectors.
- [ ] Test cancellation and content failure around a tuning boundary: old
  committed file data retained, no partial new file generation, staged job
  recovery unchanged, all worker processes gone after exit.
- [ ] Update model status and reports with the fields above. With no worker
  started, effective threads remain null. Preserve additive serialization and
  old report construction by providing defaults. Extend existing report/status
  tests and inspect CLI/TUI consumers via the index before changing payloads.
- [ ] Run the affected backend, indexing, model, and server tests, then format,
  lint, and typecheck. Commit the integrated behavior:

```bash
uv run pytest tests/test_passage_backend.py tests/test_indexing_backend.py tests/test_indexing.py tests/test_models.py tests/test_server.py
uv run ruff format .
uv run ruff check .
uv run mypy src
git add src/code_indexing_mcp/backend_coordinator.py src/code_indexing_mcp/passage_backend.py src/code_indexing_mcp/embedding_worker.py src/code_indexing_mcp/models.py src/code_indexing_mcp/indexing.py tests/test_passage_backend.py tests/test_indexing_backend.py tests/test_indexing.py tests/test_models.py tests/test_server.py
git commit -m "perf: apply cached CPU thread choices to passage indexing"
```

## Task 4: Prove throughput, overhead, memory, and output equivalence

**Files:** Create `scripts/benchmark_cpu_threads.py` and
`tests/test_cpu_thread_benchmark.py`; modify `README.md`, and add release results
under `docs/plans/data/` plus a shipped summary only after the gates pass.

**Interface produced:**

```text
uv run --extra cpu python scripts/benchmark_cpu_threads.py \
  --model-cache /absolute/path/to/models \
  --corpus /absolute/path/to/copied/source \
  --work-dir /absolute/path/to/fresh/benchmark-directory \
  --repeats 5 --offline
```

- [ ] Start from the preserved analysis scripts in this plan's `data/`
  directory. Replace machine-specific paths with required arguments. Refuse a
  work directory containing existing corpus/data; make scratch project copies,
  isolated data and tuning caches, and share only already-cached model assets.
- [ ] Make the benchmark emit JSON containing revision, corpus hashes and
  bytes, file/chunk/token counts, CPU/runtime/model identity, settings, selected
  thread count, trial decisions, all individual wall/phase times, tuning cost,
  worker peak RSS, retries, and failures. Test arithmetic/summary computation
  with synthetic records; do not put strict timing assertions in ordinary CI.
- [ ] Compare explicit baseline two threads (or the host's smaller conservative
  baseline), auto with empty tuning cache, and auto with compatible cache.
  Separate model download from cold worker load. Alternate execution order;
  do not run performance trials concurrently with tests or each other.
- [ ] Run five measured repetitions after one discarded warmup for each state
  on a real >=1,000-chunk source corpus and a mixed-language corpus. Include
  dense near-1-MiB files and long token-windowing inputs for memory acceptance.
  Hash the corpus and keep source/embedding configuration fixed.
- [ ] Measure empty-store index, forced rebuild, no-op, and one-file edit.
  Report median and p95, as well as raw samples; include calibration in the
  cold-tuning wall time. Enforce the success criteria near this plan's start.
  If the 15-second/512-KiB policy does not meet the cold gate, adjust that
  policy and repeat; do not publish a warm-only success as universal speedup.
- [ ] Compare output against baseline: exact chunk identities, content,
  prefixes, byte/line offsets, window count/bounds, file hashes, and reference
  rows; finite vector shape and normalization; minimum cosine >=0.999 and
  retrieval top-k overlap >=99% using existing golden model/acceptance helpers.
  Timing choices may introduce floating-point variation, not structural drift.
- [ ] Run the existing real-model memory acceptance gate with the tuned policy
  enabled, covering 1/2-GiB configurations, worst-case dense files, trial OOM,
  and trial deadline. Preserve the existing 256-MiB overshoot allowance,
  <=1-second breach duration, <=2-second worker exit, and bounded parent growth
  from `scripts/benchmark_index_memory.py`. Use whole-process-tree sampling;
  between-request RSS alone cannot certify the peak or one-model invariant.
- [ ] Repeat throughput/nonregression checks on another CPU architecture and
  on constrained CPU affinity/quota. Confirm accelerator selection, explicit
  thread override, and calibration-disabled paths do not start new trials.
- [ ] Document auto semantics, query versus passage threading, the large-run
  trigger, tuning/cache diagnostics, and explicit override in README. Update
  installer tests so an existing explicit integer survives an update unchanged.
- [ ] Complete the full repository gate before the final commit or push:

```bash
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -n auto
git diff --check
```

- [ ] Commit the benchmark, docs, and measured acceptance evidence after the
  gates pass. Use a PR description that distinguishes configuration-probe
  evidence from results on the implemented auto policy. Do not claim the
  historical 784-second repository build will improve by the sample's exact
  ratio; that has not been measured.

## Reproducing the analysis already completed

The scripts are analysis artifacts, not production features. Their hard-coded
model-cache path is this machine's existing offline cache. Use the analyzed
worktree as cwd and its existing installed dependencies:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/marcinh/Projects/code-indexing-mcp/.venv/bin/python docs/superpowers/plans/data/2026-09-07-cpu-thread-probe.py
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/marcinh/Projects/code-indexing-mcp/.venv/bin/python docs/superpowers/plans/data/2026-09-07-cpu-thread-e2e.py
```

The full-path script requires a working local LanceDB connection. In this
session it hung in `lancedb.connect` under the execution sandbox; the same
indexing tests passed outside the sandbox (9 passed). The benchmark then
completed outside the sandbox using temporary data only. This limitation does
not affect the raw worker probe.

## Plan review checklist

- [x] Prioritized with actual phase history and a controlled configuration probe.
- [x] Checked already-shipped cross-file batching, token planning, accelerator
  calibration, and storage batching before recommending new work.
- [x] Kept this PR to CPU-only passage threads; joint batch tuning and vector
  reuse are explicit follow-ups.
- [x] Specified cache identity, explicit-setting behavior, trial deadlines,
  memory/worker lifecycle, reporting, and first-run overhead gates.
- [x] Separated experimental results from unmeasured automatic-policy claims.
- [x] Identified index/reference blind spots in the linked design.
- [ ] Execute the proposed implementation and release gates in a subsequent task.

## Validation of this analysis worktree

- Baseline full suite: **1,917 passed, 16 skipped**, 214.24 s, run outside the
  sandbox after isolating its LanceDB connection hang. Skips cover unavailable
  accelerator/model environments, case-insensitive filesystem cases, and an
  existing reference fixture without a selectable declaration.
- Formatter: 137 files formatted/checked; only the two new analysis scripts
  required formatting. Ruff lint passed. Production source has no diff.
- Mypy checked 66 source files and reported two existing diagnostics at
  `src/code_indexing_mcp/direct_onnx.py:181`: unused `type: ignore` and
  `import-untyped` for installed `onnxruntime_ep_webgpu`. A fresh-cache run in
  the original checkout reproduced both. Resolve the baseline/environment
  mismatch before claiming the future PR's full gate is green; this planning
  task does not change that unrelated source line.
- Seven new artifacts passed whitespace checks. Saved full-path JSONL was
  parsed and checked for six runs, equal chunk/reference counts, zero errors,
  and the documented 1.440× / 30.5% arithmetic.
- The dependency environment was reused with `uv run --no-sync` and an
  explicitly selected existing `.venv`; no packages or live index settings
  were changed during analysis. For the later main-branch preservation commit,
  an isolated temporary environment was prepared with CI's locked CPU/TUI
  extras to rerun the required checks without that environment's WebGPU typing
  mismatch. The plan remains a proposal for later implementation.
- Main-branch preservation checks: formatting and lint passed. With the
  isolated CPU/TUI environment and a fresh mypy cache, source typechecking
  passed for all 66 source files; the memory-benchmark script typecheck also
  passed. The original environment's cached WebGPU diagnostics are not a
  source-code failure under the CI dependency configuration.
- Full suite on the preservation commit's main-branch tree: **1,951 passed,
  16 skipped**, 126.48 s, using the locked CPU/TUI environment. No production
  code changes were needed to pass these checks.
