# Post-Review Hardening Plan for PR #4

## Context

PR #4 ("Bound indexing memory and share a per-user daemon") has had a round of review and fixes
applied on `claude/pr-review-ad6efb`: the extractor chunk explosion, memory-ceiling accounting,
Windows `AF_UNIX` crash, socket placement, LIKE wildcard leakage, result determinism, the model
init race, compaction safety, and the silent per-file swallowing of environment failures are all
fixed and covered by 20 new tests (138 passing locally).

What remains falls into two buckets:

1. **A hang risk and a latency problem** that are still live in the merged behaviour.
2. **The PR's central claim — one model and one indexing job per user, no matter how many MCP
   clients connect — rests entirely on code inspection.** It has never been executed.

Windows CI is also red on `codex/bounded-memory-indexing` (`AttributeError: module 'socket' has no
attribute 'AF_UNIX'`, 2 failed / 115 passed on all four Windows jobs; macOS and Ubuntu pass). The
source crash is fixed, but the two daemon tests still need platform skips before CI goes green.

**Intended outcome:** PR #4 merges with no known hang, a first-index latency that is justified by
measurement rather than assumed, green CI on all three platforms, and its headline concurrency
claim backed by an executing test.

**Deliverable:** this plan committed as `docs/plans/2026-07-25-post-review-hardening.md`, following
the existing `docs/plans/` convention, then Tasks 1–6 implemented in order.

## Errata for `docs/plans/2026-07-24-indexing-memory-hardening-completion.md`

Measurements taken this session contradict parts of that plan. Add a short errata block at its top
rather than rewriting it; it remains the post-merge roadmap.

- **Task 2's premise is wrong.** It specifies 1,024-token windows to avoid overrunning the model.
  Measured: `jina-embeddings-v2-base-code` has `model_max_length = 8192`, the tokenizer truncates at
  8192, and 4,096 characters of real code is **963 tokens**. There is no silent truncation today.
  Tokenizer-aware chunking is a boundary-quality improvement, not a correctness fix — reprioritise
  it below Tasks 4–6.

  > **Correction, after Task 5 ran.** The "boundary-quality, not correctness" half of this is
  > wrong. 4,096 characters of *ordinary* code is ~963 tokens, but 4,096 characters of a minified
  > single-line file is several times that, and the resulting sequence drives the embedding worker
  > past its ceiling: measured at 2,048 / 3,072 / 4,096 MiB and batch sizes 1 and 4, the worker
  > exceeded every ceiling and `INDEX_RESOURCE_LIMIT` aborted the whole run. Character-based
  > windows bound characters, not the token count that actually drives memory. Task 2 is a
  > correctness fix for that shape; keep it below Tasks 4–6 only if minified files stay excluded.
  >
  > **Resolved (2026-07-25).** Token-bounded windowing landed as Task 7 below, so minified files no
  > longer need excluding. The 07-24 plan's Tasks 2–3 are marked done there, with the deviations
  > recorded.
- **Task 3's retry design has no effect at the shipped default.** It specifies a `4 → 2 → 1`
  microbatch backoff, but `INCODE_EMBED_BATCH_SIZE` defaults to 1, so there is nothing to halve.
  Retry only becomes meaningful once Task 3 of *this* plan raises the default.
- **Task 1's benchmark corpus is under-specified.** Deterministic Python files at varying counts do
  not exercise the memory peak, which tracks the largest *single* file. See Task 5 below.

## Task 1: Bound the `INDEX_BUSY` wait

**Problem.** `StartupCoordinator._run` ([server.py:123](../../src/incode_mcp/server.py)) retries
`index_project` in a `while True` at 20 Hz with no deadline. The polling itself is deliberate —
the `_lifespan` comment explains that lock waiters must stay cancellable between non-blocking
attempts — but nothing bounds it. PR #4 widened the lock from per-project to global
(`index-global.lock`), so *any* two indexing jobs now contend. With `INCODE_BROKER=off` and two
clients, or two roots in one session, the second client's first query blocks for the entire
duration of the first index with no timeout and no error.

**Files:** `src/incode_mcp/server.py`, `src/incode_mcp/settings.py`, `tests/test_server.py`,
`README.md`

**Approach.** Keep the cancellable polling loop; add a deadline and backoff.

- Add `index_wait_seconds` to `IndexSettings` (env `INCODE_INDEX_WAIT_SECONDS`, default 300,
  range 0–86400) using the existing `_integer` helper in
  [settings.py](../../src/incode_mcp/settings.py).
- Thread the value into `StartupCoordinator.__init__` alongside `mode`.
- Replace the fixed `await anyio.sleep(0.05)` with exponential backoff capped at 1.0s, and track a
  monotonic deadline. On expiry, re-raise the `INDEX_BUSY` `IncodeError` with the waited duration
  in `details` so `wait_for_ready` propagates a clear error instead of hanging.

**Test.** Reuse the existing `FileLock` import and `BlockingEmbedder` pattern in
`tests/test_server.py`: hold `index-global.lock` externally, call `search_code` with a short
`INCODE_INDEX_WAIT_SECONDS`, and assert the tool returns an `INDEX_BUSY` error within the deadline
rather than blocking. Add a second test asserting a job that becomes free before the deadline still
succeeds.

## Task 2: Green Windows CI

**Problem.** `tests/test_daemon.py::test_broker_application_calls_one_daemon_backend` and
`::test_daemon_does_not_idle_exit_while_request_is_active` both call `DaemonServer.serve()`, which
needs `AF_UNIX`. The fix on this branch converts the `AttributeError` into a clean `IncodeError`,
so `daemon.ready.wait()` times out and the tests still fail — just more legibly.

**Files:** `tests/test_daemon.py`

**Approach.** Skip on capability, not platform — `@pytest.mark.skipif(not daemon_supported(), ...)`
using the existing `daemon_supported()` from [daemon.py:43](../../src/incode_mcp/daemon.py). Prefer
it over `os.name == "nt"` so the guard tracks the real constraint. The two existing
`skipif(os.name == "nt")` markers on the endpoint/permission tests should move to the same
predicate where they gate socket support rather than POSIX ownership semantics.

Cross-platform coverage is preserved by the tests that already run everywhere:
`test_length_prefixed_json_frame_round_trip` (uses `socket.socketpair`, which Windows emulates),
`test_require_daemon_support_explains_unsupported_platforms`, and the two `cli` fallback tests.

**Verification.** Push and confirm all six CI jobs pass. This is the only task whose result cannot
be fully verified locally on macOS.

## Task 3: Justify first-index latency by measurement

**Problem.** Measured throughput is **20.1 chunks/s** — this repo (33 files, 481 chunks) takes 24s.
Extrapolated, a 5,000-file repo is roughly an hour, and in lazy mode that is what the user's first
`search_code` call blocks on.

**Correction to earlier analysis:** the embedding worker is *not* respawned per file. The session is
created once per `index()` call in `_index_locked` and reused across every file in that run, and in
lazy mode indexing runs once per server process. The reload therefore costs ~3–5s once per MCP
session, not per file. Worker reuse *across* runs is the wrong fix — it would keep ~1.5 GiB
resident in the daemon between refreshes, directly against this PR's purpose. Leave the per-run
lifecycle alone.

The real lever is batch size. Measured peak worker RSS is 1462–1555 MiB at `batch_size=1` against a
2048 MiB ceiling — roughly 500 MiB of headroom.

**Files:** `src/incode_mcp/settings.py`, `src/incode_mcp/models.py`, `src/incode_mcp/indexing.py`,
`README.md`

**Approach.**

- Add scan/parse/embed/commit duration fields to `IndexReport` (additive and optional, matching the
  existing `memory_budget_bytes` pattern) so the load-vs-embed split is observable. This overlaps
  the 07-24 plan's Task 3 Step 3 — implement the durations here and mark that step done there.
- Sweep `INCODE_EMBED_BATCH_SIZE` over 1, 2, 4, 8 against a fixed corpus using the harness from
  Task 5, recording throughput and peak worker RSS at each.
- Raise the default to the largest value whose peak stays within the ceiling with the 128 MiB
  hard-overshoot margin intact. **Do not guess the value** — set it from the sweep, and record the
  numbers in the README next to the setting.
- If no batch size above 1 fits, keep the default and document the measured throughput ceiling so
  users can choose `INCODE_INDEX_MODE=eager` knowingly.

**Ordering note.** The sweep consumes Task 5's harness, so implement the `IndexReport` durations
here, build Task 5, then return to set the default from the sweep.

### Measured outcome

Apple Silicon macOS, 1.0 MiB / 6,330-chunk dense-Python corpus, `INCODE_INDEX_MEMORY_MB=2048`:

| `INCODE_EMBED_BATCH_SIZE` | Wall clock | Chunks/s | Peak worker RSS | Peak combined |
| ------------------------- | ---------- | -------- | --------------- | ------------- |
| 1                         | 147.0 s    | 44.8     | 1,415 MiB       | 1,656 MiB     |
| 2                         | 136.2 s    | 48.7     | 1,419 MiB       | 1,661 MiB     |
| 4                         | 130.4 s    | 50.9     | 1,427 MiB       | 1,675 MiB     |
| 8                         | 126.7 s    | 52.5     | 1,451 MiB       | 1,691 MiB     |

**The default stays at 1** — the plan's second branch. Batch size 8 returns 17% throughput for
36 MiB more resident memory, and the shape that actually drives the peak (a minified single-line
file near the cap) already exceeds the ceiling at batch size 1, so there is no headroom to spend.
Embedding dominates at 141 s of 147 s, confirmed by the new `embed_duration_ms`. Throughput is
~45 chunks/s, higher than the 20.1 chunks/s in the problem statement above because this corpus is
denser in small definitions; both figures leave a cold index on a large repository well beyond what
a first query should block on, so the README now steers large repositories to `eager` or `manual`.

## Task 4: Prove the one-model / one-job claim

**Problem.** The PR's stated goal is that N connected MCP clients share one scheduler and one model.
Nothing executes that. The `FastEmbedder._model_lock` added on this branch closes the race visible
by inspection, but the multi-client path has never run.

**Files:** `tests/test_daemon.py`

**Approach.** Drive one `DaemonServer` with several concurrent `BrokerApplication` clients.

- Counting embedder that records every model construction and every `embed_passages` call, built on
  the existing `TinyEmbedder` in `tests/test_daemon.py`.
- Start one `DaemonServer` on a thread (existing pattern). Launch 8 client threads that each call
  `init_project` then `index_project` for the **same** canonical root, released together with a
  `threading.Barrier`.
- Assert: exactly one model construction; exactly one project registered; the successful client's
  report is complete; and every other client either shares that result or fails with `INDEX_BUSY` —
  never a partial or duplicated index.
- Assert `_active_requests` returns to 0 and the daemon shuts down cleanly.

**Note.** Genuine request coalescing (all 8 receiving the same result) is the 07-24 plan's Task 6
and is out of scope here. This test pins down current behaviour so Task 6 has a baseline to change
deliberately; write the assertions to describe what the code does today, with a comment pointing at
the scheduler task.

## Task 5: Memory acceptance for the cases that actually peak

**Problem.** Peak worker RSS tracks the largest single file, not repo size. Validation so far used a
1.1 MB corpus of ordinary source. Files near the 1 MiB `max_file_bytes` cap and single-line
minified files — the exact shapes that drive the extractor fragment path and the largest embedding
batch — are untested.

**Files:** `scripts/benchmark_index_memory.py`, `tests/test_memory_acceptance.py`,
`.github/workflows/ci.yml`, `pyproject.toml`

**Approach.** Implement the 07-24 plan's Task 1 harness, with the corpus extended to include:
a file just under `max_file_bytes` (1 MiB); a single-line minified file near the cap; a file with a
long blank run adjacent to an oversized line; and the ordinary multi-file corpus for a baseline.

Reuse the process-tree sampling approach validated this session: `psutil` parent + recursive
children at 100 ms, recording parent / worker / combined separately — combined RSS alone is what
made the original accounting bug invisible.

Register a `memory` pytest marker, keep pure `evaluate_result()` validation in normal CI, and gate
the real-model run behind a manually dispatched job that restores a model cache. Upload the
benchmark JSON even on failure.

### Measured outcome

The harness found what it was built to find. Per shape, at batch size 1 and a 2,048 MiB ceiling:

| Shape       | Result       | Peak parent / worker / combined | Notes                          |
| ----------- | ------------ | ------------------------------- | ------------------------------ |
| ordinary    | pass         | 194 / 1,415 / 1,604 MiB         | 75 chunks, 4.0 s               |
| `blank_run` | pass         | 195 / 1,430 / 1,619 MiB         | 81 chunks, 4.2 s               |
| `near_cap`  | pass         | 706 / 1,415 / 1,656 MiB         | 6,330 chunks, 147 s            |
| `minified`  | **breaches** | 261 / 2,345 / 2,534 MiB         | `INDEX_RESOURCE_LIMIT` in 2.9s |

The minified shape exceeds the ceiling at 2,048, 3,072 and 4,096 MiB alike and at batch sizes 1 and
4 — raising the ceiling does not help, so this is not a tuning problem. Because
`INDEX_RESOURCE_LIMIT` is an environment error, it aborts the entire run: one bundled or minified
file blocks indexing for the whole project. `scan.exclude` is a working stopgap (verified: the run
completes cleanly with the file excluded) and is now documented in the README. The proper fix is
the 07-24 plan's Tasks 2–3, and
`tests/test_memory_acceptance.py::test_a_minified_file_stays_within_its_ceiling` holds it as a
strict xfail so it cannot be forgotten.

`near_cap` also shows parent RSS reaching 706 MiB against 194 MiB for ordinary files — the
list-of-floats ingestion path the 07-24 plan's Task 4 replaces with Arrow staging.

> **Superseded for `minified` (2026-07-25).** Task 7 below fixed the breach. The strict xfail is now
> a plain passing gate, and the README no longer presents `scan.exclude` as a workaround for
> correctness — only as a way to avoid spending time and index space on generated files.

## Task 6: Small correctness items

**Files:** `src/incode_mcp/storage.py`, `tests/test_storage.py`

- **`_tables()` creates partitions on read.** [storage.py](../../src/incode_mcp/storage.py) calls
  `create_table(..., exist_ok=True)`, so querying an unknown project id silently creates an empty
  partition directory — reachable from `get_chunk`, which iterates every registered project. Add a
  read-only path that uses `open_table` and returns `None` when the partition is absent; keep
  create-on-write for the indexing path. Test that `get_chunk` for an unknown id creates nothing on
  disk.
- **`get_chunk` scans every project.** `chunk_id` is a one-way digest of `file_id`, which is itself
  a digest of project id and path, so the project cannot be recovered from the id. The scan is
  inherent without an id format change and re-index. Leave it; the create-on-read fix above removes
  the harmful part. Note the reasoning in a comment so it is not "fixed" later by accident.
- **Symbol over-fetch is silently truncating.** `find_symbol_chunks` scans at most
  `max(limit * 10, 200)` rows before applying exact match semantics. If the LIKE pre-filter matches
  more than that and the real matches sort late, they are dropped with no signal. Log at debug when
  the pre-filter returns exactly the cap, so the condition is diagnosable.

## Task 7: Bound embedding sequences by tokens (added 2026-07-25)

**Problem.** Task 5's harness found the one shape that still breaches: a single-line minified file
near the 1 MiB cap. It exceeds the ceiling at 2,048 / 3,072 / 4,096 MiB and at batch sizes 1 and 4,
so it is not a tuning problem, and because `INDEX_RESOURCE_LIMIT` is an environment error it aborts
the entire run — one bundled file blocks indexing for the whole project.

**Cause, measured.** Attention is quadratic in sequence length, and character windows bound the
wrong quantity. 4,096 characters of ordinary source is 984 tokens; the same 4,096 characters of
minified source is 2,157. Embedding the long sequence adds ~1,172 MiB of resident memory; the same
characters as three token-bounded windows add ~266 MiB, and finish faster (0.78 s against 1.07 s).

**Files:** `src/incode_mcp/token_batching.py`, `src/incode_mcp/embedding.py`,
`src/incode_mcp/embedding_worker.py`, `src/incode_mcp/indexing.py`, `src/incode_mcp/extractor.py`,
`src/incode_mcp/models.py`, `src/incode_mcp/settings.py`, `src/incode_mcp/application.py`,
`tests/test_token_batching.py`, `tests/test_indexing.py`, `tests/test_embedding.py`,
`tests/test_embedding_worker.py`, `tests/test_settings.py`, `tests/test_memory_acceptance.py`,
`README.md`

**Approach.** This is the 07-24 plan's Tasks 2–3, implemented ahead of the rest of that plan because
it is the only correctness defect left on this PR. The design and its deviations from those steps
are recorded under "Task 2 and Task 3 as implemented" in
`docs/plans/2026-07-24-indexing-memory-hardening-completion.md`, not duplicated here.

### Measured outcome

Peak parent / worker / combined RSS at batch size 1 against a 2,048 MiB ceiling:

| Shape                  | Before                          | After                          |
| ---------------------- | ------------------------------- | ------------------------------ |
| `minified`             | 261 / 2,345 / 2,534 MiB, aborts | 321 / 1,879 / 2,073 MiB, clean |
| `near_cap`+`blank_run` | 706 / 1,430 / 1,656 MiB         | 728 / 1,435 / 1,681 MiB        |

For `minified`, `INDEX_RESOURCE_LIMIT` in 2.9 s became 1,066 chunks in 188 s with 0 errors and 0
retries. The shapes that already passed are unchanged within noise — they never exceed the token
budget, so no chunk of theirs is windowed, and a test asserts the windowed and unwindowed paths
produce identical chunks for source that fits.

`test_a_minified_file_stays_within_its_ceiling` is now a plain passing gate that also asserts
`token_windowing` is true, so a silent fallback to whole-candidate embedding cannot pass it for the
wrong reason.

## Verification

Run at each task boundary:

```bash
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

End-to-end with the real model, against an **isolated data directory** — the live store at
`~/Library/Application Support/incode/lancedb` is schema v1 and this code migrates destructively on
first run:

```bash
export INCODE_DATA_DIR=/tmp/incode-check/data
export INCODE_CACHE_DIR="$HOME/Library/Caches/incode"
export INCODE_OFFLINE=1
uv run code-indexing-mcp index <path-to-a-checkout>
uv run code-indexing-mcp daemon status
```

Expect `state: ready`, `errors: 0`, and a chunk count consistent with the previous run. Confirm the
socket lands under `XDG_RUNTIME_DIR` or the per-user `TMPDIR` — never `/private/tmp` — and that
`daemon stop` leaves no stray process or socket.

For Task 1, verify the bound directly: hold `<data>/locks/index-global.lock` from a second shell,
issue a query, and confirm it fails with `INDEX_BUSY` at the deadline instead of hanging.

For Task 3, record the batch-size sweep numbers in the PR description alongside the existing
measurements (cold 173/1462/1635 MiB parent/worker/combined; warm 1399/~1500/2735–2829 MiB).

## Out of scope

Deferred to `docs/plans/2026-07-24-indexing-memory-hardening-completion.md`: Arrow staging with
crash-recoverable commits (Task 4), migration dual-read and validation (Task 5), the daemon
scheduler, protocol handshake and connection limits (Task 6), Windows named pipes and installer
draining (Task 7), and release qualification (Task 8).

Windows keeps the graceful direct-mode fallback and gets no shared daemon until Task 7 lands; the
README already documents this. Tokenizer-aware chunking is deprioritised per the errata above.
