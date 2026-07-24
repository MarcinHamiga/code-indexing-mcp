# Indexing Memory Hardening Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the bounded-memory, migration-safety, daemon-hardening, and release-validation work that remains after PR #4.

**Architecture:** Keep PR #4's lazy indexing, per-project storage, global lock, spawned embedding worker, and daemon facade. Add token-aware chunk planning inside the worker, packed Arrow staging with recoverable Lance version journals, a read-only legacy fallback during migration, and a daemon-owned coalescing scheduler. Finish with cross-platform IPC, installer draining, and reproducible real-model memory gates.

**Tech Stack:** Python 3.12/3.13, FastEmbed 0.8, ONNX Runtime, Hugging Face tokenizers, NumPy, PyArrow IPC, LanceDB 0.34, psutil, filelock, FastMCP, pytest.

## Delivery order and release boundaries

Implement the tasks below in order. Tasks 1–4 form the bounded-memory release, Tasks 5–7 form the daemon/migration hardening release, and Task 8 is the release gate. Do not enable automatic backup deletion or make daemon mode mandatory until Task 8 passes on Linux, macOS, and Windows.

### Task 1: Establish executable memory acceptance gates

**Files:**
- Create: `scripts/benchmark_index_memory.py`
- Create: `tests/test_memory_acceptance.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

**Step 1: Write failing unit tests for benchmark result validation**

Add tests for a pure `evaluate_result()` function covering:

- combined RSS at or below `effective_ceiling + 256 MiB` passes;
- a breach lasting more than one second fails;
- a worker still alive two seconds after completion fails;
- parent RSS growth over 128 MiB between a 1× and 10× synthetic corpus fails;
- fewer than two RSS samples is an invalid benchmark.

Run:

```bash
.venv/bin/pytest -q tests/test_memory_acceptance.py
```

Expected: FAIL because the benchmark module does not exist.

**Step 2: Implement the benchmark harness**

The script must:

- create a temporary project outside the repository;
- generate deterministic Python files at configurable corpus sizes;
- start `code-indexing-mcp index` in a child process with an isolated data/cache directory;
- sample the process tree with psutil every 100 ms;
- record parent, worker, combined, and system-available memory;
- record timestamps for cap breaches and worker exit;
- emit one JSON document containing configuration, peaks, durations, exit codes, and pass/fail reasons;
- never download a model unless `--model-cache` is explicitly supplied.

Expose `evaluate_result()` separately from process execution so ordinary CI uses synthetic fixtures without loading the real model.

**Step 3: Add opt-in pytest markers and CI smoke coverage**

- Register `memory` in `pyproject.toml`.
- Run pure benchmark validation in normal CI.
- Add a manually dispatched Linux job that restores a model cache and runs:

```bash
.venv/bin/pytest -q -m memory
```

The job must upload benchmark JSON even on failure.

**Step 4: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_memory_acceptance.py
.venv/bin/ruff check scripts tests/test_memory_acceptance.py
.venv/bin/mypy scripts/benchmark_index_memory.py
```

Commit:

```bash
git add scripts/benchmark_index_memory.py tests/test_memory_acceptance.py .github/workflows/ci.yml pyproject.toml
git commit -m "test: add indexing memory acceptance harness"
```

### Task 2: Replace character chunking with tokenizer-aware bounded chunks

**Files:**
- Create: `src/incode_mcp/token_batching.py`
- Modify: `src/incode_mcp/embedding_worker.py`
- Modify: `src/incode_mcp/extractor.py`
- Modify: `src/incode_mcp/indexing.py`
- Modify: `src/incode_mcp/models.py`
- Modify: `tests/test_extractor.py`
- Create: `tests/test_token_batching.py`
- Modify: `tests/test_embedding_worker.py`

**Step 1: Define the worker request and response models**

Add immutable internal models:

- `ChunkCandidate`: semantic metadata, source-relative byte range, and text;
- `EmbeddedSegment`: candidate identity, segment byte/line offsets, token count, packed float32 vector;
- `PackedEmbeddingBatch`: dimension, segment list, contiguous vector bytes.

The extractor must return semantic candidates without hardware-dependent splitting. Retain a 16,384-character emergency ceiling only to protect parsing and IPC; it is not the semantic chunk boundary.

**Step 2: Write failing token-window tests**

Cover:

- at most 1,024 tokenizer tokens per segment;
- 64-token overlap between adjacent segments;
- correct UTF-8 byte and line offsets;
- long single lines;
- at most 512 segments per file;
- emitted text no greater than twice the source byte size;
- deterministic boundaries regardless of memory budget or batch retry;
- explicit file error when caps cannot be satisfied.

Use a deterministic fake tokenizer that returns offset mappings. Do not require the real model in unit tests.

**Step 3: Implement token planning in the worker**

Load the tokenizer from the same FastEmbed model directory in `_worker_main`. Add a `plan_and_embed` command that:

1. tokenizes each candidate with offset mappings;
2. creates 1,024-token windows with 64-token overlap;
3. returns source-relative character offsets;
4. packs microbatches so both `item_count <= 4` and `item_count × longest_padded_tokens <= 4096`;
5. emits contiguous little-endian float32 bytes.

The parent converts tokenizer character offsets to UTF-8 byte and line offsets without decoding the whole file again.

**Step 4: Enforce file-level caps before storage**

In `Indexer`, stop accepting segments when either the 512-segment or 2× source-size cap would be exceeded. Record an `IndexIssue`, preserve the previous file generation, and do not commit a partial replacement.

**Step 5: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_extractor.py tests/test_token_batching.py tests/test_embedding_worker.py tests/test_indexing.py
```

Commit:

```bash
git add src/incode_mcp tests/test_extractor.py tests/test_token_batching.py tests/test_embedding_worker.py tests/test_indexing.py
git commit -m "feat: add tokenizer-bounded chunk planning"
```

### Task 3: Add adaptive retries and complete memory telemetry

**Files:**
- Modify: `src/incode_mcp/embedding_worker.py`
- Modify: `src/incode_mcp/indexing.py`
- Modify: `src/incode_mcp/models.py`
- Modify: `src/incode_mcp/errors.py`
- Modify: `tests/test_embedding_worker.py`
- Modify: `tests/test_indexing.py`

**Step 1: Write failing retry-state tests**

Cover:

- initial microbatch size four;
- retry sequence `4 → 2 → 1`;
- worker restart after a cap termination;
- no retry when a single 1,024-token segment exceeds the cap;
- at most two retries per original batch;
- cancellation stops retries and removes the worker within two seconds;
- successful earlier files remain committed after a later resource failure.

**Step 2: Implement retry orchestration in the parent**

Keep chunk boundaries unchanged. On `INDEX_RESOURCE_LIMIT` or `EMBEDDING_WORKER_FAILED`:

- terminate and join the failed worker;
- halve only the microbatch item limit;
- start a fresh worker;
- retry unfinished segments;
- return `INDEX_RESOURCE_LIMIT` once batch size one fails.

Do not retry model-not-found, protocol, validation, or cancellation errors.

**Step 3: Complete `IndexReport` telemetry**

Add:

- configured and effective ceiling;
- parent, worker, and combined peaks;
- batch count and retry count;
- token and segment counts;
- scan, parse, embed, commit, and maintenance durations;
- worker termination reason;
- vector-index mode.

All fields remain additive and optional for MCP compatibility.

**Step 4: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_embedding_worker.py tests/test_indexing.py
.venv/bin/mypy src
```

Commit:

```bash
git add src/incode_mcp/embedding_worker.py src/incode_mcp/indexing.py src/incode_mcp/models.py src/incode_mcp/errors.py tests/test_embedding_worker.py tests/test_indexing.py
git commit -m "feat: retry bounded embedding batches"
```

### Task 4: Stream Arrow staging and add crash-recoverable commits

**Files:**
- Create: `src/incode_mcp/staging.py`
- Modify: `src/incode_mcp/scanner.py`
- Modify: `src/incode_mcp/indexing.py`
- Modify: `src/incode_mcp/storage.py`
- Modify: `src/incode_mcp/application.py`
- Create: `tests/test_staging.py`
- Modify: `tests/test_scanner.py`
- Modify: `tests/test_indexing.py`
- Modify: `tests/test_storage.py`

**Step 1: Write failing streaming and recovery tests**

Cover:

- scanner yields one `ScannedFile` at a time and retains no source bytes;
- only one file source and one embedding microbatch are live simultaneously;
- staged vectors remain packed float32 Arrow arrays;
- no `StoredChunk.model_dump()` list-of-floats path is used;
- cancellation during staging leaves live tables unchanged;
- crash after the first live-table write restores both files and chunks table versions;
- repeated recovery is idempotent;
- staged directories are removed only after successful commit or rollback.

**Step 2: Implement the staging layout**

Use:

```text
<data>/staging/<project-id>/<job-id>/
  journal.json
  files.arrow
  chunks.arrow
```

Write JSON and Arrow files through temporary siblings, `fsync`, then atomic rename. The journal phases are `staging`, `committing`, `complete`, and `rolled_back`.

**Step 3: Add Lance version journaling**

Immediately before live writes, record `files.version` and `chunks.version`. Commit staged Arrow batches under the existing global lock. If any write, project-state update, or index-maintenance step fails:

- call `files.restore(recorded_version)`;
- call `chunks.restore(recorded_version)`;
- return both tables to latest checkout;
- mark the journal `rolled_back`;
- preserve the previous searchable project.

On application startup, recover every journal in `committing` before accepting queries.

**Step 4: Replace list-based ingestion**

Add `LanceStore.replace_files_from_arrow()` and construct fixed-size-list float32 arrays directly from worker bytes. Keep `StoredChunk` only for public single-chunk reads and test fixtures.

**Step 5: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_staging.py tests/test_scanner.py tests/test_indexing.py tests/test_storage.py
```

Commit:

```bash
git add src/incode_mcp/staging.py src/incode_mcp/scanner.py src/incode_mcp/indexing.py src/incode_mcp/storage.py src/incode_mcp/application.py tests/test_staging.py tests/test_scanner.py tests/test_indexing.py tests/test_storage.py
git commit -m "feat: stage index updates with rollback"
```

### Task 5: Make v1 migration continuously searchable and self-validating

**Files:**
- Create: `src/incode_mcp/migration.py`
- Modify: `src/incode_mcp/storage.py`
- Modify: `src/incode_mcp/search.py`
- Modify: `src/incode_mcp/application.py`
- Modify: `src/incode_mcp/cli.py`
- Modify: `tests/test_storage.py`
- Create: `tests/test_migration.py`
- Modify: `tests/test_search.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing migration-router tests**

Cover:

- v1 is opened read-only and remains searchable while v2 is pending;
- a failed v2 rebuild continues returning v1 results;
- a validated v2 project switches atomically and no longer queries v1;
- duplicate v1 chunk IDs never enter v2;
- validation checks unique files/chunks, counts, and deterministic search samples;
- missing roots remain `migration_blocked`;
- cleanup refuses to remove the backup before every reachable project passes and seven days elapse;
- cleanup is explicit when blocked projects remain.

**Step 2: Add a durable migration manifest**

Store `<data>/migration-v1.json` with:

- backup path and creation time;
- canonical project roots;
- per-project state `pending|building|validated|active|failed|blocked`;
- old and new counts;
- validation sample hashes;
- activation timestamp and failure details.

Write it atomically under the global lock.

**Step 3: Implement `StoreRouter`**

Route each project to either the v1 global tables or its v2 partition. For a pending migration:

- code queries continue using v1;
- lazy indexing builds v2 through Tasks 2–4;
- validation runs before activation;
- one manifest update switches the project to v2.

Never merge v1 and v2 rows for the same project in one response.

**Step 4: Add migration CLI operations**

Add:

```text
code-indexing-mcp storage migration status
code-indexing-mcp storage migration retry <project>
code-indexing-mcp storage migration cleanup
```

Cleanup performs a dry eligibility check first and reports why it refuses.

**Step 5: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_migration.py tests/test_storage.py tests/test_search.py tests/test_cli.py
```

Commit:

```bash
git add src/incode_mcp/migration.py src/incode_mcp/storage.py src/incode_mcp/search.py src/incode_mcp/application.py src/incode_mcp/cli.py tests/test_migration.py tests/test_storage.py tests/test_search.py tests/test_cli.py
git commit -m "feat: keep legacy indexes searchable during migration"
```

### Task 6: Add daemon protocol negotiation, idempotency, and job coalescing

**Files:**
- Create: `src/incode_mcp/scheduler.py`
- Modify: `src/incode_mcp/daemon.py`
- Modify: `src/incode_mcp/server.py`
- Modify: `src/incode_mcp/application.py`
- Create: `tests/test_scheduler.py`
- Modify: `tests/test_daemon.py`
- Modify: `tests/test_server.py`

**Step 1: Write failing scheduler tests**

Cover:

- eight automatic requests for one canonical project share one job and result;
- different projects queue behind the one active heavy job;
- priority is `explicit index > lazy query > maintenance`;
- disconnected adapters do not cancel a shared job;
- cancellation occurs only when no dependents remain;
- completed idempotency keys return the original result;
- deadlines reject queued work without starting it;
- daemon idle shutdown waits for active and queued jobs.

**Step 2: Implement `IndexScheduler`**

Use a condition-protected priority queue and one dedicated worker thread. Key automatic jobs by canonical project ID. Track dependents, stable job IDs, priority, deadline, cancellation state, and final result/error.

All daemon indexing calls must pass through the scheduler; remove adapter-side `INDEX_BUSY` polling for broker-backed applications.

**Step 3: Version the protocol**

Require a handshake before application calls:

- client sends supported min/max protocol versions and capabilities;
- daemon chooses one common version;
- response includes daemon package, protocol, and storage versions;
- no overlap returns `INVALID_CONFIGURATION` with restart/upgrade guidance.

Every request includes ID, idempotency key, canonical roots, working directory, and optional deadline.

**Step 4: Bound daemon concurrency**

- Maximum 32 accepted connections.
- Maximum 64 queued application requests.
- Existing 16 MiB frame limit remains.
- Reject excess work with a stable `DAEMON_OVERLOADED` error.

**Step 5: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_scheduler.py tests/test_daemon.py tests/test_server.py
```

Commit:

```bash
git add src/incode_mcp/scheduler.py src/incode_mcp/daemon.py src/incode_mcp/server.py src/incode_mcp/application.py tests/test_scheduler.py tests/test_daemon.py tests/test_server.py
git commit -m "feat: coalesce indexing in the shared daemon"
```

### Task 7: Finish cross-platform IPC and installer lifecycle handling

**Files:**
- Create: `src/incode_mcp/ipc.py`
- Modify: `src/incode_mcp/daemon.py`
- Modify: `src/incode_mcp/cli.py`
- Modify: `install.py`
- Modify: `tests/test_daemon.py`
- Create: `tests/test_ipc.py`
- Modify: `tests/test_installer.py`

**Step 1: Extract a transport interface**

Define `LocalTransport`, `LocalListener`, and `LocalConnection`. Implement:

- POSIX Unix-domain sockets in a mode-0700 runtime directory with mode-0600 socket and peer-UID verification;
- Windows named pipes using `multiprocessing.connection` with `AF_PIPE`, the existing per-user token, and no TCP fallback.

Both transports carry the same bounded JSON frames and protocol handshake.

**Step 2: Write platform-specific tests**

Cover endpoint permissions, stale endpoint recovery, second-daemon leader rejection, authentication failure, partial frames, oversized frames, and cleanup after idle exit. Mark POSIX and Windows-only cases with platform guards; keep common framing tests unguarded.

**Step 3: Drain the daemon during installation**

Before Git update or `uv sync`, `install.py` must:

- locate the currently installed executable;
- run `daemon status`;
- issue `daemon stop` when running;
- wait up to ten seconds;
- continue with a warning if the old executable does not support daemon commands;
- abort the update if a known daemon refuses to stop.

Do not edit any harness configuration; the existing `serve` command remains stable.

**Step 4: Add rollout switches**

Keep `INCODE_BROKER=auto|on|off`. For one release:

- fresh installs default to `auto`;
- upgrades preserve an explicit `off`;
- `serve --direct` remains available;
- startup/protocol errors include the direct-mode rollback command.

**Step 5: Verify and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_ipc.py tests/test_daemon.py tests/test_installer.py
```

Commit:

```bash
git add src/incode_mcp/ipc.py src/incode_mcp/daemon.py src/incode_mcp/cli.py install.py tests/test_ipc.py tests/test_daemon.py tests/test_installer.py
git commit -m "feat: harden daemon lifecycle across platforms"
```

### Task 8: Execute release qualification and document rollback

**Files:**
- Modify: `README.md`
- Create: `docs/indexing-memory-operations.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

**Step 1: Run the complete static and unit gate**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src
git diff --check
```

Expected: zero failures; the real-model and memory suites may be skipped only when their cache input is absent.

**Step 2: Run the real-model matrix**

On Linux and macOS, run:

```bash
.venv/bin/python scripts/benchmark_index_memory.py \
  --model-cache "$INCODE_MODEL_TEST_CACHE" \
  --corpus-scale 1 \
  --output memory-1x.json
.venv/bin/python scripts/benchmark_index_memory.py \
  --model-cache "$INCODE_MODEL_TEST_CACHE" \
  --corpus-scale 10 \
  --output memory-10x.json
```

Release criteria:

- combined RSS never exceeds the effective ceiling plus 256 MiB for one second;
- worker exits within two seconds;
- steady parent growth from 1× to 10× remains below 128 MiB;
- opening/listing tools in eight adapters loads no model;
- eight same-project first queries create one daemon, one query model, one worker, and one indexing job;
- cancellation and forced worker death preserve the previous searchable generation.

On Windows, run the same multi-adapter and recovery scenarios with the named-pipe transport.

**Step 3: Perform migration qualification**

Create fixtures for:

- a clean v1 store;
- duplicate chunk IDs;
- an incompatible model ID;
- a missing project root;
- an interrupted staging journal.

Verify continuous legacy search, bounded rebuild, activation, rollback, seven-day cleanup refusal, and explicit cleanup.

**Step 4: Document operations and rollback**

Document:

- every environment variable and precedence rule;
- interpreting memory telemetry;
- daemon status/restart/stop;
- `INCODE_BROKER=off` and `serve --direct`;
- migration status/retry/cleanup;
- locating and restoring the v1 backup;
- collecting benchmark JSON for a bug report.

**Step 5: Bump the release only after all gates pass**

Use the next unreleased version rather than relying on PR #4's provisional version. Refresh `uv.lock`, rerun the complete gate, and commit:

```bash
git add README.md docs/indexing-memory-operations.md .github/workflows/ci.yml pyproject.toml uv.lock
git commit -m "docs: qualify bounded-memory indexing release"
```

## Final acceptance criteria

- The default path performs no indexing or model loading during tool discovery.
- Only one heavy indexing job exists per user, regardless of connected MCP clients.
- Indexing terminates safely at the configured memory ceiling and preserves the prior searchable generation.
- Chunk boundaries are tokenizer-defined and independent of hardware or retry behavior.
- No indexing path expands full embedding columns into Python object graphs.
- Migration remains searchable until a validated atomic project switch.
- Unix sockets and Windows named pipes are authenticated, bounded, and cleaned up safely.
- All static, unit, multiprocess, migration, real-model, and memory gates pass before release.
