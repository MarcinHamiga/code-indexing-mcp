
# Storage, Performance, and Transparency Remediation Plan

## Context

This plan addresses the largest observed weaknesses in the MCP server's indexing
pipeline, with priority given to index disk size, write amplification, operational
auditability, and processing transparency.

The live installation measured during the review contained:

- About 6.0 GiB of physical LanceDB storage.
- About 75.8 MiB of current logical chunk data reported by Lance.
- 18,773 current chunks across eight projects.
- One 3.27 GiB partition containing only 20.2 MiB of current logical chunk data
  and 985 retained table versions.
- A 51.6 MiB project registry containing eight current rows totaling about
  6.7 KiB and 4,403 retained versions.
- A 237.7 MiB partition for this repository containing 2,472 current chunks.
- An 85.5 MiB structural-reference table for this repository containing about
  3.2 MiB of current logical data.

These measurements show that retained versions and per-file mutation overhead
dominate physical storage. Vector precision and row-schema duplication matter
after lifecycle management is corrected, but they are not the first problem to
solve.

## Decisions

- Target the latest `origin/main`, including reference indexing and the additional
  language support already merged there.
- Preserve project registrations when a schema or model change requires rebuilding
  index partitions.
- Rebuild incompatible indexes automatically within the configured indexing mode.
- Deliver the work through incremental pull requests rather than one large release.
- Automatically retain old Lance versions for 24 hours before verified cleanup.
- Keep automatic maintenance conservative: never use `delete_unverified=True` and
  never use zero-age cleanup while independent readers may be active.
- Keep vectors as `float32` until lifecycle and schema fixes are measured and a
  retrieval-quality benchmark justifies lower precision.

## Success Criteria

- An unchanged indexing run creates zero chunk, reference, and file-table versions.
- A forced reindex creates O(batches) table versions rather than O(files).
- Automatic maintenance removes every verified version older than 24 hours from
  chunks, references, files, and the project registry.
- Project registry writes occur only when project metadata or state changes.
- Progress never compares counters with different meanings or reports a completed
  count greater than its matching total.
- Every indexing run has a durable run ID, trigger, outcome, timing breakdown, and
  skip-reason summary.
- Search, symbol lookup, file outlines, structural references, and refactor analysis
  return equivalent results after compaction and schema migration.
- An incompatible partition rebuilds without losing its project registration.
- Storage benchmarks cover repeated edits, no-op runs, forced rebuilds, deletion,
  maintenance, and schema rebuilds.
- The existing multi-gigabyte set of historical versions can be reclaimed without
  deleting current rows.

## Delivery Overview

The implementation is divided into the following pull requests:

1. Correctness prerequisites for failed indexing and reference backfill.
2. Read-only storage metrics and storage-growth benchmarks.
3. Bounded multi-file commits and no-op registry writes.
4. Automatic and manual table maintenance.
5. Accurate progress and bounded audit history.
6. Scanner and freshness performance improvements.
7. Slim chunk schema, routable chunk IDs, and automatic rebuilds.
8. Registration-overlap controls and multi-project query scaling.
9. Optional vector-precision work, gated by retrieval quality.

## Preparation

Before the first implementation pull request:

- Fetch `origin/main` and create the implementation branch from `origin/main`, not
  from the currently diverged local `main`.
- Preserve the existing local commit and untracked review file.
- Record the exact Git revision in benchmark output and audit records.
- Use isolated `CODE_INDEXING_DATA_DIR` and `CODE_INDEXING_CACHE_DIR` directories for
  every benchmark and migration test.
- Capture baseline measurements for cold indexing, no-op indexing, a single-file
  edit, repeated edits, forced reindexing, file deletion, and cleanup.
- Record physical bytes, logical bytes, row counts, table versions, fragments, index
  statistics, and phase durations after every scenario.
- Do not vacuum the existing live index until maintenance and recovery tests pass.

## PR 1: Correctness Prerequisites

### Goal

Ensure that optimizing or rebuilding storage never builds on an internally divergent
file, chunk, and reference generation.

### Implementation

- Change failed replacement handling in `Indexer._index_scan.stage_failure`.
- When an existing file's new embedding fails, preserve the previous
  `content_hash`, because it describes the chunks and references that remain live.
- Update only the latest observed `size`, `mtime_ns`, error state, error text, and
  attempt timestamp.
- Derive project state from both errors in the current run and stored file errors.
- Prevent a later no-op run from promoting a project with stored file errors from
  `partial` to `ready`.
- Clear a file's stored error only after its replacement chunks and references
  commit successfully.
- Make reference-only backfill preserve the existing project state instead of
  unconditionally choosing `ready` when the backfill itself has no errors.
- Publish the `committing` phase before a reference commit begins.
- Preserve the current staging journal and all-table rollback behavior.
- Replace bare assertions around required reference tables with explicit invariant
  failures that remain active under `python -O`.

### Tests

- A failed replacement retains old chunks and references.
- The retained chunks, references, and file row carry the same indexed content hash.
- Re-running without another source change does not retry embedding.
- A later source edit retries and clears the error after success.
- A no-op run cannot change a partial project with stored errors to ready.
- Reference backfill cannot promote an existing partial project.
- A crash during any commit phase restores files, chunks, and references.
- Existing reference-index hotfix regressions continue passing.

### Primary Files

- `src/code_indexing_mcp/indexing.py`
- `src/code_indexing_mcp/storage.py`
- `src/code_indexing_mcp/staging.py`
- `src/code_indexing_mcp/models.py`
- `tests/test_indexing.py`
- `tests/test_staging.py`
- `tests/test_storage.py`

## PR 2: Storage Metrics and Regression Benchmarks

### Goal

Make storage growth observable before changing physical behavior, and establish
repeatable regression measurements.

### Storage Models

Add versioned models for table, project, and installation storage statistics. Report:

- Snapshot timestamp.
- Project and table identity.
- Current table version.
- Current row count.
- Lance-reported logical bytes.
- Filesystem-reported physical bytes.
- Fragment count and fragment-size distribution.
- Retained version count.
- Oldest and newest retained-version timestamps.
- Index names and types.
- Indexed and unindexed row counts.
- Partition total and installation total.
- A `consistent` flag indicating whether versions remained stable during collection.
- Registered-root overlap warnings.
- Shared Git common-directory warnings for worktrees.

Collect physical sizes without following symlinks. If any table version changes while
statistics are collected, return the observations with `consistent=false` rather than
presenting them as one atomic snapshot.

### User Surfaces

- Add `LanceStore.storage_stats()`.
- Add `Application.storage_status()`.
- Add daemon RPC dispatch and `BrokerApplication` forwarding.
- Add an MCP tool named `index_storage_status` or similarly explicit.
- Add `code-indexing-mcp storage status [project]`.
- Keep all status operations read-only.

### Benchmark Scenarios

Extend the existing benchmark with:

- `cold_start`: first complete index.
- `no_op`: unchanged incremental index.
- `single_file_edit`: one source edit.
- `repeated_edits`: at least 100 edits to one file.
- `forced_reindex`: all eligible files rebuilt.
- `single_file_deletion`: one deleted source file.
- `many_file_deletions`: a bounded group of removed files.
- `post_maintenance`: statistics after cleanup.

Capture table-version deltas and physical growth after each scenario. Version the
benchmark JSON contract so later phases can add metrics without ambiguity.

### Tests

- Statistics work when the optional references table is absent.
- Statistics work for a registered project with no partition.
- Physical-byte accounting does not follow symlinks.
- Concurrent mutation causes `consistent=false`.
- Values are non-negative and current logical bytes do not exceed reported physical
  partition bytes under the test layout.
- Tests avoid exact filesystem byte assertions, which are unstable across filesystems
  and Lance versions.

### Primary Files

- `src/code_indexing_mcp/storage.py`
- `src/code_indexing_mcp/models.py`
- `src/code_indexing_mcp/application.py`
- `src/code_indexing_mcp/daemon.py`
- `src/code_indexing_mcp/server.py`
- `src/code_indexing_mcp/cli.py`
- `src/code_indexing_mcp/benchmark.py`
- `tests/test_storage.py`
- `tests/test_application.py`
- `tests/test_daemon.py`
- `tests/test_server.py`
- `tests/test_cli.py`
- `tests/test_benchmark.py`

## PR 3: Eliminate Per-File Write Amplification

### Goal

Replace one Lance mutation per changed file with bounded mutations over groups of
complete files.

### Staging Changes

- Keep staged Arrow data grouped by file.
- Add iterators that combine complete file groups into bounded commit batches.
- Never split one file's rows across different source batches for a merge whose
  delete condition covers that file.
- Bound batches by file count, row count, and Arrow `nbytes`.
- Start with conservative limits such as 64 files or 64 MiB, then tune from benchmark
  evidence.
- Release each Arrow batch before loading the next one.

### Commit Changes

- Build one affected-file predicate per batch using `file_id IN (...)`.
- Execute one chunk `merge_insert` per non-empty batch.
- Include zero-chunk files in the affected predicate so their previous chunks are
  removed.
- If an entire batch has zero chunks, issue one batched delete instead of an empty
  merge.
- Apply the same strategy to reference rows.
- Batch removed-file deletes for chunks, references, and files.
- Deduplicate replacement and removal IDs before committing.
- Keep the file metadata upsert as one table operation.
- Preserve the single pre-commit `TableVersions` snapshot so any failed batch restores
  the complete previous generation.

### Registry Changes

- Compare a prospective project row with the current row while excluding
  `updated_at`.
- Skip the merge when no semantic project field or state changed.
- Change `updated_at` only for a real mutation.
- Ensure ordinary project discovery and status checks do not churn registry versions.

### Acceptance Gates

- An unchanged run creates zero partition mutations.
- One changed file creates one bounded chunk mutation and one bounded reference
  mutation.
- Five hundred changed files with a 64-file batch limit create no more than eight
  data batches per table, plus bounded index-management operations.
- Search results match those produced by the previous per-file implementation.
- A failure injected before or after any batch restores all three tables.
- Peak parent memory remains within the existing indexing-memory acceptance limit.

### Primary Files

- `src/code_indexing_mcp/staging.py`
- `src/code_indexing_mcp/storage.py`
- `src/code_indexing_mcp/indexing.py`
- `tests/test_staging.py`
- `tests/test_storage.py`
- `tests/test_indexing.py`
- `tests/test_memory_acceptance.py`

## PR 4: Automatic and Manual Maintenance

### Goal

Reclaim old verified versions from active and inactive projects without tying cleanup
to source-file deletion.

### Separate Index Creation from Maintenance

- Split `ensure_indexes()` into index-existence work and physical maintenance work.
- Keep creation of missing FTS and BTree indexes on the write path when required for
  correctness.
- Stop calling `optimize()` after every incremental commit.
- Allow searches to combine indexed rows with an unindexed tail until scheduled
  maintenance incorporates that tail.
- Move compaction, index optimization, and old-version cleanup into explicit
  maintenance methods.

### Maintenance Behavior

- Add `LanceStore.maintain_project()`.
- Add `LanceStore.maintain_registry()`.
- Optimize chunks, references, files, and the project registry.
- Use `cleanup_older_than=timedelta(hours=24)` by default.
- Never set `delete_unverified=True` automatically.
- Acquire the existing global writer lock and project lock before maintenance.
- Automatic maintenance attempts locks without waiting and skips busy projects.
- Persist the last successful maintenance timestamp in a small atomic metadata file.
- Run overdue maintenance after daemon startup.
- Repeat the check at most once per 24 hours.
- Schedule the same work in direct MCP mode.
- Run maintenance regardless of manual, lazy, or eager indexing mode because it does
  not scan source files or create a new logical generation.

### Configuration

Add narrowly scoped settings:

- `CODE_INDEXING_AUTO_MAINTENANCE`, default `true`.
- `CODE_INDEXING_VERSION_RETENTION_HOURS`, default `24`, with a safe bounded range.

Do not expose zero-hour automatic retention.

### Manual Surfaces

- Add `code-indexing-mcp storage vacuum [project] --dry-run`.
- Require an explicit execution flag for actual cleanup.
- Add an equivalent MCP maintenance tool whose default behavior is dry-run.
- Return before and after statistics, versions removed, bytes reclaimed, duration,
  skipped projects, and busy projects.
- Label pre-cleanup reclaimable bytes as an estimate.

### Tests

- Maintenance preserves current rows and search results.
- Versions inside the retention window remain readable.
- Verified versions outside the retention window are removed.
- Busy projects are skipped safely.
- Interrupted maintenance does not corrupt current tables.
- Inactive projects are cleaned even when they are never indexed again.
- Registry version count becomes bounded.
- Automatic maintenance never loads the embedding model.

This pull request is the first release point expected to reclaim most of the observed
multi-gigabyte historical storage.

## PR 5: Accurate Progress and Durable Audit History

### Goal

Make active work understandable and completed work auditable without retaining an
unbounded per-file history.

### Progress Contract

Replace ambiguous counters with counters whose names and denominators match:

- `run_id`
- `trigger`
- `phase`
- `candidates_seen`
- `candidates_total`
- `eligible_files`
- `unchanged_files`
- `changed_files`
- `parsed_files`
- `failed_files`
- `skipped_total`
- `skipped_by_reason`
- `bytes_read`
- `chunks_extracted`
- `chunks_embedded`
- `chunks_staged`
- `staged_bytes`
- `current_path`
- `started_at`
- `updated_at`
- `phase_started_at`

### Progress Rules

- Never compare candidate counts with eligible-file totals.
- Leave totals unset when they are unknown.
- Publish phase changes immediately.
- Throttle ordinary counter updates.
- Report throughput only after enough samples exist.
- Do not report an ETA unless its denominator is stable.
- Keep paths repository-relative.
- Continue using an atomic cross-process snapshot file.
- Remove the live snapshot after completion.
- Use durable history, not the progress snapshot, for completed runs.

### Audit History

Add a small SQLite history database with WAL mode and a busy timeout. Store:

- Run ID and project ID.
- Trigger: `manual`, `startup`, `watcher`, `lazy-query`, `reference-backfill`,
  `schema-rebuild`, or `maintenance`.
- Server version and Git revision when available.
- Embedding model and index schema version.
- Scan-configuration hash.
- Force flag.
- Start and finish timestamps.
- Final state.
- Phase durations.
- File and chunk counts.
- Skip counts by reason.
- Bounded error details and bounded skipped-path samples.
- Backend fallback and worker telemetry.
- Storage before and after the run.

Mark unfinished runs from dead processes as interrupted during startup. Retain at most
100 runs per project and optionally no more than 30 days. Prune in the same transaction
that records a new completed run.

### User Surfaces

- Include live progress and a compact last-run summary in `project_status`.
- Add paginated indexing history through MCP.
- Add corresponding CLI history output.
- Keep expensive storage-directory traversal out of ordinary project status.

### Tests

- The observed case of 119 eligible files plus 1,367 skipped candidates never reports
  progress as `1486/119`.
- Skip reasons distinguish ignored, unsupported, oversized, symlink, binary, encoding,
  unreadable, parse, and embedding failures.
- History pruning remains bounded.
- Concurrent daemon clients cannot corrupt history.
- A killed process leaves an interrupted audit record.
- Project status remains inexpensive and does not load all historical runs.

## PR 6: Scanner and Freshness Performance

### Goal

Avoid full repository walks and ignore processing before every related query.

### Git Repositories

- Enumerate tracked and untracked non-ignored files with
  `git ls-files --cached --others --exclude-standard -z`.
- Filter supported suffixes and configured includes before stat or content work.
- Avoid passing unsupported files through `git check-ignore`.
- Define and test behavior for tracked-but-ignored files, submodules, nested
  repositories, and worktrees.
- Preserve deterministic output.

### Non-Git Repositories

- Stream `os.walk()` results instead of retaining and globally sorting the whole tree.
- Continue pruning hard-excluded and symlinked directories.
- Load nested ignore rules incrementally.
- Keep deterministic order within each directory.
- Bound in-memory candidate and ignore batches.

### Freshness Tracking

- Let eager-mode watchers mark a project dirty immediately.
- Clear dirty state only after a successful index and race-closing reconciliation pass.
- Cache negative freshness checks briefly in lazy and direct modes.
- Invalidate that cache after registration, indexing, removal, scan-configuration
  changes, or watcher events.
- Avoid rescanning a clean project for every tool call in one agent interaction.
- Fix binary and undecodable source-extension files so they do not make a project
  permanently stale.

### Scan Inspection

- Add a paginated dry-run scan tool.
- Allow filtering by outcome or skip reason.
- Return repository-relative paths and explanations.
- Do not embed or mutate the index.
- Do not persist a complete scan manifest by default.

### Acceptance Gates

- Scanner memory scales with the largest directory or configured batch, not total
  repository file count.
- A 100,000-file repository containing mostly unsupported files performs substantially
  less work than the current implementation.
- Repeated searches against a clean repository do not repeatedly invoke a full walk.
- New and modified files remain visible according to existing lazy and eager guarantees.

## PR 7: Slim Chunk Schema and Automatic Rebuild

### Goal

Store one authoritative source-content copy per chunk, route chunk lookups directly,
and rebuild incompatible partitions safely.

### Schema Changes

- Stop persisting `embedding_text`; it is needed only before embedding.
- Replace `search_text` with compact normalized identifier terms.
- Create a multi-field FTS index over `content` and identifier terms.
- Verify multi-field FTS against both the minimum and maximum supported LanceDB
  versions.
- If the minimum supported version cannot provide correct multi-field FTS, either
  raise the LanceDB dependency floor with compatibility evidence or retain
  `search_text` temporarily.
- Remove `project_id` from partitioned chunk rows and inject it from the owning
  partition.
- Remove `content_hash` from chunk rows and obtain it from the files table when
  preserving the existing `get_chunk` response field.
- Keep one persisted copy of source content.
- Keep vectors as `float32` during this migration.
- Do not normalize the reference table until measurements show enough logical savings
  to justify more complex query joins.

### Routable Chunk IDs

- Include a project-routing prefix in newly generated chunk IDs.
- Parse the prefix in `get_chunk()` and open only the owning partition.
- Reject malformed or unknown prefixes with the existing structured not-found error.
- Treat pre-migration chunk IDs as invalid after rebuild, consistent with the current
  contract that chunk IDs change when files are reindexed.

### Automatic Rebuild Framework

Ship the generic rebuild framework before incrementing the schema version:

- Distinguish reconstructable schema/model incompatibility from corrupt registration.
- Mark a project `rebuild_required` instead of permanently raising
  `INDEX_INCOMPATIBLE`.
- Preserve `ProjectInfo` and the `.ci-mcp/project.toml` marker.
- In lazy and eager modes, rebuild before serving a query.
- In manual mode, let the next explicit index command perform the rebuild.
- Acquire writer locks before deleting an incompatible partition.
- Evict cached table handles before removal.
- Delete the old partition rather than retaining a full backup.
- Rebuild one project at a time.
- If rebuilding fails, preserve registration and leave a retryable partial/error state.
- Record rebuild reason and progress in audit history.

After the framework passes rollback tests, increment the schema version and activate
the slim layout.

### Correctness Gates

- Golden semantic and keyword queries return equivalent top results before and after
  migration.
- CamelCase, snake_case, path, and qualified-symbol matching remain covered.
- File outlines, symbol lookup, references, and refactor analysis remain unchanged.
- `get_chunk()` performs one partition lookup.
- Rolling back to the immediately preceding release triggers a rebuild to that
  release's schema rather than leaving an unreadable index.
- Rebuild failure never removes registration.

## PR 8: Registration Overlap and Query Scaling

### Goal

Prevent accidental duplicate indexing and remove linear partition latency from
multi-project reads.

### Registration Controls

- Detect exact, nested, and parent root overlaps during registration.
- Reject new nested overlaps unless `allow_overlap=true` is explicit.
- Do not invalidate existing overlapping registrations.
- Surface overlap warnings through project and storage status.
- Detect worktrees sharing a Git common directory and report them as potential
  duplicates without rejecting them.
- Add `--allow-overlap` to CLI initialization.
- Add a corresponding MCP parameter with a clear description.

### Query Changes

- Push reference-service filters into Lance queries instead of materializing the
  entire reference table.
- Run independent multi-project hybrid searches concurrently with a small bounded
  pool.
- Confirm Lance table handles are safe for concurrent reads; otherwise open per-task
  read handles.
- Preserve deterministic global result ordering.
- Benchmark one, eight, and fifty project scopes.

### Acceptance Gates

- Existing overlapping projects remain usable and clearly identified.
- New accidental overlaps require explicit intent.
- Multi-project latency approaches the slowest partition plus merge overhead rather
  than the sum of every partition, within the concurrency limit.
- Search ranking remains deterministic across runs.

## Optional PR 9: Vector Precision Experiment

### Goal

Reduce the remaining live-data footprint only if retrieval quality remains acceptable.

### Experiment

- Build a fixed retrieval corpus with relevance judgments.
- Compare `float32`, `float16`, and a suitable quantized representation.
- Measure recall at K, rank correlation, hybrid-search latency, index-build time, and
  physical bytes.
- Test both exact and HNSW modes.
- Test every supported LanceDB version if the physical vector type changes.

### Adoption Gate

Adopt lower precision only if:

- Retrieval regression remains within an agreed threshold.
- Search behavior is stable across supported platforms.
- Post-maintenance disk reduction is material.
- Reindex time and memory remain acceptable.

Treat replacing the embedding model as a separate product decision because it changes
retrieval semantics and requires every project to download a model and rebuild.

## Release Sequence

### Release A: Immediate Storage Control

Include correctness fixes, storage metrics, bounded commits, no-op registry writes,
and automatic maintenance. This release should expose and reclaim the majority of the
currently retained historical data without a schema rebuild.

### Release B: Operational Transparency

Include accurate progress, bounded audit history, scanner improvements, freshness
caching, and scan-inspection surfaces.

### Release C: Compact Live Schema

Include automatic rebuild handling, the slim chunk schema, multi-field FTS, routable
chunk IDs, overlap controls, and query concurrency.

### Release D: Optional Vector Reduction

Include lower-precision vectors only if retrieval-quality and compatibility gates pass.

## Rollout and Recovery

- Deploy maintenance before the schema-changing release.
- Run `storage status` against the live installation and save the baseline report.
- Run maintenance in dry-run mode and inspect projects, table versions, and estimated
  reclaimable bytes.
- Execute maintenance under the writer lock.
- Capture and compare the post-maintenance report.
- Monitor daemon logs and audit history for skipped busy projects and retry them later.
- Rebuild incompatible projects one at a time after the schema release.
- Prioritize active projects discovered through MCP roots before inactive
  registrations.
- Keep registration and marker data authoritative so a failed rebuild remains
  recoverable.
- Ensure the release immediately preceding the schema bump already contains generic
  rebuild handling, making binary rollback reconstructive rather than manual.

## Verification Commands

Run after every pull request:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

Also run the targeted storage and operational suites while iterating:

```bash
uv run pytest \
  tests/test_storage.py \
  tests/test_staging.py \
  tests/test_indexing.py \
  tests/test_progress.py \
  tests/test_application.py \
  tests/test_daemon.py \
  tests/test_server.py \
  tests/test_cli.py \
  tests/test_benchmark.py
```

Run isolated cold, repeated-edit, forced-reindex, maintenance, and schema-rebuild
benchmarks before each release. Include crash-injection tests around staged commits,
maintenance, and rebuild boundaries, plus golden search and reference comparisons.

## Deliberately Deferred Work

- Do not change the embedding model before lifecycle and schema measurements are
  complete.
- Do not introduce a global content-addressed vector store until overlap statistics
  show that duplicate live vectors remain a material problem after maintenance.
- Do not normalize structural-reference rows merely to remove repeated metadata until
  logical and index-size measurements justify the query complexity.
- Do not persist complete per-run file manifests by default; provide paginated
  inspection on demand instead.
