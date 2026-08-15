# Storage, Performance, and Transparency Remediation: Shipped

Closure validation for the nine-pull-request remediation plan, run 2026-08-15 at
revision `9c2df35` (`origin/main`, PR 9 merged). Every success criterion in the plan
was re-checked against the merged code, the benchmark suites, and the live
installation. No regressions were found; no source changes were required.

## Verification gates

- `ruff format`: 110 files unchanged.
- `ruff check`: clean.
- `mypy src` (strict): clean, 56 files.
- `uv run pytest -n auto`: 1369 passed, 8 skipped (environment-gated: MLX extra,
  prepared accelerator runner, and real-model memory/integration caches).

## Storage-growth benchmark (contract v2, isolated workspace)

`benchmark index --files 128 --functions-per-file 2`, real model, cold cache state:

| Scenario | Table-version delta | Physical after |
| --- | --- | --- |
| `cold_start` (incl. warmup) | files +2, chunks +9, references +10 | 853,994 B |
| `no_op` | 0 / 0 / 0 — zero mutations | 853,994 B |
| `single_file_edit` | +1 per table | 887,553 B |
| `repeated_edits` (100) | +100 per table — one bounded batch per run | 16,871,161 B |
| `forced_reindex` (128 files) | files +1, chunks +2, references +2 | 17,753,352 B |
| `single_file_deletion` | +1 per table | 17,761,992 B |
| `many_file_deletions` (8 files) | +1 per table | 17,773,692 B |
| `post_maintenance` | 0 removed, 0 reclaimed (estimate 16.9 MB) | 18,768,979 B |

- `no_op` creates zero partition mutations: the unchanged run is a true no-op.
- A forced rebuild of all 128 files costs 1–2 versions per table — O(batches),
  not O(files). Deletions batch identically.
- `post_maintenance` removing nothing is correct: the whole run is younger than
  the 24-hour retention window, and the reclaimable figure is the labeled
  estimate (physical minus logical), an upper bound by design.
- Per-edit distribution over 100 edits: median 735 ms, p95 1,282 ms.

## Multi-project search benchmark

`benchmark search --projects 50 --iterations 3`: deterministic global ordering at
every scope; median latency 28.8 ms (1 project), 122 ms (8), 756 ms (50). Latency
grows sublinearly (~26× for 50× projects) within the bounded concurrency pool.

## Vector-precision benchmark

`benchmark precision` (240 passages, 5 iterations, gates recall ≥ 0.99 and rank
correlation ≥ 0.95 against the float32-exact reference):

| Variant | Recall@8 | Rank corr. | Physical | Gate |
| --- | --- | --- | --- | --- |
| float32 exact | 1.000 | 1.000 | 819,827 B | pass |
| float16 exact | 1.000 | 1.000 | 450,954 B (−45%) | pass |
| float32 HNSW SQ8 | 0.969 | 0.902 | 1,028,348 B | fail |
| float16 HNSW SQ8 | 0.938 | 0.862 | 657,295 B | fail |

This is the measured basis for PR 9's float16 storage default: exact search keeps
full retrieval quality at roughly half the bytes, while quantized HNSW fails the
adoption gate and stays off the default path.

## Live installation validation

Against the live per-user installation (10 registered projects, daemon running
the same revision, no isolated directories):

- Baseline at session start: 144.0 MB total physical; project registry at 978
  retained versions; this repository's partition at 51.1 MB with 81–91 versions
  per table and FTS coverage at 2,441 of 3,488 chunk rows.
- **Automatic rebuild observed live**: the first query against this repository's
  schema-4/float32 partition rebuilt it to schema 5 + float16 (`index schema
  version 4 -> 5; vector storage float -> halffloat`) before serving, preserving
  the project registration, re-embedding all 177 files / 3,488 chunks in 64.1 s,
  and recording a `schema-rebuild` audit row with trigger, reason, and phase
  timings — the PR 7 framework working end to end on production data.
- **Automatic maintenance observed live** (no manual trigger): total 144.0 →
  124.7 MB, registry 978 → 214 retained versions, FTS coverage back to
  3,488/3,488.
- Manual `storage vacuum` dry-run: 28.0 MB reclaimable estimate; `--execute`
  then removed 0 versions and reclaimed 0 bytes in 676 ms with zero busy or
  failed projects — the correct steady state, because automatic maintenance had
  already removed every verified version older than 24 hours minutes earlier.
- **Golden serving-surface equivalence**: `search_code`, `find_symbol`,
  `file_outline`, `get_chunk`, `find_references`, and `analyze_refactor` outputs
  captured before and after the maintenance/vacuum boundary are byte-identical.
- The daemon kept serving throughout, including reads of the rebuilt schema-5
  partition immediately after the in-process rebuild.
- Overlap and shared-worktree warnings are present and correct for the two
  stale worktree registrations (they await their own lazy rebuilds).

## Success criteria: where each is evidenced

1. *An unchanged run creates zero versions* — `no_op` scenario above;
   `test_a_noop_run_creates_zero_partition_mutations`.
2. *Forced reindex creates O(batches) versions* — 128-file rebuild at +1/+2 per
   table; `tests/test_staging.py` batch tests.
3. *Maintenance removes verified versions older than 24 h from all tables and
   the registry* — live pass (registry 978 → 214); `tests/test_storage.py`
   retention tests; benchmark `post_maintenance` confirms nothing newer is
   touched.
4. *Registry writes only on real metadata/state change* — registry version count
   stayed flat through queries and status collection in this session;
   `test_storage.py` no-op upsert tests.
5. *Progress counters never mix meanings* —
   `test_progress.py` (119-eligible/1,367-skipped regression included).
6. *Every run has a durable audit row* — live `schema-rebuild`, `watcher`, and
   `manual` rows inspected through `history`; `tests/test_history.py`.
7. *Search and reference surfaces equivalent after compaction and migration* —
   byte-identical golden set across the schema-4→5 migration's maintenance
   boundary; golden migration tests in `tests/test_storage.py`.
8. *An incompatible partition rebuilds without losing registration* — observed
   live; `test_index_rebuilds_a_partition_written_by_an_incompatible_model`.
9. *Storage benchmarks cover edits, no-ops, rebuilds, deletion, maintenance* —
   the eight-scenario contract v2 above (schema rebuilds additionally covered by
   the live migration and `tests/test_indexing.py`).
10. *Historical multi-gigabyte storage reclaimable without losing current rows* —
    the live installation fell from ~6.0 GiB (plan's baseline measurement) to
    124.7 MB under releases A–D with all ten projects registered and serving.

## Residual state, deliberately left alone

- Two stale worktree registrations remain on the pre-PR7 schema; the lazy
  rebuild path handles them on their next query, which is the designed
  behavior rather than something to fix by hand.
- The reclaimable-bytes estimate exceeds what a given vacuum can reclaim
  whenever logical (uncompressed) bytes understate compression; it stays an
  explicitly labeled estimate.
- Deferred work from the plan (content-addressed vector store, reference-row
  normalization, per-run manifests, embedding-model change) remains deferred.
