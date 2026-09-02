# Query-Path Profiling on a Real Repository — Results (2026-09-02)

Plan: `2026-09-02-query-path-profiling-plan.md`. Harness: `scripts/profile_query_path.py`.
Raw output: `data/2026-09-02-query-path-profile-baseline-4e7a8b4.json` and
`data/2026-09-02-query-path-profile-branch-c2cd048.json`.

**Setup.** Django at `5180f82` (shallow clone, 3,122 indexed files, 49,618 chunks), Apple
M4 Pro, in-process `Application` with the broker off, lazy mode, `exact` vector index.
Baseline is the review's commit `4e7a8b4`; branch is `review-remediation` at `c2cd048`.
The branch run seeded its data directory from the baseline's, so both revisions served
the same index; the branch's registry migration and incremental refresh took under a
second. Each call reproduces what the MCP server does for a lazy query: `project_status`
per project in scope (gathered in parallel), `index_project` for any that reports
stale, then the query. Burst = ten back-to-back calls; gapped = five calls each after a
six-second pause, past the five-second freshness cache both revisions keep. Medians.

## Numbers

| Scenario | Baseline ms | Branch ms | git spawns | data files changed |
|---|---:|---:|:--:|:--:|
| `clean.embed_query` (query embedding alone) | 7.9 | 7.1 | 0 → 0 | 0 → 0 |
| `clean.burst.status_only` | 227.5 | 150.0 | 7 → 4 | 4 → 0 |
| `clean.burst.search_code` | 352.6 | 226.8 | 13 → 7 | 8 → 1 |
| `clean.gapped.search_code` | 358.7 | 246.4 | 13 → 7 | 8 → 1 |
| `clean.burst.find_symbol` | 345.9 | 208.7 | 13 → 7 | 8 → 1 |
| `clean.burst.file_outline` | 338.1 | 204.5 | 13 → 7 | 8 → 1 |
| `dirty.first.search_code` (notices and refreshes one edited file) | 2,073 | 1,607 | 31 → 23 | 41 → 38 |
| `dirty.burst.search_code` | 374.9 | 237.8 | 15 → 7 | 8 → 1 |
| `dirty.gapped.search_code` | **798.3** | **253.9** | 15 → 7 | 8 → 1 |
| `head.first.search_code` (edit committed, HEAD moved) | 1,476 | 1,214 | 30 → 24 | 31 → 22 |
| `head.gapped.search_code` | 381.5 | 260.9 | 13 → 7 | 8 → 1 |
| `head_return.first.search_code` (reset back) | 1,730 | 1,605 | 30 → 24 | 39 → 30 |
| `multi8.burst.search_code` (eight small projects) | **1,488.7** | **443.6** | 104 → 56 | 58 → 9 |
| `multi8.gapped.search_code` | 1,518.7 | 449.7 | 104 → 56 | 57 → 8 |
| `multi8.with_repo.burst.search_code` (nine, incl. Django) | 1,816.0 | 594.8 | 117 → 63 | 64 → 9 |
| `multi8.dirty.first.search_code` | 2,013 | 831.5 | 122 → 72 | 88 → 45 |
| `multi8.dirty.gapped.search_code` | 1,619.8 | 469.8 | 106 → 56 | 57 → 8 |

The edit was picked up by the first call after it on both revisions (`noticed=True` in
the JSON), so the `dirty.*` rows compare like with like.

## Verdict per claim

1. **Dirty worktree no longer walks the tree.** Baseline paid +440 ms on every call
   after the cache expired while the tree stayed dirty (`dirty.gapped` 798 ms against
   `clean.gapped` 359 ms). On the branch the two are within 8 ms of each other. Closed.
2. **Post-operation change check no longer spawns git.** Per single-project search the
   spawn count fell from 13 to 7. The three that disappeared were the second
   `rev-parse`/`symbolic-ref`/`rev-parse` probe after the query. Closed.
3. **Reads no longer write LanceDB.** Baseline changed 8 files per read (registry
   versions from the slot touch, plus the lock). The branch changes exactly one: the
   mtime of `locks/partition-slot-<id>.lock`, which is the per-project file lock being
   taken, not a table write. Closed.
4. **Multi-project scopes resolve in parallel.** Eight projects went from 1,489 ms to
   444 ms, and adding Django to the scope adds 150 ms rather than 330 ms. Closed.

## What the remaining cost is

On the branch a clean single-project search spends about 206 ms of its 227 ms in seven
git spawns at roughly 29 ms each on this machine (cProfile, `dirty.gapped` on the branch,
in the JSON). They are:

- `project_status`: `rev-parse --git-common-dir`, `symbolic-ref`, `rev-parse HEAD`,
  `status --porcelain` (4);
- the query's own `_resolve_active_target`: the same three-command probe again (3).

The query re-probes a checkout that `project_status` probed a few milliseconds earlier
in the same tool call. Sharing that probe between the status check and the query (the
server has both in hand) is worth about 90 ms per call and 24 spawns per eight-project
scope, and is the obvious next step. The query embedding is 7 ms; the search itself is
about 20 ms.

## Release gate

Re-run both `scripts/profile_query_path.py` invocations from the plan with the release
candidate on `PYTHONPATH` before tagging. The numbers that must hold:

- `dirty.gapped.search_code`: git spawns ≤ 7, data files changed ≤ 1 (the lock only),
  median within 1.1× `clean.gapped.search_code`.
- `multi8.burst.search_code`: median under one third of the baseline row above.

## Two harness lessons

- Driving `Application.search_code` directly measures the query, not the tool call: the
  lazy freshness check lives in the server (`_wait_for_startup_projects`), which calls
  `project_status` and then `index_project` before the query. The first two runs of
  this harness missed it and reported that edits were never noticed.
- Both revisions cache a clean freshness verdict for five seconds
  (`FRESHNESS_CACHE_SECONDS`), so back-to-back calls never re-walk the tree. Measuring
  the review's finding needs calls spaced past that window, hence the gapped rows.
