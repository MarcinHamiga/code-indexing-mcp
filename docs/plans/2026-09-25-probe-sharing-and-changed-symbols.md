# Git probe sharing and `changed_symbols` (2026-09-25)

Two small items shipped together: the "obvious next step" from
`2026-09-02-query-path-profiling-shipped.md`, and the `changed_symbols` entry from
`2026-08-27-feature-backlog.md`.

## 1. Query reuses the status check's Git probe

A lazy tool call runs `project_status` (a status probe: `rev-parse`, `symbolic-ref`,
`rev-parse HEAD`, `git status`) and then the query, which resolved its target with a
second three-command probe of the same checkout a few milliseconds later.

`Application._probe_git` now keeps each checkout's last successful probe for
`GIT_PROBE_REUSE_SECONDS` (5 s). A plain probe reuses it when `head_snapshot` -- the
spawn-free read of `HEAD` and its ref that `_target_changed` already relies on -- still
reports the same selector and HEAD OID. The reused state has its status fields cleared, so
callers get exactly what a fresh plain probe returns. Status probes always run fresh, since
their cleanliness answer is the freshness check itself. Non-Git and degraded probes are
never reused.

A HEAD move between the status check and the query invalidates the entry (the snapshot
differs), and a move during the query is still caught afterwards by `_target_changed`,
whose retry re-probes for the same reason.

**Measured** with `scripts/profile_query_path.py` (constants pointed at this repository:
291 files, 5,513 chunks; M-series Mac, `main` at `467198f` vs the branch; medians):

| Scenario | main ms | branch ms | git spawns |
|---|---:|---:|:--:|
| `clean.burst.search_code` | 119.3 | 91.9 | 7 → 4 |
| `clean.gapped.search_code` | 124.6 | 96.9 | 7 → 4 |
| `dirty.gapped.search_code` | 131.1 | 98.7 | 7 → 4 |
| `multi8.burst.search_code` | 302.4 | 232.8 | 56 → 32 |
| `multi8.gapped.search_code` | 314.4 | 239.6 | 56 → 32 |
| `multi8.dirty.gapped.search_code` | 328.2 | 243.7 | 56 → 32 |

Spawns cost about 9 ms each here, against 29 ms on the Django run; the saving scales with
per-spawn cost. The release gate's `dirty.gapped` spawn bound (≤ 7) now holds at 4.
`tests/test_query_path_overhead.py` pins the query at zero probes after a status check.

Still open: `project_status` itself spawns the three identity commands before
`git status`. Reusing the memo there too (only `git status` fresh) would bring a
steady-state call to one spawn; it was left out to keep this change to the query path.

## 2. `changed_symbols` tool

`changed_symbols(project, since=None, since_time=None, include_untracked=True, limit=100)`
lists files that differ from a base commit in the working tree and the indexed declarations
each change touches. MCP tool, `syndex changed-symbols` CLI command, daemon protocol 6.

- **Base.** `since` is any commit-ish (default `HEAD`: uncommitted work only), verified
  with `rev-parse --verify <rev>^{commit}`; a leading `-` is rejected. `since_time` uses
  `rev-list -1 --before=<time> HEAD`. Both together, or an unresolvable base, is
  `INVALID_FILTER`; a non-Git project is `UNSUPPORTED_OPERATION`.
- **Files.** `git diff --name-status -z --no-renames --relative <base>` plus
  `ls-files --others --exclude-standard`, so a subdirectory project sees only its subtree
  with project-relative paths. Sorted by path and cut at `limit` (≤ 500) before any patch
  is read, so a distant base costs one listing plus a bounded patch.
- **Lines.** One `git diff --unified=0` over the selected files (`:(literal)` pathspecs,
  with prefixes, quoting, colour, external drivers and textconv pinned). New-side hunk
  ranges become `changed_lines`; pure deletions become deletion points that touch only
  a declaration spanning both neighbouring lines. `---`/`+++` are only read in section
  headers, since a removed `-- comment` line reads `--- comment` inside a hunk.
- **Symbols.** One batched outline read (`outline_chunks_for_paths`) whose entries span
  every part of a split declaration (`_outline_items(span_parts=True)`); `file_outline`
  output is unchanged. Added and untracked files, and files with no line ranges (binary),
  touch every declaration; deleted files list none; mode-only changes touch none.
- **Honesty.** `index_current` compares the stored content hash with the file on disk,
  so a stale index is flagged per file rather than silently mismatched. The MCP tool
  refreshes first in lazy mode; the CLI reads the index as it is.

Not in this version: symbols removed by the change (they are gone from the index of the
current tree), rename tracking, and multi-project scope.
