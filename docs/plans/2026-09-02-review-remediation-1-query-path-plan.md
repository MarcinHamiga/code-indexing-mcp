# Track 1 — Query-Path Overhead — Implementation Plan

**Goal:** Remove the fixed per-query cost in the application layer. After this track a
lazy-mode tool call on a dirty worktree no longer walks the whole source tree, the
post-operation "did the repository move" check no longer spawns git, multi-project
scopes probe in parallel, and read queries no longer commit a LanceDB write.

**Review findings closed:** perf-major 1 (dirty-worktree walk), 2 (git spawns), 3
(write on read); arch-minor (unsynchronized `_clean_freshness_until`).

**Baseline:** see the index. Green before Step 0 and after every step.

## Decisions settled before implementation

- **D1 — Persist the status fingerprint and the dirty set on the slot.**
  `probe_git_state` already computes `status_fingerprint`, `dirty_paths`, and
  `untracked_paths` (`git_state.py:196-213`); nothing reads the fingerprint. `ProjectSlot`
  (`models.py:1100-1124`) gains `indexed_status_fingerprint: str | None = None` and
  `indexed_status_paths: str | None = None` (a JSON-encoded sorted list of
  `dirty ∪ untracked` relative paths, or `None` when the list exceeds
  `MAX_PERSISTED_STATUS_PATHS = 2000`). Both are written where `indexed_head` and
  `indexed_clean` are written today (`storage.py:622-650`, `set_slot_state` with a
  `git` argument) and cleared where those are cleared (`storage.py:647-648, 922-923,
  942-943`). Both are plain string columns in the `project_slots` arrow schema
  (`storage.py:2026-2027`) so the registry migration is "add nullable columns"; follow
  the existing `_migrate_*` pattern and make an old registry without the columns read
  as `None` (which simply disables the fast path until the next index run).
- **D2 — The fingerprint alone is not proof of currency.** A file that was already
  dirty when indexed and was edited again has the same status line. So the rule is:
  *same HEAD and same fingerprint* ⇒ stat only `dirty ∪ untracked`; *same HEAD,
  different fingerprint, stored path list present* ⇒ stat `stored ∪ current dirty ∪
  current untracked`; *stored list absent* ⇒ full walk as today. Files never listed by
  `git status --untracked-files=all` are either unchanged tracked files or ignored
  files; the scanner already excludes ignored files unless tracked, and a tracked
  file's edit always appears in status. Keep `--untracked-files=all`: `normal`
  collapses untracked directories and would make the path list incomplete.
- **D3 — Subset stat reuses the scanner's classification.** Add
  `Scanner.scan_paths(project, relative_paths, known_files)` that yields
  `ScannedFile | SkippedFile` for exactly those paths using `_classify`
  (`scanner.py:540-580`) with `ignore_specs=[]` and `standard_ignored=False` (the
  candidates come from git, which already applied ignore rules), the same symlink and
  `max_file_bytes` handling as `_scan_batch` (`scanner.py:480-535`), and
  `read_contents=False`. A path that no longer exists yields nothing.
- **D4 — Post-operation change detection reads `.git` directly.** `_target_changed`
  (`application.py:880-886`) currently re-runs the 3-spawn probe per checkout. Replace
  it with `git_state.head_snapshot(state: GitState) -> tuple[SelectorKind, str, str | None] | None`
  that reads `Path(state.checkout_identity) / "HEAD"`; for `ref: <name>` it resolves the
  ref from `<checkout>/<name>`, then `<common>/<name>` (`repository_identity`), then
  `<common>/packed-refs` (lines `<oid> <refname>`, skipping `#` and `^` lines). Any read
  failure, unparseable content, or a ref it cannot resolve returns `None`, and the
  caller falls back to `probe_git_state`. Compare selector kind, selector value, and
  head oid exactly as today.
- **D5 — Parallel resolve for multi-project scopes.** `_resolve_active_targets`
  (`application.py:844-865`) probes each project sequentially. When more than one
  project is requested, run `_resolve_active_target` in a `ThreadPoolExecutor`
  (max 8 workers, module constant) and reassemble in the same sorted order so lock
  ordering and result ordering are unchanged. `_resolve_active_target` already takes a
  per-project file lock, so concurrent activation is safe.
- **D6 — Slot touches are buffered in memory.** `LanceStore.touch_slot`
  (`storage.py:614-620`) becomes: record `slot_id -> time.time_ns()` in
  `self._pending_slot_touches` under `self._touch_lock`; write nothing. New
  `LanceStore.flush_slot_touches()` merges all pending rows in one `_merge` call
  and clears the buffer. `touch_slot` flushes inline when the oldest pending touch is
  older than `SLOT_TOUCH_FLUSH_SECONDS = 300`. Callers that flush explicitly:
  `Application.maintain_storage` and `maybe_run_maintenance` (before they read
  retention), `DaemonServer.serve` in its `finally`, and `LanceStore.close()` if one
  exists (add one if not and call it from the daemon). Every reader of
  `last_used_at` for LRU retention (find it via `branch_cache_limit` /
  `last_used_at` in `storage.py`) overlays the pending timestamps so eviction order is
  correct before a flush. `storage.py:802` (touch inside activation, already a write
  path) may call `flush_slot_touches()` after the touch so activation stays durable.
- **D7 — Freshness cache gets a lock.** `_clean_freshness_until`
  (`application.py:285`) is read and written from `asyncio.to_thread` workers and daemon
  request threads. Wrap every access in a `threading.Lock` (`self._freshness_lock`).
  No semantic change.
- **D8 — Scanner micro-costs, only if trivially safe.** In `_scan_batch`
  (`scanner.py:480-535`) and `_classify`, replace separate `is_symlink()`, `is_file()`,
  and `stat()` calls with one `os.lstat` where the current code does them on the same
  path, and stop calling `include_spec.match_file` in `_classify` when the batch was
  already prefiltered by the same spec. Skip this step entirely if it needs more than
  a local change; it is not the point of the track.

## Steps

**Step 0 — Coordinates.** Re-read `git_state.py:130-215`, `application.py:783-920`,
`application.py:1021-1135`, `application.py:1526-1550`, `storage.py:600-660`,
`storage.py:960-1000`, `storage.py:2020-2035`, `scanner.py:176-260` and `:480-580`,
`models.py:1100-1140`. Confirm D1–D8 against what is there.

**Step 1 — Persist fingerprint and paths (D1).**
Model fields, arrow schema, `_slot_row` / `_slot_from_row` (`storage.py:973-997`),
`set_slot_state`, the clear sites, and a registry migration that adds the two nullable
columns to an existing `project_slots` table. Tests: `tests/test_storage.py` — a slot
written with a git state round-trips both fields; a registry created without the
columns opens and reads them as `None`; a list over the cap stores `None` for paths
but still stores the fingerprint.

**Step 2 — `Scanner.scan_paths` (D3).**
Tests in `tests/test_scanner.py`: a listed path that is eligible yields a
`ScannedFile` with size and mtime; a symlink, an oversized file, an unsupported
extension, and a path excluded by `scan.exclude` yield `SkippedFile` or nothing
exactly as `iter_scan` would; a missing path yields nothing.

**Step 3 — Subset freshness (D2).**
In `_project_status_for_target` (`application.py:1044-1097`) add a branch between the
"HEAD moved" branch and the full `_project_is_stale` branch:

```
elif (git.probe is GitProbeOutcome.GIT
      and slot.indexed_head == git.head_oid
      and slot.indexed_status_fingerprint is not None
      and git.status_fingerprint is not None):
    candidates = set(git.dirty_paths) | set(git.untracked_paths)
    if slot.indexed_status_fingerprint != git.status_fingerprint:
        stored = decode(slot.indexed_status_paths)   # None -> fall through to full walk
        if stored is None: <full walk branch>
        candidates |= stored
    stale = self._paths_are_stale(resolved, existing_files, candidates, partition_id=...)
```

`_paths_are_stale` compares `scan_paths` output against the stored `StoredFile`
records the way `_project_is_stale` does (presence, size, mtime, language), and also
reports stale when a candidate path is present in `existing` but yields nothing from
`scan_paths` (deleted or newly ineligible). Dirty paths from git are relative to the
project prefix already (`_parse_status(..., project_prefix)`); confirm and normalise
to posix. Tests in `tests/test_application.py` with a real `git init` fixture
(`shutil.which("git")` guard, as other tests do): (a) index a repo, dirty one file,
index again, then assert `project_status` is `ready` and that
`Scanner.iter_scan` was **not** called (monkeypatch it to raise); (b) edit the
already-dirty file again → `stale`; (c) revert the dirty file to HEAD → `stale`
(exercises the stored path list); (d) create a new untracked file → `stale`;
(e) an old slot without a fingerprint takes the full-walk path unchanged.

**Step 4 — `head_snapshot` and `_target_changed` (D4).**
Tests in `tests/test_git_state.py`: loose ref, packed ref, detached HEAD, unborn
branch (returns `None` head oid), linked worktree (`.git` file → `checkout_identity`
is the worktree gitdir; refs live in `repository_identity`), and a corrupt `HEAD`
returning `None`. In `tests/test_application.py`: `_run_repository_stable_query`'s
retry still fires when HEAD moves between resolve and post-check (write a new commit
inside the operation callable), and `probe_git_state` is called exactly once per
project per query when nothing moves (count via monkeypatch).

**Step 5 — Parallel resolve (D5).**
Test: three registered projects resolve in `<` sequential time with a probe that
sleeps 50 ms (monkeypatched), and results are keyed and ordered as before.

**Step 6 — Buffered slot touches (D6).**
Tests in `tests/test_storage.py`: after N `touch_slot` calls the `project_slots`
table version is unchanged; `flush_slot_touches` writes one version; LRU eviction
under `branch_cache_limit` respects an unflushed touch; a touch older than the flush
interval flushes inline. In `tests/test_daemon.py`: stopping the daemon flushes.

**Step 7 — Freshness lock (D7), scanner micro-costs (D8).**

**Step 8 — Measure.** Add a small `tests/test_query_path_overhead.py` (or extend the
benchmark) that indexes a `git init` fixture with 200 files, dirties one, and asserts
that a `search_code` call after the initial status check performs: zero `iter_scan`
calls, at most one `probe_git_state` per project, and zero `project_slots` version
bumps. This is the regression guard for the whole track.

**Step 9 — Docs.** Update the `index_project` / `search_code` tool descriptions in
`server.py` only if their wording about freshness is now wrong (the "clean cached slot
at the same HEAD costs no scan" sentence is still true and can gain "and a dirty
checkout re-checks only its changed files"). Add a short "Freshness" note to README
where the lazy-mode refresh behaviour is described.

## Out of scope

A probe cache shared between the status check and the query (which would remove
another ~3 spawns per call) is deliberately not in this track: the double resolve is
what detects mid-request HEAD moves, and the correctness argument for caching it
deserves its own plan.

## Completion note (2026-09-02)

All nine steps implemented as planned, D1–D8 followed as written. Baseline green
at the end: `ruff format`/`ruff check`/`mypy src` clean, `pytest -n auto` **1640
passed, 8 skipped** (1618 passed, 8 skipped before this track — 22 new tests).

Notable implementation choices, none a deviation from the decisions:

- Step 3 (D2/D3) is implemented as two helpers, `Application._subset_stale_candidates`
  (returns the candidate path set or `None` for "do a full walk") and
  `Application._paths_are_stale`, invoked from one unified branch in
  `_project_status_for_target` rather than as a second literal `elif` between the
  existing branches. Same conditions, same order, less duplicated cache-bookkeeping.
- D8 (scanner micro-costs) was only partly done, per the plan's own "skip if it needs
  more than a local change" clause: `_classify` now does one `os.lstat` instead of
  separate `is_symlink()`/`is_file()`/config calls (purely local, safe). Skipping the
  redundant `include_spec.match_file` re-check was *not* done — `scan_paths` (Step 2)
  calls `_classify` on paths that were never prefiltered by `include_spec` (they come
  from `git status`, not from the prefiltered git/walk batches `iter_scan` uses), so
  threading an "already filtered" flag through `_scan_candidates`'s two call paths
  would be a correctness-sensitive, multi-method change — exactly what that clause
  says to skip.
- `set_slot_state` keeps its existing signature (`project: ProjectInfo | None`, which
  internally probes with `include_status=True`) rather than taking a `GitState`
  argument as D1's parenthetical suggested; it already computes everything D1 needs
  from that same probe, and this call happens once per index run, not per query, so
  there was no query-path cost to remove there.
- `remove_project`'s freshness-cache eviction loop (duplicate of `invalidate_freshness`)
  was folded into a call to `invalidate_freshness` while adding the D7 lock, since both
  needed the same lock-guarded logic.

Nothing was left undone.
