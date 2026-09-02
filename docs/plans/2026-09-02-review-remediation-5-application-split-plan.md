# Track 5 — Application Split — Implementation Plan

**Goal:** Take the two most cohesive responsibilities out of `Application`
(`application.py`, 2128 lines, 70 methods) without changing any caller-visible
behaviour: backend and accelerator selection with calibration, and storage
maintenance. Clean up the sideways imports and private reach-throughs the review
listed. Fold the three environment reads that bypass `IndexSettings` into it.

**Review findings closed:** arch-major (god facade), arch-minor (sideways imports,
store private reach-through), arch-minor (config read outside `IndexSettings`).

**Baseline:** see the index. This track is a refactor: the full suite must pass with
test edits limited to private-attribute paths that moved.

## Decisions settled before implementation

- **D1 — `BackendCoordinator` (new module `backend_coordinator.py`).** Owns what
  `Application.__init__` sets up at `application.py:255-285` and the methods at
  `:332-638`: `serving_providers`, `accelerator_environment`, `backend_selection`,
  `probe_cache`, `_probe_key`, `_runtime_fallback`, `embedding_batch_size`,
  `batch_calibration`, and `_select_backend`, `_runs_externally`,
  `_accelerator_launcher`, `effective_backend_selection`, `_remember_fallback`,
  `_build_probe_key`, `_cpu_probe_key`, `_cpu_max_items`, `_measurements`,
  `crossover_characters`, `_measured_crossover`, `_recommended_override`,
  `_passage_session_factory`, and the backend half of `model_status`. Constructor:
  `BackendCoordinator(paths, settings, embedder)`. `Application` exposes it as
  `self.backends` and keeps thin delegating members for everything public that
  `server.py`, `cli.py`, `daemon.py`, `indexing.py`, or `benchmark.py` reference
  (`grep -n "backend_selection\|embedding_batch_size\|batch_calibration\|effective_backend_selection\|crossover_characters\|model_status\|_passage_session_factory"` across `src/`
  first and list them in the plan comment). Tests that reach `app._remember_fallback`,
  `app._build_probe_key`, `app._accelerator_launcher`, `app._cpu_probe_key`
  (`tests/test_application.py`, `test_backends.py`, or wherever `grep` finds them)
  move to `app.backends.<name>`; assertions unchanged.
- **D2 — `MaintenanceService` (new module `maintenance.py`).** Owns the module-level
  helpers at `application.py:128-182` (`_read_maintenance_timestamp`,
  `_write_maintenance_timestamp`, `_estimate_reclaimable`, `_versions_removed`,
  `_rate`) and the methods `storage_status` (`:1162-1211`), `maintain_storage`
  (`:1213-1475`), and `maybe_run_maintenance` (`:1475-1511`). It needs the store, paths,
  settings, history, and two things only `Application` can provide: target
  resolution and locks. Pass those as explicit callables in the constructor
  (`resolve_targets: Callable[[Sequence[ProjectInfo], bool], Mapping[str, Sequence[ActiveIndexTarget]]]`,
  `root_lock: Callable[[str], FileLock]`) rather than the whole `Application`, so the
  dependency is visible and the service is constructible in tests with lambdas.
  `Application.storage_status`, `maintain_storage`, `maybe_run_maintenance` become
  one-line delegates with identical signatures. While moving `maintain_storage`,
  split its body into `_plan(...)` (the dry-run computation) and `_execute(plan, ...)`
  (the locked mutation) only if the split falls out naturally from the existing
  dry-run branch; otherwise leave the body intact — moving is the goal, not
  restructuring.
- **D3 — Sideways imports.** `reference_service.py:15` imports
  `REFERENCE_SCHEMA_VERSION, _digest` from `indexing`; move both to `models.py`
  (`REFERENCE_SCHEMA_VERSION` next to the other schema constants; `_digest` becomes a
  public `content_digest`) and re-export from `indexing` for one release so nothing
  else breaks. `indexing.py:65` imports `checkout_head` from `update_check`; move it to
  `git_state.py` and have `update_check` import it from there.
- **D4 — Store privates that are a contract become public.**
  `LanceStore._active_partition_ref` and `_chunk_project_id` are called from
  `application.py:796, 1749, 1772, 1985`, `reference_service.py:508, 701`,
  `search.py:198`. Rename to `active_partition_ref` and `chunk_project_id`; keep the
  underscored names as deprecated aliases for one release with a comment; update all
  call sites.
- **D5 — Environment reads fold into `IndexSettings`.** `CODE_INDEXING_OFFLINE`
  (`application.py:223`) becomes `IndexSettings.offline: bool` parsed in
  `from_environment` with the same accepted values; `RuntimePaths.from_environment`
  (`:190-196`) stays (paths are not indexing settings) but gains a docstring saying so.
  Any other `os.environ` read in `src/code_indexing_mcp/*.py` outside `settings.py`,
  `application.py:190-196`, and `update_check.py` is listed in the completion note
  with a reason it stays.
- **D6 — Nothing else moves.** Project registry resolution, freshness, and the
  reference-index lifecycle stay in `Application` for now; they are the next
  candidates and get their own plan.

## Steps

**Step 0 — Coordinates and reference list.** Re-read `application.py:100-300`,
`:332-640`, `:1160-1515`; grep every external reference to the members in D1 and D2;
record the list at the top of each new module's docstring.

**Step 1 — D1.** Move, delegate, update the four private test paths. Baseline.

**Step 2 — D2.** Move, delegate. Baseline. `tests/test_application.py` maintenance
tests should pass without edits; if one constructs state through a private, adjust
the path only.

**Step 3 — D3, D4.** Baseline.

**Step 4 — D5.** Test in `tests/test_settings.py`: `CODE_INDEXING_OFFLINE=1`, `true`,
`yes` set `offline`; anything else is `False`; `Application` built with an explicit
`IndexSettings(offline=True)` never reads the variable (monkeypatch `os.environ` to
raise on access to that key).

**Step 5 — Size check.** `application.py` should land near 1300 lines. Record the
before/after line and method counts in the completion note.

## Completion note (2026-09-02)

Implemented all five steps against the tree as it stood after Tracks 1-4
(`application.py` at 2499 lines / 94 `def`-lines including nested closures, 70
methods on `class Application`). Baseline green throughout:
`ruff format . && ruff check . && mypy src && pytest -n auto -q -p no:cacheprovider`
→ **1736 passed, 9 skipped** (up from 1722/9 at the start of this track — the
increase is the D5 test additions).

**Sizes.** `application.py`: 2499 → **1838 lines**, 70 → **70 methods** on
`class Application` (delegates replaced what moved, so the method *count* is
unchanged even though the class shrank by 661 lines; each delegate is 1-3
lines versus the multi-line implementations they replaced). New modules:
`backend_coordinator.py` — 415 lines, `BackendCoordinator` with 15
methods/properties. `maintenance.py` — 544 lines, `MaintenanceService` with 5
methods plus 4 module-level helpers and a `_StableQueryRunner` Protocol.

**Step 1 (D1).** Moved the constructor's backend/accelerator/calibration setup
and all methods from `_select_backend` through `model_status` into
`BackendCoordinator(paths, settings, embedder)`, constructed as
`self.backends` in `Application.__init__`. Kept delegating properties/methods
on `Application` for every public member referenced elsewhere:
`backend_selection`, `effective_backend_selection`, `embedding_batch_size`,
`batch_calibration`, `serving_providers`, `accelerator_environment`,
`probe_cache` (properties), `crossover_characters()`, `model_status()`
(methods) — used by `cli.py:model_status`, `daemon.py:model_status`,
`benchmark.py:effective_backend_selection`, and by
`tests/test_application.py` directly. The four private test paths
(`_remember_fallback`, `_build_probe_key`, `_accelerator_launcher`,
`_cpu_probe_key`) were updated in `tests/test_application.py` to
`app.backends.<name>`; assertions unchanged.

*Deviation:* `_passage_session_factory` reads `self.indexer.segment_plan` at
call time, but `Indexer` is constructed after the factory in `__init__` and
`BackendCoordinator` cannot hold a reference to `Application.indexer` without
a cycle. Added a `segment_plan: Callable[[], SegmentPlan]` parameter; the
caller passes `lambda: self.indexer.segment_plan`. No behaviour change — the
read is still lazy, at the same point in the run.

**Step 2 (D2).** Moved `_read/_write_maintenance_timestamp`,
`_estimate_reclaimable`, `_versions_removed`, `storage_status`,
`maintain_storage`, `maybe_run_maintenance`, and the
`MAINTENANCE_CHECK_INTERVAL`/`MAINTENANCE_TIMESTAMP_FILE`/`MAINTENANCE_LOCK_FILE`
constants into `maintenance.py`. `Application.maintenance` is constructed with
`store`, `paths`, `settings` directly plus five callables (`list_projects`,
`resolve_project`, `resolve_active_target`, `resolve_active_targets`,
`run_repository_stable_query`) bound to the corresponding `Application`
methods/lambdas. `storage_status`/`maintain_storage`/`maybe_run_maintenance`
on `Application` are now one-line delegates with identical signatures.
`tests/test_application.py::test_scheduled_maintenance_is_serialized_across_applications`
patched `app.maintain_storage`/`other.maintain_storage`; since
`maybe_run_maintenance` now calls `maintain_storage` on the
`MaintenanceService` instance (not on `Application`), the patch targets moved
to `app.maintenance.maintain_storage`/`other.maintenance.maintain_storage` —
the exact "adjust the path only" case Step 2 anticipated.

*Deviations from the plan text, each because the code did not support the
literal instruction:*
- **`history` dropped.** The plan says `MaintenanceService` needs "the store,
  paths, settings, and history." Grep confirms none of the three moved
  methods reference `self.history`; wiring an unused dependency would be dead
  code the baseline's `ruff`/ `ty` would have no reason to flag but that
  serves no one. Omitted.
- **`root_lock` callable dropped.** The plan specifies a
  `root_lock: Callable[[str], FileLock]` dependency. The moved methods build
  every `FileLock` directly from `paths.data / "locks" / ...`, which
  `MaintenanceService` already has via its own `paths`; there is no call site
  that needed a lock *resolution* callable. Omitted.
- **"Two callables" became five.** `target resolution`, as the plan names it,
  is not one operation in the actual code: `storage_status`/
  `maintain_storage` call `list_projects()`, `_resolve` (single-project,
  by-selector), `_resolve_active_target` (single, with a `lock_held` flag
  that must stay `True` inside the already-locked mutation path — routing it
  through the bulk resolver would attempt a second, non-reentrant lock
  acquisition and change behaviour), `_resolve_active_targets` (bulk, for the
  dry-run path), and `_run_repository_stable_query` (the retry-on-repository-change
  wrapper `storage_status` reads through, itself used by every other query
  method that stays in `Application` per D6, so it cannot move). All five are
  passed as explicit callables in `MaintenanceService.__init__`, matching the
  plan's intent ("visible dependency, constructible in tests with lambdas")
  more literally than its two-name summary allowed for.
- **`Application._primary_target` duplicated, not moved.** It is a stateless
  8-line staticmethod called from ~10 other `Application` query methods that
  D6 keeps in place, so it cannot be relocated to `maintenance.py` without
  either an import cycle (`application.py` already imports `maintenance.py`)
  or promoting a lookup to a dependency callable for no benefit. A private
  mirror `_primary_target` was added to `maintenance.py` with a docstring
  explaining why.

**Step 3 (D3, D4).**
- `REFERENCE_SCHEMA_VERSION` and `_digest` (now public `content_digest`) moved
  from `indexing.py` to `models.py`. `indexing.py` re-exports
  `REFERENCE_SCHEMA_VERSION` (`from .models import REFERENCE_SCHEMA_VERSION as REFERENCE_SCHEMA_VERSION`,
  the explicit form `mypy --strict`'s `no_implicit_reexport` requires) for
  `daemon.py`'s `from .indexing import REFERENCE_SCHEMA_VERSION`, and imports
  `content_digest as _digest` so its own ~10 call sites needed no rename.
  `reference_service.py` now imports both directly from `models.py`.
- `checkout_head` (plus its private helpers `_git_directory`, `_reference_sha`,
  `_rev_parse`) moved from `update_check.py` to `git_state.py`.
  `update_check.py` imports it back (`as checkout_head`, same explicit-reexport
  reason) so `update_check.checkout_head(...)` keeps working for `cli.py`,
  `daemon.py`, `benchmark.py`, and `installer/update.py`. *Deviation:* the
  moved `_rev_parse` calls `subprocess.run` directly instead of reusing
  `update_check._run_git` — `_run_git` is still needed by `update_check.check_remote`
  and importing it back from `update_check` would recreate the cycle the move
  was meant to remove. The inlined call uses the same command, timeout
  (`GIT_TIMEOUT_SECONDS`, numerically identical to the original
  `LS_REMOTE_TIMEOUT_SECONDS`, both 5.0), and exception handling as before.
- `LanceStore._active_partition_ref` → `active_partition_ref`,
  `_chunk_project_id` → `chunk_project_id`; the old names are one-line
  deprecated aliases. All call sites in `storage.py` itself,
  `application.py` (4), `reference_service.py` (2), and `search.py` (1) now
  use the public names; `tests/test_storage.py` and `tests/test_staging.py`
  still call the underscored aliases unchanged, as the plan allows.

**Step 4 (D5).** Added `IndexSettings.offline: bool = False`, parsed in
`from_environment` with the exact original lenient rule (`"1"`/`"true"`/`"yes"`
case-insensitively → `True`, everything else — including `"0"`/`"off"`/unset —
→ `False`, never raises), unlike the stricter `_boolean()` helper used for
every other flag. `Application.__init__` now reads `self.settings.offline`
instead of `os.environ.get("CODE_INDEXING_OFFLINE", ...)`. Added to
`tests/test_settings.py`: parametrized truthy/falsy/unset parsing tests, and
`test_application_never_reads_offline_from_the_environment`, which
monkeypatches `os.environ` with a dict subclass that raises if
`"CODE_INDEXING_OFFLINE"` is ever looked up, then builds a real `Application`
(explicit `IndexSettings(offline=True)`, default `FastEmbedder`) and asserts
`app.embedder.offline is True`.

*Other `os.environ` reads in `src/code_indexing_mcp/*.py` outside `settings.py`,
`application.py:RuntimePaths.from_environment`, and `update_check.py`
(installer/* is out of scope — it runs off the serve path):*
- `daemon.py` — `USERNAME`/`XDG_RUNTIME_DIR` (socket path, not an indexing
  setting) and `_settings_digest(os.environ)` (Track 4's whole-environment
  fingerprint for stale-daemon detection, not a single named setting).
- `git_state.py` — copies `os.environ` to set `GIT_OPTIONAL_LOCKS=0` for one
  subprocess call; a child-process environment override, not configuration.
- `worker_launcher.py` — `{**os.environ, **self._extra_environment}` builds a
  spawned worker's environment; inheritance, not a settings read.
- `accelerator_env.py` — `environment` parameter defaults to `os.environ` for
  a function whose accelerator-record fields are unrelated to `IndexSettings`.

**Left undone:** nothing from the plan's steps. D6 (nothing else moves) was
respected throughout.
