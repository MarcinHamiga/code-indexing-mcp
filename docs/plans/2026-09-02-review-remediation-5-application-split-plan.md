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
