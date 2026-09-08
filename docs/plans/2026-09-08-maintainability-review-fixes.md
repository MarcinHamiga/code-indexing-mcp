# Maintainability Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all 17 major and 33 minor fixes from `docs/reviews/2026-09-08-maintainability-review.md` while preserving the existing indexing, installer, resolver, and wire-format contracts.

**Architecture:** Keep correctness boundaries explicit: checkout identity and partition pinning remain at application/service boundaries, storage writes retain their transaction and recovery semantics, and installer/UI layers consume typed results rather than presentation strings. Extract only shared policy and lifecycle code; retain intentionally distinct full-index, backfill, bootstrap, and grammar-specific paths.

**Tech Stack:** Python 3.12, pytest/pytest-xdist, Ruff, mypy strict mode, Pydantic models, LanceDB/Arrow, Tree-sitter, Textual.

## Task 1: Prepare the isolated branch and baseline

**Files:**
- Create: `docs/plans/2026-09-08-maintainability-review-fixes.md`

**Steps:**

1. Create worktree `.worktrees/maintainability-review-2026-09-08` on branch `fix/maintainability-review-2026-09-08`.
2. Install the declared `cpu` and `tui` extras in the worktree.
3. Run `uv run pytest -n auto` and record the clean baseline.
4. Commit the plan document before production changes.

## Task 2: Contain malformed input and cleanup failures

**Files:**
- Modify: `src/code_indexing_mcp/staging.py`
- Modify: `src/code_indexing_mcp/probe_cache.py`
- Modify: `src/code_indexing_mcp/history.py`
- Modify: `src/code_indexing_mcp/installer/env_blocks.py`
- Modify: `src/code_indexing_mcp/installer/accelerator.py`
- Modify: `src/code_indexing_mcp/accelerator_env.py`
- Modify: `tests/test_staging.py`, `tests/test_probe_cache.py`, `tests/test_history.py`, `tests/test_installer.py`, `tests/test_accelerator_env.py`

**Fixes:** M04, M05, M06, M07, M12, M14.

**Steps:**

1. Add failing tests for non-object journals, invalid recovery counters, lock timeout/acquisition errors, malformed JSON objects, invalid numeric records, and writer cleanup when an earlier resource fails.
2. Add one typed journal parser shared by recovery and pending-recovery inspection; preserve unknown journals and delete only known terminal phases.
3. Replace the optional probe-cache context manager with explicit lock acquisition and guaranteed release; skip only the optional cache write on lock errors.
4. Close SQLite write connections through an owning context and make `HistoryStore.close()` deterministic.
5. Make staged writer cleanup attempt every writer/sink, reset ownership as each resource is handled, preserve the first finalization error, and clean up a sink if writer construction fails.
6. Validate decoded JSON objects before `.get()` access and validate numeric scalar types, ranges, and finite values at parsing boundaries.
7. Run the focused tests, format, lint, and mypy.

## Task 3: Preserve operational state through indexing and installer repair

**Files:**
- Modify: `src/code_indexing_mcp/server.py`
- Modify: `src/code_indexing_mcp/installer/cli.py`, `src/code_indexing_mcp/installer/wizard.py`
- Modify: `src/code_indexing_mcp/installer/tui/panels.py`, `src/code_indexing_mcp/installer/orchestrator.py`
- Modify: `tests/test_server.py`, `tests/test_installer_cli.py`, `tests/test_installer_tui.py`

**Fixes:** M02, M03, M08.

**Steps:**

1. Add a watcher/startup regression using a non-primary checkout and verify automatic indexing passes `roots=[root]` through the retry path.
2. Add a repair regression with conflicting per-harness settings and verify repair supplies no flattened updates.
3. Add a TUI completion regression that verifies configuration finalization invokes the same daemon invalidation hook as the CLI.
4. Carry project/root identity through `StartupCoordinator` and centralize post-install finalization behind the installer orchestration boundary.
5. Run the focused server and installer tests, format, lint, and mypy.

## Task 4: Restore token, worker, and backend contracts

**Files:**
- Modify: `src/code_indexing_mcp/token_batching.py`, `src/code_indexing_mcp/embedding.py`
- Modify: `src/code_indexing_mcp/embedding_worker.py`, `src/code_indexing_mcp/passage_backend.py`
- Modify: `src/code_indexing_mcp/accelerator_probe.py`, `src/code_indexing_mcp/backends.py`
- Modify: `src/code_indexing_mcp/backend_coordinator.py`
- Modify: `tests/test_token_batching.py`, `tests/test_embedding_worker.py`, `tests/test_passage_backend.py`, `tests/test_accelerator_acceptance.py`, `tests/test_backend_coordinator.py`

**Fixes:** M01, M11, M13, M15, N10, N11, N12, N13.

**Steps:**

1. Add failing tests for prefixed input token products, unknown worker statuses/cardinality, hung requests, direct-model empty provider reports, consistent setting normalization, one measurements snapshot, and retrying a missing CPU measurement after a cached accelerator result.
2. Make token-window counts represent the composed model input at the packing boundary, including prefix and overhead; reject an unfit prefix rather than widening the configured budget.
3. Introduce shared descriptor-based provider validation and map failures to installer/runtime errors at their boundaries.
4. Move setting metadata and normalization into a dependency-light runtime-owned layer while preserving the documented legacy offline behavior.
5. Add separate model-load and inference deadlines to worker requests and route expiry through existing teardown/fallback behavior.
6. Replace permissive tuple/`Any` worker replies with checked status and payload shapes; keep only an explicitly supported legacy reply if compatibility tests prove it is required.
7. Give worker telemetry ownership of snapshot/reset state and make coordinator calculations pure over one measurements snapshot.
8. Run focused tests, format, lint, and mypy.

## Task 5: Unify discovery and reference context data flow

**Files:**
- Modify: `src/code_indexing_mcp/scanner.py`, `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/reference_service.py`
- Modify: `tests/test_scanner.py`, `tests/test_application.py`, `tests/test_reference_pushdown.py`, `tests/test_references.py`

**Fixes:** M16, M17, N03, N16, N17, N22.

**Steps:**

1. Add discovery/scan parity tests for nested repositories, force-added ignored files, and reference context construction count.
2. Factor a stat-only eligibility iterator shared by discovery and the real scanner, preserving Git enumeration, ignore semantics, bounded batches, and subprocess cleanup.
3. Pass immutable `_ReferenceContext` data through classification and override analysis; construct only declaration-dependent receiver maps per query.
4. Extract the repeated application reference-query preparation/partition boundary while preserving repository-stability retries.
5. Add strict reference/impact cursor models behind a shared exact-key Base64/JSON codec, and replace untyped edge/classification dictionaries with typed dataclasses using existing model aliases.
6. Extract patch conflict construction and verified-edit results without weakening source, digest, containment, BOM, or overlap checks.
7. Run resolver, pushdown, scanner, application, and patch tests, then format, lint, and mypy.

## Task 6: Consolidate search, staging, storage, and maintenance paths

**Files:**
- Modify: `src/code_indexing_mcp/search.py`, `src/code_indexing_mcp/storage.py`
- Modify: `src/code_indexing_mcp/indexing.py`, `src/code_indexing_mcp/maintenance.py`
- Modify: `tests/test_search.py`, `tests/test_storage.py`, `tests/test_application.py`, `tests/test_indexing.py`

**Fixes:** N01, N02, N04, N05, N06, N07, N08, N09.

**Steps:**

1. Add parameterized search filter/partition tests and a dry-run test asserting pending touches and table versions are unchanged.
2. Extract typed search preparation/hit collection and resolve example language/extraction once; log recoverable detection failures without hiding unexpected errors.
3. Split maintenance inspection, project mutation, and registry maintenance into typed helpers while preserving lock order and per-project exception isolation.
4. Remove the consecutive duplicate chunk normalization pass.
5. Use ordered-set semantics for staged IDs, extract the shared chunk/reference Arrow merge helper, and extract staging-job construction with begin-failure cleanup.
6. Expand registry table inventory to projects, project slots, and active slots and report maintenance statistics against the same inventory.
7. Run focused search/storage/indexing/maintenance tests, format, lint, and mypy.

## Task 7: Type extractor contracts and simplify grammar plumbing

**Files:**
- Modify: `src/code_indexing_mcp/extractor.py`, `src/code_indexing_mcp/language_rules.py`
- Modify: `tests/test_language_rules.py`, `tests/test_extractor.py`, `tests/test_reference_pushdown.py`

**Fixes:** N18, N19, N20, N21, N31.

**Steps:**

1. Add an independent registry integration test covering language rules, bound handlers, and packaged query compilation.
2. Remove unreachable positional-only state while preserving separator-based parameter rewriting.
3. Separate identifier exclusion decisions into named predicates/rule data without changing branch precedence.
4. Replace handler-name strings and variadic `Any` callbacks with typed protocols and bound callable registrations.
5. Consolidate query compilation behind one locked helper while retaining intention-revealing wrappers.
6. Run extractor and resolver-corpus tests, format, lint, and mypy.

## Task 8: Centralize model, benchmark, daemon, and CLI lifecycle helpers

**Files:**
- Modify: `src/code_indexing_mcp/embedding.py`, `src/code_indexing_mcp/embedding_worker.py`
- Modify: `src/code_indexing_mcp/benchmark.py`, `src/code_indexing_mcp/daemon.py`, `src/code_indexing_mcp/cli.py`
- Modify: `src/code_indexing_mcp/installer/daemon_control.py`, `src/code_indexing_mcp/installer/update.py`, `src/code_indexing_mcp/installer/cli.py`, `src/code_indexing_mcp/tui/service.py`, `src/code_indexing_mcp/server.py`
- Modify: `src/code_indexing_mcp/update_check.py`
- Modify: `tests/test_benchmark.py`, `tests/test_daemon.py`, `tests/test_cli.py`, `tests/test_installer_cli.py`, `tests/test_update_check.py`

**Fixes:** N14, N15, N23, N27, N28, N29, N30.

**Steps:**

1. Add tests for benchmark workspace cleanup/provenance, daemon exception logging, update-check parsing, backend auto/on/off selection, stop timeout handling, and typed configure requests.
2. Extract a public embedding model adapter used by in-process and worker paths; keep optional backend imports lazy and installer-independent.
3. Extract shared benchmark workspace/application setup and make each benchmark report the backend it actually exercised.
4. Centralize backend/application creation with one validated settings snapshot and preserve direct-mode fallback policy.
5. Move daemon stop-and-wait behavior to a runtime helper with typed timeout completion and use it from restart and installer flows.
6. Replace installer argv reconstruction with a typed configure request, and reuse the runtime update-check implementation while retaining installer-specific error presentation.
7. Add `logger.exception` at the daemon translation boundary without exposing request contents.
8. Run focused benchmark/daemon/CLI/installer tests, format, lint, and mypy.

## Task 9: Type installer outcomes and harness metadata

**Files:**
- Modify: `src/code_indexing_mcp/installer/harnesses.py`, `src/code_indexing_mcp/installer/env_blocks.py`
- Modify: `src/code_indexing_mcp/installer/orchestrator.py`, `src/code_indexing_mcp/installer/update.py`, `src/code_indexing_mcp/installer/uninstall.py`
- Modify: `tests/test_installer.py`, `tests/test_installer_cli.py`, `tests/test_installer_tui.py`

**Fixes:** N24, N26.

**Steps:**

1. Add tests for structured skill outcomes and registry-driven read/write/remove behavior across schema families.
2. Define typed step/status/outcome models and carry structured skill results through install, update, uninstall, and TUI rendering.
3. Define harness schema metadata once and drive common environment/object access from it while leaving exceptional TOML/version/platform branches explicit.
4. Run installer tests, format, lint, and mypy.

## Task 10: Simplify TUI output and share test support

**Files:**
- Modify: `src/code_indexing_mcp/tui/app.py`
- Create or modify: `tests/support.py` or the existing test support module
- Modify: `tests/test_application.py`, `tests/test_server.py`, `tests/test_daemon.py`, `tests/test_query_path_overhead.py`, `tests/test_cli.py`

**Fixes:** N25, N32.

**Steps:**

1. Add/adjust render tests for outline, references, and impact visible output.
2. Remove discarded styled text-tree construction and render instructions/entries once.
3. Move the identical deterministic embedder into shared test support and import it from the five suites, leaving specialized fakes local.
4. Run TUI and affected application tests, format, lint, and mypy.

## Task 11: Finish comments and branch-wide verification

**Files:**
- Modify: `src/code_indexing_mcp/reference_service.py`, `src/code_indexing_mcp/application.py`, `src/code_indexing_mcp/extractor.py`

**Fixes:** N33.

**Steps:**

1. Replace historical control-flow commentary with present-tense invariants and direct links to design documents; preserve safety rationale.
2. Rename `_rename_analysis` to reflect rename and signature-change analysis, using indexed references to update call sites.
3. Run `uv run ruff format .`, `uv run ruff format --check .`, `uv run ruff check .`, `uv run mypy src`, and `uv run pytest -n auto`.
4. Review `git diff --check`, `git status`, and the full diff against the branch base. Confirm every M01–M17 and N01–N33 item has a code change or an explicit test-backed explanation where the current tip already contains the fix.

