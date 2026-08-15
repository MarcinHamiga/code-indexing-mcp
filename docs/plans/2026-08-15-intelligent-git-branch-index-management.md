# Intelligent Git Branch Index Management Implementation Plan

**Goal:** Make a registered project aware of its Git checkout and retain independent,
branch-specific index slots so switching back to a previously indexed branch does not repeatedly
parse and embed the repository.

**Architecture:** Keep the public project ID stable and add an internal branch-slot identity.
Each slot owns a separate physical Lance partition. Application entrypoints resolve an immutable
active index target immediately before accessing storage, while a registry pointer records the
currently selected slot. Clean slots at the same HEAD can be restored without scanning source
metadata. Dirty, advanced, reset, or degraded checkouts reuse the branch slot but must pass normal
freshness validation before being considered current.

**Tech stack:** Python 3.12, Git, LanceDB, PyArrow, Pydantic, SQLite, AnyIO/asyncio, FastMCP,
pytest.

## Product Decisions

- A branch without a cached slot becomes `pending`; lazy and eager modes build it on demand.
- A pending slot never falls back to or serves results from the previously active branch.
- Manual mode exposes the pending or stale same-branch slot without automatically indexing it.
- Inactive slots use bounded least-recently-used retention.
- Add `CODE_INDEXING_BRANCH_CACHE_LIMIT`, default `4`, counting the active slot.
- V1 exposes branch and slot information through existing status surfaces only.
- Linked worktrees remain isolated because each checkout can have different dirty files,
  untracked files, sparse-checkout rules, and worktree-local configuration.
- Existing Git projects receive one conservative branch-aware rebuild. Existing non-Git projects
  can adopt their current partition as the workspace slot.
- V1 does not clone or seed a new branch slot from another branch's partition.

## Existing Constraints

- `.ci-mcp/project.toml` holds one checkout-local project ID and must remain backward compatible
  (`src/code_indexing_mcp/projects.py`).
- The project registry currently combines logical registration and active index-generation state
  in one row (`src/code_indexing_mcp/storage.py:LanceStore.upsert_project`).
- Physical partition paths, table caches, generation counters, staging journals, and chunk routing
  currently assume `project_id == partition_id`.
- Freshness currently compares paths, size, mtime, and language. The indexer can skip reading a
  file when size and mtime match, which is insufficient after a branch switch or reset.
- Lazy server preflight and the final application query are separate operations. Correct active
  slot selection therefore belongs at the application/storage boundary, not only in the MCP
  coordinator.
- `get_chunk` bypasses project-scoped server preflight and must independently resolve the active
  slot.
- Staging recovery records only a logical project ID and table versions. Recovery must never
  restore those versions into whichever slot happens to be active later.
- Reference cursors currently bind a logical project and table version. Equal version numbers can
  exist in different branch partitions, so the cursor must also bind the slot.
- Lance table versions cannot be branch slots because maintenance deliberately removes historical
  versions.

## Core Invariants

1. Public APIs, markers, history lookup, and project resolution continue to use the logical
   project ID.
2. Every table read, write, rollback, rebuild, and maintenance operation carries an immutable
   physical partition ID.
3. One logical project can have only one active slot at a time, but inactive slots remain durable.
4. Slot activation never makes an old branch partition visible under a new branch selector.
5. No operation publishes rows when the Git selector or HEAD changed while those rows were being
   produced.
6. A chunk ID from an inactive slot never returns different content from the active slot.
7. Recovery restores the exact partition named by its journal, never the current active pointer.
8. LRU eviction never removes the active slot, an indexing slot, or a slot with pending recovery.
9. A transient Git failure never overwrites a known branch slot with workspace-fallback data.
10. Non-Git projects retain the current single-partition behavior through one stable workspace
    slot.

## Data Model

### Git state

Add `src/code_indexing_mcp/git_state.py` with typed, read-only models for:

- Probe outcome: `git`, `not_git`, `unavailable`, `timeout`, or `invalid`.
- Resolved repository identity from `--git-common-dir`.
- Resolved checkout identity from `--git-dir`.
- Resolved worktree top-level and project root relative to that top-level.
- HEAD selector:
  - attached or unborn: full symbolic ref, such as `refs/heads/main`;
  - detached: full commit OID;
  - workspace fallback: checkout-local project-root identity.
- Current HEAD OID, which is optional for an unborn branch.
- Worktree classification: clean, tracked dirty, untracked, mixed, or unknown.
- Dirty and untracked project-relative paths for internal validation.
- A bounded status fingerprint for cache invalidation and diagnostics.

Do not put the attached branch OID, dirty state, dirty paths, scan configuration, model, or schema
version in the slot key. They are mutable properties of one slot.

Build the canonical slot key from:

```text
(
    "git-slot-v1",
    logical_project_id,
    repository_identity,
    checkout_identity,
    project_prefix,
    selector_kind,
    selector_value,
)
```

Serialize the tuple deterministically and hash it with SHA-256. Use an opaque, path-safe slot ID
and partition ID; never place a branch name directly in a filesystem path.

### Slot registry

Add two Lance registry tables rather than changing the schema of the existing projects table:

`project_slots`:

```text
slot_id: string, primary key
project_id: string
partition_id: string
selector_kind: string
selector_value: string
repository_identity: string | null
checkout_identity: string | null
project_prefix: string
indexed_head: string | null
indexed_clean: bool
scan_config_hash: string
model_id: string
vector_dimension: int
schema_version: int
state: string
created_at: int
last_used_at: int
```

`active_slots`:

```text
project_id: string, primary key
slot_id: string
activation_epoch: int
updated_at: int
```

Use a separate active-pointer row instead of an `active` flag in `project_slots`; switching an
`active` flag would otherwise require two non-atomic row updates.

Keep the existing projects row as the logical registration. Its generation columns may remain an
active-slot compatibility summary during migration, but all new correctness decisions must use
the slot row.

### Immutable operation target

Add an internal frozen model or dataclass, tentatively `ActiveIndexTarget`, containing:

```text
project: ProjectInfo
slot: ProjectSlot
partition_id: string
activation_epoch: int
git_state: GitState
```

Pass this target through the indexer, search service, reference service, staging, storage, and
status paths. Do not let lower layers re-resolve the active pointer midway through an operation.

## Physical Layout

Keep physical partitions flat:

```text
lancedb/
  registry/
    projects.lance
    project_slots.lance
    active_slots.lance
  projects/
    <legacy-project-id>/
    slot-<opaque-id>/
  partition-generations/
    <physical-partition-id>
```

The flat layout lets the old `projects/<project-id>` directory become a legacy partition without
moving Lance data. Nested `projects/<project-id>/slots/...` would require a crash-safe directory
relocation because `projects/<project-id>` is already a Lance database.

## Delivery Sequence

### Task 1: Git state probe and slot key

**Files:**

- Create: `src/code_indexing_mcp/git_state.py`
- Modify: `src/code_indexing_mcp/models.py`
- Test: `tests/test_git_state.py`

1. Add failing tests for normal branches, detached HEAD, unborn branches, subdirectory project
   roots, dirty tracked files, untracked files, and non-Git directories.
2. Add real linked-worktree coverage proving that the common repository identity matches while
   the checkout identity differs.
3. Add injected-runner tests for missing Git, timeout, malformed output, deleted worktree metadata,
   and relative Git directory paths.
4. Implement a bounded-timeout, shell-free Git runner with `GIT_OPTIONAL_LOCKS=0`.
5. Resolve relative `--git-common-dir` and `--git-dir` output against the registered root, matching
   Git's documented query-directory semantics.
6. Read the symbolic ref and OID without abbreviating either value.
7. Run `git status --porcelain=v1 -z --untracked-files=all` only when cleanliness or dirty paths
   are required; keep the selector/OID probe cheap enough for query entrypoints.
8. Implement deterministic slot-key hashing and prove that mutable HEAD and dirty state do not
   change an attached branch's slot ID.
9. Prove branch rename, detached OID, repository replacement, checkout replacement, and project
   prefix changes do change the slot ID.

### Task 2: Configuration and public status models

**Files:**

- Modify: `src/code_indexing_mcp/settings.py`
- Modify: `src/code_indexing_mcp/installer/settings_spec.py`
- Modify: `src/code_indexing_mcp/models.py`
- Test: `tests/test_settings.py`
- Test: `tests/test_installer_settings_spec.py`

1. Add `branch_cache_limit` to `IndexSettings`.
2. Parse `CODE_INDEXING_BRANCH_CACHE_LIMIT` as an integer from `1` to `32`, default `4`.
3. Add the setting to the installer catalog under the Indexing or Maintenance group.
4. Add optional, backward-compatible Git and slot fields to `ProjectStatus`.
5. Add versioned per-slot storage status models with active flag, selector, indexed HEAD, state,
   last-use timestamp, and physical bytes.
6. Bump only response schema versions whose serialized contract changes.
7. Keep `ProjectInfo` and the marker schema unchanged.

### Task 3: Slot registry and legacy adoption

**Files:**

- Modify: `src/code_indexing_mcp/storage.py`
- Test: `tests/test_storage.py`

1. Add failing tests for idempotent creation and round-trip of `project_slots` and `active_slots`.
2. Add store operations to create, list, resolve, touch, activate, and remove slots.
3. Publish the active pointer only after the target slot row exists.
4. Increment `activation_epoch` on every pointer change.
5. Key open-table caches and partition-generation counters by physical partition ID.
6. Change `_tables`, `_existing_tables`, partition access, table versions, compatibility checks,
   rebuild deletion, and statistics to accept a physical partition identity.
7. Preserve a logical project ID in file and reference rows; chunk rows continue relying on
   partition ownership.
8. Add lazy legacy adoption under the logical project writer lock:
   - non-Git projects map `partition_id=project_id` into the workspace slot;
   - Git projects record the old partition as an unscoped legacy slot but do not serve it as the
     current branch;
   - registered projects without a partition create only a pending slot row.
9. Ensure repeated startup and interrupted adoption are idempotent.
10. Keep reads from materializing a partition for a pending slot.

### Task 4: Slot-aware storage and chunk routing

**Files:**

- Modify: `src/code_indexing_mcp/storage.py`
- Modify: `src/code_indexing_mcp/search.py`
- Modify: `src/code_indexing_mcp/reference_service.py`
- Modify: `src/code_indexing_mcp/indexing.py`
- Test: `tests/test_storage.py`
- Test: `tests/test_search.py`
- Test: `tests/test_references.py`

1. Change storage read methods to take `ActiveIndexTarget` or an explicit immutable partition
   reference.
2. Change multi-project search to carry a logical-project-to-partition mapping and continue
   reporting logical project IDs in hits.
3. Keep chunk IDs prefixed by the logical project ID for direct project resolution.
4. Include the slot ID in the chunk identity digest so identical content in two branch slots does
   not create an accidentally cross-slot selector.
5. Change `_chunk_project_id` to validate the logical project registry instead of probing
   `projects/<project-id>`.
6. Make `get_chunk` query only the resolved active slot. An inactive or evicted slot's chunk ID
   returns `CHUNK_NOT_FOUND`.
7. Update declaration selection by chunk ID to reject a chunk not present in the active slot.
8. Extend reference cursors with slot ID and activation epoch. Return `STALE_CURSOR` when the
   cursor slot is no longer active.
9. Add tests where two slots have equal Lance table versions and prove cursors and chunks cannot
   cross between them.

### Task 5: Slot-aware staging and crash recovery

**Files:**

- Modify: `src/code_indexing_mcp/staging.py`
- Modify: `src/code_indexing_mcp/storage.py`
- Modify: `src/code_indexing_mcp/indexing.py`
- Test: `tests/test_staging.py`

1. Bump the staging journal format.
2. Record logical project ID, slot ID, immutable partition ID, and all three table versions.
3. Store new jobs under `staging/<project-id>/<slot-id>/<job-id>`.
4. Teach recovery to discover both the legacy two-level and new three-level layouts.
5. Interpret legacy journals as `partition_id == project_id`, the only layout that could have
   written them.
6. Restore versions directly into the recorded physical partition without consulting the active
   pointer.
7. Never create a missing partition while recovering a removed project or evicted slot.
8. Make pending-recovery checks recursive by logical project and expose the protected slot IDs to
   LRU eviction.
9. Add crash tests for pointer changes before, during, and after a commit.
10. Add a test proving startup recovery of an inactive slot cannot alter the active slot.

### Task 6: Active target resolution at the application boundary

**Files:**

- Modify: `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/search.py`
- Modify: `src/code_indexing_mcp/reference_service.py`
- Test: `tests/test_application.py`
- Test: `tests/test_references.py`
- Test: `tests/test_refactors.py`

1. Add one application helper that probes Git, resolves or creates the slot, updates the active
   pointer under the logical project lock, and returns `ActiveIndexTarget`.
2. Resolve a target immediately before project status, explicit indexing, semantic search, symbol
   search, file outline, chunk fetch, reference analysis, refactor analysis, storage status, and
   maintenance.
3. Make a first-time branch slot pending and empty before returning control to server scheduling.
4. Never serve the previously active slot while a new slot is pending.
5. On Git probe failure, select a separate degraded workspace slot rather than modifying a known
   branch slot.
6. Resolve all targets for a multi-project query in stable logical-project order.
7. Re-probe target identities after a query. Retry once if any selector or HEAD changed; return a
   structured repository-changed error after a second transition.
8. Ensure `get_chunk` uses the same active-target boundary even though it has no explicit project
   argument.
9. Preserve manual-mode behavior: direct searches can read a stale same-branch slot, but never a
   different branch's slot.

### Task 7: Indexing guard and branch-aware freshness

**Files:**

- Modify: `src/code_indexing_mcp/indexing.py`
- Modify: `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/progress.py`
- Test: `tests/test_indexing.py`
- Test: `tests/test_application.py`
- Test: `tests/test_progress.py`

1. Make `Indexer.index` and reference backfill accept `ActiveIndexTarget`.
2. Keep writer lock ownership at the logical project level so two slots for one checkout cannot
   index concurrently.
3. Write progress with logical project ID, slot ID, selector, expected HEAD, and activation epoch.
4. Capture Git selector and HEAD before scanning and verify both before the staged commit begins.
5. If either changes, discard staged data, leave the slot stale or pending, and return a structured
   retryable error.
6. Update only the target slot's state and generation metadata after a successful commit.
7. Store `indexed_head`, `indexed_clean`, scan hash, model, dimension, and schema on the slot.
8. Extend the file-processing path with `verify_paths` and `verify_all` controls that bypass the
   size/mtime early return but retain content-hash reuse.
9. On same-slot HEAD changes, use `git diff --name-only -z <old> <new> -- <project-prefix>` to
   identify tracked paths requiring hash validation. Fall back to `verify_all` if the diff cannot
   be computed.
10. Force hash validation for dirty and untracked paths reported by Git, preventing equal
    size/mtime from hiding a worktree edit.
11. Treat a clean slot as immediately ready without a source scan only when selector, HEAD, scan
    hash, model, dimension, schema, and clean state all match.
12. Key the in-memory freshness cache by logical project ID, slot ID, activation epoch, HEAD, and
    scan-config fingerprint.
13. Add a switch-away/switch-back test proving no source scanner, parser, or embedder work occurs
    for an exact clean cache hit.

### Task 8: Lazy, eager, daemon, and watcher integration

**Files:**

- Modify: `src/code_indexing_mcp/server.py`
- Modify: `src/code_indexing_mcp/daemon.py`
- Test: `tests/test_server.py`
- Test: `tests/test_daemon.py`

1. Carry slot ID and activation epoch through daemon request and response models where an operation
   can outlive target resolution.
2. Make completed coordinator jobs reusable only when their slot and activation epoch still match
   the root's active target.
3. In lazy mode, schedule and wait for a pending or stale active slot before querying it.
4. In eager mode, treat a branch transition as a dirty generation even if filesystem watcher
   events were coalesced or missed.
5. Retry a repository-changed indexing result through the existing bounded coordinator loop.
6. Filter progress displays to the active slot so a finishing prior-branch job cannot satisfy or
   overwrite the new branch's status.
7. Add branch-switch tests for direct application mode and broker mode.
8. Add a race test that switches branches between server preflight and daemon query execution and
   proves the application boundary prevents old-branch results.
9. Add a race test that switches branches during an index scan and proves no mixed generation
   commits.

### Task 9: History and operational observability

**Files:**

- Modify: `src/code_indexing_mcp/history.py`
- Modify: `src/code_indexing_mcp/models.py`
- Modify: `src/code_indexing_mcp/indexing.py`
- Modify: `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/server.py`
- Modify: `src/code_indexing_mcp/daemon.py`
- Test: `tests/test_history.py`
- Test: `tests/test_application.py`
- Test: `tests/test_server.py`

1. Preserve the existing history `git_revision` as server-build provenance.
2. Add nullable history columns for source slot ID, selector kind/value, source HEAD, clean state,
   and physical partition ID.
3. Add guarded SQLite migrations for installations with the current history schema.
4. Continue listing and pruning history by logical project ID.
5. Include partition identity alongside storage table versions so version numbers remain
   interpretable after slot switching.
6. Extend `project_status` with current selector, HEAD, probe state, clean state, active slot ID,
   cached-slot count, and whether a branch build is pending.
7. Extend `index_storage_status` with all slots, active flags, per-slot table statistics, and the
   aggregate physical bytes for the logical project.
8. Ensure status collection detects an active-pointer change while collecting and returns
   `consistent=false` rather than presenting a mixed snapshot.
9. Update MCP tool descriptions and daemon serialization tests for the additive fields.

### Task 10: LRU retention, maintenance, rebuild, and removal

**Files:**

- Modify: `src/code_indexing_mcp/storage.py`
- Modify: `src/code_indexing_mcp/application.py`
- Modify: `src/code_indexing_mcp/indexing.py`
- Test: `tests/test_storage.py`
- Test: `tests/test_application.py`
- Test: `tests/test_indexing.py`

1. Add an LRU selector ordered by `last_used_at`, then stable slot ID for deterministic ties.
2. Count the active slot toward `branch_cache_limit`.
3. Never choose active, indexing, deleting, or recovery-protected slots.
4. Allow the limit to be exceeded temporarily while a new active slot is pending or failed; prune
   only after a successful build or during maintenance so a failed first build does not destroy a
   usable cache unnecessarily.
5. Remove a slot through a durable deleting state: evict cached handles, advance its physical
   generation, delete the partition, remove the slot row, and resume safely after a crash.
6. Make compatibility checks and schema rebuilds operate only on the active physical slot.
7. Make scheduled and manual maintenance iterate every retained slot, not just the active slot.
8. Aggregate maintenance results under the logical project while retaining per-slot details.
9. Make `remove_project` delete the active pointer, every slot row, every owned partition,
   generation metadata, pending progress, and staging data without deleting the local marker.
10. Detect and report orphan physical partitions; reclaim them only after proving no slot or journal
    references them.
11. Add tests for LRU order, protected slots, failed builds, detached-slot churn, multi-slot
    maintenance, schema rebuild isolation, and complete project removal.

### Task 11: Scanner fallback correctness

**Files:**

- Modify: `src/code_indexing_mcp/scanner.py`
- Test: `tests/test_scanner.py`

1. Add a regression test where Git worktree detection succeeds but both `git ls-files` and
   `git check-ignore` fail.
2. Prove the fallback walk still applies in-process `.gitignore` specifications and does not index
   ignored source files.
3. Keep nested repositories and submodules opaque under both Git and walk enumeration.
4. Confirm existing tracked-but-ignored and linked-worktree scanner behavior remains unchanged.

### Task 12: Documentation and final verification

**Files:**

- Modify: `README.md`
- Modify: MCP tool descriptions in `src/code_indexing_mcp/server.py`
- Modify: installer setting documentation in `src/code_indexing_mcp/installer/settings_spec.py`

1. Document branch-aware behavior for lazy, eager, and manual modes.
2. Document `CODE_INDEXING_BRANCH_CACHE_LIMIT`, including that the active slot counts toward the
   limit and protection can temporarily exceed it.
3. Explain detached HEAD caching, degraded Git fallback, one-time migration rebuild, and
   checkout-local worktree isolation.
4. Update status examples with active branch and cached-slot storage.
5. State explicitly that the tool indexes only the checked-out working tree and does not index
   arbitrary refs from Git objects.
6. Run focused test suites after each task.
7. Run final formatting and static checks:

```sh
uv run ruff format .
uv run ruff check .
uv run mypy src
```

8. Run the full test gate:

```sh
uv run pytest -n auto
```

9. Review `git diff --check`, the migration paths, lock ordering, journal compatibility, and all
   user-visible schema changes before merge.

## Required Behavior Matrix

| Scenario | Expected behavior |
| --- | --- |
| Switch to cached clean branch at the same HEAD | Activate immediately; no source scan, parse, or embedding |
| Switch to an unseen branch | Activate an empty pending slot; lazy/eager builds on demand |
| Search unseen branch in manual mode | Return no old-branch rows; remain pending until explicit indexing |
| Commit on current branch | Reuse the slot and incrementally validate changed paths |
| Reset or reuse branch name at another OID | Reuse selector slot but force content validation |
| Dirty tracked files | Update current branch slot; dirty state does not create another slot |
| Untracked eligible files | Include through existing scanner semantics and validate in the current slot |
| Detached HEAD | Cache by full commit OID and subject the slot to normal LRU retention |
| Unborn branch then first commit | Keep the same symbolic-ref slot and advance its indexed HEAD |
| Branch rename | Create a new selector slot; old slot becomes LRU eligible |
| Branch deletion while inactive | Leave the slot cached until normal LRU cleanup |
| Git unavailable or timed out | Use a degraded workspace slot; do not mutate branch slots |
| Linked worktree | Use an isolated checkout-local slot set |
| Branch changes during scan | Discard staged rows and retry; never commit a mixed generation |
| Branch changes during query | Retry once against the new slot, then return a structured error |
| Old branch chunk ID | Return `CHUNK_NOT_FOUND` unless that slot is active and still contains it |
| Reference cursor after switch | Return `STALE_CURSOR` |
| Project removal | Remove every slot and physical partition while leaving the marker |

## Acceptance Gates

- Switching from branch A to B never returns a chunk, symbol, outline item, reference, or refactor
  finding that exists only on A.
- Switching from A to B and back to an unchanged, clean A performs zero parsing and embedding on
  the return switch.
- A first visit to B performs at most one full build, after which B is reusable until evicted or
  invalidated by model/schema changes.
- A same-branch commit or reset cannot be hidden by equal source size and mtime.
- A branch transition during indexing leaves all three active tables on one coherent generation.
- Crash recovery restores only the journal's physical partition.
- Pending and manual-mode slots never expose the prior branch as a fallback.
- Existing non-Git registrations continue to work without a forced rebuild.
- Existing Git registrations require at most one conservative migration rebuild.
- The configured LRU limit bounds durable branch partitions after successful maintenance, except
  for explicitly protected or failed-active slots reported by status.
- Storage status includes every retained slot in project and installation byte totals.
- Existing project selectors, marker files, and logical project IDs remain stable.
- All formatter, Ruff, mypy, and parallel pytest gates pass.

## Deferred Work

- Sharing clean partitions between linked worktrees.
- Sharing indexes between clones based on remote URL or Git common directory.
- Copy-on-write seeding or vector/chunk deduplication between branch slots.
- Detecting a branch rename and transferring its slot automatically.
- Eagerly enumerating all refs to delete branch slots immediately after ref deletion.
- Indexing branches that are not checked out.
- Dirty-content-addressed immutable snapshots.
- MCP tools for pinning, activating, listing, or manually pruning branch slots.

These optimizations require stronger equivalence proofs for project prefix, sparse-checkout state,
ignore configuration, scan settings, model/schema generation, committed tree, and untracked files.
They should not be mixed into the v1 correctness boundary.
