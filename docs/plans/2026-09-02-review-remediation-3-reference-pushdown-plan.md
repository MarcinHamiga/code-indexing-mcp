# Track 3 — Reference Query Pushdown — Implementation Plan

**Goal:** Stop materializing the whole reference table into Python for every
`find_references`, `analyze_refactor`, `emit_refactor_patch`, and each node of
`impact_radius`. Push a superset prefilter into the LanceDB `where` clause, fetch the
per-file and per-import context once per traversal, and select a declaration by its
indexed columns instead of decoding every chunk.

**Review findings closed:** perf-major 4 (whole reference table per call, per node in
`impact_radius`, traversal re-run per page), perf-major 5 (`_select` full chunk scan).

**Baseline:** see the index. The resolver corpus (`tests/test_resolver_corpus.py`),
`tests/test_references.py`, `tests/test_refactors.py`, and
`tests/test_reference_extraction.py` are the correctness contract for this track and
must pass **unchanged**.

## The one principle

**The Python filters stay the arbiter.** `_may_refer` (`reference_service.py:2230-2267`)
and every classification step after it are not modified. The storage condition only
excludes rows that `_may_refer` could never accept. If a pushdown cannot be proven a
superset for some branch, that branch fetches unfiltered as today. Correctness is
therefore unchanged by construction; only the row count that reaches Python drops.

## Decisions settled before implementation

- **D1 — What `_may_refer` can accept, and the superset condition.** Reading the
  function: a `reference` row passes when (a) `selected.symbol` equals
  `target_name`, `written_name`, or the last `.`-separated segment of either; or
  (c) the row is in a Java file with an on-demand import and its spelling
  (`receiver_text or written_name`) equals the symbol; or (d) some import in the row's
  file whose spelling (`alias or written_name or imported_name`) equals the row's
  spelling targets the selected declaration via `_import_targets` or
  `_reexport_targets_symbol`. Hence the superset:

  ```
  record_kind = 'reference' AND (
        target_name = S OR written_name = S OR receiver_text = S
     OR target_name LIKE '%.S' OR written_name LIKE '%.S'
     OR receiver_text IN (A) OR written_name IN (A) )
  ```

  where `S` is the selected symbol and `A` is the set of spellings of every
  `import` row in the project for which `_import_targets` or `_reexport_targets_symbol`
  is true for the selection (computed in Python from the import rows, D2). `S`
  contains only identifier characters (assert this; fall back to unfiltered fetch
  otherwise) so `LIKE` needs no escaping beyond `_quoted`; an underscore in `S` acts as
  a wildcard, which only widens the superset. Confirm DataFusion accepts `LIKE` and
  `IN` in a LanceDB `where` on the installed version (a quick probe test in
  `tests/test_storage.py` is enough). Empty `A` drops the two `IN` terms.
- **D2 — Context rows are fetched by kind, once.** Import and export rows
  (`record_kind = 'reference' AND kind IN ('import','export')`) and coverage rows
  (`record_kind = 'coverage'`) are O(imports) and O(files) respectively, far smaller
  than all references. They feed `_imports_by_file` (`:2175`), the re-export map
  (`:2226`), `known_paths`, the module index, and the staleness gate
  (`_hits_and_limitations`, `:1092`). Fetch them through two new narrow calls on
  `LanceStore.list_reference_records` (it already accepts `record_kinds`; add an
  optional `kinds` parameter and an optional raw `extra_condition` for D1) and bundle
  them in a `_ReferenceContext(version, partition_id, coverage_rows, import_rows,
  export_rows, known_paths, module_index)` built by a new
  `_load_reference_context(project_id, partition, version)`.
- **D3 — `_find_references_with_records` takes an optional context.** Its signature
  (`:478`) gains `context: _ReferenceContext | None = None`. When `None` it loads one.
  It then computes `A` from `context.import_rows`, fetches candidate rows with the D1
  condition, and assembles `records = coverage_rows + import_rows + export_rows +
  candidate_rows` (deduplicated by `reference_id`) so every downstream consumer that
  reads `query.records` sees the same row *kinds* it sees today, minus references
  that cannot match. Audit every consumer of `.records` and of the local `records`
  (`:799, :1039, :1092, :1437, :1508, :1933, :1939, :2839` and any others) — each
  either needs only coverage/import/export/candidate rows, or needs a *different*
  narrow set (e.g. `kind = 'inheritance'` rows whose `target_name` names the selected
  class for signature analysis at `:1939`) which it fetches explicitly with its own
  condition. Document the audit result in a comment at the top of
  `_find_references_with_records`.
- **D4 — `impact_radius` shares one context per traversal and caches pages.**
  `impact_radius` (`:673-1002`) calls `_find_references_with_records` per frontier
  node (`:760-775`); pass the same `_ReferenceContext` to every call (same partition
  and pinned version, so it is valid for the whole traversal). Pagination
  (`:917-935`) re-runs the traversal per page: add a small process-local LRU
  (`functools.lru_cache`-style dict, 8 entries, guarded by a lock) keyed by
  `(project_id, partition_id, version, selector key, max_depth, include_likely,
  sorted kinds, max_nodes)` holding the fully computed layers, so a cursor request
  reuses them. A cursor whose version no longer matches already raises
  `STALE_CURSOR`; the cache is keyed by version so it cannot serve stale layers.
- **D5 — `_select` by path and symbol uses indexed columns.** Replace the
  `list_chunks` scan (`:2103-2112`) with a projected query
  `path = ? AND qualified_symbol = ?` through `_projected_chunks`
  (`storage.py:1508-1525`; check it accepts a condition, else add a
  `find_declarations(project_id, path, qualified_symbol, partition_id)` on the store
  that projects `INDEXED_CHUNK_COLUMNS` minus `content`). The "did you mean" branch
  that lists nearby symbols selects only `qualified_symbol` where `path = ?`. Result
  ordering and the `AMBIGUOUS_SYMBOL` behaviour stay identical.
- **D6 — Indexes.** `target_name`, `kind`, `record_kind`, `file_id`, `schema_version`
  already have BTree indexes (`storage.py:1843-1855`). Add `written_name` and
  `receiver_text` to that list so the `= S` terms are index-assisted; `LIKE '%.S'`
  will scan but on a narrow projected column. Index creation runs where the existing
  ones are created, so an old partition gains them on its next maintenance or index
  run; do not force a rebuild.

## Steps

**Step 0 — Coordinates and audit.** Re-read `reference_service.py:478-700`,
`:673-1002`, `:1003-1210`, `:1207-1520`, `:1900-1960`, `:2074-2270`, `:2830-2850`,
and `storage.py:2520-2580`, `:1500-1530`, `:1840-1860`. Write the `.records`
consumer audit (D3) before touching code.

**Step 1 — Store: narrow fetches (D2, D6).** `list_reference_records` gains `kinds`
and `extra_condition`; add the two indexes; probe test for `LIKE`/`IN`.

**Step 2 — Context and superset fetch (D1–D3).** Implement `_ReferenceContext`,
`_load_reference_context`, the spelling set `A`, the candidate condition, and the
record assembly. Run the four contract test files after this step alone.

**Step 3 — Traversal context and page cache (D4).**

**Step 4 — `_select` pushdown (D5).**

**Step 5 — Regression guard.** Add `tests/test_reference_pushdown.py`: build a
fixture project with ~50 files and ~2000 reference rows where only 5 rows can refer to
the selected symbol; wrap `LanceStore._reference_rows` to count returned rows and
assert `find_references` pulls fewer than 200 (context rows plus candidates), that
`impact_radius` at depth 3 over 20 nodes calls `_load_reference_context` once, that
the second page of an `impact_radius` cursor does not call
`_find_references_with_records` at all, and that `_select` by path and symbol never
calls `list_chunks`. Also assert result equality: for every declaration in the
resolver corpus fixtures, `find_references` output with pushdown equals the output
with pushdown disabled (add a private module flag `_PUSHDOWN_ENABLED` used only by
this test to force the unfiltered path).

**Step 6 — Docs.** No tool description changes expected. Note the change in
`docs/plans/2026-08-07-reference-index-hardening.md`'s follow-ups list if one exists.

## Completion note (2026-09-02)

Implemented Steps 0-6 in full, including D1-D3 applied to every caller of
`_find_references_with_records`. Baseline green at the end: `ruff format
--check`, `ruff check`, `mypy src`, `pytest -n auto` all pass. The four
contract files (`test_resolver_corpus.py`, `test_references.py`,
`test_refactors.py`, `test_reference_extraction.py`) were touched only where
the coordinator explicitly narrowed the "do not edit" rule: two spy/mock
call-shape assertions (`test_references.py`'s `spy_list_reference_records`
and its `record_kinds_seen` assertion, `test_refactors.py`'s
`counting_list_reference_records`) were updated to match the new fetch
shape, with every correctness assertion (hits, kinds, limitations, patch
bytes, the narrowed declaration file set) left unchanged.

**Final design.** `_find_references_with_records` always loads a
`_ReferenceContext` (when the caller has not already supplied one) and
fetches candidates through `_candidate_records`'s SQL-pushed-down D1
superset condition -- no code path fetches the whole reference table
unfiltered any more. `find_references` and `analyze_refactor`'s single
lookup now costs 3 `list_reference_records` calls (coverage rows,
import/export rows, one candidate query carrying an `extra_condition` naming
the selected symbol) instead of one unfiltered full-table call; the row
*count* those calls return is what actually shrank. `emit_refactor_patch`
reuses the same `_rename_analysis` fetch, so it benefits identically.
`impact_radius` still loads one context per whole traversal and reuses it
across every frontier node (D4), and its cursor page cache is unchanged.
`_context_from_records` (the interim single-fetch-derived context used while
the two contract tests were unmodifiable) is deleted as dead code.

**D5 deviation (unchanged, still applies):** the "did you mean" suggestion in
`_select` searches every declaration in the project for a matching name
tail, not just the requested path (the plan's text described it as
path-scoped); the code it replaced already searched project-wide, so the
pushdown (`LanceStore.declaration_symbols_by_tail`) preserves that behaviour
rather than narrowing what callers see.

**Row-count reduction measured by the regression guard:** in a fixture with
~50 files and ~800 `reference`-kind rows where only 5 rows can refer to the
selected symbol, plain `find_references`, `analyze_refactor`, and
`impact_radius` at `max_depth=1` each pull under 200 rows total across every
`LanceStore._reference_rows` call -- versus the ~800 an unfiltered fetch
would have pulled. `_load_reference_context` is called exactly once for a
20+ node, `max_depth=3` `impact_radius` traversal (not once per frontier
node), and a second cursor page of that traversal calls
`_find_references_with_records` zero times (served from the page cache).
The on/off equality guard (`_PUSHDOWN_ENABLED`) still covers every
selectable declaration in the resolver corpus, now exercised through the
same code path `find_references` itself runs.

Nothing else from the plan was left undone.
