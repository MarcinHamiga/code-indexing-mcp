# Reference Index Hardening Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the silent-miss, storage-corruption, and honesty defects recorded in
`2026-08-07-reference-index-hardening.md` so that `find_references` and `analyze_refactor`
can be trusted for Python *and* the JS/TS family, and so this class of defect cannot ship
green again.

**Backlog reference:** `docs/plans/2026-08-07-reference-index-hardening.md` (defect IDs E1–E13,
R1–R4, S1–S6, T1–T7 used throughout). Every backlog claim was re-verified against
`hotfix/reference-index` HEAD (`6046623`) on 2026-08-07, including empirically re-running the
extractor for all thirteen E-items. All claims hold. The backlog's `indexing.py`,
`application.py`, `reference_service.py` line references are pre-hotfix coordinates; this plan
uses HEAD coordinates.

**Baseline:** `uv run pytest -q` on `hotfix/reference-index`. Full environment:
`uv sync --all-groups --extra cpu --extra tui --locked`.

## Corrections and additions from re-verification

Findings that refine or extend the backlog; the plan below accounts for all of them.

- **E1 refined:** TS *interface* heritage (`extends_type_clause`) already extracts correctly —
  its named children are `type_identifier`s. Only `class_heritage` is broken in TS/TSX, and
  `extends Base<T>` additionally needs the E2 descent (inner node is `generic_type`). There is
  no compensating `read` row: `class_heritage`/`extends_type_clause` are in the
  `_identifier_record` hard-exclusion set (`extractor.py:369-370`), so the miss is total.
- **E5 refined:** one `read` of the receiver object survives (`config`); it is the property
  (`TIMEOUT`) that never appears in any form, and the assignment-LHS occurrence is swallowed
  whole by the `left` field exclusion (`extractor.py:394-402`).
- **E10 is worse than recorded:** besides `abstract_class_declaration`,
  `abstract_method_signature` and class-field arrows (`public_field_definition`) also produce
  no declaration, and a missing class declaration silently de-qualifies
  `source_qualified_symbol` for *every* reference inside the class (`extractor.py:862-883`).
  `queries/tsx.scm` is byte-identical to `queries/typescript.scm`; both copies need every fix.
- **Capture names are decorative.** The reference-query dispatch (`extractor.py:332-343`)
  ignores everything after `reference.`; handlers re-dispatch on `node.type`
  (`_python_records` `extractor.py:577-669`, `_javascript_records` `extractor.py:671-821`).
  Adding a `.scm` pattern without a matching handler branch does nothing (constrains E6, E9,
  E13). Sub-captures like `arguments:` are match *constraints* — the mechanism behind E4.
- **E1 blocks R1 for TypeScript.** Override analysis built on inheritance rows works for
  Python/JS today but silently does nothing for TS until E1 lands. Sequence R1 after E1.
- **R3 dedupe constraint:** the synthetic declaration finding and the export hit share only
  their *edit span* (`edit_start_byte`/`edit_end_byte`) — `reference_id`, `kind`, and outer
  spans all differ, and the edit span is `(None, None)` when the identifier is not unique in
  the body, so a span key must never merge two `None` spans. The synthetic finding is emitted
  only when `cursor is None` (`reference_service.py:223`), so dedupe must consult the full
  pre-slice hit list, not the current page.
- **R4 nuance:** `limitations` is page-independent (computed before the slice at
  `reference_service.py:167`), but `review`/`likely_change` are built from the sliced page.
  The false `complete` occurs exactly when there are zero limitations project-wide and the
  last page holds only `exact` hits.
- **Cursor gaps beyond T2 (new):** the cursor payload does not bind the refactor operation or
  the page `limit`, so page 2 of `analyze_refactor` silently accepts a different `new_name`
  or signature spec, and inconsistent page sizes across pages.
- **S1/S2 compound:** the S1 chain is what *creates* backfill work for an embed-failed file;
  that backfill commit is what triggers the S2 promotion to `ready`. Fix together.
- **S5 extension (new):** two more bare asserts guard the same invariant (`storage.py:272`,
  `storage.py:309`) beyond the cited `:265`/`:360`; `restore_versions` (`storage.py:303-304`)
  shows the correct `RuntimeError` pattern. Also, the hotfix removed the only raiser of
  `ErrorCode.REFERENCE_INDEX_UNAVAILABLE` (`errors.py:26`) — it is now defined but never
  raised, so the S5 masking is thinner than when the review was written.
- **S6 is worse than recorded:** `backfill_references` calls `progress.clear()` in a
  `finally` immediately after the locked body returns (`indexing.py:242-243`), so the
  after-the-fact `committing` phase is published and erased with nothing in between.
- **Root cause of shipping green (new):** `tests/test_extractor_equivalence.py` already
  implements a committed-snapshot corpus gate over `tests/fixtures/extractor_corpus/` — but
  its `fingerprint()` (lines 33-53) covers `result.chunks` only. `references` and
  `declarations` are absent from the snapshot. This is why every E-item passed CI.
- **JSX component references (new, backlog gap):** `<Widget prop={value} />` yields only
  `read` rows; there is no component-reference/`type_use` kind for JSX element names, and
  closing tags are deduped by byte range. Tracked here as **E14** (Low, TSX only).

## Implementation constraints

- `@superpowers:test-driven-development` for every production-code task.
- Prefer `likely`/`unresolved` over a false `exact`; a construct starts as `likely` until a
  corpus fixture proves exact resolution is sound (design doc, "Resolver corpus").
- The failure path stays non-destructive: a failed file preserves its previous committed
  chunks and references. S1's fix must not regress this.
- Never make `completeness: "complete"` claim more than the extractor can deliver — the
  standing-limitation task (0.3) is the stopgap, corpus-gated removal (2.7) is the exit.
- `queries/tsx.scm` and `reference_queries/tsx.scm` mirror their `typescript` counterparts;
  every query change is applied to both copies (there is no include mechanism).
- Any new reference kinds/rows (E5 especially) grow the table; re-run the benchmark disk and
  cold-index gates (≤25% over baseline) before merging Phase 2.
- The extractor snapshot (`tests/fixtures/extractor_snapshot.json`) is regenerated
  deliberately via `python -m tests.test_extractor_equivalence` with the diff reviewed —
  never to make a failing test pass.

---

## Phase 0 — Regression net and honesty (before any behavior change)

The backlog buries its most important improvement in a closing remark: the corpus that would
have caught all of this was designed but never built. Build the net first, so every later
phase lands with a reviewed, corpus-visible diff, and stop the tool over-claiming while the
extraction gaps are still live.

### Task 0.1: Extend the extractor snapshot to structural records

**Files:** `tests/test_extractor_equivalence.py`, `tests/fixtures/extractor_corpus/*`,
`tests/fixtures/extractor_snapshot.json`

- Extend `fingerprint()` to emit tuples for `result.references` (kind, target_name,
  written_name, module_path, imported_name, alias, byte span, source_qualified_symbol,
  call-shape summary) and `result.declarations` (qualified_symbol, kind, parameter tuples).
- Extend the corpus fixtures with the defect constructs so the *current wrong output* is
  frozen: TS class heritage with generics and `implements` (E1), generic/union/intersection
  types (E2), `export *` / `export * as ns` (E3), tagged templates, `new Widget;`, Python
  generator-sole-argument (E4), member reads/writes and `{ onSave }` shorthand (E5), JS/TS
  decorators (E6), multi-key destructured params (E7 — `widget.tsx` currently hides this with
  a single-key pattern), callback-typed params (E8), bare/dynamic imports (E9), abstract
  class + class-field arrow (E10), `for (const item of items)` (E11), `metaclass=` (E12),
  `__all__` (E13), a JSX element (E14).
- Regenerate the snapshot deliberately. From here on, every extractor fix produces an
  explicit snapshot diff that review can check against the intended change.

### Task 0.2: Resolver corpus with per-defect expectations

**Files:** create `tests/fixtures/resolver_corpus/<language>/…`, `tests/test_resolver_corpus.py`

- Drive it with the existing `_indexed_service(tmp_path, files)` harness
  (`tests/test_references.py:26-42`), which builds a real repo, indexes it with
  `TinyEmbedder`, and returns a live `ReferenceService`.
- Each fixture is a small multi-file repo plus an expected reference set:
  `(path, line, kind, resolution, reason_code)` and, for refactor cases, the expected
  finding buckets, counts, and completeness state.
- Hard gate: **zero false positives in the `exact` category** (design doc contract).
- Encode the known gaps as `xfail(strict=True)` cases tagged with their defect ID
  (E1, E2, E3, E4, E5, E6, E7, E9, R1, R2, R3, R4). Each later fix must flip its xfails —
  that is the per-defect exit criterion, and `strict=True` means an accidental early fix or
  regression is also caught.
- Include verbatim the backlog's repro: `Base`/`Child` with `extends Base`, rename
  `Base -> Foundation`, expecting the inheritance finding and honest completeness.

### Task 0.3: Standing limitation for JS/TS extraction gaps

**Files:** `src/code_indexing_mcp/reference_service.py` (completeness block, `:327-348`,
and `_coverage_limitations`, `:801-847`), `README.md`, tool descriptions in `server.py`

- Until Phase 2 lands, any analysis whose scope includes JS/TS/TSX files appends a
  limitation (code e.g. `extraction_gaps`) stating that inheritance, type references,
  re-exports, member accesses, and decorators are not yet fully extracted, and the
  completeness state is capped at `complete_with_dynamic_limitations` (or `incomplete` for
  rename of a class/type, where the miss is a known wrong answer).
- This is the backlog's own "consider emitting a standing limitation" suggestion, promoted
  from a parenthetical to a required first-phase task: the tool must stop claiming
  `complete` for TS renames *now*, not after the extraction work.
- Removed in Task 2.7, gated on the corpus xfails for E1/E2/E3/E5 flipping.

---

## Phase 1 — Storage integrity (S1 + S2 + S6)

S1 is the only remaining defect that corrupts stored state and hides its own evidence; every
measurement taken while it is live is suspect. S2 compounds it. Both are small, localized
fixes in `indexing.py`.

### Task 1.1: Backfill must not launder embed-failed files (S1)

**Files:** `src/code_indexing_mcp/indexing.py` (`:254-272`, `:318`), `tests/test_indexing.py`

- In `_backfill_references_locked`, skip files whose stored record has `has_errors=True`:
  report them in `incomplete_paths` (already a supported channel —
  `ReferenceBackfillReport.incomplete_paths`, `models.py:294`, surfaced as limitations)
  instead of staging references for content the chunk table does not contain.
- This preserves the deliberately non-destructive failure path (`stage_failure`,
  `indexing.py:476-486`, keeps prior chunks) — the alternative of reverting `content_hash`
  would defeat the "don't re-parse every run" guard at `indexing.py:625` and is rejected.
- Add a divergence tripwire: a debug-level check (or test-only invariant) that compares
  chunk-row `content_hash` against the files row for a sample, so a future path that
  re-introduces divergence is detectable rather than self-hiding.
- Regression test: index a project where embedding fails for one file → run backfill → the
  file appears in `incomplete_paths`, its references are not committed, coverage hash does
  not equal the files hash for it, and a subsequent successful index heals it.

### Task 1.2: Backfill preserves project state (S2)

**Files:** `src/code_indexing_mcp/indexing.py` (`:362`, `_commit_staged` `:754-799`),
`tests/test_indexing.py`

- Backfill reads the prior state (`store.project_state`, `storage.py:227-231`) and preserves
  it through the commit — either by giving `_commit_staged` an explicit state pass-through or
  by re-marking via the existing `mark_project_state` (`storage.py:313-327`).
- Regression test: failed index leaves state `partial` → reference query triggers backfill →
  state is still `partial` and `project_status` still shows the failed files.

### Task 1.3: Backfill progress and report fidelity (S6)

**Files:** `src/code_indexing_mcp/indexing.py` (`:354-368`), `tests/test_indexing.py`

- Publish the `committing` phase *before* `_commit_staged`, not after (the current publish is
  immediately erased by the `finally: progress.clear()` at `:242-243`).
- Include `files_backfilled` (work parsed, then discarded) in the `stale_paths` early-return
  report so the retry loop's cost is visible.

---

## Phase 2 — JS/TS extraction correctness (E1, E2, E3, E5, E10, E11 + E6)

The silent misses that make TS/JS untrustworthy. Structured around the two shared mechanics
the verification identified, not item-by-item patches. Every task flips its corpus xfails
and produces a reviewed snapshot diff.

### Task 2.1: Name-descent helper for wrapper nodes (E1, E2, E12)

**Files:** `src/code_indexing_mcp/extractor.py` (`:784-791`, `:806-821`, `:639-648`),
`reference_queries/typescript.scm`, `reference_queries/tsx.scm`

- Add a helper that descends a wrapper node to its identifying name node(s):
  `type_identifier`/`identifier` leaves, unwrapping `generic_type` (emit the head name plus a
  `type_use` per type argument), `union_type`, `intersection_type`, `array_type`,
  `function_type`, and `type_arguments`.
- E1: in the `class_heritage` branch, descend `extends_clause`/`implements_clause` children
  to identifiers instead of `_capture_name` on the raw clause. Do not touch
  `extends_type_clause` — interface heritage already works; add a corpus case pinning that.
- E2: replace the verbatim `generic_type`/`type_annotation` capture with per-inner-name
  `type_use` rows.
- E12: in the Python `superclasses` branch, skip `keyword_argument` wrappers (their values
  already surface as `read`).

### Task 2.2: Broaden identifier fallback (E2/E5 safety net, E11)

**Files:** `src/code_indexing_mcp/extractor.py` (`_identifier_record` `:346-422`),
all four `reference_queries/*.scm`

- Extend the identifier capture/handling to `type_identifier`,
  `shorthand_property_identifier`, and `shorthand_property_identifier_pattern` so names the
  structured branches miss degrade to `read` instead of vanishing. Re-tune the hard-exclusion
  set (`:361-372`) in the same change — it currently suppresses exactly these contexts.
- E11: add `for_in_statement` to the field-exclusion map with `("left",)` so the JS `for…of`
  binding stops registering as a spurious `read`.

### Task 2.3: Module-edge completeness for exports and imports (E3, E9)

**Files:** `src/code_indexing_mcp/extractor.py` (`:674-723` imports, `:724-783` exports),
`tests/test_reference_extraction.py`

- E3: `export * from './x'` emits an `export` row carrying `module_path` (barrel edges become
  visible); `export * as ns from './x'` emits an `export` row with `alias="ns"` and the
  module path, replacing the current bogus `read` (add `namespace_export` handling; note the
  `_identifier_record` exclusion list has `export_clause` but not `namespace_export`).
- E9: a side-effect `import './polyfill'` (no `import_clause`) emits an `import` row with
  `module_path` and no imported name; `require('…')` and dynamic `import('…')` (grammar:
  `call_expression` with `function` of node type `import`) keep their call rows but gain
  `module_path` so they remain visible as likely/unresolved edges, per the design.

### Task 2.4: Member access, writes, and shorthand (E5)

**Files:** `src/code_indexing_mcp/extractor.py` (`:394-406`), `src/code_indexing_mcp/models.py`
(kind already declared at `:130`), `tests/test_reference_extraction.py`

- Emit `read` for non-call member access properties (`config.TIMEOUT`) — for JS/TS this means
  handling `property_identifier` in `member_expression` (it can never match `(identifier)`);
  for Python the `attribute` field exclusion at `:405-406` is the load-bearing line to relax.
- Emit `write` (first-ever producer of the declared kind) for assignment targets, including
  member-expression LHS currently swallowed by the `left` exclusion, and Python attribute
  assignment.
- Object shorthand `{ onSave }` records a `read` (covered mechanically by Task 2.2).
- Resolver check: `_lexical_declaration`'s kind filter (`reference_service.py:617`,
  `{"call","read","write"}`) already anticipates `write`; verify classification and the
  rename path treat the new rows sanely (corpus cases: rename a module constant).
- **Gate:** this multiplies row count; re-run benchmark disk/row-count numbers and record
  them in the PR. This raises S4's urgency (Phase 5) — note the interaction there.

### Task 2.5: TS declaration coverage (E10)

**Files:** `queries/typescript.scm`, `queries/tsx.scm` (byte-identical copies — patch both),
`tests/test_extractor.py`

- Add chunk-query patterns for `abstract_class_declaration` (class), `abstract_method_signature`
  (method), and `public_field_definition` with an arrow/function value (method).
- Assert the downstream effect the backlog flags: members inside an abstract class regain
  `A.run` qualification (`source_qualified_symbol` correctness) and the class becomes
  selectable by `analyze_refactor`.
- Note: this changes the *chunk* snapshot too — expect and review a chunk-fingerprint diff.

### Task 2.6: JS/TS decorators (E6)

**Files:** `reference_queries/javascript.scm`, `typescript.scm`, `tsx.scm`,
`src/code_indexing_mcp/extractor.py` (`_javascript_records`)

- Add `decorator` query patterns *and* a `node.type == "decorator"` handler branch mirroring
  the Python one (`extractor.py:628-638`) — a query pattern alone does nothing because
  capture names are discarded. Handle `@Name`, `@ns.Name`, and `@Factory()` (the factory
  call keeps its `call` row; the decorator row is additional).
- Keep `"decorator"` in the read-exclusion set so the identifier fallback doesn't duplicate.

### Task 2.7: Retire the standing limitation

**Files:** `src/code_indexing_mcp/reference_service.py`, corpus

- With E1/E2/E3/E5 corpus xfails flipped, remove the Task 0.3 cap for the covered constructs.
  Keep a narrow limitation only for what genuinely remains (e.g. E14 JSX component renames if
  not yet done).

---

## Phase 3 — Remaining silent misses (E4, R1, E7, E8)

### Task 3.1: Dropped call forms (E4)

**Files:** all four `reference_queries/*.scm`, `src/code_indexing_mcp/extractor.py`
(`_call_shape` `:526-554`, `:407-408`)

- Relax the mandatory `arguments: (arguments)` constraint so tagged templates
  (`arguments: template_string`), Python generator-sole-argument
  (`arguments: generator_expression`), and `new Widget;` (no `arguments` child at all) still
  produce `call` rows.
- `_call_shape` must tolerate non-`arguments` nodes: a template literal is not a positional
  arg list (do not count `string_fragment`s); a generator argument is one positional with
  spread-like uncertainty → signature analysis routes to `review`, not a fabricated match.
- Corpus: `gql`/`styled.div` templates, `new Widget;`, `summarize(x for x in items)`.

### Task 3.2: Override analysis (R1) — after E1

**Files:** `src/code_indexing_mcp/reference_service.py`, `tests/test_refactors.py`, corpus

- Renaming `Base.handle` must surface `Child.handle`. All the data exists:
  - Inheritance rows read "class `<source_qualified_symbol>` extends name `<target_name>`"
    (`extractor.py:638-648` Python, `:783-791` JS/TS, correct for TS once E1 lands).
  - Binding the base name to the selected declaration's file reuses `_import_targets`
    (`reference_service.py:703-714`) plus same-file declarations.
  - The override itself is the declaration row with qualified name `<subclass>.<method>` —
    parent-class recovery via `rsplit(".", 1)` is already idiomatic in this module
    (`:626`, `:727`).
- Emit overrides as `likely_change` (rename should usually propagate) with a dedicated
  reason code (e.g. `override_of_renamed_method`), never `exact` — dynamic dispatch can't be
  proven structurally. Walk transitive subclasses with a visited set.
- TS coverage is corpus-gated on E1 (Task 2.1); Python/JS cases must pass regardless.

### Task 3.3: Destructured parameters (E7)

**Files:** `src/code_indexing_mcp/extractor.py` (`:488-497`), `tests/test_reference_extraction.py`

- Reuse the existing recursive pattern walker `_binding_identifiers` (`extractor.py:556-575`,
  already correct for `object_pattern`/`array_pattern`/`pair_pattern`/`assignment_pattern`/
  `rest_pattern`) to expand destructured parameters.
- Decide the `ParameterShape` semantics explicitly rather than fabricating: represent the
  destructured slot as one positional parameter *marked as destructured* (so positional
  arity stays correct) and route signature comparisons that depend on its inner names to
  `review`. Expanding to N flat params would corrupt positional matching for every caller.
- Replace the single-key TSX fixture that hides this with a multi-key one (done in 0.1).

### Task 3.4: Default-detection heuristic (E8)

**Files:** `src/code_indexing_mcp/extractor.py` (`:516-517`)

- TS needs no text heuristic at all: `required_parameter` vs `optional_parameter` node types
  plus the `value` field are authoritative. Restrict the `"=" in child_text` fallback to
  grammars that need it (Python's default nodes expose a `value` field too — check whether
  the heuristic has any remaining justification; delete it if not).
- Test: `function h(handler: (e: Event) => void, n: number)` → `handler` is required;
  `missing_required_parameter` fires for a callback parameter.

---

## Phase 4 — Resolution and surface honesty (R2, R3, R4, T2)

### Task 4.1: Findings dedupe (R3)

**Files:** `src/code_indexing_mcp/reference_service.py` (`:219-305`), `tests/test_refactors.py`

- Suppress the synthetic declaration finding's duplicate by keying on
  `(path, edit_start_byte, edit_end_byte)` against the **full pre-slice hit list** (the
  export hit may fall on a later page than the page-1-only synthetic finding), never merging
  `(None, None)` spans.
- Cover all three shapes the verification identified: `export function answer()`,
  `export const answer = () => {}`, `export default class Foo`.
- `counts.must_change` counts one entry per distinct edit.

### Task 4.2: Page-independent completeness and counts (R4)

**Files:** `src/code_indexing_mcp/reference_service.py` (`:327-348`), `tests/test_refactors.py:199-219`

- Compute `counts` and `completeness` from the full classified result set *before* slicing,
  so every page reports the same totals and state; only `findings` pages. (`limitations` is
  already page-independent; `review`/`likely_change` presence is the part that must move
  pre-slice.)
- `state` stays `incomplete` while `cursor is not None` is wrong in the other direction too —
  a mid-stream page is not "incomplete coverage". Either keep it and document it, or add a
  distinct indicator (`has_more_pages`) and reserve `incomplete` for coverage gaps. Choose
  one; the test currently encoding the false `complete` (`test_refactors.py:219`) is
  rewritten either way.

### Task 4.3: Cursor hardening (T2 + new gaps)

**Files:** `src/code_indexing_mcp/reference_service.py` (`_decode_cursor` `:860-869`,
cursor block `:75-90`, encode `:855-858`), `src/code_indexing_mcp/errors.py`,
`tests/test_references.py:222`

- Validate all six payload fields (presence and type) in `_decode_cursor`; raise
  `CodeIndexingError` with a structured code (reuse `STALE_CURSOR` or add `INVALID_CURSOR`)
  instead of bare `KeyError`/`ValueError` — plain exceptions bypass `_with_error_details`
  (`server.py:601-619`) and reach the client as `'version'`.
- Bind the missing dimensions into the payload: a digest of the refactor operation (so
  page 2 with a different `new_name`/signature is rejected) and the page `limit`.
- Update `tests/test_references.py:222` (currently asserts loose `ValueError`); check
  `errors.py` for whether `CodeIndexingError` subclasses `ValueError` before choosing how.

### Task 4.4: Re-export chains (R2)

**Files:** `src/code_indexing_mcp/reference_service.py` (`_import_targets` `:703-714`,
`_module_matches` `:731-753`, `_classify` `:638-701`), corpus

- Add a chain-following variant *beside* `_import_targets` (it is called from both candidate
  admission `:607` and classification `:665/:678` — loosening it in place changes both):
  when the import's module resolves to a file that is not the declaration's, look for an
  import/export row in that file binding the same name and recurse with a visited set and a
  small depth cap.
  - JS/TS: `export { x } from './y'` rows already carry `module_path`/`imported_name`
    (`extractor.py:724-747`); `storage.py:422 imports_for` is the fitting pushdown.
  - Python: the chain hop is the `from .impl import b` *import* row inside `pkg/__init__.py`
    (no export kind exists until E13).
- A proven chain classifies `exact` with a dedicated reason (`reexport_chain`); an unproven
  one gets a *distinguishable* degraded reason (e.g. `unproven_reexport`) instead of the
  generic `name_only_candidate`/`ambiguous_symbol`. Note the current asymmetry:
  `ambiguous_symbol` is in `_LIMITATION_REASONS` but `name_only_candidate` is not — decide
  deliberately which new reasons surface as limitations.
- Corpus requirement: zero false `exact` — a same-named symbol exported from a *different*
  chain must not bind.

---

## Phase 5 — Scale, robustness, and cleanup (S3, S4, S5, E13, E14, T1, T3–T7)

### Task 5.1: Pushdown resolver queries (S4) — before pointing at a large repo

**Files:** `src/code_indexing_mcp/storage.py` (`:388-392`, `:415-430`, `:855-878`),
`src/code_indexing_mcp/reference_service.py` (`:93`, `:204`)

- The three existing primitives (`declaration_shapes`, `imports_for`,
  `target_name_candidates`) are necessary but not sufficient: the resolver also needs all
  coverage rows (`_coverage_limitations` `:801-818`), class declarations (`:103-107`),
  per-file declarations (`_lexical_declaration`), and per-file imports
  (`_imports_by_file` `:585-590`). Add `version=` kwargs (mechanical — `_reference_rows`
  already supports it), a `record_kind` filter to `target_name_candidates`, and the missing
  query shapes (coverage-only; declarations/imports restricted to the candidate files).
- Fetch plan: candidates by target name/alias → the files they live in → per-file
  declaration/import context for exactly those files → coverage rows. Eliminate
  `analyze_refactor`'s second full scan (`:204`) by reusing the first fetch.
- Measure before/after on this repo (verified baseline: 21,079 rows, ~1.5 rows per source
  line — and growing after Task 2.4) and record the numbers.

### Task 5.2: Rejected-file tombstones (S3)

**Files:** `src/code_indexing_mcp/indexing.py` (`:614-624`), `src/code_indexing_mcp/application.py`
(`_project_is_stale` `:653-670`), `tests/test_indexing.py`

- Instead of dropping a NUL-byte/undecodable file from storage while the path-based scanner
  (`scanner.py:233-273` — never decodes content) keeps yielding it forever, persist a files
  row flagged as rejected (`has_errors=True`, rejection reason, no chunks/references). Then
  `current.keys() == existing.keys()` holds and every reference query stops triggering a
  full re-index under the global lock (`application.py:621-627`).
- Task 1.1's `has_errors` skip already makes such rows safe for backfill — sequence after it.
- Test: a `.py` file full of NULs → project indexes, goes `ready`/`partial` deterministically,
  and `ensure_reference_index` does not re-index on every call.

### Task 5.3: Missing-table guard and asserts (S5)

**Files:** `src/code_indexing_mcp/reference_service.py`, `src/code_indexing_mcp/storage.py`
(`:265`, `:272`, `:309`, `:360`)

- `ReferenceService` raises `CodeIndexingError(REFERENCE_INDEX_UNAVAILABLE)` (currently
  defined but raised nowhere) when the reference table is missing, distinguishing it from a
  legitimately empty table, instead of trusting callers to have run `ensure_reference_index`.
- Replace all four bare `assert`s (two beyond the backlog's citations) with real exceptions,
  following the `restore_versions` pattern (`storage.py:303-304`).

### Task 5.4: Python exports and JSX components (E13, E14)

- E13: add an `export` emission for `__all__` entries (string list → export rows naming the
  target) so a rename flags the `__all__` entry; needs both a `python.scm` pattern and a
  `_python_records` branch. Also feeds the later safe-deletion phase.
- E14: emit a component-reference row (kind `type_use` or a documented choice) for JSX
  element names in TSX so renaming a component finds its usages.

### Task 5.5: Benchmark and test truthfulness (T1, T3)

**Files:** `src/code_indexing_mcp/benchmark.py` (`:55-66`), `src/code_indexing_mcp/indexing.py`
(`:644-645`), `tests/test_benchmark.py`

- T1: time reference extraction separately (its own `timer.measure` around
  `_structural_records` inside `extract`, or an extractor-reported split) instead of
  labeling the whole parse+chunk+reference phase; report per-run staged structural rows,
  not the whole-project total that makes all four scenarios identical.
- T3: make the benchmark test exercise real values — give the fake application a real (tiny)
  store or assert nonzero via a seeded fixture; remove the `getattr(app, "store", None)`
  silent-zero fallback that a rename would never trip.

### Task 5.6: Docs, skills, and schema tests (T4–T7)

- T4: recurse into `$defs` in `tests/test_server.py:1490-1501`; add `Field(description=…)`
  to all four `DeclarationSelector` fields (`models.py:302-308`) — `path` is repo-relative
  POSIX, `project` accepts id/name/path, `qualified_symbol` is dotted.
- T5: assert skills state the language coverage (Python/JS/TS/TSX; `UNSUPPORTED_LANGUAGE`
  otherwise) and completeness handling — currently neither skill states coverage at all.
- T6: fix `skills/indexed-review/SKILL.md:39` — the last remaining "definitions and call
  sites" claim for `find_symbol` (all other surfaces already corrected).
- T7: fix the `evidence` description in `README.md:307-310` *and* the same wording in
  `server.py:1093-1094`: for `signature_change` the bucket holds compatible call sites, not
  just rename aliases.

---

## Acceptance gates

1. **Corpus:** zero false `exact` across the resolver corpus; every defect-tagged
   `xfail(strict=True)` flipped by the end of its phase. The backlog's `Base`/`Foundation`
   repro reports the `extends Base` finding and an honest completeness state.
2. **Snapshot discipline:** every extractor change ships with a reviewed
   `extractor_snapshot.json` diff; no regeneration to silence a failure.
3. **Performance:** cold-index ≤ 25% over the pre-#23 baseline (design gate) re-verified
   after Phase 2's row growth; S4 numbers measured and recorded before/after Task 5.1.
4. **Honesty invariant:** at no point between phases may `completeness: "complete"` be
   reachable for a construct the extractor is known not to produce — the standing
   limitation (0.3) covers the window until 2.7 retires it.
5. Full suite green: `uv run pytest -q` with the all-groups environment.

## Sequencing rationale (changes from the backlog's suggestion)

The backlog proposed S1 → {E1,E2,E3,E5} → {R1,E4} → S4 → rest. This plan keeps that spine
and changes four things, each grounded in the re-verification:

1. **The corpus moves from a closing remark to Phase 0.** It was designed, never built, and
   its absence is the stated reason every defect shipped green. Building it first turns all
   later phases into reviewed, gated diffs — and `test_extractor_equivalence.py` already
   provides the template, missing only structural records in its fingerprint.
2. **The standing TS/JS limitation is promoted from "consider" to a required Phase 0 task.**
   It is a few lines, and it stops the tool giving known-wrong `complete` verdicts for TS
   renames months before the extraction work finishes.
3. **E10 is pulled into Phase 2** (backlog: "everything else"). It corrupts
   `source_qualified_symbol` for every reference inside an abstract class and makes the
   class unselectable — that is resolution-input corruption, not cleanup, and it is a
   two-file query fix.
4. **Explicit dependencies:** R1 after E1 (TS inheritance rows are unusable until then),
   S2 with S1 (the same backfill commit is both the laundering and the promotion),
   S3 after S1's `has_errors` skip (which makes tombstone rows safe for backfill).
