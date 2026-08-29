# Refactor Patch Emission — Implementation Plan

**Goal:** Add an `emit_refactor_patch` tool that turns the deterministic
`must_change` findings of a rename into a byte-exact unified diff the caller can review
and apply with `git apply` or their own edit tooling. The server's no-write stance is
preserved: producing a patch is analysis, applying it stays with the client.

**Design reference:** `docs/plans/2026-08-27-refactor-patch-emission-design.md`.
All code coordinates below were verified against the current tree on 2026-08-28.

**Baseline:** `uv run ruff format --check . && uv run ruff check . && uv run mypy src &&
uv run pytest -n auto` green before Step 0, and after every step.

## Decisions settled before implementation

- **D1 — Shared unpaginated analysis core (settled).** `analyze_refactor` page-filters
  findings (`page_ids`, `src/code_indexing_mcp/reference_service.py:629`) and
  `_find_references_with_records` hard-caps `limit` at 1–500
  (`reference_service.py:189-190`). Emission needs *all* findings, so cursor-looping the
  public method would re-materialize the full record set once per 500-hit page. Instead,
  extract the body of `ReferenceService.analyze_refactor`
  (`reference_service.py:534-662`) into a private `_rename_analysis(..., paginate: bool)`
  core; `analyze_refactor` becomes a thin wrapper with pagination on (zero behavior
  change), `emit_refactor_patch` calls the same core with pagination off. Same function,
  so the two tools cannot disagree about what is `must_change`.
- **D2 — Conflict mapping reuses the serve-time staleness gate.** The analysis already
  suppresses hits from files whose on-disk bytes fail the coverage-hash gate
  (`_hits_and_limitations`, `reference_service.py:433-478`) and reports `stale_file`
  limitations (`:515-530`) — stale files never reach `must_change`. Emission maps
  `stale_file` limitation paths into the `conflicted` bucket (reason vocabulary
  preserved) and keeps per-offset byte-slice equality plus overlap checks as
  defense-in-depth, per the design.
- **D3 — Encoding is UTF-8, by construction.** `_edit_span` matches identifiers as
  UTF-8-encoded bytes (`reference_service.py:2160`), so any span the resolver matched
  *is* the UTF-8 encoding of the name; replacement bytes =
  `new_name.encode("utf-8")` satisfies "encoded as the file is" for every emittable
  hunk. The renderer operates on raw on-disk bytes with the BOM intact — edit offsets
  are raw-absolute because `_edit_span` adds the removed BOM length back
  (`reference_service.py:2156-2163`, `_BOM` at `:56`; proven by
  `tests/test_refactors.py:79`).
- **D4 — New `ErrorCode.UNSUPPORTED_OPERATION`** (`src/code_indexing_mcp/errors.py`)
  for signature-change emission. The message names the limitation (synthesized argument
  lists are language-specific and easy to get silently wrong) and points back to
  `analyze_refactor` for the analysis.
- **D5 — The renderer is a new leaf module** `src/code_indexing_mcp/patching.py`, pure
  bytes → unified diff, importing nothing from the resolver or storage — design
  delivery step 1 (fixture-level tests independent of the resolver).
- **D6 — README gets the count fix only (settled).** Update the "twelve tools" sentence
  (`README.md:260`) to the true registered count (17 after this feature) and add just
  the `emit_refactor_patch` table row. The four storage/history tools already
  registered but missing from the table stay a separate cleanup; no broader table
  repair here.
- **D7 — Byte-fidelity tests use real `git apply` (settled).** Apply the emitted patch
  in a `git init`-ed copy of the fixture repo and compare the result to the expected
  renamed sources — the design's stated interface, so testing against the real tool is
  the strongest contract. Guarded with `pytest.mark.skipif(not shutil.which("git"))`;
  CI always has git.
- **D8 — `conflicted`/`unapplied` entries reuse `RefactorFinding` verbatim (settled).**
  Findings keep their full shape — path, offsets, `reason_code`, `explanation` — so the
  `analyze_refactor` reason vocabulary carries through unchanged; conflicts append a
  conflict explanation. No slimmer omission record.

## Mechanics discovered during research (accounted for below)

- **Every tool crosses five layers, each needing a sibling:** the MCP tool
  (`analyze_refactor`, `src/code_indexing_mcp/server.py:1460-1496`) →
  `Application.analyze_refactor` (`src/code_indexing_mcp/application.py:1842-1861`) +
  `_analyze_refactor_for_target` (`:1863-1881`, via `_prepare_reference_query`
  `:1762-1780` and `store.partition_access`) → `ReferenceService.analyze_refactor`
  (`reference_service.py:534`); plus the daemon dispatch entry
  (`src/code_indexing_mcp/daemon.py:408-411`) and `BrokerApplication.analyze_refactor`
  (`daemon.py:689-694`).
- **`_ReferenceQuery`** (`reference_service.py:91-104`) already carries everything the
  patch response needs for correlation: `response.snapshot_version`, `partition_id`,
  `slot_id`, `activation_epoch`, `root`, and the per-query `sources` byte cache.
- **`_operation_digest`** (`reference_service.py:2274-2288`) already fingerprints an
  operation for cursor binding; reuse verbatim in the patch response.
- **`_file_bytes`** (`reference_service.py:2199-2218`) reads with project-root
  confinement, strips the BOM, and caches per query; emission reconstructs raw bytes by
  re-prepending `_BOM` when the removed offset is non-zero.
- **Gates pinning the tool surface** (all move with Step 4):
  `tests/test_server.py:128-146` (exact tool set + `len(tools) == 16`), `:1650-1667`
  (READ_ONLY / AUTO_REGISTERING / WRITE annotation frozensets), `:1818-1843`
  (input-schema and description contract tests); `tests/test_skills.py:64-72`
  (skills must name the structural analysis tools and their language/completeness
  contracts); `README.md:260` (tool count sentence), `:278` (table row),
  `:317-331` (refactoring workflow); the backlog entry
  (`docs/plans/2026-08-27-feature-backlog.md:14`).
- **README count drift exists today:** the sentence says "twelve tools" while 16 are
  registered (the four storage/history tools are documented elsewhere in the file).
  D6 addresses the count only.
- **Existing no-regression proofs for the core extraction** (Step 2):
  `tests/test_refactors.py:200` (pagination independent of completeness and counts),
  `:426` (reference table fetched once), `:463` (full hit list classified once),
  `:1072` (edit spans suppressed for a stale file).

## Cross-cutting invariants (apply to every step)

1. **`analyze_refactor` behavior is unchanged.** The Step 2 core extraction keeps every
   existing refactors test passing untouched; the four tests cited above are the proof.
2. **No write.** Emission reads source bytes and returns text; no code path opens a
   file for writing. The tool carries `_READS_AND_REGISTERS` (registering-read), never
   `readOnlyHint`.
3. **Determinism.** Identical selector, operation, file bytes, and snapshot produce a
   byte-identical patch: files render in sorted path order, edits apply in offset
   order, nothing depends on time, concurrent-read ordering, or locale.
4. **A partial patch never reads as finished.** `completeness.state` is `complete` only
   when the underlying analysis was `complete` *and* `unapplied` and `conflicted` are
   empty; otherwise `complete_with_dynamic_limitations`, or `incomplete` when the
   analysis was.
5. **After every batch:** `uv run ruff format . && uv run ruff check . && uv run mypy
   src && uv run pytest -n auto` (per AGENTS.md; format before push — CI rejects
   unformatted code at the Format step and nothing else runs until it passes).

---

## Step 0 — Renderer module + fixture unit tests (no resolver, no index)

1. **`src/code_indexing_mcp/patching.py`:**
   - `apply_edits(original: bytes, edits)` — splice sorted, non-overlapping
     `(start, end, replacement)` byte spans into `original`; raise on overlap. Shared
     by the emission pipeline and by tests constructing expected bytes.
   - `render_unified_diff(path, original: bytes, edited: bytes, context_lines: int)
     -> str | None` — `None` when the byte strings are identical. Headers
     `diff --git a/{path} b/{path}`, `--- a/{path}`, `+++ b/{path}`; hunks from
     `difflib.SequenceMatcher.get_grouped_opcodes` over keepends line lists with
     standard `@@ -start,count +start,count @@` offsets; `\ No newline at end of
     file` markers after any `-`/`+` line whose source lacks a final terminator
     (difflib omits them; `git apply` needs them); CRLF and BOM preserved by working
     purely on bytes.
2. **`tests/test_patching.py`** (inline byte fixtures only): identical bytes → `None`;
   single edit mid-file; adjacent edits collapsing into one hunk; edits at byte 0 and
   at EOF; gap smaller than context merges two hunks, larger splits; `context_lines=0`;
   CRLF file; BOM-prefixed file; missing final newline on the `-` side only, the `+`
   side only, and both; non-ASCII identifier replacement with multibyte content before
   the edit (byte offsets and hunk line numbers stay correct); two calls byte-identical
   (determinism).

**Verify:** full gate green; no other module touched.

## Step 1 — Models + error code

1. `errors.py`: add `UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"` to `ErrorCode`.
2. `models.py`, after `RefactorAnalysis` (~`:515`), both `FrozenModel`:
   - `PatchEdit`: `path`, `edit_start_byte`, `edit_end_byte`, `old_text`, `new_text`.
   - `RefactorPatch`: `selected`, `operation`, `patch: str` (empty when nothing
     applied), `edits: list[PatchEdit]`, `applied: int`, `unapplied:
     list[RefactorFinding]`, `conflicted: list[RefactorFinding]`,
     `snapshot_version: int`, `slot_id: str | None`, `operation_digest: str`,
     `completeness: CompletenessReport` (D8: findings reused verbatim).

**Verify:** full gate green.

## Step 2 — Service: core extraction + `ReferenceService.emit_refactor_patch`

1. **Extract** `reference_service.py:534-662` into
   `_rename_analysis(selector, operation, *, limit, cursor, backfill, partition,
   paginate: bool)` returning the `RefactorAnalysis` plus the query metadata
   (`snapshot_version`, `slot_id`) emission needs. `analyze_refactor` delegates with
   `paginate=True`; the `page_ids` filter (`:629`) and page-cursor handling apply only
   when paginating. Existing tests `:200`, `:426`, `:463` are the no-regression proof.
2. **`emit_refactor_patch(selector, operation, *, context_lines=3, backfill,
   partition) -> RefactorPatch`:**
   - Not a `RenameOperation` → `CodeIndexingError(UNSUPPORTED_OPERATION, ...)`
     (D4); rename → reuse `_validate_rename` (`reference_service.py:665-672`).
   - Run `_rename_analysis(..., paginate=False)` against the pinned snapshot.
   - Partition findings: `must_change` with non-null `edit_start_byte`/`edit_end_byte`
     → candidates; null-offset `must_change`, `likely_change`, `review`, and override
     findings → `unapplied` with existing reason codes; `stale_file` limitation paths
     → `conflicted` (D2).
   - Per file in sorted path order: read bytes through `_file_bytes` (same
     project-root confinement), reconstruct raw bytes with the BOM re-prepended.
     Verify each candidate's fresh slice `[edit_start:edit_end]` equals the
     analysis-time slice from the `sources` cache — mismatch or unreadable file →
     `conflicted`. Sort by `(edit_start, edit_end)`; a span intersecting its
     predecessor → `conflicted` (defensive against resolver regressions).
   - `patching.apply_edits` with `new_name.encode("utf-8")` replacements (D3), then
     `render_unified_diff` per file; concatenate in sorted path order.
   - Completeness per invariant 4; counts applied/unapplied/conflicted;
     `snapshot_version`/`slot_id` from the query; `operation_digest` via
     `_operation_digest`.
3. **Tests** (extend `tests/test_refactors.py`, reuse `_indexed_service`):
   - Multi-file rename: patch contains both files in sorted order with correct
     `a/`/`b/` headers; `edits` match the `must_change` offsets; applying with
     `git apply` in a `git init`-ed copy reproduces the expected renamed sources (D7,
     skipif no git).
   - Aliased import (`authorize as check`): only the imported name is replaced.
   - Qualified member call (`auth.authorize(u)`): only the member name is replaced.
   - `likely_change`/`review` findings (spread-call fixture) never appear in the patch
     and always appear in `unapplied` with their reasons.
   - Stale file after indexing (mirror `:1072`): path reported in `conflicted` with
     `stale_file`, patch omits its hunks, completeness degrades; after a re-index the
     hunks appear.
   - Synthetic-findings unit tests for the slice-mismatch and overlap defenses
     (overlap → omitted, never merged).
   - `UNSUPPORTED_OPERATION` for a signature-change operation.
   - CRLF and non-ASCII-identifier fixtures emit byte-exact hunks.
   - Emit twice → byte-identical output.
   - `paginate=False` returns all findings where `analyze_refactor(limit=1)` returns
     one (mirror the pattern at `:200`).

**Verify:** full gate green; `tests/test_refactors.py:200/:426/:463/:1072` untouched
and passing.

## Step 3 — Application + daemon plumbing

1. `Application.emit_refactor_patch` + `_emit_refactor_patch_for_target` mirroring
   `application.py:1842-1881`: same `_resolve_reference_project`,
   `_run_repository_stable_query`, `_prepare_reference_query`, and
   `store.partition_access` wrapper; threads `context_lines` through.
2. `daemon.py`: `BrokerApplication.emit_refactor_patch` (sibling of `:689-694`,
   validating the operation via `_REFACTOR_OPERATION.validate_python`) and a
   `_dispatch` entry (sibling of `:408-411`).
3. Tests: `tests/test_application.py` and `tests/test_daemon.py` happy-path and
   dispatch smoke, mirroring the existing `analyze_refactor` coverage; the daemon path
   round-trips the `RefactorPatch` model.

**Verify:** full gate green.

## Step 4 — MCP tool, contracts, docs sweep

1. **`server.py`**, registered directly after `analyze_refactor` (`:1460-1496`):
   `emit_refactor_patch(selector, operation, context_lines=3)` with
   `_READS_AND_REGISTERS` (registering-read per design — it refreshes a stale index
   before analyzing), `@_with_error_details`, selector/operation parameters identical
   to `analyze_refactor`, and `context_lines: Annotated[int, Field(ge=0, le=50)] = 3`
   (cap in the style of the search limits). Description states: emission-only (never
   edits source; application stays with the caller's tooling, e.g. `git apply`),
   rename-only (signature change → `UNSUPPORTED_OPERATION`), review
   `likely_change`/`review` findings before applying, and the completeness contract —
   a partial patch can never read as a finished rename.
2. **`tests/test_server.py`:** tool set + `len(tools) == 17` (`:128-146`);
   `AUTO_REGISTERING_TOOLS` gains `emit_refactor_patch` (`:1653`); input-schema test
   (properties == selector/operation/context_lines, mirroring `:1818`); description
   contract test naming emission-only semantics and `UNSUPPORTED_OPERATION`
   (mirroring `:1830`).
3. **`README.md` (D6):** tool-table row after `:278`; the `:260` count sentence moves
   to the true registered count; the refactoring-workflow section (`:317-331`) gains
   an emission paragraph — run `analyze_refactor` first, review, then
   `emit_refactor_patch` for the deterministic subset.
4. **Skills:** `src/code_indexing_mcp/skills/impact-analysis/SKILL.md` and
   `feature-dev/SKILL.md` mention `mcp__code-indexing-mcp__emit_refactor_patch` as the
   patch-production step after reviewing an `analyze_refactor` result;
   `tests/test_skills.py:64` pin extended to require it alongside the two analysis
   tools.
5. **Housekeeping:** remove the "Patch emission" entry from
   `docs/plans/2026-08-27-feature-backlog.md` "In flight"; ship note
   (`2026-08-27-refactor-patch-emission-shipped.md`) per repo convention.

**Verify:** full gate green; tool-schema snapshot shows the new tool with
`_READS_AND_REGISTERS` annotations.

## Sequencing and degradation

Step 0 is resolver-independent and independently shippable; Steps 1 → 2 → 3 → 4 land
in order with the gate green after each. Degradation is structural, not behavioral:
every non-deterministic, stale, or unverifiable finding lands in `unapplied` or
`conflicted` — never in the patch — and completeness degrades before any hunk is
trusted. If byte-slice verification proves noisy in practice it can only over-report
conflicts, never emit a wrong patch.

## Later phases (out of scope here)

- Signature-change edit scripts with per-language argument synthesis (design non-goal;
  revisit with per-language edit-script generators).
- Patch emission for C, C++, and Lua renames as their structural support lands
  (`2026-08-27-structural-references-more-languages-design.md`; the eight current
  `STRUCTURAL_LANGUAGES` work on day one because emission is language-agnostic).
- Composability with the transitive impact radius
  (`2026-08-27-transitive-impact-radius-design.md`): one patch covering the
  deterministic subset of a multi-hop rename cascade.
