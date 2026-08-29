# Refactor Patch Emission: Shipped

## Outcome

`emit_refactor_patch` turns the deterministic `must_change` findings of a rename into a
byte-exact unified diff the caller can review and apply with `git apply` or their own edit
tooling. The server's no-write stance is preserved: producing a patch is analysis, applying it
stays with the client. Emission re-derives the findings through the same code path
`analyze_refactor` serves, so the two tools cannot disagree about what is `must_change`.

## Delivered

### Renderer (Step 0)

- `src/code_indexing_mcp/patching.py`, a pure bytes → diff leaf with no imports from the
  resolver or storage: `apply_edits` splices sorted, non-overlapping byte spans (raising on
  overlap rather than guessing an order), and `render_unified_diff` emits
  `diff --git a/{path}` headers with `@@ -start,count +start,count @@` hunks from
  `difflib.get_grouped_opcodes` over keepends line lists.
- Difflib omits `\ No newline at end of file`, which `git apply` needs; the renderer re-adds
  it after any line (context included) whose content lacks a final terminator, and terminates
  the patch line itself before the marker. Working purely on bytes keeps CRLF terminators and
  a leading BOM intact. 16 fixture-level tests pin all of this, including a real
  `git apply` smoke.

### Models and error code (Step 1)

- `ErrorCode.UNSUPPORTED_OPERATION` (`errors.py`) for signature-change emission; the message
  names the limitation (synthesized argument lists are language-specific and easy to get
  silently wrong) and points back to `analyze_refactor`.
- `PatchEdit` and `RefactorPatch` (`models.py`, both `FrozenModel`): the structured edit list
  beside the diff, `applied`/`unapplied`/`conflicted` with the `analyze_refactor` reason
  vocabulary (`RefactorFinding` reused verbatim), `snapshot_version`/`slot_id`/
  `operation_digest` for correlating with the motivating analysis, and a `completeness`
  report.

### Service (Step 2)

- `ReferenceService.analyze_refactor`'s body moved into `_rename_analysis(..., paginate)`;
  the public method delegates with pagination on (zero behavior change — the fetch-once,
  classify-once, and pagination-independence proofs pass untouched), and emission calls the
  same core with pagination off, so it sees every finding without cursor-looping pages.
- `ReferenceService.emit_refactor_patch` → `_render_patch`: partition findings
  (`must_change` with offsets → candidates; everything else → `unapplied`), recompute the
  serve-time coverage-hash gate over every file the analysis read, verify each candidate's
  fresh byte slice still spells the identifier, and refuse overlapping spans. Only then are
  edits spliced with `new_name.encode("utf-8")` against raw on-disk bytes (BOM re-prepended)
  and rendered per file in sorted path order.
- `stale_file` reporting is structured: files whose hits the gate had already suppressed
  carry no findings, so `_render_patch` synthesizes one conflicted finding per such path
  (`reason_code="stale_file"`), and per-candidate conflicts reuse the finding with a conflict
  explanation appended. `completeness` degrades before any hunk is trusted: `incomplete`
  inherits from the analysis, and any `unapplied`/`conflicted` entry caps the patch at
  `complete_with_dynamic_limitations`.
- Emission is deterministic: identical selector, operation, bytes, and snapshot produce a
  byte-identical patch.

### Plumbing (Step 3)

- `Application.emit_refactor_patch` + `_emit_refactor_patch_for_target` mirror the
  `analyze_refactor` path (`_resolve_reference_project`, `_run_repository_stable_query`,
  `_prepare_reference_query`, `partition_access`), threading `context_lines` through.
- `daemon.py`: `BrokerApplication.emit_refactor_patch` (round-trips the `RefactorPatch`
  model) and a `_dispatch` entry validating the operation through `_REFACTOR_OPERATION`.

### Tool surface and contracts (Step 4)

- `server.py` registers `emit_refactor_patch(selector, operation, context_lines=3)` directly
  after `analyze_refactor`, with `_READS_AND_REGISTERS` (registering-read: it refreshes a
  stale index before analyzing), `@_with_error_details`, and `context_lines` bounded
  `ge=0, le=50`. The description states the contract: emission-only (never edits source;
  application stays with the caller's tooling), rename-only (`UNSUPPORTED_OPERATION`),
  review `likely_change`/`review` before applying, and the completeness rule that a partial
  patch can never read as a finished rename.
- Contract tests: exact tool set grew to 17, `AUTO_REGISTERING_TOOLS` gained the tool, the
  input schema (properties and `context_lines` bounds) and the description wording are
  pinned.
- `impact-analysis` and `feature-dev` skills name `emit_refactor_patch` as the
  patch-production step after reviewing an `analyze_refactor` result; the skills test pins
  it alongside the two analysis tools. README: tool table row, the corrected registered-tool
  count, and an emission paragraph in the refactoring workflow.

## Verification

After every step: `uv run ruff format . && uv run ruff check . && uv run mypy src && uv run
pytest -n auto` green. The pre-existing `analyze_refactor` proofs (pagination independence,
cursor binding, one table fetch, one classification pass, stale suppression) pass unchanged
after the Step 2 extraction. Byte fidelity is proven end-to-end by applying the emitted
patch with real `git` in a `git init`-ed copy of the fixture repo and comparing the renamed
sources, plus renderer tests for CRLF, BOM, non-ASCII identifiers, and missing final
newlines.

## Later phases (out of scope here)

- Signature-change edit scripts with per-language argument synthesis.
- Patch emission for C, C++, and Lua renames as their structural support lands (the eight
  current `STRUCTURAL_LANGUAGES` work on day one; emission is language-agnostic).
- Composability with the transitive impact radius: one patch covering the deterministic
  subset of a multi-hop rename cascade.
