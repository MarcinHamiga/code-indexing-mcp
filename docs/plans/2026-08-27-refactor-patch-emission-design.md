# Refactor Patch Emission Design

## Context

`analyze_refactor` computes deterministic rename findings — `must_change` items carrying
`edit_start_byte`/`edit_end_byte` offsets that cover exactly the identifier — but the
server "never edits source files". An agent renaming a symbol used at twenty sites must
hand-apply twenty edits from offsets, which is exactly the mechanical, error-prone work
the offsets were meant to prevent.

The missing piece is not application but serialization: a byte-exact unified diff that
the caller can review and apply with `git apply` or their own edit tooling. This design
adds emission only. The server's no-write stance is preserved: producing a patch is
analysis, applying it stays with the client.

## Goals

- Emit a unified diff from the deterministic `must_change` findings of a rename.
- Byte-exact hunks built from the working-tree bytes the resolver already reads
  (`ReferenceService._file_bytes`), not from stored chunk text.
- Freshness verification per touched file: if the bytes at the recorded offsets no longer
  match what the analysis saw, report a conflict and omit the hunk, never emit a patch
  against stale expectations.
- A machine-readable structured edit list beside the diff.
- Determinism: identical inputs produce a byte-identical patch.

## Non-goals

- Applying edits to source files, now or behind a flag.
- Signature-change emission. Synthesizing reordered or defaulted argument lists is
  language-specific and easy to get silently wrong; the analysis stays, the patch does
  not. Revisit with per-language edit-script generators.
- Conflict resolution or three-way merge semantics. A stale file omits its hunks and is
  reported; the caller re-runs after refreshing.
- Rename operations that move or create files (module moves, import rewrites). Import
  lines rename the identifier only.

## Considered approaches

### New `emit_refactor_patch` tool re-deriving the analysis — selected

The tool takes the same selector and a `{"kind":"rename","new_name":...}` operation,
re-runs the same deterministic analysis path `analyze_refactor` uses
(`_find_references_with_records` → `_rename_findings`), and renders the results as a
diff. It carries the same registering-read annotation, so it refreshes a stale index
before analyzing rather than emitting against old data.

Re-deriving keeps `analyze_refactor`'s contract untouched (it stays pure analysis with
no output-format coupling) and avoids persisting findings between calls, which would
need a digest-addressed store and its own staleness story.

### `analyze_refactor` gains an `emit_patch` flag

One call instead of two, but it welds presentation into the analysis tool, makes every
`analyze_refactor` description carry patch caveats, and tempts later callers to skip the
review step `likely_change`/`review` items exist for. Rejected.

### Returning an edit script instead of a diff

Offsets plus replacement text, no unified diff. Sufficient for programmatic clients and
in fact included in the response as the `edits` field — but agents and humans review
diffs, and `git apply` exists everywhere. The diff is the interface; the script is the
fallback.

## Architecture

### Tool surface

`emit_refactor_patch(selector, operation, context_lines=3)` — accepts only the rename
operation; a signature-change operation returns `UNSUPPORTED_OPERATION` naming the
limitation.

### Emission pipeline

1. Resolve the selector and run the deterministic rename analysis against the pinned
   snapshot, reusing the exact code paths `analyze_refactor` uses so the two tools cannot
   disagree about what is `must_change`.
2. Partition findings by file. Keep only `must_change` items with non-null
   `edit_start_byte`/`edit_end_byte`; everything else (`likely_change`, `review`,
   null-offset evidence) accumulates into `unapplied` with its existing reason codes.
3. For each file, read the current bytes through `_file_bytes` (same project-root
   confinement as today). For each finding, verify the byte slice at the recorded
   offsets equals the bytes the analysis matched — the identifier occurrence the
   resolver classified. A mismatch, or a file that cannot be read, marks that finding
   `conflict` and omits it.
4. Sort a file's edits by `edit_start_byte`; overlapping edits (a finding whose span
   intersects the previous one) are `conflict`-omitted too, defensive against resolver
   regressions.
5. Render a unified diff: `a/`/`b/` prefixes with repo-relative paths, configurable
   context (default 3), LF or CRLF preserved from the file bytes, replacement text the
   new identifier encoded as the file is. Header lines and hunk offsets are computed
   with a standard difflib-style implementation over the edited bytes.

### Response

- `patch`: the unified diff text, empty when nothing applied (all findings
  `unapplied`/`conflict`).
- `edits`: the structured list — path, `edit_start_byte`, `edit_end_byte`, old text, new
  text — for programmatic application.
- `applied`/`unapplied`/`conflicted` counts, and per-item detail for the latter two with
  the `analyze_refactor` reason vocabulary.
- `snapshot_version`, slot identity, and the operation digest, so a caller can correlate
  the patch with the `analyze_refactor` run that motivated it.

`completeness` reports `complete` only when every finding in the analysis was
deterministic, applied, and verified — otherwise
`complete_with_dynamic_limitations` with the same `limitations` list, so a partial patch
can never read as a finished rename.

### Safety properties

- No write: the tool reads source bytes and returns text; it never opens a file for
  writing.
- Stale-world protection is per-offset byte equality, which is strictly stronger than
  mtime checks: content that changed and changed back still matches.
- Determinism: the same selector, operation, file bytes, and snapshot produce identical
  output; nothing in the pipeline depends on time, ordering of concurrent reads, or
  locale.

## Testing strategy

- **Byte-exact fixtures**: corpus projects with committed expected diffs, covering
  multi-file renames, adjacent edits in one hunk, rename at file start/end, CRLF files,
  and non-ASCII identifiers (offsets are byte offsets from Tree-sitter).
- **Conflicts**: a file edited after indexing (same length, different content) omits its
  hunks and reports `conflict`; re-running after refresh emits them.
- **Unapplied**: `likely_change`/`review` findings never appear in the patch and always
  appear in `unapplied`.
- **Overlap defense**: synthetic overlapping findings are omitted, not merged.
- **Operation guard**: signature-change selectors return `UNSUPPORTED_OPERATION`.
- **Contracts**: tool description states emission-only semantics and directs application
  to the caller's tooling.

## Delivery sequence

1. Internal patch renderer (bytes → unified diff) with fixture-level unit tests,
   independent of the resolver.
2. `emit_refactor_patch` service method reusing the rename analysis path, freshness
   verification, and conflict handling.
3. MCP tool registration with the registering-read annotation, contract tests, and the
   README refactoring-workflow section update.

## Later phases

- Signature-change edit scripts with per-language argument synthesis.
- Patch emission for Go, Rust, Java, and C# renames as their structural support lands
  (`2026-08-27-structural-references-more-languages-design.md`).
- Composability with the impact radius: emit one patch covering the deterministic subset
  of a multi-hop rename cascade.
