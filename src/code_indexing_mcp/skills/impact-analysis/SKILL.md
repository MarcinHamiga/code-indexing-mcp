---
name: impact-analysis
description: Map the blast radius of changing a symbol or API using code-indexing-mcp structural references plus semantic search before any edit
type: prompt
whenToUse: When the user plans a rename, refactor, signature change, move, or removal and wants to know what it affects, and the code-indexing-mcp tools are available
arguments:
  - target
---

You are analyzing the impact of changing: `$target` (a symbol, module, or API plus the intended change). If the target or the intended change is unclear, ask the user with the AskUserQuestion tool before starting.

The core rule of this skill: **map usages with the index tools, not with grep.**

## 1. Ensure the index is ready

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files the change touches, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.

## 2. Resolve the target

- `mcp__code-indexing-mcp__find_symbol` with an exact match on the target name to locate every definition. If several unrelated symbols match, ask the user which one is meant — do not guess.
- `mcp__code-indexing-mcp__get_chunk` — read the definition itself so the analysis is grounded in what the symbol actually does and exposes.

## 3. Map direct usages and refactor impact

- `mcp__code-indexing-mcp__find_references` and `mcp__code-indexing-mcp__analyze_refactor` only understand Python, JavaScript, TypeScript, and TSX declarations; selecting a declaration in any other language returns `UNSUPPORTED_LANGUAGE` rather than an empty (and misleadingly clean) result. If the target lives in another language, say so and fall back to `search_code`/grep for that part of the analysis.
- `mcp__code-indexing-mcp__find_references` — pass the selected declaration's `chunk_id`, or its project/path/qualified-symbol tuple, to retrieve structural uses. Treat `exact` as binding evidence, `likely` as a required review, and `unresolved` plus limitations as blind spots; do not promote them to exact yourself.
- `mcp__code-indexing-mcp__analyze_refactor` — for a rename or signature proposal, use its discriminated `operation` input before planning edits. `must_change` is deterministic, `likely_change` and `review` need human inspection, and `evidence` can show aliases that bind the target but need no spelling edit (or, for a signature change, compatible call sites).
- Read `completeness.state` before you characterise the blast radius. Only `complete` means every indexed file was analyzed; `complete_with_dynamic_limitations` means everything was analyzed but some findings rest on conservative, non-structural evidence; `incomplete` means whole files were not analyzed at all. The named `limitations` (other languages, parse failures, stale files) name the gap in either non-`complete` case. Report that gap to the user instead of presenting the finding list as exhaustive.
- If you apply the edits, use each finding's `edit_start_byte`/`edit_end_byte`, which cover just the identifier. The wider `start_byte`/`end_byte` span the whole reference, so replacing that range turns `auth.authorize(u)` into `permit(u)` and drops the alias from `import authorize as check`. Null edit offsets mean edit that site by hand.
- For each distinct file in the results, `mcp__code-indexing-mcp__file_outline` to place the usage in context, and `mcp__code-indexing-mcp__get_chunk` where the exact call matters (signature changes, argument reordering).

## 4. Hunt indirect usages

Semantic search finds what name-based lookup misses:

- `mcp__code-indexing-mcp__search_code` — targeted queries for re-exports and wrappers around the symbol, duck-typed or dynamic call sites, serialization or config that names it, and tests that exercise it indirectly.
- Grep is allowed only for literal string occurrences of the exact name (reflection, registries, config keys) — say so when you use it.

## 5. Report

Produce a structured impact report:

- **Target and intended change**, with `path:line` of the definition.
- **Must change** — call sites that break, each with `path:line` and why.
- **Should review** — indirect or dynamic usages that may break silently.
- **Tests** — the test files covering the target, found via the index.
- **Change checklist** — the ordered list of edits the change requires, grounded in the findings above.

Report only what you actually inspected through the tools; list blind spots the index could not cover.
