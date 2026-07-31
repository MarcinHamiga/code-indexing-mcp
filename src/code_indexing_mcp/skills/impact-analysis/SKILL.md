---
name: impact-analysis
description: Map the blast radius of changing or removing a symbol, module, or API using code-indexing-mcp (find_symbol for definitions and call sites, search_code for indirect usages) before any edit
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

## 3. Map direct usages

- `mcp__code-indexing-mcp__find_symbol` — collect all references and call sites of the resolved symbol.
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
