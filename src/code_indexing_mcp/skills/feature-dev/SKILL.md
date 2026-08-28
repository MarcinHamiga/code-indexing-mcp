---
name: feature-dev
description: Use when the user asks to implement a new feature, enhancement, or behavior change in a codebase and the code-indexing-mcp tools are available
type: prompt
whenToUse: When the user asks for a feature to be designed and implemented, and the code-indexing-mcp tools are available
arguments:
  - request
---

You are running a structured feature-development workflow. Follow the phases in order. Do not skip or merge phases.

The feature request is `$request`. If it is missing or too vague to plan against (no clear goal or acceptance signal), ask the user with the AskUserQuestion tool before starting Phase 1.

## Phase 1 — Gather current state with code-indexing-mcp

The core rule of this phase: **the code-indexing-mcp tools are the primary navigation mechanism. Do not fall back to Grep/Glob/Read for code exploration while they are available.**

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files the feature will touch, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.
3. Explore the relevant parts of the codebase:
   - `mcp__code-indexing-mcp__search_code` — issue several targeted semantic queries derived from the feature request (the area being changed, the patterns it must follow, the tests that cover it). Not one broad query.
   - `mcp__code-indexing-mcp__file_outline` — get a file's structure before reading it; decide which parts matter instead of dumping whole files.
   - `mcp__code-indexing-mcp__find_symbol` — locate the declarations the feature will touch. It matches declaration names only; use `find_references` for call sites.
   - When changing an existing declaration, `mcp__code-indexing-mcp__find_references` — identify structural uses from its selected `chunk_id` or project/path/qualified-symbol tuple. For a rename or signature change, use `mcp__code-indexing-mcp__analyze_refactor` and retain its `likely_change`, `review`, and limitation findings in the plan. Both tools cover only C#, Go, Java, JavaScript, Python, Rust, TSX, and TypeScript declarations — selecting one in any other language returns `UNSUPPORTED_LANGUAGE`, not an empty result, so fall back to `search_code`/grep for that part of the codebase instead of treating silence as "no usages". Check `completeness.state` (`complete`, `complete_with_dynamic_limitations`, or `incomplete`) and the `limitations` list before trusting a finding set as exhaustive.
   - `mcp__code-indexing-mcp__get_chunk` — pull the full body of a chunk when you need exact code.
4. Only use Grep when the task genuinely needs a literal match semantic search cannot express (exact string literals, config keys, TODO/FIXME markers) or when the MCP tools are unavailable — and say so when you do.

End the phase with a short **current-state summary** for the user: the modules/files involved, existing patterns and conventions the feature must follow, extension points, and how that area is tested.

## Phase 2 — Prepare the implementation plan

Ground every step in what Phase 1 actually found — real file paths, real symbols, real test conventions. The plan must state:

- **Changes per file** — which files are created or edited, and what changes in each (new symbols, modified functions, wired-in call sites).
- **Design decisions** — any place you deviate from an existing convention, with the reason.
- **Tests** — which test files, which cases, following the project's existing test patterns.
- **Verification** — the exact commands to run (tests, lint, typecheck) and any manual smoke test.

Keep the plan minimal: the smallest change that delivers the request, no speculative extras.

## Phase 3 — Wait for explicit approval

Present the current-state summary and the plan, then STOP and wait for the user's explicit approval (via AskUserQuestion, or ExitPlanMode if in plan mode).

**No file edits, no implementation, no "starting with the obvious part" before approval.** If the user asks for changes, revise the plan and wait again.

## Phase 4 — Implement and verify

Only after approval:

1. Implement the plan as approved, matching the codebase conventions found in Phase 1.
2. Run the verification commands from the plan and look at the results.
3. Report what was changed (`path:line` references) and the verification outcome. If verification fails, fix it before claiming completion — do not declare done with red tests.

## Red flags — STOP and return to the right phase

- Reading files with Read/Grep as the main exploration while the MCP tools are available
- Writing a plan from memory or convention instead of Phase 1 findings
- Editing any file before the user has approved the plan
- Treating silence, "sounds good so far", or a clarifying question as approval
- Claiming completion without running the verification commands
