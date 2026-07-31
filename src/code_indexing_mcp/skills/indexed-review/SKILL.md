---
name: indexed-review
description: Review the codebase from a user-chosen angle (security, logic, UI, UX, performance, architecture, etc.) using code-indexing-mcp semantic code analysis instead of grep-style searching
type: prompt
whenToUse: When the user asks for a code review, audit, or targeted analysis of the codebase and the code-indexing-mcp tools are available
arguments:
  - angle
  - scope
---

You are running a structured, angle-focused code review. Follow this workflow exactly.

## 1. Require a review angle

The review angle is `$angle` (valid examples: security, logic, UI, UX, performance, architecture, API design, error handling, testing).

If no angle was provided (or it is ambiguous), you MUST stop and ask the user with the AskUserQuestion tool before doing any analysis. Offer these options:

- **security** — vulnerabilities, injection, secrets, authn/authz, unsafe input handling
- **logic** — correctness, edge cases, error handling, race conditions, off-by-one
- **UI / UX** — interface structure, consistency, accessibility, user-facing flows
- **performance** — hot paths, allocations, I/O patterns, caching, complexity

Do not guess the angle. A review without a stated angle is not started.

Scope is `$scope` (a directory, module, feature, or symbol set). If scope is empty, ask the user or infer it from their request (e.g. recent changes); state the scope you settled on before reviewing.

## 2. Prepare the index

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files under review, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.

## 3. Analyze with the index, not grep

This is the core rule of this skill: **use the code-indexing-mcp tools as the primary navigation and analysis mechanism. Do not fall back to Grep/Glob for code exploration.**

- `mcp__code-indexing-mcp__search_code` — semantic natural-language queries to find relevant code for the angle (see playbooks below). Issue several targeted queries, not one broad one.
- `mcp__code-indexing-mcp__file_outline` — get a file's structure before reading it; decide which parts matter instead of dumping whole files.
- `mcp__code-indexing-mcp__find_symbol` — trace a function/class to all its definitions and call sites when evaluating impact or correctness.
- `mcp__code-indexing-mcp__get_chunk` — pull the full body of a chunk returned by search when you need exact code.

Only use Grep when the task genuinely needs a literal match that semantic search cannot express (exact string literals, secret-looking patterns, TODO/FIXME markers) or when the code-indexing-mcp tools are unavailable — and say so when you do.

## 4. Angle playbooks

Tune your `search_code` queries and checklist to the chosen angle:

- **security**: query for "user input validation", "SQL or query construction", "subprocess / shell command execution", "authentication and permission checks", "file path handling", "deserialization", "token or password handling". Check: injection sinks, missing authz, hardcoded secrets, unsafe defaults, data exposure in logs/errors.
- **logic**: query for the core decision-making code in scope, "boundary handling", "retry and timeout logic", "concurrency and locking". Check: edge cases (empty, None, overflow), error paths that swallow exceptions, incorrect conditionals, state mutation bugs.
- **UI / UX**: query for "rendering / component structure", "user-facing text and messages", "input and interaction handling", "loading and error states". Check: consistency, accessibility, feedback on failure, dead ends in flows.
- **performance**: query for "loops over collections", "database or network calls", "caching", "serialization". Check: N+1 patterns, repeated work in hot paths, unbounded growth, blocking I/O on critical paths.
- **architecture**: query for "module boundaries", "dependency direction", "shared state". Check: layering violations, god objects, leaky abstractions, coupling.
- **API design**: query for "public endpoints / exported functions", "request and response schemas". Check: naming consistency, versioning, error contract, backward compatibility.

For any other angle, derive 3–5 semantic queries from the angle's intent and proceed the same way.

## 5. Report

Produce a structured review report:

- **Angle and scope** actually reviewed, and index freshness used.
- **Findings**, ordered by severity (critical / major / minor). Each finding: `path:line` reference, the evidence (short code excerpt via `get_chunk`), why it matters for this angle, and a concrete fix suggestion.
- **What looks good** — brief, only genuine positives.
- **Blind spots** — areas the semantic search could not cover confidently, so the user knows what was not verified.

Keep findings honest: report only what you actually inspected through the tools, never speculate about code you did not read.
