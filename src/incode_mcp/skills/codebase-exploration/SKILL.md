---
name: codebase-exploration
description: Explore and navigate a codebase by narrowing down with the code-indexing-mcp index tools (search_code, find_symbol, file_outline, get_chunk) before reading any file
type: prompt
whenToUse: When the user asks where something is, how something works, or wants to understand or navigate code, and the code-indexing-mcp tools are available
arguments:
  - question
---

You are exploring a codebase to answer: `$question`. If the question is missing or too vague to search against, ask the user with the AskUserQuestion tool before starting.

The core rule of this skill: **narrow down with the index first; read files last.** Do not fall back to Grep/Glob/Read for code exploration while the code-indexing-mcp tools are available.

## 1. Ensure the index is ready

1. Call `mcp__code-indexing-mcp__project_status` for the target project. If the project is unknown, call `mcp__code-indexing-mcp__list_projects` and, if needed, `mcp__code-indexing-mcp__init_project`.
2. If the index is missing or stale relative to the files in question, call `mcp__code-indexing-mcp__index_project` first. Tell the user if indexing will take a moment.

## 2. Narrow down semantically

- `mcp__code-indexing-mcp__search_code` — issue several targeted natural-language queries derived from the question (the concept, the mechanism, the data it touches). Not one broad query.
- `mcp__code-indexing-mcp__find_symbol` — once you have a concrete function/class/module name, jump straight to its definitions and references instead of more searching.

Keep a shortlist of candidate files and symbols; discard paths the results rule out.

## 3. Outline before reading

- `mcp__code-indexing-mcp__file_outline` — get each candidate file's structure and decide which symbols matter. Never dump a whole file just to find one thing.

## 4. Read only what matters

- `mcp__code-indexing-mcp__get_chunk` — pull the full body of the specific chunks that answer the question.
- Use Read/Grep only for content the index does not cover (non-code files, unindexed paths, exact literal matches semantic search cannot express) — and say so when you do.

## 5. Answer

Answer the question with `path:line` references for every claim, the shortest explanation that is complete, and an explicit note of anything the index could not cover so the user knows what was not verified.
