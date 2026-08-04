---
name: cross-repo-debugging
description: Use when debugging a failure whose producer, contract, transport, generated client, API, or consumer spans two or more repositories registered with code-indexing-mcp
---

# Cross-Repo Debugging

## Overview

Trace failures as evidence chains across explicit repository boundaries. Rankings, copied errors, and matching symbols are not root-cause proof.

## Workflow

1. Call `mcp__code-indexing-mcp__list_projects`. Select only related projects by ID or unambiguous path; never substitute `search_code(all_projects=true)` for deliberate scope.
2. Write the expected chain, such as `producer -> contract -> consumer -> persistence`. Treat each boundary as a first-divergence hypothesis.
3. Check `mcp__code-indexing-mcp__project_status` when freshness is uncertain. In manual mode, refresh stale indexes with `mcp__code-indexing-mcp__index_project` before trusting missing results.
4. Call `mcp__code-indexing-mcp__search_across_projects` with explicit selectors. Query separately for stable identifiers, producer serialization, consumer validation, and contract versions, aliases, artifacts, or configuration.
5. If one repository dominates the global ranking, do not infer absence elsewhere. Filter by `paths`, `languages`, or `kinds`, or search overlapping boundary pairs or smaller shards.
6. Pivot concepts into exact identifiers. Use project-scoped `mcp__code-indexing-mcp__find_symbol` for declarations and semantic search for usages; `find_symbol` does not resolve call sites.
7. Call `mcp__code-indexing-mcp__file_outline` before `mcp__code-indexing-mcp__get_chunk`. Fetch only the producer, boundary contract/configuration, and consumer code needed to compare names, casing, fields, versions, routes/topics, status handling, defaults, and environment keys.
8. Record evidence as `project/path:line`. Stop when it proves the first broken boundary and why it breaks, or when static analysis narrows the failure to one boundary and names the missing runtime evidence.

Use dynamic request and correlation IDs in logs or traces, not as primary source queries. When static code aligns, request the minimum payload, route/topic, deployment version, configuration, and outcome evidence for the narrowed boundary.

## Quick Reference

| Need | Action |
| --- | --- |
| Discover scope | `list_projects`, then retain explicit IDs/paths |
| Correlate code | `search_across_projects` with focused queries |
| Ranking hides a repo | Filter or split into boundary pairs/shards |
| Inspect evidence | `find_symbol`; `file_outline`; `get_chunk` |
| Prove cause | Compare both sides of the first broken boundary |

## Example

For a TypeScript client sending `customer_id` to an API expecting `customerId`, search the client, API, and schema generator. Inspect the call site, serializer/version, naming rule, and aliases. Fix only the proven divergence.

## Common Mistakes and Red Flags

| Mistake | Correction |
| --- | --- |
| “Search every project; it is faster.” | Select related repositories; unrelated hits obscure boundaries. |
| “The top hit is the cause.” | Classify construction, translation, logging copies, and consumers separately. |
| “No hit means no code.” | Check freshness and ranking concentration before concluding absence. |
| “A matching request ID should be in source.” | Correlate dynamic IDs in runtime evidence. |
| “The contracts look aligned, so guess a fix.” | Stop with the narrowed boundary and name the missing runtime evidence. |
