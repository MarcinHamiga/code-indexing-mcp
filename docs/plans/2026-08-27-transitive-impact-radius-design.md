# Transitive Impact Radius Design

## Context

`find_references` resolves direct references to one declaration. An agent preparing a
breaking change — a signature change, a semantic change to a widely-called function, a
trait or interface modification — needs the closure: which declarations depend on the
targets, which depend on those, and how deep the blast radius reaches. Today the agent
would have to loop `find_references` itself, one call per declaration, with no shared
snapshot, no cycle handling, and inconsistent treatment of ambiguous hits.

The structural reference index already holds every edge of that graph as
`ReferenceOccurrence` rows, and `ReferenceService._find_references_with_records` already
resolves one node's edges against a pinned table snapshot. The transitive impact radius
is an iteration over that existing machinery, not a new index.

## Goals

- One tool call that returns the N-hop dependents of a selected declaration.
- Layered output by hop distance, with reference kinds on every edge.
- Exact edges traversed by default; `likely` edges opt-in and clearly marked.
- Cycles deduplicated with the first discovery depth kept.
- The whole traversal pinned to one immutable structural snapshot and one slot/activation
  epoch, with the same stale-cursor semantics as `find_references`.
- Bounded cost with explicit truncation reporting, never silent.

## Non-goals

- A runtime call graph, control flow, or dynamic-dispatch resolution.
- Persisting a materialized graph; edges are derived at query time from occurrences.
- Cross-project traversal. This shares the deferred scope of module moves and
  cross-project reference tracing (`2026-08-27-feature-backlog.md`).
- "Depends-on" (reverse) direction, library-style "who do I depend on" queries, and
  diffing two radii. These are thin variants to consider once the core traversal exists.

## Considered approaches

### Query-time BFS over the resolver — selected

Start from the selected declaration, resolve its references with the existing
`_find_references_with_records`, and enqueue each uniquely-identified referencing
declaration for the next hop. Reuse `_select` for enqueueing so a hop target is resolved
with the same uniqueness rules as a user-supplied selector.

Costs no write-path changes, no schema, and no migration. The trade-off is repeated
resolution work per traversal, bounded by the node budget below and amortized by the
store's existing structural-table caching.

### Materialized edge table

Persist resolved declaration-to-declaration edges at index time. Traversal becomes a
cheap table scan, but edges duplicate occurrence data, must be rewritten whenever
resolution rules improve, and add staging complexity for zero benefit at the depths and
node counts a refactoring agent actually inspects. Rejected for now; worth revisiting if
traversals routinely hit the budget.

### Client-driven looping

The MCP client pages `find_references` per declaration. No server work at all, but every
client re-implements cycle detection and snapshot pinning incorrectly, and each page
re-pays classification. Rejected as the primary interface; it remains the fallback for
paging huge results.

## Architecture

### Tool surface

`impact_radius(selector, max_depth=2, include_likely=false, kinds=None, max_nodes=500)` —
a new MCP tool alongside `find_references` and `analyze_refactor`, registered with the
same registering-read annotation (its first call can backfill/refresh the index).

- `max_depth` (1–10, default 2): hops to traverse.
- `include_likely` (default false): also traverse `likely` edges, marked as `possible` in
  the output.
- `kinds`: same reference-kind filter as `find_references`, applied per hop.
- `max_nodes` (default 500, hard cap 2000): total visited-declaration budget.

### Traversal

1. Resolve the entry selector via `_select` under the partition resolved for the project
   (same active-slot rules as `find_references`).
2. For each depth `d` from 1 to `max_depth`: for every node enqueued at `d-1`, run
   `_find_references_with_records` against the pinned snapshot (the same
   `list_reference_records` version, so no hop can observe a mid-traversal refresh).
3. Classify each hit with the existing `_classify`. An `exact` hit whose referencing
   declaration can be uniquely selected becomes an edge; the target node is enqueued
   unless already visited (a revisited node is recorded as a `cycle` edge to its original
   depth and not re-enqueued).
4. A hit that cannot be uniquely attributed to a declaration (member access on an
   unresolved receiver, dynamic import, glob re-export) does not become an edge; it is
   reported in the layer as `review`, matching `analyze_refactor`'s vocabulary.
5. With `include_likely`, `likely` hits become `possible` edges and are traversed, but
   every node downstream of one carries `tainted: true` in the output.

### Response

```
impact_radius(selector, max_depth, include_likely, kinds, max_nodes, limit, cursor)
```

Result fields: `layers` (one per depth, each with `edges` — target declaration, kinds,
`possible`/`tainted` flags — and `review` items), `visited` node count,
`budget_exhausted: bool` (with the depth and node at which the budget cut the traversal),
`completeness` reusing the existing states — `complete` only when every traversed edge
was `exact` and coverage had no gaps, `complete_with_dynamic_limitations` otherwise —
plus the same `limitations` list `find_references` reports. Traversal repeats the
resolver's `must_change`-style exactness discipline: nothing downstream of an ambiguous
edge is claimed as safe.

Pagination: a single response carries up to `limit` edges (default 100, cap 500) with the
same opaque-cursor paging as `find_references`, the cursor binding `max_depth`,
`include_likely`, `kinds`, `max_nodes`, the snapshot version, and the slot/activation
epoch so a later page cannot silently change the traversal's meaning.

### Cost bounds

Per-hop resolution is the dominant cost: one classification pass per visited node. The
`max_nodes` budget caps it; when the frontier at the next hop would exceed the budget,
the tool stops, sets `budget_exhausted`, and reports the unvisited frontier size so the
caller can restart from a deeper selector or raise the budget explicitly.

## Testing strategy

- **Graph shapes**: chains, diamonds (two paths to one node), self-cycles, and mutual
  cycles assert deduplication and first-discovery depths.
- **Classification**: `likely` edges excluded by default, traversed and `tainted` when
  included; unattributable hits land in `review`.
- **Budgets**: exhaustion reports depth and frontier size; the hard cap refuses rather
  than loops.
- **Snapshot pinning**: a refresh between pages yields `STALE_CURSOR`, not mixed-depth
  results; a branch switch invalidates like `find_references`.
- **Coverage**: an `unsupported_language` or `parse_error` file in the radius degrades
  `completeness` exactly as `find_references` does.
- **Contracts**: tool description documents layering, tainting, and budget semantics;
  `completeness.state` guidance mirrors `analyze_refactor`'s.

## Delivery sequence

1. Internal traversal over `_find_references_with_records` with visited/budget logic and
   unit tests on the resolver corpus.
2. `impact_radius` MCP tool, response models, paging cursor, and contract tests.
3. `include_likely` tainting and `review` attribution polish, driven by fixture corpora
   in Python and TypeScript first, the new structural languages as they land
   (`2026-08-27-structural-references-more-languages-design.md`).

## Later phases

- Reverse direction ("who do I depend on") reusing the same traversal with edges
  inverted.
- Radius diffing against a stored baseline — "dependents added since last week" —
  building on index history.
- Cross-project radius once the cross-project symbol catalogue exists.
- Persisted edge cache if real-world traversals routinely exhaust the budget.
