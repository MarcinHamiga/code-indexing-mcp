# Feature Backlog

> Origin: brainstorming session on 2026-08-27. Ideas 1–3 from that session moved into
> active design docs (linked below); everything else recorded here for future work.
> When an item starts, move it to its own dated design/plan doc and delete the entry.

## In flight

- **Structural references for more languages** (Go, Rust, Java, C#) —
  see `2026-08-27-structural-references-more-languages-design.md`.
- **Transitive impact radius** (N-hop closure over the reference graph) —
  see `2026-08-27-transitive-impact-radius-design.md`.
- **Search by example** (paste a snippet, get structurally similar code) —
  see `2026-09-04-search-by-example-plan.md`.

## Backlog

### Field-scoped query syntax

`lang:go path:src/ "exact phrase"` inside `search_code` queries, instead of separate
filter parameters. Parsed client-side into the existing `languages`/`paths` filters, so
the index scan pushdown (`docs/plans/2026-07-27-search-paths-pushdown.md`) is reused
unchanged. Quoting rules and conflicts with explicit filter parameters need a decision:
error on conflict, or let the query string win.

### Similar-chunk clustering

Near-duplicate detection across a project to surface refactor opportunities (copy-paste
variants, parallel implementations). Can run offline against stored vectors; the open
question is presentation (a report tool vs enriching `search_code` results with a
`duplicate_of` field) and threshold tuning.

### Ranking explanations

Per-hit breakdown of why it ranked: term-match contribution vs vector-similarity
contribution vs boost. Improves trust and debuggability of hybrid search. Requires the
 scorer to return its components alongside the final score; purely additive to the
 response schema.

### `changed_symbols` tool

Symbols touched since a commit or timestamp: intersect the scanner's changed-path
validation with the chunk table and report the declarations in those files. Natural
pairing with review workflows and the index-freshness monitoring work. Cheap first
version: files → outlines of changed files. Richer version maps byte ranges to specific
symbols via the structural table.

### Blame/staleness enrichment

Last-touched date per chunk from `git blame`, enabling "only code older than N months"
filters and staleness badges on search hits. Cost model needs care: blame is expensive
per line and must be refreshed lazily per chunk, never during indexing. Storage is a
per-chunk annotation table keyed like the structural table.

### PR-branch comparison slots

Index a PR branch into its existing branch slot and diff search results against the
main slot: new/removed declarations, changed call sites. Mostly a presentation layer
over per-branch slots (`docs/plans/2026-08-15-intelligent-git-branch-index-management.md`);
the hard part is a meaningful diff of two chunk sets and keeping both slots warm within
LRU retention.

### Chunk-level embedding cache

Edits currently re-embed whole files. Content-hash each chunk so an edited function
re-embeds alone and untouched chunks reuse stored vectors. Requires stable chunk
identity across parses (path + qualified name + content hash) and care when token
windowing shifts boundaries. Large indexing-speed win for small edits; interacts with
the probe cache and memory ceiling bookkeeping.

### File-watcher mode

Daemon proactively refreshes on filesystem events so the first query after edits never
waits. Needs platform watchers (FSEvents/inotify), debouncing, and coordination with
lazy/eager index modes and the scheduler. Watch out for editor temp-file churn and
network-filesystem false positives; a conservative fallback poll may still be needed.

### Cross-project reference tracing

`find_references` spanning registered projects (service + generated client, protocol
library + consumers), building on `search_across_projects`' explicit multi-project
scope. Requires a cross-project symbol catalogue and import-edge inference across repo
boundaries; the refactoring-reference-index design explicitly deferred this to a later
module-move phase, and this item is that phase's query-side counterpart.
