# Search by Example — Implementation Plan

**Goal:** A read-side tool that takes a pasted code snippet, chunks it with the real
extractor so its declaration metadata shapes the embedding, and returns the indexed
chunks most similar to it — the "where else do we do this" query, single-project by
default and across explicitly selected projects through `search_across_projects`.

**Origin:** the Search-by-example entry in `docs/plans/2026-08-27-feature-backlog.md`
(brainstorm 2026-08-27). Per that doc's convention the entry is deleted now that work
started; this plan is its design and implementation record.

All code coordinates below were verified against the current tree on 2026-09-04.

**Baseline:** `uv run ruff format --check . && uv run ruff check . && uv run mypy src &&
uv run pytest -n auto` green before Step 0, and after every step.

## Decisions settled before implementation

- **D1 — New tool plus an `example` mode on `search_across_projects`; no `search_code`
  overload.** `search_code`'s `query` is contractually natural language or keywords, and
  its hybrid scorer feeds the whole string to the FTS side (`MultiMatchQuery` with
  `FullTextOperator.OR`, `src/code_indexing_mcp/storage.py:1855-1865`) — a pasted
  snippet there would rank by term soup, not similarity. One service method backs two
  tool entry points: `search_by_example` (scope semantics identical to `search_code`:
  `projects`/`all_projects`, default active root) and `search_across_projects(example=...)`
  (mutually exclusive with `query`, still requiring ≥2 distinct resolved projects).
- **D2 — The snippet is chunked by the production extractor under a pseudo-path.**
  `TreeSitterExtractor.extract(path, language, source)`
  (`src/code_indexing_mcp/extractor.py:279`) is called with `Path("example" + suffix)`
  for the resolved language, so every extracted chunk carries the exact prefix indexing
  would have produced — `language: …`, `path: …`, `kind: …`, `symbol: …` — via
  `_make_chunk` (`extractor.py:2936-2973`). That prefix is the backlog's "declaration
  metadata weights the match": it is part of the embedded passage, not a separate boost.
  The pseudo-path is a constant; it never reaches storage or responses.
- **D3 — Language resolution: explicit hint, then parse-attempt detection, then whole-
  snippet fallback.** An explicit `language` wins. Without one, detection runs `extract`
  over a fixed-priority candidate list and accepts the first language whose result has
  `has_errors == False` and at least one non-`module` definition chunk (Tree-sitter
  parses a snippet in microseconds; 18 candidates bound the cost). If nothing qualifies,
  the snippet is embedded whole as a single prefix-less passage and `language` is
  reported as `null` — an honest degradation, not an error.
- **D4 — Passage-role embedding, no query role, no re-windowing.** Stored vectors were
  produced by `embed_passages` over `compose_passage(prefix, content)`
  (`src/code_indexing_mcp/embedding.py:91`, `indexing.py:1610`), so the snippet must be
  embedded through the same role and composition or it lands in the wrong region of the
  space; `embed_query` is a different role and stays wrong here. Token windowing is not
  reimplemented query-side: the 16 KiB size cap (D7) fits the model's context, and the
  extractor's own `_chunks_for_range` splitting already produces `_part` chunks for
  oversized definitions. This also keeps the `Embedder` protocol
  (`QueryEmbedder + PassageEmbedder`, `embedding.py:293`) as the only requirement —
  every test double in the suite implements exactly those two methods.
- **D5 — Vector-only search per snippet segment, merged by best distance.** For each
  snippet chunk's vector, run a vector-only LanceDB query (`query_type="vector"`,
  same `vector_column_name="vector"`, same select list, same `ef`/`refine_factor`/
  `bypass_vector_index` handling as `_hybrid_search_rows`, `storage.py:1887-1893`) —
  no `.text()` side. Rows are merged across segments by minimum `_distance` per
  `chunk_id`; score = `1.0 - _distance` (cosine distance → similarity). The partition
  fan-out, concurrency, and request-order tie-breaking mirror `hybrid_search`
  (`storage.py:1764-1826`); implementation extracts the shared task-building/executor
  body into one private helper both methods call. No language condition is imposed from
  the resolved snippet language — a Python snippet may legitimately match its ported
  Java twin; the caller keeps the existing `languages` filter for that.
- **D6 — Post-processing reuses `search_code`'s path.** Pushed-down path conditions via
  `path_condition`, the `_FALLBACK_FETCH_ROWS` widening when paths cannot push
  (`src/code_indexing_mcp/search.py:66-85`), Python-side path filter, dedup on
  `(project_id, path, start_line, end_line)`, sort by `(-score, path, start_line)`,
  `_hit` for response shaping. Response is a new frozen model
  `ExampleSearchResponse(language: str | None, segments: int, hits: list[SearchHit])`
  next to `SearchResponse` (`src/code_indexing_mcp/models.py:839`); the snippet is
  never echoed back and never logged (log sizes and language only).
- **D7 — Limits.** Example capped at 16,384 characters (`INVALID_FILTER` naming the
  cap); `limit` clamped 1–50 exactly like `search_code`; an explicit unknown language
  is rejected by the `LanguageName` literal schema (`UNSUPPORTED_LANGUAGE` for
  non-schema callers through the daemon path).
- **D8 — Normalization deferred, explicitly.** The backlog's open question — embed the
  snippet as-is or normalized first — is answered "as-is" for this plan: the only
  normalization is the extractor's existing `utf-8-sig` handling (`extractor.py:281`).
  Comment stripping or identifier renaming would break the symmetric-embedding
  assumption (indexed chunks are unnormalized) and is recorded as a non-goal; revisit
  only with measured evidence that raw snippets retrieve poorly.

## Mechanics discovered during research (accounted for below)

- The query-side `Application` already owns an extractor: `self.indexer` is built with
  `extractor=TreeSitterExtractor()` (`src/code_indexing_mcp/application.py:440`), and
  `self.search = SearchService(self.store, embedder)` at `application.py:462` — the
  service needs the extractor handed to it (new optional constructor parameter).
- `_make_chunk` already computes everything the query side needs per chunk:
  `embedding_prefix`, `content`, and windowing-ready shape; no extractor change.
- `SearchService.search_code` (`search.py:43-131`) is the template for scope
  validation, partition mapping checks, and `_in_condition` filters; `find_symbol`'s
  partition-ref validation (`search.py:143-148`) shows the single-partition variant.
- `Application.search_code` (`application.py:1367-1400`) is the template for
  `_scope_checkouts` → `_run_repository_stable_query` with `_ensure_query_generations`
  (`application.py:816-847`) — the new tool must run under the same repository-stable
  retry and freshness machinery, not beside it.
- Daemon mode must be wired or the tool breaks when served over the daemon: the
  dispatcher (`src/code_indexing_mcp/daemon.py:630-696`) routes by method name
  (`search_code` at `:664`) and `DaemonClient` wraps each call
  (`daemon.py:1007-1008`); both need `search_by_example` entries.
- The server funnels `search_code` and `search_across_projects` through one helper,
  `search_resolved_projects` (`src/code_indexing_mcp/server.py:824-849`); the example
  variant extends or siblings that helper rather than duplicating the wait/resolve/to_thread
  scaffolding.
- Documentation gates that enumerate tools: the server instruction string
  (`server.py:63-69`), `README.md` tools table (`README.md:289-290` region), and
  `tests/test_server.py:138-139` asserts the tool-name list.

## Steps

**Step 0 — Baseline.** Run the four gate commands; confirm green before touching code.

**Step 1 — Models and backlog.** Add `ExampleSearchResponse` to `models.py` next to
`SearchResponse`. Delete the Search-by-example entry from
`docs/plans/2026-08-27-feature-backlog.md` and add an "In flight" line pointing here.
Unit-test the response model round-trips with `language=None` and `segments=0`.

**Step 2 — Snippet preparation in `search.py`.** Module-level `_EXAMPLE_SUFFIX`
mapping every `LanguageName` to one representative suffix, the ordered
`_DETECTION_ORDER` tuple, and two helpers: `detect_example_language(extractor, source)
-> str | None` (D3) and `_example_passages(extractor, example, language) -> tuple[
str | None, list[str]]` returning the resolved language and the composed passage texts
(D2, D3 fallback, D4). `SearchService.__init__` gains `extractor: TreeSitterExtractor
| None = None` (default keeps every existing constructor call working). Tests
(`tests/test_search.py`): a Python snippet yields a passage starting with the
`language: python` prefix and containing the function body; detection picks python for
a `def` snippet and typescript for an `interface` snippet; prose falls back to
`None` with one prefix-less passage; every suffix entry parses via the extractor;
oversized and empty examples raise `INVALID_FILTER`.

**Step 3 — Store: `LanceStore.example_search`.** Signature
`example_search(vectors, project_ids, condition, limit, *, partition_ids=None)`
mirroring `hybrid_search`; extract the shared partition fan-out into a private helper
both methods call; per (partition, vector) vector-only query per D5; merge by best
distance, inject `project_id` per row, sort by score desc with request-order
tie-breaks. Tests (`tests/test_storage.py`, following its existing
index-then-search fixtures): two snippet vectors where the nearer one wins the shared
chunk; per-segment limits fan in and merge; condition prefilter (`language IN (…)`)
applies; pinned partitions and multi-project scoping behave as in `hybrid_search`.

**Step 4 — `SearchService.search_by_example`.** Full parameter set of `search_code`
plus `language: str | None`, returning `ExampleSearchResponse`; validations, filters,
fallback fetch widening, dedup, and sort per D6–D7. Tests (`tests/test_search.py`,
using the existing `SemanticEmbedder` convention whose `embed_passages`/`embed_query`
make snippet and chunk vectors comparable): pasting a function from an indexed file
ranks that chunk first; a structurally similar-but-renamed function outranks unrelated
code; `kinds`/`languages`/`paths` filters compose; `limit` clamps; partitions map
validation errors match `search_code`'s.

**Step 5 — Application wiring.** `Application.search_by_example` mirroring
`Application.search_code` (`application.py:1367-1400`) with `_scope_checkouts`,
`_ensure_query_generations`, `_run_repository_stable_query`; pass
`self.indexer.extractor` to `SearchService` at `application.py:462`; extend the
`ApplicationLike` protocol (`application.py:223-234` region). Tests
(`tests/test_application.py`): scope resolution defaults to the active root,
`all_projects` expands, and the repository-stable retry path is exercised the same way
existing search tests do.

**Step 6 — Daemon and server surface.** Dispatcher entry plus `DaemonClient` wrapper
(`daemon.py`, mirroring `search_code` at `:664`/`:1007`). Server: new
`search_by_example` tool (title, description, `_READS_AND_REGISTERS`,
`_with_error_details`) with `example`, optional `language: LanguageName | None`, and
`search_code`'s scope/filter/limit parameters; `search_across_projects` gains
`example: str | None = None`, with `query` becoming optional and exactly one of the
two required (`INVALID_FILTER` otherwise), keeping the ≥2-distinct-projects rule
(`server.py:1300-1365`); extend the shared resolved-projects helper. Update the
instruction string (`server.py:63-69`) and the README tools table. Tests
(`tests/test_daemon.py`, `tests/test_server.py`): the tool-name list assertion; happy
path over an indexed tmp project; mutual-exclusion and size-cap errors surface with
their `ErrorCode` details; cross-project example search returns one globally ranked
list.

**Step 7 — Polish and shipped note.** Full gate run; write
`2026-09-04-search-by-example-shipped.md` recording the decisions that changed during
implementation, the measured latency of a warm `search_by_example` call next to
`search_code`'s (the extra cost is one extraction plus one passage-embedding batch),
and the deferred-normalization verdict from D8 as the follow-up question.

## Cross-cutting invariants

1. The snippet is untrusted input: it is parsed, embedded, and discarded — never
   written to storage, never echoed in full, never logged (sizes and language only).
2. Read-only with respect to the index: the tool registers/indexes roots exactly like
   `search_code` (lazy freshness), but adds no tables and no schema changes.
3. No ranking regressions for `search_code`: its hybrid path must be untouched except
   for the shared fan-out helper extraction, verified by the existing
   `tests/test_search.py` suite passing unmodified.
