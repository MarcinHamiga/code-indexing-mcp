# Review Follow-Ups: Plan Index

Five independent plans addressing the 2026-07-26 review of `main` (commit `961a7c2`). Each plan
produces working, tested software on its own and can be executed and merged separately.

Baseline at the time of review: **190 passed, 3 skipped** (`.venv/bin/python -m pytest -q`). The
three skips are real-model gates requiring `INCODE_MODEL_TEST_CACHE`.

## The plans

| # | Plan | Fixes | Primary files |
|---|---|---|---|
| 1 | [MCP tool surface](2026-07-27-mcp-tool-surface.md) | Empty tool descriptions, missing annotations, loose schemas, dropped error details | `server.py`, `errors.py` |
| 2 | [Search paths pushdown](2026-07-27-search-paths-pushdown.md) | `paths` filter silently returns zero hits | `path_filter.py` (new), `search.py` |
| 3 | [get_chunk projection](2026-07-27-get-chunk-projection.md) | `get_chunk` returns the 768-dim vector and triplicated text | `models.py`, `storage.py`, `search.py` |
| 4 | [Extractor performance](2026-07-27-extractor-performance.md) | Quadratic in definitions per file; query recompiled per file | `extractor.py` |
| 5 | [Scanner I/O and store cache](2026-07-27-scanner-io-and-store-cache.md) | Double read per changed file, unbounded partition cache, unbounded `list_chunks` | `scanner.py`, `storage.py`, `indexing.py` |

## Recommended order

Plans 2, 4, and 5 touch disjoint files and can run in parallel with anything.

Plans **1 and 3 both edit the `get_chunk` tool** in `server.py`, and plans **1 and 2 both edit
`search_code`** in `search.py`/`server.py`. Run **1 before 3** and **1 before 2** to avoid resolving
the same hunks twice. If they must run in parallel, expect a small rebase in `server.py`.

Suggested sequence, highest value first:

```
1 (tool surface)  ->  2 (paths bug)  ->  3 (get_chunk)  ->  4 (extractor)  ->  5 (I/O + cache)
```

## Global Constraints

These apply to every plan; individual plans do not repeat them.

- **Python** `>=3.12,<3.14`. Target 3.12 syntax — the repo runs 3.12.13 locally.
- **Pinned dependency ranges** must not change: `mcp>=1.27,<2`, `lancedb>=0.25,<1`,
  `pydantic>=2.11,<3`. Verified behaviour below was measured on `mcp` 1.x, `lancedb` 0.34.0,
  `pydantic` 2.13.4.
- **No new runtime dependencies.** Every plan is implementable with what is already in
  `pyproject.toml`.
- **Run the tools the repo already configures** — `ruff`, `mypy`, `pytest` — from `.venv/bin/`.
  `pytest-timeout` is *not* installed; do not pass `--timeout`.
- **Lint and type-check must stay clean.** `.venv/bin/ruff check .`,
  `.venv/bin/ruff format --check .`, and `.venv/bin/mypy src tests` all pass on `main`.
- **Preserve the public error contract.** `ErrorCode` members may be *added*; existing members must
  not be renamed or removed — `daemon.py` maps codes across the socket boundary by value.
- **Comment style:** this codebase explains *why*, not *what*, and the review specifically credited
  that. Match it. Do not add comments restating the code.
- **Commit convention:** `feat:`, `fix:`, `perf:`, `docs:`, `test:` prefixes, as in
  `git log --oneline`.
- **Every plan ends green:** the full suite must pass with no new skips before the final commit.

## Verified facts the plans rely on

Each was measured in this repo rather than assumed. Plans cite these instead of re-deriving them.

- `@mcp.tool()` accepts `name`, `title`, `description`, `annotations`, `icons`, `meta`,
  `structured_output`. `ToolAnnotations` fields are `title`, `readOnlyHint`, `destructiveHint`,
  `idempotentHint`, `openWorldHint`.
- `Annotated[T, Field(description=..., ge=..., le=...)]` and `Literal[...]` render into
  `inputSchema` as `description`, `minimum`/`maximum`, and `enum`. Out-of-range arguments then fail
  with a Pydantic validation error at call time instead of being silently clamped.
- A `functools.wraps` decorator between `@mcp.tool()` and the handler preserves the generated
  schema, the parameter descriptions, and `Context` exclusion.
- LanceDB supports `regexp_match(column, 'pattern')` as a `prefilter=True` predicate on hybrid
  search, combines with `AND`/`OR`, and passes backslashes through string literals unharmed.
  `re.escape` output is accepted as valid Rust regex for `- + ( ) space $ ^ { } \ % _ '`.
- Extraction of 35 repo files: **72.9 ms** baseline, **40.8 ms** with the compiled Tree-sitter
  query cached (44% faster). A 699 KB / 16,384-definition generated file takes **31.3 s** to
  extract against **8 ms** of Tree-sitter parsing.
- A cold index issues exactly **2** `Path.read_bytes` calls per file; a warm re-index issues **0**.

### Algorithms in these plans were run before being written down

The two non-mechanical rewrites were validated against the code they replace, not just reasoned
about:

- **Plan 4's `_content_range` bisect rewrite** was run alongside the current list-comprehension
  version over 36 real files — **572 definitions, 79 of them containers (the branch that changes),
  0 mismatches.**
- **Plan 4's `_LineIndex.line_at`** was compared to `source[:offset].count(b"\n") + 1` at sampled
  offsets across 12 files plus the empty, newline-only, and CRLF edge cases — **0 mismatches.**
  `bisect_left` is the correct bisect here; `bisect_right` would be off by one at an offset that
  lands exactly on a newline.
- **Plan 5's `StoredChunk(IndexedChunk)` split** preserves the field order of the current
  `StoredChunk` exactly, `vector` still last, with `IndexedChunk` at 18 fields matching
  `INDEXED_CHUNK_COLUMNS` one-for-one.
- **Plan 2's glob translation** agrees with `PurePosixPath.match` over 870 paths × 20 patterns with
  0 mismatches in either direction, and its `re.escape` output is accepted by LanceDB's Rust regex
  for every special character tested.

Plan 4 Task 1 still installs a committed output snapshot before any of that lands. The validation
above is why the refactor is expected to pass; the snapshot is what proves it did.
