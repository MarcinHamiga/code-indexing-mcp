# Search by Example: Shipped (2026-09-04)

## Outcome

`search_by_example` allows searching registered repositories by providing an arbitrary code snippet (e.g. from an error trace, an external repository, or code being refactored) rather than guessing keywords or natural-language phrasing. It parses the snippet using the production `TreeSitterExtractor`, applies language-specific declaration metadata to format passage-symmetric embedding text, embeds chunks via `embed_passages`, searches LanceDB vector-only across all active index partitions, and merges hits across chunks by minimum distance (`score = 1.0 - _distance`).

In addition, `search_across_projects` now supports an `example` parameter (mutually exclusive with `query`) to search for structurally similar code across multiple related projects with global score ranking.

The server's untrusted-input and read-only invariants are maintained:
- Snippets are parsed and embedded in-memory and discarded — never written to disk, indexed, or logged in full (bounded to 16 KiB).
- Search does not alter the LanceDB index.
- Existing `search_code` hybrid ranking remains untouched and regressions-free.

---

## Architecture & Decisions

### 1. New Tool & Dedicated Mode (D1)
`search_by_example` is introduced as a distinct tool alongside `search_code`. Mixing `example` directly into `search_code` was rejected to avoid ambiguous ranking semantics (hybrid vs. vector-only) and schema pollution. `search_across_projects` was extended with `example: str | None = None`, mutually exclusive with `query: str | None = None` (`ErrorCode.INVALID_FILTER` raised if neither or both are provided).

### 2. Passage-Symmetric Embedding Representation (D2)
Because chunks in the index are embedded as passages with declaration prefixes (e.g., `language: python\npath: ...\nclass: ...\ndef: ...\n<code>`), search queries generated from code examples use the exact same template format produced by `_passage_text` rather than query prefixes (`query: ...`). This aligns query and document representations in the embedding space.

### 3. Tree-sitter Segmentation & Fallback (D3)
Input snippets are parsed using `TreeSitterExtractor`. If the user does not provide `language`, `detect_example_language` attempts to parse the snippet against supported languages in a deterministic order. If valid declarations or statements are extracted, each chunk forms a query segment. If extraction yields no chunks (or the language lacks structural grammars), a whole-snippet pseudo-chunk is created with fallback metadata.

### 4. Vector-Only Search & Segment Merging (D4, D5)
Hybrid search is not suitable for raw code snippets because BM25 keyword search over language syntax characters (braces, punctuation, keywords) degrades rank quality. `search_by_example` executes vector-only search per segment vector via LanceDB (`query_type="vector"`). Hits are merged across segments by selecting the minimum distance per chunk, mapped to `score = 1.0 - _distance`, deduplicated, and globally sorted by `(-score, path, start_line)`.

### 5. Shared Partition Fan-out (D6)
Extracted `_fan_out_partitions` helper in `storage.py` shared by both `hybrid_search` and `example_search` to avoid duplicating parallel partition query execution, retry handling, and slot validation.

### 6. Single Response Model for Cross-Project Search (D9 / Adaptation)
When FastMCP encounters a Python union return type (`SearchResponse | ExampleSearchResponse`), it wraps tool output into a nested `{"result": ...}` dictionary, breaking client expectations. To maintain a clean, unwrapped JSON object schema and backward compatibility, `SearchAcrossProjectsResponse(hits: list[SearchHit], query: str | None = None, language: str | None = None, segments: int | None = None)` was introduced.

### 7. Deferred-Normalization Verdict (D8)
Under cosine distance on normalized embeddings (such as production `FastEmbedder`), vector distance ranges between 0.0 and 2.0 (and typically <= 1.0 for positive similarities), resulting in scores in `[0.0, 1.0]`. With synthetic, unnormalized unit test vectors (e.g., `TinyEmbedder`), raw L2 distances may exceed 1.0, producing negative `1.0 - distance` values. Score normalization across segments was intentionally deferred:
- Segment scores must remain comparable across queries within the same embedding space without artificial squashing.
- Preserving raw `1.0 - distance` aligns exactly with how LanceDB surfaces distances and avoids loss of resolution.
- As a follow-up, if future embedders do not guarantee unit L2 normalization, explicit L2 projection before LanceDB queries can be introduced at the embedder boundary rather than in search ranking.

---

## Measured Performance

Benchmarked warm query latency on Apple Silicon across 10 Python modules using production `FastEmbedder`:

| Operation | Warm Latency | Overhead vs `search_code` |
|---|---:|---:|
| `search_code` (hybrid vector + FTS) | 48.05 ms | baseline |
| `search_by_example` (extract + vector) | 52.13 ms | +4.08 ms (+8.5%) |

The ~4.08 ms delta is accounted for by in-memory Tree-sitter parsing of the snippet, metadata extraction, and computing passage-prefixed embeddings.

---

## Delivered Components

1. **Models (`models.py`)**:
   - `ExampleSearchResponse`: `language: str | None`, `segments: int`, `hits: list[SearchHit]`.
   - `SearchAcrossProjectsResponse`: unified unwrapped cross-project search response model.
2. **Search Service (`search.py`)**:
   - `MAX_EXAMPLE_LENGTH = 16_384`.
   - `detect_example_language`: AST error-rate heuristics across candidate grammars.
   - `_example_passages`: chunks and formats snippet into passage-symmetric vectors.
   - `SearchService.search_by_example`: coordinates multi-segment vector retrieval and merging.
3. **Storage (`storage.py`)**:
   - `LanceStore.example_search`: vector-only LanceDB scan per segment vector, partitioned fan-out via `_fan_out_partitions`.
4. **Application (`application.py`)**:
   - `Application.search_by_example` and `ApplicationLike` interface; wired `self.indexer.extractor` to `SearchService`.
5. **Daemon & Broker (`daemon.py`)**:
   - Dispatcher entry for `search_by_example`.
   - `BrokerApplication.search_by_example` round-tripping through Unix domain socket.
6. **Server & Tools (`server.py`)**:
   - `@mcp.tool search_by_example`.
   - `search_across_projects` updated to accept `example`.
   - Updated `SERVER_INSTRUCTIONS` and `README.md`.
7. **Test Coverage**:
   - Unit tests in `tests/test_search.py`, `tests/test_storage.py`, `tests/test_application.py`.
   - Integration & protocol tests in `tests/test_server.py` and `tests/test_daemon.py`.
