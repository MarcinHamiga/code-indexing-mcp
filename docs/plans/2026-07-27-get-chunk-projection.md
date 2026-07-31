# `get_chunk` Projection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `get_chunk` from returning the 768-dimension embedding vector and three copies of the
same code, cutting its response from ~5,400 tokens to ~580 for the same content.

**Architecture:** `CodeChunk` stops being a bare subclass of `StoredChunk` and becomes an explicit
projection listing only the fields a caller can use. `LanceStore.get_chunk` selects just those
columns, so the vector is never read off disk either. No indexing, search-ranking, or storage-schema
changes — the stored rows are untouched, so no re-index is required.

**Tech Stack:** Pydantic v2 models, LanceDB column projection.

## Global Constraints

See [the plan index](2026-07-27-review-followups-index.md#global-constraints). Additionally:

- **No storage schema change and no re-index.** `_chunk_schema` and `SCHEMA_VERSION` stay exactly as
  they are. This plan only narrows what is *read*.
- **`CodeChunk.content` must keep returning the full stored text.** The whole point of the tool is
  that it is the untruncated counterpart to `SearchHit.snippet`, which is capped at 4,000 chars.
- **Run `code-indexing-mcp daemon restart` after deploying.** See the version-skew note in the
  Self-Review; a running daemon serves the old shape.

---

## Problem

`models.py:240-241` is the whole definition:

```python
class CodeChunk(StoredChunk):
    pass
```

So the `get_chunk` tool returns every storage column, including `vector: list[float]`. Measured on a
real chunk from this repository:

| field | chars |
|---|---|
| **vector** | **15,556** |
| search_text | 1,839 |
| embedding_text | 1,815 |
| content | 1,753 |
| chunk_id, file_id, project_id, content_hash, path, offsets, … | ~290 |
| **total** | **21,546 chars ≈ 5,386 tokens** |

That is **1,753 characters of actual code** delivered in a 21.5 KB payload:

- **72% of the response is the embedding vector**, which is meaningless to a model consumer.
- **The code is sent three times.** `content` is the text; `embedding_text` is
  `prefix + "\n" + content`; `search_text` is `embedding_text + "\n" + normalized_identifiers`
  (`extractor.py:356-379`). 5,232 chars to convey 1,697.

Roughly **13× amplification on the one tool whose only job is "give me the full text of this
chunk"**. `LanceStore.get_chunk` also reads all of it off disk, via `_rows()` at `storage.py:193`,
which issues an unprojected `table.search()`.

For contrast, the sibling read paths are already careful: `hybrid_search` projects its columns
(`storage.py:225-239`), `ChunkPreview` is documented as *"A query result that deliberately excludes
embedding and index payloads"* (`models.py:144-145`), and `tests/test_search.py:87-101` asserts
`vector`, `embedding_text`, and `search_text` never appear in search rows. `get_chunk` is the one
path that missed the pattern.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `src/code_indexing_mcp/models.py` | `CodeChunk` becomes an explicit projection model, independent of `StoredChunk`. |
| `src/code_indexing_mcp/storage.py` | `get_chunk` selects only the projected columns and returns `CodeChunk`. |
| `src/code_indexing_mcp/search.py` | `get_chunk` returns the store's value directly instead of re-validating a full `model_dump`. |
| `tests/test_search.py` | Gains a payload-shape contract test. |
| `tests/test_storage.py` | Existing `get_chunk` returns-`None` test keeps passing; gains a projection assertion. |
| `README.md` | Documents what `get_chunk` returns. |

---

### Task 1: Make `CodeChunk` an explicit projection

**Files:**
- Modify: `src/code_indexing_mcp/models.py:240-241`
- Modify: `src/code_indexing_mcp/storage.py:183-196`
- Modify: `src/code_indexing_mcp/search.py:120-124`
- Test: `tests/test_search.py`, `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing from other plans.
- Produces: `CodeChunk` with fields `chunk_id`, `file_id`, `project_id`, `path`, `language`, `kind`,
  `symbol`, `qualified_symbol`, `parent_symbol`, `start_byte`, `end_byte`, `start_line`, `end_line`,
  `content`, `content_hash`, `part_index` — and **no** `vector`, `embedding_text`, or `search_text`.
  `LanceStore.get_chunk(chunk_id: str) -> CodeChunk | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search.py`, reusing the `_indexed_source` helper already at line 175:

```python
CHUNK_PAYLOAD_FIELDS = {
    "chunk_id",
    "file_id",
    "project_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "content_hash",
    "part_index",
}


def test_get_chunk_excludes_embedding_and_duplicated_text(tmp_path: Path) -> None:
    """get_chunk is the full-text counterpart to a snippet, not a storage row dump.

    The vector was 72% of the old payload and the code text appeared three times
    over, in content, embedding_text, and search_text.
    """
    search, project = _indexed_source(
        tmp_path, "def authenticate(user):\n    return user.token\n"
    )
    hit = search.search_code("authenticate", [project]).hits[0]

    payload = search.get_chunk(hit.chunk_id).model_dump()

    assert set(payload) == CHUNK_PAYLOAD_FIELDS
    assert "vector" not in payload
    assert "embedding_text" not in payload
    assert "search_text" not in payload
    assert "return user.token" in payload["content"]


def test_get_chunk_payload_is_dominated_by_content(tmp_path: Path) -> None:
    import json

    source = "def authenticate(user):\n" + "".join(
        f"    step_{index} = user.token\n" for index in range(80)
    )
    search, project = _indexed_source(tmp_path, source)
    hit = search.search_code("authenticate", [project]).hits[0]

    chunk = search.get_chunk(hit.chunk_id)
    encoded = json.dumps(chunk.model_dump(mode="json"))

    # Metadata is ~290 chars of ids and offsets; anything beyond a small multiple of
    # the content means a payload field crept back in.
    assert len(encoded) < len(chunk.content) * 2
```

Append to `tests/test_storage.py`:

```python
def test_get_chunk_does_not_read_the_vector_column(tmp_path: Path) -> None:
    from code_indexing_mcp.models import CodeChunk

    store, project, chunk_id = _store_with_one_chunk(tmp_path)

    chunk = store.get_chunk(chunk_id)

    assert isinstance(chunk, CodeChunk)
    assert not hasattr(chunk, "vector")
    assert store.get_chunk("no-such-chunk") is None
```

`_store_with_one_chunk` must return `(store, project, chunk_id)`. `tests/test_storage.py` already
builds `LanceStore`, `ProjectInfo`, `StoredFile`, and `StoredChunk` rows for its existing tests —
factor the setup those tests share into that helper rather than writing a fourth copy. Inspect it
first with `grep -n "StoredChunk(\|def test_\|replace_file" tests/test_storage.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -k get_chunk tests/test_storage.py -k get_chunk -v`

Expected: FAIL — `set(payload) == CHUNK_PAYLOAD_FIELDS` fails because the actual set also contains
`vector`, `embedding_text`, and `search_text`.

- [ ] **Step 3: Redefine `CodeChunk`**

In `src/code_indexing_mcp/models.py`, replace lines 240-241:

```python
class CodeChunk(FrozenModel):
    """One indexed chunk as returned to a caller.

    Deliberately not a StoredChunk subclass. Inheriting the storage row shipped the
    768-dimension vector and both derived text columns to MCP clients: 72% of the
    response was the vector, and the code arrived three times over as content,
    embedding_text, and search_text. Adding a storage column must not silently
    widen this payload, so the fields are listed rather than inherited.
    """

    chunk_id: str
    file_id: str
    project_id: str
    path: str
    language: str
    kind: str
    symbol: str | None = None
    qualified_symbol: str | None = None
    parent_symbol: str | None = None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    content: str
    content_hash: str
    part_index: int = 0
```

`file_id` and `content_hash` stay: both are already in the payload's noise floor at 66 chars each,
and `content_hash` is how a caller detects that a chunk it cached has been re-indexed. `start_byte`
and `end_byte` stay because they are what an editing caller needs to splice the chunk back into the
file.

- [ ] **Step 4: Project the columns in storage**

In `src/code_indexing_mcp/storage.py`, add `CodeChunk` to the `from .models import (...)` block, add the
projection constant next to `OVERFETCH_FACTOR`:

```python
# Columns get_chunk reads. The vector and the two derived text columns are excluded:
# nothing outside indexing and ranking can use them, and reading them made a
# single-chunk fetch an order of magnitude larger than the code it returned.
CHUNK_PAYLOAD_COLUMNS = [
    "chunk_id",
    "file_id",
    "project_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "content_hash",
    "part_index",
]
```

and replace `get_chunk` (lines 183-196):

```python
    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        # chunk_id is a one-way digest of file_id, which is itself a digest of
        # the project id and path, so the owning project cannot be recovered
        # from the id. Scanning every project is inherent without an id-format
        # change and a full re-index; do not "fix" it by narrowing the loop.
        # The partitions open read-only so the scan leaves nothing behind.
        for project in self.list_projects():
            tables = self._existing_tables(project.id)
            if tables is None:
                continue
            rows = cast(
                list[dict[str, Any]],
                tables.chunks.search()
                .where(f"chunk_id = {_quoted(chunk_id)}")
                .select(CHUNK_PAYLOAD_COLUMNS)
                .to_list(),
            )
            if rows:
                return CodeChunk.model_validate(rows[0])
        return None
```

- [ ] **Step 5: Simplify the search service**

In `src/code_indexing_mcp/search.py`, replace `get_chunk` (lines 120-124):

```python
    def get_chunk(self, chunk_id: str) -> CodeChunk:
        chunk = self.store.get_chunk(chunk_id)
        if chunk is None:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown chunk: {chunk_id}")
        return chunk
```

The `CodeChunk.model_validate(chunk.model_dump())` round trip is gone — the store already returns the
projection.

> If [plan 1](2026-07-27-mcp-tool-surface.md) has already landed, this method raises
> `ErrorCode.CHUNK_NOT_FOUND` with the recovery hint instead. Keep that version and change only the
> return statement.

Remove `StoredChunk` from `search.py`'s model imports if nothing else in the file uses it — check
with `grep -n "StoredChunk" src/code_indexing_mcp/search.py`. It is still referenced by `_hit`'s type
union at `search.py:131`, so it most likely stays.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py tests/test_storage.py -q`

Expected: all pass, including the pre-existing
`test_search_truncates_snippet_and_get_chunk_returns_full_content` at line 148, which asserts
`len(full.content) > len(hits[0].snippet)` — `content` is retained, so it is unaffected.

- [ ] **Step 7: Measure the improvement**

Run:

```bash
.venv/bin/python - <<'PY'
import json, os, random, shutil, tempfile
from pathlib import Path
os.environ["CODE_INDEXING_OFFLINE"] = "1"
os.environ["CODE_INDEXING_INDEX_EXECUTION"] = "in-process"
from code_indexing_mcp.application import Application, RuntimePaths

random.seed(0)
class F:
    model_id = "fake-768"; dimension = 768
    def embed_passages(self, texts): return [[random.random() for _ in range(768)] for _ in texts]
    def embed_query(self, text): return [random.random() for _ in range(768)]

tmp = Path(tempfile.mkdtemp()); proj = tmp / "proj"; proj.mkdir()
shutil.copytree("src/code_indexing_mcp", proj / "code_indexing_mcp")
(proj / "pyproject.toml").write_text("[project]\nname='x'\n")
app = Application(RuntimePaths(data=tmp / "d", cache=tmp / "c"), embedder=F(), cwd=proj)
info = app.init_project(proj); app.index_project(info.id)
hit = app.search_code("token window planning", projects=[info.id], limit=8).hits[0]
chunk = app.get_chunk(hit.chunk_id)
payload = json.dumps(chunk.model_dump(mode="json"))
print(f"get_chunk payload: {len(payload):,} chars (~{len(payload)//4:,} tokens)")
print(f"  content: {len(chunk.content):,} chars")
print(f"  overhead ratio: {len(payload)/len(chunk.content):.2f}x")
print(f"  fields: {sorted(chunk.model_dump())}")
PY
```

Expected: roughly **2,300 chars / ~580 tokens** with an overhead ratio near `1.3x`, against 21,546
chars / ~5,386 tokens and `12.3x` before. The field list must contain no `vector`,
`embedding_text`, or `search_text`.

- [ ] **Step 8: Run the full suite, lint, type-check**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
```

Expected: green. `mypy` is the important one here: it catches any remaining caller that expected
`CodeChunk.vector`. `application.py:237`, `daemon.py:473`, and `server.py:438` all just pass the
value through and need no change.

- [ ] **Step 9: Document it**

In `README.md`, in the tool table or list, make the `get_chunk` entry explicit:

```markdown
`get_chunk` returns one chunk's full stored text with its path, symbol, line range, byte range, and
content hash. It deliberately excludes the embedding vector and the derived `embedding_text` and
`search_text` columns, which exist for ranking and are not useful to a caller.
```

- [ ] **Step 10: Commit**

```bash
git add src/code_indexing_mcp/models.py src/code_indexing_mcp/storage.py src/code_indexing_mcp/search.py \
        tests/test_search.py tests/test_storage.py README.md
git commit -m "perf: return a projection from get_chunk instead of the whole storage row"
```

---

## Self-Review

**Spec coverage.** Review item A3 is fully covered: the vector and both duplicated text columns are
removed from the model (Task 1 Step 3) and from the disk read (Step 4), with the reduction measured
in Step 7.

**Type consistency.** `CodeChunk`'s 16 fields in Task 1 Step 3 match `CHUNK_PAYLOAD_COLUMNS` in Step
4 one-for-one and `CHUNK_PAYLOAD_FIELDS` in the Step 1 test. If a reviewer adds a field, all three
lists must change together — the Step 1 test fails loudly if they drift, which is the intent.

**Version skew during upgrade — read this before deploying.** `BrokerApplication.get_chunk`
(`daemon.py:473-474`) validates whatever the daemon sends. A *newly upgraded client* talking to an
*already-running old daemon* is fine, because Pydantic ignores the extra `vector` field. The reverse
is not: an old client validating a new daemon's response fails, because its `CodeChunk` inherits
`StoredChunk` and requires `vector`. The daemon exits after 300 s idle
(`DaemonServer.idle_timeout_seconds`), so the window is short, but run
`code-indexing-mcp daemon restart` after upgrading to close it deliberately. Add that line to the
release notes.

**Deliberately out of scope.** `SearchHit.snippet` is capped at 4,000 chars (`search.py:132`), so
`search_code(limit=8)` measured 10,497 chars. That is defensible for eight results and is not part
of this plan. Lowering the cap or making it a parameter is a separate judgement call about recall
versus context.
