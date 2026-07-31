# Scanner I/O and Store Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound the two places where the long-lived daemon's memory grows without a ceiling, and
stop reading every changed file off disk twice during a cold index.

**Architecture:** Three independent changes. Task 1 puts an LRU bound on the per-project LanceDB
table cache. Task 2 stops `list_chunks` from materialising embedding vectors it has no caller for.
Task 3 — **optional, lowest value in the whole review, read its scoping note before starting** —
moves binary and encoding validation from the scanner into the indexer so each changed file is read
once instead of twice.

**Tech Stack:** `collections.OrderedDict` for the LRU, LanceDB column projection, Pydantic model
inheritance.

## Global Constraints

See [the plan index](2026-07-27-review-followups-index.md#global-constraints). Additionally:

- **Never break the incremental contract.** Size + mtime + content hash change detection
  (`indexing.py:182-207`) and the "record the failure so the file is not re-read" behaviour
  (`indexing.py:240-265`) must survive intact. A file that has not changed must still be neither
  read, parsed, nor embedded.
- **Evicting a cache entry must not disturb an in-flight query.** The daemon runs one thread per
  connection (`daemon.py:222-227`), so eviction may only drop a dictionary reference; it must never
  close a table another thread holds.
- **Task 3 changes `IndexReport` semantics.** If it is executed, the `skipped_files` accounting and
  one scanner test change with it. Do not start it without reading its scoping note.

---

## Problem

### D1 — the partition cache never evicts

`LanceStore._partitions` (`storage.py:73`) is populated at `storage.py:437` and `storage.py:463` and
removed **only** by `remove_project` (`storage.py:350-351`). Every project the process has ever
touched keeps two open `LanceTable` handles plus their internal caches, for the life of the process.

That is fine for a CLI invocation. The daemon is the problem: it is a per-user, long-lived process
that serves every client (`daemon.py:162-241`) and only exits after 300 s fully idle. `get_chunk`
makes it worse, because it deliberately walks **every registered project** looking for the chunk
(`storage.py:183-196`), so one `get_chunk` call is enough to fault in the partition of every project
the user has ever indexed. The cache is bounded by project count, not by memory, and there is no
LRU.

### D3 — `list_chunks` materialises every vector, for tests only

`list_chunks` (`storage.py:173-181`) loads every chunk of every requested project into a list of
`StoredChunk`, each carrying a 768-float vector. Grepped across the repository: **production never
calls it.** The only callers are `tests/test_indexing.py` (10 call sites) and
`tests/test_storage.py`. `tests/test_search.py:70-84` even exists to assert that structural queries
*don't* call it.

The vector is not what the tests want either — they read `chunk_id`, `start_byte`, `end_byte`,
`content`, `start_line`, `end_line`, and `search_text`. Grepping `\.vector\b` across `tests/` finds
exactly one hit, `tests/test_embedding_worker.py:227`, and that is an `EmbeddedSegment`, not a stored
chunk. So the vectors are read off disk, decoded into Python floats, and discarded.

### C4 — every changed file is read from disk twice

`scanner.py:129` reads the full bytes of each changed file to test for NUL bytes and UTF-8 validity,
then deliberately discards them (`content=None`, `scanner.py:149`). `indexing.py:193-197` then reads
the same file again. Instrumented on a 25-file project:

```
cold index: 25 files indexed
  read_bytes calls per file: [2]   (total 50)
warm re-index (no changes): read_bytes calls = 0
```

The incremental path is genuinely excellent — zero reads when nothing changed. But a cold index pays
**2× read I/O for no memory benefit**, because the validity check could run where the bytes are
already in hand. `ScannedFile.content` exists for precisely this (`models.py:69`) and is always
`None`.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `src/code_indexing_mcp/storage.py` | `_partitions` becomes a bounded LRU; `list_chunks` projects away the vector. |
| `src/code_indexing_mcp/models.py` | `IndexedChunk` (stored chunk minus vector) is introduced and `StoredChunk` extends it. |
| `src/code_indexing_mcp/scanner.py` | Task 3 only: stops reading file contents. |
| `src/code_indexing_mcp/indexing.py` | Task 3 only: validates content where it is already read, and accounts for skips. |
| `tests/test_storage.py` | Gains LRU eviction and projection tests. |
| `tests/test_scanner.py`, `tests/test_indexing.py` | Task 3 only: the binary/encoding assertions move from scanner to indexer. |

---

### Task 1: Bound the partition cache with an LRU

**Files:**
- Modify: `src/code_indexing_mcp/storage.py:58-83` (constructor), `:347-355` (`remove_project`),
  `:419-464` (`_tables`, `_existing_tables`)
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MAX_CACHED_PARTITIONS: int` module constant, and two private helpers
  `LanceStore._cached(project_id) -> _ProjectTables | None` and
  `LanceStore._remember(project_id, tables) -> _ProjectTables`. `_tables` and `_existing_tables`
  keep their current signatures and return types.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_storage.py`:

```python
def test_partition_cache_evicts_least_recently_used(tmp_path: Path) -> None:
    """The daemon is long-lived and get_chunk faults in every project's partition.

    Without a bound, two open LanceTable handles per project accumulate for the life
    of the process.
    """
    from code_indexing_mcp import storage as storage_module

    store = LanceStore(tmp_path / "data", vector_dimension=4)
    projects = []
    for index in range(storage_module.MAX_CACHED_PARTITIONS + 3):
        root = tmp_path / f"p{index}"
        root.mkdir()
        project = ProjectInfo(id=f"id-{index:02d}", name=f"p{index}", root=root)
        store.upsert_project(project, model_id="test")
        store._tables(project.id)  # fault the partition in
        projects.append(project)

    assert len(store._partitions) == storage_module.MAX_CACHED_PARTITIONS
    # The oldest three were evicted; the most recent are still resident.
    assert projects[0].id not in store._partitions
    assert projects[1].id not in store._partitions
    assert projects[2].id not in store._partitions
    assert projects[-1].id in store._partitions


def test_partition_cache_keeps_recently_used_entries(tmp_path: Path) -> None:
    from code_indexing_mcp import storage as storage_module

    store = LanceStore(tmp_path / "data", vector_dimension=4)
    ids = []
    for index in range(storage_module.MAX_CACHED_PARTITIONS):
        root = tmp_path / f"p{index}"
        root.mkdir()
        project = ProjectInfo(id=f"id-{index:02d}", name=f"p{index}", root=root)
        store.upsert_project(project, model_id="test")
        store._tables(project.id)
        ids.append(project.id)

    # Touch the oldest so it is no longer the eviction candidate, then overflow by one.
    store._tables(ids[0])
    overflow_root = tmp_path / "overflow"
    overflow_root.mkdir()
    overflow = ProjectInfo(id="id-overflow", name="overflow", root=overflow_root)
    store.upsert_project(overflow, model_id="test")
    store._tables(overflow.id)

    assert ids[0] in store._partitions, "a freshly used partition must not be evicted"
    assert ids[1] not in store._partitions


def test_evicted_partition_reopens_with_its_data(tmp_path: Path) -> None:
    """Eviction is a cache decision, never a data decision."""
    from code_indexing_mcp import storage as storage_module

    store, project, chunk_id = _store_with_one_chunk(tmp_path)
    for index in range(storage_module.MAX_CACHED_PARTITIONS + 1):
        root = tmp_path / f"filler{index}"
        root.mkdir()
        filler = ProjectInfo(id=f"filler-{index:02d}", name=f"f{index}", root=root)
        store.upsert_project(filler, model_id="test")
        store._tables(filler.id)

    assert project.id not in store._partitions
    assert store.count_chunks([project.id]) == 1
    assert store.get_chunk(chunk_id) is not None
```

These reuse `_store_with_one_chunk(tmp_path) -> (store, project, chunk_id)`. If
[plan 3](2026-07-27-get-chunk-projection.md) has not landed, create it here by factoring the setup
that `tests/test_storage.py` already repeats — inspect with
`grep -n "def test_\|StoredChunk(\|replace_file\|upsert_project" tests/test_storage.py` and lift the
shared block. Also confirm `ProjectInfo` is imported in that file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage.py -k partition_cache -v`

Expected: FAIL — `AttributeError: module 'code_indexing_mcp.storage' has no attribute
'MAX_CACHED_PARTITIONS'`.

- [ ] **Step 3: Write the implementation**

In `src/code_indexing_mcp/storage.py`, add the import and constant:

```python
from collections import OrderedDict
```

```python
# Open partitions kept resident. Each entry holds two LanceTable handles and their
# caches, and nothing evicted them before: the daemon is long-lived and get_chunk
# walks every registered project, so one call could fault in every project a user
# has ever indexed. Sixteen covers the projects one developer works across while
# keeping the ceiling independent of how many they have registered.
MAX_CACHED_PARTITIONS = 16
```

Change the cache to an `OrderedDict` in `__init__`:

```python
        self._partitions: OrderedDict[str, _ProjectTables] = OrderedDict()
        self._partitions_lock = threading.Lock()
```

Add the two helpers next to `_tables`:

```python
    def _cached(self, project_id: str) -> _ProjectTables | None:
        """Return the cached partition for *project_id*, marking it recently used."""
        with self._partitions_lock:
            cached = self._partitions.get(project_id)
            if cached is not None:
                self._partitions.move_to_end(project_id)
            return cached

    def _remember(self, project_id: str, tables: _ProjectTables) -> _ProjectTables:
        """Cache *tables*, evicting the least recently used partition past the bound.

        Eviction only drops this dictionary's reference. A caller mid-query holds its
        own reference to the tables, so the underlying dataset stays open until that
        caller is done — the daemon serves each client on its own thread and must not
        have a table closed underneath it.
        """
        with self._partitions_lock:
            existing = self._partitions.get(project_id)
            if existing is not None:
                # Another thread opened it first; keep one instance so both callers
                # share a single set of handles.
                self._partitions.move_to_end(project_id)
                return existing
            self._partitions[project_id] = tables
            while len(self._partitions) > MAX_CACHED_PARTITIONS:
                self._partitions.popitem(last=False)
            return tables
```

Rewrite `_tables` and `_existing_tables` to use them. Note that the LanceDB connection now happens
**outside** the lock, which also removes the previous behaviour of holding the lock across I/O:

```python
    def _tables(self, project_id: str) -> _ProjectTables:
        """Open *project_id*'s partition, creating it. For write paths only."""
        cached = self._cached(project_id)
        if cached is not None:
            return cached
        database = lancedb.connect(
            self.directory / "projects" / project_id,
            read_consistency_interval=timedelta(0),
        )
        tables = _ProjectTables(
            files=self._table(database, "files", self._file_schema()),
            chunks=self._table(
                database,
                "chunks",
                self._chunk_schema(self.vector_dimension),
            ),
        )
        return self._remember(project_id, tables)

    def _existing_tables(self, project_id: str) -> _ProjectTables | None:
        """Open *project_id*'s partition without creating it, or return None.

        Reads must not materialise storage for a project they are only looking
        at. get_chunk in particular scans every registered project, so going
        through the create-on-write _tables() would leave an empty partition
        directory behind for each project that has never been indexed.
        """
        cached = self._cached(project_id)
        if cached is not None:
            return cached
        directory = self.directory / "projects" / project_id
        if not directory.is_dir():
            return None
        database = lancedb.connect(directory, read_consistency_interval=timedelta(0))
        try:
            tables = _ProjectTables(
                files=cast(LanceTable, database.open_table("files")),
                chunks=cast(LanceTable, database.open_table("chunks")),
            )
        except (ValueError, FileNotFoundError):
            return None
        return self._remember(project_id, tables)
```

`remove_project` needs no change: `OrderedDict.pop(key, None)` behaves identically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage.py -v`

Expected: all pass, including the three new tests and the existing partition/migration tests.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: green. `tests/test_daemon.py` exercises the multi-client path and is the one most likely to
surface a locking mistake.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/storage.py tests/test_storage.py
git commit -m "fix: bound the per-project table cache with an LRU"
```

---

### Task 2: Stop `list_chunks` reading embedding vectors

**Files:**
- Modify: `src/code_indexing_mcp/models.py:122-141`
- Modify: `src/code_indexing_mcp/storage.py:173-181`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `IndexedChunk` in `models.py` — every `StoredChunk` field except `vector`. `StoredChunk`
  becomes `class StoredChunk(IndexedChunk)` adding only `vector`, so its field set and field **order**
  are unchanged. `LanceStore.list_chunks(...) -> list[IndexedChunk]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_storage.py`:

```python
def test_list_chunks_does_not_materialize_vectors(tmp_path: Path) -> None:
    """Nothing in production calls list_chunks; the vectors were read for no one."""
    from code_indexing_mcp.models import IndexedChunk

    store, project, _ = _store_with_one_chunk(tmp_path)

    chunks = store.list_chunks([project.id])

    assert chunks
    assert all(isinstance(chunk, IndexedChunk) for chunk in chunks)
    assert not any(hasattr(chunk, "vector") for chunk in chunks)
    # The fields the tests actually read must survive the projection.
    assert chunks[0].search_text
    assert chunks[0].content
    assert chunks[0].chunk_id


def test_stored_chunk_still_carries_its_vector(tmp_path: Path) -> None:
    """The write path is unaffected: StoredChunk keeps the vector it commits."""
    from code_indexing_mcp.models import IndexedChunk, StoredChunk

    assert issubclass(StoredChunk, IndexedChunk)
    assert "vector" in StoredChunk.model_fields
    assert "vector" not in IndexedChunk.model_fields
    # Field order matters to nothing in LanceDB, but the schema lists vector last
    # and keeping it there makes the inheritance a pure refactor.
    assert list(StoredChunk.model_fields)[-1] == "vector"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_storage.py -k "list_chunks_does_not or still_carries" -v`

Expected: FAIL — `ImportError: cannot import name 'IndexedChunk' from 'code_indexing_mcp.models'`.

- [ ] **Step 3: Split the model**

In `src/code_indexing_mcp/models.py`, replace the `StoredChunk` definition at lines 122-141:

```python
class IndexedChunk(FrozenModel):
    """A committed chunk without its embedding vector.

    Read paths that only need chunk text and offsets use this so a whole project's
    768-float vectors are not decoded into Python lists for no consumer.
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
    embedding_text: str
    search_text: str
    content_hash: str
    part_index: int = 0


class StoredChunk(IndexedChunk):
    """A chunk as written to storage, vector included."""

    vector: list[float]
```

- [ ] **Step 4: Project the columns in `list_chunks`**

In `src/code_indexing_mcp/storage.py`, add `IndexedChunk` to the models import, add the column list beside
the other projections:

```python
# Every chunk column except the vector. list_chunks has no production caller and its
# test callers read text and offsets, so decoding vectors was pure waste.
INDEXED_CHUNK_COLUMNS = [
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
    "embedding_text",
    "search_text",
    "content_hash",
    "part_index",
]
```

and replace `list_chunks`:

```python
    def list_chunks(self, project_ids: Iterable[str] | None = None) -> list[IndexedChunk]:
        ids = list(project_ids or [project.id for project in self.list_projects()])
        chunks: list[IndexedChunk] = []
        for project_id in ids:
            tables = self._existing_tables(project_id)
            if tables is None:
                continue
            rows = cast(
                list[dict[str, Any]],
                tables.chunks.search().select(INDEXED_CHUNK_COLUMNS).to_list(),
            )
            chunks.extend(IndexedChunk.model_validate(row) for row in rows)
        return chunks
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_storage.py tests/test_indexing.py -q`

Expected: all pass. `tests/test_indexing.py` has ten `list_chunks` call sites reading `chunk_id`,
`start_byte`, `end_byte`, `content`, `start_line`, `end_line`, and `search_text` — all present on
`IndexedChunk`. If one fails on a missing attribute, that field belongs in
`INDEXED_CHUNK_COLUMNS` and `IndexedChunk`; add it to both rather than reverting.

- [ ] **Step 6: Run the full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/models.py src/code_indexing_mcp/storage.py tests/test_storage.py
git commit -m "perf: project away embedding vectors in list_chunks"
```

---

### Task 3 (optional): Read each changed file once

> **Scoping note — decide before starting.** This is the lowest-value item in the entire review, and
> a reviewer may reasonably drop it. It halves read syscalls on a **cold** index only; a warm
> re-index already performs zero reads, and the README's own benchmark shows embedding dominating a
> cold index at 141 s of 147 s. It also **changes `IndexReport.skipped_files` accounting** and moves
> one scanner test. Take it if cold-index I/O on large repositories matters to you; skip it
> otherwise. Tasks 1 and 2 stand alone without it.

**Files:**
- Modify: `src/code_indexing_mcp/scanner.py:114-151`
- Modify: `src/code_indexing_mcp/indexing.py:168-296`
- Modify: `src/code_indexing_mcp/models.py:63-69` (drop the now-dead `content` field)
- Test: `tests/test_scanner.py:76-92`, `tests/test_indexing.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces: `SourceScanner.scan` no longer reads file contents and no longer emits `"binary"` or
  `"encoding"` skip reasons. `ScannedFile.content` is removed. `IndexReport.skipped_files` becomes
  `len(scan.skipped) + <content-rejected count>`.

- [ ] **Step 1: Write the failing tests**

Replace the content assertions in `tests/test_scanner.py:76-92`. The scanner keeps rejecting
oversized files and symlinks, which are `stat`-only decisions; binary and encoding move out:

```python
def test_scanner_rejects_oversized_and_symlink_files_without_reading(tmp_path: Path) -> None:
    """Size and symlink checks are stat-only; content checks belong to the indexer.

    The scanner used to read every changed file to test for NUL bytes and UTF-8
    validity, then discard the bytes, so the indexer read the same file again.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.py").write_text("def ok():\n    return 1\n")
    (root / "big.py").write_text("x = 1\n" * 400_000)
    (root / "link.py").symlink_to(root / "ok.py")
    project = initialize_project(root)
    project = project.model_copy(
        update={"scan": project.scan.model_copy(update={"max_file_bytes": 1_024})}
    )

    result = SourceScanner().scan(project)

    reasons = {skip.reason for skip in result.skipped}
    assert "oversized" in reasons
    assert "symlink" in reasons
    assert "binary" not in reasons
    assert "encoding" not in reasons
    assert {item.path.as_posix() for item in result.files} == {"ok.py"}


def test_scanner_does_not_read_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.py").write_text("def ok():\n    return 1\n")
    project = initialize_project(root)

    def reject_read(self: Path) -> bytes:
        raise AssertionError(f"scan must not read {self}")

    monkeypatch.setattr(Path, "read_bytes", reject_read)

    assert len(SourceScanner().scan(project).files) == 1
```

Append to `tests/test_indexing.py`:

```python
def test_binary_and_undecodable_files_are_skipped_by_the_indexer(tmp_path: Path) -> None:
    """Content rejection moved to where the bytes are already read.

    The file must not be indexed, must not be committed as a stored file, and any
    chunks from an earlier text version of it must be dropped.
    """
    store, indexer, project, root = _indexer(tmp_path)  # existing helper in this file
    (root / "good.py").write_text("def good():\n    return 1\n")
    (root / "turned_binary.py").write_text("def old():\n    return 0\n")
    indexer.index(project)
    assert {chunk.path for chunk in store.list_chunks([project.id])} >= {"turned_binary.py"}

    (root / "turned_binary.py").write_bytes(b"def old():\x00\n")
    (root / "latin.py").write_bytes(b"# caf\xe9\ndef latin():\n    return 2\n")
    report = indexer.index(project)

    indexed = {chunk.path for chunk in store.list_chunks([project.id])}
    assert indexed == {"good.py"}
    assert report.skipped_files >= 2
    assert {issue.path for issue in report.errors} == set()


def test_each_changed_file_is_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, indexer, project, root = _indexer(tmp_path)
    for index in range(5):
        (root / f"m{index}.py").write_text(f"def f{index}():\n    return {index}\n")

    reads: dict[str, int] = {}
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        if self.suffix == ".py":
            reads[str(self)] = reads.get(str(self), 0) + 1
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    indexer.index(project)

    assert len(reads) == 5
    assert set(reads.values()) == {1}, f"expected one read per file, got {reads}"
```

`_indexer(tmp_path) -> (store, indexer, project, root)` must be factored from the setup
`tests/test_indexing.py` already repeats; inspect with
`grep -n "def _\|Indexer(\|initialize_project" tests/test_indexing.py` and reuse the existing helper
if one already matches.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scanner.py tests/test_indexing.py -k "does_not_read or read_once or turned_binary or without_reading" -v`

Expected: FAIL — `AssertionError: scan must not read .../ok.py`, and
`expected one read per file, got {...: 2, ...}`.

- [ ] **Step 3: Stop the scanner reading**

In `src/code_indexing_mcp/scanner.py`, replace the body of the per-candidate loop in `scan` from the
`previous = known_files.get(...)` line through the `files.append(...)` call:

```python
            files.append(
                ScannedFile(
                    path=relative,
                    absolute_path=absolute,
                    language=language,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )
```

Delete the `previous = known_files.get(relative.as_posix())` block and the NUL/UTF-8 checks entirely.
`known_files` is now unused by `scan`; keep the parameter for call compatibility and mark it:

```python
    def scan(
        self, project: ProjectInfo, known_files: dict[str, StoredFile] | None = None
    ) -> ScanResult:
        """Discover eligible source files using stat-only decisions.

        *known_files* is accepted for call compatibility and no longer consulted:
        content checks moved to the indexer, which is where the bytes it needs are
        already read, and change detection lives there too.
        """
        del known_files
```

Then remove `content: bytes | None = None` from `ScannedFile` in `models.py:63-69` — it was always
`None` and nothing reads it now. `mypy` will point at any remaining reference.

- [ ] **Step 4: Validate content in the indexer**

In `src/code_indexing_mcp/indexing.py`, add the helper above the `Indexer` class:

```python
def _content_rejection(source: bytes) -> str | None:
    """Return why *source* cannot be indexed, or None when it can.

    Runs where the bytes are already in hand. The scanner used to do this and throw
    the bytes away, which cost every changed file a second full read.
    """
    if b"\x00" in source:
        return "binary"
    try:
        source.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "encoding"
    return None
```

In `_index_scan`, add a counter beside the others and apply the check right after the read. Replace
the read block at `indexing.py:191-207`:

```python
            content_hash: str | None = None
            try:
                with timer.measure("scan"):
                    source = item.absolute_path.read_bytes()
                    rejection = _content_rejection(source)
                    if rejection is None:
                        content_hash = _digest(source)
                if rejection is not None:
                    # Not indexable and not an error against the file's syntax. Drop
                    # anything a previous text version left behind so the index does
                    # not keep serving chunks for content that is now unreadable.
                    with timer.measure("commit"):
                        if previous is not None:
                            self.store.remove_file(project.id, previous.file_id)
                    content_rejected += 1
                    continue
                if not force and previous is not None and previous.content_hash == content_hash:
```

Initialise the counter with the others at `indexing.py:176`:

```python
        indexed = parsed = embedded = unchanged = metadata_only = removed = 0
        content_rejected = 0
```

`item.content` no longer exists, so the `item.content if item.content is not None else` branch is
gone — that is the line the double read came from.

The rejected path uses `continue`, so `current_paths` still contains the file and the removal loop at
`indexing.py:267-271` will not double-remove it.

Finally, fold the counter into the report at `indexing.py:290`:

```python
            skipped_files=len(scan.skipped) + content_rejected,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scanner.py tests/test_indexing.py -q`

Expected: all pass. Existing tests that assert an exact `skipped_files` number will shift by the
number of binary/undecodable files in their fixtures — verify each change is the expected
reclassification rather than a lost file before updating the number.

- [ ] **Step 6: Confirm the halved I/O**

```bash
.venv/bin/python - <<'PY'
import os, tempfile
from pathlib import Path
os.environ["CODE_INDEXING_OFFLINE"] = "1"
os.environ["CODE_INDEXING_INDEX_EXECUTION"] = "in-process"
from code_indexing_mcp.application import Application, RuntimePaths

class F:
    model_id = "f"; dimension = 8
    def embed_passages(self, texts): return [[0.0] * 8 for _ in texts]
    def embed_query(self, text): return [0.0] * 8

reads: dict[str, int] = {}
original = Path.read_bytes
def counting(self):
    reads[str(self)] = reads.get(str(self), 0) + 1
    return original(self)
Path.read_bytes = counting

tmp = Path(tempfile.mkdtemp()); proj = tmp / "p"; proj.mkdir()
(proj / "pyproject.toml").write_text("[project]\nname='x'\n")
for i in range(25):
    (proj / f"m{i}.py").write_text(f"def f{i}():\n    return {i}\n")
app = Application(RuntimePaths(data=tmp / "d", cache=tmp / "c"), embedder=F(), cwd=proj)
info = app.init_project(proj)
reads.clear(); report = app.index_project(info.id)
py = {k: v for k, v in reads.items() if k.endswith(".py")}
print(f"cold index: {report.indexed_files} files, reads per file {sorted(set(py.values()))}, "
      f"total {sum(py.values())} (was 50)")
reads.clear(); warm = app.index_project(info.id)
print(f"warm re-index: unchanged={warm.unchanged_files}, reads="
      f"{sum(v for k, v in reads.items() if k.endswith('.py'))}")
PY
```

Expected: `reads per file [1], total 25 (was 50)` and `warm re-index: unchanged=25, reads=0`.

- [ ] **Step 7: Full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/scanner.py src/code_indexing_mcp/indexing.py src/code_indexing_mcp/models.py tests/
git commit -m "perf: read each changed file once by validating content in the indexer"
```

- [ ] **Step 8: Document the accounting change**

In `README.md`, in the incremental-indexing list around lines 146-151, add:

```markdown
- Binary and undecodable files are detected while the file is read for indexing, not during the
  scan, so each changed file is read once. They count toward `skipped_files` and are not recorded as
  per-file errors.
```

---

## Self-Review

**Spec coverage.** Review items D1 (Task 1), D3 (Task 2), and C4 (Task 3) are covered. Two review
items are deliberately **not** tasks here:

- **D2 — `_walk` materialises every path in the tree** (`scanner.py:157-173`), verified at 526
  candidates for 25 indexable files. Filtering by suffix during the walk would fix it, but every
  non-source file currently produces a `SkippedFile(reason="unsupported")`, so filtering early
  changes what `skipped_files` counts and breaks the assertion at `tests/test_scanner.py:28`. The
  win only materialises on very large monorepos. Worth doing as its own change, with its own
  decision about whether per-file "unsupported" records are worth keeping.
- **D4 — one unbounded thread per daemon connection** (`daemon.py:222-227`). Bounding it needs a
  worker pool and a decision about queueing versus rejecting under load. Not a cache fix.
- **D5 — nothing bounds query-side memory.** The 300 s idle exit reclaims it. Bounding it properly
  means unloading the ONNX query model on idle, which is a design change to the daemon lifecycle.

**Type consistency.** `MAX_CACHED_PARTITIONS`, `_cached`, and `_remember` are defined in Task 1
Step 3 and used in the same step. `IndexedChunk` is defined in Task 2 Step 3, consumed by
`list_chunks` in Step 4, and `INDEXED_CHUNK_COLUMNS` matches its 18 fields one-for-one.
`_content_rejection` is defined in Task 3 Step 4 and used in the same step.

**Task independence.** Tasks 1 and 2 both edit `storage.py` but in disjoint regions — the cache
helpers versus `list_chunks`. Task 3 touches neither. Any order works; running 1 then 2 avoids a
trivial context overlap.

**If Task 3 is skipped**, note it in the branch description so a later reader does not assume the
double read was measured away. The `ScannedFile.content` field stays dead in that case, which is
itself worth a comment pointing at this plan.
