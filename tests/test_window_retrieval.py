"""Embedding windows are matches inside a retrievable source chunk."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import lancedb
import pytest
from test_search import SemanticEmbedder
from test_token_batching import FakeEncoding, fake_encode

from code_indexing_mcp.embedding import (
    EmbeddedSegment,
    PassageCandidate,
    SegmentPlan,
    embed_planned_segments,
    pack_vector,
)
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import DeclarationSelector, ExtractedChunk, ProjectInfo
from code_indexing_mcp.passage_cache import PassageCacheNamespace, PassageReuseContext
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.search import SearchService
from code_indexing_mcp.storage import LanceStore, PartitionRef
from code_indexing_mcp.token_batching import content_token_offsets, plan_token_windows


class AdaptiveEmbedder(SemanticEmbedder):
    tokenizer_available = True

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return super().embed_passages(texts)

    @staticmethod
    def encode(text: str) -> FakeEncoding:
        encoding = fake_encode(text)
        # The reserved header ends in a separator. A complete passage adds
        # boundary cost that is absent from the header-only measurement.
        if "\n" in text and not text.endswith("\n"):
            return FakeEncoding(
                offsets=[*encoding.offsets, (0, 0), (0, 0)],
                special_tokens_mask=[*encoding.special_tokens_mask, 1, 1],
            )
        return encoding

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        return [
            [
                EmbeddedSegment(w.start_char, w.end_char, w.token_count, pack_vector(vector))
                for w, vector in windows
            ]
            for windows in embed_planned_segments(
                self.encode, self.embed_passages, candidates, plan
            )
        ]


SOURCE = (
    "# π and 世界 before the declaration\n"
    "def process_data(user):\n"
    "    beginning_marker = 'héllo 世界'\n"
    + "".join(f"    stage_{i} = user.value + {i}\n" for i in range(8))
    + "    ending_marker = user.invoice\n"
    "    return ending_marker\n"
)


def indexed_windows(
    tmp_path: Path,
    source: str = SOURCE,
    *,
    cache: bool = False,
    max_tokens: int = 48,
    embedder: AdaptiveEmbedder | None = None,
) -> tuple[Indexer, LanceStore, SearchService, ProjectInfo, list[ExtractedChunk]]:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_bytes(source.encode("utf-8"))
    project = initialize_project(root)
    embedder = embedder or AdaptiveEmbedder()
    extractor = TreeSitterExtractor()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    namespace = PassageCacheNamespace(
        project_id=project.id,
        artifact_digest="artifact",
        tokenizer_digest="tokenizer",
        producer="test-float32",
        runtime_version="test",
        dimension=embedder.dimension,
        precision="float32-le",
    )

    def cache_factory(current: ProjectInfo, force: bool) -> PassageReuseContext:
        return PassageReuseContext(
            tmp_path / "passage.sqlite3", replace(namespace, project_id=current.id), force=force
        )

    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=extractor,
        embedder=embedder,
        lock_directory=tmp_path / "locks",
        segment_plan=SegmentPlan(max_tokens=max_tokens, overlap_tokens=4),
        passage_cache_factory=cache_factory if cache else None,
    )
    report = indexer.index(project)
    assert report.errors == []
    original = extractor.extract(Path("main.py"), "python", source.encode("utf-8")).chunks
    return indexer, store, SearchService(store, embedder), project, original


def test_every_adaptive_window_retrieves_the_original_indexed_chunk(tmp_path: Path) -> None:
    _, store, search, project, original = indexed_windows(tmp_path)
    expected = next(chunk for chunk in original if chunk.symbol == "process_data")
    # This source exercises adaptive replanning, not only the older windowing path.
    header = expected.embedding_prefix + "\n"
    initial = plan_token_windows(
        content_token_offsets(AdaptiveEmbedder.encode(expected.content)),
        text_length=len(expected.content),
        max_tokens=48 - len(AdaptiveEmbedder.encode(header).offsets),
        overlap_tokens=4,
    )
    assert any(
        len(AdaptiveEmbedder.encode(header + expected.content[w.start_char : w.end_char]).offsets)
        > 48
        for w in initial
    )
    windows = [
        chunk for chunk in store.list_chunks([project.id]) if chunk.symbol == expected.symbol
    ]
    assert len(windows) > 1
    # Retrieval must use the indexed snapshot, including when the checkout is gone.
    (project.root / "main.py").unlink()

    for window in windows:
        retrieved = search.get_chunk(window.chunk_id)
        assert retrieved.chunk_id == window.chunk_id
        assert retrieved.content == expected.content
        assert (retrieved.start_byte, retrieved.end_byte) == (
            expected.start_byte,
            expected.end_byte,
        )
        assert (retrieved.start_line, retrieved.end_line) == (
            expected.start_line,
            expected.end_line,
        )
        assert (
            SOURCE.encode()[retrieved.start_byte : retrieved.end_byte].decode() == retrieved.content
        )


@pytest.mark.parametrize("query", ["beginning_marker", "ending_marker", "billing invoice"])
def test_search_matches_retrieve_complete_context(tmp_path: Path, query: str) -> None:
    _, _, search, project, original = indexed_windows(tmp_path)
    expected = next(chunk for chunk in original if chunk.symbol == "process_data")

    response = search.search_code(query, [project.id])
    hits = [hit for hit in response.hits if hit.symbol == "process_data"]

    assert len(hits) == 1
    assert hits[0].truncated
    assert search.get_chunk(hits[0].chunk_id).content == expected.content


def test_symbol_lookup_and_outline_span_the_original_chunk(tmp_path: Path) -> None:
    _, _, search, project, original = indexed_windows(tmp_path)
    expected = next(chunk for chunk in original if chunk.symbol == "process_data")

    found = search.find_symbol("process_data", project.id)
    assert len(found.hits) == 1
    hit = found.hits[0]
    assert (hit.start_line, hit.end_line) == (expected.start_line, expected.end_line)
    # The hit spans the whole chunk but shows one window, so it must say so.
    assert hit.truncated
    assert expected.content.startswith(hit.snippet)
    assert search.get_chunk(hit.chunk_id).content == expected.content
    item = next(
        item
        for item in search.file_outline("main.py", project.id).items
        if item.symbol == expected.symbol
    )
    assert (item.start_line, item.end_line) == (expected.start_line, expected.end_line)


def test_dense_line_symbol_hits_represent_their_first_window(tmp_path: Path) -> None:
    # Every window of a one-line declaration shares its start line, so only
    # source byte order can pick the window that opens the declaration.
    source = "".join(
        f"def dense{i}(user): return " + " + ".join(f"user.v{j}" for j in range(60)) + "\n"
        for i in range(8)
    )
    _, store, search, project, original = indexed_windows(tmp_path, source)
    references = ReferenceService(store)

    for index in range(8):
        name = f"dense{index}"
        expected = next(chunk for chunk in original if chunk.symbol == name)
        windows = [chunk for chunk in store.list_chunks([project.id]) if chunk.symbol == name]
        assert len(windows) > 1
        assert len({window.start_line for window in windows}) == 1
        first = min(windows, key=lambda window: window.start_byte)

        hit = search.find_symbol(name, project.id).hits[0]
        assert hit.chunk_id == first.chunk_id
        assert hit.truncated
        assert hit.snippet == first.content
        assert expected.content.startswith(hit.snippet)
        selected = references.find_references(
            DeclarationSelector(project=project.id, path="main.py", qualified_symbol=name)
        ).selected
        assert selected.chunk_id == first.chunk_id


def test_windowing_does_not_make_a_declaration_selector_ambiguous(tmp_path: Path) -> None:
    _, store, _, project, original = indexed_windows(tmp_path)
    expected = next(chunk for chunk in original if chunk.symbol == "process_data")

    response = ReferenceService(store).find_references(
        DeclarationSelector(project=project.id, path="main.py", qualified_symbol="process_data")
    )

    assert response.selected.start_line == expected.start_line
    assert response.selected.end_line == expected.end_line


def test_reindex_replaces_whole_retrieval_without_mixing_generations(tmp_path: Path) -> None:
    indexer, store, search, project, _ = indexed_windows(tmp_path)
    old_ids = [chunk.chunk_id for chunk in store.list_chunks([project.id])]
    changed = SOURCE.replace("héllo", "changed π")
    (project.root / "main.py").write_bytes(changed.encode())

    assert indexer.index(project).errors == []
    expected = next(
        chunk
        for chunk in TreeSitterExtractor()
        .extract(Path("main.py"), "python", changed.encode())
        .chunks
        if chunk.symbol == "process_data"
    )
    for window in store.list_chunks([project.id]):
        if window.symbol == expected.symbol:
            assert search.get_chunk(window.chunk_id).content == expected.content
    for chunk_id in old_ids:
        with pytest.raises(CodeIndexingError) as caught:
            search.get_chunk(chunk_id)
        assert caught.value.code is ErrorCode.CHUNK_NOT_FOUND


def test_cached_windows_retrieve_current_original_offsets(tmp_path: Path) -> None:
    embedder = AdaptiveEmbedder()
    indexer, store, search, project, original = indexed_windows(
        tmp_path, cache=True, embedder=embedder
    )
    expected = next(chunk for chunk in original if chunk.symbol == "process_data")
    before = len(embedder.embedded_texts)
    insertion = "# inserted π 世界\n"
    (project.root / "main.py").write_bytes((insertion + SOURCE).encode())

    report = indexer.index(project)

    assert report.errors == []
    assert report.reused_segments > 1
    # The inserted comment changes the module candidate, but every function
    # window should reuse its embedding despite the shifted source offsets.
    assert all("symbol: process_data" not in text for text in embedder.embedded_texts[before:])
    for window in store.list_chunks([project.id]):
        if window.symbol == expected.symbol:
            retrieved = search.get_chunk(window.chunk_id)
            assert retrieved.content == expected.content
            assert retrieved.start_byte == expected.start_byte + len(insertion.encode())
            assert retrieved.start_line == expected.start_line + 1


def test_same_named_declarations_are_not_joined_by_retrieval(tmp_path: Path) -> None:
    source = SOURCE + "\n" + SOURCE.replace("ending_marker", "second_marker")
    _, store, search, project, original = indexed_windows(tmp_path, source)
    originals = [chunk for chunk in original if chunk.symbol == "process_data"]
    assert len(originals) == 2

    found = search.find_symbol("process_data", project.id)
    assert len(found.hits) == 2
    assert {search.get_chunk(hit.chunk_id).content for hit in found.hits} == {
        chunk.content for chunk in originals
    }
    with pytest.raises(CodeIndexingError) as caught:
        ReferenceService(store).find_references(
            DeclarationSelector(project=project.id, path="main.py", qualified_symbol="process_data")
        )
    assert caught.value.code is ErrorCode.AMBIGUOUS_SYMBOL
    # Preserve the existing outline policy for repeated names: the first
    # declaration is shown, without extending through an unrelated declaration.
    item = next(
        item
        for item in search.file_outline("main.py", project.id).items
        if item.symbol == "process_data"
    )
    assert item.end_line == originals[0].end_line


def test_extractor_parts_remain_separate_retrieval_units(tmp_path: Path) -> None:
    source = "def huge(user):\n" + "".join(f"    step_{i} = user.value + {i}\n" for i in range(300))
    _, store, search, project, original = indexed_windows(tmp_path, source, max_tokens=256)
    originals = {chunk.part_index: chunk for chunk in original if chunk.symbol == "huge"}
    assert len(originals) > 1

    for window in store.list_chunks([project.id]):
        expected = originals[window.part_index]
        retrieved = search.get_chunk(window.chunk_id)
        assert retrieved.content == expected.content
        assert (retrieved.start_byte, retrieved.end_byte) == (
            expected.start_byte,
            expected.start_byte + len(expected.content.encode()),
        )
        assert (
            source.encode()[retrieved.start_byte : retrieved.end_byte].decode() == expected.content
        )


@pytest.mark.parametrize("removed", [0, 1, -1])
def test_missing_windows_raise_instead_of_returning_incomplete_source(
    tmp_path: Path, removed: int
) -> None:
    _, store, search, project, _ = indexed_windows(tmp_path)
    windows = sorted(
        (chunk for chunk in store.list_chunks([project.id]) if chunk.symbol == "process_data"),
        key=lambda chunk: chunk.start_byte,
    )
    discarded = windows.pop(removed)
    table = store._project_tables(project.id).chunks
    table.delete(f"chunk_id = '{discarded.chunk_id}'")

    with pytest.raises(CodeIndexingError) as caught:
        search.get_chunk(windows[0].chunk_id)
    assert caught.value.code is ErrorCode.INDEX_INCOMPATIBLE


def test_inconsistent_overlaps_raise_instead_of_silently_joining_source(tmp_path: Path) -> None:
    _, store, search, project, _ = indexed_windows(tmp_path)
    windows = sorted(
        (chunk for chunk in store.list_chunks([project.id]) if chunk.symbol == "process_data"),
        key=lambda chunk: chunk.start_byte,
    )
    target = windows[1]
    changed = "X" + target.content[1:]
    table = store._project_tables(project.id).chunks
    table.update(where=f"chunk_id = '{target.chunk_id}'", values={"content": changed})

    with pytest.raises(CodeIndexingError) as caught:
        search.get_chunk(windows[0].chunk_id)
    assert caught.value.code is ErrorCode.INDEX_INCOMPATIBLE


class OrderedEmbedder(AdaptiveEmbedder):
    @staticmethod
    def _vector(text: str) -> list[float]:
        name = re.search(r"function(\d+)", text)
        index = int(name[1]) if name else 0
        return [1.0, index * 0.02, 0.0, 0.0]


@pytest.mark.parametrize("example", [False, True])
def test_window_duplicates_do_not_exhaust_the_search_result_limit(
    tmp_path: Path, example: bool
) -> None:
    source = "\n".join(
        f"def function{i}(user):\n"
        + "".join(f"    step_{j} = user.value + {j}\n" for j in range(30))
        for i in range(22)
    )
    _, _, search, project, _ = indexed_windows(tmp_path, source, embedder=OrderedEmbedder())

    response = (
        search.search_by_example(
            "def probe():\n    return 0", [project.id], language="python", limit=20
        )
        if example
        else search.search_code("ordered retrieval", [project.id], limit=20)
    )

    assert len(response.hits) == 20
    assert len({hit.qualified_symbol for hit in response.hits}) == 20


def _second_checkout(store: LanceStore, project: ProjectInfo) -> PartitionRef:
    """Register a second branch slot of *project* without building its partition."""
    first = store.active_partition(project.id)
    first_slot = store.get_slot(first.slot_id)
    assert first_slot is not None
    second_slot = first_slot.model_copy(
        update={
            "slot_id": "second-checkout",
            "partition_id": "slot-second-checkout",
            "selector_kind": "ref",
            "selector_value": "refs/heads/second-checkout",
            "state": "ready",
        }
    )
    store.upsert_slot(second_slot)
    return PartitionRef(
        project.id, second_slot.slot_id, second_slot.partition_id, first.activation_epoch
    )


def test_sibling_slot_without_window_columns_is_rebuilt_before_queries(tmp_path: Path) -> None:
    indexer, store, search, project, _ = indexed_windows(tmp_path)
    second = _second_checkout(store, project)
    # A sibling slot built before the window columns existed. Rebuilding the
    # first slot already stamped the shared registry row with this schema.
    source = store._project_tables(project.id)
    database = lancedb.connect(store.directory / "projects" / second.partition_id)
    database.create_table("files", source.files.to_arrow())
    database.create_table(
        "chunks", source.chunks.to_arrow().drop_columns(["source_start_byte", "source_end_byte"])
    )
    database.create_table("references", source.references.to_arrow())

    reason = store.incompatibility_reason(
        project.id, indexer.embedder.model_id, partition_id=second.partition_id
    )
    assert "chunk columns missing: source_start_byte, source_end_byte" in (reason or "")

    assert indexer.index(project, partition=second).errors == []
    assert (
        store.incompatibility_reason(
            project.id, indexer.embedder.model_id, partition_id=second.partition_id
        )
        is None
    )
    assert len(search.find_symbol("process_data", project.id, partition=second).hits) == 1


@pytest.mark.parametrize("example", [False, True])
def test_identical_windows_in_pinned_checkouts_do_not_exhaust_search_results(
    tmp_path: Path, example: bool
) -> None:
    source = "\n".join(
        f"def function{i}(user):\n"
        + "".join(f"    step_{j} = user.value + {j}\n" for j in range(43))
        for i in range(10)
    )
    _, store, search, project, _ = indexed_windows(tmp_path, source, embedder=OrderedEmbedder())
    first = store.active_partition(project.id)
    second = _second_checkout(store, project)
    # Equal source in separate slots has equal logical source metadata but
    # distinct chunk IDs, since physical slot identity participates in the ID.
    rows = store._project_tables(project.id).chunks.to_arrow().to_pylist()
    for row in rows:
        digest = hashlib.sha256(f"{second.slot_id}:{row['chunk_id']}".encode()).hexdigest()
        row["chunk_id"] = f"{project.id}:{digest}"
    store._project_tables(project.id, partition_id=second.partition_id).chunks.add(rows)
    store.ensure_indexes(project.id, partition_id=second.partition_id)
    partitions = {project.id: [first, second]}

    response = (
        search.search_by_example(
            "def probe():\n    return 0",
            [project.id],
            language="python",
            kinds=["function"],
            limit=8,
            partitions=partitions,
        )
        if example
        else search.search_code(
            "ordered retrieval",
            [project.id],
            kinds=["function"],
            limit=8,
            partitions=partitions,
        )
    )

    assert len(response.hits) == 8
    assert len({hit.qualified_symbol for hit in response.hits}) == 8
