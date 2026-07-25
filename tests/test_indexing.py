import itertools
import os
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest
from filelock import FileLock
from test_token_batching import fake_encode

from incode_mcp.embedding import (
    EmbeddedSegment,
    PassageCandidate,
    SegmentPlan,
    embed_planned_segments,
)
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.extractor import TreeSitterExtractor
from incode_mcp.indexing import Indexer
from incode_mcp.projects import initialize_project
from incode_mcp.scanner import SourceScanner
from incode_mcp.storage import LanceStore


class RecordingEmbedder:
    model_id = "test/code"
    dimension = 4

    def __init__(self) -> None:
        self.passage_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if any("RAISE_EMBEDDING" in text for text in texts):
            raise RuntimeError("embedding failed")
        self.passage_batches.append(texts)
        return [[float(len(text) % 7), 1.0, 2.0, 3.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text) % 7), 1.0, 2.0, 3.0]


class SessionEmbedder(RecordingEmbedder):
    def __enter__(self) -> "SessionEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def make_indexer(tmp_path: Path, embedder: RecordingEmbedder) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    return (
        Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=tmp_path / "locks",
        ),
        store,
    )


def test_indexer_skips_unchanged_and_metadata_only_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    first = indexer.index(project)
    original_read_bytes = Path.read_bytes

    def fail_if_source_is_read(path: Path) -> bytes:
        if path == source:
            raise AssertionError("unchanged source was read")
        return original_read_bytes(path)

    with patch.object(Path, "read_bytes", fail_if_source_is_read):
        second = indexer.index(project)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    metadata_only = indexer.index(project)

    assert first.indexed_files == 1
    assert first.parsed_files == 1
    assert len(embedder.passage_batches) == 1
    assert second.unchanged_files == 1
    assert second.parsed_files == 0
    assert second.embedded_chunks == 0
    assert metadata_only.metadata_only_files == 1
    assert metadata_only.parsed_files == 0
    assert len(embedder.passage_batches) == 1
    assert len(store.list_files(project.id)) == 1


def test_index_report_splits_duration_into_phases(tmp_path: Path) -> None:
    """Embedding cost must be separable from scan/parse/commit cost in the report."""

    class SlowEmbedder(RecordingEmbedder):
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            time.sleep(0.05)
            return super().embed_passages(texts)

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, SlowEmbedder())

    report = indexer.index(project)

    assert report.embed_duration_ms is not None
    assert report.embed_duration_ms >= 45
    phases = [
        report.scan_duration_ms,
        report.parse_duration_ms,
        report.embed_duration_ms,
        report.commit_duration_ms,
    ]
    assert all(phase is not None and phase >= 0 for phase in phases)
    # The phases partition the indexing work, so together they cannot exceed the
    # total, which also covers lock acquisition and project bookkeeping.
    assert sum(phase or 0 for phase in phases) <= report.duration_ms


def test_indexer_replaces_changed_files_and_removes_deleted_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 41\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    original_ids = {chunk.chunk_id for chunk in store.list_chunks([project.id])}

    source.write_text("def renamed_answer():\n    return 42\n")
    changed = indexer.index(project)
    changed_ids = {chunk.chunk_id for chunk in store.list_chunks([project.id])}
    source.unlink()
    removed = indexer.index(project)

    assert changed.indexed_files == 1
    assert changed_ids != original_ids
    assert removed.removed_files == 1
    assert store.list_chunks([project.id]) == []


def test_failed_changed_file_preserves_previous_chunks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def stable():\n    return 1\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    original = store.list_chunks([project.id])

    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    report = indexer.index(project)

    assert len(report.errors) == 1
    assert store.list_chunks([project.id]) == original
    # A failed file is recorded with its current size/mtime, so subsequent
    # runs skip it instead of re-reading, re-parsing, and re-embedding it.
    batches = len(embedder.passage_batches)
    second = indexer.index(project)
    assert len(embedder.passage_batches) == batches
    assert second.unchanged_files == 1
    assert store.list_chunks([project.id]) == original


def test_global_index_lock_serializes_different_projects(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())

    lock = FileLock(tmp_path / "locks" / "index-global.lock")
    with lock, pytest.raises(IncodeError) as caught:
        indexer.index(project)

    assert caught.value.code is ErrorCode.INDEX_BUSY


def test_indexer_uses_passage_worker_session_when_configured(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    parent_embedder = RecordingEmbedder()
    worker_embedder = SessionEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=parent_embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=parent_embedder,
        lock_directory=tmp_path / "locks",
        passage_session_factory=lambda: worker_embedder,
    )

    indexer.index(project)

    assert worker_embedder.passage_batches
    assert parent_embedder.passage_batches == []


def test_force_reindexes_previously_failed_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    failed = indexer.index(project)
    assert len(failed.errors) == 1

    source.write_text("def stable():\n    return 3\n")
    forced = indexer.index(project, force=True)

    assert forced.errors == []
    assert forced.indexed_files == 1
    assert store.list_files(project.id)[0].has_errors is False


def test_unexpected_index_failure_marks_project_error(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("x = 1\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    with (
        patch.object(SourceScanner, "scan", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        indexer.index(project)

    assert store.project_state(project.id) == "error"


def test_model_unavailable_marks_project_error(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    with (
        patch.object(
            embedder,
            "embed_passages",
            side_effect=IncodeError(ErrorCode.MODEL_UNAVAILABLE, "model missing"),
        ),
        pytest.raises(IncodeError) as raised,
    ):
        indexer.index(project)

    assert raised.value.code is ErrorCode.MODEL_UNAVAILABLE
    assert store.project_state(project.id) == "error"


def test_store_keeps_projects_isolated_and_removal_does_not_touch_markers(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    projects = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "main.py").write_text(f"def {name}():\n    return '{name}'\n")
        project = initialize_project(root)
        projects.append(project)
        indexer.index(project)

    store.remove_project(projects[0].id)

    assert {project.id for project in store.list_projects()} == {projects[1].id}
    assert {chunk.project_id for chunk in store.list_chunks()} == {projects[1].id}
    assert (projects[0].root / ".ci-mcp" / "project.toml").exists()


class ResourceLimitEmbedder(RecordingEmbedder):
    """Trips the memory ceiling on the second file, as a busy machine would."""

    def __init__(self) -> None:
        super().__init__()
        self.files_seen = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.files_seen += 1
        if self.files_seen == 2:
            raise IncodeError(
                ErrorCode.INDEX_RESOURCE_LIMIT, "Indexing exceeded its memory ceiling"
            )
        return super().embed_passages(texts)


def test_resource_limit_aborts_instead_of_poisoning_the_file_record(tmp_path: Path) -> None:
    """A transient ceiling trip must not mark the file as up to date."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("value = 1\n")
    (root / "b.py").write_text("value = 2\n")
    project = initialize_project(root)
    embedder = ResourceLimitEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    with pytest.raises(IncodeError) as caught:
        indexer.index(project)

    assert caught.value.code is ErrorCode.INDEX_RESOURCE_LIMIT
    # The file that failed was never recorded, so a later run retries it rather
    # than skipping it as unchanged forever.
    recorded = {record.path for record in store.list_files(project.id)}
    assert len(recorded) < 2

    healthy, _ = make_indexer(tmp_path, RecordingEmbedder())
    report = healthy.index(project)

    assert report.errors == []
    assert len(store.list_files(project.id)) == 2
    assert store.project_state(project.id) == "ready"


class WindowingEmbedder(RecordingEmbedder):
    """A double that windows candidates the way the real worker does.

    It runs the production planner over the deterministic tokenizer from
    ``test_token_batching``, so the offsets under test are the ones the real
    path produces, without loading a model.
    """

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        planned = embed_planned_segments(fake_encode, self.embed_passages, candidates, plan)
        return [
            [
                EmbeddedSegment(window.start_char, window.end_char, window.token_count, vector)
                for window, vector in segments
            ]
            for segments in planned
        ]


def make_windowing_indexer(
    tmp_path: Path, embedder: RecordingEmbedder, plan: SegmentPlan
) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    return (
        Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=tmp_path / "locks",
            segment_plan=plan,
        ),
        store,
    )


DENSE_SOURCE = (
    "def answer():\n"
    "    total = 0\n"
    "    for index in range(10):\n"
    "        total = total + index\n"
    "    return total\n"
)


def test_a_token_dense_chunk_is_split_into_several_stored_chunks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(DENSE_SOURCE)
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    report = indexer.index(project)
    chunks = store.list_chunks([project.id])

    assert report.errors == []
    assert len(chunks) > 1
    assert {chunk.qualified_symbol for chunk in chunks} == {"answer"}


def test_windowed_chunk_offsets_still_slice_the_original_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(DENSE_SOURCE)
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    indexer.index(project)
    source = (root / "main.py").read_bytes()

    for chunk in store.list_chunks([project.id]):
        assert source[chunk.start_byte : chunk.end_byte].decode("utf-8") == chunk.content
        assert source[: chunk.start_byte].decode("utf-8").count("\n") + 1 == chunk.start_line
        assert chunk.end_line == chunk.start_line + chunk.content.count("\n")


def test_every_window_keeps_the_context_header_and_identifier_tail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(DENSE_SOURCE)
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    indexer.index(project)
    chunks = store.list_chunks([project.id])

    for chunk in chunks:
        assert chunk.embedding_text.startswith("language: python\npath: main.py\n")
        assert "symbol: answer" in chunk.embedding_text
        assert chunk.embedding_text.endswith(chunk.content)
        assert chunk.search_text.startswith(chunk.embedding_text)
        assert chunk.search_text.endswith("main py answer answer")


def test_windows_cover_the_symbol_without_dropping_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(DENSE_SOURCE)
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    indexer.index(project)
    chunks = sorted(store.list_chunks([project.id]), key=lambda chunk: chunk.start_byte)
    source = (root / "main.py").read_bytes()

    covered = source[chunks[0].start_byte : chunks[0].end_byte]
    for previous, chunk in itertools.pairwise(chunks):
        assert chunk.start_byte <= previous.end_byte
        covered += source[max(chunk.start_byte, previous.end_byte) : chunk.end_byte]
    assert covered == source[chunks[0].start_byte : chunks[-1].end_byte]
    assert covered.decode("utf-8").strip() == DENSE_SOURCE.strip()


def test_a_chunk_within_the_budget_is_stored_unchanged(tmp_path: Path) -> None:
    """Windowing must be invisible to files that never exceed the budget."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    windowing, windowed_store = make_windowing_indexer(
        tmp_path / "windowed", WindowingEmbedder(), SegmentPlan(max_tokens=1_024)
    )
    plain, plain_store = make_indexer(tmp_path / "plain", RecordingEmbedder())

    windowing.index(project)
    plain.index(project)

    def comparable(store: LanceStore) -> list[tuple[object, ...]]:
        return sorted(
            (chunk.chunk_id, chunk.start_byte, chunk.end_byte, chunk.content, chunk.search_text)
            for chunk in store.list_chunks([project.id])
        )

    assert comparable(windowed_store) == comparable(plain_store)


class UnplannableEmbedder(WindowingEmbedder):
    """Rejects one file's candidates the way an exploding window plan would."""

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        if any("REJECT" in candidate.content for candidate in candidates):
            raise ValueError("Token planning exceeded 16 windows")
        return super().plan_and_embed(candidates, plan)


def test_an_unplannable_file_is_charged_to_the_file_not_the_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "good.py").write_text("value = 1\n")
    (root / "bad.py").write_text("REJECT = 'x'\n")
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, UnplannableEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    report = indexer.index(project)

    # The run completes and the healthy file is indexed; only the file that
    # could not be planned is recorded as an issue.
    assert [issue.path for issue in report.errors] == ["bad.py"]
    assert store.project_state(project.id) == "partial"
    assert {chunk.path for chunk in store.list_chunks([project.id])} == {"good.py"}
