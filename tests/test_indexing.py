import os
from pathlib import Path
from unittest.mock import patch

import pytest

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
