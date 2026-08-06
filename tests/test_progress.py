"""Live indexing progress: in-process listeners and the cross-process snapshot."""

import json
import time
from pathlib import Path

from test_indexing import RecordingEmbedder, make_indexer

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.progress import (
    IndexProgress,
    ProgressPublisher,
    progress_path,
    read_progress,
)
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


def _repo(tmp_path: Path, files: int = 3) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for number in range(files):
        (root / f"module_{number}.py").write_text(f"def answer_{number}():\n    return {number}\n")
    return root


def test_indexing_reports_file_counts_as_it_goes(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    report = indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    assert report.indexed_files == 3
    assert seen, "an index that takes any work must report progress"
    assert [progress.project_id for progress in seen] == [project.id] * len(seen)
    assert max(progress.files_seen for progress in seen) == 3
    assert max(progress.files_indexed for progress in seen) == 3
    assert max(progress.chunks_embedded for progress in seen) > 0
    assert seen[-1].phase == "committing"
    # File counts are a running total, never a per-update delta.
    assert [progress.files_seen for progress in seen] == sorted(
        progress.files_seen for progress in seen
    )


def test_a_second_run_knows_roughly_how_many_files_to_expect(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    seen: list[IndexProgress] = []

    indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    assert seen[0].files_total == 3
    assert seen[0].fraction is not None


def test_the_first_run_admits_it_has_no_total(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    assert seen[0].files_total is None
    assert seen[0].fraction is None
    assert "files" in seen[0].describe()


def test_reference_extraction_has_a_distinct_progress_description() -> None:
    progress = IndexProgress(project_id="project", phase="extracting_references")

    assert progress.describe() == "Extracting structural references"


def test_another_process_can_read_the_snapshot_and_it_is_gone_afterwards(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    embedder = RecordingEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    progress_directory = tmp_path / "progress"
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
        progress_directory=progress_directory,
    )
    published: list[IndexProgress | None] = []

    # Reading from the listener stands in for the separate process that polls the
    # file while the daemon indexes: it can only see what has been written.
    indexer.index(
        project,
        on_progress=lambda _: published.append(read_progress(progress_directory, project.id)),
    )

    assert any(snapshot is not None for snapshot in published)
    assert read_progress(progress_directory, project.id) is None
    assert not progress_path(progress_directory, project.id).exists()


def test_updates_are_throttled_but_the_forced_ones_always_land(tmp_path: Path) -> None:
    clock = iter([0.0, 0.05, 0.10, 0.15])
    seen: list[int] = []
    publisher = ProgressPublisher(
        "project",
        listener=lambda progress: seen.append(progress.files_seen),
        interval_seconds=1.0,
        clock=lambda: next(clock),
    )

    publisher.update(files_seen=1)
    publisher.update(files_seen=2)
    publisher.update(files_seen=3)
    publisher.update(files_seen=4, force=True)

    assert seen == [1, 4]


def test_a_snapshot_left_behind_by_a_dead_process_is_ignored(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stale = IndexProgress(project_id="abc", updated_at=time.time() - 3600)
    progress_path(tmp_path, "abc").write_text(stale.model_dump_json())

    assert read_progress(tmp_path, "abc") is None
    assert read_progress(tmp_path, "abc", stale_after_seconds=7200) is not None


def test_a_corrupt_snapshot_is_treated_as_absent(tmp_path: Path) -> None:
    progress_path(tmp_path, "abc").write_text("{not json")

    assert read_progress(tmp_path, "abc") is None


def test_the_snapshot_is_replaced_atomically(tmp_path: Path) -> None:
    publisher = ProgressPublisher("abc", directory=tmp_path)

    publisher.update(files_seen=1, force=True)
    publisher.update(files_seen=2, force=True)

    payload = json.loads(progress_path(tmp_path, "abc").read_text())
    assert payload["files_seen"] == 2
    assert list(tmp_path.iterdir()) == [progress_path(tmp_path, "abc")]
