"""Live indexing progress: in-process listeners and the cross-process snapshot."""

import json
import time
from pathlib import Path

import pytest
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
    assert [progress.run_id for progress in seen] == [seen[0].run_id] * len(seen)
    assert max(progress.candidates_seen for progress in seen) == 3
    assert max(progress.eligible_files for progress in seen) == 3
    assert max(progress.changed_files for progress in seen) == 3
    assert max(progress.chunks_embedded for progress in seen) > 0
    assert seen[-1].phase == "committing"
    assert seen[-1].candidates_total == 3
    assert seen[-1].parsed_files == 3
    assert seen[-1].failed_files == 0
    assert seen[-1].bytes_read > 0
    assert seen[-1].chunks_extracted > 0
    # Counts are a running total, never a per-update delta.
    assert [progress.candidates_seen for progress in seen] == sorted(
        progress.candidates_seen for progress in seen
    )
    # A candidate count is never compared with an eligible-file total: the
    # denominator of a progress fraction is always candidate-scoped.
    assert all(progress.fraction is None or progress.fraction <= 1.0 for progress in seen)


def test_observed_119_eligible_plus_1367_skipped_never_reads_as_1486_over_119() -> None:
    """The review-measured case: 119 eligible files, 1,367 skipped candidates.

    The old contract counted every candidate in the seen counter and only the
    eligible files in the total, so describe() could print ``1486/~119``.
    """
    progress = IndexProgress(
        project_id="p",
        run_id="r",
        candidates_seen=1486,
        candidates_total=1486,
        eligible_files=119,
        skipped_total=1367,
    )

    description = progress.describe()
    assert "/~119" not in description
    assert progress.fraction == 1.0
    # Without a candidate total the fraction must stay unknown rather than
    # falling back to an eligible-file denominator.
    ambiguous = progress.model_copy(update={"candidates_total": None})
    assert ambiguous.fraction is None
    assert "1486 candidates, 119 eligible" in ambiguous.describe()


def test_the_first_run_admits_it_has_no_total(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    assert seen[0].candidates_total is None
    assert seen[0].fraction is None
    counting = next(progress for progress in seen if progress.candidates_seen)
    assert "candidates" in counting.describe()


def test_skipped_candidates_are_aggregated_by_reason(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "notes.txt").write_text("unsupported")
    (root / "binary.py").write_bytes(b"a\x00b")
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    last = seen[-1]
    assert last.skipped_total == 2
    assert last.skipped_by_reason == {"unsupported": 1, "binary": 1}


def test_reference_extraction_has_a_distinct_progress_description() -> None:
    progress = IndexProgress(
        project_id="project",
        phase="extracting_references",
        run_id="abc123def456",
        trigger="reference-backfill",
        candidates_seen=2,
        candidates_total=5,
        eligible_files=9,
        changed_files=1,
        current_path="pkg/mod.py",
    )

    text = progress.describe()
    assert text.startswith("Extracting structural references")
    assert "2/~5 candidates" in text
    assert "1 changed" in text
    assert "pkg/mod.py" in text
    assert "reference-backfill" in text


def test_describe_carries_counters_in_every_phase() -> None:
    base = {
        "project_id": "project",
        "run_id": "abc123def456",
        "trigger": "watcher",
        "candidates_seen": 10,
        "candidates_total": 20,
        "eligible_files": 8,
        "changed_files": 5,
        "parsed_files": 5,
        "chunks_extracted": 12,
        "chunks_embedded": 9,
        "chunks_staged": 9,
        "current_path": "pkg/mod_001.py",
        "slot_id": "slot-1abcdef",
        "selector": "ref:refs/heads/main",
    }
    for phase in ("scanning", "embedding", "extracting_references", "committing"):
        text = IndexProgress(**base, phase=phase).describe()  # type: ignore[arg-type]
        for expected in (
            "5 changed",
            "5 parsed",
            "12 chunks extracted",
            "9 chunks embedded",
            "9 chunks staged",
            "pkg/mod_001.py",
            "abc123de",
            "watcher",
            "slot-1ab",
            "ref:refs/heads/main",
        ):
            assert expected in text, (phase, text)
    # A run that has not seen anything yet still names the wait.
    assert IndexProgress(project_id="project").describe() == "Scanning for changed files"


def test_embedding_progress_advances_with_every_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One file with three chunks: small enough to flush once, big enough for
    # two groups at COUNT=2, so the middle of the phase is one flush's groups
    # rather than several flushes' bracket updates.
    monkeypatch.setattr("code_indexing_mcp.indexing.CANDIDATE_GROUP_COUNT", 2)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text(
        "def answer_0():\n    return 0\n\n\n"
        "def answer_1():\n    return 1\n\n\n"
        "def answer_2():\n    return 2\n"
    )
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    report = indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    assert seen[-1].chunks_extracted >= 3, "the fixture must span at least two groups"
    embedded = [progress.chunks_embedded for progress in seen if progress.phase == "embedding"]
    assert embedded, "the longest phase must publish progress"
    # Initial, per-group, final: a first-to-last comparison alone cannot tell
    # per-group publishing from a frozen middle, so demand distinct mid-flush
    # levels too.
    assert len(embedded) > 2, "each embedding group must publish its running total"
    assert len(set(embedded)) > 2, "each embedding group must advance the running total"
    assert embedded == sorted(embedded), "running totals never move backwards"
    assert embedded[0] < embedded[-1]
    assert embedded[-1] == report.embedded_chunks
    assert report.embedded_chunks > 0
    staged = [progress.chunks_staged for progress in seen if progress.phase == "embedding"]
    assert len(staged) > 2, "each embedding group must publish its running total"
    assert len(set(staged)) > 2, "each embedding group must advance the running total"
    assert staged == sorted(staged), "running totals never move backwards"
    assert staged[0] < staged[-1]
    assert staged[-1] == report.chunks_staged
    # Only the per-group update names the file in flight: the flush brackets
    # publish with no current path.
    assert any(
        progress.current_path is not None for progress in seen if progress.phase == "embedding"
    ), "each embedding group must publish the file in flight"


def test_mid_flush_totals_exclude_files_that_already_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailSecondBatchEmbedder(RecordingEmbedder):
        """Banks one batch, then fails every later batch like a dying worker."""

        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            if self.passage_batches:
                raise RuntimeError("embedding failed")
            return super().embed_passages(texts)

    # One file, two chunks, one chunk per group: the first group banks a
    # chunk, the second group fails the file, and the file's banked chunk
    # must leave the published total the moment the failure lands.
    monkeypatch.setattr("code_indexing_mcp.indexing.CANDIDATE_GROUP_COUNT", 1)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text(
        "def answer_0():\n    return 0\n\n\ndef answer_1():\n    return 1\n"
    )
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, FailSecondBatchEmbedder())
    seen: list[IndexProgress] = []

    report = indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    assert report.embedded_chunks == 0
    assert report.errors, "the second group must fail the file"
    embedded = [progress.chunks_embedded for progress in seen if progress.phase == "embedding"]
    assert max(embedded) > 0, "the first group must bank a chunk before the failure"
    assert embedded[-2] == 0, "a failed file's banked chunks must leave the published total"


def test_committing_snapshot_carries_run_totals(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    report = indexer.index(project, on_progress=lambda progress: seen.append(progress.model_copy()))

    last = seen[-1]
    assert last.phase == "committing"
    assert last.changed_files == report.indexed_files == 3
    assert last.chunks_embedded == report.embedded_chunks
    assert report.embedded_chunks > 0
    assert last.chunks_staged == report.chunks_staged
    assert report.chunks_staged > 0


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
    # The first tick is consumed for the initial phase anchor.
    clock = iter([0.0, 0.05, 0.10, 0.15, 0.20])
    seen: list[int] = []
    publisher = ProgressPublisher(
        "project",
        listener=lambda progress: seen.append(progress.candidates_seen),
        interval_seconds=1.0,
        clock=lambda: next(clock),
    )

    publisher.update(candidates_seen=1)
    publisher.update(candidates_seen=2)
    publisher.update(candidates_seen=3)
    publisher.update(candidates_seen=4, force=True)

    assert seen == [1, 4]


def test_a_phase_change_stamps_phase_started_at_and_publishes_immediately() -> None:
    publisher = ProgressPublisher("project", listener=lambda _: None, interval_seconds=1.0)
    before = publisher.state.phase_started_at

    publisher.update(phase="embedding", candidates_seen=1, force=True)

    assert publisher.state.phase == "embedding"
    assert publisher.state.phase_started_at > before


def test_every_run_carries_a_run_id_and_trigger(tmp_path: Path) -> None:
    project = initialize_project(_repo(tmp_path))
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    seen: list[IndexProgress] = []

    indexer.index(
        project,
        trigger="watcher",
        on_progress=lambda progress: seen.append(progress.model_copy()),
    )

    assert seen
    assert all(progress.run_id for progress in seen)
    assert [progress.trigger for progress in seen] == ["watcher"] * len(seen)


def test_retained_snapshots_do_not_change_when_the_source_dict_is_mutated() -> None:
    publisher = ProgressPublisher("project", listener=lambda _: None, interval_seconds=0.0)
    reasons = {"binary": 1}
    publisher.update(skipped_by_reason=reasons, force=True)
    snapshot = publisher.state.model_copy()

    reasons["binary"] = 99
    publisher.update(skipped_by_reason=reasons, force=True)

    assert snapshot.skipped_by_reason == {"binary": 1}
    assert publisher.state.skipped_by_reason == {"binary": 99}


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

    publisher.update(candidates_seen=1, force=True)
    publisher.update(candidates_seen=2, force=True)

    payload = json.loads(progress_path(tmp_path, "abc").read_text())
    assert payload["candidates_seen"] == 2
    assert list(tmp_path.iterdir()) == [progress_path(tmp_path, "abc")]


def test_slot_identity_is_carried_on_every_published_snapshot() -> None:
    seen: list[object] = []
    publisher = ProgressPublisher(
        "project",
        listener=seen.append,
        interval_seconds=0.0,
        slot_id="slot-a",
        selector="ref:refs/heads/main",
        expected_head="0" * 40,
        activation_epoch=3,
    )

    publisher.update(phase="scanning", candidates_seen=1)

    assert [(p.slot_id, p.selector, p.expected_head, p.activation_epoch) for p in seen] == [
        ("slot-a", "ref:refs/heads/main", "0" * 40, 3)
    ]
