"""Durable audit history: bounded SQLite storage for indexing runs."""

import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil
import pytest

from code_indexing_mcp.history import (
    MAX_RUN_AGE_DAYS,
    MAX_RUNS_PER_PROJECT,
    HistoryStore,
)
from code_indexing_mcp.models import IndexIssue, RunAudit


def _dead_pid() -> int:
    """A pid that existed but whose process has certainly exited."""
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    assert not psutil.pid_exists(process.pid)
    return process.pid


def _audit(run_id: str, started_at: str | None = None, *, pid: int | None = None) -> RunAudit:
    return RunAudit(
        run_id=run_id,
        project_id="project-1",
        trigger="manual",
        server_version="0.4.0",
        git_revision="abc1234",
        model_id="test/tiny",
        schema_version=2,
        scan_config_hash="feedface",
        force=True,
        pid=os.getpid() if pid is None else pid,
        started_at=started_at or datetime.now(UTC).isoformat(),
    )


def test_a_completed_run_round_trips_all_audit_fields(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    audit = _audit("run-1")

    store.begin(audit)
    store.finish(
        "run-1",
        state="completed",
        finished_at=datetime.now(UTC).isoformat(),
        phase_durations={"scan": 10, "embed": 40},
        eligible_files=3,
        changed_files=2,
        unchanged_files=1,
        parsed_files=2,
        failed_files=0,
        removed_files=0,
        skipped_total=5,
        chunks_extracted=4,
        chunks_embedded=4,
        chunks_staged=4,
        staged_bytes=1024,
        bytes_read=2048,
        skip_reasons={"unsupported": 3, "binary": 2},
        errors=[IndexIssue(path="a.py", message="boom")],
        skipped_samples=["b.txt", "c.bin"],
        embedding_backend="cpu",
        worker_used=True,
        storage_before={"files": 1, "chunks": 2, "references": 3},
        storage_after={"files": 2, "chunks": 4, "references": 5},
    )

    page = store.list_runs("project-1")
    assert page.project is None
    assert page.next_cursor is None
    run = page.runs[0]
    assert run.run_id == "run-1"
    assert run.trigger == "manual"
    assert run.server_version == "0.4.0"
    assert run.git_revision == "abc1234"
    assert run.model_id == "test/tiny"
    assert run.schema_version == 2
    assert run.scan_config_hash == "feedface"
    assert run.force is True
    assert run.state == "completed"
    assert run.phase_durations == {"scan": 10, "embed": 40}
    assert run.eligible_files == 3
    assert run.changed_files == 2
    assert run.unchanged_files == 1
    assert run.skipped_total == 5
    assert run.skip_reasons == {"unsupported": 3, "binary": 2}
    assert run.errors == [IndexIssue(path="a.py", message="boom")]
    assert run.skipped_samples == ["b.txt", "c.bin"]
    assert run.worker_used is True
    assert run.storage_before == {"files": 1, "chunks": 2, "references": 3}
    assert run.storage_after == {"files": 2, "chunks": 4, "references": 5}
    assert run.project_id == "project-1"
    assert run.pid == os.getpid()


def test_unknown_finish_fields_are_rejected(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("run-1"))

    with pytest.raises(ValueError, match="unknown audit finish fields"):
        store.finish("run-1", made_up_column=1)


def test_finish_with_no_updates_is_rejected(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("run-1"))

    with pytest.raises(ValueError, match="at least one field"):
        store.finish("run-1")


def test_a_malformed_cursor_is_rejected(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("run-1"))

    with pytest.raises(ValueError, match="invalid history cursor"):
        store.list_runs("project-1", cursor="not-a-cursor")


def test_history_is_pruned_to_a_bounded_window_per_project(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    for number in range(MAX_RUNS_PER_PROJECT + 5):
        run_id = f"run-{number:04d}"
        started = datetime.now(UTC) - timedelta(minutes=MAX_RUNS_PER_PROJECT - number)
        store.begin(_audit(run_id, started.isoformat()))
        store.finish(run_id, state="completed", finished_at=datetime.now(UTC).isoformat())

    page = store.list_runs("project-1", limit=1000)
    assert len(page.runs) == MAX_RUNS_PER_PROJECT
    # The newest runs survive; the oldest are pruned.
    assert page.runs[0].run_id == "run-0104"
    assert all(run.run_id != "run-0000" for run in page.runs)


def test_history_is_pruned_by_age_in_the_same_transaction(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    old = datetime.now(UTC) - timedelta(days=MAX_RUN_AGE_DAYS + 1)
    store.begin(_audit("old-run", old.isoformat()))
    store.finish("old-run", state="completed", finished_at=datetime.now(UTC).isoformat())
    store.begin(_audit("new-run"))
    store.finish("new-run", state="completed", finished_at=datetime.now(UTC).isoformat())

    assert {run.run_id for run in store.list_runs("project-1").runs} == {"new-run"}


def test_pagination_returns_an_opaque_cursor(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    for number in range(5):
        run_id = f"run-{number}"
        started = datetime.now(UTC) - timedelta(minutes=5 - number)
        store.begin(_audit(run_id, started.isoformat()))
        store.finish(run_id, state="completed", finished_at=datetime.now(UTC).isoformat())

    first = store.list_runs("project-1", limit=2)
    assert [run.run_id for run in first.runs] == ["run-4", "run-3"]
    assert first.next_cursor is not None

    second = store.list_runs("project-1", limit=2, cursor=first.next_cursor)
    assert [run.run_id for run in second.runs] == ["run-2", "run-1"]
    assert second.next_cursor is not None

    third = store.list_runs("project-1", limit=2, cursor=second.next_cursor)
    assert [run.run_id for run in third.runs] == ["run-0"]
    assert third.next_cursor is None


def test_a_killed_process_leaves_a_running_row_that_startup_marks_interrupted(
    tmp_path: Path,
) -> None:
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("run-1", pid=_dead_pid()))
    assert store.list_runs("project-1").runs[0].state == "running"

    # A fresh store instance stands in for the next process start.
    restarted = HistoryStore(tmp_path / "history")
    restarted.mark_interrupted()

    assert store.list_runs("project-1").runs[0].state == "interrupted"


def test_mark_interrupted_never_touches_runs_owned_by_live_processes(
    tmp_path: Path,
) -> None:
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("live-run", pid=os.getpid()))

    # A fresh store instance stands in for a second process starting while
    # the first process's run is still in flight.
    restarted = HistoryStore(tmp_path / "history")
    restarted.mark_interrupted()

    assert store.list_runs("project-1").runs[0].state == "running"


def test_mark_interrupted_distinguishes_dead_and_live_owners(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("dead-run", pid=_dead_pid()))
    store.begin(_audit("live-run", pid=os.getpid()))

    store.mark_interrupted()

    states = {run.run_id: run.state for run in store.list_runs("project-1").runs}
    assert states == {"dead-run": "interrupted", "live-run": "running"}


def test_recent_returns_a_compact_summary_and_nothing_else(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    assert store.recent("project-1") is None

    store.begin(_audit("run-1"))
    store.finish(
        "run-1",
        state="completed",
        finished_at=datetime.now(UTC).isoformat(),
        eligible_files=7,
        changed_files=2,
        failed_files=1,
        skipped_total=3,
        chunks_embedded=9,
    )

    summary = store.recent("project-1")
    assert summary is not None
    assert summary.run_id == "run-1"
    assert summary.trigger == "manual"
    assert summary.state == "completed"
    assert summary.eligible_files == 7
    assert summary.changed_files == 2
    assert summary.failed_files == 1
    assert summary.skipped_total == 3
    assert summary.chunks_embedded == 9
    assert summary.duration_ms >= 0


def test_recent_skips_running_and_interrupted_stubs(tmp_path: Path) -> None:
    """``last_run`` is the most recent *finished* run: a newer in-flight row
    (covered by live progress) or a crash's interrupted stub, both carrying
    only zeroed counters, must not hide the completed run's summary."""
    store = HistoryStore(tmp_path / "history")
    store.begin(_audit("run-1"))
    store.finish(
        "run-1",
        state="completed",
        finished_at=datetime.now(UTC).isoformat(),
        eligible_files=7,
    )

    later = (datetime.now(UTC) + timedelta(seconds=1)).isoformat()
    store.begin(_audit("run-2", later))
    summary = store.recent("project-1")
    assert summary is not None and summary.run_id == "run-1"

    store.begin(_audit("run-3", later, pid=_dead_pid()))
    store.mark_interrupted()
    summary = store.recent("project-1")
    assert summary is not None and summary.run_id == "run-1"


def test_concurrent_writers_do_not_corrupt_history(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    errors: list[Exception] = []

    def writer(number: int) -> None:
        try:
            for index in range(40):
                run_id = f"w{number}-{index}"
                store.begin(_audit(run_id))
                store.finish(run_id, state="completed", finished_at=datetime.now(UTC).isoformat())
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(number,)) for number in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    page = store.list_runs("project-1", limit=1000)
    assert len(page.runs) == MAX_RUNS_PER_PROJECT
    assert len({run.run_id for run in page.runs}) == len(page.runs)


def test_the_database_uses_wal_mode(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history")
    connection = sqlite3.connect(store.path)
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    connection.close()
    assert journal_mode == "wal"
