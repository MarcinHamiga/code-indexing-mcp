from __future__ import annotations

import os
import signal
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest
from support import DeterministicEmbedder

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import DeclarationSelector, IndexProgress
from code_indexing_mcp.staging import StagingJob
from code_indexing_mcp.storage import LanceStore


def _make_paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")


def _write_python_corpus(root: Path, file_count: int, functions_per_file: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for file_idx in range(file_count):
        lines = [f"GLOBAL_{file_idx} = {file_idx}\n\n"]
        for fn_idx in range(functions_per_file):
            lines.append(
                f"def compute_{file_idx}_{fn_idx}(value: int) -> int:\n"
                f"    total = value + GLOBAL_{file_idx}\n"
                f"    if total > 0:\n"
                f"        return helper_step_{file_idx}(total)\n"
                f"    return total\n\n"
            )
        lines.append(f"def helper_step_{file_idx}(x: int) -> int:\n    return x + 1\n")
        (root / f"mod_{file_idx:03d}.py").write_text("".join(lines))


def test_index_worker_indexes_backfills_streams_progress_and_closes(tmp_path: Path) -> None:
    root = tmp_path / "project"
    _write_python_corpus(root, file_count=4, functions_per_file=3)
    app = Application(
        _make_paths(tmp_path),
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        project = app.init_project(root)
        worker_pid = app._get_index_worker().ping()
        assert worker_pid != os.getpid()

        updates: list[IndexProgress] = []
        report = app.index_project(project.id, on_progress=updates.append)

        assert report.indexed_files == 4
        assert updates
        assert app._get_index_worker().pid == worker_pid

        # Parent reads worker commits immediately via read_consistency_interval=0.
        found = app.find_symbol("compute_0_0", project.id)
        assert len(found.hits) == 1
        hits = app.search_code("compute_0_0", projects=[project.id])
        assert hits.hits

        backfill = app.ensure_reference_index(project.id)
        assert backfill.stale_paths == []
        assert backfill.incomplete_paths == []
    finally:
        app.close()

    assert app._index_worker is None


def test_index_worker_preserves_code_indexing_error(tmp_path: Path) -> None:
    from filelock import FileLock

    paths = _make_paths(tmp_path)
    app = Application(
        paths,
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        root = tmp_path / "project"
        root.mkdir()
        (root / "main.py").write_text("def hello():\n    return 1\n")
        project = app.init_project(root)
        lock_file = paths.data / "locks" / f"{project.id}.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_file), pytest.raises(CodeIndexingError) as exc_info:
            app.index_project(project.id, wait_for_lock=False)
        assert exc_info.value.code == ErrorCode.INDEX_BUSY
    finally:
        app.close()


def test_search_and_find_symbol_stay_responsive_during_worker_index_run(tmp_path: Path) -> None:
    """F1 gate: query p50 during an active index run stays <= 2x idle p50."""
    query_repo = tmp_path / "query_repo"
    build_repo = tmp_path / "build_repo"
    _write_python_corpus(query_repo, file_count=4, functions_per_file=4)
    _write_python_corpus(build_repo, file_count=25, functions_per_file=20)

    app = Application(
        _make_paths(tmp_path),
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        query_project = app.init_project(query_repo)
        build_project = app.init_project(build_repo)
        app.index_project(query_project.id)
        app.index_project(build_project.id)

        # Warm query paths before sampling idle baseline.
        for _ in range(3):
            app.search_code("compute_0_0", projects=[query_project.id])
            app.find_symbol("compute_0_0", query_project.id)

        idle_samples: list[float] = []
        for _ in range(15):
            t0 = time.perf_counter()
            assert app.search_code("compute_0_0", projects=[query_project.id]).hits
            assert app.find_symbol("compute_0_0", query_project.id).hits
            idle_samples.append(time.perf_counter() - t0)
        idle_p50 = statistics.median(idle_samples)

        busy_samples: list[float] = []
        indexing_done = threading.Event()

        def run_index() -> None:
            try:
                app.index_project(build_project.id, force=True)
            finally:
                indexing_done.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_index)
            while not indexing_done.is_set() and len(busy_samples) < 40:
                t0 = time.perf_counter()
                assert app.search_code("compute_0_0", projects=[query_project.id]).hits
                assert app.find_symbol("compute_0_0", query_project.id).hits
                busy_samples.append(time.perf_counter() - t0)
            future.result()

        assert len(busy_samples) >= 3
        busy_p50 = statistics.median(busy_samples)
        assert busy_p50 <= max(idle_p50 * 2.0, idle_p50 + 0.015), (
            f"idle_p50={idle_p50 * 1000:.1f}ms, busy_p50={busy_p50 * 1000:.1f}ms"
        )
    finally:
        app.close()


def _remove_reference_generation(store: LanceStore, project_id: str) -> None:
    for record in store.list_files(project_id):
        store.replace_files_from_arrow(
            project_id,
            files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
            chunk_batches=(),
            reference_batches=[
                (
                    [record.file_id],
                    pa.Table.from_batches([], schema=LanceStore.reference_arrow_schema()),
                )
            ],
        )


def test_index_worker_wait_for_lock_false_raises_index_busy_immediately(tmp_path: Path) -> None:
    """Finding 1a: concurrent wait_for_lock=False fails immediately with INDEX_BUSY."""
    root_a = tmp_path / "project_a"
    root_b = tmp_path / "project_b"
    _write_python_corpus(root_a, file_count=4, functions_per_file=3)
    _write_python_corpus(root_b, file_count=2, functions_per_file=2)

    app = Application(
        _make_paths(tmp_path),
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        project_a = app.init_project(root_a)
        project_b = app.init_project(root_b)
        started = threading.Event()
        release = threading.Event()

        def hold_worker(_progress: IndexProgress) -> None:
            started.set()
            assert release.wait(timeout=10.0)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                app.index_project,
                project_a.id,
                on_progress=hold_worker,
            )
            assert started.wait(timeout=10.0)
            worker = app._get_index_worker()
            assert worker.active_project_id == project_a.id

            t0 = time.monotonic()
            with pytest.raises(CodeIndexingError) as exc_info:
                app.index_project(project_b.id, wait_for_lock=False)
            elapsed = time.monotonic() - t0
            assert exc_info.value.code is ErrorCode.INDEX_BUSY
            assert elapsed < 2.0

            target_b = app._resolve_active_target(project_b)
            t0 = time.monotonic()
            with pytest.raises(CodeIndexingError) as backfill_exc:
                worker.backfill_references(target_b, wait_for_lock=True, lock_timeout=0.05)
            assert backfill_exc.value.code is ErrorCode.INDEX_BUSY
            assert time.monotonic() - t0 < 2.0

            release.set()
            report = future.result(timeout=10.0)
            assert report.indexed_files == 4
            assert worker.active_project_id is None
    finally:
        app.close()


def test_settled_references_and_cross_project_backfill_do_not_stall_on_worker(
    tmp_path: Path,
) -> None:
    """Finding 1b: settled queries skip worker and cross-project backfill uses disposable worker."""
    root_a = tmp_path / "project_a"
    root_b = tmp_path / "project_b"
    _write_python_corpus(root_a, file_count=3, functions_per_file=2)
    _write_python_corpus(root_b, file_count=3, functions_per_file=2)

    app = Application(
        _make_paths(tmp_path),
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        project_a = app.init_project(root_a)
        project_b = app.init_project(root_b)
        app.index_project(project_a.id)
        helper_hit = app.find_symbol("helper_step_0", project_a.id).hits[0]
        selector = DeclarationSelector(chunk_id=helper_hit.chunk_id)

        started = threading.Event()
        release = threading.Event()

        def hold_worker(_progress: IndexProgress) -> None:
            started.set()
            assert release.wait(timeout=10.0)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                app.index_project,
                project_b.id,
                force=True,
                on_progress=hold_worker,
            )
            assert started.wait(timeout=10.0)
            worker = app._get_index_worker()
            assert worker.active_project_id == project_b.id

            # Settled references never touch the busy worker.
            with patch.object(
                worker,
                "backfill_references",
                side_effect=AssertionError("settled query touched busy worker"),
            ):
                refs = app.find_references(selector)
                assert len(refs.hits) >= 2

            # Unsettled parse-only backfill on Project A runs in a disposable worker
            # instead of queueing behind Project B's active index run.
            _remove_reference_generation(app.store, project_a.id)
            backfill = app.ensure_reference_index(project_a.id)
            assert backfill.files_backfilled == 3
            assert backfill.complete is True
            assert app.find_references(selector).hits

            release.set()
            assert future.result(timeout=10.0).indexed_files == 3
    finally:
        app.close()


def test_worker_sigkill_mid_run_rolls_back_committing_journal_and_marks_history_interrupted(
    tmp_path: Path,
) -> None:
    """Finding 2: SIGKILL mid-run rolls back PHASE_COMMITTING and marks history interrupted."""
    root = tmp_path / "project"
    root.mkdir()
    target_file = root / "main.py"
    target_file.write_text("def before_crash() -> int:\n    return 1\n")

    paths = _make_paths(tmp_path)
    app = Application(
        paths,
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        project = app.init_project(root)
        app.index_project(project.id)
        assert len(app.find_symbol("before_crash", project.id).hits) == 1

        target = app._resolve_active_target(project)
        target_file.write_text("def after_crash() -> int:\n    return 2\n")
        killed = False

        def simulate_mid_commit_crash(_progress: IndexProgress) -> None:
            nonlocal killed
            if killed:
                return
            runs = app.history.list_runs(project.id).runs
            if not runs or runs[0].state != "running":
                return
            worker_pid = app._get_index_worker().pid
            assert worker_pid is not None

            versions = app.store.table_versions(project.id, partition_id=target.partition_id)
            file_ids = [
                record.file_id
                for record in app.store.list_files(project.id, partition_id=target.partition_id)
            ]
            # Write a PHASE_COMMITTING journal and corrupt live rows to simulate a torn commit.
            writer = StagingJob(
                paths.data / "staging",
                project.id,
                file_schema=LanceStore.file_arrow_schema(),
                chunk_schema=LanceStore.chunk_arrow_schema(
                    app.store.vector_dimension, app.store.vector_dtype
                ),
                reference_schema=LanceStore.reference_arrow_schema(),
                partition=target.partition,
                job_id="torn-commit",
            )
            writer.begin()
            writer.begin_commit(versions)
            app.store.replace_files_from_arrow(
                project.id,
                files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
                chunk_batches=(),
                removed_file_ids=file_ids,
                partition_id=target.partition_id,
            )
            killed = True
            os.kill(worker_pid, signal.SIGKILL)

        with pytest.raises(CodeIndexingError) as exc_info:
            app.index_project(project.id, on_progress=simulate_mid_commit_crash)
        assert killed
        assert exc_info.value.code is ErrorCode.INTERNAL_ERROR

        # Recovery ran before returning: history is interrupted, journal is rolled back,
        # and the pre-crash generation remains searchable.
        runs = app.history.list_runs(project.id).runs
        assert runs[0].state == "interrupted"
        assert not list((paths.data / "staging").rglob("journal.json"))
        assert len(app.find_symbol("before_crash", project.id).hits) == 1
        assert app._get_index_worker().ping() != os.getpid()
    finally:
        app.close()


def test_on_progress_exception_tears_down_worker_and_request_id_prevents_stale_frames(
    tmp_path: Path,
) -> None:
    """Finding 3: exception in on_progress tears down worker cleanly with no pipe desync."""
    root = tmp_path / "project"
    _write_python_corpus(root, file_count=4, functions_per_file=3)

    app = Application(
        _make_paths(tmp_path),
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        project = app.init_project(root)
        first_pid = app._get_index_worker().ping()

        def failing_progress(_progress: IndexProgress) -> None:
            raise ValueError("progress listener exploded")

        with pytest.raises(ValueError, match="progress listener exploded"):
            app.index_project(project.id, on_progress=failing_progress)

        worker = app._get_index_worker()
        assert worker.pid is None
        next_pid = worker.ping()
        assert next_pid != first_pid

        report = app.index_project(project.id, force=True)
        assert report.indexed_files == 4
        assert app.find_symbol("compute_0_0", project.id).hits
    finally:
        app.close()


def test_request_id_mismatch_tears_down_worker_and_recovers(tmp_path: Path) -> None:
    """Finding 3: mismatched response request_id raises RuntimeError and tears down worker."""
    from unittest.mock import MagicMock

    from code_indexing_mcp.index_worker import IndexWorkerSession
    from code_indexing_mcp.settings import IndexSettings
    from code_indexing_mcp.worker_launcher import LaunchedWorker

    crashes: list[str] = []
    session = IndexWorkerSession(
        _make_paths(tmp_path),
        tmp_path,
        IndexSettings.from_environment(),
        on_worker_crash=lambda: crashes.append("crashed"),
    )
    fake_conn = MagicMock()
    fake_conn.recv.return_value = (999, "pong", 12345)
    fake_proc = MagicMock()
    fake_proc.is_alive.side_effect = [True, False, False, False]
    session._launched = LaunchedWorker(process=fake_proc, connection=fake_conn)

    with pytest.raises(RuntimeError, match="Mismatched index worker response id"):
        session.ping()

    assert session._launched is None
    assert crashes == ["crashed"]
    fake_conn.close.assert_called_once()


def test_ensure_query_generations_rebuilds_in_worker_process(tmp_path: Path) -> None:
    """Finding 6: _ensure_query_generations delegates schema-rebuild to the worker process."""
    root = tmp_path / "project"
    _write_python_corpus(root, file_count=2, functions_per_file=2)
    paths = _make_paths(tmp_path)

    app_v1 = Application(
        paths,
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        project = app_v1.init_project(root)
        app_v1.index_project(project.id)
    finally:
        app_v1.close()

    embedder_v2 = DeterministicEmbedder()
    embedder_v2.model_id = "test/tiny-v2"
    app_v2 = Application(
        paths,
        embedder=embedder_v2,
        cwd=tmp_path,
        _enable_index_worker=True,
    )
    try:
        hits = app_v2.search_code("compute_0_0", projects=[project.id])
        assert hits.hits
        worker_pid = app_v2._get_index_worker().pid
        assert worker_pid is not None
        assert worker_pid != os.getpid()

        latest_run = app_v2.history.list_runs(project.id).runs[0]
        assert latest_run.trigger == "schema-rebuild"
        assert latest_run.pid == worker_pid
    finally:
        app_v2.close()
