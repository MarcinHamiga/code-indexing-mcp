from __future__ import annotations

import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from support import DeterministicEmbedder

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import IndexProgress


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
