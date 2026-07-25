from __future__ import annotations

import time
from multiprocessing.connection import Connection

import pytest

from incode_mcp.embedding_worker import (
    MINIMUM_WORKER_BYTES,
    EmbeddingWorkerSession,
    WorkerConfig,
    WorkerTarget,
    effective_memory_ceiling,
    indexing_memory_bytes,
)
from incode_mcp.errors import ErrorCode, IncodeError


def _fake_worker(connection: Connection, _: WorkerConfig) -> None:
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        vectors = [[float(len(text)), 1.0, 2.0, 3.0] for text in payload]
        connection.send(("ok", vectors))


def test_effective_memory_ceiling_reserves_system_memory() -> None:
    ceiling = effective_memory_ceiling(
        configured_bytes=2 * 1024**3,
        available_bytes=1400 * 1024**2,
    )

    assert ceiling == 888 * 1024**2


def test_effective_memory_ceiling_uses_configured_limit_when_ram_is_available() -> None:
    ceiling = effective_memory_ceiling(
        configured_bytes=1536 * 1024**2,
        available_bytes=8 * 1024**3,
    )

    assert ceiling == 1536 * 1024**2


def test_embedding_worker_round_trips_vectors_and_stops() -> None:
    session = EmbeddingWorkerSession(
        WorkerConfig(
            cache_directory="unused",
            offline=True,
            threads=1,
            enable_cpu_mem_arena=False,
            dimension=4,
        ),
        effective_ceiling_bytes=2 * 1024**3,
        target=_fake_worker,
    )

    with session:
        assert session.pid is None
        vectors = session.embed_passages(["a", "abcd"])
        pid = session.pid

    assert vectors == [[1.0, 1.0, 2.0, 3.0], [4.0, 1.0, 2.0, 3.0]]
    assert pid is not None
    assert session.pid is None


def test_embedding_worker_refuses_unsafe_effective_budget() -> None:
    with pytest.raises(IncodeError) as caught:
        EmbeddingWorkerSession(
            WorkerConfig(
                cache_directory="unused",
                offline=True,
                threads=1,
                enable_cpu_mem_arena=False,
                dimension=4,
            ),
            effective_ceiling_bytes=MINIMUM_WORKER_BYTES - 1,
        )

    assert caught.value.code is ErrorCode.INDEX_RESOURCE_LIMIT


def _slow_worker(connection: Connection, _: WorkerConfig) -> None:
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        time.sleep(0.4)
        connection.send(("ok", [[0.0, 1.0, 2.0, 3.0] for _ in payload]))


def _session(target: WorkerTarget, ceiling_bytes: int) -> EmbeddingWorkerSession:
    return EmbeddingWorkerSession(
        WorkerConfig(
            cache_directory="unused",
            offline=True,
            threads=1,
            enable_cpu_mem_arena=False,
            dimension=4,
        ),
        effective_ceiling_bytes=ceiling_bytes,
        target=target,
    )


def test_indexing_memory_excludes_the_parent_baseline() -> None:
    budgeted = indexing_memory_bytes(
        parent_bytes=4 * 1024**3,
        worker_bytes=700 * 1024**2,
        parent_baseline_bytes=4 * 1024**3,
    )

    assert budgeted == 700 * 1024**2


def test_indexing_memory_counts_parent_growth_during_indexing() -> None:
    budgeted = indexing_memory_bytes(
        parent_bytes=4 * 1024**3 + 300 * 1024**2,
        worker_bytes=700 * 1024**2,
        parent_baseline_bytes=4 * 1024**3,
    )

    assert budgeted == 1000 * 1024**2


def test_indexing_memory_never_goes_negative_when_the_parent_shrinks() -> None:
    budgeted = indexing_memory_bytes(
        parent_bytes=1024**3,
        worker_bytes=64 * 1024**2,
        parent_baseline_bytes=2 * 1024**3,
    )

    assert budgeted == 64 * 1024**2


def test_resident_parent_memory_does_not_trip_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon already holding a query model must still be able to index."""
    session = _session(_slow_worker, 1024**3)
    resident = 4 * 1024**3  # far above the ceiling, but present before indexing
    start = session._start

    def start_with_resident_parent() -> None:
        start()
        session._parent_baseline_bytes = resident

    monkeypatch.setattr(session, "_start", start_with_resident_parent)
    monkeypatch.setattr(session, "_sample_rss", lambda: (resident, 32 * 1024**2))

    with session:
        vectors = session.embed_passages(["a"])

    assert vectors == [[0.0, 1.0, 2.0, 3.0]]
    assert session.peak_combined_rss >= resident


def test_worker_growth_still_trips_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(_slow_worker, 1024**3)
    monkeypatch.setattr(session, "_sample_rss", lambda: (0, 8 * 1024**3))

    with session, pytest.raises(IncodeError) as caught:
        session.embed_passages(["a"])

    assert caught.value.code is ErrorCode.INDEX_RESOURCE_LIMIT
    assert caught.value.details["indexing_memory_bytes"] == 8 * 1024**3
