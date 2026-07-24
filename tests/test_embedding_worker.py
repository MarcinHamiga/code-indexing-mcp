from __future__ import annotations

from multiprocessing.connection import Connection

import pytest

from incode_mcp.embedding_worker import (
    MINIMUM_WORKER_BYTES,
    EmbeddingWorkerSession,
    WorkerConfig,
    effective_memory_ceiling,
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
