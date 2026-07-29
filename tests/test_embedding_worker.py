from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest
from test_token_batching import fake_encode

from incode_mcp.embedding import (
    PROBE_TEXTS,
    PassageCandidate,
    SegmentPlan,
    embed_windows,
    pack_vector,
    plan_passages,
)
from incode_mcp.embedding_worker import (
    MINIMUM_WORKER_BYTES,
    EmbeddingWorkerSession,
    WorkerConfig,
    WorkerTarget,
    _load_model,
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


class _ToListOnlyVector:
    """A model row exposing only ``tolist()``, which the contract permits."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return list(self._values)


def test_pack_vector_accepts_rows_that_only_expose_tolist() -> None:
    # FastEmbedder._vectors has always accepted any row with tolist(), so the
    # packing path must too; a non-numpy row previously raised inside asarray.
    values = [0.5, -1.5, 2.0, 3.25]

    assert pack_vector(_ToListOnlyVector(values)) == pack_vector(np.asarray(values, dtype="<f4"))


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


def _planning_worker(connection: Connection, _: WorkerConfig) -> None:
    """Windows candidates with the deterministic tokenizer, no model involved."""
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        candidates_raw, plan = payload
        candidates = [PassageCandidate(prefix, content) for prefix, content in candidates_raw]
        try:
            windows = plan_passages(fake_encode, candidates, plan)
        except ValueError as exc:
            connection.send(("plan_error", str(exc)))
            continue
        planned = embed_windows(
            lambda texts: [
                np.asarray([float(len(text)), 1.0, 2.0, 3.0], dtype="<f4").tobytes()
                for text in texts
            ],
            candidates,
            windows,
            plan,
        )
        connection.send(
            (
                "planned",
                (
                    [
                        [(w.start_char, w.end_char, w.token_count, vector) for w, vector in group]
                        for group in planned
                    ],
                    True,
                ),
            )
        )


def test_plan_and_embed_returns_a_segment_per_token_window() -> None:
    session = _session(_planning_worker, 2 * 1024**3)
    content = " ".join(f"tok{index}" for index in range(30))

    with session:
        segments = session.plan_and_embed(
            [PassageCandidate("kind: module", content)],
            SegmentPlan(max_tokens=8, overlap_tokens=2),
        )

    assert len(segments[0]) > 1
    assert all(len(segment.vector) == 4 * 4 for segment in segments[0])
    assert segments[0][0].start_char == 0
    assert segments[0][-1].end_char == len(content)
    assert session.segment_count == len(segments[0])
    assert session.token_count == sum(segment.token_count for segment in segments[0])
    assert session.tokenizer_available is True


def test_an_unplannable_candidate_raises_a_plain_error_the_file_absorbs() -> None:
    session = _session(_planning_worker, 2 * 1024**3)
    content = " ".join(f"tok{index}" for index in range(500))

    with session, pytest.raises(ValueError, match="exceeded 2 windows"):
        session.plan_and_embed(
            [PassageCandidate("", content)],
            SegmentPlan(max_tokens=8, overlap_tokens=2, max_windows=2),
        )

    # Deliberately not an IncodeError: the indexer's environment-error set would
    # abort the whole run for what is one bad file.
    assert session.termination_reason is None


def _batch_size_sensitive_worker(connection: Connection, _: WorkerConfig) -> None:
    """Dies whenever a microbatch carries more than one item."""
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        _, plan = payload
        if plan.max_items > 1:
            os._exit(1)
        connection.send(("planned", ([[(0, 1, 1, np.zeros(4, dtype="<f4").tobytes())]], True)))


def test_a_failed_batch_is_retried_at_a_halved_microbatch_size() -> None:
    session = _session(_batch_size_sensitive_worker, 2 * 1024**3)

    with session:
        segments = session.plan_and_embed(
            [PassageCandidate("", "x")], SegmentPlan(max_tokens=8, max_items=4)
        )

    # 4 -> 2 -> 1, so two retries, and the run survives.
    assert session.retry_count == 2
    assert len(segments[0]) == 1


def _always_failing_worker(connection: Connection, _: WorkerConfig) -> None:
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        os._exit(1)


def test_retries_stop_and_the_error_surfaces_once_the_batch_cannot_shrink() -> None:
    session = _session(_always_failing_worker, 2 * 1024**3)

    with session, pytest.raises(IncodeError) as caught:
        session.plan_and_embed([PassageCandidate("", "x")], SegmentPlan(max_tokens=8, max_items=2))

    assert caught.value.code is ErrorCode.EMBEDDING_WORKER_FAILED
    assert session.retry_count == 1
    # Either the liveness poll or the closed pipe notices first, depending on
    # how the two processes interleave; both name a dead worker.
    assert session.termination_reason in {"worker_exited", "channel_closed"}


class _BreaksAt:
    """A worker that died mid-request, noticed at *stage* as *error*."""

    def __init__(self, stage: str, error: BaseException) -> None:
        self._stage = stage
        self._error = error

    def send(self, _payload: object) -> None:
        if self._stage == "send":
            raise self._error

    def poll(self, _timeout: float = 0.0) -> bool:
        if self._stage == "poll":
            raise self._error
        return True

    def recv(self) -> tuple[str, object]:
        raise self._error

    def close(self) -> None:
        return None


class _NeverExits:
    """Reports itself alive, so only the channel can reveal the death."""

    pid = 4321

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("send", BrokenPipeError(32, "Broken pipe")),
        ("poll", BrokenPipeError(109, "The pipe has been ended")),
        ("recv", ConnectionResetError(104, "Connection reset by peer")),
        ("recv", EOFError()),
    ],
)
def test_a_broken_channel_is_reported_however_it_breaks(stage: str, error: BaseException) -> None:
    """No way of losing the worker escapes as a raw channel error.

    Which exception a dead worker produces depends on the platform and on
    whether the channel is a pipe or a socket, so the cases that never occur on
    the machine running this are injected rather than left to a race: a leak
    here reaches indexing, which can only degrade to CPU on an IncodeError.
    """
    session = _session(_fake_worker, 2 * 1024**3)
    session._process = _NeverExits()  # type: ignore[assignment]
    session._connection = _BreaksAt(stage, error)  # type: ignore[assignment]

    with pytest.raises(IncodeError) as caught:
        session.initialize()

    assert caught.value.code is ErrorCode.EMBEDDING_WORKER_FAILED
    assert caught.value.__cause__ is error
    assert session.termination_reason == "channel_closed"


def _protocol_worker(connection: Connection, config: WorkerConfig) -> None:
    """Answers the lifecycle commands the backend contract added."""
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "memory":
            connection.send(("memory", 4096))
            continue
        if command == "probe":
            connection.send(
                ("probed", [_unit_row() for _ in range(len(PROBE_TEXTS))]),
            )
            continue
        connection.send(("packed", [_unit_row() for _ in payload]))


def _unit_row() -> bytes:
    row = np.zeros(4, dtype="<f4")
    row[0] = 1.0
    return row.tobytes()


def test_initialize_reports_the_providers_the_session_resolved() -> None:
    config = WorkerConfig(
        cache_directory="unused",
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        dimension=4,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        accelerator="cuda",
    )
    session = EmbeddingWorkerSession(
        config, effective_ceiling_bytes=2 * 1024**3, target=_protocol_worker
    )

    with session:
        info = session.initialize()

    assert info.resolved_providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert info.dimension == 4


def test_a_probe_that_returns_usable_vectors_is_accepted() -> None:
    session = _session(_protocol_worker, 2 * 1024**3)

    with session:
        vectors = session.probe()

    assert len(vectors) == len(PROBE_TEXTS)


def test_report_memory_returns_the_workers_own_footprint() -> None:
    session = _session(_protocol_worker, 2 * 1024**3)

    with session:
        assert session.report_memory() == 4096


def _bad_dimension_worker(connection: Connection, _: WorkerConfig) -> None:
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        connection.send(("probed", [np.zeros(8, dtype="<f4").tobytes() for _ in PROBE_TEXTS]))


def _unnormalized_worker(connection: Connection, _: WorkerConfig) -> None:
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        connection.send(("probed", [np.full(4, 5.0, dtype="<f4").tobytes() for _ in PROBE_TEXTS]))


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (_bad_dimension_worker, "wide"),
        (_unnormalized_worker, "norm"),
    ],
)
def test_a_probe_whose_vectors_could_not_search_an_index_is_rejected(
    target: WorkerTarget, message: str
) -> None:
    # Deliberately a ValueError rather than an IncodeError: the caller decides
    # whether an unusable backend means fall back or fail.
    session = _session(target, 2 * 1024**3)

    with session, pytest.raises(ValueError, match=message):
        session.probe()


def test_the_default_worker_config_requests_no_providers() -> None:
    """The CPU path must call the model exactly as it always has."""
    config = WorkerConfig(
        cache_directory="unused",
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        dimension=4,
    )

    assert config.is_cpu is True
    assert config.providers == ()


@pytest.mark.parametrize("accelerator", ["webgpu", "migraphx"])
def test_direct_accelerators_do_not_load_fastembed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accelerator: str,
) -> None:
    from incode_mcp import direct_onnx, embedding_worker

    direct_model = object()
    direct_options: list[dict[str, object]] = []

    def direct(**options: object) -> object:
        direct_options.append(options)
        return direct_model

    monkeypatch.setattr(direct_onnx, "DirectOnnxEmbedding", direct)
    monkeypatch.setattr(
        embedding_worker,
        "TextEmbedding",
        lambda **options: pytest.fail(f"FastEmbed was loaded with {options}"),
        raising=False,
    )
    config = WorkerConfig(
        cache_directory=str(tmp_path),
        offline=True,
        threads=2,
        enable_cpu_mem_arena=False,
        dimension=768,
        providers=(
            "WebGpuExecutionProvider" if accelerator == "webgpu" else "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ),
        accelerator=accelerator,
    )

    assert _load_model(config) is direct_model
    assert direct_options == [
        {
            "cache_directory": tmp_path,
            "offline": True,
            "threads": 2,
            "enable_cpu_mem_arena": False,
            "providers": config.providers,
            "model_id": config.model_id,
            "accelerator": accelerator,
        }
    ]


def test_telemetry_names_the_backend_the_worker_ran_on() -> None:
    session = _session(_protocol_worker, 2 * 1024**3)

    with session:
        session.embed_passages(["a"])

    assert session.telemetry().backend == "cpu"
    assert session.telemetry().memory_budget_bytes == 2 * 1024**3
