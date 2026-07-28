from __future__ import annotations

import os
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest

from incode_mcp.backends import (
    CPU_BACKEND,
    CPU_PROVIDER,
    Accelerator,
    BackendDescriptor,
    BackendSelection,
    Precision,
    Stability,
)
from incode_mcp.embedding import PROBE_TEXTS, PassageCandidate, SegmentPlan
from incode_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig, WorkerTarget
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.passage_backend import PassageBackendSession
from incode_mcp.probe_cache import ProbeCache, ProbeKey

DIMENSION = 4
CUDA_PROVIDER = "CUDAExecutionProvider"

CUDA_BACKEND = BackendDescriptor(
    accelerator=Accelerator.CUDA,
    provider=CUDA_PROVIDER,
    device="cuda:0",
    stability=Stability.AUTOMATIC,
    precision=Precision.FLOAT32,
)


def _unit_vector(value: float) -> bytes:
    row = np.zeros(DIMENSION, dtype="<f4")
    row[0] = value
    return row.tobytes()


def _packed_segments(count: int) -> list[list[tuple[int, int, int, bytes]]]:
    return [[(0, 1, 1, _unit_vector(1.0))] for _ in range(count)]


def _accelerator_selection() -> BackendSelection:
    return BackendSelection(
        requested=Accelerator.CUDA,
        descriptor=CUDA_BACKEND,
        available_providers=(CUDA_PROVIDER, CPU_PROVIDER),
    )


# -- worker doubles --------------------------------------------------------


def _healthy_worker(connection: Connection, config: WorkerConfig) -> None:
    """A worker that initializes, probes, and embeds without complaint."""
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            connection.send(("probed", [_unit_vector(1.0) for _ in PROBE_TEXTS]))
            continue
        if command == "memory":
            connection.send(("memory", 1024))
            continue
        if command == "plan_and_embed":
            candidates, _plan = payload
            connection.send(("planned", (_packed_segments(len(candidates)), True)))
            continue
        connection.send(("packed", [_unit_vector(1.0) for _ in payload]))


def _dead_on_initialize_worker(connection: Connection, _: WorkerConfig) -> None:
    """A provider that cannot be initialised at all."""
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        os._exit(1)


def _wrong_provider_worker(connection: Connection, config: WorkerConfig) -> None:
    """Loads, but silently on CPU -- exactly what ONNX Runtime does on a drop."""
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", ((CPU_PROVIDER,), config.dimension)))
            continue
        connection.send(("planned", (_packed_segments(1), True)))


def _bad_probe_worker(connection: Connection, config: WorkerConfig) -> None:
    """Initialises and then returns vectors an index could not search."""
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            broken = np.full(DIMENSION, np.nan, dtype="<f4").tobytes()
            connection.send(("probed", [broken for _ in PROBE_TEXTS]))
            continue
        connection.send(("planned", (_packed_segments(1), True)))


def _unplannable_worker(connection: Connection, config: WorkerConfig) -> None:
    """Healthy, but rejects the candidate it is given as unwindowable."""
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            connection.send(("probed", [_unit_vector(1.0) for _ in PROBE_TEXTS]))
            continue
        connection.send(("plan_error", "exceeded 2 windows"))


def _dies_after_first_batch_worker(connection: Connection, config: WorkerConfig) -> None:
    """Survives the probe and the first group, then dies mid-run."""
    served = 0
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            connection.send(("probed", [_unit_vector(1.0) for _ in PROBE_TEXTS]))
            continue
        if served:
            os._exit(1)
        served += 1
        candidates, _plan = payload
        connection.send(("planned", (_packed_segments(len(candidates)), True)))


# -- fixtures --------------------------------------------------------------


def _config(accelerator: str, providers: tuple[str, ...]) -> WorkerConfig:
    return WorkerConfig(
        cache_directory="unused",
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        dimension=DIMENSION,
        providers=providers,
        accelerator=accelerator,
    )


def _session(target: WorkerTarget, config: WorkerConfig) -> EmbeddingWorkerSession:
    return EmbeddingWorkerSession(config, effective_ceiling_bytes=2 * 1024**3, target=target)


def _backend(
    accelerator_target: WorkerTarget,
    *,
    cpu_target: WorkerTarget = _healthy_worker,
    strict: bool = False,
    selection: BackendSelection | None = None,
    probe_cache: ProbeCache | None = None,
    probe_key: ProbeKey | None = None,
) -> PassageBackendSession:
    cpu_started: list[int] = []

    def cpu_factory() -> EmbeddingWorkerSession:
        cpu_started.append(1)
        return _session(cpu_target, _config(Accelerator.CPU.value, CPU_BACKEND.providers))

    session = PassageBackendSession(
        selection or _accelerator_selection(),
        accelerator_factory=lambda: _session(
            accelerator_target, _config(Accelerator.CUDA.value, CUDA_BACKEND.providers)
        ),
        cpu_factory=cpu_factory,
        strict=strict,
        probe_cache=probe_cache,
        probe_key=probe_key,
        batch_size=8,
        dimension=DIMENSION,
    )
    session.cpu_starts = cpu_started  # type: ignore[attr-defined]
    return session


def _candidates(count: int) -> list[PassageCandidate]:
    return [PassageCandidate("", f"chunk-{index}") for index in range(count)]


PLAN = SegmentPlan(max_tokens=8, max_items=1)


# -- selection is honoured -------------------------------------------------


def test_a_working_accelerator_is_used_and_reported(tmp_path: Path) -> None:
    with _backend(_healthy_worker) as backend:
        segments = backend.plan_and_embed(_candidates(2), PLAN)

        assert len(segments) == 2
        assert backend.selection.accelerator is Accelerator.CUDA
        assert backend.probe_state == "verified"
        assert backend.fallback_count == 0
        assert backend.cpu_starts == []  # type: ignore[attr-defined]

    assert backend.telemetry().backend == "cuda"


def test_a_cpu_selection_never_starts_an_accelerator() -> None:
    cpu_only = BackendSelection(
        requested=Accelerator.CPU,
        descriptor=CPU_BACKEND,
        available_providers=(CPU_PROVIDER,),
    )

    def refuse() -> EmbeddingWorkerSession:
        raise AssertionError("the accelerator must not be started for a CPU selection")

    session = PassageBackendSession(
        cpu_only,
        accelerator_factory=refuse,
        cpu_factory=lambda: _session(
            _healthy_worker, _config(Accelerator.CPU.value, CPU_BACKEND.providers)
        ),
        dimension=DIMENSION,
    )

    with session as backend:
        assert len(backend.plan_and_embed(_candidates(1), PLAN)) == 1

    assert backend.telemetry().backend == "cpu"


# -- degradation paths -----------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected_probe_state"),
    [
        (_dead_on_initialize_worker, "failed"),
        (_wrong_provider_worker, "failed"),
        (_bad_probe_worker, "failed"),
    ],
)
def test_a_backend_that_fails_verification_is_replaced_by_cpu(
    target: WorkerTarget, expected_probe_state: str
) -> None:
    with _backend(target) as backend:
        segments = backend.plan_and_embed(_candidates(1), PLAN)

        assert len(segments) == 1
        assert backend.selection.accelerator is Accelerator.CPU
        assert backend.probe_state == expected_probe_state
        assert backend.fallback_count == 1
        assert backend.cpu_starts == [1]  # type: ignore[attr-defined]

    report = backend.telemetry()
    assert report.backend == "cpu"
    assert report.fallback_reason


def test_a_provider_that_is_silently_dropped_is_caught_not_trusted() -> None:
    """A "CUDA" session that really runs on CPU must not pass as CUDA."""
    with _backend(_wrong_provider_worker) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    assert backend.selection.accelerator is Accelerator.CPU
    assert CPU_PROVIDER in (backend.fallback_reason or "")


def test_a_worker_dying_mid_run_moves_the_remaining_chunks_to_cpu() -> None:
    with _backend(_dies_after_first_batch_worker) as backend:
        first = backend.plan_and_embed(_candidates(2), PLAN)
        assert backend.selection.accelerator is Accelerator.CUDA

        # The group that killed the accelerator is re-embedded on CPU, so the
        # caller receives its rows rather than losing them.
        second = backend.plan_and_embed(_candidates(3), PLAN)

    assert len(first) == 2
    assert len(second) == 3
    assert backend.selection.accelerator is Accelerator.CPU
    assert backend.fallback_count == 1


def test_the_run_stays_on_cpu_once_it_has_fallen_back() -> None:
    with _backend(_dies_after_first_batch_worker) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)
        backend.plan_and_embed(_candidates(1), PLAN)
        starts_after_fallback = len(backend.cpu_starts)  # type: ignore[attr-defined]
        backend.plan_and_embed(_candidates(1), PLAN)

        # One CPU worker serves the rest of the run; it is not respawned per
        # group, and the dead accelerator is never retried.
        assert len(backend.cpu_starts) == starts_after_fallback  # type: ignore[attr-defined]
        assert backend.fallback_count == 1


def test_the_accelerator_worker_is_terminated_when_it_is_abandoned() -> None:
    backend = _backend(_dies_after_first_batch_worker)
    with backend:
        backend.plan_and_embed(_candidates(1), PLAN)
        accelerator_pid = backend._session.pid if backend._session else None
        backend.plan_and_embed(_candidates(1), PLAN)

    assert accelerator_pid is not None
    # Releasing device memory is the point of the teardown; the pid must be
    # gone rather than merely unreferenced.
    with pytest.raises(OSError):
        os.kill(accelerator_pid, 0)


# -- strict mode -----------------------------------------------------------


def test_strict_mode_refuses_the_fallback_when_verification_fails() -> None:
    with (
        pytest.raises(IncodeError) as caught,
        _backend(_dead_on_initialize_worker, strict=True) as backend,
    ):
        backend.plan_and_embed(_candidates(1), PLAN)

    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE
    assert caught.value.details["accelerator"] == "cuda"


def test_strict_mode_refuses_the_fallback_when_a_worker_dies_mid_run() -> None:
    with (
        pytest.raises(IncodeError) as caught,
        _backend(_dies_after_first_batch_worker, strict=True) as backend,
    ):
        backend.plan_and_embed(_candidates(1), PLAN)
        backend.plan_and_embed(_candidates(1), PLAN)

    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE


def test_strict_mode_fails_before_any_work_when_the_selection_was_denied() -> None:
    denied = BackendSelection(
        requested=Accelerator.CUDA,
        descriptor=CPU_BACKEND,
        available_providers=(CPU_PROVIDER,),
        honored=False,
        fallback_reason="CUDAExecutionProvider is not installed",
    )

    with (
        pytest.raises(IncodeError) as caught,
        _backend(_healthy_worker, strict=True, selection=denied),
    ):
        pass

    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE


def test_strict_mode_allows_an_auto_selection_that_settled_on_cpu() -> None:
    """``auto`` resolving to CPU is a correct outcome, not a denied request."""
    resolved = BackendSelection(
        requested=Accelerator.AUTO,
        descriptor=CPU_BACKEND,
        available_providers=(CPU_PROVIDER,),
        fallback_reason="no accelerator has reached automatic selection",
    )

    with _backend(_healthy_worker, strict=True, selection=resolved) as backend:
        assert len(backend.plan_and_embed(_candidates(1), PLAN)) == 1


# -- probe caching ---------------------------------------------------------


def _probe_key() -> ProbeKey:
    return ProbeKey(
        model_id="test-model",
        model_artifact="artifact",
        accelerator="cuda",
        provider=CUDA_PROVIDER,
        runtime_version="1.20.0",
        platform="test",
        device="cuda:0",
    )


def test_a_successful_probe_is_cached_and_then_reused(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    key = _probe_key()

    with _backend(_healthy_worker, probe_cache=cache, probe_key=key) as first:
        first.plan_and_embed(_candidates(1), PLAN)
    assert first.probe_state == "verified"

    record = cache.load(key)
    assert record is not None
    assert record.batch_size == 8
    assert record.dimension == DIMENSION

    with _backend(_healthy_worker, probe_cache=cache, probe_key=key) as second:
        second.plan_and_embed(_candidates(1), PLAN)

    # The second run still loads the model -- that is what proves the provider
    # initialises on this boot -- but skips the inference.
    assert second.probe_state == "cached"


def test_a_failed_probe_is_never_cached(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    key = _probe_key()

    with _backend(_bad_probe_worker, probe_cache=cache, probe_key=key) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    assert cache.load(key) is None


def test_a_cached_probe_for_another_configuration_does_not_apply(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    cache.store(_probe_key(), batch_size=8, dimension=DIMENSION)
    moved = ProbeKey(**{**_probe_key().__dict__, "runtime_version": "9.9.9"})

    with _backend(_healthy_worker, probe_cache=cache, probe_key=moved) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    assert backend.probe_state == "verified"


# -- telemetry -------------------------------------------------------------


def test_telemetry_sums_the_work_done_across_both_backends() -> None:
    with _backend(_dies_after_first_batch_worker) as backend:
        backend.plan_and_embed(_candidates(2), PLAN)
        backend.plan_and_embed(_candidates(2), PLAN)

    report = backend.telemetry()
    # Two segments on the accelerator plus the two the CPU re-embedded after it
    # died. The retired accelerator session's counters survive its teardown,
    # and its failed attempt contributed nothing.
    assert report.segment_count == 4
    assert report.backend == "cpu"
    assert report.fallback_count >= 1
    assert report.tokenizer_available is True
    assert report.memory_budget_bytes == 2 * 1024**3


def test_a_content_error_is_not_treated_as_a_backend_failure() -> None:
    """A file the planner cannot window must not cost the run its accelerator."""
    with _backend(_unplannable_worker) as backend:
        with pytest.raises(ValueError, match="exceeded 2 windows"):
            backend.plan_and_embed(_candidates(1), PLAN)

        assert backend.selection.accelerator is Accelerator.CUDA
        assert backend.fallback_count == 0


def test_an_ordinary_cpu_run_reports_no_fallback_reason() -> None:
    """``auto`` settling on CPU is not a degradation and must not read as one."""
    resolved = BackendSelection(
        requested=Accelerator.AUTO,
        descriptor=CPU_BACKEND,
        available_providers=(CPU_PROVIDER,),
        fallback_reason="no accelerator has reached automatic selection",
    )

    with _backend(_healthy_worker, selection=resolved) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    assert backend.telemetry().fallback_reason is None
    assert backend.telemetry().fallback_count == 0
