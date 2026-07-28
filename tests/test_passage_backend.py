from __future__ import annotations

import os
from collections.abc import Callable
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
    calibrated_batch_size: int = 8,
    accelerator_factory: Callable[[], EmbeddingWorkerSession] | None = None,
    on_degrade: Callable[[BackendSelection], None] | None = None,
) -> PassageBackendSession:
    cpu_started: list[int] = []

    def cpu_factory() -> EmbeddingWorkerSession:
        cpu_started.append(1)
        return _session(cpu_target, _config(Accelerator.CPU.value, CPU_BACKEND.providers))

    session = PassageBackendSession(
        selection or _accelerator_selection(),
        accelerator_factory=accelerator_factory
        or (
            lambda: _session(
                accelerator_target, _config(Accelerator.CUDA.value, CUDA_BACKEND.providers)
            )
        ),
        cpu_factory=cpu_factory,
        strict=strict,
        probe_cache=probe_cache,
        probe_key=probe_key,
        calibrated_batch_size=calibrated_batch_size,
        dimension=DIMENSION,
        on_degrade=on_degrade,
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


# -- a terminated accelerator still reports what it measured ---------------


def _starved_accelerator() -> EmbeddingWorkerSession:
    """An accelerator whose model load alone overruns the memory ceiling."""
    session = _session(_healthy_worker, _config(Accelerator.CUDA.value, CUDA_BACKEND.providers))
    # Sampled, not allocated, so the gate is exercised without the test itself
    # needing gigabytes.
    session._sample_rss = lambda: (0, 8 * 1024**3)  # type: ignore[method-assign]
    return session


def test_an_accelerator_killed_while_verifying_still_reports_its_measurements() -> None:
    """The evidence for a fallback must survive the backend that caused it.

    A run that fell back because its accelerator blew the ceiling has to say
    so with numbers. Discarding the terminated session would leave a report
    naming a memory failure beside a peak that never came close to one.
    """
    backend = _backend(_healthy_worker, accelerator_factory=_starved_accelerator)

    with backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    report = backend.telemetry()
    assert report.backend == "cpu"
    assert report.termination_reason == "memory_ceiling"
    assert report.peak_memory_bytes >= 8 * 1024**3
    # for_client() rather than str(): the ceiling and the overrun are details
    # on the error, and the reason field is the only place they can travel.
    assert "indexing_memory_bytes=" in (report.fallback_reason or "")


def test_a_degraded_backend_tells_its_owner_so_the_next_run_can_skip_it() -> None:
    remembered: list[BackendSelection] = []

    with _backend(_dead_on_initialize_worker, on_degrade=remembered.append) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    assert [selection.accelerator for selection in remembered] == [Accelerator.CPU]
    assert remembered[0].requested is Accelerator.CUDA
    assert remembered[0].fallback_reason


def test_strict_mode_reports_no_degradation_because_it_permits_none() -> None:
    """Strict mode raises instead of degrading, so there is nothing to memoize."""
    remembered: list[BackendSelection] = []

    with (
        pytest.raises(IncodeError),
        _backend(_dead_on_initialize_worker, strict=True, on_degrade=remembered.append) as backend,
    ):
        backend.plan_and_embed(_candidates(1), PLAN)

    assert remembered == []


def test_an_uncalibrated_probe_records_no_batch_size(tmp_path: Path) -> None:
    """Nothing measures a batch size yet, and the cache must not imply one."""
    cache = ProbeCache(tmp_path / "probes.json")
    key = _probe_key()

    with _backend(
        _healthy_worker, probe_cache=cache, probe_key=key, calibrated_batch_size=0
    ) as backend:
        backend.plan_and_embed(_candidates(1), PLAN)

    record = cache.load(key)
    assert record is not None
    assert record.batch_size == 0


# -- a respawned worker is not trusted on its predecessor's verification ---


def _flaky_then_healthy_worker(connection: Connection, config: WorkerConfig) -> None:
    """Dies on the first real batch of its first process, serves after that.

    Spawn identity lives on disk because each respawn is a fresh process with
    no memory of the last one -- which is precisely the condition that makes
    re-verification necessary.
    """
    scratch = Path(config.cache_directory)
    spawn = len(list(scratch.glob("spawn-*")))
    (scratch / f"spawn-{spawn}").write_text("")
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            (scratch / f"initialized-{spawn}").write_text("")
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            connection.send(("probed", [_unit_vector(1.0) for _ in PROBE_TEXTS]))
            continue
        if spawn == 0:
            os._exit(1)
        candidates, _plan = payload
        connection.send(("planned", (_packed_segments(len(candidates)), True)))


def test_a_worker_respawned_by_a_batch_retry_is_verified_before_more_content(
    tmp_path: Path,
) -> None:
    """A batch retry replaces the process; the successor has proven nothing.

    The retry itself still runs on the fresh worker -- that is what a retry
    is -- but every group after it goes through verification first.
    """
    config = WorkerConfig(
        cache_directory=str(tmp_path),
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        dimension=DIMENSION,
        providers=CUDA_BACKEND.providers,
        accelerator=Accelerator.CUDA.value,
    )
    # max_items above 1 so the in-worker retry loop is reachable at all; it
    # short-circuits at 1, which is the shipped default.
    plan = SegmentPlan(max_tokens=8, max_items=4)
    backend = _backend(
        _healthy_worker,
        accelerator_factory=lambda: EmbeddingWorkerSession(
            config, effective_ceiling_bytes=2 * 1024**3, target=_flaky_then_healthy_worker
        ),
    )

    with backend:
        backend.plan_and_embed(_candidates(4), plan)
        assert len(list(tmp_path.glob("spawn-*"))) == 2, "the retry should have respawned"
        backend.plan_and_embed(_candidates(4), plan)

    # Both processes were verified: the original before its first batch, the
    # successor before it was given a second group.
    assert sorted(p.name for p in tmp_path.glob("initialized-*")) == [
        "initialized-0",
        "initialized-1",
    ]
    assert backend.selection.accelerator is Accelerator.CUDA
    assert backend.fallback_count == 0
