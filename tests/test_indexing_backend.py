"""Phase 2 release gate: a failing backend must not kill the server or the index.

These run the real ``Indexer`` against real spawned worker processes, with the
model replaced by a deterministic stand-in. What is being checked is the
lifecycle -- a worker that dies, a worker that overruns its memory ceiling, a
provider that never initialises -- not the embedding arithmetic.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import pytest
from test_indexing import RecordingEmbedder
from test_token_batching import fake_encode

from code_indexing_mcp.backends import (
    CPU_BACKEND,
    CPU_PROVIDER,
    Accelerator,
    BackendDescriptor,
    BackendSelection,
    Precision,
    Stability,
)
from code_indexing_mcp.embedding import (
    PROBE_TEXTS,
    PassageCandidate,
    SegmentPlan,
    embed_windows,
    plan_passages,
)
from code_indexing_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig, WorkerTarget
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.passage_backend import PassageBackendSession
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore

DIMENSION = 4
CUDA_PROVIDER = "CUDAExecutionProvider"

CUDA_BACKEND = BackendDescriptor(
    accelerator=Accelerator.CUDA,
    provider=CUDA_PROVIDER,
    device="cuda:0",
    stability=Stability.AUTOMATIC,
    precision=Precision.FLOAT32,
)


def _row(text: str) -> bytes:
    """A deterministic unit vector, so probe validation accepts it."""
    row = np.zeros(DIMENSION, dtype="<f4")
    row[len(text) % DIMENSION] = 1.0
    return row.tobytes()


def _serve_plan(connection: Connection, payload: object) -> None:
    """Window and embed a request exactly as the real worker would."""
    raw_candidates, plan = payload  # type: ignore[misc]
    candidates = [PassageCandidate(prefix, content) for prefix, content in raw_candidates]
    try:
        windows = plan_passages(fake_encode, candidates, plan)
    except ValueError as exc:
        connection.send(("plan_error", str(exc)))
        return
    planned = embed_windows(lambda texts: [_row(text) for text in texts], candidates, windows, plan)
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


def _healthy_worker(connection: Connection, config: WorkerConfig) -> None:
    while True:
        command, payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            connection.send(("probed", [_row(text) for text in PROBE_TEXTS]))
            continue
        if command == "memory":
            connection.send(("memory", 1024))
            continue
        _serve_plan(connection, payload)


def _crashing_worker(connection: Connection, config: WorkerConfig) -> None:
    """Passes its probe, then dies on the first real batch."""
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        if command == "initialize":
            connection.send(("initialized", (tuple(config.providers), config.dimension)))
            continue
        if command == "probe":
            connection.send(("probed", [_row(text) for text in PROBE_TEXTS]))
            continue
        # A segfaulting provider looks exactly like this from the parent.
        os._exit(1)


def _unloadable_worker(connection: Connection, _: WorkerConfig) -> None:
    """A provider that cannot be initialised at all."""
    while True:
        command, _payload = connection.recv()
        if command == "stop":
            return
        os._exit(1)


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


def _selection() -> BackendSelection:
    return BackendSelection(
        requested=Accelerator.CUDA,
        descriptor=CUDA_BACKEND,
        available_providers=(CUDA_PROVIDER, CPU_PROVIDER),
    )


def _session_factory(
    accelerator_target: WorkerTarget,
    *,
    strict: bool = False,
    starve_accelerator: bool = False,
    crossover_characters: int = 0,
) -> Callable[[], PassageBackendSession]:
    def accelerator_session() -> EmbeddingWorkerSession:
        session = EmbeddingWorkerSession(
            _config(Accelerator.CUDA.value, CUDA_BACKEND.providers),
            effective_ceiling_bytes=2 * 1024**3,
            target=accelerator_target,
        )
        if starve_accelerator:
            # Simulate device memory the ceiling cannot absorb. Sampling, not
            # allocation, so the gate is exercised without the test itself
            # needing gigabytes.
            session._sample_rss = lambda: (0, 8 * 1024**3)  # type: ignore[method-assign]
        return session

    def cpu_session() -> EmbeddingWorkerSession:
        return EmbeddingWorkerSession(
            _config(Accelerator.CPU.value, CPU_BACKEND.providers),
            effective_ceiling_bytes=2 * 1024**3,
            target=_healthy_worker,
        )

    def factory() -> PassageBackendSession:
        return PassageBackendSession(
            _selection(),
            accelerator_factory=accelerator_session,
            cpu_factory=cpu_session,
            strict=strict,
            dimension=DIMENSION,
            crossover_characters=crossover_characters,
        )

    return factory


def _indexer(
    tmp_path: Path, factory: Callable[[], PassageBackendSession] | None
) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_path / "data", vector_dimension=DIMENSION)
    return (
        Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=RecordingEmbedder(),
            lock_directory=tmp_path / "locks",
            segment_plan=SegmentPlan(max_tokens=64, max_items=4),
            passage_session_factory=factory,
            staging_directory=tmp_path / "staging",
        ),
        store,
    )


def _repository(tmp_path: Path, files: int = 3) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    for index in range(files):
        (root / f"module_{index}.py").write_text(
            f"def function_{index}(value):\n    return value + {index}\n"
        )
    return root


# -- the happy path stays on the accelerator -------------------------------


def test_a_working_accelerator_indexes_and_is_named_in_the_report(tmp_path: Path) -> None:
    project = initialize_project(_repository(tmp_path))
    indexer, store = _indexer(tmp_path, _session_factory(_healthy_worker))

    report = indexer.index(project)

    assert report.errors == []
    assert report.indexed_files == 3
    assert report.embedding_backend == "cuda"
    assert report.embedding_fallback_reason is None
    assert report.worker_used is True
    assert len(store.list_chunks([project.id])) == report.embedded_chunks


def test_a_run_too_small_to_repay_the_accelerator_says_so_rather_than_failing_over(
    tmp_path: Path,
) -> None:
    """A report has to distinguish CPU by design from CPU after a failure."""
    project = initialize_project(_repository(tmp_path))
    indexer, store = _indexer(
        tmp_path, _session_factory(_healthy_worker, crossover_characters=10**6)
    )

    report = indexer.index(project)

    assert report.errors == []
    assert report.embedding_backend == "cpu"
    assert report.embedding_fallback_reason is None
    assert report.fallback_count == 0
    assert report.embedded_characters and report.embedded_characters > 0
    assert report.embedding_crossover_characters == 10**6
    assert report.embedding_selection_reason
    assert "crossover" in report.embedding_selection_reason
    assert len(store.list_chunks([project.id])) == report.embedded_chunks


# -- degradation completes the run -----------------------------------------


@pytest.mark.parametrize("target", [_crashing_worker, _unloadable_worker])
def test_a_failing_accelerator_still_produces_a_complete_index(
    tmp_path: Path, target: WorkerTarget
) -> None:
    project = initialize_project(_repository(tmp_path))
    indexer, store = _indexer(tmp_path, _session_factory(target))

    report = indexer.index(project)

    # Every file is indexed, and none of them is charged with the backend's
    # failure -- the chunks were simply re-embedded somewhere else.
    assert report.errors == []
    assert report.indexed_files == 3
    assert report.embedding_backend == "cpu"
    assert report.embedding_fallback_reason
    assert report.fallback_count >= 1
    assert store.project_state(project.id) == "ready"
    assert len(store.list_chunks([project.id])) == report.embedded_chunks


def test_an_accelerator_that_overruns_its_memory_ceiling_falls_back(tmp_path: Path) -> None:
    project = initialize_project(_repository(tmp_path))
    indexer, store = _indexer(tmp_path, _session_factory(_healthy_worker, starve_accelerator=True))

    report = indexer.index(project)

    assert report.errors == []
    assert report.embedding_backend == "cpu"
    assert store.project_state(project.id) == "ready"
    assert store.list_chunks([project.id])


def test_a_fallback_run_matches_what_cpu_alone_would_have_produced(tmp_path: Path) -> None:
    """The device a chunk was embedded on must not change the rows stored."""
    reference_root = _repository(tmp_path / "reference")
    reference_project = initialize_project(reference_root)
    reference_indexer, reference_store = _indexer(
        tmp_path / "reference", _session_factory(_healthy_worker)
    )
    reference_indexer.index(reference_project)

    degraded_root = _repository(tmp_path / "degraded")
    degraded_project = initialize_project(degraded_root)
    degraded_indexer, degraded_store = _indexer(
        tmp_path / "degraded", _session_factory(_unloadable_worker)
    )
    degraded_indexer.index(degraded_project)

    def rows(store: LanceStore, project_id: str) -> list[tuple[str, str, int, int]]:
        return sorted(
            (chunk.path, chunk.kind, chunk.start_byte, chunk.end_byte)
            for chunk in store.list_chunks([project_id])
        )

    assert rows(degraded_store, degraded_project.id) == rows(reference_store, reference_project.id)


# -- strict mode -----------------------------------------------------------


def test_strict_mode_fails_the_run_instead_of_falling_back(tmp_path: Path) -> None:
    project = initialize_project(_repository(tmp_path))
    indexer, store = _indexer(tmp_path, _session_factory(_unloadable_worker, strict=True))

    with pytest.raises(CodeIndexingError) as caught:
        indexer.index(project)

    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE
    # A backend failure is never charged to whichever file was in flight, so
    # nothing is stamped as permanently broken.
    assert store.project_state(project.id) == "error"
    assert store.list_files(project.id) == []


def test_a_strict_failure_leaves_an_existing_index_intact(tmp_path: Path) -> None:
    """The release gate: a dead worker must not corrupt what was already there."""
    root = _repository(tmp_path)
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=DIMENSION)

    def indexer_with(factory: Callable[[], PassageBackendSession]) -> Indexer:
        return Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=RecordingEmbedder(),
            lock_directory=tmp_path / "locks",
            segment_plan=SegmentPlan(max_tokens=64, max_items=4),
            passage_session_factory=factory,
            staging_directory=tmp_path / "staging",
        )

    good = indexer_with(_session_factory(_healthy_worker)).index(project)
    committed = sorted(chunk.chunk_id for chunk in store.list_chunks([project.id]))
    assert committed

    # Change every file so the next run must re-embed all of them, then let
    # the accelerator die with the fallback forbidden.
    for index in range(3):
        (root / f"module_{index}.py").write_text(f"def changed_{index}():\n    return {index}\n")

    with pytest.raises(CodeIndexingError) as caught:
        indexer_with(_session_factory(_crashing_worker, strict=True)).index(project)

    assert caught.value.code is ErrorCode.BACKEND_UNAVAILABLE
    # The failed run staged everything and committed nothing: the previous
    # index is byte-for-byte what it was.
    assert sorted(chunk.chunk_id for chunk in store.list_chunks([project.id])) == committed
    assert len(store.list_files(project.id)) == good.indexed_files


def test_the_index_recovers_on_the_next_run_after_a_backend_failure(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=DIMENSION)

    def indexer_with(factory: Callable[[], PassageBackendSession]) -> Indexer:
        return Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=RecordingEmbedder(),
            lock_directory=tmp_path / "locks",
            segment_plan=SegmentPlan(max_tokens=64, max_items=4),
            passage_session_factory=factory,
            staging_directory=tmp_path / "staging",
        )

    with pytest.raises(CodeIndexingError):
        indexer_with(_session_factory(_unloadable_worker, strict=True)).index(project)

    # No file was stamped with the environment's failure, so the retry is a
    # full index rather than a run that skips everything it already "saw".
    recovered = indexer_with(_session_factory(_healthy_worker)).index(project)

    assert recovered.errors == []
    assert recovered.indexed_files == 3
    assert store.project_state(project.id) == "ready"
