"""Opt-in correctness and performance gates for dedicated accelerator runners."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_model_integration import GOLDEN_PASSAGES, GOLDEN_QUERIES

from code_indexing_mcp.accelerator_env import load_environment
from code_indexing_mcp.acceptance import cosine_rows, top_k_overlap
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.backends import Accelerator, backend_for
from code_indexing_mcp.benchmark import write_benchmark_corpus
from code_indexing_mcp.embedding import DEFAULT_DIMENSION, FastEmbedder
from code_indexing_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig
from code_indexing_mcp.settings import IndexSettings
from code_indexing_mcp.worker_launcher import ExternalInterpreterLauncher

GATED_ACCELERATORS = (Accelerator.MLX, Accelerator.WEBGPU, Accelerator.MIGRAPHX)
_NAMES = ", ".join(accelerator.value for accelerator in GATED_ACCELERATORS)


def _accelerator() -> Accelerator:
    configured = os.environ.get("CODE_INDEXING_TEST_ACCELERATOR")
    if not configured:
        pytest.skip(f"set CODE_INDEXING_TEST_ACCELERATOR to one of {_NAMES} on a prepared runner")
    try:
        accelerator = Accelerator(configured.lower())
    except ValueError:
        pytest.fail(f"CODE_INDEXING_TEST_ACCELERATOR names unknown backend {configured!r}")
    if accelerator not in GATED_ACCELERATORS:
        pytest.fail(f"CODE_INDEXING_TEST_ACCELERATOR must be one of {_NAMES}")
    return accelerator


def _cache() -> Path:
    configured = os.environ.get("CODE_INDEXING_MODEL_TEST_CACHE")
    if not configured:
        pytest.fail("CODE_INDEXING_MODEL_TEST_CACHE must point at the prepared model cache")
    return Path(configured)


def _prepared_worker(accelerator: Accelerator, cache: Path) -> EmbeddingWorkerSession:
    status = load_environment(Path.cwd())
    record = status.environment
    if record is None:
        pytest.fail(
            "CODE_INDEXING_ACCEL_ENV must point at a verified accelerator record"
            + (f": {status.reason}" if status.reason else "")
        )
    if record.accelerator is not accelerator:
        pytest.fail(
            f"accelerator record prepares {record.accelerator.value}, "
            f"not requested {accelerator.value}"
        )
    descriptor = backend_for(accelerator)
    assert descriptor is not None
    return EmbeddingWorkerSession(
        WorkerConfig(
            cache_directory=str(cache),
            offline=True,
            threads=1,
            enable_cpu_mem_arena=False,
            dimension=DEFAULT_DIMENSION,
            providers=descriptor.providers,
            accelerator=accelerator.value,
        ),
        effective_ceiling_bytes=4 * 1024**3,
        launcher=ExternalInterpreterLauncher(record.interpreter),
    )


@pytest.mark.accelerator
def test_prepared_accelerator_preserves_cpu_vectors_and_rankings() -> None:
    accelerator = _accelerator()
    cache = _cache()
    reference = FastEmbedder(cache, offline=True, threads=1)
    cpu_vectors = np.asarray(reference.embed_passages(list(GOLDEN_PASSAGES)), dtype=np.float32)
    query_vectors = np.asarray(
        [reference.embed_query(query) for query in GOLDEN_QUERIES],
        dtype=np.float32,
    )
    descriptor = backend_for(accelerator)
    assert descriptor is not None

    with _prepared_worker(accelerator, cache) as worker:
        info = worker.initialize()
        worker.probe()
        accelerator_vectors = np.asarray(
            worker.embed_passages(list(GOLDEN_PASSAGES)),
            dtype=np.float32,
        )

    assert descriptor.provider in info.resolved_providers
    assert accelerator_vectors.shape == cpu_vectors.shape
    assert np.all(np.isfinite(accelerator_vectors))
    assert np.allclose(np.linalg.norm(accelerator_vectors, axis=1), 1.0, atol=1e-5)
    assert float(np.min(cosine_rows(cpu_vectors, accelerator_vectors))) >= 0.999
    assert top_k_overlap(query_vectors, cpu_vectors, accelerator_vectors, k=5) >= 0.99


def _forced_index_seconds(app: Application, root: Path) -> tuple[float, int]:
    project = app.init_project(root)
    started = time.monotonic()
    report = app.index_project(project.id, force=True)
    return time.monotonic() - started, report.embedded_chunks


@pytest.mark.accelerator
def test_accelerator_forced_index_is_faster_on_at_least_one_thousand_chunks(
    tmp_path: Path,
) -> None:
    accelerator = _accelerator()
    cache = _cache()
    cpu_root = tmp_path / "cpu-corpus"
    accelerator_root = tmp_path / "accelerator-corpus"
    write_benchmark_corpus(cpu_root, files=500, functions_per_file=2)
    write_benchmark_corpus(accelerator_root, files=500, functions_per_file=2)
    base = replace(
        IndexSettings.from_environment(),
        embedding_batch_size=8,
        embedding_batch_auto=False,
        embedding_strict=True,
        index_execution="worker",
        broker_mode="off",
    )
    cpu = Application(
        RuntimePaths(data=tmp_path / "cpu-data", cache=tmp_path / "cpu-cache"),
        embedder=FastEmbedder(cache, offline=True, threads=base.embedding_threads),
        cwd=cpu_root,
        settings=replace(base, embedding_accelerator=Accelerator.CPU),
    )
    accelerated = Application(
        RuntimePaths(
            data=tmp_path / "accelerator-data",
            cache=tmp_path / "accelerator-cache",
        ),
        embedder=FastEmbedder(cache, offline=True, threads=base.embedding_threads),
        cwd=accelerator_root,
        settings=replace(base, embedding_accelerator=accelerator),
    )

    cpu_seconds, cpu_chunks = _forced_index_seconds(cpu, cpu_root)
    accelerator_seconds, accelerator_chunks = _forced_index_seconds(
        accelerated,
        accelerator_root,
    )

    assert cpu_chunks >= 1_000
    assert accelerator_chunks == cpu_chunks
    speedup = cpu_seconds / accelerator_seconds
    assert speedup >= 1.25, (
        f"{accelerator.value} indexed {accelerator_chunks} chunks in "
        f"{accelerator_seconds:.3f}s vs CPU {cpu_seconds:.3f}s ({speedup:.2f}x)"
    )


def test_similarity_metrics_compare_corresponding_rows_and_rankings() -> None:
    reference = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    assert np.all(cosine_rows(reference, candidates) > 0.9)
    assert top_k_overlap(queries, reference, candidates, k=1) == 1.0
