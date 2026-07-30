import os
import sys
from pathlib import Path

import numpy as np
import pytest

from incode_mcp.acceptance import cosine_rows, top_k_overlap
from incode_mcp.backends import CPU_BACKEND
from incode_mcp.direct_onnx import DirectOnnxEmbedding
from incode_mcp.embedding import DEFAULT_DIMENSION, FastEmbedder
from incode_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig
from incode_mcp.worker_launcher import ExternalInterpreterLauncher

GOLDEN_PASSAGES = (
    "def authorize(user, action):\n    return action in user.permissions",
    "async def fetch_with_retry(client, url):\n    for attempt in range(3):\n        try:\n"
    "            return await client.get(url)\n        except TimeoutError:\n            pass",
    "class Transaction:\n    def commit(self) -> None:\n        self.connection.execute('COMMIT')",
    "SELECT project_id, updated_at FROM source_files WHERE digest = ?",
    "func (s *Store) Close() error { return s.database.Close() }",
    "public record UserId(UUID value) {}",
    "const normalizePath = (value: string) => value.replaceAll('\\\\', '/');",
    "fn parse_version(input: &str) -> Result<Version, Error> { input.parse() }",
    'resource "aws_s3_bucket" "artifacts" { bucket = var.artifact_bucket }',
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: index-worker",
    "CREATE INDEX source_chunks_vector_idx ON source_chunks "
    "USING hnsw (embedding vector_cosine_ops)",
    "def rollback_staged_commit(path: Path) -> None:\n    path.unlink(missing_ok=True)",
    "interface CacheEntry<T> { value: T; expiresAt: number; }",
    'class PermissionDenied(Exception):\n    """Raised when a principal lacks a capability."""',
    "pub async fn serve(listener: TcpListener) -> anyhow::Result<()> { loop { accept().await?; } }",
    "MATCH (owner)-[:MAINTAINS]->(module) RETURN owner, count(module)",
    "def cosine_similarity(left, right):\n    return dot(left, right) / (norm(left) * norm(right))",
    "ALTER TABLE projects ADD COLUMN indexed_at TIMESTAMP WITH TIME ZONE",
    "function exponentialBackoff(attempt) { return Math.min(1000 * 2 ** attempt, 30000); }",
    "package queue\n\nfunc (q *Queue) Push(item Item) { q.items = append(q.items, item) }",
)
GOLDEN_QUERIES = (
    "where are permission checks performed",
    "database transaction commit and rollback",
    "retry timed out network requests",
    "vector similarity index schema",
)


def _model_cache() -> Path:
    configured_cache = os.environ.get("INCODE_MODEL_TEST_CACHE")
    if not configured_cache:
        pytest.skip("set INCODE_MODEL_TEST_CACHE to opt into the real model test")
    return Path(configured_cache)


@pytest.mark.model
def test_prepared_model_embeds_without_network() -> None:
    cache = _model_cache()
    FastEmbedder(cache, offline=False).prepare()

    vector = FastEmbedder(cache, offline=True).embed_query("find permission checks")

    assert len(vector) == DEFAULT_DIMENSION


@pytest.mark.model
def test_the_real_worker_serves_embeddings_from_another_interpreter() -> None:
    """An accelerator runs in an environment of its own; prove that path works.

    This machine has no second environment to point at, so it points at its own
    interpreter through the external launcher: the code under test is the socket
    handshake and the model actually loading and embedding on the far side of
    it, neither of which cares whose environment answered.
    """
    cache = _model_cache()
    FastEmbedder(cache, offline=False).prepare()
    config = WorkerConfig(
        cache_directory=str(cache),
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        dimension=DEFAULT_DIMENSION,
        providers=CPU_BACKEND.providers,
        accelerator=CPU_BACKEND.accelerator.value,
    )

    with EmbeddingWorkerSession(
        config,
        effective_ceiling_bytes=4 * 1024**3,
        launcher=ExternalInterpreterLauncher(Path(sys.executable)),
    ) as session:
        info = session.initialize()
        session.probe()
        vectors = session.embed_passages(["def probe() -> int:\n    return 0\n"])

    assert info.dimension == DEFAULT_DIMENSION
    assert len(vectors) == 1
    assert len(vectors[0]) == DEFAULT_DIMENSION


@pytest.mark.model
def test_direct_onnx_passages_preserve_cpu_vectors_and_rankings() -> None:
    cache = _model_cache()
    reference = FastEmbedder(cache, offline=True, threads=1)
    cpu_vectors = np.asarray(reference.embed_passages(list(GOLDEN_PASSAGES)), dtype=np.float32)
    query_vectors = np.asarray(
        [reference.embed_query(query) for query in GOLDEN_QUERIES],
        dtype=np.float32,
    )
    direct = DirectOnnxEmbedding(
        cache,
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        providers=CPU_BACKEND.providers,
        accelerator="",
    )
    direct_vectors = np.asarray(
        list(direct.passage_embed(list(GOLDEN_PASSAGES))),
        dtype=np.float32,
    )

    assert direct_vectors.shape == cpu_vectors.shape == (len(GOLDEN_PASSAGES), DEFAULT_DIMENSION)
    assert np.all(np.isfinite(direct_vectors))
    assert np.allclose(np.linalg.norm(direct_vectors, axis=1), 1.0, atol=1e-5)
    assert float(np.min(cosine_rows(cpu_vectors, direct_vectors))) >= 0.999
    assert top_k_overlap(query_vectors, cpu_vectors, direct_vectors, k=5) >= 0.99
