import os
import sys
from pathlib import Path

import pytest

from incode_mcp.backends import CPU_BACKEND
from incode_mcp.embedding import DEFAULT_DIMENSION, FastEmbedder
from incode_mcp.embedding_worker import EmbeddingWorkerSession, WorkerConfig
from incode_mcp.worker_launcher import ExternalInterpreterLauncher


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
