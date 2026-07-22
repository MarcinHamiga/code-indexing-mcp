import os
from pathlib import Path

import pytest

from incode_mcp.embedding import DEFAULT_DIMENSION, FastEmbedder


@pytest.mark.model
def test_prepared_model_embeds_without_network() -> None:
    configured_cache = os.environ.get("INCODE_MODEL_TEST_CACHE")
    if not configured_cache:
        pytest.skip("set INCODE_MODEL_TEST_CACHE to opt into the real model test")
    cache = Path(configured_cache)
    FastEmbedder(cache, offline=False).prepare()

    vector = FastEmbedder(cache, offline=True).embed_query("find permission checks")

    assert len(vector) == DEFAULT_DIMENSION
