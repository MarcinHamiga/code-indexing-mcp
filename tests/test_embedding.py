from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from incode_mcp import embedding
from incode_mcp.embedding import FastEmbedder


class _FakeModel:
    def passage_embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_concurrent_first_use_builds_a_single_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The daemon serves clients on separate threads; one model, not one per thread."""
    builds: list[float] = []

    def slow_build(**_: Any) -> _FakeModel:
        builds.append(time.monotonic())
        # Widen the window a naive check-then-set would race through.
        time.sleep(0.05)
        return _FakeModel()

    monkeypatch.setattr(embedding, "TextEmbedding", slow_build)
    embedder = FastEmbedder(tmp_path / "models")
    barrier = threading.Barrier(8)

    def prepare() -> None:
        barrier.wait()
        embedder.prepare()

    threads = [threading.Thread(target=prepare) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert len(builds) == 1
