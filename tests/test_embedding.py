from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from incode_mcp import embedding
from incode_mcp.embedding import (
    FastEmbedder,
    PassageCandidate,
    SegmentPlan,
    compose_passage,
    embed_planned_segments,
    plan_passages,
    resolve_session_providers,
    resolve_tokenizer,
)


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


def test_compose_passage_matches_the_extractor_layout() -> None:
    assert compose_passage("kind: module", "value = 1") == "kind: module\nvalue = 1"
    assert compose_passage("", "value = 1") == "value = 1"


def test_resolve_tokenizer_finds_the_nested_fastembed_tokenizer() -> None:
    class Tokenizer:
        def encode(self, text: str) -> object:
            return text

    class Inner:
        tokenizer = Tokenizer()

    class Model:
        model = Inner()

    assert isinstance(resolve_tokenizer(Model()), Tokenizer)


def test_resolve_tokenizer_returns_none_when_the_layout_moved() -> None:
    class Model:
        pass

    assert resolve_tokenizer(Model()) is None


def test_direct_model_reports_the_provider_attached_through_the_plugin_api() -> None:
    class CpuLookingSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    class DirectModel:
        resolved_providers = ("WebGpuExecutionProvider", "CPUExecutionProvider")
        model = CpuLookingSession()

    assert resolve_session_providers(DirectModel()) == (
        "WebGpuExecutionProvider",
        "CPUExecutionProvider",
    )


def test_planning_without_a_tokenizer_leaves_candidates_whole() -> None:
    candidates = [PassageCandidate("kind: module", "value = 1")]

    windows = plan_passages(None, candidates, SegmentPlan())

    assert [(window.start_char, window.end_char) for window in windows[0]] == [(0, 9)]


def test_embedding_without_a_tokenizer_sends_the_whole_candidate() -> None:
    seen: list[list[str]] = []

    def embed(texts: list[str]) -> list[str]:
        seen.append(texts)
        return ["vector" for _ in texts]

    candidates = [PassageCandidate("kind: module", "value = 1")]
    result = embed_planned_segments(None, embed, candidates, SegmentPlan())

    assert seen == [["kind: module\nvalue = 1"]]
    assert len(result[0]) == 1
