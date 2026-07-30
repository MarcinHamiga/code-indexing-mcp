from __future__ import annotations

import numpy as np
import pytest

from incode_mcp import accelerator_probe, embedding_worker
from incode_mcp.backends import CPU_PROVIDER, Accelerator
from incode_mcp.embedding import DEFAULT_DIMENSION, DEFAULT_MODEL, PROBE_TEXTS


class _WebGpuModel:
    resolved_providers = ("WebGpuExecutionProvider", CPU_PROVIDER)

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        assert texts == list(PROBE_TEXTS)
        row = np.zeros(DEFAULT_DIMENSION, dtype=np.float32)
        row[0] = 1.0
        return [row.copy() for _ in texts]


def test_plugin_provider_is_discovered_by_loading_the_direct_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        accelerator_probe,
        "available_execution_providers",
        lambda: (CPU_PROVIDER,),
    )
    monkeypatch.setattr(embedding_worker, "_load_model", lambda config: _WebGpuModel())

    report = accelerator_probe.probe(
        Accelerator.WEBGPU,
        offline=True,
        model_id=DEFAULT_MODEL,
        dimension=DEFAULT_DIMENSION,
    )

    assert report["accelerator"] == "webgpu"
    assert report["resolved_providers"] == [
        "WebGpuExecutionProvider",
        CPU_PROVIDER,
    ]
    assert report["providers"] == [CPU_PROVIDER, "WebGpuExecutionProvider"]


class _OpaqueFastEmbedModel:
    """A FastEmbed-shaped model whose private layout resolution cannot read."""

    def passage_embed(self, texts: list[str]) -> list[np.ndarray]:
        row = np.zeros(DEFAULT_DIMENSION, dtype=np.float32)
        row[0] = 1.0
        return [row.copy() for _ in texts]


def test_unknown_fastembed_providers_are_tolerated_rather_than_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider resolution walks FastEmbed's private layout, so an empty result
    means "unknown", and a FastEmbed refactor must not fail a working CUDA
    probe. Only the direct backends report their sessions authoritatively."""
    monkeypatch.setattr(
        accelerator_probe,
        "available_execution_providers",
        lambda: ("CUDAExecutionProvider", CPU_PROVIDER),
    )
    monkeypatch.setattr(embedding_worker, "_load_model", lambda config: _OpaqueFastEmbedModel())

    report = accelerator_probe.probe(
        Accelerator.CUDA,
        offline=True,
        model_id=DEFAULT_MODEL,
        dimension=DEFAULT_DIMENSION,
    )

    assert report["ok"] is True
    assert report["resolved_providers"] == []
    assert report["providers"] == ["CUDAExecutionProvider", CPU_PROVIDER]


def test_a_direct_session_reporting_no_providers_fails_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenDirectModel(_WebGpuModel):
        resolved_providers = ()

    monkeypatch.setattr(
        accelerator_probe,
        "available_execution_providers",
        lambda: (CPU_PROVIDER,),
    )
    monkeypatch.setattr(embedding_worker, "_load_model", lambda config: _BrokenDirectModel())

    with pytest.raises(RuntimeError, match="cannot be verified"):
        accelerator_probe.probe(
            Accelerator.WEBGPU,
            offline=True,
            model_id=DEFAULT_MODEL,
            dimension=DEFAULT_DIMENSION,
        )


def test_a_direct_session_that_dropped_its_provider_fails_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CpuOnlyDirectModel(_WebGpuModel):
        resolved_providers = (CPU_PROVIDER,)

    monkeypatch.setattr(
        accelerator_probe,
        "available_execution_providers",
        lambda: (CPU_PROVIDER,),
    )
    monkeypatch.setattr(embedding_worker, "_load_model", lambda config: _CpuOnlyDirectModel())

    with pytest.raises(RuntimeError, match="session runs on"):
        accelerator_probe.probe(
            Accelerator.WEBGPU,
            offline=True,
            model_id=DEFAULT_MODEL,
            dimension=DEFAULT_DIMENSION,
        )
