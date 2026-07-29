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
