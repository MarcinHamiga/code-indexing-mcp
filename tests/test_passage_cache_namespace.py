from dataclasses import replace
from pathlib import Path

import pytest

from code_indexing_mcp.application import RuntimePaths
from code_indexing_mcp.backend_coordinator import BackendCoordinator
from code_indexing_mcp.backends import CPU_BACKEND, Accelerator, BackendSelection, Runtime
from code_indexing_mcp.embedding import DEFAULT_MODEL, FastEmbedder
from code_indexing_mcp.mlx_backend import converted_weights_path


def _coordinator(tmp_path: Path) -> tuple[BackendCoordinator, Path]:
    cache = tmp_path / "models"
    snapshot = cache / f"models--{DEFAULT_MODEL.replace('/', '--')}" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.onnx").write_bytes(b"source weights")
    refs = snapshot.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text("revision")
    weights = converted_weights_path(cache, snapshot)
    weights.parent.mkdir()
    coordinator = object.__new__(BackendCoordinator)
    coordinator.paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    coordinator.embedder = FastEmbedder(cache_directory=cache, offline=True)
    coordinator._runtime_fallback = None
    coordinator.backend_selection = BackendSelection(
        requested=Accelerator.MLX,
        descriptor=replace(
            CPU_BACKEND, accelerator=Accelerator.MLX, runtime=Runtime.MLX, runtime_version="test"
        ),
        available_providers=(),
    )
    return coordinator, weights


def test_mlx_namespace_changes_with_converted_weights(tmp_path: Path) -> None:
    coordinator, weights = _coordinator(tmp_path)
    weights.write_bytes(b"first weights")
    first = coordinator.passage_cache_namespace("project")
    weights.write_bytes(b"other weights")
    second = coordinator.passage_cache_namespace("project")
    assert first is not None and second is not None
    assert first.artifact_digest != second.artifact_digest
    assert first.tokenizer_digest == second.tokenizer_digest


def test_mlx_namespace_is_disabled_without_converted_weights(tmp_path: Path) -> None:
    coordinator, _ = _coordinator(tmp_path)
    assert coordinator.passage_cache_namespace("project") is None


def test_mlx_namespace_changes_with_conversion_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from code_indexing_mcp import mlx_backend

    coordinator, weights = _coordinator(tmp_path)
    weights.write_bytes(b"same weights")
    first = coordinator.passage_cache_namespace("project")
    monkeypatch.setattr(mlx_backend, "WEIGHT_LAYOUT_VERSION", mlx_backend.WEIGHT_LAYOUT_VERSION + 1)
    snapshot = (
        coordinator.embedder.cache_directory
        / f"models--{DEFAULT_MODEL.replace('/', '--')}"
        / "snapshots"
        / "revision"
    )
    converted_weights_path(coordinator.embedder.cache_directory, snapshot).write_bytes(
        b"same weights"
    )
    second = coordinator.passage_cache_namespace("project")
    assert first is not None and second is not None
    assert first.artifact_digest != second.artifact_digest


def test_cpu_namespace_is_distinct_from_the_mlx_one(tmp_path: Path) -> None:
    """EMB-9: CPU vectors need an identity of their own, with the source-model
    digest and the CPU runtime, never the MLX conversion's."""
    coordinator, weights = _coordinator(tmp_path)
    weights.write_bytes(b"first weights")
    mlx = coordinator.passage_cache_namespace("project")
    cpu = coordinator.cpu_passage_cache_namespace("project")
    assert mlx is not None and cpu is not None
    assert mlx.producer == "mlx-float32"
    assert cpu.producer == "cpu-float32"
    assert cpu.artifact_digest == mlx.tokenizer_digest != mlx.artifact_digest
    assert cpu.project_id == mlx.project_id
