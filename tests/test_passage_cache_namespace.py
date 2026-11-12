from dataclasses import replace
from pathlib import Path

import pytest

from code_indexing_mcp.application import RuntimePaths
from code_indexing_mcp.backend_coordinator import BackendCoordinator
from code_indexing_mcp.backends import Accelerator, BackendSelection, backend_for
from code_indexing_mcp.direct_onnx import converted_onnx_path
from code_indexing_mcp.embedding import DEFAULT_MODEL, FastEmbedder
from code_indexing_mcp.mlx_backend import converted_weights_path


def _coordinator(tmp_path: Path) -> tuple[BackendCoordinator, Path, Path]:
    cache = tmp_path / "models"
    snapshot = cache / f"models--{DEFAULT_MODEL.replace('/', '--')}" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"source weights")
    refs = snapshot.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text("revision")
    weights = converted_weights_path(cache, snapshot)
    weights.parent.mkdir()
    onnx_file = converted_onnx_path(cache, snapshot)
    onnx_file.parent.mkdir()
    mlx = backend_for(Accelerator.MLX)
    assert mlx is not None
    coordinator = object.__new__(BackendCoordinator)
    coordinator.paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    coordinator.embedder = FastEmbedder(cache_directory=cache, offline=True)
    coordinator._runtime_fallback = None
    coordinator.backend_selection = BackendSelection(
        requested=Accelerator.MLX,
        descriptor=replace(mlx, runtime_version="test"),
        available_providers=(),
    )
    return coordinator, weights, onnx_file


def test_mlx_namespace_changes_with_converted_weights(tmp_path: Path) -> None:
    coordinator, weights, _ = _coordinator(tmp_path)
    weights.write_bytes(b"first weights")
    first = coordinator.passage_cache_namespace("project")
    weights.write_bytes(b"other weights")
    second = coordinator.passage_cache_namespace("project")
    assert first is not None and second is not None
    assert first.artifact_digest != second.artifact_digest
    assert first.tokenizer_digest == second.tokenizer_digest


def test_mlx_and_cpu_namespaces_are_disabled_without_converted_artifacts(tmp_path: Path) -> None:
    coordinator, _, _ = _coordinator(tmp_path)
    assert coordinator.passage_cache_namespace("project") is None
    assert coordinator.cpu_passage_cache_namespace("project") is None


def test_mlx_and_cpu_namespaces_change_with_conversion_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from code_indexing_mcp import direct_onnx, mlx_backend

    coordinator, weights, onnx_file = _coordinator(tmp_path)
    weights.write_bytes(b"same weights")
    onnx_file.write_bytes(b"same onnx")
    first_mlx = coordinator.passage_cache_namespace("project")
    first_cpu = coordinator.cpu_passage_cache_namespace("project")

    monkeypatch.setattr(mlx_backend, "WEIGHT_LAYOUT_VERSION", mlx_backend.WEIGHT_LAYOUT_VERSION + 1)
    monkeypatch.setattr(direct_onnx, "ONNX_LAYOUT_VERSION", direct_onnx.ONNX_LAYOUT_VERSION + 1)
    snapshot = (
        coordinator.embedder.cache_directory
        / f"models--{DEFAULT_MODEL.replace('/', '--')}"
        / "snapshots"
        / "revision"
    )
    converted_weights_path(coordinator.embedder.cache_directory, snapshot).write_bytes(
        b"same weights"
    )
    converted_onnx_path(coordinator.embedder.cache_directory, snapshot).write_bytes(b"same onnx")

    second_mlx = coordinator.passage_cache_namespace("project")
    second_cpu = coordinator.cpu_passage_cache_namespace("project")
    assert first_mlx is not None and second_mlx is not None
    assert first_cpu is not None and second_cpu is not None
    assert first_mlx.artifact_digest != second_mlx.artifact_digest
    assert first_cpu.artifact_digest != second_cpu.artifact_digest


def test_cpu_namespace_is_distinct_from_the_mlx_one(tmp_path: Path) -> None:
    coordinator, weights, onnx_file = _coordinator(tmp_path)
    weights.write_bytes(b"first weights")
    onnx_file.write_bytes(b"first onnx")
    mlx = coordinator.passage_cache_namespace("project")
    cpu = coordinator.cpu_passage_cache_namespace("project")
    assert mlx is not None and cpu is not None
    assert mlx.producer == "mlx-float16"
    assert mlx.precision == "float16"
    assert cpu.producer == "cpu-float32"
    assert cpu.tokenizer_digest == mlx.tokenizer_digest
    assert cpu.artifact_digest != mlx.artifact_digest
    assert cpu.project_id == mlx.project_id
