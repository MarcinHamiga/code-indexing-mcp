"""Unit tests for the MLX passage backend.

These run in the ``mlx`` extra's environment, which is the only one that has
MLX in it:

    uv run --extra mlx pytest tests/test_mlx_backend.py

The forward pass is checked against an independent NumPy implementation of the
same exported graph, so a divergence shows up as a numeric disagreement between
two implementations rather than as a plausible vector nobody can check. Parity
with the real CPU model is a separate, opt-in gate that has to run across two
environments -- see ``tests/test_accelerator_acceptance.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

mx = pytest.importorskip("mlx.core", reason="the mlx extra's environment is required")
onnx = pytest.importorskip("onnx", reason="the mlx extra's environment is required")

from onnx import helper, numpy_helper  # noqa: E402

from code_indexing_mcp.mlx_backend import (  # noqa: E402
    MASK_FILL,
    JinaBertMlx,
    ModelConfig,
    converted_weights_path,
    ensure_converted_weights,
    extract_weights,
    mask_mean_normalize,
    read_model_config,
)

HIDDEN = 8
HEADS = 2
LAYERS = 2
INTERMEDIATE = 16
VOCAB = 11
EPS = 1e-12

CONFIG = ModelConfig(
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    num_attention_heads=HEADS,
    intermediate_size=INTERMEDIATE,
    layer_norm_eps=EPS,
)


def _random(*shape: int) -> NDArray[np.float32]:
    generator = np.random.default_rng(abs(hash(shape)) % (2**32))
    return generator.standard_normal(shape).astype(np.float32) * np.float32(0.2)


def _config_json(**overrides: Any) -> dict[str, Any]:
    document = {
        "model_type": "bert",
        "position_embedding_type": "alibi",
        "feed_forward_type": "geglu",
        "hidden_act": "gelu",
        "emb_pooler": "mean",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "intermediate_size": INTERMEDIATE,
        "layer_norm_eps": EPS,
    }
    document.update(overrides)
    return document


def _slopes() -> NDArray[np.float32]:
    return np.asarray(
        [[[-(2.0 ** -(index + 1))]] for index in range(HEADS)],
        dtype=np.float32,
    )


def _stub_graph(
    *,
    weight_shapes: dict[str, tuple[int, ...]] | None = None,
    rename: dict[str, str] | None = None,
) -> onnx.ModelProto:
    """Build a graph carrying the names and shapes extraction reads.

    Only the initializers and the nodes that point at them matter here: nothing
    executes this graph, so it names the tensors of the real export without
    reproducing its 1,823 nodes.
    """
    rename = rename or {}
    weight_shapes = weight_shapes or {}
    initializers = []
    nodes = []

    def matmul(node_name: str, weight_name: str, shape: tuple[int, ...]) -> None:
        shape = weight_shapes.get(node_name, shape)
        initializers.append(numpy_helper.from_array(_random(*shape), weight_name))
        nodes.append(
            helper.make_node(
                "MatMul",
                ["hidden", weight_name],
                [f"{node_name}_output_0"],
                name=rename.get(node_name, node_name),
            )
        )

    def tensor(name: str, *shape: int) -> None:
        initializers.append(numpy_helper.from_array(_random(*shape), rename.get(name, name)))

    tensor("embeddings.word_embeddings.weight", VOCAB, HIDDEN)
    tensor("embeddings.token_type_embeddings.weight", 2, HIDDEN)
    tensor("embeddings.LayerNorm.weight", HIDDEN)
    tensor("embeddings.LayerNorm.bias", HIDDEN)

    slopes = numpy_helper.from_array(_slopes(), "slopes")
    nodes.append(
        helper.make_node(
            "Constant",
            [],
            ["/encoder/Constant_7_output_0"],
            name="/encoder/Constant_7",
            value=slopes,
        )
    )
    nodes.append(
        helper.make_node(
            "Mul",
            ["/encoder/Constant_7_output_0", "distance"],
            ["/encoder/Mul_1_output_0"],
            name="/encoder/Mul_1",
        )
    )

    for index in range(LAYERS):
        prefix = f"/encoder/layer.{index}"
        parameter = f"encoder.layer.{index}"
        for role in ("query", "key", "value"):
            matmul(f"{prefix}/attention/self/{role}/MatMul", f"w.{index}.{role}", (HIDDEN, HIDDEN))
            tensor(f"{parameter}.attention.self.{role}.bias", HIDDEN)
        for role in ("layer_norm_q", "layer_norm_k"):
            tensor(f"{parameter}.attention.self.{role}.weight", HIDDEN)
            tensor(f"{parameter}.attention.self.{role}.bias", HIDDEN)
        matmul(f"{prefix}/attention/output/dense/MatMul", f"w.{index}.out", (HIDDEN, HIDDEN))
        tensor(f"{parameter}.attention.output.dense.bias", HIDDEN)
        tensor(f"{parameter}.attention.output.LayerNorm.weight", HIDDEN)
        tensor(f"{parameter}.attention.output.LayerNorm.bias", HIDDEN)
        tensor(f"{parameter}.layer_norm_1.weight", HIDDEN)
        tensor(f"{parameter}.layer_norm_1.bias", HIDDEN)
        matmul(f"{prefix}/mlp/up_gated_layer/MatMul", f"w.{index}.up", (HIDDEN, 2 * INTERMEDIATE))
        matmul(f"{prefix}/mlp/down_layer/MatMul", f"w.{index}.down", (INTERMEDIATE, HIDDEN))
        tensor(f"{parameter}.mlp.down_layer.bias", HIDDEN)
        tensor(f"{parameter}.layer_norm_2.weight", HIDDEN)
        tensor(f"{parameter}.layer_norm_2.bias", HIDDEN)

    graph = helper.make_graph(
        nodes,
        "stub",
        [helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, ["batch", "sequence"])],
        [
            helper.make_tensor_value_info(
                "last_hidden_state", onnx.TensorProto.FLOAT, ["batch", "sequence", HIDDEN]
            )
        ],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def _snapshot(directory: Path, **graph_options: Any) -> Path:
    """Write a snapshot-shaped directory holding a stub artifact and config."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(_config_json()), encoding="utf-8")
    artifact = directory / "onnx"
    artifact.mkdir(exist_ok=True)
    onnx.save(_stub_graph(**graph_options), str(artifact / "model.onnx"))
    return directory


# -- configuration ---------------------------------------------------------


def test_the_config_of_the_supported_architecture_is_read(tmp_path: Path) -> None:
    config = read_model_config(_snapshot(tmp_path / "snapshot"))

    assert config == CONFIG
    assert config.head_dim == HIDDEN // HEADS


@pytest.mark.parametrize(
    "field",
    ["model_type", "position_embedding_type", "feed_forward_type", "hidden_act", "emb_pooler"],
)
def test_a_config_describing_another_architecture_is_refused(tmp_path: Path, field: str) -> None:
    """Reproducing a graph means refusing to guess at a graph that moved.

    Every one of these fields names something this implementation hard-codes, so
    a changed value is a different model wearing the same name.
    """
    directory = _snapshot(tmp_path / "snapshot")
    (directory / "config.json").write_text(
        json.dumps(_config_json(**{field: "something-else"})), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=field):
        read_model_config(directory)


def test_a_hidden_size_the_heads_do_not_divide_is_refused(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    (directory / "config.json").write_text(
        json.dumps(_config_json(hidden_size=9)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="attention heads"):
        read_model_config(directory)


# -- weight extraction -----------------------------------------------------


def test_every_weight_and_the_alibi_slopes_come_out_of_the_artifact(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")

    weights = extract_weights(directory / "onnx" / "model.onnx", CONFIG)

    assert weights["embeddings.word_embeddings"].shape == (VOCAB, HIDDEN)
    assert weights["embeddings.token_type"].shape == (2, HIDDEN)
    assert weights["alibi_slopes"].shape == (HEADS, 1, 1)
    # Read from the graph rather than recomputed, so a model whose slopes were
    # not the textbook powers of two is reproduced instead of replaced.
    assert np.allclose(weights["alibi_slopes"], _slopes())
    for index in range(LAYERS):
        assert weights[f"layers.{index}.query.weight"].shape == (HIDDEN, HIDDEN)
        assert weights[f"layers.{index}.up_gated.weight"].shape == (HIDDEN, 2 * INTERMEDIATE)
        assert weights[f"layers.{index}.down.weight"].shape == (INTERMEDIATE, HIDDEN)
        assert weights[f"layers.{index}.norm_q.bias"].shape == (HIDDEN,)
    assert all(value.dtype == np.float32 for value in weights.values())


def test_a_renamed_node_is_refused_rather_than_silently_skipped(tmp_path: Path) -> None:
    directory = _snapshot(
        tmp_path / "snapshot",
        rename={"/encoder/layer.1/mlp/down_layer/MatMul": "/encoder/layer.1/mlp/renamed/MatMul"},
    )

    with pytest.raises(ValueError, match="down_layer/MatMul"):
        extract_weights(directory / "onnx" / "model.onnx", CONFIG)


def test_a_weight_of_the_wrong_shape_is_refused(tmp_path: Path) -> None:
    directory = _snapshot(
        tmp_path / "snapshot",
        weight_shapes={"/encoder/layer.0/mlp/up_gated_layer/MatMul": (HIDDEN, INTERMEDIATE)},
    )

    with pytest.raises(ValueError, match="up_gated"):
        extract_weights(directory / "onnx" / "model.onnx", CONFIG)


def test_a_non_float32_weight_is_refused(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    model_path = directory / "onnx" / "model.onnx"
    model = onnx.load(str(model_path))
    name = "embeddings.LayerNorm.weight"
    tensor = next(tensor for tensor in model.graph.initializer if tensor.name == name)
    values = numpy_helper.to_array(tensor).astype(np.float16)
    tensor.CopyFrom(numpy_helper.from_array(values, name))
    onnx.save(model, str(model_path))

    with pytest.raises(ValueError, match=r"embedding normalization.*FLOAT16.*expected FLOAT"):
        extract_weights(model_path, CONFIG)


def test_a_weight_stored_outside_the_artifact_is_refused(tmp_path: Path) -> None:
    """The artifact is parsed without its external data and the snapshot carries
    no sidecar, so an initializer pointing outside it has to say so here rather
    than resolve a path against whatever the working directory happens to be."""
    directory = _snapshot(tmp_path / "snapshot")
    model_path = directory / "onnx" / "model.onnx"
    model = onnx.load(str(model_path))
    tensor = next(
        tensor
        for tensor in model.graph.initializer
        if tensor.name == "embeddings.word_embeddings.weight"
    )
    tensor.ClearField("raw_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    entry = tensor.external_data.add()
    entry.key, entry.value = "location", "model.onnx_data"
    onnx.save(model, str(model_path))

    with pytest.raises(ValueError, match=r"word embedding.*stored outside model.onnx"):
        extract_weights(model_path, CONFIG)


# -- conversion cache ------------------------------------------------------


def test_conversion_runs_once_and_is_reused_by_revision(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "0a1b2c3")
    cache = tmp_path / "cache"
    conversions = 0

    def counting(model_path: Path, config: ModelConfig) -> dict[str, NDArray[np.float32]]:
        nonlocal conversions
        conversions += 1
        return extract_weights(model_path, config)

    first = ensure_converted_weights(directory, cache, CONFIG, convert=counting)
    second = ensure_converted_weights(directory, cache, CONFIG, convert=counting)

    assert first == second == converted_weights_path(cache, directory)
    assert first.is_file()
    assert conversions == 1
    # The revision the snapshot resolved to is in the name, so a model that
    # moves is converted again instead of being read from a stale file.
    assert "0a1b2c3" in first.name
    assert list(cache.rglob("*.tmp*")) == []


def test_a_conversion_of_another_revision_is_discarded(tmp_path: Path) -> None:
    """Each conversion is 600 MB, and nothing else ever revisits the one this
    installation stopped resolving to."""
    cache = tmp_path / "cache"
    superseded = converted_weights_path(cache, tmp_path / "0a1b2c3")
    superseded.parent.mkdir(parents=True, exist_ok=True)
    superseded.write_bytes(b"an earlier revision")
    in_flight = superseded.with_name("4d5e6f7-jina-v1-f32.999999.tmp.safetensors")
    in_flight.write_bytes(b"another process, mid-conversion")

    path = ensure_converted_weights(_snapshot(tmp_path / "4d5e6f7"), cache, CONFIG)

    assert path.is_file()
    assert not superseded.exists()
    # Another process's unfinished write is not this one's to remove.
    assert in_flight.is_file()


def test_a_converted_file_holds_every_extracted_tensor(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")

    path = ensure_converted_weights(directory, tmp_path / "cache", CONFIG)

    loaded = mx.load(str(path))
    expected = extract_weights(directory / "onnx" / "model.onnx", CONFIG)
    assert set(loaded) == set(expected)
    for name, value in expected.items():
        assert np.allclose(np.asarray(loaded[name]), value)


# -- a NumPy statement of the same graph -----------------------------------


def _layer_norm(x: NDArray[Any], weight: NDArray[Any], bias: NDArray[Any]) -> NDArray[Any]:
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = np.mean(np.square(centred), axis=-1, keepdims=True)
    return centred / np.sqrt(variance + EPS) * weight + bias


def _gelu(x: NDArray[Any]) -> NDArray[Any]:
    return x * 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _reference_forward(
    weights: dict[str, NDArray[np.float32]],
    input_ids: NDArray[np.int64],
    attention_mask: NDArray[np.int64],
) -> NDArray[np.float32]:
    """The exported graph, written out again in NumPy."""
    _, sequence = input_ids.shape
    head_dim = HIDDEN // HEADS
    hidden = weights["embeddings.word_embeddings"][input_ids] + weights["embeddings.token_type"][0]
    hidden = _layer_norm(hidden, weights["embeddings.norm.weight"], weights["embeddings.norm.bias"])
    positions = np.arange(sequence)
    distance = np.abs(positions[:, None] - positions[None, :]).astype(np.float32)
    bias = weights["alibi_slopes"] * distance + (
        (1.0 - attention_mask.astype(np.float32))[:, None, None, :] * MASK_FILL
    )

    for index in range(LAYERS):
        layer = f"layers.{index}"

        def project(role: str, source: NDArray[Any], layer: str = layer) -> NDArray[Any]:
            return source @ weights[f"{layer}.{role}.weight"] + weights[f"{layer}.{role}.bias"]

        def heads(value: NDArray[Any]) -> NDArray[Any]:
            return value.reshape(*value.shape[:2], HEADS, head_dim).transpose(0, 2, 1, 3)

        query = _layer_norm(
            project("query", hidden),
            weights[f"{layer}.norm_q.weight"],
            weights[f"{layer}.norm_q.bias"],
        )
        key = _layer_norm(
            project("key", hidden),
            weights[f"{layer}.norm_k.weight"],
            weights[f"{layer}.norm_k.bias"],
        )
        value = project("value", hidden)
        scores = heads(query) @ heads(key).transpose(0, 1, 3, 2) / math.sqrt(head_dim) + bias
        scores = scores - scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        context = (probabilities @ heads(value)).transpose(0, 2, 1, 3).reshape(*hidden.shape)
        attention = _layer_norm(
            project("attention_output", context) + hidden,
            weights[f"{layer}.attention_norm.weight"],
            weights[f"{layer}.attention_norm.bias"],
        )
        residual = _layer_norm(
            hidden + attention,
            weights[f"{layer}.norm_1.weight"],
            weights[f"{layer}.norm_1.bias"],
        )
        gated = residual @ weights[f"{layer}.up_gated.weight"]
        projected = (gated[..., :INTERMEDIATE] * _gelu(gated[..., INTERMEDIATE:])) @ weights[
            f"{layer}.down.weight"
        ] + weights[f"{layer}.down.bias"]
        hidden = _layer_norm(
            residual + projected,
            weights[f"{layer}.norm_2.weight"],
            weights[f"{layer}.norm_2.bias"],
        )
    return np.asarray(hidden, dtype=np.float32)


def _reference_pooled(
    hidden: NDArray[np.float32], attention_mask: NDArray[np.int64]
) -> NDArray[np.float32]:
    mask = attention_mask.astype(np.float32)[..., None]
    pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
    return np.asarray(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))


def _inputs() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    ids = np.asarray([[3, 5, 1, 7, 0, 0], [2, 9, 4, 6, 8, 10]], dtype=np.int64)
    mask = np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=np.int64)
    return ids, mask


def test_the_mlx_forward_pass_reproduces_the_graph(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    weights = extract_weights(directory / "onnx" / "model.onnx", CONFIG)
    ids, mask = _inputs()

    model = JinaBertMlx(CONFIG, {name: mx.array(value) for name, value in weights.items()})
    hidden = np.asarray(model(mx.array(ids), mx.array(mask)))

    expected = _reference_forward(weights, ids, mask)
    assert hidden.shape == expected.shape == (2, 6, HIDDEN)
    assert np.all(np.isfinite(hidden))
    assert np.allclose(hidden, expected, atol=2e-4)


def test_padding_does_not_change_the_rows_that_are_not_padding(tmp_path: Path) -> None:
    """Masked keys must contribute nothing, or a batch would embed differently
    depending on which other passages happened to share it."""
    directory = _snapshot(tmp_path / "snapshot")
    weights = extract_weights(directory / "onnx" / "model.onnx", CONFIG)
    model = JinaBertMlx(CONFIG, {name: mx.array(value) for name, value in weights.items()})
    ids, mask = _inputs()

    padded = mask_mean_normalize(model(mx.array(ids), mx.array(mask)), mx.array(mask))
    unpadded_ids = ids[:1, :4]
    unpadded_mask = mask[:1, :4]
    alone = mask_mean_normalize(
        model(mx.array(unpadded_ids), mx.array(unpadded_mask)), mx.array(unpadded_mask)
    )

    assert np.allclose(np.asarray(padded)[0], np.asarray(alone)[0], atol=2e-5)


def test_pooling_masks_padding_and_returns_unit_rows() -> None:
    _, mask = _inputs()
    hidden = np.random.default_rng(7).standard_normal((2, 6, HIDDEN)).astype(np.float32)

    pooled = np.asarray(mask_mean_normalize(mx.array(hidden), mx.array(mask)))

    assert pooled.shape == (2, HIDDEN)
    assert pooled.dtype == np.float32
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=1e-6)
    assert np.allclose(pooled, _reference_pooled(hidden, mask), atol=1e-6)


def test_pooling_a_row_with_no_unmasked_tokens_stays_finite() -> None:
    hidden = np.ones((1, 3, HIDDEN), dtype=np.float32)
    mask = np.zeros((1, 3), dtype=np.int64)

    pooled = np.asarray(mask_mean_normalize(mx.array(hidden), mx.array(mask)))

    assert np.all(np.isfinite(pooled))
