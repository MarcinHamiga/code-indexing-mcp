"""Unit tests for the MLX Qwen3 passage backend (F2LLM-v2-160M).

Run on Apple Silicon without mutating the project environment:

    uv run --with "mlx==0.32.0" --with "onnx>=1.22,<2" pytest tests/test_mlx_backend.py
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

from code_indexing_mcp.direct_onnx import (  # noqa: E402
    DEFAULT_MODEL_ARTIFACT,
    ModelConfig,
    export_qwen3_onnx,
    last_token_pool_and_normalize,
    read_model_config,
)
from code_indexing_mcp.mlx_backend import (  # noqa: E402
    FULL_PRECISION_TENSORS,
    Qwen3Mlx,
    converted_weights_path,
    ensure_converted_weights,
    extract_weights,
    last_token_normalize,
)
from code_indexing_mcp.safetensors_io import write_safetensors_bf16  # noqa: E402

HIDDEN = 8
QH = 4
KVH = 2
HD = 4
LAYERS = 2
INTERMEDIATE = 12
VOCAB = 16
EPS = 1e-6
ROPE_THETA = 1_000_000.0

CONFIG = ModelConfig(
    hidden_size=HIDDEN,
    num_hidden_layers=LAYERS,
    num_attention_heads=QH,
    num_key_value_heads=KVH,
    head_dim=HD,
    intermediate_size=INTERMEDIATE,
    vocab_size=VOCAB,
    rms_norm_eps=EPS,
    rope_theta=ROPE_THETA,
)


def _config_json(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": QH,
        "num_key_value_heads": KVH,
        "head_dim": HD,
        "intermediate_size": INTERMEDIATE,
        "vocab_size": VOCAB,
        "rms_norm_eps": EPS,
        "rope_theta": ROPE_THETA,
    }
    document.update(overrides)
    return document


def _synthetic_tensors(
    *,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
    omit: set[str] | None = None,
) -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(2026)
    shape_overrides = shape_overrides or {}
    omit = omit or set()

    def rand(name: str, *shape: int) -> NDArray[np.float32]:
        actual = shape_overrides.get(name, shape)
        return (rng.standard_normal(actual) * 0.15).astype(np.float32)

    tensors: dict[str, NDArray[np.float32]] = {
        "embed_tokens.weight": rand("embed_tokens.weight", VOCAB, HIDDEN),
        "norm.weight": np.ones(shape_overrides.get("norm.weight", (HIDDEN,)), dtype=np.float32),
    }
    for i in range(LAYERS):
        p = f"layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones((HIDDEN,), dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones((HIDDEN,), dtype=np.float32)
        tensors[f"{p}.self_attn.q_norm.weight"] = np.ones((HD,), dtype=np.float32)
        tensors[f"{p}.self_attn.k_norm.weight"] = np.ones((HD,), dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = rand(
            f"{p}.self_attn.q_proj.weight", QH * HD, HIDDEN
        )
        tensors[f"{p}.self_attn.k_proj.weight"] = rand(
            f"{p}.self_attn.k_proj.weight", KVH * HD, HIDDEN
        )
        tensors[f"{p}.self_attn.v_proj.weight"] = rand(
            f"{p}.self_attn.v_proj.weight", KVH * HD, HIDDEN
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = rand(
            f"{p}.self_attn.o_proj.weight", HIDDEN, QH * HD
        )
        tensors[f"{p}.mlp.gate_proj.weight"] = rand(
            f"{p}.mlp.gate_proj.weight", INTERMEDIATE, HIDDEN
        )
        tensors[f"{p}.mlp.up_proj.weight"] = rand(f"{p}.mlp.up_proj.weight", INTERMEDIATE, HIDDEN)
        tensors[f"{p}.mlp.down_proj.weight"] = rand(
            f"{p}.mlp.down_proj.weight", HIDDEN, INTERMEDIATE
        )
    for key in omit:
        tensors.pop(key, None)
    return tensors


def _snapshot(directory: Path, **tensor_options: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(_config_json()), encoding="utf-8")
    write_safetensors_bf16(directory / DEFAULT_MODEL_ARTIFACT, _synthetic_tensors(**tensor_options))
    return directory


# -- configuration ---------------------------------------------------------


def test_the_config_of_the_supported_architecture_is_read(tmp_path: Path) -> None:
    config = read_model_config(_snapshot(tmp_path / "snapshot"))
    assert config == CONFIG
    assert config.num_key_value_groups == QH // KVH


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_type", "bert"),
        ("hidden_act", "gelu"),
        ("attention_bias", True),
        ("use_sliding_window", True),
    ],
)
def test_a_config_describing_another_architecture_is_refused(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    (directory / "config.json").write_text(
        json.dumps(_config_json(**{field: bad_value})), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=field):
        read_model_config(directory)


def test_incompatible_kv_head_grouping_is_refused(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    (directory / "config.json").write_text(
        json.dumps(_config_json(num_key_value_heads=3)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="num_key_value_heads"):
        read_model_config(directory)


# -- weight extraction -----------------------------------------------------


def test_every_qwen3_weight_comes_out_pre_transposed_and_fused(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    weights = extract_weights(directory / DEFAULT_MODEL_ARTIFACT, CONFIG)

    assert weights["embed_tokens.weight"].shape == (VOCAB, HIDDEN)
    assert weights["norm.weight"].shape == (HIDDEN,)
    for index in range(LAYERS):
        assert weights[f"layers.{index}.q_proj.weight"].shape == (HIDDEN, QH * HD)
        assert weights[f"layers.{index}.k_proj.weight"].shape == (HIDDEN, KVH * HD)
        assert weights[f"layers.{index}.v_proj.weight"].shape == (HIDDEN, KVH * HD)
        assert weights[f"layers.{index}.o_proj.weight"].shape == (QH * HD, HIDDEN)
        assert weights[f"layers.{index}.gate_up.weight"].shape == (HIDDEN, 2 * INTERMEDIATE)
        assert weights[f"layers.{index}.down_proj.weight"].shape == (INTERMEDIATE, HIDDEN)
        assert weights[f"layers.{index}.q_norm.weight"].shape == (HD,)
        assert weights[f"layers.{index}.k_norm.weight"].shape == (HD,)
    assert all(value.dtype == np.float32 for value in weights.values())


def test_missing_or_wrong_shaped_safetensor_is_refused(tmp_path: Path) -> None:
    missing_dir = _snapshot(tmp_path / "missing", omit={"layers.1.mlp.down_proj.weight"})
    with pytest.raises(ValueError, match="down_proj"):
        extract_weights(missing_dir / DEFAULT_MODEL_ARTIFACT, CONFIG)

    bad_shape_dir = _snapshot(
        tmp_path / "bad_shape",
        shape_overrides={"layers.0.mlp.gate_proj.weight": (INTERMEDIATE + 1, HIDDEN)},
    )
    with pytest.raises(ValueError, match="gate_proj"):
        extract_weights(bad_shape_dir / DEFAULT_MODEL_ARTIFACT, CONFIG)


# -- conversion cache ------------------------------------------------------


def test_conversion_runs_once_reuses_revision_and_prunes_legacy_jina_files(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "0a1b2c3")
    cache = tmp_path / "cache"
    target = converted_weights_path(cache, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy_jina = target.with_name("0a1b2c3-jina-v2-f16.safetensors")
    legacy_jina.write_bytes(b"old jina weights")
    in_flight = target.with_name("other.999999.tmp.safetensors")
    in_flight.write_bytes(b"in flight")

    conversions = 0

    def counting(model_path: Path, config: ModelConfig) -> dict[str, NDArray[np.float32]]:
        nonlocal conversions
        conversions += 1
        return extract_weights(model_path, config)

    first = ensure_converted_weights(directory, cache, CONFIG, convert=counting)
    second = ensure_converted_weights(directory, cache, CONFIG, convert=counting)

    assert first == second == target
    assert conversions == 1
    assert "-f2llm-v3-f16.safetensors" in first.name
    assert not legacy_jina.exists()
    assert in_flight.is_file()


def test_a_converted_file_holds_every_extracted_tensor_in_float16(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    path = ensure_converted_weights(directory, tmp_path / "cache", CONFIG)
    loaded = mx.load(str(path))
    expected = extract_weights(directory / DEFAULT_MODEL_ARTIFACT, CONFIG)
    assert set(loaded) == set(expected)
    for name, value in expected.items():
        stored = loaded[name]
        assert name not in FULL_PRECISION_TENSORS
        assert stored.dtype == mx.float16
        assert np.allclose(np.asarray(stored), value, rtol=1e-3, atol=1e-6)


# -- independent NumPy Qwen3 reference & ONNX cross-parity -----------------


def _rms_norm_np(x: NDArray[Any], weight: NDArray[Any]) -> NDArray[Any]:
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + EPS)
    return (x / rms) * weight


def _rope_np(x: NDArray[Any]) -> NDArray[Any]:
    # x: [B, H, S, HD]
    seq = x.shape[2]
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))
    angles = np.arange(seq, dtype=np.float32)[:, None] * inv_freq[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, None, :, :]
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, None, :, :]
    x1 = x[..., : HD // 2]
    x2 = x[..., HD // 2 :]
    rotated = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


def _reference_forward(
    weights: dict[str, NDArray[np.float32]],
    input_ids: NDArray[np.int64],
) -> NDArray[np.float32]:
    batch, seq = input_ids.shape
    hidden = weights["embed_tokens.weight"][input_ids]
    pos = np.arange(seq)
    causal_mask = np.where(pos[None, :] > pos[:, None], -1e9, 0.0).astype(np.float32)[
        None, None, :, :
    ]
    group = QH // KVH
    for i in range(LAYERS):
        layer = f"layers.{i}"
        normed = _rms_norm_np(hidden, weights[f"{layer}.input_layernorm.weight"])
        q = (
            (normed @ weights[f"{layer}.q_proj.weight"])
            .reshape(batch, seq, QH, HD)
            .transpose(0, 2, 1, 3)
        )
        k = (
            (normed @ weights[f"{layer}.k_proj.weight"])
            .reshape(batch, seq, KVH, HD)
            .transpose(0, 2, 1, 3)
        )
        v = (
            (normed @ weights[f"{layer}.v_proj.weight"])
            .reshape(batch, seq, KVH, HD)
            .transpose(0, 2, 1, 3)
        )
        q = _rope_np(_rms_norm_np(q, weights[f"{layer}.q_norm.weight"]))
        k = _rope_np(_rms_norm_np(k, weights[f"{layer}.k_norm.weight"]))
        k_exp = np.repeat(k, group, axis=1)
        v_exp = np.repeat(v, group, axis=1)
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) / math.sqrt(HD) + causal_mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        ctx = (probs @ v_exp).transpose(0, 2, 1, 3).reshape(batch, seq, QH * HD)
        hidden = hidden + (ctx @ weights[f"{layer}.o_proj.weight"])

        post = _rms_norm_np(hidden, weights[f"{layer}.post_attention_layernorm.weight"])
        gate_up = post @ weights[f"{layer}.gate_up.weight"]
        gate = gate_up[..., :INTERMEDIATE]
        up = gate_up[..., INTERMEDIATE:]
        silu = gate * (1.0 / (1.0 + np.exp(-gate)))
        hidden = hidden + ((silu * up) @ weights[f"{layer}.down_proj.weight"])
    return np.asarray(_rms_norm_np(hidden, weights["norm.weight"]), dtype=np.float32)


def _inputs() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    ids = np.asarray([[3, 5, 1, 7, 0, 0], [2, 9, 4, 6, 8, 10]], dtype=np.int64)
    mask = np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=np.int64)
    return ids, mask


def test_mlx_forward_matches_numpy_reference_and_exported_onnx(tmp_path: Path) -> None:
    import onnxruntime

    directory = _snapshot(tmp_path / "snapshot")
    weights = extract_weights(directory / DEFAULT_MODEL_ARTIFACT, CONFIG)
    ids, mask = _inputs()

    # Float32 MLX execution vs NumPy reference
    model_f32 = Qwen3Mlx(CONFIG, {name: mx.array(value) for name, value in weights.items()})
    hidden_f32 = np.asarray(model_f32(mx.array(ids), mx.array(mask)))
    expected = _reference_forward(weights, ids)
    assert hidden_f32.shape == expected.shape == (2, 6, HIDDEN)
    assert np.allclose(hidden_f32, expected, atol=1e-5)

    # Exported ONNX Runtime execution vs MLX float16 converted model
    onnx_path = tmp_path / "exported.onnx"
    export_qwen3_onnx(directory / DEFAULT_MODEL_ARTIFACT, CONFIG, onnx_path)
    ort_session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_pooled = last_token_pool_and_normalize(ort_session.run(None, {"input_ids": ids})[0], mask)

    f16_path = ensure_converted_weights(directory, tmp_path / "cache", CONFIG)
    model_f16 = Qwen3Mlx(CONFIG, mx.load(str(f16_path)))
    hidden_f16 = model_f16(mx.array(ids), mx.array(mask))
    assert hidden_f16.dtype == mx.float16
    mlx_pooled = np.asarray(last_token_normalize(hidden_f16, mx.array(mask)))

    assert mlx_pooled.dtype == np.float32
    assert np.allclose(np.linalg.norm(mlx_pooled, axis=1), 1.0, atol=1e-6)
    assert np.all(np.sum(mlx_pooled * ort_pooled, axis=1) > 0.9999)


def test_right_padding_and_single_token_sequence_stay_invariant_and_finite(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path / "snapshot")
    f16_path = ensure_converted_weights(directory, tmp_path / "cache", CONFIG)
    model = Qwen3Mlx(CONFIG, mx.load(str(f16_path)))
    ids, mask = _inputs()

    padded = np.asarray(last_token_normalize(model(mx.array(ids), mx.array(mask)), mx.array(mask)))
    alone = np.asarray(
        last_token_normalize(
            model(mx.array(ids[:1, :4]), mx.array(mask[:1, :4])),
            mx.array(mask[:1, :4]),
        )
    )
    assert float(np.sum(padded[0] * alone[0])) > 0.99999

    # S == 1 edge case (mx.fast.scaled_dot_product_attention requires mask=None for S==1)
    single_id = mx.array([[5]], dtype=mx.int64)
    single_mask = mx.array([[1]], dtype=mx.int64)
    single_vec = np.asarray(last_token_normalize(model(single_id, single_mask), single_mask))
    assert single_vec.shape == (1, HIDDEN)
    assert np.all(np.isfinite(single_vec))

    # Empty mask safety
    zero_mask = mx.zeros((1, 3), dtype=mx.int64)
    hidden = mx.ones((1, 3, HIDDEN), dtype=mx.float32)
    assert np.all(np.isfinite(np.asarray(last_token_normalize(hidden, zero_mask))))
