"""Metal passage embedding through MLX, for Apple Silicon.

MLX cannot execute ONNX directly, so this module reproduces the pinned
``codefuse-ai/F2LLM-v2-160M`` Qwen3 encoder in MLX over weights extracted from
the same official ``model.safetensors`` snapshot that ``direct_onnx`` converts.

Weights and activations execute in half precision (``float16``), while final
``<|im_end|>`` pooling and L2 normalization run in ``float32`` so vectors
handed to the index match CPU ``float32`` to ``cosine >= 0.999998`` with
identical retrieval rankings.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from .backends import MLX_PROVIDER
from .direct_onnx import (
    DEFAULT_MODEL_ARTIFACT,
    FloatArray,
    ModelConfig,
    load_tokenizer,
    read_model_config,
    resolve_model_snapshot,
)
from .embedding import DEFAULT_MODEL
from .safetensors_io import open_safetensors

if TYPE_CHECKING:  # pragma: no cover - imported for types only
    import mlx.core as mx

logger = logging.getLogger(__name__)

# Bumped whenever the set, shape, or meaning of the converted tensors changes,
# so an existing conversion (including legacy Jina weights) is rebuilt rather
# than read under new expectations.
WEIGHT_LAYOUT_VERSION = 3

WEIGHT_PRECISION = "f16"
FULL_PRECISION_TENSORS: frozenset[str] = frozenset()

# What a conversion in flight is called. It stays ``.safetensors`` because MLX
# appends that to a name that lacks it, and would write somewhere this never
# moves or cleans up.
_TEMPORARY_SUFFIX = ".tmp.safetensors"


def extract_weights(model_path: Path, config: ModelConfig) -> dict[str, FloatArray]:
    """Lift and pre-transpose every Qwen3 parameter out of *model_path*."""

    h = config.hidden_size
    qh = config.num_attention_heads
    kvh = config.num_key_value_heads
    hd = config.head_dim
    ff = config.intermediate_size
    vocab = config.vocab_size

    with open_safetensors(model_path) as archive:
        weights: dict[str, FloatArray] = {
            "embed_tokens.weight": archive.take_float32(
                "embed_tokens.weight", (vocab, h), "word embedding"
            ),
            "norm.weight": archive.take_float32("norm.weight", (h,), "final RMSNorm"),
        }
        for index in range(config.num_hidden_layers):
            prefix = f"layers.{index}"
            weights[f"{prefix}.input_layernorm.weight"] = archive.take_float32(
                f"{prefix}.input_layernorm.weight",
                (h,),
                f"layer {index} input RMSNorm",
            )
            weights[f"{prefix}.post_attention_layernorm.weight"] = archive.take_float32(
                f"{prefix}.post_attention_layernorm.weight",
                (h,),
                f"layer {index} post-attention RMSNorm",
            )
            weights[f"{prefix}.q_norm.weight"] = archive.take_float32(
                f"{prefix}.self_attn.q_norm.weight",
                (hd,),
                f"layer {index} q_norm",
            )
            weights[f"{prefix}.k_norm.weight"] = archive.take_float32(
                f"{prefix}.self_attn.k_norm.weight",
                (hd,),
                f"layer {index} k_norm",
            )
            weights[f"{prefix}.q_proj.weight"] = np.ascontiguousarray(
                archive.take_float32(
                    f"{prefix}.self_attn.q_proj.weight",
                    (qh * hd, h),
                    f"layer {index} q_proj",
                ).T
            )
            weights[f"{prefix}.k_proj.weight"] = np.ascontiguousarray(
                archive.take_float32(
                    f"{prefix}.self_attn.k_proj.weight",
                    (kvh * hd, h),
                    f"layer {index} k_proj",
                ).T
            )
            weights[f"{prefix}.v_proj.weight"] = np.ascontiguousarray(
                archive.take_float32(
                    f"{prefix}.self_attn.v_proj.weight",
                    (kvh * hd, h),
                    f"layer {index} v_proj",
                ).T
            )
            weights[f"{prefix}.o_proj.weight"] = np.ascontiguousarray(
                archive.take_float32(
                    f"{prefix}.self_attn.o_proj.weight",
                    (h, qh * hd),
                    f"layer {index} o_proj",
                ).T
            )
            gate_w = archive.take_float32(
                f"{prefix}.mlp.gate_proj.weight",
                (ff, h),
                f"layer {index} gate_proj",
            )
            up_w = archive.take_float32(
                f"{prefix}.mlp.up_proj.weight",
                (ff, h),
                f"layer {index} up_proj",
            )
            weights[f"{prefix}.gate_up.weight"] = np.ascontiguousarray(
                np.concatenate([gate_w.T, up_w.T], axis=-1)
            )
            weights[f"{prefix}.down_proj.weight"] = np.ascontiguousarray(
                archive.take_float32(
                    f"{prefix}.mlp.down_proj.weight",
                    (h, ff),
                    f"layer {index} down_proj",
                ).T
            )
    return weights


def converted_weights_path(cache_directory: Path, model_directory: Path) -> Path:
    """Where the MLX conversion of *model_directory* is kept."""

    revision = model_directory.name or "unknown"
    return (
        cache_directory
        / "mlx"
        / f"{revision}-f2llm-v{WEIGHT_LAYOUT_VERSION}-{WEIGHT_PRECISION}.safetensors"
    )


def ensure_converted_weights(
    model_directory: Path,
    cache_directory: Path,
    config: ModelConfig,
    *,
    convert: Callable[[Path, ModelConfig], dict[str, FloatArray]] = extract_weights,
) -> Path:
    """Return the converted MLX weights for *model_directory*, building them once."""

    import mlx.core as mx

    target = converted_weights_path(cache_directory, model_directory)
    if target.is_file():
        return target
    logger.info("Converting the safetensors weights of %s for MLX once", model_directory.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    extracted = convert(model_directory / DEFAULT_MODEL_ARTIFACT, config)
    weights = {}
    for name in list(extracted):
        tensor = extracted.pop(name)
        if name not in FULL_PRECISION_TENSORS:
            tensor = tensor.astype(np.float16)
        weights[name] = mx.array(tensor)
    temporary = target.with_name(f"{target.stem}.{os.getpid()}{_TEMPORARY_SUFFIX}")
    try:
        mx.save_safetensors(str(temporary), weights)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _discard_superseded_conversions(target)
    return target


def _discard_superseded_conversions(target: Path) -> None:
    """Remove conversions of other revisions, models, or layouts left beside *target*."""

    for path in target.parent.glob(f"*{target.suffix}"):
        if path == target or path.name.endswith(_TEMPORARY_SUFFIX):
            continue
        try:
            path.unlink()
        except OSError:  # pragma: no cover - a cache this cannot prune still works
            logger.debug("Could not remove the superseded MLX conversion %s", path)
        else:
            logger.info("Removed the superseded MLX conversion %s", path.name)


def last_token_normalize(hidden: mx.array, attention_mask: mx.array) -> mx.array:
    """Gather the final non-padding token (``<|im_end|>``) and L2-normalize in float32."""

    import mlx.core as mx

    if hidden.ndim != 3 or hidden.shape[1] == 0:
        raise ValueError("Cannot pool an empty token sequence")
    lengths = mx.sum(attention_mask.astype(mx.int64), axis=1)
    eos_positions = mx.maximum(lengths - 1, 0)
    batch_indices = mx.arange(hidden.shape[0])
    pooled = hidden[batch_indices, eos_positions].astype(mx.float32)
    norms = mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return cast("mx.array", pooled / norms)


class Qwen3Mlx:
    """The Qwen3 causal encoder trunk of ``codefuse-ai/F2LLM-v2-160M``, executed by MLX."""

    def __init__(self, config: ModelConfig, weights: Mapping[str, mx.array]) -> None:
        self.config = config
        self.weights = weights

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        import mlx.core as mx

        del attention_mask  # Causal masking keeps <|im_end|> invariant to trailing right-padding.
        weights = self.weights
        config = self.config
        batch, sequence = input_ids.shape
        qh = config.num_attention_heads
        kvh = config.num_key_value_heads
        hd = config.head_dim
        ff = config.intermediate_size
        eps = config.rms_norm_eps
        scale = 1.0 / math.sqrt(hd)

        hidden = mx.take(weights["embed_tokens.weight"], input_ids.reshape(-1), axis=0).reshape(
            batch, sequence, config.hidden_size
        )

        # mx.fast.scaled_dot_product_attention rejects mask="causal" when S == 1;
        # for a single token there are no future keys to mask anyway.
        attn_mask: str | None = "causal" if sequence > 1 else None

        for index in range(config.num_hidden_layers):
            layer = f"layers.{index}"
            normed = mx.fast.rms_norm(hidden, weights[f"{layer}.input_layernorm.weight"], eps)

            q = (
                (normed @ weights[f"{layer}.q_proj.weight"])
                .reshape(batch, sequence, qh, hd)
                .transpose(0, 2, 1, 3)
            )
            k = (
                (normed @ weights[f"{layer}.k_proj.weight"])
                .reshape(batch, sequence, kvh, hd)
                .transpose(0, 2, 1, 3)
            )
            v = (
                (normed @ weights[f"{layer}.v_proj.weight"])
                .reshape(batch, sequence, kvh, hd)
                .transpose(0, 2, 1, 3)
            )

            q = mx.fast.rms_norm(q, weights[f"{layer}.q_norm.weight"], eps)
            k = mx.fast.rms_norm(k, weights[f"{layer}.k_norm.weight"], eps)
            q = mx.fast.rope(
                q, dims=hd, traditional=False, base=config.rope_theta, scale=1.0, offset=0
            )
            k = mx.fast.rope(
                k, dims=hd, traditional=False, base=config.rope_theta, scale=1.0, offset=0
            )

            context = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attn_mask)
            context = context.transpose(0, 2, 1, 3).reshape(batch, sequence, qh * hd)
            hidden = hidden + (context @ weights[f"{layer}.o_proj.weight"])

            post = mx.fast.rms_norm(
                hidden, weights[f"{layer}.post_attention_layernorm.weight"], eps
            )
            gate_up = post @ weights[f"{layer}.gate_up.weight"]
            gate = gate_up[..., :ff]
            up = gate_up[..., ff:]
            activated = (gate * mx.sigmoid(gate)) * up
            hidden = hidden + (activated @ weights[f"{layer}.down_proj.weight"])

        return cast("mx.array", mx.fast.rms_norm(hidden, weights["norm.weight"], eps))


class MlxEmbedding:
    """The index-compatible F2LLM-v2-160M passage model executed on Metal by MLX."""

    def __init__(
        self,
        cache_directory: Path,
        *,
        offline: bool,
        model_id: str = DEFAULT_MODEL,
    ) -> None:
        import mlx.core as mx

        if model_id != DEFAULT_MODEL:
            raise ValueError(
                f"The MLX backend only supports the index model {DEFAULT_MODEL}; got {model_id}"
            )
        model_directory = resolve_model_snapshot(
            cache_directory, model_id=model_id, offline=offline
        )
        config = read_model_config(model_directory)
        weights_path = ensure_converted_weights(model_directory, cache_directory, config)
        self.tokenizer = load_tokenizer(model_directory)
        self.model = Qwen3Mlx(config, mx.load(str(weights_path)))
        mx.eval(self.model(mx.zeros((1, 2), dtype=mx.int64), mx.ones((1, 2), dtype=mx.int64)))
        self.resolved_providers: tuple[str, ...] = (MLX_PROVIDER,)

    def passage_embed(self, documents: str | Iterable[str]) -> Iterable[FloatArray]:
        import mlx.core as mx

        texts = [documents] if isinstance(documents, str) else list(documents)
        if not texts:
            return
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = mx.array(np.asarray([row.ids for row in encoded], dtype=np.int64))
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Tokenizer produced an empty token sequence")
        attention_mask = mx.array(
            np.asarray([row.attention_mask for row in encoded], dtype=np.int64)
        )
        pooled = last_token_normalize(self.model(input_ids, attention_mask), attention_mask)
        mx.eval(pooled)
        rows: NDArray[np.float32] = np.asarray(pooled, dtype=np.float32)
        mx.clear_cache()
        yield from rows
