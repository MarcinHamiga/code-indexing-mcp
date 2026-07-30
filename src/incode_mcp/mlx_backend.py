"""Metal passage embedding through MLX, for Apple Silicon.

WebGPU reached 1.11x of CPU on an Apple M4 Pro against a 1.25x promotion gate,
which is what activated the long-term plan's designated Metal fallback. MLX
cannot execute ONNX, so this reproduces the pinned model rather than running it:
the float32 initializers are lifted out of the same ``onnx/model.onnx`` artifact
FastEmbed loads, and the JinaBERT v2 graph they belong to is written out again in
MLX. That keeps the one thing an index depends on -- the vectors -- identical in
origin, while the execution changes.

Like ``direct_onnx``, this is deliberately model-specific. The configuration is
checked against the architecture reproduced here and the extraction refuses an
artifact whose tensors have moved, because a backend that guesses at a changed
graph does not fail: it returns plausible vectors that no longer retrieve.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .backends import MLX_PROVIDER
from .direct_onnx import (
    DEFAULT_MODEL_ARTIFACT,
    FloatArray,
    load_tokenizer,
    resolve_model_snapshot,
)
from .embedding import DEFAULT_MODEL

if TYPE_CHECKING:  # pragma: no cover - imported for types only
    import mlx.core as mx

logger = logging.getLogger(__name__)

# Bumped whenever the set or meaning of the converted tensors changes, so an
# existing conversion is rebuilt rather than read under new expectations.
WEIGHT_LAYOUT_VERSION = 1

# What the export adds to a masked score is -3.4028235e38; adding a negative
# ALiBi bias to that overflows to -inf, which can leave a fused attention kernel
# subtracting -inf from -inf. This underflows to exactly zero through exp in
# float32 and cannot overflow, so it is the same softmax without the NaN risk.
MASK_FILL = -1e9

# The architectural facts this implementation hard-codes. A different value is a
# different model, and reproducing it would need different code.
SUPPORTED_ARCHITECTURE = {
    "model_type": "bert",
    "position_embedding_type": "alibi",
    "feed_forward_type": "geglu",
    "hidden_act": "gelu",
    "emb_pooler": "mean",
}

# The node whose first input carries the per-head ALiBi slopes.
_ALIBI_SLOPE_NODE = "/encoder/Mul_1"


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def read_model_config(model_directory: Path) -> ModelConfig:
    """Read and vet the configuration of the snapshot in *model_directory*."""

    path = model_directory / "config.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read model configuration at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Model configuration at {path} is not a JSON object")
    for field, expected in SUPPORTED_ARCHITECTURE.items():
        actual = document.get(field)
        if actual != expected:
            raise ValueError(
                f"The MLX backend reproduces {field}={expected!r}; this model declares {actual!r}"
            )

    def positive(field: str) -> int:
        value = document.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Model configuration has no positive {field}")
        return value

    hidden_size = positive("hidden_size")
    heads = positive("num_attention_heads")
    if hidden_size % heads:
        raise ValueError(
            f"hidden_size {hidden_size} does not divide evenly across {heads} attention heads"
        )
    epsilon = document.get("layer_norm_eps")
    if not isinstance(epsilon, int | float) or epsilon <= 0:
        raise ValueError("Model configuration has no positive layer_norm_eps")
    return ModelConfig(
        hidden_size=hidden_size,
        num_hidden_layers=positive("num_hidden_layers"),
        num_attention_heads=heads,
        intermediate_size=positive("intermediate_size"),
        layer_norm_eps=float(epsilon),
    )


def _layer_matmuls(index: int, config: ModelConfig) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The linear projections of one layer, by converted name.

    The export folds each projection's weight into an anonymous
    ``onnx::MatMul_*`` initializer, transposed into ``[in, out]``, so a weight is
    found through the node that consumes it rather than by parameter name.
    """
    hidden = config.hidden_size
    prefix = f"/encoder/layer.{index}/attention"
    return {
        "query": (f"{prefix}/self/query/MatMul", (hidden, hidden)),
        "key": (f"{prefix}/self/key/MatMul", (hidden, hidden)),
        "value": (f"{prefix}/self/value/MatMul", (hidden, hidden)),
        "attention_output": (f"{prefix}/output/dense/MatMul", (hidden, hidden)),
        "up_gated": (
            f"/encoder/layer.{index}/mlp/up_gated_layer/MatMul",
            (hidden, 2 * config.intermediate_size),
        ),
        "down": (
            f"/encoder/layer.{index}/mlp/down_layer/MatMul",
            (config.intermediate_size, hidden),
        ),
    }


def _layer_tensors(index: int, config: ModelConfig) -> dict[str, tuple[str, tuple[int, ...]]]:
    """The biases and normalization parameters of one layer, by converted name."""
    hidden = (config.hidden_size,)
    parameter = f"encoder.layer.{index}"
    named: dict[str, tuple[str, tuple[int, ...]]] = {
        "query.bias": (f"{parameter}.attention.self.query.bias", hidden),
        "key.bias": (f"{parameter}.attention.self.key.bias", hidden),
        "value.bias": (f"{parameter}.attention.self.value.bias", hidden),
        "attention_output.bias": (f"{parameter}.attention.output.dense.bias", hidden),
        "down.bias": (f"{parameter}.mlp.down_layer.bias", hidden),
    }
    for converted, source in (
        ("norm_q", f"{parameter}.attention.self.layer_norm_q"),
        ("norm_k", f"{parameter}.attention.self.layer_norm_k"),
        ("attention_norm", f"{parameter}.attention.output.LayerNorm"),
        ("norm_1", f"{parameter}.layer_norm_1"),
        ("norm_2", f"{parameter}.layer_norm_2"),
    ):
        named[f"{converted}.weight"] = (f"{source}.weight", hidden)
        named[f"{converted}.bias"] = (f"{source}.bias", hidden)
    return named


def extract_weights(model_path: Path, config: ModelConfig) -> dict[str, FloatArray]:
    """Lift every parameter this implementation needs out of the ONNX artifact.

    Each initializer's payload is released as it is converted, so the peak stays
    near one copy of the weights rather than holding the parsed protobuf and the
    converted arrays at full size at the same time.
    """
    import onnx
    from onnx import numpy_helper

    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Could not read the ONNX artifact at {model_path}: {exc}") from exc
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    by_name = {node.name: node for node in graph.node}
    by_output = {output: node for node in graph.node for output in node.output}

    def take(name: str, shape: tuple[int, ...], described_as: str) -> FloatArray:
        tensor = initializers.get(name)
        if tensor is None:
            raise ValueError(f"The ONNX artifact has no {described_as} tensor named {name!r}")
        array = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
        # Frees the protobuf's own copy; nothing reads this tensor twice.
        tensor.ClearField("raw_data")
        if array.shape != shape:
            raise ValueError(
                f"The {described_as} tensor {name!r} is {array.shape}, expected {shape}"
            )
        return array

    def take_weight(node_name: str, shape: tuple[int, ...], described_as: str) -> FloatArray:
        node = by_name.get(node_name)
        if node is None or len(node.input) < 2:
            raise ValueError(
                f"The ONNX artifact has no {described_as} projection at node {node_name!r}"
            )
        return take(node.input[1], shape, described_as)

    weights: dict[str, FloatArray] = {
        "embeddings.word_embeddings": take(
            "embeddings.word_embeddings.weight",
            (_vocabulary_size(initializers), config.hidden_size),
            "word embedding",
        ),
        "embeddings.token_type": take(
            "embeddings.token_type_embeddings.weight",
            (_token_type_count(initializers), config.hidden_size),
            "token type embedding",
        ),
        "embeddings.norm.weight": take(
            "embeddings.LayerNorm.weight", (config.hidden_size,), "embedding normalization"
        ),
        "embeddings.norm.bias": take(
            "embeddings.LayerNorm.bias", (config.hidden_size,), "embedding normalization"
        ),
        "alibi_slopes": _alibi_slopes(by_output, numpy_helper, config),
    }
    for index in range(config.num_hidden_layers):
        for converted, (node_name, shape) in _layer_matmuls(index, config).items():
            weights[f"layers.{index}.{converted}.weight"] = take_weight(
                node_name, shape, f"layer {index} {converted}"
            )
        for converted, (name, shape) in _layer_tensors(index, config).items():
            weights[f"layers.{index}.{converted}"] = take(name, shape, f"layer {index} {converted}")
    return weights


def _vocabulary_size(initializers: Mapping[str, Any]) -> int:
    tensor = initializers.get("embeddings.word_embeddings.weight")
    if tensor is None or len(tensor.dims) != 2:
        raise ValueError("The ONNX artifact has no two-dimensional word embedding table")
    return int(tensor.dims[0])


def _token_type_count(initializers: Mapping[str, Any]) -> int:
    tensor = initializers.get("embeddings.token_type_embeddings.weight")
    if tensor is None or len(tensor.dims) != 2:
        raise ValueError("The ONNX artifact has no two-dimensional token type embedding table")
    return int(tensor.dims[0])


def _alibi_slopes(
    by_output: Mapping[str, Any], numpy_helper: Any, config: ModelConfig
) -> FloatArray:
    """Read the per-head ALiBi slopes the export baked into the graph.

    Recomputing them from the head count would silently replace the slopes of a
    model that did not use the textbook powers of two.
    """
    consumer = next(
        (node for node in by_output.values() if node.name == _ALIBI_SLOPE_NODE),
        None,
    )
    if consumer is None or not consumer.input:
        raise ValueError(f"The ONNX artifact has no ALiBi bias node named {_ALIBI_SLOPE_NODE!r}")
    producer = by_output.get(consumer.input[0])
    if producer is None or producer.op_type != "Constant" or not producer.attribute:
        raise ValueError("The ALiBi slopes are not a constant in this ONNX artifact")
    slopes = np.asarray(numpy_helper.to_array(producer.attribute[0].t), dtype=np.float32)
    expected = (config.num_attention_heads, 1, 1)
    if slopes.shape != expected:
        raise ValueError(f"The ALiBi slopes are {slopes.shape}, expected {expected}")
    if not np.all(slopes < 0):
        raise ValueError("The ALiBi slopes are not all negative, so they do not penalise distance")
    return slopes


def converted_weights_path(cache_directory: Path, model_directory: Path) -> Path:
    """Where the conversion of *model_directory* is kept.

    The snapshot's revision is in the name, so a model that moves is converted
    again instead of being read out of a file describing the old one.
    """
    revision = model_directory.name or "unknown"
    return cache_directory / "mlx" / f"{revision}-jina-v{WEIGHT_LAYOUT_VERSION}-f32.safetensors"


def ensure_converted_weights(
    model_directory: Path,
    cache_directory: Path,
    config: ModelConfig,
    *,
    convert: Callable[[Path, ModelConfig], dict[str, FloatArray]] = extract_weights,
) -> Path:
    """Return the converted weights for *model_directory*, building them once.

    Building them is local file work over a snapshot already on disk, not a
    package installation, so it stays allowed in a worker -- but it reads and
    rewrites 600 MB, so it is logged rather than done quietly.
    """
    import mlx.core as mx

    target = converted_weights_path(cache_directory, model_directory)
    if target.is_file():
        return target
    logger.info("Converting the ONNX weights of %s for MLX once", model_directory.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        name: mx.array(value)
        for name, value in convert(model_directory / DEFAULT_MODEL_ARTIFACT, config).items()
    }
    # Written aside and moved into place: a conversion interrupted halfway must
    # not leave a truncated file that later loads look complete. The suffix stays
    # ``.safetensors`` because MLX appends it to a name that lacks it, and would
    # write somewhere this never moves or cleans up.
    temporary = target.with_name(f"{target.stem}.{os.getpid()}.tmp{target.suffix}")
    try:
        mx.save_safetensors(str(temporary), weights)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def mask_mean_normalize(hidden: mx.array, attention_mask: mx.array) -> mx.array:
    """Attention-mask mean pool and L2-normalize rows, as the CPU model does."""
    import mlx.core as mx

    mask = attention_mask.astype(mx.float32)[..., None]
    summed = mx.sum(hidden * mask, axis=1)
    counts = mx.maximum(mx.sum(mask, axis=1), 1e-9)
    pooled = summed / counts
    norms = mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    return cast("mx.array", pooled / norms)


class JinaBertMlx:
    """The exported JinaBERT v2 graph, executed by MLX.

    ALiBi in place of position embeddings, post-norm on the query and key
    projections, and a GEGLU feed-forward whose gate is the second half of one
    fused projection -- each of them read off the export rather than assumed.
    """

    def __init__(self, config: ModelConfig, weights: Mapping[str, mx.array]) -> None:
        self.config = config
        self.weights = weights

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        import mlx.core as mx

        weights = self.weights
        config = self.config
        batch, sequence = input_ids.shape
        heads = config.num_attention_heads
        head_dim = config.head_dim

        def normalize(x: mx.array, name: str) -> mx.array:
            return mx.fast.layer_norm(
                x, weights[f"{name}.weight"], weights[f"{name}.bias"], config.layer_norm_eps
            )

        # The artifact takes no token_type_ids input: it derives zeros, so every
        # token uses the first token-type row.
        hidden = (
            mx.take(weights["embeddings.word_embeddings"], input_ids.reshape(-1), axis=0).reshape(
                batch, sequence, config.hidden_size
            )
            + weights["embeddings.token_type"][0]
        )
        hidden = normalize(hidden, "embeddings.norm")

        positions = mx.arange(sequence)
        distance = mx.abs(positions[:, None] - positions[None, :]).astype(mx.float32)
        # Masked keys are pushed far below any score so they contribute nothing,
        # which is what keeps a passage's vector independent of the padding its
        # batch happened to need.
        keys_masked = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * MASK_FILL
        bias = weights["alibi_slopes"] * distance + keys_masked

        def project(name: str, source: mx.array) -> mx.array:
            return source @ weights[f"{name}.weight"] + weights[f"{name}.bias"]

        def split_heads(value: mx.array) -> mx.array:
            return value.reshape(batch, sequence, heads, head_dim).transpose(0, 2, 1, 3)

        for index in range(config.num_hidden_layers):
            layer = f"layers.{index}"
            query = normalize(project(f"{layer}.query", hidden), f"{layer}.norm_q")
            key = normalize(project(f"{layer}.key", hidden), f"{layer}.norm_k")
            value = project(f"{layer}.value", hidden)
            context = mx.fast.scaled_dot_product_attention(
                split_heads(query),
                split_heads(key),
                split_heads(value),
                scale=1.0 / math.sqrt(head_dim),
                mask=bias,
            )
            context = context.transpose(0, 2, 1, 3).reshape(batch, sequence, config.hidden_size)
            attention = normalize(
                project(f"{layer}.attention_output", context) + hidden,
                f"{layer}.attention_norm",
            )
            residual = normalize(hidden + attention, f"{layer}.norm_1")
            gated = residual @ weights[f"{layer}.up_gated.weight"]
            up = gated[..., : config.intermediate_size]
            gate = gated[..., config.intermediate_size :]
            projected = (
                up * _exact_gelu(gate) @ weights[f"{layer}.down.weight"]
                + weights[f"{layer}.down.bias"]
            )
            hidden = normalize(residual + projected, f"{layer}.norm_2")
        return hidden


def _exact_gelu(x: mx.array) -> mx.array:
    """GELU as the export computes it: ``x * 0.5 * (1 + erf(x / sqrt(2)))``."""
    import mlx.core as mx

    return cast("mx.array", x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0))))


class MlxEmbedding:
    """The index-compatible Jina passage model executed on Metal by MLX."""

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
        self.model = JinaBertMlx(config, mx.load(str(weights_path)))
        # Constructing the model does not touch the device, so one real forward
        # pass over the smallest possible input is what proves Metal is there to
        # be used -- and it makes the report below a statement rather than a
        # claim about an installed package.
        self.model(mx.zeros((1, 2), dtype=mx.int64), mx.ones((1, 2), dtype=mx.int64))
        mx.eval(self.model.weights)
        self.resolved_providers: tuple[str, ...] = (MLX_PROVIDER,)

    def passage_embed(self, documents: str | Iterable[str]) -> Iterable[FloatArray]:
        import mlx.core as mx

        texts = [documents] if isinstance(documents, str) else list(documents)
        if not texts:
            return
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = mx.array(np.asarray([row.ids for row in encoded], dtype=np.int64))
        attention_mask = mx.array(
            np.asarray([row.attention_mask for row in encoded], dtype=np.int64)
        )
        pooled = mask_mean_normalize(self.model(input_ids, attention_mask), attention_mask)
        mx.eval(pooled)
        rows: NDArray[np.float32] = np.asarray(pooled, dtype=np.float32)
        # Unified memory the batch's intermediates reserved is returned rather
        # than held against the worker's ceiling until the process exits.
        mx.clear_cache()
        yield from rows
