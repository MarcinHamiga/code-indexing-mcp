"""Direct ONNX query and passage embedding for ``codefuse-ai/F2LLM-v2-160M``.

Every ONNX Runtime backend (``cpu``, ``cuda``, ``webgpu``, ``vulkan``, and
``coreml``) runs through this module. The official Hugging Face snapshot ships
``model.safetensors`` (BF16) rather than an ONNX graph, so this module builds
and caches a self-contained opset-18 float32 ONNX graph locally once per
snapshot revision and layout version, using the same atomic write-and-replace
contract as ``mlx_backend.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .embedding import DEFAULT_MODEL, DEFAULT_QUERY_PROMPT
from .safetensors_io import FloatArray as FloatArray
from .safetensors_io import open_safetensors

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ARTIFACT = "model.safetensors"
_MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config_sentence_transformers.json",
    DEFAULT_MODEL_ARTIFACT,
]

# Bumped whenever the exported ONNX graph structure, initializers, or pooling
# contract changes, so existing cached graphs are rebuilt automatically.
ONNX_LAYOUT_VERSION = 1
ONNX_PRECISION = "f32"
_TEMPORARY_ONNX_SUFFIX = ".tmp.onnx"

# Dawn's native Vulkan backend is compiled into the Linux and Windows WebGPU
# plugin wheels; on macOS Dawn only ships Metal.
VULKAN_DAWN_PLATFORMS = frozenset({"linux", "win32"})

SUPPORTED_ARCHITECTURE: dict[str, object] = {
    "model_type": "qwen3",
    "hidden_act": "silu",
    "attention_bias": False,
    "use_sliding_window": False,
}

IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ModelConfig:
    """Validated Qwen3 architecture parameters read from ``config.json``."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class _Encoding(Protocol):
    ids: Sequence[int]
    attention_mask: Sequence[int]
    offsets: Sequence[tuple[int, int]]

    def __len__(self) -> int: ...


class _Tokenizer(Protocol):
    def encode_batch(self, documents: list[str]) -> list[_Encoding]: ...

    def encode(self, text: str) -> _Encoding: ...


class _SessionInput(Protocol):
    name: str


class _Session(Protocol):
    def get_inputs(self) -> Sequence[_SessionInput]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(
        self, output_names: object, input_feed: Mapping[str, NDArray[Any]]
    ) -> Sequence[NDArray[Any]]: ...


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read model configuration at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model configuration at {path} is not a JSON object")
    return cast(dict[str, Any], value)


def read_model_config(model_directory: Path) -> ModelConfig:
    """Read and validate Qwen3 architecture invariants from ``config.json``."""

    raw = _json_object(model_directory / "config.json")
    for key, expected in SUPPORTED_ARCHITECTURE.items():
        actual = raw.get(key)
        # Some Qwen3 configs omit use_sliding_window when sliding_window is null;
        # accept missing False flags only when False is the default.
        if actual is None and key == "use_sliding_window" and raw.get("sliding_window") is None:
            continue
        if actual != expected:
            raise ValueError(
                f"Unsupported {key} {actual!r} in {model_directory / 'config.json'}; "
                f"expected {expected!r}"
            )

    def positive(name: str) -> int:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Model configuration has no positive integer {name}")
        return value

    hidden_size = positive("hidden_size")
    layers = positive("num_hidden_layers")
    heads = positive("num_attention_heads")
    kv_heads = positive("num_key_value_heads")
    if heads % kv_heads != 0:
        raise ValueError(
            f"num_attention_heads ({heads}) is not divisible by num_key_value_heads ({kv_heads})"
        )
    raw_head_dim = raw.get("head_dim")
    if raw_head_dim is None:
        if hidden_size % heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) is not divisible by num_attention_heads ({heads})"
            )
        head_dim = hidden_size // heads
    else:
        head_dim = positive("head_dim")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for rotary position embeddings")

    eps = raw.get("rms_norm_eps")
    if isinstance(eps, bool) or not isinstance(eps, int | float) or eps <= 0:
        raise ValueError("Model configuration has no positive rms_norm_eps")
    rope_theta = raw.get("rope_theta", 1_000_000.0)
    if isinstance(rope_theta, bool) or not isinstance(rope_theta, int | float) or rope_theta <= 0:
        raise ValueError("Model configuration has no positive rope_theta")

    return ModelConfig(
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=positive("intermediate_size"),
        vocab_size=positive("vocab_size"),
        rms_norm_eps=float(eps),
        rope_theta=float(rope_theta),
    )


def read_query_prompt(model_directory: Path) -> str:
    """Read the query instruction prompt from ``config_sentence_transformers.json``."""

    path = model_directory / "config_sentence_transformers.json"
    if not path.is_file():
        return DEFAULT_QUERY_PROMPT
    raw = _json_object(path)
    prompts = raw.get("prompts")
    if not isinstance(prompts, dict):
        return DEFAULT_QUERY_PROMPT
    query_prompt = prompts.get("query")
    if not isinstance(query_prompt, str):
        return DEFAULT_QUERY_PROMPT
    return query_prompt


def resolve_model_snapshot(
    cache_directory: Path, *, model_id: str = DEFAULT_MODEL, offline: bool
) -> Path:
    """Resolve the pinned Hugging Face model snapshot into the shared cache."""

    if model_id != DEFAULT_MODEL:
        raise ValueError(
            f"The direct ONNX backend only supports the index model {DEFAULT_MODEL}; got {model_id}"
        )
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=model_id,
        allow_patterns=list(_MODEL_FILES),
        cache_dir=str(cache_directory),
        local_files_only=offline,
    )
    return Path(resolved)


def _extract_token_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def load_tokenizer(model_directory: Path) -> _Tokenizer:
    """Load tokenizer configuration with dynamic right-padding and EOS rules."""

    from tokenizers import AddedToken, Tokenizer

    config = _json_object(model_directory / "config.json")
    tokenizer_config = _json_object(model_directory / "tokenizer_config.json")
    special_tokens = _json_object(model_directory / "special_tokens_map.json")

    raw_model_max = tokenizer_config.get("model_max_length")
    raw_max = tokenizer_config.get("max_length")
    raw_pos_max = config.get("max_position_embeddings")
    lengths = [
        int(value)
        for value in (raw_model_max, raw_max, raw_pos_max)
        if not isinstance(value, bool) and isinstance(value, int | float) and value > 0
    ]
    if not lengths:
        raise ValueError("Tokenizer config has no positive model_max_length or max_length")
    max_context = min(lengths)

    tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
    for raw_token in special_tokens.values():
        if isinstance(raw_token, str):
            tokenizer.add_special_tokens([raw_token])
        elif isinstance(raw_token, list):
            strs = [item for item in raw_token if isinstance(item, str)]
            if strs:
                tokenizer.add_special_tokens(strs)
        elif isinstance(raw_token, dict) and isinstance(raw_token.get("content"), str):
            tokenizer.add_special_tokens([AddedToken(**raw_token)])

    tokenizer.enable_truncation(max_length=max_context)

    # Always clear any pre-baked fixed-length padding in tokenizer.json so
    # batches pad dynamically to their own longest member rather than max_length.
    tokenizer.no_padding()
    pad_token = _extract_token_string(tokenizer_config.get("pad_token")) or _extract_token_string(
        special_tokens.get("pad_token")
    )
    if not isinstance(pad_token, str):
        raise ValueError("Tokenizer config has no pad_token")

    raw_pad_id = config.get("pad_token_id")
    if not isinstance(raw_pad_id, bool) and isinstance(raw_pad_id, int) and raw_pad_id >= 0:
        pad_id = raw_pad_id
    else:
        resolved_pad_id = tokenizer.token_to_id(pad_token)
        if resolved_pad_id is None:
            raise ValueError(f"Tokenizer vocabulary does not contain pad_token {pad_token!r}")
        pad_id = int(resolved_pad_id)
    tokenizer.enable_padding(direction="right", pad_id=pad_id, pad_token=pad_token)
    return cast(_Tokenizer, tokenizer)


def last_token_pool_and_normalize(
    model_output: NDArray[Any], attention_mask: IntArray
) -> FloatArray:
    """Gather the final non-padding token (``<|im_end|>``) and L2-normalize in float32."""

    output = np.asarray(model_output, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.int64)
    if output.ndim != 3 or output.shape[1] == 0:
        raise ValueError("Cannot pool an empty token sequence")
    lengths = np.sum(mask, axis=1)
    eos_positions = np.maximum(lengths - 1, 0)
    batch_indices = np.arange(output.shape[0], dtype=np.int64)
    pooled = output[batch_indices, eos_positions]
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.maximum(norms, np.float32(1e-12))
    return np.asarray(normalized, dtype=np.float32)


def mean_pool_and_normalize(model_output: NDArray[Any], attention_mask: IntArray) -> FloatArray:
    """Attention-mask mean pool and L2-normalize rows in float32."""

    output = np.asarray(model_output, dtype=np.float32)
    mask = np.asarray(attention_mask, dtype=np.float32)[..., np.newaxis]
    summed = np.sum(output * mask, axis=1, dtype=np.float32)
    counts = np.maximum(np.sum(mask, axis=1, dtype=np.float32), np.float32(1e-9))
    pooled = np.asarray(summed / counts, dtype=np.float32)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.maximum(norms, np.float32(1e-12))
    return np.asarray(normalized, dtype=np.float32)


def converted_onnx_path(cache_directory: Path, model_directory: Path) -> Path:
    """Where the local ONNX conversion of *model_directory* is cached."""

    revision = model_directory.name or "unknown"
    return (
        cache_directory / "onnx" / f"{revision}-f2llm-v{ONNX_LAYOUT_VERSION}-{ONNX_PRECISION}.onnx"
    )


def _discard_superseded_onnx_conversions(target: Path) -> None:
    """Remove older ONNX revisions or layout versions beside *target*."""

    for path in target.parent.glob(f"*{target.suffix}"):
        if path == target or path.name.endswith(_TEMPORARY_ONNX_SUFFIX):
            continue
        try:
            path.unlink()
        except OSError:  # pragma: no cover - best-effort cache pruning
            logger.debug("Could not remove superseded ONNX conversion %s", path)
        else:
            logger.info("Removed superseded ONNX conversion %s", path.name)


def export_qwen3_onnx(safetensors_path: Path, config: ModelConfig, output_path: Path) -> None:
    """Build a self-contained opset-18 float32 Qwen3 ONNX graph from *safetensors_path*."""

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    h = config.hidden_size
    layers = config.num_hidden_layers
    qh = config.num_attention_heads
    kvh = config.num_key_value_heads
    hd = config.head_dim
    ff = config.intermediate_size
    vocab = config.vocab_size
    group = config.num_key_value_groups

    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def add_init(name: str, array: NDArray[Any]) -> str:
        initializers.append(numpy_helper.from_array(array, name=name))
        return name

    with open_safetensors(safetensors_path) as archive:
        add_init(
            "embed_tokens.weight",
            archive.take_float32("embed_tokens.weight", (vocab, h), "word embedding"),
        )
        add_init("norm.weight", archive.take_float32("norm.weight", (h,), "final RMSNorm"))
        add_init("eps", np.array(config.rms_norm_eps, dtype=np.float32))
        add_init("attn_scale", np.array(1.0 / math.sqrt(hd), dtype=np.float32))
        add_init("neg_inf", np.array(-1e9, dtype=np.float32))
        add_init("zero_f", np.array(0.0, dtype=np.float32))
        add_init("zero_i", np.array(0, dtype=np.int64))
        add_init("one_i", np.array(1, dtype=np.int64))
        add_init("ax_m1", np.array([-1], dtype=np.int64))
        add_init("ax_0", np.array([0], dtype=np.int64))
        add_init("ax_1", np.array([1], dtype=np.int64))
        add_init("ax_2", np.array([2], dtype=np.int64))
        add_init("ax_3", np.array([3], dtype=np.int64))
        add_init("ax_01", np.array([0, 1], dtype=np.int64))
        add_init("ax_rope_pos", np.array([0, 1, 3], dtype=np.int64))
        add_init("s_0", np.array([0], dtype=np.int64))
        add_init("s_half", np.array([hd // 2], dtype=np.int64))
        add_init("s_full", np.array([hd], dtype=np.int64))
        add_init("s_ff", np.array([ff], dtype=np.int64))
        add_init("s_2ff", np.array([2 * ff], dtype=np.int64))

        inv_freq = (
            1.0 / (config.rope_theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        ).reshape(1, 1, 1, hd // 2)
        add_init("inv_freq", inv_freq)

        # Dynamic [batch, seq] extraction
        nodes.append(helper.make_node("Shape", ["input_ids"], ["ids_shape"]))
        nodes.append(helper.make_node("Gather", ["ids_shape", "zero_i"], ["B_sc"], axis=0))
        nodes.append(helper.make_node("Gather", ["ids_shape", "one_i"], ["S_sc"], axis=0))
        nodes.append(helper.make_node("Unsqueeze", ["B_sc", "ax_0"], ["B_1d"]))
        nodes.append(helper.make_node("Unsqueeze", ["S_sc", "ax_0"], ["S_1d"]))

        # Dynamic RoPE cos/sin for arbitrary sequence length up to 40,960
        nodes.append(helper.make_node("Range", ["zero_i", "S_sc", "one_i"], ["pos_1d"]))
        nodes.append(helper.make_node("Cast", ["pos_1d"], ["pos_f"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Unsqueeze", ["pos_f", "ax_rope_pos"], ["pos_4d"]))
        nodes.append(helper.make_node("Mul", ["pos_4d", "inv_freq"], ["angles_half"]))
        nodes.append(
            helper.make_node("Concat", ["angles_half", "angles_half"], ["angles_full"], axis=3)
        )
        nodes.append(helper.make_node("Cos", ["angles_full"], ["rope_cos"]))
        nodes.append(helper.make_node("Sin", ["angles_full"], ["rope_sin"]))

        # Causal attention mask [1, 1, S, S]: trailing right-pad tokens sit strictly
        # to the right of <|im_end|>, so causal masking keeps <|im_end|> invariant
        # to batch padding without needing a 4D per-row padding mask.
        nodes.append(helper.make_node("Unsqueeze", ["pos_1d", "ax_1"], ["pos_row"]))
        nodes.append(helper.make_node("Unsqueeze", ["pos_1d", "ax_0"], ["pos_col"]))
        nodes.append(helper.make_node("Greater", ["pos_col", "pos_row"], ["causal_bool"]))
        nodes.append(helper.make_node("Where", ["causal_bool", "neg_inf", "zero_f"], ["causal_2d"]))
        nodes.append(helper.make_node("Unsqueeze", ["causal_2d", "ax_01"], ["causal_mask"]))

        c_qh = add_init("q_h_hd", np.array([qh, hd], dtype=np.int64))
        c_kvh = add_init("kv_h_hd", np.array([kvh, hd], dtype=np.int64))
        nodes.append(helper.make_node("Concat", ["B_1d", "S_1d", c_qh], ["q_shape4d"], axis=0))
        nodes.append(helper.make_node("Concat", ["B_1d", "S_1d", c_kvh], ["kv_shape4d"], axis=0))
        c_qh_1d = add_init("c_qh", np.array([qh], dtype=np.int64))
        c_hd_1d = add_init("c_hd", np.array([hd], dtype=np.int64))
        nodes.append(
            helper.make_node("Concat", ["B_1d", c_qh_1d, "S_1d", c_hd_1d], ["gqa_4d"], axis=0)
        )
        c_qdim = add_init("c_qdim", np.array([qh * hd], dtype=np.int64))
        nodes.append(helper.make_node("Concat", ["B_1d", "S_1d", c_qdim], ["ctx_3d"], axis=0))

        def rms_norm(x_name: str, weight_name: str, prefix: str) -> str:
            nodes.append(helper.make_node("Mul", [x_name, x_name], [f"{prefix}_sq"]))
            nodes.append(
                helper.make_node(
                    "ReduceMean", [f"{prefix}_sq", "ax_m1"], [f"{prefix}_mean"], keepdims=1
                )
            )
            nodes.append(helper.make_node("Add", [f"{prefix}_mean", "eps"], [f"{prefix}_veps"]))
            nodes.append(helper.make_node("Sqrt", [f"{prefix}_veps"], [f"{prefix}_rms"]))
            nodes.append(helper.make_node("Div", [x_name, f"{prefix}_rms"], [f"{prefix}_normed"]))
            nodes.append(
                helper.make_node("Mul", [f"{prefix}_normed", weight_name], [f"{prefix}_out"])
            )
            return f"{prefix}_out"

        def apply_rope(tensor_name: str, prefix: str) -> str:
            nodes.append(
                helper.make_node("Slice", [tensor_name, "s_0", "s_half", "ax_3"], [f"{prefix}_x1"])
            )
            nodes.append(
                helper.make_node(
                    "Slice", [tensor_name, "s_half", "s_full", "ax_3"], [f"{prefix}_x2"]
                )
            )
            nodes.append(helper.make_node("Neg", [f"{prefix}_x2"], [f"{prefix}_nx2"]))
            nodes.append(
                helper.make_node(
                    "Concat", [f"{prefix}_nx2", f"{prefix}_x1"], [f"{prefix}_rot"], axis=3
                )
            )
            nodes.append(helper.make_node("Mul", [tensor_name, "rope_cos"], [f"{prefix}_tc"]))
            nodes.append(helper.make_node("Mul", [f"{prefix}_rot", "rope_sin"], [f"{prefix}_ts"]))
            nodes.append(
                helper.make_node("Add", [f"{prefix}_tc", f"{prefix}_ts"], [f"{prefix}_out"])
            )
            return f"{prefix}_out"

        nodes.append(
            helper.make_node("Gather", ["embed_tokens.weight", "input_ids"], ["x_0"], axis=0)
        )
        cur_x = "x_0"
        for index in range(layers):
            layer = f"layers.{index}"
            rn1_w = add_init(
                f"l{index}_rn1_w",
                archive.take_float32(
                    f"{layer}.input_layernorm.weight", (h,), f"layer {index} norm 1"
                ),
            )
            rn2_w = add_init(
                f"l{index}_rn2_w",
                archive.take_float32(
                    f"{layer}.post_attention_layernorm.weight", (h,), f"layer {index} norm 2"
                ),
            )
            qn_w = add_init(
                f"l{index}_qn_w",
                archive.take_float32(
                    f"{layer}.self_attn.q_norm.weight", (hd,), f"layer {index} q_norm"
                ),
            )
            kn_w = add_init(
                f"l{index}_kn_w",
                archive.take_float32(
                    f"{layer}.self_attn.k_norm.weight", (hd,), f"layer {index} k_norm"
                ),
            )
            qw = add_init(
                f"l{index}_q_w",
                np.ascontiguousarray(
                    archive.take_float32(
                        f"{layer}.self_attn.q_proj.weight", (qh * hd, h), f"layer {index} q_proj"
                    ).T
                ),
            )
            kw = add_init(
                f"l{index}_k_w",
                np.ascontiguousarray(
                    archive.take_float32(
                        f"{layer}.self_attn.k_proj.weight", (kvh * hd, h), f"layer {index} k_proj"
                    ).T
                ),
            )
            vw = add_init(
                f"l{index}_v_w",
                np.ascontiguousarray(
                    archive.take_float32(
                        f"{layer}.self_attn.v_proj.weight", (kvh * hd, h), f"layer {index} v_proj"
                    ).T
                ),
            )
            ow = add_init(
                f"l{index}_o_w",
                np.ascontiguousarray(
                    archive.take_float32(
                        f"{layer}.self_attn.o_proj.weight", (h, qh * hd), f"layer {index} o_proj"
                    ).T
                ),
            )
            gate_t = archive.take_float32(
                f"{layer}.mlp.gate_proj.weight", (ff, h), f"layer {index} gate_proj"
            ).T
            up_t = archive.take_float32(
                f"{layer}.mlp.up_proj.weight", (ff, h), f"layer {index} up_proj"
            ).T
            gu_w = add_init(
                f"l{index}_gu_w",
                np.ascontiguousarray(np.concatenate([gate_t, up_t], axis=-1)),
            )
            dw = add_init(
                f"l{index}_d_w",
                np.ascontiguousarray(
                    archive.take_float32(
                        f"{layer}.mlp.down_proj.weight", (h, ff), f"layer {index} down_proj"
                    ).T
                ),
            )

            normed = rms_norm(cur_x, rn1_w, f"l{index}_rn1")
            nodes.append(helper.make_node("MatMul", [normed, qw], [f"l{index}_q_lin"]))
            nodes.append(helper.make_node("MatMul", [normed, kw], [f"l{index}_k_lin"]))
            nodes.append(helper.make_node("MatMul", [normed, vw], [f"l{index}_v_lin"]))

            nodes.append(
                helper.make_node("Reshape", [f"l{index}_q_lin", "q_shape4d"], [f"l{index}_q_4d"])
            )
            nodes.append(
                helper.make_node("Reshape", [f"l{index}_k_lin", "kv_shape4d"], [f"l{index}_k_4d"])
            )
            nodes.append(
                helper.make_node("Reshape", [f"l{index}_v_lin", "kv_shape4d"], [f"l{index}_v_4d"])
            )

            qn = rms_norm(f"l{index}_q_4d", qn_w, f"l{index}_qn")
            kn = rms_norm(f"l{index}_k_4d", kn_w, f"l{index}_kn")

            nodes.append(helper.make_node("Transpose", [qn], [f"l{index}_qt"], perm=[0, 2, 1, 3]))
            nodes.append(helper.make_node("Transpose", [kn], [f"l{index}_kt"], perm=[0, 2, 1, 3]))
            nodes.append(
                helper.make_node(
                    "Transpose", [f"l{index}_v_4d"], [f"l{index}_vt"], perm=[0, 2, 1, 3]
                )
            )

            qr = apply_rope(f"l{index}_qt", f"l{index}_qrope")
            kr = apply_rope(f"l{index}_kt", f"l{index}_krope")

            if group == 1:
                kgqa = kr
                vgqa = f"l{index}_vt"
            else:
                nodes.append(helper.make_node("Unsqueeze", [kr, "ax_2"], [f"l{index}_ku"]))
                nodes.append(
                    helper.make_node("Concat", [f"l{index}_ku"] * group, [f"l{index}_kc"], axis=2)
                )
                nodes.append(
                    helper.make_node("Reshape", [f"l{index}_kc", "gqa_4d"], [f"l{index}_kgqa"])
                )
                nodes.append(
                    helper.make_node("Unsqueeze", [f"l{index}_vt", "ax_2"], [f"l{index}_vu"])
                )
                nodes.append(
                    helper.make_node("Concat", [f"l{index}_vu"] * group, [f"l{index}_vc"], axis=2)
                )
                nodes.append(
                    helper.make_node("Reshape", [f"l{index}_vc", "gqa_4d"], [f"l{index}_vgqa"])
                )
                kgqa = f"l{index}_kgqa"
                vgqa = f"l{index}_vgqa"

            nodes.append(helper.make_node("Transpose", [kgqa], [f"l{index}_kT"], perm=[0, 1, 3, 2]))
            nodes.append(helper.make_node("MatMul", [qr, f"l{index}_kT"], [f"l{index}_qk"]))
            nodes.append(helper.make_node("Mul", [f"l{index}_qk", "attn_scale"], [f"l{index}_qks"]))
            nodes.append(
                helper.make_node("Add", [f"l{index}_qks", "causal_mask"], [f"l{index}_qkm"])
            )
            nodes.append(
                helper.make_node("Softmax", [f"l{index}_qkm"], [f"l{index}_probs"], axis=-1)
            )
            nodes.append(helper.make_node("MatMul", [f"l{index}_probs", vgqa], [f"l{index}_ctx4d"]))
            nodes.append(
                helper.make_node(
                    "Transpose", [f"l{index}_ctx4d"], [f"l{index}_ctx_t"], perm=[0, 2, 1, 3]
                )
            )
            nodes.append(
                helper.make_node("Reshape", [f"l{index}_ctx_t", "ctx_3d"], [f"l{index}_ctx"])
            )
            nodes.append(helper.make_node("MatMul", [f"l{index}_ctx", ow], [f"l{index}_attn_out"]))
            nodes.append(helper.make_node("Add", [cur_x, f"l{index}_attn_out"], [f"l{index}_res1"]))

            mlp_in = rms_norm(f"l{index}_res1", rn2_w, f"l{index}_rn2")
            nodes.append(helper.make_node("MatMul", [mlp_in, gu_w], [f"l{index}_gu"]))
            nodes.append(
                helper.make_node(
                    "Slice", [f"l{index}_gu", "s_0", "s_ff", "ax_2"], [f"l{index}_gate"]
                )
            )
            nodes.append(
                helper.make_node(
                    "Slice", [f"l{index}_gu", "s_ff", "s_2ff", "ax_2"], [f"l{index}_up"]
                )
            )
            nodes.append(helper.make_node("Sigmoid", [f"l{index}_gate"], [f"l{index}_sig"]))
            nodes.append(
                helper.make_node("Mul", [f"l{index}_gate", f"l{index}_sig"], [f"l{index}_silu"])
            )
            nodes.append(
                helper.make_node("Mul", [f"l{index}_silu", f"l{index}_up"], [f"l{index}_gated"])
            )
            nodes.append(helper.make_node("MatMul", [f"l{index}_gated", dw], [f"l{index}_mlp_out"]))
            nodes.append(
                helper.make_node("Add", [f"l{index}_res1", f"l{index}_mlp_out"], [f"x_{index + 1}"])
            )
            cur_x = f"x_{index + 1}"

        final_h = rms_norm(cur_x, "norm.weight", "final_rn")
        nodes.append(helper.make_node("Identity", [final_h], ["last_hidden_state"]))

    graph = helper.make_graph(
        nodes,
        "F2LLM_v2_160M",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        [
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.FLOAT, ["batch", "seq", h]
            )
        ],
        initializer=initializers,
    )
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save_model(model_proto, str(output_path))


def ensure_converted_onnx(
    model_directory: Path,
    cache_directory: Path,
    config: ModelConfig,
    *,
    convert: Callable[[Path, ModelConfig, Path], None] = export_qwen3_onnx,
) -> Path:
    """Return the cached ONNX graph for *model_directory*, building it atomically once."""

    target = converted_onnx_path(cache_directory, model_directory)
    if target.is_file():
        return target
    safetensors_path = model_directory / DEFAULT_MODEL_ARTIFACT
    if not safetensors_path.is_file():
        raise ValueError(f"The model snapshot has no safetensors artifact at {safetensors_path}")

    logger.info("Converting the safetensors weights of %s to ONNX once", model_directory.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.{os.getpid()}{_TEMPORARY_ONNX_SUFFIX}")
    try:
        convert(safetensors_path, config, temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _discard_superseded_onnx_conversions(target)
    return target


def _session_options(threads: int | None, enable_cpu_mem_arena: bool) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = enable_cpu_mem_arena
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    return options


def create_session(
    model_path: Path,
    *,
    providers: tuple[str, ...],
    threads: int | None,
    enable_cpu_mem_arena: bool,
) -> _Session:
    """Create a conventional ONNX Runtime session for a built-in provider."""

    import onnxruntime as ort

    kwargs: dict[str, Any] = {
        "sess_options": _session_options(threads, enable_cpu_mem_arena),
    }
    if providers:
        kwargs["providers"] = list(providers)
    return cast(
        _Session,
        ort.InferenceSession(str(model_path), **kwargs),
    )


def _normalized_platform_name(platform_name: str | None = None) -> str:
    raw = (platform_name or sys.platform).lower()
    return "win32" if raw.startswith("win") else raw


def create_webgpu_session(
    model_path: Path,
    *,
    threads: int | None,
    enable_cpu_mem_arena: bool,
    dawn_backend_type: str | None = None,
    platform_name: str | None = None,
) -> tuple[_Session, str]:
    """Register the native WebGPU plugin and attach its discovered device.

    When *dawn_backend_type* is ``"Vulkan"``, Dawn is explicitly instructed to
    open a Vulkan adapter rather than its OS default (D3D12 on Windows). Platforms
    whose Dawn build lacks Vulkan (macOS) are refused here before Dawn aborts.
    """

    if dawn_backend_type == "Vulkan":
        normalized = _normalized_platform_name(platform_name)
        if normalized not in VULKAN_DAWN_PLATFORMS:
            raise RuntimeError(
                f"Dawn's Vulkan backend is only available on Linux and Windows, not {normalized}"
            )

    import onnxruntime as ort
    import onnxruntime_ep_webgpu as webgpu_ep  # type: ignore[import-not-found]

    provider = str(webgpu_ep.get_ep_name())
    ort.register_execution_provider_library(
        "code-indexing-mcp_webgpu_ep", webgpu_ep.get_library_path()
    )
    devices = [device for device in ort.get_ep_devices() if str(device.ep_name) == provider]
    if not devices:
        raise RuntimeError("The WebGPU plugin registered but exposed no WebGPU device")

    options = _session_options(threads, enable_cpu_mem_arena)
    if dawn_backend_type is not None:
        options.add_session_config_entry(
            "ep.webgpuexecutionprovider.dawnBackendType", dawn_backend_type
        )
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(str(model_path), sess_options=options)
    return cast(_Session, session), provider


class DirectOnnxEmbedding:
    """The index-compatible ``F2LLM-v2-160M`` model executed directly by ONNX Runtime."""

    def __init__(
        self,
        cache_directory: Path,
        *,
        offline: bool,
        threads: int | None,
        enable_cpu_mem_arena: bool,
        providers: tuple[str, ...] = (),
        model_id: str = DEFAULT_MODEL,
        accelerator: str = "",
    ) -> None:
        model_directory = resolve_model_snapshot(
            cache_directory, model_id=model_id, offline=offline
        )
        config = read_model_config(model_directory)
        model_path = ensure_converted_onnx(model_directory, cache_directory, config)
        self.tokenizer = load_tokenizer(model_directory)
        self.query_prompt = read_query_prompt(model_directory)
        plugin_provider: str | None = None
        if accelerator in {"webgpu", "vulkan"}:
            self.model, plugin_provider = create_webgpu_session(
                model_path,
                threads=threads,
                enable_cpu_mem_arena=enable_cpu_mem_arena,
                dawn_backend_type="Vulkan" if accelerator == "vulkan" else None,
            )
            if providers and plugin_provider != providers[0]:
                raise RuntimeError(
                    f"The WebGPU plugin registered {plugin_provider}, expected {providers[0]}"
                )
        else:
            self.model = create_session(
                model_path,
                providers=providers,
                threads=threads,
                enable_cpu_mem_arena=enable_cpu_mem_arena,
            )
        # Plugin registration and device discovery only prove that a provider
        # was available to request. The created session is authoritative: ONNX
        # Runtime may still omit a provider it could not initialize, and the
        # caller must see that omission so it can reject a silent CPU fallback.
        self.resolved_providers = tuple(
            dict.fromkeys(str(name) for name in self.model.get_providers())
        )

    def _embed_raw(self, texts: list[str]) -> FloatArray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.asarray([row.ids for row in encoded], dtype=np.int64)
        attention_mask = np.asarray([row.attention_mask for row in encoded], dtype=np.int64)
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("Tokenizer produced an empty token sequence")
        input_names = {item.name for item in self.model.get_inputs()}
        inputs: dict[str, NDArray[Any]] = {"input_ids": input_ids}
        if "attention_mask" in input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros(input_ids.shape, dtype=np.int64)
        outputs = self.model.run(None, inputs)
        if not outputs:
            raise ValueError("The ONNX model returned no outputs")
        return last_token_pool_and_normalize(outputs[0], attention_mask)

    def passage_embed(self, documents: str | Iterable[str]) -> Iterable[FloatArray]:
        texts = [documents] if isinstance(documents, str) else list(documents)
        if not texts:
            return
        yield from self._embed_raw(texts)

    def query_embed(self, query: str | Iterable[str]) -> Iterable[FloatArray]:
        queries = [query] if isinstance(query, str) else list(query)
        if not queries:
            return
        prompted = [f"{self.query_prompt}{item}" if self.query_prompt else item for item in queries]
        yield from self._embed_raw(prompted)
