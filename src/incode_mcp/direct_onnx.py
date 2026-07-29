"""Direct ONNX passage embedding for providers FastEmbed cannot configure.

The implementation is deliberately model-specific. Index compatibility depends
on more than an ONNX file: the tokenizer, pooling, normalization, dimension, and
artifact revision all have to retain the semantics of the CPU query model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .embedding import DEFAULT_MODEL

DEFAULT_MODEL_ARTIFACT = "onnx/model.onnx"
_MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    DEFAULT_MODEL_ARTIFACT,
]

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class _Encoding(Protocol):
    ids: Sequence[int]
    attention_mask: Sequence[int]

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


def resolve_model_snapshot(
    cache_directory: Path, *, model_id: str = DEFAULT_MODEL, offline: bool
) -> Path:
    """Resolve the exact model snapshot FastEmbed uses into the shared cache."""

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


def load_tokenizer(model_directory: Path) -> _Tokenizer:
    """Load tokenizer configuration with the same rules as the CPU model."""

    from tokenizers import AddedToken, Tokenizer

    config = _json_object(model_directory / "config.json")
    tokenizer_config = _json_object(model_directory / "tokenizer_config.json")
    special_tokens = _json_object(model_directory / "special_tokens_map.json")

    raw_model_max = tokenizer_config.get("model_max_length")
    raw_max = tokenizer_config.get("max_length")
    lengths = [
        int(value)
        for value in (raw_model_max, raw_max)
        if isinstance(value, int | float) and value > 0
    ]
    if not lengths:
        raise ValueError("Tokenizer config has no positive model_max_length or max_length")
    max_context = min(lengths)

    tokenizer = Tokenizer.from_file(str(model_directory / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=max_context)
    if not tokenizer.padding:
        pad_token = tokenizer_config.get("pad_token")
        if not isinstance(pad_token, str):
            raise ValueError("Tokenizer config has no pad_token")
        tokenizer.enable_padding(pad_id=int(config.get("pad_token_id", 0)), pad_token=pad_token)

    for raw_token in special_tokens.values():
        if isinstance(raw_token, str):
            tokenizer.add_special_tokens([raw_token])
        elif isinstance(raw_token, dict):
            tokenizer.add_special_tokens([AddedToken(**raw_token)])
    return cast(_Tokenizer, tokenizer)


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


def create_session(
    model_path: Path,
    *,
    providers: tuple[str, ...],
    threads: int | None,
    enable_cpu_mem_arena: bool,
) -> _Session:
    """Create a conventional ONNX Runtime session for a built-in provider."""

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = enable_cpu_mem_arena
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    return cast(
        _Session,
        ort.InferenceSession(
            str(model_path),
            providers=list(providers),
            sess_options=options,
        ),
    )


def create_webgpu_session(
    model_path: Path,
    *,
    threads: int | None,
    enable_cpu_mem_arena: bool,
) -> tuple[_Session, str]:
    """Register the native WebGPU plugin and attach its discovered device."""

    import onnxruntime as ort
    import onnxruntime_ep_webgpu as webgpu_ep  # type: ignore[import-not-found]

    provider = str(webgpu_ep.get_ep_name())
    ort.register_execution_provider_library("incode_webgpu_ep", webgpu_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if str(device.ep_name) == provider]
    if not devices:
        raise RuntimeError("The WebGPU plugin registered but exposed no WebGPU device")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = enable_cpu_mem_arena
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
    options.add_provider_for_devices(devices, {})
    session = ort.InferenceSession(str(model_path), sess_options=options)
    return cast(_Session, session), provider


class DirectOnnxEmbedding:
    """The index-compatible Jina passage model executed directly by ONNX Runtime."""

    def __init__(
        self,
        cache_directory: Path,
        *,
        offline: bool,
        threads: int | None,
        enable_cpu_mem_arena: bool,
        providers: tuple[str, ...],
        model_id: str = DEFAULT_MODEL,
        accelerator: str = "",
    ) -> None:
        model_directory = resolve_model_snapshot(
            cache_directory, model_id=model_id, offline=offline
        )
        model_path = model_directory / DEFAULT_MODEL_ARTIFACT
        if not model_path.is_file():
            raise ValueError(f"The model snapshot has no ONNX artifact at {model_path}")
        self.tokenizer = load_tokenizer(model_directory)
        plugin_provider: str | None = None
        if accelerator == "webgpu":
            self.model, plugin_provider = create_webgpu_session(
                model_path,
                threads=threads,
                enable_cpu_mem_arena=enable_cpu_mem_arena,
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

    def passage_embed(self, documents: str | Iterable[str]) -> Iterable[FloatArray]:
        texts = [documents] if isinstance(documents, str) else list(documents)
        if not texts:
            return
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.asarray([row.ids for row in encoded], dtype=np.int64)
        attention_mask = np.asarray([row.attention_mask for row in encoded], dtype=np.int64)
        input_names = {item.name for item in self.model.get_inputs()}
        inputs: dict[str, NDArray[Any]] = {"input_ids": input_ids}
        if "attention_mask" in input_names:
            inputs["attention_mask"] = attention_mask
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros(input_ids.shape, dtype=np.int64)
        outputs = self.model.run(None, inputs)
        if not outputs:
            raise ValueError("The ONNX model returned no outputs")
        pooled = mean_pool_and_normalize(outputs[0], attention_mask)
        yield from pooled
