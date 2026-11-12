from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime
import pytest

from code_indexing_mcp.direct_onnx import (
    DEFAULT_MODEL_ARTIFACT,
    DirectOnnxEmbedding,
    ModelConfig,
    converted_onnx_path,
    create_webgpu_session,
    ensure_converted_onnx,
    export_qwen3_onnx,
    last_token_pool_and_normalize,
    load_tokenizer,
    mean_pool_and_normalize,
    read_model_config,
    read_query_prompt,
    resolve_model_snapshot,
)
from code_indexing_mcp.embedding import DEFAULT_MODEL, DEFAULT_QUERY_PROMPT
from code_indexing_mcp.safetensors_io import write_safetensors_bf16

SYNTH_CONFIG = ModelConfig(
    hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    intermediate_size=12,
    vocab_size=16,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
)


def _synthetic_qwen3_tensors(config: ModelConfig = SYNTH_CONFIG) -> dict[str, np.ndarray[Any, Any]]:
    rng = np.random.default_rng(1234)

    def rand(*shape: int) -> np.ndarray[Any, Any]:
        return (rng.standard_normal(shape) * 0.15).astype(np.float32)

    h = config.hidden_size
    qh = config.num_attention_heads
    kvh = config.num_key_value_heads
    hd = config.head_dim
    ff = config.intermediate_size
    tensors: dict[str, np.ndarray[Any, Any]] = {
        "embed_tokens.weight": rand(config.vocab_size, h),
        "norm.weight": np.ones((h,), dtype=np.float32),
    }
    for i in range(config.num_hidden_layers):
        p = f"layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones((h,), dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones((h,), dtype=np.float32)
        tensors[f"{p}.self_attn.q_norm.weight"] = np.ones((hd,), dtype=np.float32)
        tensors[f"{p}.self_attn.k_norm.weight"] = np.ones((hd,), dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = rand(qh * hd, h)
        tensors[f"{p}.self_attn.k_proj.weight"] = rand(kvh * hd, h)
        tensors[f"{p}.self_attn.v_proj.weight"] = rand(kvh * hd, h)
        tensors[f"{p}.self_attn.o_proj.weight"] = rand(h, qh * hd)
        tensors[f"{p}.mlp.gate_proj.weight"] = rand(ff, h)
        tensors[f"{p}.mlp.up_proj.weight"] = rand(ff, h)
        tensors[f"{p}.mlp.down_proj.weight"] = rand(h, ff)
    return tensors


def _write_synthetic_snapshot(directory: Path, config: ModelConfig = SYNTH_CONFIG) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_act": "silu",
                "attention_bias": False,
                "use_sliding_window": False,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "intermediate_size": config.intermediate_size,
                "vocab_size": config.vocab_size,
                "rms_norm_eps": config.rms_norm_eps,
                "rope_theta": config.rope_theta,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )
    (directory / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": DEFAULT_QUERY_PROMPT, "document": ""}}),
        encoding="utf-8",
    )
    write_safetensors_bf16(directory / DEFAULT_MODEL_ARTIFACT, _synthetic_qwen3_tensors(config))
    return directory


class _Encoding:
    def __init__(self, ids: list[int], attention_mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = attention_mask
        self.offsets = [(i, i + 1) for i in range(len(ids))]

    def __len__(self) -> int:
        return len(self.ids)


class _Tokenizer:
    def __init__(self) -> None:
        self.padding: dict[str, object] | None = {"pre_baked": True}
        self.cleared_padding = False
        self.truncation: dict[str, object] | None = None
        self.added: list[object] = []
        self.documents: list[str] = []

    def enable_truncation(self, *, max_length: int) -> None:
        self.truncation = {"max_length": max_length}

    def no_padding(self) -> None:
        self.cleared_padding = True
        self.padding = None

    def enable_padding(self, *, direction: str = "right", pad_id: int, pad_token: str) -> None:
        self.padding = {"direction": direction, "pad_id": pad_id, "pad_token": pad_token}

    def token_to_id(self, token: str) -> int | None:
        return 9 if token == "<|endoftext|>" else None

    def add_special_tokens(self, tokens: list[object]) -> None:
        self.added.extend(tokens)

    def encode_batch(self, documents: list[str]) -> list[_Encoding]:
        self.documents = documents
        if len(documents) == 1:
            return [_Encoding([4, 5, 6], [1, 1, 1])]
        return [
            _Encoding([1, 2, 0], [1, 1, 0]),
            _Encoding([3, 4, 5], [1, 1, 1]),
        ]

    def encode(self, text: str) -> _Encoding:
        return _Encoding([ord(char) % 16 for char in text], [1] * len(text))


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    def __init__(self) -> None:
        self.last_inputs: dict[str, np.ndarray[Any, Any]] = {}

    def get_inputs(self) -> list[_Input]:
        return [_Input("input_ids")]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(
        self, outputs: object, inputs: dict[str, np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        self.last_inputs = inputs
        batch = inputs["input_ids"].shape[0]
        if batch == 1:
            return [np.asarray([[[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]]], dtype=np.float32)]
        return [
            np.asarray(
                [
                    [[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]],
                    [[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
                ],
                dtype=np.float32,
            )
        ]


def test_model_snapshot_uses_the_shared_cache_and_offline_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []
    snapshot = tmp_path / "snapshot"

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )

    resolved = resolve_model_snapshot(
        tmp_path / "models",
        model_id=DEFAULT_MODEL,
        offline=True,
    )

    assert resolved == snapshot
    assert calls == [
        {
            "repo_id": DEFAULT_MODEL,
            "allow_patterns": [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "config_sentence_transformers.json",
                DEFAULT_MODEL_ARTIFACT,
            ],
            "cache_dir": str(tmp_path / "models"),
            "local_files_only": True,
        }
    ]


def test_only_the_index_compatible_model_is_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only supports"):
        resolve_model_snapshot(tmp_path, model_id="another/model", offline=True)


def test_tokenizer_configuration_clears_fixed_padding_and_resolves_pad_token_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Qwen3 config.json omits pad_token_id; load_tokenizer resolves it via token_to_id.
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 40960}))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_max_length": 8192,
                "max_length": 8192,
                "pad_token": {"content": "<|endoftext|>"},
            }
        )
    )
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "eos_token": "<|im_end|>",
                "additional_special_tokens": {
                    "content": "<special>",
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                },
            }
        )
    )
    tokenizer = _Tokenizer()

    class FakeTokenizer:
        @staticmethod
        def from_file(path: str) -> _Tokenizer:
            assert path == str(tmp_path / "tokenizer.json")
            return tokenizer

    class FakeAddedToken:
        def __init__(self, **values: object) -> None:
            self.values = values

    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(AddedToken=FakeAddedToken, Tokenizer=FakeTokenizer),
    )

    loaded = load_tokenizer(tmp_path)

    assert loaded is tokenizer
    assert tokenizer.cleared_padding is True
    assert tokenizer.truncation == {"max_length": 8192}
    assert tokenizer.padding == {
        "direction": "right",
        "pad_id": 9,
        "pad_token": "<|endoftext|>",
    }
    assert tokenizer.added[0] == "<|im_end|>"
    assert isinstance(tokenizer.added[1], FakeAddedToken)
    assert tokenizer.added[1].values["content"] == "<special>"


def test_last_token_and_mean_pooling_ignore_padding_and_normalize_float32_rows() -> None:
    output = np.asarray(
        [
            [[1.0, 0.0], [3.0, 4.0], [100.0, 100.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    attention = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.int64)

    last_pooled = last_token_pool_and_normalize(output, attention)
    np.testing.assert_allclose(last_pooled[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(last_pooled[1], [0.0, 0.0], atol=1e-6)
    assert last_pooled.dtype == np.float32

    mean_pooled = mean_pool_and_normalize(output, attention)
    np.testing.assert_allclose(mean_pooled[0], [2.0 / np.hypot(2.0, 2.0), 2.0 / np.hypot(2.0, 2.0)])
    np.testing.assert_allclose(mean_pooled[1], [0.0, 0.0])
    assert mean_pooled.dtype == np.float32

    with pytest.raises(ValueError, match="empty token sequence"):
        last_token_pool_and_normalize(
            np.zeros((1, 0, 2), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
        )


def test_export_and_ensure_converted_onnx_caches_reuses_and_executes_dynamic_rope(
    tmp_path: Path,
) -> None:
    snapshot = _write_synthetic_snapshot(tmp_path / "rev123")
    config = read_model_config(snapshot)
    assert config == SYNTH_CONFIG
    assert read_query_prompt(snapshot) == DEFAULT_QUERY_PROMPT

    cache = tmp_path / "cache"
    superseded = converted_onnx_path(cache, tmp_path / "oldrev")
    superseded.parent.mkdir(parents=True, exist_ok=True)
    superseded.write_bytes(b"stale")
    in_flight = superseded.with_name("other.9999.tmp.onnx")
    in_flight.write_bytes(b"writing")

    calls = 0

    def counting_export(src: Path, cfg: ModelConfig, dst: Path) -> None:
        nonlocal calls
        calls += 1
        export_qwen3_onnx(src, cfg, dst)

    first = ensure_converted_onnx(snapshot, cache, config, convert=counting_export)
    second = ensure_converted_onnx(snapshot, cache, config, convert=counting_export)

    assert first == second == converted_onnx_path(cache, snapshot)
    assert calls == 1
    assert not superseded.exists()
    assert in_flight.is_file()

    # Execute through real ONNX Runtime CPUExecutionProvider: check right-padding
    # invariance and dynamic RoPE past 2048 tokens!
    session = onnxruntime.InferenceSession(str(first), providers=["CPUExecutionProvider"])
    unpadded_ids = np.array([[1, 5, 9, 12]], dtype=np.int64)
    padded_ids = np.array([[1, 5, 9, 12, 0, 0], [2, 3, 4, 5, 6, 7]], dtype=np.int64)
    unpadded_mask = np.array([[1, 1, 1, 1]], dtype=np.int64)
    padded_mask = np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=np.int64)

    out_unpadded = last_token_pool_and_normalize(
        session.run(None, {"input_ids": unpadded_ids})[0], unpadded_mask
    )
    out_padded = last_token_pool_and_normalize(
        session.run(None, {"input_ids": padded_ids})[0], padded_mask
    )
    np.testing.assert_allclose(out_unpadded[0], out_padded[0], atol=1e-6)

    # Dynamic RoPE beyond 2048 tokens
    long_ids = np.ones((1, 2100), dtype=np.int64)
    long_out = session.run(None, {"input_ids": long_ids})[0]
    assert long_out.shape == (1, 2100, SYNTH_CONFIG.hidden_size)
    assert np.all(np.isfinite(long_out))


def test_direct_model_builds_exact_onnx_inputs_and_handles_asymmetric_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = _write_synthetic_snapshot(tmp_path / "snapshot")
    onnx_file = tmp_path / "models" / "onnx" / "cached.onnx"
    onnx_file.parent.mkdir(parents=True)
    onnx_file.write_bytes(b"onnx")
    tokenizer = _Tokenizer()
    session = _Session()
    built: list[tuple[Path, tuple[str, ...], int | None, bool]] = []

    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.resolve_model_snapshot",
        lambda *args, **kwargs: model_directory,
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.ensure_converted_onnx",
        lambda *args, **kwargs: onnx_file,
    )
    monkeypatch.setattr("code_indexing_mcp.direct_onnx.load_tokenizer", lambda path: tokenizer)

    def build_session(
        path: Path,
        *,
        providers: tuple[str, ...],
        threads: int | None,
        enable_cpu_mem_arena: bool,
    ) -> _Session:
        built.append((path, providers, threads, enable_cpu_mem_arena))
        return session

    monkeypatch.setattr("code_indexing_mcp.direct_onnx.create_session", build_session)

    model = DirectOnnxEmbedding(
        cache_directory=tmp_path / "models",
        offline=True,
        threads=3,
        enable_cpu_mem_arena=False,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        accelerator="cuda",
    )
    vectors = list(model.passage_embed(["short", "longer"]))

    assert built == [
        (
            onnx_file,
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            3,
            False,
        )
    ]
    assert tokenizer.documents == ["short", "longer"]
    assert session.last_inputs["input_ids"].dtype == np.int64
    np.testing.assert_allclose(vectors, [[1.0, 0.0], [0.0, 1.0]])
    assert model.resolved_providers == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )

    # Query embedding prepends the asymmetric instruction prompt; passage embedding does not.
    query_vectors = list(model.query_embed("find auth"))
    assert tokenizer.documents == [f"{DEFAULT_QUERY_PROMPT}find auth"]
    np.testing.assert_allclose(query_vectors, [[0.6, 0.8]], atol=1e-6)


def test_webgpu_and_vulkan_sessions_register_plugin_and_configure_dawn_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []
    webgpu_device = SimpleNamespace(ep_name="WebGpuExecutionProvider")
    cpu_device = SimpleNamespace(ep_name="CPUExecutionProvider")
    session = _Session()

    class SessionOptions:
        graph_optimization_level: object | None = None
        enable_cpu_mem_arena = True
        intra_op_num_threads = 0
        inter_op_num_threads = 0

        def add_session_config_entry(self, key: str, value: str) -> None:
            events.append(("config", (key, value)))

        def add_provider_for_devices(self, devices: list[object], options: dict[str, str]) -> None:
            events.append(("devices", (devices, options)))

    fake_ort = SimpleNamespace(
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        SessionOptions=SessionOptions,
        register_execution_provider_library=lambda name, path: events.append(
            ("register", (name, path))
        ),
        get_ep_devices=lambda: [cpu_device, webgpu_device],
        InferenceSession=lambda path, *, sess_options: (
            events.append(("session", (path, sess_options))) or session
        ),
    )
    fake_plugin = SimpleNamespace(
        get_library_path=lambda: "/runtime/webgpu.so",
        get_ep_name=lambda: "WebGpuExecutionProvider",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", fake_plugin)

    resolved_session, provider = create_webgpu_session(
        tmp_path / "model.onnx",
        threads=2,
        enable_cpu_mem_arena=False,
        dawn_backend_type="Vulkan",
        platform_name="linux",
    )

    assert resolved_session is session
    assert provider == "WebGpuExecutionProvider"
    assert events[0] == (
        "register",
        ("code-indexing-mcp_webgpu_ep", "/runtime/webgpu.so"),
    )
    assert events[1] == (
        "config",
        ("ep.webgpuexecutionprovider.dawnBackendType", "Vulkan"),
    )
    assert events[2] == ("devices", ([webgpu_device], {}))
    assert events[3][0] == "session"

    # Vulkan Dawn backend must be refused cleanly on macOS before touching Dawn C++.
    with pytest.raises(RuntimeError, match="only available on Linux and Windows"):
        create_webgpu_session(
            tmp_path / "model.onnx",
            threads=1,
            enable_cpu_mem_arena=False,
            dawn_backend_type="Vulkan",
            platform_name="darwin",
        )


def test_webgpu_session_refuses_a_registered_plugin_with_no_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_ort = SimpleNamespace(
        register_execution_provider_library=lambda name, path: None,
        get_ep_devices=lambda: [SimpleNamespace(ep_name="CPUExecutionProvider")],
    )
    fake_plugin = SimpleNamespace(
        get_library_path=lambda: "/runtime/webgpu.dylib",
        get_ep_name=lambda: "WebGpuExecutionProvider",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", fake_plugin)

    with pytest.raises(RuntimeError, match="no WebGPU device"):
        create_webgpu_session(
            tmp_path / "model.onnx",
            threads=2,
            enable_cpu_mem_arena=False,
        )


@pytest.mark.parametrize("accelerator", ["webgpu", "vulkan"])
def test_webgpu_and_vulkan_models_report_only_providers_the_session_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accelerator: str
) -> None:
    model_directory = _write_synthetic_snapshot(tmp_path / "snapshot")
    onnx_file = tmp_path / "models" / "onnx" / "cached.onnx"
    onnx_file.parent.mkdir(parents=True)
    onnx_file.write_bytes(b"onnx")
    session = _Session()
    captured_dawn: list[object] = []
    monkeypatch.setattr(session, "get_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.resolve_model_snapshot",
        lambda *args, **kwargs: model_directory,
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.ensure_converted_onnx",
        lambda *args, **kwargs: onnx_file,
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.load_tokenizer",
        lambda path: _Tokenizer(),
    )

    def fake_webgpu(*args: object, **kwargs: object) -> tuple[_Session, str]:
        captured_dawn.append(kwargs.get("dawn_backend_type"))
        return session, "WebGpuExecutionProvider"

    monkeypatch.setattr("code_indexing_mcp.direct_onnx.create_webgpu_session", fake_webgpu)

    model = DirectOnnxEmbedding(
        cache_directory=tmp_path / "models",
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
        providers=("WebGpuExecutionProvider", "CPUExecutionProvider"),
        accelerator=accelerator,
    )

    assert model.resolved_providers == ("CPUExecutionProvider",)
    assert captured_dawn == ["Vulkan" if accelerator == "vulkan" else None]


def test_direct_model_refuses_empty_token_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = _write_synthetic_snapshot(tmp_path / "snapshot")
    onnx_file = tmp_path / "models" / "onnx" / "cached.onnx"
    onnx_file.parent.mkdir(parents=True)
    onnx_file.write_bytes(b"onnx")
    tokenizer = _Tokenizer()
    monkeypatch.setattr(
        tokenizer,
        "encode_batch",
        lambda documents: [_Encoding([], []) for _ in documents],
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.resolve_model_snapshot",
        lambda *args, **kwargs: model_directory,
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.ensure_converted_onnx",
        lambda *args, **kwargs: onnx_file,
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.load_tokenizer",
        lambda path: tokenizer,
    )
    monkeypatch.setattr(
        "code_indexing_mcp.direct_onnx.create_session",
        lambda *args, **kwargs: _Session(),
    )

    model = DirectOnnxEmbedding(
        cache_directory=tmp_path / "models",
        offline=True,
        threads=1,
        enable_cpu_mem_arena=False,
    )
    with pytest.raises(ValueError, match="empty token sequence"):
        list(model.passage_embed([""]))
