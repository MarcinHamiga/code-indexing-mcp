from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from incode_mcp.direct_onnx import (
    DEFAULT_MODEL_ARTIFACT,
    DirectOnnxEmbedding,
    create_webgpu_session,
    load_tokenizer,
    mean_pool_and_normalize,
    resolve_model_snapshot,
)


class _Encoding:
    def __init__(self, ids: list[int], attention_mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.ids)


class _Tokenizer:
    def __init__(self) -> None:
        self.padding: dict[str, object] | None = None
        self.truncation: dict[str, object] | None = None
        self.added: list[object] = []
        self.documents: list[str] = []

    def enable_truncation(self, *, max_length: int) -> None:
        self.truncation = {"max_length": max_length}

    def enable_padding(self, *, pad_id: int, pad_token: str) -> None:
        self.padding = {"pad_id": pad_id, "pad_token": pad_token}

    def add_special_tokens(self, tokens: list[object]) -> None:
        self.added.extend(tokens)

    def encode_batch(self, documents: list[str]) -> list[_Encoding]:
        self.documents = documents
        return [
            _Encoding([11, 12, 0], [1, 1, 0]),
            _Encoding([21, 22, 23], [1, 1, 1]),
        ]

    def encode(self, text: str) -> _Encoding:
        return _Encoding([ord(char) for char in text], [1] * len(text))


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    def __init__(self) -> None:
        self.last_inputs: dict[str, np.ndarray[Any, Any]] = {}

    def get_inputs(self) -> list[_Input]:
        return [_Input("input_ids"), _Input("attention_mask"), _Input("token_type_ids")]

    def get_providers(self) -> list[str]:
        return ["MIGraphXExecutionProvider", "CPUExecutionProvider"]

    def run(
        self, outputs: object, inputs: dict[str, np.ndarray[Any, Any]]
    ) -> list[np.ndarray[Any, Any]]:
        self.last_inputs = inputs
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
        model_id="jinaai/jina-embeddings-v2-base-code",
        offline=True,
    )

    assert resolved == snapshot
    assert calls == [
        {
            "repo_id": "jinaai/jina-embeddings-v2-base-code",
            "allow_patterns": [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                DEFAULT_MODEL_ARTIFACT,
            ],
            "cache_dir": str(tmp_path / "models"),
            "local_files_only": True,
        }
    ]


def test_only_the_index_compatible_model_is_accepted(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only supports"):
        resolve_model_snapshot(tmp_path, model_id="another/model", offline=True)


def test_tokenizer_configuration_matches_the_model_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"pad_token_id": 7}))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 8192, "max_length": 4096, "pad_token": "<pad>"})
    )
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "unk_token": "<unk>",
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
    assert tokenizer.truncation == {"max_length": 4096}
    assert tokenizer.padding == {"pad_id": 7, "pad_token": "<pad>"}
    assert tokenizer.added[0] == "<unk>"
    assert isinstance(tokenizer.added[1], FakeAddedToken)
    assert tokenizer.added[1].values["content"] == "<special>"


def test_mean_pooling_ignores_padding_and_normalizes_float32_rows() -> None:
    output = np.asarray(
        [
            [[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    attention = np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.int64)

    pooled = mean_pool_and_normalize(output, attention)

    np.testing.assert_allclose(pooled[0], [1.0, 0.0])
    np.testing.assert_allclose(pooled[1], [0.0, 0.0])
    assert pooled.dtype == np.float32


def test_direct_model_builds_exact_onnx_inputs_and_reports_resolved_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_directory = tmp_path / "snapshot"
    (model_directory / "onnx").mkdir(parents=True)
    (model_directory / DEFAULT_MODEL_ARTIFACT).write_bytes(b"onnx")
    tokenizer = _Tokenizer()
    session = _Session()
    built: list[tuple[Path, tuple[str, ...], int | None, bool]] = []

    monkeypatch.setattr(
        "incode_mcp.direct_onnx.resolve_model_snapshot",
        lambda *args, **kwargs: model_directory,
    )
    monkeypatch.setattr("incode_mcp.direct_onnx.load_tokenizer", lambda path: tokenizer)

    def build_session(
        path: Path,
        *,
        providers: tuple[str, ...],
        threads: int | None,
        enable_cpu_mem_arena: bool,
    ) -> _Session:
        built.append((path, providers, threads, enable_cpu_mem_arena))
        return session

    monkeypatch.setattr("incode_mcp.direct_onnx.create_session", build_session)

    model = DirectOnnxEmbedding(
        cache_directory=tmp_path / "models",
        offline=True,
        threads=3,
        enable_cpu_mem_arena=False,
        providers=("MIGraphXExecutionProvider", "CPUExecutionProvider"),
    )
    vectors = list(model.passage_embed(["short", "longer"]))

    assert built == [
        (
            model_directory / DEFAULT_MODEL_ARTIFACT,
            ("MIGraphXExecutionProvider", "CPUExecutionProvider"),
            3,
            False,
        )
    ]
    assert tokenizer.documents == ["short", "longer"]
    assert session.last_inputs["input_ids"].dtype == np.int64
    assert session.last_inputs["attention_mask"].dtype == np.int64
    np.testing.assert_array_equal(
        session.last_inputs["token_type_ids"], np.zeros((2, 3), dtype=np.int64)
    )
    np.testing.assert_allclose(vectors, [[1.0, 0.0], [0.0, 1.0]])
    assert model.resolved_providers == (
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    )


def test_webgpu_session_registers_the_plugin_and_attaches_its_device(
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

        def add_provider_for_devices(
            self, devices: list[object], options: dict[str, str]
        ) -> None:
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
        get_library_path=lambda: "/runtime/webgpu.dylib",
        get_ep_name=lambda: "WebGpuExecutionProvider",
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", fake_plugin)

    resolved_session, provider = create_webgpu_session(
        tmp_path / "model.onnx",
        threads=2,
        enable_cpu_mem_arena=False,
    )

    assert resolved_session is session
    assert provider == "WebGpuExecutionProvider"
    assert events[0] == (
        "register",
        ("incode_webgpu_ep", "/runtime/webgpu.dylib"),
    )
    assert events[1] == ("devices", ([webgpu_device], {}))
    assert events[2][0] == "session"


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
