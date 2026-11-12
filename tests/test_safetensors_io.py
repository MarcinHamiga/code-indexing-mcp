from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from code_indexing_mcp.safetensors_io import open_safetensors, write_safetensors_bf16


def _write_raw_safetensors(path: Path, header: dict[str, object], payload: bytes) -> None:
    raw_header = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def test_bf16_round_trip_and_metadata_handling(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    original = np.array([[1.0, -2.5, 0.125], [3.75, 0.0, -0.0625]], dtype=np.float32)
    write_safetensors_bf16(path, {"embed_tokens.weight": original})

    with open_safetensors(path) as archive:
        loaded = archive.take_float32("embed_tokens.weight", (2, 3), "word embedding")

    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, original, rtol=1e-2, atol=1e-3)


def test_f16_and_f32_tensors_are_loaded_as_float32(tmp_path: Path) -> None:
    path = tmp_path / "mixed.safetensors"
    f16 = np.array([1.5, -0.5, 2.0, 4.0], dtype="<f2")
    f32 = np.array([[0.25, -1.25], [3.5, 7.0]], dtype="<f4")
    payload = f16.tobytes() + f32.tobytes()
    header = {
        "norm.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
        "proj.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [8, 24]},
    }
    _write_raw_safetensors(path, header, payload)

    with open_safetensors(path) as archive:
        norm = archive.take_float32("norm.weight", (4,), "norm")
        proj = archive.take_float32("proj.weight", (2, 2), "projection")

    np.testing.assert_allclose(norm, [1.5, -0.5, 2.0, 4.0])
    np.testing.assert_allclose(proj, [[0.25, -1.25], [3.5, 7.0]])


def test_missing_and_misshapen_tensors_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors_bf16(path, {"norm.weight": np.ones((4,), dtype=np.float32)})

    with open_safetensors(path) as archive:
        with pytest.raises(ValueError, match="no final norm tensor named 'missing'"):
            archive.take_float32("missing", (4,), "final norm")
        with pytest.raises(ValueError, match=r"final norm tensor 'norm.weight' is \(4,\)"):
            archive.take_float32("norm.weight", (8,), "final norm")


@pytest.mark.parametrize(
    ("raw_bytes", "pattern"),
    [
        (b"short", "truncated"),
        (struct.pack("<Q", 0), "invalid header length"),
        (struct.pack("<Q", 64) + b"{}", "ends before its 64-byte header"),
        (struct.pack("<Q", 2) + b"[]", "not a JSON object"),
    ],
)
def test_corrupt_archive_framing_is_refused(tmp_path: Path, raw_bytes: bytes, pattern: str) -> None:
    path = tmp_path / "corrupt.safetensors"
    path.write_bytes(raw_bytes)

    with pytest.raises(ValueError, match=pattern), open_safetensors(path):
        pass


def test_out_of_bounds_or_mismatched_offsets_are_refused(tmp_path: Path) -> None:
    oob = tmp_path / "oob.safetensors"
    _write_raw_safetensors(
        oob,
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 16]}},
        b"\x00" * 8,
    )
    with pytest.raises(ValueError, match="out-of-bounds"), open_safetensors(oob):
        pass

    wrong_span = tmp_path / "span.safetensors"
    _write_raw_safetensors(
        wrong_span,
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
        b"\x00" * 8,
    )
    with pytest.raises(ValueError, match="spans 4 bytes, expected 8"), open_safetensors(wrong_span):
        pass


def test_unsupported_dtype_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "i8.safetensors"
    _write_raw_safetensors(
        path,
        {"w": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}},
        b"\x00" * 4,
    )
    with pytest.raises(ValueError, match="unsupported dtype 'I8'"), open_safetensors(path):
        pass
