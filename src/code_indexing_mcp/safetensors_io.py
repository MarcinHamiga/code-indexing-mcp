"""Minimal, strictly validated reader for ``.safetensors`` weight archives.

Both the direct ONNX converter and the Apple Silicon MLX converter read the
same official ``model.safetensors`` snapshot from Hugging Face, without adding
PyTorch or a compiled ``safetensors`` wheel to any runtime environment.

The file format is fixed:
  * 8-byte unsigned little-endian header length ``N``;
  * ``N`` UTF-8 JSON bytes mapping each tensor name (plus an optional
    ``"__metadata__"`` object) to ``{"dtype": ..., "shape": [...],
    "data_offsets": [start, end]}``;
  * raw little-endian tensor payloads addressed relative to ``8 + N``.
"""

from __future__ import annotations

import json
import mmap
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

MAX_HEADER_BYTES = 100 * 1024 * 1024
_ELEMENT_SIZES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}


@dataclass(frozen=True)
class TensorEntry:
    """Validated metadata for one tensor inside a ``.safetensors`` archive."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


class SafetensorsArchive:
    """Memory-mapped ``.safetensors`` reader with strict shape and bounds checks."""

    def __init__(
        self,
        path: Path,
        entries: dict[str, TensorEntry],
        buffer: mmap.mmap | bytes,
        data_offset: int,
    ) -> None:
        self.path = path
        self.entries = entries
        self._buffer = buffer
        self._data_offset = data_offset

    def take_float32(
        self,
        name: str,
        shape: tuple[int, ...],
        described_as: str,
    ) -> FloatArray:
        """Read *name*, verify *shape*, and return a contiguous ``float32`` array."""

        entry = self.entries.get(name)
        if entry is None:
            raise ValueError(
                f"The safetensors archive at {self.path} has no {described_as} tensor "
                f"named {name!r}"
            )
        if entry.shape != shape:
            raise ValueError(
                f"The {described_as} tensor {name!r} is {entry.shape}, expected {shape}"
            )
        view = memoryview(self._buffer)[
            self._data_offset + entry.start : self._data_offset + entry.end
        ]
        if entry.dtype == "BF16":
            raw_u16 = np.frombuffer(view, dtype="<u2")
            shifted = raw_u16.astype(np.uint32) << np.uint32(16)
            return np.array(shifted.view(np.float32).reshape(shape), dtype=np.float32, copy=True)
        if entry.dtype == "F16":
            raw_f16 = np.frombuffer(view, dtype="<f2").reshape(shape)
            return np.asarray(raw_f16, dtype=np.float32).copy()
        if entry.dtype == "F32":
            raw_f32 = np.frombuffer(view, dtype="<f4").reshape(shape)
            return np.array(raw_f32, dtype=np.float32, copy=True)
        raise ValueError(
            f"The {described_as} tensor {name!r} has unsupported safetensors dtype "
            f"{entry.dtype!r}; expected BF16, F16, or F32"
        )


def _parse_header(path: Path, raw_header: bytes, payload_bytes: int) -> dict[str, TensorEntry]:
    try:
        parsed = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse the safetensors header at {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"The safetensors header at {path} is not a JSON object")

    entries: dict[str, TensorEntry] = {}
    for name, value in parsed.items():
        if name == "__metadata__":
            if not isinstance(value, dict):
                raise ValueError(f"Invalid __metadata__ entry in {path}")
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Invalid tensor descriptor for {name!r} in {path}")
        dtype = value.get("dtype")
        raw_shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _ELEMENT_SIZES:
            raise ValueError(
                f"Tensor {name!r} in {path} has unsupported dtype {dtype!r}; "
                "expected BF16, F16, or F32"
            )
        if not isinstance(raw_shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in raw_shape
        ):
            raise ValueError(f"Tensor {name!r} in {path} has invalid shape {raw_shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(pos, bool) or not isinstance(pos, int) for pos in offsets)
        ):
            raise ValueError(f"Tensor {name!r} in {path} has invalid data_offsets {offsets!r}")
        start, end = offsets[0], offsets[1]
        if start < 0 or end < start or end > payload_bytes:
            raise ValueError(
                f"Tensor {name!r} in {path} has out-of-bounds data_offsets [{start}, {end}] "
                f"for payload of {payload_bytes} bytes"
            )
        shape = tuple(int(dim) for dim in raw_shape)
        element_count = 1
        for dim in shape:
            element_count *= dim
        expected_bytes = element_count * _ELEMENT_SIZES[dtype]
        if end - start != expected_bytes:
            raise ValueError(
                f"Tensor {name!r} in {path} spans {end - start} bytes, "
                f"expected {expected_bytes} for {dtype} {shape}"
            )
        entries[name] = TensorEntry(
            name=name,
            dtype=dtype,
            shape=shape,
            start=start,
            end=end,
        )
    return entries


@contextmanager
def open_safetensors(path: Path) -> Iterator[SafetensorsArchive]:
    """Open *path* as a validated, memory-mapped :class:`SafetensorsArchive`."""

    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"Could not open the safetensors archive at {path}: {exc}") from exc
    with handle:
        size = path.stat().st_size
        if size < 8:
            raise ValueError(f"The safetensors archive at {path} is truncated ({size} bytes)")
        (header_size,) = struct.unpack("<Q", handle.read(8))
        if header_size == 0 or header_size > MAX_HEADER_BYTES:
            raise ValueError(
                f"The safetensors archive at {path} has invalid header length {header_size}"
            )
        data_offset = 8 + header_size
        if data_offset > size:
            raise ValueError(
                f"The safetensors archive at {path} ends before its {header_size}-byte header"
            )
        raw_header = handle.read(header_size)
        entries = _parse_header(path, raw_header, size - data_offset)
        if size == data_offset:
            yield SafetensorsArchive(path, entries, b"", 0)
            return
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            yield SafetensorsArchive(path, entries, mapped, data_offset)
        finally:
            mapped.close()


def write_safetensors_bf16(path: Path, tensors: dict[str, FloatArray]) -> None:
    """Serialize ``float32`` arrays as a ``BF16`` ``.safetensors`` file (for tests)."""

    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    chunks: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        f32 = np.ascontiguousarray(array, dtype="<f4")
        u32 = f32.view(np.uint32)
        # Round-to-nearest-even to BF16 so round-trips match PyTorch/MLX BF16.
        rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
        bf16 = ((u32 + rounding_bias) >> np.uint32(16)).astype("<u2")
        payload = bf16.tobytes()
        end = offset + len(payload)
        header[name] = {
            "dtype": "BF16",
            "shape": list(f32.shape),
            "data_offsets": [offset, end],
        }
        chunks.append(payload)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for chunk in chunks:
            handle.write(chunk)
