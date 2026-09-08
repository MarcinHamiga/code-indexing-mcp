"""Exercise Windows handle policy without requiring a Windows host."""

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_indexing_mcp import windows_source_io as windows


class FakeKernel:
    def __init__(self, path: Path, attributes: int = 0x10) -> None:
        self.path = str(path)
        self.attributes = attributes
        self.open_calls: list[tuple[object, ...]] = []
        self.closed: list[int] = []

    def CreateFileW(self, *args: object) -> int:
        self.open_calls.append(args)
        return 123

    def GetFileInformationByHandleEx(self, handle, kind, output, size):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_uint32))[0] = self.attributes
        return 1

    def GetFinalPathNameByHandleW(self, handle, output, length, flags):
        output.value = self.path
        return len(self.path)

    def GetFileType(self, handle):
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


def test_directory_handles_reject_reparse_points_and_close_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeKernel(tmp_path, attributes=0x410)
    monkeypatch.setattr(windows, "_kernel32", lambda: api)
    with pytest.raises(OSError, match="reparse"):
        windows._open_handle(tmp_path, directory=True)
    assert api.closed == [123]
    # Opening the reparse point itself prevents following the substituted link.
    assert api.open_calls[0][5] & 0x00200000
    # No FILE_SHARE_DELETE: the directory stays fixed while held.
    assert not api.open_calls[0][2] & 0x00000004


def test_opened_handle_must_name_the_expected_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = FakeKernel(tmp_path / "outside")
    monkeypatch.setattr(windows, "_kernel32", lambda: api)
    with pytest.raises(OSError, match="changed"):
        windows._open_handle(tmp_path / "repo", directory=True)
    assert api.closed == [123]


def test_pinned_directory_releases_all_parents_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    closed: list[int] = []

    def open_handle(path: Path, *, directory: bool) -> int:
        opened.append(path)
        if path.name == "bad":
            raise OSError("reparse")
        return len(opened)

    monkeypatch.setattr(windows, "_open_handle", open_handle)
    monkeypatch.setattr(windows, "_close_handle", closed.append)
    with (
        pytest.raises(OSError, match="reparse"),
        windows.pinned_directory(tmp_path, Path("sub/bad")),
    ):
        pytest.fail("must not enter an unsafe directory")
    assert opened == [tmp_path, tmp_path / "sub", tmp_path / "sub/bad"]
    assert closed == [2, 1]


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (r"\\?\C:\Repo\source.py", r"c:\repo\source.py"),
        (r"\\?\UNC\server\share\repo", r"\\server\share\repo"),
    ],
)
def test_final_path_normalizes_windows_device_prefixes(actual: str, expected: str) -> None:
    assert windows._normalized_path(actual) == windows._normalized_path(expected)


def test_source_descriptor_is_binary_and_validated_before_parent_pins_are_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"x\r\n\x1a")
    descriptor = os.open(source, os.O_RDONLY)
    events: list[tuple[str, int]] = []

    def transfer(handle: int, flags: int) -> int:
        assert flags & 0x8000
        events.append(("transfer", handle))
        return descriptor

    monkeypatch.setattr(os, "O_BINARY", 0x8000, raising=False)
    monkeypatch.setattr(windows, "_open_handle", lambda path, directory: 1 if directory else 2)
    monkeypatch.setattr(windows, "_close_handle", lambda handle: events.append(("close", handle)))
    monkeypatch.setattr(windows, "_kernel32", lambda: None)
    monkeypatch.setattr(
        windows, "_validate_handle_path", lambda api, handle, path: events.append(("check", handle))
    )
    monkeypatch.setattr(
        windows.importlib,
        "import_module",
        lambda name: SimpleNamespace(open_osfhandle=transfer, get_osfhandle=lambda fd: 2),
    )
    try:
        assert windows.open_source(tmp_path, Path("source.py")) == descriptor
        assert events == [("transfer", 2), ("check", 2), ("close", 1)]
        assert os.read(descriptor, 10) == b"x\r\n\x1a"
    finally:
        os.close(descriptor)
