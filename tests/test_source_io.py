"""Source access rejects unsafe live paths before reading their contents."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from code_indexing_mcp.errors import CodeIndexingError
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.source_io import SourceReadError, read_source


def test_reader_preserves_bytes_and_enforces_exact_size(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_bytes(b"\xef\xbb\xbfhello")
    content, info = read_source(tmp_path, Path("source.py"), 8)
    assert content == b"\xef\xbb\xbfhello"
    assert info.st_size == 8
    with pytest.raises(SourceReadError, match="oversized"):
        read_source(tmp_path, Path("source.py"), 7)


@pytest.mark.skipif(os.name == "nt", reason="Exercises unsupported non-Windows fallback")
def test_reader_fails_closed_without_safe_platform_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_bytes(b"source")
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(OSError, match=r"safe.*unavailable"):
        read_source(tmp_path, Path("source.py"), 1024)


def test_reader_preserves_crlf_and_control_z(tmp_path: Path) -> None:
    source = b"a = 1\r\n# control-z: \x1a\r\nb = 2\r\n"
    (tmp_path / "source.py").write_bytes(source)
    assert read_source(tmp_path, Path("source.py"), 1024)[0] == source


@pytest.mark.parametrize("relative", ["../outside.py", "/outside.py"])
def test_reader_rejects_non_relative_paths(tmp_path: Path, relative: str) -> None:
    with pytest.raises(SourceReadError, match="path_escapes_root"):
        read_source(tmp_path, Path(relative), 1024)


def test_reference_source_reader_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"SECRET = 1\n")
    (root / "source.py").symlink_to(outside)
    with pytest.raises(CodeIndexingError) as caught:
        ReferenceService._file_bytes(root, "source.py", {})
    assert caught.value.details["reason"] == "symlink"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
def test_reference_source_reader_does_not_block_on_fifo(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "source.py")
    code = (
        "from pathlib import Path; import sys; "
        "from code_indexing_mcp.reference_service import ReferenceService; "
        "ReferenceService._file_bytes(Path(sys.argv[1]), 'source.py', {})"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, timeout=5
    )
    assert result.returncode != 0
    assert b"not_regular" in result.stderr


def test_patch_service_rejects_a_consumer_replaced_with_an_external_symlink(tmp_path: Path) -> None:
    from test_refactors import _emit, _indexed_service

    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "consumer.py": "from lib import answer\nanswer()\n",
        },
    )
    outside = tmp_path / "outside.py"
    outside.write_text("OUTSIDE = 123\n")
    consumer = tmp_path / "repo" / "consumer.py"
    consumer.unlink()
    consumer.symlink_to(outside)
    with pytest.raises(CodeIndexingError) as caught:
        _emit(service, project_id, "lib.py", "answer", "renamed")
    assert caught.value.details["path"] == "consumer.py"
    assert caught.value.details["reason"] == "symlink"
    assert outside.read_text() == "OUTSIDE = 123\n"


def test_reference_source_reader_rejects_oversized_live_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("code_indexing_mcp.reference_service.MAX_FILE_BYTES_CEILING", 8)
    (tmp_path / "source.py").write_bytes(b"x" * 9)
    with pytest.raises(CodeIndexingError) as caught:
        ReferenceService._file_bytes(tmp_path, "source.py", {})
    assert caught.value.details["reason"] == "oversized"


def test_reader_rejects_growth_without_unbounded_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"small")
    original = os.read
    requested = []

    def growing_read(descriptor: int, count: int) -> bytes:
        requested.append(count)
        with source.open("ab") as writer:
            writer.write(b"x" * 100)
        return original(descriptor, count)

    monkeypatch.setattr("code_indexing_mcp.source_io.os.read", growing_read)
    with pytest.raises(SourceReadError, match="oversized"):
        read_source(tmp_path, Path("source.py"), 8)
    assert requested == [9]
