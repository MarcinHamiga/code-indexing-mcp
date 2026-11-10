"""Source access rejects unsafe live paths before reading their contents."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from code_indexing_mcp.errors import CodeIndexingError
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.source_io import SourceReadError, checked_source_stat, read_source


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


def test_vetted_directory_cache_spares_ancestor_lstats_but_keeps_final_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    for name in ("one.py", "two.py"):
        (deep / name).write_text("x = 1\n")
    calls: list[Path] = []
    real_lstat = Path.lstat

    def counting_lstat(self: Path) -> os.stat_result:
        calls.append(self)
        return real_lstat(self)

    monkeypatch.setattr(Path, "lstat", counting_lstat)
    vetted: set[Path] = set()

    first = checked_source_stat(tmp_path, Path("a/b/one.py"), vetted_directories=vetted)
    first_calls = len(calls)
    second = checked_source_stat(tmp_path, Path("a/b/two.py"), vetted_directories=vetted)

    assert stat.S_ISREG(first.st_mode) and stat.S_ISREG(second.st_mode)
    assert first_calls == 3
    assert len(calls) - first_calls == 1
    assert vetted == {Path("a"), Path("a/b")}


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_vetted_directory_cache_never_hides_a_symlinked_component(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "ok.py").write_text("x = 1\n")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    (tmp_path / "real" / "sub").mkdir()
    (tmp_path / "real" / "sub" / "f.py").write_text("x = 1\n")
    vetted: set[Path] = set()

    checked_source_stat(tmp_path, Path("real/ok.py"), vetted_directories=vetted)
    with pytest.raises(SourceReadError, match="symlink"):
        checked_source_stat(tmp_path, Path("link/ok.py"), vetted_directories=vetted)
    # A deeper path under a vetted ancestor still vets its new directories.
    (tmp_path / "real" / "sub").rename(tmp_path / "real" / "sub_moved")
    (tmp_path / "real" / "sub").symlink_to(tmp_path / "real" / "sub_moved")
    with pytest.raises(SourceReadError, match="symlink"):
        checked_source_stat(tmp_path, Path("real/sub/f.py"), vetted_directories=vetted)
