"""Source access rejects unsafe live paths before reading their contents."""

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from code_indexing_mcp.errors import CodeIndexingError
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.source_io import (
    PinnedDirectoryCache,
    SourceReadError,
    checked_source_stat,
    read_source,
)


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


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="POSIX dir_fd cache")
def test_pinned_directory_cache_reuses_ancestor_descriptors_and_closes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "one.py").write_bytes(b"one = 1\n")
    (tmp_path / "a" / "b" / "two.py").write_bytes(b"two = 2\n")
    (tmp_path / "a" / "c" / "three.py").write_bytes(b"three = 3\n")

    real_open = os.open
    real_close = os.close
    dir_opens: list[str] = []
    live_fds: set[int] = set()

    def tracking_open(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        fd = (
            real_open(path, flags, mode)
            if dir_fd is None
            else real_open(path, flags, mode, dir_fd=dir_fd)
        )
        live_fds.add(fd)
        if flags & getattr(os, "O_DIRECTORY", 0):
            dir_opens.append(Path(path).name if path != tmp_path else "<root>")
        return fd

    def tracking_close(fd: int) -> None:
        live_fds.discard(fd)
        real_close(fd)

    monkeypatch.setattr("code_indexing_mcp.source_io.os.open", tracking_open)
    monkeypatch.setattr("code_indexing_mcp.source_io.os.close", tracking_close)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, tracking_open})

    vetted: set[Path] = set()
    with PinnedDirectoryCache(tmp_path) as cache:
        for rel_str, expected in (
            ("a/b/one.py", b"one = 1\n"),
            ("a/b/two.py", b"two = 2\n"),
            ("a/c/three.py", b"three = 3\n"),
        ):
            rel = Path(rel_str)
            st = checked_source_stat(tmp_path, rel, vetted_directories=vetted)
            data, _ = read_source(tmp_path, rel, 1024, expected_stat=st, directory_cache=cache)
            assert data == expected

    # <root>, a, b opened once for one.py + two.py; only c opened when switching to a/c/three.py.
    assert dir_opens == ["<root>", "a", "b", "c"]
    assert live_fds == set()


@pytest.mark.skipif(
    os.name == "nt" or os.open not in os.supports_dir_fd,
    reason="POSIX symlink + dir_fd race check",
)
def test_pinned_directory_cache_rejects_mid_scan_directory_symlink_swap(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "one.py").write_bytes(b"inside_one = 1\n")
    (root / "sub" / "two.py").write_bytes(b"inside_two = 2\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "two.py").write_bytes(b"OUTSIDE_SECRET = 99\n")

    vetted: set[Path] = set()
    with PinnedDirectoryCache(root) as cache:
        st_one = checked_source_stat(root, Path("sub/one.py"), vetted_directories=vetted)
        data_one, _ = read_source(
            root, Path("sub/one.py"), 1024, expected_stat=st_one, directory_cache=cache
        )
        assert data_one == b"inside_one = 1\n"

        # Swap `sub` for a symlink to `outside` while `sub` is already vetted and pinned.
        (root / "sub").rename(tmp_path / "sub_moved")
        (root / "sub").symlink_to(outside, target_is_directory=True)

        # The vetted set is not re-checked, so the stat follows the new link...
        st_two = checked_source_stat(root, Path("sub/two.py"), vetted_directories=vetted)
        assert st_two.st_ino == (outside / "two.py").stat().st_ino
        # ...but the read does not: the pinned descriptor yields a different
        # inode, and the fresh chain walked on retry refuses the symlink.
        with pytest.raises(SourceReadError, match="symlink") as raised:
            read_source(root, Path("sub/two.py"), 1024, expected_stat=st_two, directory_cache=cache)
        assert raised.value.reason == "symlink"
        assert not cache.pinned


@pytest.mark.skipif(
    os.name == "nt" or os.open not in os.supports_dir_fd,
    reason="POSIX symlink + dir_fd race check",
)
def test_pinned_directory_cache_reports_unpinned_symlinked_directory_as_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "one.py").write_bytes(b"inside_one = 1\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "one.py").write_bytes(b"OUTSIDE_SECRET = 99\n")

    vetted: set[Path] = set()
    with PinnedDirectoryCache(root) as cache:
        # Vetted by a stat that never read, so `sub` is trusted but not pinned.
        checked_source_stat(root, Path("sub/one.py"), vetted_directories=vetted)
        (root / "sub").rename(tmp_path / "sub_moved")
        (root / "sub").symlink_to(outside, target_is_directory=True)

        st = checked_source_stat(root, Path("sub/one.py"), vetted_directories=vetted)
        # O_DIRECTORY | O_NOFOLLOW answers ENOTDIR here; the reason must still say symlink.
        with pytest.raises(SourceReadError) as raised:
            read_source(root, Path("sub/one.py"), 1024, expected_stat=st, directory_cache=cache)
        assert raised.value.reason == "symlink"

    (tmp_path / "plain").write_bytes(b"not a directory\n")
    with PinnedDirectoryCache(tmp_path) as cache:
        with pytest.raises(SourceReadError) as raised:
            cache.open_parent(("plain",), Path("plain/x.py"))
        assert raised.value.reason == "not_directory"


@pytest.mark.skipif(os.open not in os.supports_dir_fd, reason="POSIX dir_fd cache")
@pytest.mark.parametrize("replace", ["delete", "rename"])
def test_pinned_directory_cache_recovers_when_a_pinned_directory_is_recreated(
    tmp_path: Path, replace: str
) -> None:
    root = tmp_path / "repo"
    (root / "gen").mkdir(parents=True)
    (root / "gen" / "one.py").write_bytes(b"one = 1\n")
    (root / "gen" / "two.py").write_bytes(b"two = 1\n")

    vetted: set[Path] = set()
    with PinnedDirectoryCache(root) as cache:
        st_one = checked_source_stat(root, Path("gen/one.py"), vetted_directories=vetted)
        read_source(root, Path("gen/one.py"), 1024, expected_stat=st_one, directory_cache=cache)

        # A generator or checkout replaces the pinned directory between two reads.
        if replace == "delete":
            shutil.rmtree(root / "gen")
        else:
            (root / "gen").rename(tmp_path / "gen_old")
        (root / "gen").mkdir()
        (root / "gen" / "two.py").write_bytes(b"two = 2\n")

        st_two = checked_source_stat(root, Path("gen/two.py"), vetted_directories=vetted)
        data, _ = read_source(
            root, Path("gen/two.py"), 1024, expected_stat=st_two, directory_cache=cache
        )
        assert data == b"two = 2\n"

        # A file that is genuinely gone still fails, and only after one retry.
        (root / "gen" / "two.py").unlink()
        with pytest.raises(FileNotFoundError):
            read_source(root, Path("gen/two.py"), 1024, expected_stat=st_two, directory_cache=cache)
