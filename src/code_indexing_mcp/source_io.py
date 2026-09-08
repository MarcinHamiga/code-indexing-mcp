"""Bounded source access anchored to a trusted checkout directory."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from . import windows_source_io


class SourceReadError(OSError):
    """A source path failed a safety check before its bytes could be returned."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(f"{reason}: {path}")
        self.reason = reason


def checked_source_stat(root: Path, relative: Path) -> os.stat_result:
    """Reject every symlink component, including Windows junctions."""
    if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
        raise SourceReadError("path_escapes_root", relative)
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or current.is_junction():
            raise SourceReadError("symlink", relative)
        if index < len(relative.parts) - 1:
            if not stat.S_ISDIR(info.st_mode):
                raise SourceReadError("not_directory", relative)
        elif not stat.S_ISREG(info.st_mode):
            raise SourceReadError("not_regular", relative)
    return info


@contextmanager
def open_source(root: Path, relative: Path, max_bytes: int) -> Iterator[tuple[int, os.stat_result]]:
    """Open without following checkout symlinks; validate the descriptor before reading.

    POSIX traversal pins each directory with openat/O_NOFOLLOW. Windows opens
    reparse points themselves and pins directories against rename, then checks
    the final descriptor path. Other platforms without dir_fd fail closed.
    Nonblocking open prevents a substituted FIFO from waiting for a writer.
    """
    before = checked_source_stat(root, relative)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    directory: int | None = None
    descriptor: int | None = None
    try:
        if os.open in os.supports_dir_fd:
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            directory = os.open(root, directory_flags)
            for part in relative.parts[:-1]:
                child = os.open(part, directory_flags, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(relative.name, flags, dir_fd=directory)
        else:
            descriptor = windows_source_io.open_source(root, relative)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SourceReadError("not_regular", relative)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SourceReadError("changed_during_open", relative)
        if opened.st_size > max_bytes:
            raise SourceReadError("oversized", relative)
        yield descriptor, opened
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def read_source(root: Path, relative: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    """Read at most max_bytes + 1 and reject growth beyond the configured ceiling."""
    with open_source(root, relative, max_bytes) as (descriptor, info):
        content = bytearray()
        while len(content) <= max_bytes:
            block = os.read(descriptor, min(65536, max_bytes + 1 - len(content)))
            if not block:
                break
            content.extend(block)
        if len(content) > max_bytes:
            raise SourceReadError("oversized", relative)
        after = os.fstat(descriptor)
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise SourceReadError("changed_during_read", relative)
        return bytes(content), info
