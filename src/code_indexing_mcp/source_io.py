"""Bounded source access anchored to a trusted checkout directory."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import TracebackType

from . import windows_source_io

_CHECK_JUNCTIONS = os.name == "nt"
_SYMLINK_ERRNOS = frozenset(
    code for code in (getattr(errno, "ELOOP", None), getattr(errno, "EMLINK", None)) if code
)


class SourceReadError(OSError):
    """A source path failed a safety check before its bytes could be returned."""

    def __init__(self, reason: str, path: Path) -> None:
        super().__init__(f"{reason}: {path}")
        self.reason = reason


class PinnedDirectoryCache:
    """Reuse O_DIRECTORY | O_NOFOLLOW descriptors across sequential scan reads.

    Sorted scans visit sibling and child files together, so keeping the active
    ancestor chain pinned avoids reopening every directory component from
    *root* on each file. Descriptors outside the shared prefix are closed as
    soon as traversal moves elsewhere, and :meth:`close` releases the rest.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root_fd: int | None = None
        self._chain: list[tuple[str, int]] = []

    def __enter__(self) -> PinnedDirectoryCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open_parent(self, parts: tuple[str, ...], relative: Path) -> int:
        """Return an open descriptor for *parts* under *root* without following symlinks."""
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if self._root_fd is None:
            try:
                self._root_fd = os.open(self._root, directory_flags)
            except OSError as exc:
                if exc.errno in _SYMLINK_ERRNOS:
                    raise SourceReadError("symlink", relative) from exc
                raise
        common = 0
        limit = min(len(self._chain), len(parts))
        while common < limit and self._chain[common][0] == parts[common]:
            common += 1
        self._trim_to(common)
        parent_fd = self._chain[-1][1] if self._chain else self._root_fd
        for part in parts[common:]:
            try:
                child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            except OSError as exc:
                self.close()
                if exc.errno in _SYMLINK_ERRNOS:
                    raise SourceReadError("symlink", relative) from exc
                if exc.errno == errno.ENOTDIR:
                    raise SourceReadError("not_directory", relative) from exc
                raise
            self._chain.append((part, child_fd))
            parent_fd = child_fd
        return parent_fd

    def _trim_to(self, depth: int) -> None:
        while len(self._chain) > depth:
            _, descriptor = self._chain.pop()
            with suppress(OSError):
                os.close(descriptor)

    def close(self) -> None:
        """Close every pinned directory descriptor."""
        self._trim_to(0)
        if self._root_fd is not None:
            root_fd = self._root_fd
            self._root_fd = None
            with suppress(OSError):
                os.close(root_fd)


def checked_source_stat(
    root: Path,
    relative: Path,
    *,
    vetted_directories: set[Path] | None = None,
    leaf_path: Path | None = None,
) -> os.stat_result:
    """Reject every symlink component, including Windows junctions.

    *vetted_directories* lets one bulk walk (a scan) remember ancestor
    directories, given relative to *root*, that already passed the symlink and
    directory checks, so a directory shared by many files is lstat'd once
    rather than once per file. The final component is always lstat'd. Standalone
    reads never pass the set: ``open_source`` re-vets the whole chain unless
    paired with a scan's fresh stat and pinned directory cache.
    """
    if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
        raise SourceReadError("path_escapes_root", relative)
    parts = relative.parts
    last = len(parts) - 1
    if last == 0 or (vetted_directories is not None and relative.parent in vetted_directories):
        leaf = leaf_path if leaf_path is not None else root / relative
        info = leaf.lstat()
        if stat.S_ISLNK(info.st_mode) or (_CHECK_JUNCTIONS and leaf.is_junction()):
            raise SourceReadError("symlink", relative)
        if not stat.S_ISREG(info.st_mode):
            raise SourceReadError("not_regular", relative)
        return info
    start = 0
    current = root
    if vetted_directories is not None and last > 0:
        # Resume below the deepest ancestor already vetted.
        for depth in range(last, 0, -1):
            if relative.parents[last - depth] in vetted_directories:
                start = depth
                break
        if start:
            current = root.joinpath(*parts[:start])
    for index in range(start, len(parts)):
        current = current / parts[index]
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or (_CHECK_JUNCTIONS and current.is_junction()):
            raise SourceReadError("symlink", relative)
        if index < last:
            if not stat.S_ISDIR(info.st_mode):
                raise SourceReadError("not_directory", relative)
            if vetted_directories is not None:
                vetted_directories.add(relative.parents[last - index - 1])
        elif not stat.S_ISREG(info.st_mode):
            raise SourceReadError("not_regular", relative)
    return info


@contextmanager
def open_source(
    root: Path,
    relative: Path,
    max_bytes: int,
    *,
    expected_stat: os.stat_result | None = None,
    directory_cache: PinnedDirectoryCache | None = None,
) -> Iterator[tuple[int, os.stat_result]]:
    """Open without following checkout symlinks; validate the descriptor before reading.

    POSIX traversal pins each directory with openat/O_NOFOLLOW. Windows opens
    reparse points themselves and pins directories against rename, then checks
    the final descriptor path. Other platforms without dir_fd fail closed.
    Nonblocking open prevents a substituted FIFO from waiting for a writer.
    """
    if expected_stat is None:
        before = checked_source_stat(root, relative)
    else:
        if relative.is_absolute() or relative.drive or not relative.parts or ".." in relative.parts:
            raise SourceReadError("path_escapes_root", relative)
        before = expected_stat
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
            if directory_cache is not None:
                parent_fd = directory_cache.open_parent(relative.parts[:-1], relative)
            else:
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                directory = os.open(root, directory_flags)
                for part in relative.parts[:-1]:
                    child = os.open(part, directory_flags, dir_fd=directory)
                    os.close(directory)
                    directory = child
                parent_fd = directory
            try:
                descriptor = os.open(relative.name, flags, dir_fd=parent_fd)
            except OSError as exc:
                if directory_cache is not None:
                    directory_cache.close()
                if exc.errno in _SYMLINK_ERRNOS:
                    raise SourceReadError("symlink", relative) from exc
                raise
        else:
            descriptor = windows_source_io.open_source(root, relative)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SourceReadError("not_regular", relative)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            if directory_cache is not None:
                directory_cache.close()
            raise SourceReadError("changed_during_open", relative)
        if opened.st_size > max_bytes:
            raise SourceReadError("oversized", relative)
        yield descriptor, opened
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory is not None:
            os.close(directory)


def read_source(
    root: Path,
    relative: Path,
    max_bytes: int,
    *,
    expected_stat: os.stat_result | None = None,
    directory_cache: PinnedDirectoryCache | None = None,
) -> tuple[bytes, os.stat_result]:
    """Read at most max_bytes + 1 and reject growth beyond the configured ceiling."""
    with open_source(
        root,
        relative,
        max_bytes,
        expected_stat=expected_stat,
        directory_cache=directory_cache,
    ) as (descriptor, info):
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
