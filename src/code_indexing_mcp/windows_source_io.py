"""Windows source handles and directory pins for safe path-based mutations."""

from __future__ import annotations

import ctypes
import errno
import importlib
import ntpath
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import Any, cast


class _AttributeTagInfo(ctypes.Structure):
    _fields_ = [("attributes", ctypes.c_uint32), ("reparse_tag", ctypes.c_uint32)]


def _kernel32() -> Any:
    if os.name != "nt":
        raise OSError(errno.ENOTSUP, "safe Windows filesystem access unavailable")
    platform_ctypes: Any = ctypes
    api = platform_ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    api.CreateFileW.restype = wintypes.HANDLE
    api.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    api.GetFileInformationByHandleEx.restype = wintypes.BOOL
    api.GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    api.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    api.GetFileType.argtypes = [wintypes.HANDLE]
    api.GetFileType.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    return api


def _windows_error() -> OSError:
    platform_ctypes: Any = ctypes
    return cast(OSError, platform_ctypes.WinError())


def _normalized_path(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return ntpath.normcase(ntpath.normpath(value))


def _validate_handle_path(api: Any, handle: int, expected: Path) -> None:
    capacity = 512
    while capacity <= 32768:
        buffer = ctypes.create_unicode_buffer(capacity)
        length = api.GetFinalPathNameByHandleW(handle, buffer, capacity, 0)
        if not length:
            raise _windows_error()
        if length < capacity:
            if _normalized_path(buffer.value) != _normalized_path(str(expected.absolute())):
                raise OSError(errno.EPERM, "source path changed during open", str(expected))
            return
        capacity = length + 1
    raise OSError(errno.ENAMETOOLONG, "source handle path exceeds Windows limit", str(expected))


def _open_handle(path: Path, *, directory: bool) -> int:
    api = _kernel32()
    # OPEN_REPARSE_POINT opens the entry itself. Excluding FILE_SHARE_DELETE
    # prevents rename/replacement while a directory or source handle is held.
    handle = api.CreateFileW(
        str(path),
        0x80 if directory else 0x80000000,
        0x1 | 0x2,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise _windows_error()
    try:
        if api.GetFileType(handle) != 1:  # FILE_TYPE_DISK
            raise OSError(errno.EPERM, "source is not a disk file", str(path))
        info = _AttributeTagInfo()
        if not api.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise _windows_error()
        if info.attributes & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise OSError(errno.EPERM, "source is a reparse point", str(path))
        if bool(info.attributes & 0x10) != directory:
            raise OSError(errno.EPERM, "source has an unexpected file type", str(path))
        _validate_handle_path(api, handle, path)
    except BaseException:
        api.CloseHandle(handle)
        raise
    return cast(int, handle)


def _close_handle(handle: int) -> None:
    _kernel32().CloseHandle(handle)


@contextmanager
def pinned_directory(root: Path, relative: Path) -> Iterator[None]:
    """Hold every parent against replacement until path-based work completes."""
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise OSError(errno.EPERM, "directory path escapes root", str(relative))
    with ExitStack() as stack:
        current = root
        for part in (None, *relative.parts):
            if part is not None:
                current = current / part
            handle = _open_handle(current, directory=True)
            stack.callback(_close_handle, handle)
        yield


def open_source(root: Path, relative: Path) -> int:
    """Return a binary descriptor whose pinned path and final handle were checked."""
    with pinned_directory(root, relative.parent):
        handle = _open_handle(root / relative, directory=False)
        try:
            msvcrt = importlib.import_module("msvcrt")
            descriptor = cast(
                int, msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            )
        except BaseException:
            _close_handle(handle)
            raise
        # The descriptor now owns the handle. Validate that exact descriptor
        # before releasing directory pins or permitting any source read.
        try:
            _validate_handle_path(_kernel32(), msvcrt.get_osfhandle(descriptor), root / relative)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
