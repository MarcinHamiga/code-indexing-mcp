"""Byte-exact unified-diff rendering for deterministic refactor edits.

A pure bytes -> patch-text leaf: the emission pipeline hands it the on-disk
bytes and verified edit spans, and it returns reviewable patch text a caller
can apply with `git apply`. It knows nothing about the resolver, the store,
or what the edits mean, so its tests run on inline fixtures alone.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import Final, NamedTuple

# difflib renders a missing final terminator as nothing, but `git apply`
# needs the marker to know the resulting file does not end in a newline.
_NO_NEWLINE_MARKER: Final = b"\\ No newline at end of file\n"


class ByteEdit(NamedTuple):
    """One replacement of the half-open byte span [start, end) in one file."""

    start: int
    end: int
    replacement: bytes


def apply_edits(original: bytes, edits: Sequence[ByteEdit]) -> bytes:
    """Splice sorted, non-overlapping byte spans into ``original``.

    Edits are applied in offset order regardless of the order they arrive
    in, so a caller can hand findings straight from the resolver. Two spans
    that overlap mean a resolver regression rather than a mergeable state,
    so this raises instead of guessing an order.
    """
    parts: list[bytes] = []
    cursor = 0
    for edit in sorted(edits):
        if edit.end < edit.start:
            raise ValueError(f"edit span [{edit.start}, {edit.end}) is negative")
        if edit.start < cursor:
            raise ValueError(f"edit span [{edit.start}, {edit.end}) overlaps the previous edit")
        parts.append(original[cursor : edit.start])
        parts.append(edit.replacement)
        cursor = edit.end
    parts.append(original[cursor:])
    return b"".join(parts)


def _format_range(start: int, stop: int) -> str:
    """Render one 0-based half-open line range in ``start,count`` form."""
    beginning = start + 1
    length = stop - start
    if length == 1:
        return str(beginning)
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


def _append_lines(out: list[bytes], prefix: bytes, lines: Sequence[bytes]) -> None:
    """Prefix each line and mark any that does not end in a newline.

    The marker follows git's own output: it describes the file's missing
    final terminator, so it follows context lines as well as -/+ lines.
    """
    for line in lines:
        if line.endswith(b"\n"):
            out.append(prefix + line)
            continue
        # The patch line needs its own terminator even when the file content
        # lacks one; the marker then records that fact for `git apply`.
        out.append(prefix + line + b"\n")
        out.append(_NO_NEWLINE_MARKER)


def render_unified_diff(
    path: str,
    original: bytes,
    edited: bytes,
    context_lines: int = 3,
) -> str | None:
    """Render ``original`` -> ``edited`` as one file's unified diff.

    Works purely on bytes, so CRLF terminators and a leading byte-order mark
    survive untouched. Returns None when the byte strings are identical, so
    callers render one diff per changed file and concatenate the results.
    """
    if original == edited:
        return None
    original_lines = original.splitlines(keepends=True)
    edited_lines = edited.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=original_lines, b=edited_lines, autojunk=False)
    out: list[bytes] = [
        f"diff --git a/{path} b/{path}\n".encode(),
        f"--- a/{path}\n".encode(),
        f"+++ b/{path}\n".encode(),
    ]
    for group in matcher.get_grouped_opcodes(context_lines):
        first, last = group[0], group[-1]
        out.append(
            (
                f"@@ -{_format_range(first[1], last[2])} +{_format_range(first[3], last[4])} @@\n"
            ).encode()
        )
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                _append_lines(out, b" ", original_lines[i1:i2])
                continue
            if tag in ("replace", "delete"):
                _append_lines(out, b"-", original_lines[i1:i2])
            if tag in ("replace", "insert"):
                _append_lines(out, b"+", edited_lines[j1:j2])
    return b"".join(out).decode("utf-8", errors="replace")
