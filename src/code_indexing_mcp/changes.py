"""What changed in a checkout since a base revision, as files and line ranges.

Backs ``changed_symbols``: Git reports which files differ from the base and
which current-file lines each diff hunk covers; the index then names the
declarations those lines fall in. Every Git call runs from the registered
project root with ``--relative``, so a project registered in a subdirectory
sees only its own subtree and every path comes back project-relative.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .errors import CodeIndexingError, ErrorCode
from .git_state import GitRunner, GitRunnerError, run_git
from .models import FileChangeKind, LineRange, OutlineItem

# Pinned so user configuration cannot change what the parsers below read:
# prefixes, quoting, colour, external diff drivers, and textconv filters.
_DIFF_OPTIONS = (
    "--no-renames",
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--relative",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)
_GIT = ("git", "-c", "core.quotePath=false")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_STATUS_KINDS: dict[str, FileChangeKind] = {
    "A": "added",
    "D": "deleted",
    "M": "modified",
    "T": "modified",
}


@dataclass(frozen=True)
class FileChange:
    """One changed file: its kind and, for a textual diff, the touched lines.

    ``ranges`` are inclusive current-file line ranges that were added or
    rewritten. ``deletion_points`` are current-file lines after which lines
    were removed. ``whole_file`` means every declaration counts as touched.
    """

    path: str
    change: FileChangeKind
    ranges: tuple[tuple[int, int], ...] = ()
    deletion_points: tuple[int, ...] = ()
    whole_file: bool = False

    def line_ranges(self) -> list[LineRange]:
        return [LineRange(start_line=start, end_line=end) for start, end in self.ranges]

    def touched(self, items: Sequence[OutlineItem]) -> list[OutlineItem]:
        """The declarations whose line span meets a change."""
        if self.change == "deleted":
            return []
        if self.whole_file:
            return list(items)
        return [
            item
            for item in items
            if any(start <= item.end_line and item.start_line <= end for start, end in self.ranges)
            # Removed lines sit between current lines `point` and `point + 1`;
            # only a declaration spanning both contained them.
            or any(item.start_line <= point < item.end_line for point in self.deletion_points)
        ]


@dataclass
class _Hunks:
    ranges: list[tuple[int, int]] = field(default_factory=list)
    deletion_points: list[int] = field(default_factory=list)
    binary: bool = False


@dataclass(frozen=True)
class ChangeSet:
    base: str
    files: list[FileChange]
    total_files: int


def resolve_base(
    root: Path,
    *,
    since: str | None,
    since_time: str | None,
    runner: GitRunner | None = None,
) -> str:
    """Resolve the base revision to a commit OID.

    ``since`` names any commit-ish (default ``HEAD``: only uncommitted work);
    ``since_time`` picks the last commit on ``HEAD`` made before that time.
    """
    run = run_git if runner is None else runner
    if since is not None and since_time is not None:
        raise CodeIndexingError(
            ErrorCode.INVALID_FILTER, "Pass either since or since_time, not both"
        )
    if since_time is not None:
        if not since_time.strip():
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, "since_time must not be empty")
        result = _run(run, ["rev-list", "-1", f"--before={since_time}", "HEAD"], root)
        oid = result.strip() if result is not None else ""
        if not oid:
            raise CodeIndexingError(
                ErrorCode.INVALID_FILTER,
                f"No commit on HEAD is older than {since_time!r}",
                since_time=since_time,
            )
        return oid
    revision = since or "HEAD"
    # A leading dash would be read as an option, not a revision.
    if not revision.strip() or revision.startswith("-"):
        raise CodeIndexingError(
            ErrorCode.INVALID_FILTER, f"Invalid revision {revision!r}", since=revision
        )
    result = _run(run, ["rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"], root)
    if result is None or not result.strip():
        raise CodeIndexingError(
            ErrorCode.INVALID_FILTER,
            f"{revision!r} does not name a commit in this repository",
            since=revision,
        )
    return result.strip()


def collect_changes(
    root: Path,
    base: str,
    *,
    include_untracked: bool,
    limit: int,
    runner: GitRunner | None = None,
) -> ChangeSet:
    """List files that differ from *base* in the working tree, first *limit* by path.

    Only the returned files are diffed line by line, so a base far in the past
    costs one name listing plus a bounded patch.
    """
    run = run_git if runner is None else runner
    listing = _run(run, ["diff", *_DIFF_OPTIONS, "--name-status", "-z", base, "--"], root)
    if listing is None:
        raise CodeIndexingError(
            ErrorCode.INTERNAL_ERROR, f"git diff against {base} failed", base=base
        )
    kinds = parse_name_status(listing)
    if include_untracked:
        others = _run(run, ["ls-files", "--others", "--exclude-standard", "-z"], root)
        for path in (others or "").split("\0"):
            if path and path not in kinds:
                kinds[path] = "untracked"
    ordered = sorted(kinds)
    selected = ordered[:limit]
    diffed = [path for path in selected if kinds[path] in {"modified", "added"}]
    hunks: dict[str, _Hunks] = {}
    if diffed:
        patch = _run(
            run,
            [
                "diff",
                *_DIFF_OPTIONS,
                "--unified=0",
                base,
                "--",
                *(f":(literal){path}" for path in diffed),
            ],
            root,
        )
        hunks = parse_unified_zero(patch or "")
    files: list[FileChange] = []
    for path in selected:
        kind = kinds[path]
        found = hunks.get(path)
        if kind in {"added", "untracked"} or found is None or found.binary:
            # No usable line ranges: an added file is new throughout, and a
            # modified file whose patch could not be read is treated likewise
            # rather than reported as touching nothing.
            files.append(FileChange(path=path, change=kind, whole_file=kind != "deleted"))
            continue
        files.append(
            FileChange(
                path=path,
                change=kind,
                ranges=tuple(found.ranges),
                deletion_points=tuple(found.deletion_points),
            )
        )
    return ChangeSet(base=base, files=files, total_files=len(ordered))


def parse_name_status(output: str) -> dict[str, FileChangeKind]:
    """Parse ``git diff --name-status -z`` (renames disabled) into path kinds."""
    fields = output.split("\0")
    kinds: dict[str, FileChangeKind] = {}
    index = 0
    while index + 1 < len(fields):
        status, path = fields[index], fields[index + 1]
        index += 2
        if not status or not path:
            continue
        kinds[path] = _STATUS_KINDS.get(status[0], "modified")
    return kinds


def parse_unified_zero(patch: str) -> dict[str, _Hunks]:
    """Collect each file's new-side hunk ranges from a ``--unified=0`` patch."""
    files: dict[str, _Hunks] = {}
    current: _Hunks | None = None
    old_path: str | None = None
    in_header = False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            # A section whose header names its file is recorded even without
            # hunks: a mode-only change touches no lines.
            in_header, old_path = True, None
            path = _header_path(line[len("diff --git ") :])
            current = files.setdefault(path, _Hunks()) if path else None
            continue
        if in_header:
            # Only the section header carries ---/+++ lines; inside a hunk a
            # removed "-- comment" line reads "--- comment" too.
            if line.startswith("--- "):
                old_path = _patch_path(line[4:], "a/")
            elif line.startswith("+++ "):
                path = _patch_path(line[4:], "b/") or old_path
                if path:
                    current = files.setdefault(path, _Hunks())
            elif line.startswith("Binary files ") and line.endswith(" differ"):
                path = _binary_path(line)
                if path:
                    current = files.setdefault(path, _Hunks())
                if current is not None:
                    current.binary = True
            elif line.startswith("@@ "):
                in_header = False
        if not in_header and current is not None and (match := _HUNK.match(line)):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            if count:
                current.ranges.append((start, start + count - 1))
            else:
                current.deletion_points.append(start)
    return files


def _header_path(value: str) -> str | None:
    """The path in an unquoted ``a/P b/P`` header; renames are disabled, so both match."""
    if value.startswith('"'):
        return None
    length, remainder = divmod(len(value) - 5, 2)
    if length <= 0 or remainder:
        return None
    old, new = value[: length + 2], value[length + 3 :]
    if not (old.startswith("a/") and new.startswith("b/") and old[2:] == new[2:]):
        return None
    return old[2:]


def _patch_path(value: str, prefix: str) -> str | None:
    # Git appends a tab to a ---/+++ path that contains a space.
    value = value.removesuffix("\t")
    if value == "/dev/null":
        return None
    value = _unquote(value)
    return value[len(prefix) :] if value.startswith(prefix) else None


def _binary_path(line: str) -> str | None:
    # "Binary files a/x and b/x differ" -- the new side names the file unless
    # the file was deleted.
    body = line[len("Binary files ") : -len(" differ")]
    old, separator, new = body.partition(" and ")
    if not separator:
        return None
    return _patch_path(new, "b/") or _patch_path(old, "a/")


def _unquote(value: str) -> str:
    """Undo Git's C-style quoting of a path with special characters."""
    if len(value) < 2 or not (value.startswith('"') and value.endswith('"')):
        return value
    # With core.quotePath=false non-ASCII stays raw; only escapes need decoding.
    # typeshed types escape_decode's result as str; it is bytes for bytes input.
    raw = cast(bytes, codecs.escape_decode(value[1:-1].encode("utf-8"))[0])
    return raw.decode("utf-8", errors="replace")


def _run(run: GitRunner, arguments: list[str], root: Path) -> str | None:
    try:
        result = run([*_GIT, *arguments], root)
    except GitRunnerError:
        return None
    return result.stdout if result.returncode == 0 else None
