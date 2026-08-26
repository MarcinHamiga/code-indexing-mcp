"""Typed, read-only Git state for a registered project root.

The models here answer one question -- "which physical index slot does this
checkout currently map to?" -- without touching storage. Everything is derived
from read-only Git queries so a probe can run on every application entrypoint
before the index is opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

from .models import FrozenModel

GIT_TIMEOUT_SECONDS = 5.0
SLOT_KEY_VERSION = "git-slot-v1"
_SLOT_PARTITION_PREFIX = "slot-"
_SLOT_PARTITION_HEX_CHARS = 32


class GitProbeOutcome(StrEnum):
    """Result of consulting Git about a registered root."""

    GIT = "git"
    NOT_GIT = "not_git"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID = "invalid"


class SelectorKind(StrEnum):
    """What a slot's selector value is anchored to."""

    REF = "ref"
    COMMIT = "commit"
    WORKSPACE = "workspace"


class WorktreeStatus(StrEnum):
    CLEAN = "clean"
    TRACKED_DIRTY = "tracked_dirty"
    UNTRACKED = "untracked"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class GitRunnerError(Exception):
    """Git could not be consulted at all."""


class GitUnavailable(GitRunnerError):
    """The Git binary is missing or cannot be executed."""


class GitTimeout(GitRunnerError):
    """Git did not answer within the bounded timeout."""


class GitCommandResult(FrozenModel):
    returncode: int
    stdout: str
    stderr: str = ""


GitRunner = Callable[[Sequence[str], Path], GitCommandResult]


def run_git(command: Sequence[str], cwd: Path) -> GitCommandResult:
    """Run one Git command without a shell and with bounded patience.

    ``GIT_OPTIONAL_LOCKS=0`` keeps even ``git status`` from taking the
    repository's index lock: a probe must never disturb the working checkout.
    """
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitTimeout(f"git did not finish within {GIT_TIMEOUT_SECONDS}s") from exc
    except OSError as exc:
        raise GitUnavailable(f"git could not be executed: {exc}") from exc
    return GitCommandResult(
        returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
    )


class GitState(FrozenModel):
    """Immutable snapshot of one registered root's Git state."""

    probe: GitProbeOutcome
    selector_kind: SelectorKind
    selector_value: str
    # Optional because an unborn branch has no commit object yet, and because
    # a degraded probe never guesses at a HEAD it could not read.
    head_oid: str | None = None
    repository_identity: str | None = None
    checkout_identity: str | None = None
    toplevel: str | None = None
    project_prefix: str = ""
    worktree: WorktreeStatus = WorktreeStatus.UNKNOWN
    dirty_paths: tuple[str, ...] = ()
    untracked_paths: tuple[str, ...] = ()
    status_fingerprint: str | None = None


def probe_git_state(
    root: Path,
    *,
    include_status: bool = False,
    runner: GitRunner | None = None,
) -> GitState:
    """Probe *root*'s checkout identity, HEAD selector, and optional dirtiness.

    The identity and selector queries are cheap rev-parse-class calls so query
    entrypoints can run this on every request; ``git status`` is spawned only
    when ``include_status`` asks for cleanliness or dirty paths.
    """
    run = run_git if runner is None else runner
    resolved = root.resolve()
    try:
        identities = run(
            ["git", "rev-parse", "--git-common-dir", "--git-dir", "--show-toplevel"], root
        )
    except GitTimeout:
        return _fallback_state(root, GitProbeOutcome.TIMEOUT)
    except GitRunnerError:
        return _fallback_state(root, GitProbeOutcome.UNAVAILABLE)
    if identities.returncode != 0:
        return _fallback_state(root, GitProbeOutcome.NOT_GIT)
    lines = [line.strip() for line in identities.stdout.splitlines()]
    if len(lines) != 3 or not all(lines):
        return _fallback_state(root, GitProbeOutcome.INVALID)
    common = _resolve_query_directory(lines[0], root)
    git_dir = _resolve_query_directory(lines[1], root)
    toplevel = Path(lines[2])
    if not toplevel.is_absolute():
        return _fallback_state(root, GitProbeOutcome.INVALID)
    try:
        project_prefix = _project_prefix(resolved, toplevel.resolve())
    except ValueError:
        return _fallback_state(root, GitProbeOutcome.INVALID)

    try:
        symbolic = run(["git", "symbolic-ref", "-q", "HEAD"], root)
        head = run(["git", "rev-parse", "HEAD"], root)
    except GitTimeout:
        return _fallback_state(root, GitProbeOutcome.TIMEOUT)
    except GitRunnerError:
        return _fallback_state(root, GitProbeOutcome.UNAVAILABLE)

    head_oid = head.stdout.strip() if head.returncode == 0 else None
    if head_oid is not None and not _looks_like_oid(head_oid):
        return _fallback_state(root, GitProbeOutcome.INVALID)
    if symbolic.returncode == 0:
        ref = symbolic.stdout.strip()
        if not ref.startswith("refs/"):
            return _fallback_state(root, GitProbeOutcome.INVALID)
        selector_kind, selector_value = SelectorKind.REF, ref
    elif head_oid is not None:
        selector_kind, selector_value = SelectorKind.COMMIT, head_oid
    else:
        # Neither a symbolic ref nor a resolvable object: HEAD's metadata is
        # damaged rather than merely unborn.
        return _fallback_state(root, GitProbeOutcome.INVALID)

    worktree = WorktreeStatus.UNKNOWN
    dirty: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()
    fingerprint: str | None = None
    if include_status:
        try:
            status = run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], root)
        except GitRunnerError:
            status = None
        if status is not None and status.returncode == 0:
            worktree, dirty, untracked, fingerprint = _parse_status(status.stdout, project_prefix)

    return GitState(
        probe=GitProbeOutcome.GIT,
        selector_kind=selector_kind,
        selector_value=selector_value,
        head_oid=head_oid,
        repository_identity=str(common),
        checkout_identity=str(git_dir),
        toplevel=str(toplevel.resolve()),
        project_prefix=project_prefix,
        worktree=worktree,
        dirty_paths=dirty,
        untracked_paths=untracked,
        status_fingerprint=fingerprint,
    )


def slot_key(project_id: str, state: GitState) -> tuple[str, str, str, str, str, str, str]:
    """Return the identity tuple of the physical index slot for *state*.

    Deliberately excludes the mutable properties of one checkout: the HEAD
    OID, dirty state, dirty paths, scan configuration, model, and schema
    version. A branch therefore keeps one slot across commits and local
    edits, while a rename, a detached OID, or a different checkout selects a
    different slot.
    """
    return (
        SLOT_KEY_VERSION,
        project_id,
        state.repository_identity or "",
        state.checkout_identity or "",
        state.project_prefix,
        state.selector_kind.value,
        state.selector_value,
    )


def slot_id(project_id: str, state: GitState) -> str:
    """Return the opaque slot identifier derived from the slot key."""
    payload = json.dumps(slot_key(project_id, state), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def partition_id(slot: str) -> str:
    """Return the physical partition directory name for a slot identifier.

    The name is opaque and path-safe; a branch name never reaches the
    filesystem.
    """
    return f"{_SLOT_PARTITION_PREFIX}{slot[:_SLOT_PARTITION_HEX_CHARS]}"


def changed_paths_between(
    root: Path, old_oid: str, new_oid: str, *, project_prefix: str = ""
) -> frozenset[str] | None:
    """Project-relative tracked paths whose content differs between two commits.

    Runs ``git diff --name-only -z <old> <new> -- .`` from the registered root,
    so a project registered inside a subdirectory only sees its own subtree.
    Git prints repository-root-relative paths; those are re-rooted onto the
    project exactly like status output. Returns ``None`` whenever the diff
    cannot be computed -- a missing repository, an unreachable object after a
    history rewrite, any nonzero exit -- so the caller can fall back to
    validating every path instead of trusting a partial answer.
    """
    if not old_oid or not new_oid or old_oid == new_oid:
        return frozenset()
    try:
        result = run_git(
            ("git", "diff", "--name-only", "-z", old_oid, new_oid, "--", "."), cwd=root
        )
    except GitRunnerError:
        return None
    if result.returncode != 0:
        return None
    changed: set[str] = set()
    for path in result.stdout.split("\0"):
        if not path:
            continue
        relative = _project_relative_path(path, project_prefix)
        if relative is not None:
            changed.add(relative)
    return frozenset(changed)


def _fallback_state(root: Path, outcome: GitProbeOutcome) -> GitState:
    """Route a non-Git or degraded probe to the checkout-local workspace slot.

    A transient Git failure must never claim a branch identity it could not
    read, so every degraded outcome shares the same workspace selector as a
    plain non-Git directory.
    """
    resolved = str(root.resolve())
    return GitState(
        probe=outcome,
        selector_kind=SelectorKind.WORKSPACE,
        selector_value=resolved,
        toplevel=resolved if outcome is GitProbeOutcome.NOT_GIT else None,
    )


def _resolve_query_directory(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        # --git-common-dir and --git-dir are relative to the query directory,
        # which is the registered root, not the repository toplevel.
        path = root / path
    return path.resolve()


def _project_prefix(root: Path, toplevel: Path) -> str:
    relative = os.path.relpath(root, toplevel)
    parts = Path(relative).parts
    if relative == ".":
        return ""
    if parts and parts[0] == "..":
        raise ValueError(f"{root} is outside the worktree {toplevel}")
    return Path(relative).as_posix()


def _looks_like_oid(value: str) -> bool:
    return len(value) in (40, 64) and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _parse_status(
    stdout: str, project_prefix: str
) -> tuple[WorktreeStatus, tuple[str, ...], tuple[str, ...], str]:
    fields = stdout.split("\0")
    records: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if len(field) < 4 or field[2] != " ":
            continue
        code, path = field[:2], field[3:]
        # In -z format a rename or copy record is followed by a separate
        # NUL-terminated field carrying the source path; that field is not a
        # record of its own.
        if code[0] in "RC":
            index += 1
        relative = _project_relative_path(path, project_prefix)
        if relative is not None:
            records.append((code, relative))
    tracked = tuple(path for code, path in records if code != "??")
    untracked = tuple(path for code, path in records if code == "??")
    if tracked and untracked:
        worktree = WorktreeStatus.MIXED
    elif tracked:
        worktree = WorktreeStatus.TRACKED_DIRTY
    elif untracked:
        worktree = WorktreeStatus.UNTRACKED
    else:
        worktree = WorktreeStatus.CLEAN
    digest = hashlib.sha256()
    for code, path in sorted(records):
        digest.update(f"{code} {path}".encode())
        digest.update(b"\0")
    return worktree, tracked, untracked, digest.hexdigest()


def _project_relative_path(path: str, project_prefix: str) -> str | None:
    if not project_prefix:
        return path
    marker = f"{project_prefix}/"
    return path[len(marker) :] if path.startswith(marker) else None
