"""Throttled "an update is available" check for managed installations.

This runs on the serve path, so it is stdlib-only and deliberately cheap: the
local revision is read straight out of ``.git`` rather than through a
subprocess, the remote is contacted at most once a day, and every failure --
no network, no git, a corrupt cache -- is swallowed. An update check may never
fail the command that happened to trigger it.

It is also silent for anything that is not a managed install: a development
checkout has no ``code-indexing-mcp update`` to run, so it must never be
nagged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

CACHE_FILENAME = "update-check.json"
# Once a day: the checkout only ever moves when a human runs an update, and the
# notice is a convenience rather than a security signal.
CHECK_INTERVAL_SECONDS = 86400
# A hung network must not hold up an interactive command for longer than the
# command itself would plausibly take.
LS_REMOTE_TIMEOUT_SECONDS = 5.0
DISABLE_VARIABLE = "CODE_INDEXING_UPDATE_CHECK"
# Bumped whenever a stored record's meaning changes. Records written by another
# version are treated as absent rather than reinterpreted.
CACHE_SCHEMA_VERSION = 1

_DISABLED_VALUES = frozenset({"off", "0", "false", "no"})
_INSTALL_DIRECTORY_VARIABLE = "CODE_INDEXING_MCP_INSTALL_DIR"
_REMOTE_BRANCH_REF = "refs/heads/main"

# The subprocess seam, injectable so tests never need a network or a git.
_Runner = Callable[[list[str], Path, float], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class UpdateStatus:
    checked_at: float
    local_sha: str
    remote_sha: str

    @property
    def update_available(self) -> bool:
        return bool(self.local_sha) and bool(self.remote_sha) and self.local_sha != self.remote_sha


def install_context(*, environment: Mapping[str, str] | None = None) -> Path | None:
    """Return the managed install directory, or ``None`` when this is not one.

    ``None`` turns every other entry point here into a no-op. The interpreter
    must be running out of the install's own environment, which is tested
    through ``sys.prefix`` -- the virtualenv root -- rather than
    ``sys.executable``, which symlink-resolves to the base interpreter outside
    the checkout.
    """
    values = os.environ if environment is None else environment
    configured = values.get(_INSTALL_DIRECTORY_VARIABLE, "")
    try:
        # Kept in step with installer.orchestrator.default_install_directory,
        # which cannot be imported here: the installer stays off the serve path.
        if configured:
            directory = Path(configured).expanduser()
        else:
            directory = Path.home() / ".local" / "share" / "code-indexing-mcp"
        resolved = directory.resolve()
        # A worktree's ``.git`` is a file, so existence is the test, not is_dir.
        if not (resolved / ".git").exists():
            return None
        if not Path(sys.prefix).resolve().is_relative_to(resolved):
            return None
    except OSError:
        return None
    return resolved


def checkout_head(directory: Path) -> str | None:
    """Return the checked-out revision of *directory*, reading files first.

    ``git rev-parse`` is the last resort only: on the serve path this has to
    cost microseconds, and the plain-file layout covers every case a managed
    install can be in.
    """
    try:
        git_directory = _git_directory(directory)
        if git_directory is None:
            return None
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        reference = head[len("ref:") :].strip()
        sha = _reference_sha(git_directory, reference)
        if sha is not None:
            return sha
        return _rev_parse(directory)
    except (OSError, ValueError):
        return None


def read_cache(cache_directory: Path) -> UpdateStatus | None:
    """Return the cached status, treating anything unreadable as absent."""
    try:
        raw = json.loads((cache_directory / CACHE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    try:
        return UpdateStatus(
            checked_at=float(raw["checked_at"]),
            local_sha=str(raw["local_sha"]),
            remote_sha=str(raw["remote_sha"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_cache(cache_directory: Path, status: UpdateStatus) -> None:
    """Persist *status* atomically, so a reader never sees a partial file."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "checked_at": status.checked_at,
        "local_sha": status.local_sha,
        "remote_sha": status.remote_sha,
    }
    path = cache_directory / CACHE_FILENAME
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def check_remote(
    install_directory: Path,
    *,
    timeout: float = LS_REMOTE_TIMEOUT_SECONDS,
    run_command: _Runner | None = None,
) -> UpdateStatus:
    """Ask the remote for the tip of main. Raises when the check fails."""
    runner = _run_git if run_command is None else run_command
    # "origin" rather than a URL, so an install created with
    # CODE_INDEXING_MCP_REPO_URL keeps checking the remote it came from.
    completed = runner(
        ["git", "ls-remote", "origin", _REMOTE_BRANCH_REF], install_directory, timeout
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"no remote branch {_REMOTE_BRANCH_REF} at origin")
    return UpdateStatus(
        checked_at=time.time(),
        local_sha=checkout_head(install_directory) or "",
        remote_sha=lines[0].split()[0],
    )


def refresh_if_due(
    install_directory: Path,
    cache_directory: Path,
    *,
    now: float | None = None,
    run_command: _Runner | None = None,
) -> None:
    """Re-check the remote when the cached answer has aged out."""
    if _disabled():
        return
    moment = time.time() if now is None else now
    cached = read_cache(cache_directory)
    if cached is not None and moment - cached.checked_at < CHECK_INTERVAL_SECONDS:
        return
    try:
        write_cache(cache_directory, check_remote(install_directory, run_command=run_command))
    except Exception:
        # Missing git, no network, a timeout, an unwritable cache: none of them
        # are this caller's problem, and none may surface as its failure.
        return


def start_background_refresh(cache_directory: Path) -> threading.Thread | None:
    """Start the refresh off the hot path, or return ``None`` when it is moot.

    The throttle is re-checked here on purpose: the point is to not spawn a
    thread at all on the overwhelmingly common already-checked-today path.
    """
    if _disabled():
        return None
    install_directory = install_context()
    if install_directory is None:
        return None
    cached = read_cache(cache_directory)
    if cached is not None and time.time() - cached.checked_at < CHECK_INTERVAL_SECONDS:
        return None
    thread = threading.Thread(
        target=refresh_if_due,
        args=(install_directory, cache_directory),
        daemon=True,
    )
    thread.start()
    return thread


def notice(cache_directory: Path) -> str | None:
    """Return the update message, or ``None`` when there is nothing to say.

    The cached remote is compared against the *live* head rather than the
    cached one, so an update applied by any means -- the update command, a
    re-run of install.sh, a manual pull -- silences this immediately.
    """
    cached = read_cache(cache_directory)
    if cached is None:
        return None
    install_directory = install_context()
    if install_directory is None:
        return None
    local = checkout_head(install_directory) or ""
    if not local or not cached.remote_sha or local == cached.remote_sha:
        return None
    return (
        f"A code-indexing-mcp update is available "
        f"({local[:7]} -> {cached.remote_sha[:7]}). Run: code-indexing-mcp update"
    )


def _disabled(environment: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environment is None else environment
    return values.get(DISABLE_VARIABLE, "").strip().lower() in _DISABLED_VALUES


def _git_directory(directory: Path) -> Path | None:
    candidate = directory / ".git"
    if candidate.is_dir():
        return candidate
    if not candidate.is_file():
        return None
    content = candidate.read_text(encoding="utf-8").strip()
    if not content.startswith("gitdir:"):
        return None
    target = Path(content[len("gitdir:") :].strip())
    return target if target.is_absolute() else directory / target


def _reference_sha(git_directory: Path, reference: str) -> str | None:
    try:
        return (git_directory / reference).read_text(encoding="utf-8").strip() or None
    except OSError:
        pass
    try:
        packed = (git_directory / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed.splitlines():
        if line.startswith(("#", "^")):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[1] == reference:
            return parts[0]
    return None


def _rev_parse(directory: Path) -> str | None:
    try:
        completed = _run_git(["git", "rev-parse", "HEAD"], directory, LS_REMOTE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _run_git(command: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
