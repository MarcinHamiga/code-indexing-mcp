"""Self-update for a managed installation, in two phases.

Phase one runs from the environment it is about to rewrite: it fast-forwards
the checkout, re-syncs the locked environment, and then hands off to a child
process that starts from the new files on disk. Phase two (``--finalize``)
reconciles the accelerator environment, stops the daemon, re-runs the
post-install checks, and reports what moved.

Every import in this module is at the top for that reason: the module is
imported before the pull, and once the files on disk have changed an import
resolved later would load new code into a process running the old code.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from filelock import FileLock, Timeout

from .. import update_check
from ..application import RuntimePaths
from ..daemon import BrokerApplication, daemon_status, daemon_supported
from .accelerator import (
    ACCELERATOR_EXTRAS,
    _run_command,
    _uv_executable,
    accelerator_lock_fingerprint,
    accelerator_record_path,
    configure_accelerator,
    environment_python,
    prepared_accelerator,
    server_executable,
)
from .config_files import InstallerError
from .harnesses import configuration_path
from .orchestrator import default_install_directory
from .verify import format_check, run_update_checks
from .wizard import load_prefill

# Kept in lockstep with install.py, which must stay self-contained: it is
# downloaded and run on its own, before this package exists on the machine.
DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git"
SERVING_EXTRAS = ("cpu", "tui")

UPDATE_BRANCH = "main"
REMOTE_BRANCH_REF = "refs/heads/main"
UPDATE_LOCK_NAME = ".update.lock"
# Suffix for a Windows launcher renamed out of the way of `uv sync`.
STALE_SUFFIX = ".stale-"
LOG_LINE_LIMIT = 15
DIRTY_PATH_LIMIT = 5
LS_REMOTE_TIMEOUT_SECONDS = 10.0
# Distinct from 1 (error) and 2 (CodeIndexingError), so a script can tell
# "an update is available" from "the check itself failed".
CHECK_UPDATE_AVAILABLE_EXIT = 10

RunCommand = Callable[..., "subprocess.CompletedProcess[str]"]
Spawn = Callable[[list[str], Path], int]


class _AcceleratorOutcome(NamedTuple):
    status: str  # "ok" | "skipped" | "warning"
    detail: str
    rebuilt: bool
    prepared: str | None


def _canonical_repository_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        return f"github.com/{value.removeprefix('git@github.com:').lower()}"
    for prefix in ("https://github.com/", "http://github.com/", "ssh://git@github.com/"):
        if value.startswith(prefix):
            return f"github.com/{value.removeprefix(prefix).lower()}"
    if "://" not in value:
        return str(Path(value).expanduser().resolve())
    return value


def _error(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _print_status(step: str, status: str, detail: str) -> None:
    stream = sys.stderr if status == "warning" else sys.stdout
    print(f"[{step}] {status}: {detail}", file=stream)


def _git(run_command: RunCommand, directory: Path, *arguments: str) -> str:
    return run_command(["git", *arguments], cwd=directory).stdout.strip()


def _fast_forward_possible(directory: Path) -> bool:
    """Ask git whether HEAD is an ancestor of what was just fetched.

    This one call answers through its exit status, so it cannot go through the
    command helper every other call uses -- that one raises on non-zero.
    """

    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _spawn_finalize(argv: list[str], cwd: Path) -> int:
    # subprocess rather than os.execv: exec is unavailable on Windows in any
    # form that survives a running console, and it would forfeit the chance to
    # report a launch failure at all.
    try:
        return subprocess.run(argv, cwd=cwd).returncode
    except OSError as exc:
        return _error(f"could not start the updated code to finish the update: {exc}")


def update_main(
    *,
    install_dir: str | None,
    check: bool,
    skip_accelerator: bool,
    finalize: bool,
    previous_sha: str | None,
    run_command: RunCommand = _run_command,
    uv_executable: str | None = None,
    platform_name: str | None = None,
    spawn: Spawn = _spawn_finalize,
) -> int:
    """Entry for ``code-indexing-mcp update`` and its two hidden sub-modes."""

    directory = (
        Path(install_dir).expanduser().resolve()
        if install_dir
        else default_install_directory().resolve()
    )
    if finalize:
        return _finalize_main(
            directory,
            previous_sha=previous_sha,
            skip_accelerator=skip_accelerator,
            run_command=run_command,
        )
    if check:
        return _check_main(directory, run_command=run_command)
    return _phase1(
        directory,
        skip_accelerator=skip_accelerator,
        run_command=run_command,
        uv_executable=uv_executable,
        platform_name=platform_name,
        spawn=spawn,
    )


def _preflight(
    directory: Path,
    *,
    run_command: RunCommand,
    uv_executable: str | None,
) -> tuple[str, str, str]:
    """Return ``(uv, head, remote)``, or raise before anything has been touched."""

    if shutil.which("git") is None:
        raise InstallerError("git is required to update but was not found in PATH")
    if not (directory / ".git").exists():
        raise InstallerError(
            f"{directory} is not a git checkout; reinstall with install.sh to update it "
            "with this command"
        )
    try:
        origin = _git(run_command, directory, "remote", "get-url", "origin")
    except InstallerError as exc:
        raise InstallerError(f"{directory} has no origin remote to update from: {exc}") from exc
    expected = os.environ.get("CODE_INDEXING_MCP_REPO_URL", DEFAULT_REPOSITORY_URL)
    if _canonical_repository_url(origin) != _canonical_repository_url(expected):
        raise InstallerError(
            f"the checkout at {directory} tracks {origin}, not {expected}; update it manually "
            "or point CODE_INDEXING_MCP_REPO_URL at the remote it came from"
        )
    branch = _git(run_command, directory, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != UPDATE_BRANCH:
        raise InstallerError(
            f"the checkout at {directory} is on branch {branch}, not {UPDATE_BRANCH}; "
            f"switch it back with `git switch {UPDATE_BRANCH}` before updating"
        )
    # Tracked changes only: the environment, its caches, and this command's own
    # lock file all live untracked inside the checkout, and none of them is a
    # reason to refuse a fast-forward.
    status = _git(run_command, directory, "status", "--porcelain", "--untracked-files=no")
    dirty = status.splitlines()
    if dirty:
        names = ", ".join(line.strip().split(maxsplit=1)[-1] for line in dirty[:DIRTY_PATH_LIMIT])
        if len(dirty) > DIRTY_PATH_LIMIT:
            names = f"{names} and {len(dirty) - DIRTY_PATH_LIMIT} more"
        raise InstallerError(
            f"the checkout at {directory} has uncommitted changes ({names}); commit, stash, "
            "or discard them before updating"
        )
    # Resolved before the pull on purpose: a checkout left ahead of an
    # environment that cannot be synced is the one state this refuses to create.
    uv = _uv_executable(uv_executable)
    try:
        run_command(["git", "fetch", "origin", UPDATE_BRANCH], cwd=directory)
    except InstallerError as exc:
        raise InstallerError(
            f"could not fetch origin/{UPDATE_BRANCH}; check the network connection and try "
            f"again: {exc}"
        ) from exc
    head = _git(run_command, directory, "rev-parse", "HEAD")
    remote = _git(run_command, directory, "rev-parse", "FETCH_HEAD")
    if head != remote and not _fast_forward_possible(directory):
        raise InstallerError(
            f"the checkout at {directory} has diverged from origin/{UPDATE_BRANCH}; update it "
            "manually or reinstall"
        )
    return uv, head, remote


def _phase1(
    directory: Path,
    *,
    skip_accelerator: bool,
    run_command: RunCommand,
    uv_executable: str | None,
    platform_name: str | None,
    spawn: Spawn,
) -> int:
    if not server_executable(directory, platform_name=platform_name).is_file():
        return _error(f"no installation found at {directory}")
    lock = FileLock(directory / UPDATE_LOCK_NAME)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return _error(f"another update is already running in {directory}; wait for it to finish")
    try:
        _discard_stale_scripts(directory, platform_name=platform_name)
        try:
            uv, head, remote = _preflight(
                directory, run_command=run_command, uv_executable=uv_executable
            )
        except InstallerError as exc:
            return _error(str(exc))
        if head == remote:
            # Still synced and finalized below: an interrupted update leaves the
            # checkout current and everything after it undone.
            print("Already up to date.")
        else:
            try:
                run_command(["git", "merge", "--ff-only", "FETCH_HEAD"], cwd=directory)
            except InstallerError as exc:
                return _error(f"could not fast-forward the checkout at {directory}: {exc}")
        shielded = _shield_running_script(directory, platform_name=platform_name)
        extras = [flag for extra in SERVING_EXTRAS for flag in ("--extra", extra)]
        try:
            run_command([uv, "sync", "--locked", *extras], cwd=directory)
        except InstallerError as exc:
            _unshield(shielded, directory, platform_name=platform_name)
            return _error(
                f"{exc}\nthe checkout was updated but its environment was not; re-run "
                "`code-indexing-mcp update`"
            )
        _unshield(shielded, directory, platform_name=platform_name)
        python = environment_python(directory / ".venv", platform_name=platform_name)
        argv = [
            str(python),
            "-m",
            "code_indexing_mcp",
            "update",
            "--finalize",
            "--previous-sha",
            head,
        ]
        if skip_accelerator:
            argv.append("--skip-accelerator")
        argv += ["--install-dir", str(directory)]
        return spawn(argv, directory)
    finally:
        lock.release()


def _shield_running_script(directory: Path, *, platform_name: str | None = None) -> Path | None:
    """Rename the running launcher aside so ``uv sync`` can replace it.

    Windows refuses to overwrite a mapped image but allows renaming one, which
    is what makes an update that runs through this very launcher possible.
    """

    platform_name = platform_name or sys.platform
    if not platform_name.startswith("win"):
        return None
    executable = server_executable(directory, platform_name=platform_name)
    if not executable.is_file():
        return None
    stale = executable.with_name(f"{executable.name}{STALE_SUFFIX}{os.getpid()}")
    try:
        executable.rename(stale)
    except OSError:
        return None
    return stale


def _unshield(shielded: Path | None, directory: Path, *, platform_name: str | None = None) -> None:
    """Put the launcher back, unless uv wrote a new one over the gap."""

    if shielded is None:
        return
    executable = server_executable(directory, platform_name=platform_name)
    if not executable.exists():
        with contextlib.suppress(OSError):
            shielded.rename(executable)
        return
    # An unlink that fails leaves a copy this process still has mapped; the
    # next update collects it.
    with contextlib.suppress(OSError):
        shielded.unlink()


def _discard_stale_scripts(directory: Path, *, platform_name: str | None = None) -> None:
    platform_name = platform_name or sys.platform
    if not platform_name.startswith("win"):
        return
    scripts = directory / ".venv" / "Scripts"
    try:
        leftovers = list(scripts.glob(f"*{STALE_SUFFIX}*"))
    except OSError:
        return
    for path in leftovers:
        try:
            path.unlink()
        except OSError:
            continue


def _recorded_lock_fingerprint(directory: Path) -> str:
    try:
        record = json.loads(accelerator_record_path(directory).read_text(encoding="utf-8"))
    except (OSError, ValueError, InstallerError):
        return ""
    if not isinstance(record, dict):
        return ""
    return str(record.get("lock_fingerprint", ""))


def _reconcile_accelerator(directory: Path, *, skip_accelerator: bool) -> _AcceleratorOutcome:
    """Rebuild the accelerator environment only when the pull invalidated it."""

    prepared = prepared_accelerator(directory)
    if prepared is None:
        # An update never detects: an installation that is on CPU stays on CPU
        # until someone asks for something else.
        return _AcceleratorOutcome("skipped", "no accelerator environment is recorded", False, None)
    extra = ACCELERATOR_EXTRAS.get(prepared)
    if extra is None:
        return _AcceleratorOutcome(
            "skipped", f"the recorded {prepared} runtime needs no separate environment", False, None
        )
    try:
        expected = accelerator_lock_fingerprint(directory, extra)
    except InstallerError as exc:
        return _AcceleratorOutcome(
            "warning", f"the {prepared} environment could not be checked: {exc}", False, prepared
        )
    if _recorded_lock_fingerprint(directory) == expected:
        # Matching fingerprints answer the question without a single detection
        # subprocess, the slowest of which can sit on nvidia-smi for ~30s.
        return _AcceleratorOutcome(
            "skipped",
            f"the {prepared} environment is unchanged since the last build",
            False,
            prepared,
        )
    if skip_accelerator:
        return _AcceleratorOutcome(
            "warning",
            f"the {prepared} environment was resolved from an older lockfile and keeps serving "
            f"as it is, because the server does not re-check it; rebuild it with "
            f"`code-indexing-mcp configure --accelerator {prepared}`",
            False,
            prepared,
        )
    try:
        plan = configure_accelerator(directory, prepared)
    except InstallerError as exc:
        return _AcceleratorOutcome(
            "warning", f"the {prepared} environment could not be rebuilt: {exc}", False, prepared
        )
    # A rebuild that fell back has already cleared the record and degraded to
    # CPU, which costs speed and nothing else -- a warning, not a failed update.
    return _AcceleratorOutcome(
        "ok" if plan.honored else "warning",
        f"{plan.accelerator} ({plan.reason})",
        True,
        plan.accelerator if plan.prepares_environment else None,
    )


def _wait_until_stopped(
    paths: RuntimePaths, *, attempts: int = 100, interval: float = 0.05
) -> bool:
    for _ in range(attempts):
        if not daemon_status(paths)["running"]:
            return True
        time.sleep(interval)
    return False


def _stop_daemon(paths: RuntimePaths, *, changed: bool) -> tuple[str, str]:
    """Stop the daemon; the next client respawns it on the updated code."""

    if not changed:
        return "skipped", "nothing changed, so the running daemon is already current"
    if not daemon_supported():
        return "skipped", "this platform has no shared daemon"
    try:
        if not daemon_status(paths)["running"]:
            return "skipped", "no daemon is running"
        BrokerApplication(paths).stop()
        if not _wait_until_stopped(paths):
            return "warning", "the daemon did not stop; run `code-indexing-mcp daemon stop`"
    except Exception as exc:
        return "warning", f"the daemon could not be stopped: {exc}"
    return "ok", "stopped; it restarts on the updated code with the next client"


def _print_summary(
    directory: Path,
    *,
    previous_sha: str | None,
    head: str,
    accelerator: _AcceleratorOutcome,
    run_command: RunCommand,
) -> None:
    if previous_sha and previous_sha != head:
        print(f"Updated {previous_sha[:7]} -> {head[:7]}")
        try:
            lines = _git(
                run_command,
                directory,
                "log",
                "--oneline",
                "--no-decorate",
                f"{previous_sha}..HEAD",
            ).splitlines()
        except InstallerError:
            lines = []
        for line in lines[:LOG_LINE_LIMIT]:
            print(f"  {line}")
        if len(lines) > LOG_LINE_LIMIT:
            print(f"  ... and {len(lines) - LOG_LINE_LIMIT} more commits")
    else:
        print(f"Already at {head[:7]}; the environment was re-synced and re-checked.")
    print(f"Accelerator: {accelerator.detail}")
    print("Restart your MCP clients to load the updated server.")


def _finalize_main(
    directory: Path,
    *,
    previous_sha: str | None,
    skip_accelerator: bool,
    run_command: RunCommand,
) -> int:
    head = update_check.checkout_head(directory) or ""
    accelerator = _reconcile_accelerator(directory, skip_accelerator=skip_accelerator)
    _print_status("accelerator", accelerator.status, accelerator.detail)

    paths = RuntimePaths.from_environment()
    changed = accelerator.rebuilt or (previous_sha or "") != head
    status, detail = _stop_daemon(paths, changed=changed)
    _print_status("daemon", status, detail)

    configured = [(slug, configuration_path(slug)) for slug in load_prefill().configured_slugs]
    checks = run_update_checks(
        directory, configured, accelerator_was_prepared=accelerator.prepared is not None
    )
    for check in checks:
        print(format_check(check), file=sys.stdout if check.ok else sys.stderr)

    # Silencing the notifier is a courtesy; failing to is not a failed update.
    with contextlib.suppress(OSError):
        update_check.write_cache(
            paths.cache,
            update_check.UpdateStatus(checked_at=time.time(), local_sha=head, remote_sha=head),
        )

    _print_summary(
        directory,
        previous_sha=previous_sha,
        head=head,
        accelerator=accelerator,
        run_command=run_command,
    )
    unusable = [
        check for check in checks if check.status == "fail" and check.name == "server executable"
    ]
    return 1 if unusable else 0


def _check_main(directory: Path, *, run_command: RunCommand) -> int:
    if not (directory / ".git").exists():
        return _error(f"{directory} is not a git checkout; there is no update to check for")
    try:
        origin = _git(run_command, directory, "remote", "get-url", "origin")
    except InstallerError as exc:
        return _error(f"{directory} has no origin remote to check against: {exc}")
    expected = os.environ.get("CODE_INDEXING_MCP_REPO_URL", DEFAULT_REPOSITORY_URL)
    if _canonical_repository_url(origin) != _canonical_repository_url(expected):
        return _error(f"the checkout at {directory} tracks {origin}, not {expected}")
    try:
        completed = subprocess.run(
            ["git", "ls-remote", "origin", REMOTE_BRANCH_REF],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=LS_REMOTE_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _error(f"could not reach origin to check for updates: {exc}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return _error(f"origin has no {REMOTE_BRANCH_REF} to compare against")
    status = update_check.UpdateStatus(
        checked_at=time.time(),
        local_sha=update_check.checkout_head(directory) or "",
        remote_sha=lines[0].split()[0],
    )
    print(
        json.dumps(
            {
                "install_dir": str(directory),
                "local_sha": status.local_sha,
                "remote_sha": status.remote_sha,
                "update_available": status.update_available,
            },
            indent=2,
            sort_keys=True,
        )
    )
    with contextlib.suppress(OSError):
        update_check.write_cache(RuntimePaths.from_environment().cache, status)
    return CHECK_UPDATE_AVAILABLE_EXIT if status.update_available else 0
