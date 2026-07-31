#!/usr/bin/env python3
"""Bootstrap installer for Code Indexing MCP.

This file is stdlib-only and self-contained: install.sh downloads it into a
temporary directory and runs it before any virtual environment exists. It
clones or updates the repository, builds the locked environment, and delegates
everything else to ``python -m incode_mcp.installer`` inside that environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git"
# The serving environment always gets the CPU extra (it is the fallback every
# accelerator degrades to) plus the TUI extra for the interactive wizard.
SERVING_EXTRAS = ("cpu", "tui")
ACCELERATOR_CHOICES = ("auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml")


class InstallerError(RuntimeError):
    """An actionable installer failure."""


def _run_command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=None if environment is None else {**os.environ, **environment},
        )
    except FileNotFoundError as exc:
        raise InstallerError(f"Required command was not found: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        command = " ".join(arguments)
        message = f"Command failed: {command}"
        if detail:
            message = f"{message}\n{detail}"
        raise InstallerError(message) from exc


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


def clone_or_update_repository(repository_url: str, install_directory: Path) -> str:
    """Clone a fresh checkout or fast-forward an existing clean checkout."""

    git = shutil.which("git")
    if git is None:
        raise InstallerError("Git is required but was not found in PATH")
    install_directory = install_directory.expanduser().resolve()

    if not install_directory.exists():
        install_directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _run_command([git, "clone", "--", repository_url, str(install_directory)])
        return "installed"

    if not (install_directory / ".git").exists():
        raise InstallerError(
            f"Install target exists but is not a Git repository: {install_directory}"
        )

    origin = _run_command(
        [git, "remote", "get-url", "origin"],
        cwd=install_directory,
    ).stdout.strip()
    if _canonical_repository_url(origin) != _canonical_repository_url(repository_url):
        raise InstallerError(
            "Existing checkout origin does not match the requested repository: "
            f"{origin} != {repository_url}"
        )

    status = _run_command(
        [git, "status", "--porcelain"],
        cwd=install_directory,
    ).stdout
    if status.strip():
        raise InstallerError(
            f"Existing checkout has uncommitted changes; update it manually: {install_directory}"
        )

    _run_command([git, "pull", "--ff-only"], cwd=install_directory)
    return "updated"


def server_executable(
    install_directory: Path,
    *,
    platform_name: str | None = None,
) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return install_directory / ".venv" / "Scripts" / "code-indexing-mcp.exe"
    return install_directory / ".venv" / "bin" / "code-indexing-mcp"


def environment_python(directory: Path, *, platform_name: str | None = None) -> Path:
    """Return the interpreter inside a virtual environment directory."""

    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _uv_executable(uv_executable: str | None) -> str:
    uv = uv_executable or shutil.which("uv")
    if uv is None:
        raise InstallerError(
            "uv is required but was not found in PATH. Install it from https://docs.astral.sh/uv/"
        )
    return uv


def sync_environment(
    install_directory: Path,
    *,
    uv_executable: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Create or refresh the locked virtual environment and return its server command."""

    uv = _uv_executable(uv_executable)
    command = [uv, "sync", "--locked"]
    for extra in SERVING_EXTRAS:
        command += ["--extra", extra]
    _run_command(command, cwd=install_directory)
    command_path = server_executable(install_directory, platform_name=platform_name)
    if not command_path.is_file():
        raise InstallerError(f"uv sync completed but the MCP executable is missing: {command_path}")
    return command_path


def _default_install_directory() -> Path:
    configured = os.environ.get("CODE_INDEXING_MCP_INSTALL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "code-indexing-mcp"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install or update Code Indexing MCP and configure it for selected MCP harnesses."
        )
    )
    parser.add_argument(
        "--install-dir",
        default=str(_default_install_directory()),
        help="checkout location (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("CODE_INDEXING_MCP_REPO_URL", DEFAULT_REPOSITORY_URL),
        help="Git repository to clone or update (default: %(default)s)",
    )
    parser.add_argument(
        "--accelerator",
        choices=ACCELERATOR_CHOICES,
        default=os.environ.get("CODE_INDEXING_MCP_ACCELERATOR", "auto"),
        help=(
            "which accelerator to prepare for passage indexing (default: %(default)s). "
            "auto detects one; anything that cannot be detected, built, or probed "
            "falls back to CPU with the reason reported"
        ),
    )
    parser.add_argument(
        "--harnesses",
        help=(
            "comma-separated harness numbers/slugs or 'all'; omit for the interactive menu "
            "(codex, claude-code, kimi-code, claude-desktop, opencode, kilocode)"
        ),
    )
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set a managed INCODE_* value in harness configs; repeatable",
    )
    parser.add_argument(
        "--unset",
        dest="unsets",
        action="append",
        default=[],
        metavar="NAME",
        help="remove a managed INCODE_* value from harness configs; repeatable",
    )
    parser.add_argument("--tui", action="store_true", help="force the interactive wizard")
    parser.add_argument("--no-tui", action="store_true", help="force the plain text interface")
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="never prompt; a missing harness selection configures none",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("INCODE_OFFLINE", "").lower() in {"1", "true", "yes"},
        help="never download the embedding model",
    )
    return parser


def tui_available() -> bool:
    """True when the terminal can host the Textual wizard."""

    term = os.environ.get("TERM", "")
    return sys.stdin.isatty() and sys.stdout.isatty() and bool(term) and term != "dumb"


def _delegate(install_directory: Path, tail: list[str]) -> int:
    python = environment_python(install_directory / ".venv")
    try:
        completed = subprocess.run(
            [str(python), "-m", "incode_mcp.installer", *tail],
            cwd=install_directory,
        )
    except OSError as exc:
        print(f"Error: could not launch the installer module: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    install_directory = Path(arguments.install_dir).expanduser().resolve()

    try:
        action = clone_or_update_repository(arguments.repo_url, install_directory)
        print(f"{action.title()} repository: {install_directory}")
        command = sync_environment(install_directory)
        print(f"Prepared MCP executable: {command}")
    except InstallerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Installation cancelled.", file=sys.stderr)
        return 130

    tail = ["--install-dir", str(install_directory), "--accelerator", arguments.accelerator]
    if arguments.harnesses is not None:
        tail += ["--harnesses", arguments.harnesses]
    for pair in arguments.settings:
        tail += ["--set", pair]
    for name in arguments.unsets:
        tail += ["--unset", name]
    if arguments.offline:
        tail.append("--offline")
    if arguments.no_prompt:
        tail.append("--no-prompt")
    use_tui = arguments.tui or (not arguments.no_tui and tui_available())
    if use_tui:
        tail.append("--tui")

    returncode = _delegate(install_directory, tail)
    if use_tui and returncode not in (0, 1, 130):
        print(
            "The interactive installer failed; re-run with --no-tui for the plain interface.",
            file=sys.stderr,
        )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
