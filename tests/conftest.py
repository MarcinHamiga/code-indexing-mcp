"""Helpers shared by the installer test modules."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_git(*arguments: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def create_test_remote(tmp_path: Path, name: str = "remote") -> tuple[Path, Path]:
    remote = tmp_path / f"{name}.git"
    publisher = tmp_path / f"{name}-publisher"
    run_git("init", "--bare", "--initial-branch=main", str(remote))
    run_git("init", "--initial-branch=main", str(publisher))
    run_git("config", "user.name", "Installer Tests", cwd=publisher)
    run_git("config", "user.email", "installer@example.test", cwd=publisher)
    (publisher / "version.txt").write_text("one\n")
    run_git("add", "version.txt", cwd=publisher)
    run_git("commit", "-m", "initial", cwd=publisher)
    run_git("remote", "add", "origin", str(remote), cwd=publisher)
    run_git("push", "-u", "origin", "main", cwd=publisher)
    return remote, publisher
