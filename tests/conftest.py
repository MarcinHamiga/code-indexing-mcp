"""Helpers shared by the test suite."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def case_insensitive_path_alias() -> Callable[[Path], Path]:
    """Return a differently-cased alias, or skip when the filesystem is case-sensitive."""

    def alias(path: Path) -> Path:
        candidate = path.with_name(path.name.swapcase())
        if candidate.name == path.name or not candidate.exists() or not candidate.samefile(path):
            pytest.skip("test requires a case-insensitive filesystem")
        return candidate

    return alias


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
