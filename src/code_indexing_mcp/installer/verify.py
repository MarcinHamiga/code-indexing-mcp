"""Post-install checks: proof the installation works, not just that files were written.

Every step before this one reports what it *did*. These checks report what is
now *true* -- the executable runs, the launcher resolves, each harness config
still parses and names a command that exists. A wizard that says "complete" and
leaves a client silently unable to start the server is the failure mode worth
spending a few seconds to rule out.

Nothing here can fail an installation. A check that does not pass is a warning
attached to a result that already happened.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import accelerator, harnesses, shell_path
from .env_blocks import command_from_entry

# Long enough for a cold interpreter start on a loaded machine, short enough
# that a hung executable does not hold the wizard open.
HELP_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _server_runs(install_directory: Path) -> Check:
    executable = accelerator.server_executable(install_directory)
    if not executable.is_file():
        return Check("server executable", "fail", f"missing: {executable}")
    try:
        completed = subprocess.run(
            [str(executable), "--help"],
            capture_output=True,
            timeout=HELP_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError as exc:
        return Check("server executable", "fail", f"{executable} could not be launched: {exc}")
    except subprocess.TimeoutExpired:
        return Check(
            "server executable",
            "fail",
            f"{executable} did not answer --help within {HELP_TIMEOUT_SECONDS:.0f}s",
        )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        return Check(
            "server executable",
            "fail",
            f"{executable} --help exited {completed.returncode}: {message[-1] if message else ''}",
        )
    return Check("server executable", "ok", str(executable))


def _launcher_resolves(
    launcher: shell_path.LauncherResult | None,
    profiles_updated: Sequence[Path],
    *,
    environment: Mapping[str, str] | None = None,
) -> Check:
    environment = os.environ if environment is None else environment
    if launcher is None:
        return Check("command on PATH", "warn", "no launcher was requested")
    if not launcher.ok:
        return Check("command on PATH", "warn", f"launcher not created: {launcher.detail}")
    found = shutil.which(shell_path.LAUNCHER_NAME, path=environment.get("PATH", ""))
    if found is not None and Path(found).resolve() == launcher.path.resolve():
        return Check("command on PATH", "ok", found)
    if profiles_updated:
        # The block was written into a profile this shell read before it existed.
        # That is the expected state right after an install, not a problem. The
        # hint has to come from the environment being verified, not this process:
        # they differ whenever the caller passed one in.
        hint = shell_path.activation_hint(profiles_updated, environment=environment)
        return Check("command on PATH", "warn", f"resolves once you start a new shell ({hint})")
    if found is not None:
        return Check("command on PATH", "warn", f"{found} is found first, not {launcher.path}")
    return Check(
        "command on PATH",
        "warn",
        f"{launcher.path.parent} is not on PATH; add it to use the command by name",
    )


def _harness_entries(
    configured: Sequence[tuple[str, Path]],
    install_directory: Path,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[Check]:
    """Re-read what was just written: a merge that parsed can still land wrong."""

    expected = accelerator.server_executable(install_directory)
    checks: list[Check] = []
    for slug, path in configured:
        name = f"{slug} configuration"
        try:
            entry = harnesses.read_server_entry(slug, home=home, environment=environment)
        except Exception as exc:  # a harness path this platform cannot even name
            checks.append(Check(name, "warn", f"could not be re-read: {exc}"))
            continue
        if entry is None:
            checks.append(Check(name, "warn", f"no server entry found in {path} after writing it"))
            continue
        command = command_from_entry(slug, entry)
        if command is None:
            checks.append(Check(name, "warn", f"the entry in {path} names no command"))
        elif Path(command) != expected:
            checks.append(Check(name, "warn", f"the entry names {command}, expected {expected}"))
        else:
            checks.append(Check(name, "ok", str(path)))
    return checks


def _accelerator_recorded(install_directory: Path) -> Check:
    prepared = accelerator.prepared_accelerator(install_directory)
    if prepared is None:
        return Check("accelerator record", "warn", "no record; the server will resolve at startup")
    return Check("accelerator record", "ok", prepared)


def _skill_links(
    slugs: Sequence[str],
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[Check]:
    checks: list[Check] = []
    for slug in slugs:
        directory = harnesses.skill_directory(slug, home=home, environment=environment)
        if directory is None or not directory.is_dir():
            continue
        # Only the links this installer made. A broken symlink the user put in
        # their own skill directory is not this installation's to report on.
        broken = [
            entry.name
            for entry in directory.iterdir()
            if harnesses.is_bundled_skill_link(entry) and not entry.resolve().is_dir()
        ]
        name = f"{slug} skills"
        if broken:
            checks.append(Check(name, "warn", f"links point nowhere: {', '.join(sorted(broken))}"))
        else:
            checks.append(Check(name, "ok", str(directory)))
    return checks


def run_checks(
    install_directory: Path,
    configured: Sequence[tuple[str, Path]] = (),
    *,
    launcher: shell_path.LauncherResult | None = None,
    profiles_updated: Sequence[Path] = (),
    accelerator_was_prepared: bool = True,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[Check, ...]:
    """Check the finished installation. Never raises; problems come back as warnings."""

    checks = [_server_runs(install_directory)]
    checks.append(_launcher_resolves(launcher, profiles_updated, environment=environment))
    checks.extend(
        _harness_entries(configured, install_directory, home=home, environment=environment)
    )
    if accelerator_was_prepared:
        checks.append(_accelerator_recorded(install_directory))
    checks.extend(
        _skill_links([slug for slug, _ in configured], home=home, environment=environment)
    )
    return tuple(checks)


def format_check(check: Check) -> str:
    """One display line per check.

    No square brackets: this string is handed to Rich-rendered widgets, which
    would read ``[ok]`` as a markup tag and swallow it.
    """

    marker = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}[check.status]
    line = f"{marker} - {check.name}"
    return f"{line}: {check.detail}" if check.detail else line
