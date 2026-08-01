"""Taking back exactly what the installer put down.

The rule throughout: remove only what this installer created, and recognise it
by evidence rather than by name. A file at the launcher's path that is not one
of our symlinks, a skill directory entry that is not one of our links, a PATH
block whose end marker the user deleted -- each stays where it is, and the
uninstaller says so instead of guessing.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import harnesses, shell_path
from .config_files import InstallerError
from .links import is_under
from .orchestrator import StepEvent, default_install_directory


@dataclass(frozen=True)
class UninstallPlan:
    install_directory: Path
    harness_slugs: tuple[str, ...] = ()
    bin_directory: Path | None = None
    remove_launcher: bool = True
    remove_path_block: bool = True
    # Off by default: indexes cost minutes of CPU to rebuild, and an uninstall
    # that silently discards them is not one the user can undo.
    remove_data: bool = False
    # Off by default: the checkout may be a working copy the user edits.
    remove_checkout: bool = False


@dataclass
class UninstallResult:
    harnesses_cleared: tuple[tuple[str, Path, bool], ...] = ()
    skills: tuple[tuple[str, str], ...] = ()
    launcher_removed: Path | None = None
    profiles_cleared: tuple[Path, ...] = ()
    directories_removed: tuple[Path, ...] = ()
    failures: list[tuple[str, str]] = field(default_factory=list)


def data_directories(
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """The index and cache directories, honouring the settings that move them."""

    from .settings_spec import BY_NAME, default_value

    environment = os.environ if environment is None else environment
    directories: list[Path] = []
    for name in ("CODE_INDEXING_DATA_DIR", "CODE_INDEXING_CACHE_DIR"):
        setting = BY_NAME.get(name)
        if setting is None:  # pragma: no cover - the catalog defines both
            continue
        configured = environment.get(name) or default_value(setting)
        if configured:
            directories.append(Path(configured).expanduser())
    return tuple(directories)


# Files and directories the server itself creates under its data or cache
# directory. One of these present is what distinguishes "our index lives here"
# from "the user pointed the setting at a directory that holds other things".
_DATA_MARKERS = (
    "lancedb",
    "locks",
    "staging",
    "accelerator.json",
    "daemon.token",
    "daemon.log",
    "models",
)


def _refuse_reason(directory: Path, *, checkout: bool, home: Path | None = None) -> str | None:
    """Why ``directory`` must not be deleted, or None when removing it is safe.

    ``--purge`` and ``--remove-checkout`` take their targets from a setting and a
    flag, either of which can name somewhere that is not ours. The confirmation
    prompt shows what will go, but a prompt is not a safety net for a recursive
    delete that cannot be undone: the directory has to look like ours as well.
    """

    home = home or Path.home()
    try:
        resolved = directory.resolve()
    except (OSError, ValueError) as exc:
        return f"cannot be resolved ({exc})"
    if resolved.parent == resolved:
        return "is a filesystem root"
    if resolved == home.resolve():
        return "is your home directory"
    if is_under(home.resolve(), resolved):
        return "contains your home directory"
    if checkout:
        if not (resolved / "pyproject.toml").is_file():
            return "does not look like a code-indexing-mcp checkout (no pyproject.toml)"
        if not (resolved / "src" / "code_indexing_mcp").is_dir():
            return "does not look like a code-indexing-mcp checkout (no src/code_indexing_mcp)"
        return None
    if resolved.name == "code-indexing-mcp":
        return None
    if any((resolved / marker).exists() for marker in _DATA_MARKERS):
        return None
    return "holds no code-indexing-mcp index or cache, so it is not ours to delete"


def run_uninstall(
    plan: UninstallPlan,
    on_event: Callable[[StepEvent], None] = lambda event: None,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> UninstallResult:
    """Undo an installation, step by step, reporting each removal.

    Every step isolates its own failures: a config file that cannot be written
    must not stop the launcher and the PATH entry from being taken back.
    """

    result = UninstallResult()

    on_event(StepEvent("harnesses", "started", ", ".join(plan.harness_slugs) or "none selected"))
    cleared, failures = harnesses.deconfigure_selected_harnesses(
        list(plan.harness_slugs), home=home, environment=environment
    )
    result.harnesses_cleared = tuple(cleared)
    result.failures.extend(failures)
    for slug, path, changed in cleared:
        on_event(
            StepEvent(
                "harnesses",
                "finished" if changed else "skipped",
                f"{slug}: {'removed from' if changed else 'no entry in'} {path}",
            )
        )
    for slug, message in failures:
        on_event(StepEvent("harnesses", "failed", f"{slug}: {message}"))

    on_event(StepEvent("skills", "started"))
    result.skills = tuple(
        harnesses.remove_skills(
            list(plan.harness_slugs),
            plan.install_directory,
            home=home,
            environment=environment,
        )
    )
    for slug, message in result.skills:
        on_event(StepEvent("skills", "finished", f"{slug}: {message}"))

    _remove_launcher(plan, result, on_event, home=home, environment=environment)
    _remove_directories(plan, result, on_event, home=home, environment=environment)
    return result


def _remove_launcher(
    plan: UninstallPlan,
    result: UninstallResult,
    on_event: Callable[[StepEvent], None],
    *,
    home: Path | None,
    environment: Mapping[str, str] | None,
) -> None:
    bin_directory = plan.bin_directory or shell_path.default_bin_directory(
        home=home, environment=environment
    )
    on_event(StepEvent("path", "started", str(bin_directory)))
    if plan.remove_launcher:
        try:
            removed = shell_path.remove_launcher(bin_directory, plan.install_directory)
        except OSError as exc:
            result.failures.append(("launcher", str(exc)))
            on_event(StepEvent("path", "failed", str(exc)))
        else:
            result.launcher_removed = removed
            on_event(
                StepEvent(
                    "path",
                    "finished" if removed else "skipped",
                    f"removed {removed}" if removed else f"no launcher of ours in {bin_directory}",
                )
            )
    if not plan.remove_path_block:
        return
    cleared: list[Path] = []
    for profile in shell_path.shell_profiles(home=home, environment=environment):
        try:
            if shell_path.remove_path_block(profile):
                cleared.append(profile)
                on_event(StepEvent("path", "finished", f"removed the PATH block from {profile}"))
        except OSError as exc:
            result.failures.append((str(profile), str(exc)))
            on_event(StepEvent("path", "failed", f"{profile}: {exc}"))
    result.profiles_cleared = tuple(cleared)


def _remove_directories(
    plan: UninstallPlan,
    result: UninstallResult,
    on_event: Callable[[StepEvent], None],
    *,
    home: Path | None,
    environment: Mapping[str, str] | None,
) -> None:
    targets: list[tuple[Path, bool]] = []
    if plan.remove_data:
        for directory in data_directories(environment=environment):
            targets.append((directory, False))
    if plan.remove_checkout:
        targets.append((plan.install_directory, True))
    if not targets:
        return
    on_event(StepEvent("directories", "started"))
    removed: list[Path] = []
    for directory, is_checkout in targets:
        if not directory.is_dir():
            on_event(StepEvent("directories", "skipped", f"{directory} does not exist"))
            continue
        refusal = _refuse_reason(directory, checkout=is_checkout, home=home)
        if refusal is not None:
            result.failures.append((str(directory), f"not removed: it {refusal}"))
            on_event(StepEvent("directories", "failed", f"{directory} {refusal}; left alone"))
            continue
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            result.failures.append((str(directory), str(exc)))
            on_event(StepEvent("directories", "failed", f"{directory}: {exc}"))
        else:
            removed.append(directory)
            on_event(StepEvent("directories", "finished", f"removed {directory}"))
    result.directories_removed = tuple(removed)


def resolve_slugs(
    selection: str | None,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Which harnesses to clear: the named ones, or every one still configured."""

    if selection is not None:
        return harnesses.parse_harness_selection(selection)
    from .wizard import load_prefill

    return list(load_prefill(home=home, environment=environment).configured_slugs)


def describe_plan(
    plan: UninstallPlan,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """What the uninstall will do, for the confirmation prompt."""

    lines: list[str] = []
    if plan.harness_slugs:
        lines.append("Remove the code-indexing-mcp entry from:")
        for slug in plan.harness_slugs:
            lines.append(
                f"  {slug}: "
                f"{harnesses.configuration_path(slug, home=home, environment=environment)}"
            )
        lines.append("Unlink the bundled skills from those harnesses.")
    else:
        lines.append("No configured harnesses to clear.")
    bin_directory = plan.bin_directory or shell_path.default_bin_directory(
        home=home, environment=environment
    )
    if plan.remove_launcher:
        lines.append(f"Remove the launcher at {shell_path.launcher_path(bin_directory)}.")
    if plan.remove_path_block:
        lines.append("Remove the PATH block from your shell profiles.")
    if plan.remove_data:
        for directory in data_directories(environment=environment):
            lines.append(f"DELETE the index/cache directory {directory}.")
    if plan.remove_checkout:
        lines.append(f"DELETE the installation checkout {plan.install_directory}.")
    return lines


def uninstall_main(
    *,
    install_dir: str | None = None,
    harnesses_selection: str | None = None,
    bin_dir: str | None = None,
    keep_launcher: bool = False,
    keep_path: bool = False,
    purge: bool = False,
    remove_checkout: bool = False,
    assume_yes: bool = False,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    error_output: Callable[[str], None] = print,
) -> int:
    """Entry for ``code-indexing-mcp uninstall``."""

    install_directory = (
        Path(install_dir).expanduser().resolve() if install_dir else default_install_directory()
    )
    try:
        slugs = resolve_slugs(harnesses_selection)
    except InstallerError as exc:
        error_output(f"Error: {exc}")
        return 1
    plan = UninstallPlan(
        install_directory=install_directory,
        harness_slugs=tuple(slugs),
        bin_directory=Path(bin_dir).expanduser() if bin_dir else None,
        remove_launcher=not keep_launcher,
        remove_path_block=not keep_path,
        remove_data=purge,
        remove_checkout=remove_checkout,
    )
    for line in describe_plan(plan):
        output(line)
    if not assume_yes:
        answer = input_fn("Proceed? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            output("Uninstall cancelled.")
            return 130

    def report(event: StepEvent) -> None:
        line = f"[{event.step}] {event.status}: {event.detail}"
        (error_output if event.status == "failed" else output)(line)

    result = run_uninstall(plan, report)
    if result.failures:
        error_output(f"Uninstall finished with {len(result.failures)} problem(s); see above.")
        return 1
    output("Uninstall complete.")
    if not plan.remove_data:
        output("Indexes and caches were kept; re-run with --purge to delete them.")
    return 0
