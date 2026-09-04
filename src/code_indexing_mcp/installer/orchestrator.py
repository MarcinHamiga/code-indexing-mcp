"""The install pipeline as event-emitting steps, shared by the CLI and the TUI."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import accelerator, harnesses, shell_path, verify
from .config_files import InstallerError
from .shell_path import LauncherResult

EMBED_ACCELERATOR_SETTING = "CODE_INDEXING_EMBED_ACCELERATOR"


def default_install_directory() -> Path:
    configured = os.environ.get("CODE_INDEXING_MCP_INSTALL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "code-indexing-mcp"


@dataclass(frozen=True)
class InstallPlan:
    install_directory: Path
    # None skips the accelerator step: the reconfigure default, which keeps the
    # backend the last install prepared.
    accelerator: str | None = "auto"
    harness_slugs: tuple[str, ...] = ()
    env_updates: Mapping[str, str | None] = field(default_factory=dict)
    offline: bool = False
    # None resolves to the default bin directory at run time, so a plan built
    # before the environment was inspected still lands in the right place.
    bin_directory: Path | None = None
    install_launcher: bool = True
    modify_shell_profiles: bool = True


@dataclass(frozen=True)
class StepEvent:
    step: str  # "accelerator" | "path" | "harnesses" | "skills"
    status: str  # "started" | "finished" | "warning" | "failed" | "skipped"
    detail: str = ""


@dataclass(frozen=True)
class InstallResult:
    accelerator_plan: accelerator.AcceleratorPlan | None
    configured: tuple[tuple[str, Path], ...]
    failures: tuple[tuple[str, str], ...]
    skills: tuple[tuple[str, str], ...]
    launcher: LauncherResult | None = None
    profiles_updated: tuple[Path, ...] = ()
    checks: tuple[verify.Check, ...] = ()
    # The env updates actually applied to the configured harnesses, accelerator
    # additions included -- exposed so a caller (D6: `configure`) can tell
    # whether a daemon-consumed setting changed without recomputing the
    # accelerator merge itself.
    env_written: Mapping[str, str | None] = field(default_factory=dict)
    tui_launcher: LauncherResult | None = None

    @property
    def warnings(self) -> tuple[verify.Check, ...]:
        return tuple(check for check in self.checks if not check.ok)


def run_install(
    plan: InstallPlan,
    on_event: Callable[[StepEvent], None] = lambda event: None,
    should_continue: Callable[[], bool] = lambda: True,
) -> InstallResult:
    """Run the pipeline, emitting an event at every step boundary.

    ``should_continue`` is checked between steps so a UI can cancel cleanly;
    a step already running always finishes.
    """

    accelerator_plan: accelerator.AcceleratorPlan | None = None
    if plan.accelerator is None:
        on_event(StepEvent("accelerator", "skipped", "keeping the prepared backend"))
    elif should_continue():
        on_event(StepEvent("accelerator", "started", plan.accelerator))
        accelerator_plan = accelerator.configure_accelerator(
            plan.install_directory, plan.accelerator, offline=plan.offline
        )
        status = "finished" if accelerator_plan.honored else "warning"
        detail = f"{accelerator_plan.accelerator} ({accelerator_plan.reason})"
        on_event(StepEvent("accelerator", status, detail))

    env_updates = dict(plan.env_updates)
    if (
        accelerator_plan is not None
        and plan.accelerator is not None
        and EMBED_ACCELERATOR_SETTING not in env_updates
    ):
        # The installer choice is also the user's runtime choice. In particular,
        # an experimental backend is never eligible for runtime ``auto``, so
        # preparing it without writing this setting silently leaves indexing on
        # CPU. Record the backend the installer actually resolved (which may be
        # a deliberate fallback), while ``auto`` removes an older override.
        env_updates[EMBED_ACCELERATOR_SETTING] = (
            None if plan.accelerator == "auto" else accelerator_plan.accelerator
        )

    launcher: LauncherResult | None = None
    tui_launcher: LauncherResult | None = None
    profiles_updated: tuple[Path, ...] = ()
    if not plan.install_launcher:
        on_event(StepEvent("path", "skipped", "launcher not requested"))
    elif should_continue():
        launcher, tui_launcher, profiles_updated = _install_launcher(plan, on_event)

    configured: list[tuple[str, Path]] = []
    failures: list[tuple[str, str]] = []
    skills: list[tuple[str, str]] = []
    if should_continue():
        command = accelerator.server_executable(plan.install_directory)
        if plan.harness_slugs and not command.is_file():
            # This path is what every configured client will try to launch. A
            # directory without a prepared environment would leave each of them
            # pointing at nothing, which fails silently at the client end.
            raise InstallerError(
                f"No prepared installation at {plan.install_directory}: "
                f"expected the server executable at {command}"
            )
        on_event(
            StepEvent("harnesses", "started", ", ".join(plan.harness_slugs) or "none selected")
        )
        configured, failures = harnesses.configure_selected_harnesses(
            list(plan.harness_slugs), command, env=env_updates
        )
        for slug, path in configured:
            on_event(StepEvent("harnesses", "finished", f"{slug}: {path}"))
        for slug, message in failures:
            on_event(StepEvent("harnesses", "failed", f"{slug}: {message}"))
    if should_continue():
        on_event(StepEvent("skills", "started"))
        skills = harnesses.install_skills(list(plan.harness_slugs), plan.install_directory)
        for slug, message in skills:
            on_event(StepEvent("skills", "finished", f"{slug}: {message}"))

    checks: tuple[verify.Check, ...] = ()
    if should_continue():
        on_event(StepEvent("verify", "started"))
        checks = verify.run_checks(
            plan.install_directory,
            tuple(configured),
            launcher=launcher,
            profiles_updated=profiles_updated,
            accelerator_was_prepared=plan.accelerator is not None,
        )
        for check in checks:
            # A check never fails the install: everything above it already
            # happened, and a warning is the honest way to say "and yet".
            on_event(
                StepEvent(
                    "verify",
                    "finished" if check.ok else "warning",
                    f"{check.name}: {check.detail}",
                )
            )
    return InstallResult(
        accelerator_plan,
        tuple(configured),
        tuple(failures),
        tuple(skills),
        launcher=launcher,
        profiles_updated=profiles_updated,
        checks=checks,
        env_written=env_updates,
        tui_launcher=tui_launcher,
    )


def _install_launcher(
    plan: InstallPlan,
    on_event: Callable[[StepEvent], None],
) -> tuple[LauncherResult, LauncherResult, tuple[Path, ...]]:
    """Create both launchers and, if asked, put their directory on PATH.

    Nothing here raises. A missing launcher costs the user a convenience; a
    raise would cost them the harness configuration that comes after it.
    """

    bin_directory = plan.bin_directory or shell_path.default_bin_directory()
    on_event(StepEvent("path", "started", str(bin_directory)))
    launcher = shell_path.install_launcher(
        plan.install_directory, bin_directory, command=shell_path.LAUNCHER_NAME
    )
    on_event(
        StepEvent(
            "path",
            "finished" if launcher.ok else "warning",
            f"{launcher.path}: {launcher.status} ({launcher.detail})",
        )
    )
    tui_launcher = shell_path.install_launcher(
        plan.install_directory, bin_directory, command=shell_path.TUI_LAUNCHER_NAME
    )
    on_event(
        StepEvent(
            "path",
            "finished" if tui_launcher.ok else "warning",
            f"{tui_launcher.path}: {tui_launcher.status} ({tui_launcher.detail})",
        )
    )
    if not plan.modify_shell_profiles:
        return launcher, tui_launcher, ()
    if shell_path.is_on_path(bin_directory):
        on_event(StepEvent("path", "skipped", f"{bin_directory} is already on PATH"))
        return launcher, tui_launcher, ()
    profiles = shell_path.shell_profiles()
    if not profiles:
        on_event(
            StepEvent("path", "warning", f"add {bin_directory} to PATH yourself on this platform")
        )
        return launcher, tui_launcher, ()
    written, profile_failures = shell_path.update_profiles(bin_directory, profiles)
    for profile in written:
        on_event(StepEvent("path", "finished", f"added {bin_directory} to PATH in {profile}"))
    for profile, message in profile_failures:
        on_event(StepEvent("path", "warning", f"{profile}: {message}"))
    if not written and not profile_failures:
        on_event(StepEvent("path", "skipped", "the shell profiles already set PATH"))
    return launcher, tui_launcher, tuple(written)
