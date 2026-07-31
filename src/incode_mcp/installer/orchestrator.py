"""The install pipeline as event-emitting steps, shared by the CLI and the TUI."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import accelerator, harnesses
from .config_files import InstallerError


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


@dataclass(frozen=True)
class StepEvent:
    step: str  # "accelerator" | "harnesses" | "skills"
    status: str  # "started" | "finished" | "warning" | "failed" | "skipped"
    detail: str = ""


@dataclass(frozen=True)
class InstallResult:
    accelerator_plan: accelerator.AcceleratorPlan | None
    configured: tuple[tuple[str, Path], ...]
    failures: tuple[tuple[str, str], ...]
    skills: tuple[tuple[str, str], ...]


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
            list(plan.harness_slugs), command, env=plan.env_updates
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
    return InstallResult(accelerator_plan, tuple(configured), tuple(failures), tuple(skills))
