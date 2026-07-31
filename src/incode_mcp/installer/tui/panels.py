"""Wizard panels. Each panel owns its widgets and a commit() into WizardState."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Static,
)

from .. import accelerator as accelerator_module
from .. import harnesses
from ..orchestrator import InstallResult, run_install
from ..wizard import WizardState
from .settings_form import SettingsPanel

if TYPE_CHECKING:
    from .app import InstallerApp

__all__ = [
    "AcceleratorPanel",
    "DonePanel",
    "HarnessesPanel",
    "LocationPanel",
    "ProgressPanel",
    "SettingsPanel",
    "SummaryPanel",
    "WelcomePanel",
]


class WelcomePanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        if self.state.mode == "reconfigure":
            headline = "Reconfigure Code Indexing MCP"
            detail = (
                f"Installation: {self.state.install_directory}\n"
                "Your current settings were read from the configured harnesses. "
                "Walk through the sections and confirm on the summary screen."
            )
        else:
            headline = "Install Code Indexing MCP"
            detail = (
                f"Installation: {self.state.install_directory}\n"
                "This wizard prepares the accelerator, configures your MCP clients, "
                "and lets you customize the server's settings."
            )
        yield Label(headline, id="welcome-headline")
        yield Static(detail)
        if self.state.disagreements:
            names = ", ".join(self.state.disagreements)
            yield Static(
                f"Your harnesses disagree on: {names}. The value from the earliest "
                "configured harness in the list is prefilled; confirming unifies them.",
                classes="help",
            )


class LocationPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Install location")
        yield Static(
            "Where the repository is cloned. Changing this moves only the checkout; "
            "indexes and caches live in the data directory (Indexing section).",
            classes="help",
        )
        with Collapsible(title="Advanced", collapsed=True):
            yield Label("Install directory")
            yield Input(value=str(self.state.install_directory), id="install-dir")
            yield Label("Repository URL")
            yield Input(value=self.state.repo_url, id="repo-url")
        yield Label("", id="location-error", classes="error")

    def commit(self) -> bool:
        directory = self.query_one("#install-dir", Input).value.strip()
        if not directory:
            self.query_one("#location-error", Label).update(
                "Install directory cannot be empty."
            )
            return False
        self.state.install_directory = Path(directory).expanduser()
        self.state.repo_url = self.query_one("#repo-url", Input).value.strip()
        return True


class AcceleratorPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Passage embedding accelerator")
        yield Static(
            "auto detects a supported GPU and prepares it; anything that cannot be "
            "detected, built, or probed leaves the installation on CPU and says why.",
            classes="help",
        )
        yield Static("\n".join(accelerator_module.detection_report()), id="detection")
        with RadioSet(id="accel-choices"):
            if self.state.mode == "reconfigure":
                prepared = self.state.prepared_accelerator or "none prepared"
                yield RadioButton(
                    f"Keep the prepared backend ({prepared})", id="accel-keep", value=True
                )
            for choice in accelerator_module.ACCELERATOR_CHOICES:
                yield RadioButton(
                    "auto (recommended)" if choice == "auto" else choice,
                    id=f"accel-{choice}",
                    value=self.state.accelerator == choice,
                )
        yield Static(
            "Preparing an accelerator downloads the embedding model and can take several "
            "minutes and a few gigabytes; a matching record is reused next time.",
            classes="help",
        )

    def commit(self) -> bool:
        if self.state.mode == "reconfigure" and self.query_one("#accel-keep", RadioButton).value:
            self.state.accelerator = None
            return True
        for choice in accelerator_module.ACCELERATOR_CHOICES:
            if self.query_one(f"#accel-{choice}", RadioButton).value:
                self.state.accelerator = choice
                return True
        self.state.accelerator = "auto"
        return True


class HarnessesPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("MCP clients to configure")
        yield Static(
            "The server entry is merged into each selected client's user-wide "
            "configuration; existing files are backed up with a .bak suffix first.",
            classes="help",
        )
        for choice in harnesses.HARNESS_CHOICES:
            path = harnesses.configuration_path(choice.slug)
            existing = choice.slug in self.state.configured_slugs
            skills = harnesses.skill_directory(choice.slug) is not None
            notes = [str(path)]
            if existing:
                notes.append("already configured")
            if skills:
                notes.append("skills supported")
            yield Checkbox(
                f"{choice.label} — {', '.join(notes)}",
                value=choice.slug in self.state.harness_slugs,
                id=f"harness-{choice.slug}",
            )

    def commit(self) -> bool:
        self.state.harness_slugs = [
            choice.slug
            for choice in harnesses.HARNESS_CHOICES
            if self.query_one(f"#harness-{choice.slug}", Checkbox).value
        ]
        return True


class SummaryPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Summary")
        yield Static("", id="summary-body")
        yield Static("Confirm to run the installation.", classes="help")

    def on_became_visible(self) -> None:
        lines = [f"Install directory: {self.state.install_directory}"]
        if self.state.accelerator is None:
            prepared = self.state.prepared_accelerator or "none"
            lines.append(f"Accelerator: keep the prepared backend ({prepared})")
        else:
            lines.append(f"Accelerator: {self.state.accelerator}")
            if self.state.accelerator in accelerator_module.ACCELERATOR_EXTRAS:
                lines.append(
                    "  Building this environment downloads several gigabytes and runs "
                    "a real inference probe."
                )
        lines.append("Harnesses: " + (", ".join(self.state.harness_slugs) or "none"))
        updates = self.state.env_updates()
        if updates:
            lines.append("Settings:")
            for name, value in sorted(updates.items()):
                lines.append(f"  {name} = {value if value is not None else '(removed)'}")
        else:
            lines.append("Settings: all defaults (nothing written to env blocks)")
        if self.state.harness_slugs:
            lines.append("Files that will be written:")
            for slug in self.state.harness_slugs:
                lines.append(f"  {harnesses.configuration_path(slug)}")
            if self.state.accelerator is not None:
                lines.append("  the accelerator record in the server's data directory")
        if self.state.disagreements:
            lines.append(
                "Harnesses that disagreed (" + ", ".join(self.state.disagreements) + ") "
                "will be unified on the values above."
            )
        self.query_one("#summary-body", Static).update("\n".join(lines))


class ProgressPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.cancelled = False

    def compose(self) -> ComposeResult:
        yield Label("Running the installation")
        yield Static(
            "The accelerator environment build and its probe can take several minutes.",
            classes="help",
        )
        yield RichLog(id="progress-log")
        yield Button("Cancel", id="progress-cancel", variant="error")

    def start(self) -> None:
        self.cancelled = False
        self.query_one("#progress-cancel", Button).disabled = False
        self._run_pipeline()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "progress-cancel":
            self.cancelled = True
            event.button.disabled = True
            self._log_line("cancel requested; stopping after the current step")
            event.stop()

    def _log_line(self, line: str) -> None:
        self.query_one("#progress-log", RichLog).write(line)

    @work(thread=True)
    def _run_pipeline(self) -> None:
        result: InstallResult | None = None
        error: Exception | None = None
        try:
            result = run_install(
                self.state.to_plan(),
                on_event=lambda event: self.app.call_from_thread(
                    self._log_line,
                    f"[{event.step}] {event.status}: {event.detail}",
                ),
                should_continue=lambda: not self.cancelled,
            )
        except Exception as exc:  # surfaced on the Done screen
            error = exc
        app = cast("InstallerApp", self.app)
        app.call_from_thread(app.finish, result, error=error, cancelled=self.cancelled)


class DonePanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Label(id="done-title")
        yield Static("", id="done-body")
        yield Button("Exit", id="exit", variant="primary")

    def show_result(
        self,
        result: InstallResult | None,
        *,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        lines: list[str] = []
        if cancelled:
            title = "Installation cancelled"
            lines.append("Stopped between steps; anything already written above still applies.")
        elif error is not None:
            title = "Installation failed"
            lines.append(str(error))
        elif result is None:
            title = "Installation failed"
            lines.append("No result was produced.")
        else:
            title = "Installation complete"
            if result.accelerator_plan is not None:
                plan = result.accelerator_plan
                marker = "" if plan.honored else " (fell back to CPU)"
                lines.append(f"Accelerator: {plan.accelerator}{marker}\n  {plan.reason}")
            for slug, path in result.configured:
                lines.append(f"Configured {slug}: {path}")
            for slug, message in result.failures:
                lines.append(f"FAILED {slug}: {message}")
            for slug, message in result.skills:
                lines.append(f"Skills for {slug}: {message}")
            if result.failures:
                title = "Installation complete with failures"
        lines.append("")
        lines.append("Restart configured clients to load the MCP server.")
        lines.append("Reconfigure later with: code-indexing-mcp configure")
        self.query_one("#done-title", Label).update(title)
        self.query_one("#done-body", Static).update("\n".join(lines))
