"""Wizard panels. Each panel owns its widgets and a commit() into WizardState."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.timer import Timer
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
from .. import harnesses, shell_path
from ..orchestrator import InstallResult, StepEvent, run_install
from ..verify import format_check
from ..wizard import WizardState
from .settings_form import SettingsPanel

if TYPE_CHECKING:
    from .app import InstallerApp

__all__ = [
    "AcceleratorPanel",
    "DonePanel",
    "HarnessesPanel",
    "LocationPanel",
    "PathPanel",
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
        hints = self._repair_hints()
        if hints:
            yield Static("\n".join(hints), id="welcome-repair", classes="error")

    def _repair_hints(self) -> list[str]:
        """Point at --repair when an existing install is visibly missing a piece.

        Only filesystem questions are asked here. Launching the server to prove
        it runs is what the verify step does at the end of a run, and is far too
        slow to sit between the user and the first screen.
        """

        if self.state.mode != "reconfigure":
            return []
        hints: list[str] = []
        executable = accelerator_module.server_executable(self.state.install_directory)
        if not executable.is_file():
            hints.append(f"This installation has no server executable at {executable}.")
        launcher = shell_path.launcher_path(self.state.bin_directory)
        if not launcher.exists():
            hints.append(
                f"The {launcher.name} launcher is missing from {launcher.parent}; "
                "the Command-line access step will put it back."
            )
        if hints:
            hints.append(
                "To restore everything without walking the wizard, quit and run: "
                "code-indexing-mcp configure --repair"
            )
        return hints


class LocationPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Install location")
        yield Static(
            "The checkout this wizard configures. It has already been cloned and its "
            "environment built; point this elsewhere only to configure a different "
            "existing installation. Indexes and caches live in the data directory "
            "(Indexing section).",
            classes="help",
        )
        with Collapsible(title="Advanced", collapsed=True):
            yield Label("Install directory")
            yield Input(value=str(self.state.install_directory), id="install-dir")
        yield Label("", id="location-error", classes="error")

    def commit(self) -> bool:
        error = self.query_one("#location-error", Label)
        value = self.query_one("#install-dir", Input).value.strip()
        if not value:
            error.update("Install directory cannot be empty.")
            return False
        directory = Path(value).expanduser()
        # The path written into every harness config is derived from this
        # directory, so a location without a built environment would configure
        # each client to launch a command that does not exist.
        executable = accelerator_module.server_executable(directory)
        if not executable.is_file():
            error.update(f"No prepared installation there; expected {executable}.")
            return False
        self.state.install_directory = directory
        error.update("")
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


class PathPanel(Vertical):
    """Whether to create the `code-indexing-mcp` launcher, and where."""

    # Inspecting a directory reads every shell profile and walks PATH with
    # shutil.which. That is far too much work to repeat on each keystroke, so
    # typing schedules it and only the last keystroke of a burst pays for it.
    INSPECT_DEBOUNCE_SECONDS = 0.4

    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self._inspect_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Label("Command-line access")
        yield Static(
            "Your MCP clients launch the server by absolute path and do not need this. "
            "The launcher is for you: it makes `code-indexing-mcp configure`, `index`, "
            "`status`, and `daemon` work from any shell.",
            classes="help",
        )
        yield Checkbox(
            "Create the code-indexing-mcp launcher",
            value=self.state.install_launcher,
            id="path-launcher",
        )
        yield Checkbox(
            "Add its directory to my shell profile",
            value=self.state.modify_shell_profiles,
            id="path-profile",
        )
        yield Static("", id="path-status", classes="help")
        with Collapsible(title="Advanced", collapsed=True):
            yield Label("Launcher directory")
            yield Input(value=str(self.state.bin_directory), id="path-bin-dir")
        yield Label("", id="path-error", classes="error")

    def on_became_visible(self) -> None:
        self._refresh_status()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "path-bin-dir":
            return
        if self._inspect_timer is not None:
            self._inspect_timer.stop()
        self._inspect_timer = self.set_timer(self.INSPECT_DEBOUNCE_SECONDS, self._refresh_status)

    def _refresh_status(self) -> None:
        self._inspect_timer = None
        raw = self.query_one("#path-bin-dir", Input).value.strip()
        if not raw:
            self.query_one("#path-status", Static).update("")
            return
        state = shell_path.inspect(Path(raw).expanduser())
        lines = [f"Launcher: {state.launcher}"]
        profile_checkbox = self.query_one("#path-profile", Checkbox)
        if state.on_path:
            lines.append(f"{state.bin_directory} is already on PATH; nothing to add.")
            # Nothing for the checkbox to do, so it says so rather than
            # implying a profile edit that will not happen.
            profile_checkbox.value = False
            profile_checkbox.disabled = True
        else:
            if profile_checkbox.disabled:
                # Coming back from a directory that was already on PATH: restore
                # the answer the user actually gave rather than leaving the
                # unchecked box this panel set for its own display reasons.
                profile_checkbox.value = self.state.modify_shell_profiles
            profile_checkbox.disabled = False
            if state.profiles:
                names = ", ".join(str(profile) for profile in state.profiles)
                lines.append(f"{state.bin_directory} is not on PATH; would edit: {names}")
            else:
                lines.append(
                    f"{state.bin_directory} is not on PATH, and this platform's profile "
                    "cannot be edited safely; add it yourself afterwards."
                )
        if state.shadowed_by is not None:
            lines.append(
                f"Note: {state.shadowed_by} comes earlier on PATH and would keep winning the name."
            )
        self.query_one("#path-status", Static).update("\n".join(lines))

    def commit(self) -> bool:
        error = self.query_one("#path-error", Label)
        self.state.install_launcher = self.query_one("#path-launcher", Checkbox).value
        profile_checkbox = self.query_one("#path-profile", Checkbox)
        if not profile_checkbox.disabled:
            # A disabled box was unchecked by _refresh_status for display, not by
            # the user; recording that as their answer would lose their choice if
            # they moved the launcher somewhere that is not on PATH.
            self.state.modify_shell_profiles = profile_checkbox.value
        raw = self.query_one("#path-bin-dir", Input).value.strip()
        if not raw:
            if not self.state.install_launcher:
                # No launcher means no directory to put one in. Demanding a path
                # that nothing will use would block the user for nothing.
                return True
            error.update("Launcher directory cannot be empty.")
            return False
        directory = Path(raw).expanduser()
        # A directory that cannot be created is worth catching here rather than
        # halfway through the pipeline, where the harness configs are already
        # written and the failure reads as a mystery.
        if self.state.install_launcher and not directory.is_dir():
            try:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            except OSError as exc:
                error.update(f"Cannot create {directory}: {exc}")
                return False
        self.state.bin_directory = directory
        error.update("")
        return True


class SummaryPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    # The panels worth jumping straight back to when the summary reads wrong.
    # Stepping back one screen at a time to fix one setting is the tedium this
    # removes.
    JUMP_TARGETS: tuple[tuple[str, str], ...] = (
        ("accelerator", "Accelerator"),
        ("harnesses", "Clients"),
        ("path", "Command-line access"),
        ("indexing", "Indexing"),
        ("embedding", "Embedding"),
    )

    def compose(self) -> ComposeResult:
        yield Label("Summary")
        yield Static("", id="summary-body")
        yield Static("Confirm to run the installation, or go back to:", classes="help")
        with Horizontal(id="summary-jumps"):
            for panel, label in self.JUMP_TARGETS:
                yield Button(label, id=f"jump-{panel}", classes="jump")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id or ""
        if button.startswith("jump-"):
            event.stop()
            cast("InstallerApp", self.app).show_panel(button.removeprefix("jump-"))

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
        lines.extend(self._launcher_lines())
        updates = self.state.env_updates()
        if updates:
            lines.append("Settings:")
            for name, value in sorted(updates.items()):
                lines.append(f"  {name} = {value if value is not None else '(removed)'}")
        else:
            lines.append("Settings: all defaults (nothing written to env blocks)")
        written = [harnesses.configuration_path(slug) for slug in self.state.harness_slugs]
        written.extend(self._path_files())
        if written:
            lines.append("Files that will be written:")
            lines.extend(f"  {path}" for path in written)
            if self.state.accelerator is not None:
                lines.append("  the accelerator record in the server's data directory")
        if self.state.disagreements:
            lines.append(
                "Harnesses that disagreed (" + ", ".join(self.state.disagreements) + ") "
                "will be unified on the values above."
            )
        self.query_one("#summary-body", Static).update("\n".join(lines))

    def _path_state(self) -> shell_path.PathState:
        return shell_path.inspect(self.state.bin_directory)

    def _launcher_lines(self) -> list[str]:
        if not self.state.install_launcher:
            return ["Launcher: not created"]
        state = self._path_state()
        lines = [f"Launcher: {state.launcher}"]
        if state.on_path:
            lines.append("  its directory is already on PATH")
        elif self.state.modify_shell_profiles and state.profiles:
            lines.append("  its directory will be added to PATH in your shell profile")
        else:
            lines.append("  its directory is not on PATH; add it yourself to use the command")
        return lines

    def _path_files(self) -> list[Path]:
        if not self.state.install_launcher:
            return []
        state = self._path_state()
        files = [state.launcher]
        if self.state.modify_shell_profiles and not state.on_path:
            files.extend(state.profiles)
        return files


# The steps in the order run_install runs them, with the label each row shows.
# A step absent from a given run (no accelerator to prepare, no launcher wanted)
# still gets a row, and ends up marked "skipped" rather than silently missing.
PIPELINE_STEPS: tuple[tuple[str, str], ...] = (
    ("accelerator", "Prepare the embedding accelerator"),
    ("path", "Install the launcher and PATH entry"),
    ("harnesses", "Configure MCP clients"),
    ("skills", "Link bundled skills"),
    ("verify", "Check the installation"),
)

_STEP_MARKERS = {
    "pending": "   ",
    "running": ">> ",
    "finished": "ok ",
    "skipped": "-- ",
    "warning": " ! ",
    "failed": "XX ",
}


class ProgressPanel(Vertical):
    """One row per pipeline step, with the raw event log kept underneath.

    The accelerator build takes minutes. A log that stops printing during it is
    indistinguishable from a hang, which is why every row carries its own
    elapsed clock while it runs.
    """

    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.cancelled = False
        self.result: InstallResult | None = None
        self._error: Exception | None = None
        # step -> status; the row's clock reads from _started/_elapsed.
        self._status: dict[str, str] = {}
        self._running_step: str | None = None
        self._elapsed: dict[str, int] = {}
        self._ticker: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Label("Running the installation")
        yield Static(
            "The accelerator environment build and its probe can take several minutes.",
            classes="help",
        )
        yield Static("", id="progress-steps")
        with Collapsible(title="Details", collapsed=True, id="progress-details"):
            yield RichLog(id="progress-log", markup=False, wrap=True)
        with Horizontal(id="progress-buttons"):
            yield Button("Cancel", id="progress-cancel", variant="error")
            yield Button("Retry failed steps", id="progress-retry", variant="primary")
            yield Button("Continue anyway", id="progress-continue")

    def on_mount(self) -> None:
        self._reset_rows()
        self._show_recovery_buttons(False)

    def _show_recovery_buttons(self, visible: bool) -> None:
        for button_id in ("#progress-retry", "#progress-continue"):
            self.query_one(button_id, Button).display = visible

    def _reset_rows(self) -> None:
        self._status = {step: "pending" for step, _ in PIPELINE_STEPS}
        self._elapsed = dict.fromkeys(self._status, 0)
        self._running_step = None
        self._render_rows()

    def _render_rows(self) -> None:
        lines = []
        for step, label in PIPELINE_STEPS:
            status = self._status.get(step, "pending")
            clock = ""
            seconds = self._elapsed.get(step, 0)
            if status == "running" or (seconds and status != "pending"):
                clock = f"  ({seconds}s)"
            lines.append(f"{_STEP_MARKERS[status]}{label}{clock}")
        self.query_one("#progress-steps", Static).update("\n".join(lines))

    def _tick(self) -> None:
        if self._running_step is None:
            return
        self._elapsed[self._running_step] += 1
        self._render_rows()

    def start(self) -> None:
        self.cancelled = False
        self.result = None
        self._reset_rows()
        self.query_one("#progress-log", RichLog).clear()
        self.query_one("#progress-cancel", Button).disabled = False
        self.query_one("#progress-cancel", Button).display = True
        self._show_recovery_buttons(False)
        if self._ticker is None:
            self._ticker = self.set_interval(1.0, self._tick)
        self._run_pipeline()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "progress-cancel":
            self.cancelled = True
            event.button.disabled = True
            self._log_line("cancel requested; stopping after the current step")
            event.stop()
        elif event.button.id == "progress-retry":
            event.stop()
            self._retry()
        elif event.button.id == "progress-continue":
            event.stop()
            app = cast("InstallerApp", self.app)
            app.finish(self.result, error=self._error, cancelled=False)

    def _retry(self) -> None:
        """Re-run only what failed, keeping the accelerator build already paid for."""

        failed = [slug for slug, _ in (self.result.failures if self.result else ())]
        if failed:
            self.state.harness_slugs = failed
        self.state.accelerator = None
        self.start()

    def _log_line(self, line: str) -> None:
        self.query_one("#progress-log", RichLog).write(line)

    def _on_step_event(self, event: StepEvent) -> None:
        self._log_line(f"[{event.step}] {event.status}: {event.detail}")
        if event.step not in self._status:
            return
        if event.status == "started":
            self._running_step = event.step
            self._status[event.step] = "running"
        else:
            if self._running_step == event.step:
                self._running_step = None
            current = self._status[event.step]
            # A step reports once per unit of work, so the row keeps the worst
            # outcome it saw rather than whatever happened to arrive last.
            rank = {"pending": 0, "running": 1, "skipped": 2, "finished": 3, "warning": 4}
            if event.status == "failed" or current == "failed":
                self._status[event.step] = "failed"
            elif rank.get(event.status, 0) >= rank.get(current, 0):
                self._status[event.step] = event.status
        self._render_rows()

    def _settle_rows(self) -> None:
        """Anything still pending or running when the pipeline ends never ran."""

        for step in self._status:
            if self._status[step] in {"pending", "running"}:
                self._status[step] = "skipped"
        self._running_step = None
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        self._render_rows()

    def finished(
        self,
        result: InstallResult | None,
        error: Exception | None,
        cancelled: bool,
    ) -> None:
        self.result = result
        self._error = error
        self._settle_rows()
        app = cast("InstallerApp", self.app)
        failed = bool(error) or bool(result and result.failures)
        if failed and not cancelled:
            # Stay put and offer the retry: moving straight to Done would make a
            # one-file permission problem cost another full run to put right.
            self.query_one("#progress-cancel", Button).display = False
            self._show_recovery_buttons(True)
            self.query_one("#progress-retry", Button).focus()
            self.query_one("#progress-details", Collapsible).collapsed = False
            self._log_line("")
            self._log_line(
                "Retry re-runs the failed clients only, keeping the prepared accelerator."
            )
            return
        app.finish(result, error=error, cancelled=cancelled)

    @work(thread=True)
    def _run_pipeline(self) -> None:
        result: InstallResult | None = None
        error: Exception | None = None
        try:
            result = run_install(
                self.state.to_plan(),
                on_event=lambda event: self.app.call_from_thread(self._on_step_event, event),
                should_continue=lambda: not self.cancelled,
            )
        except Exception as exc:  # surfaced on the Done screen
            error = exc
        app = cast("InstallerApp", self.app)
        app.call_from_thread(self.finished, result, error, self.cancelled)


def _next_step_lines(result: InstallResult | None) -> list[str]:
    """How to actually run the command, given what the install managed to do.

    Telling someone to run `code-indexing-mcp configure` is only useful once
    their shell can resolve it, which depends on whether a launcher was made and
    whether the PATH entry is live in the session they are sitting in.
    """

    reconfigure = "Reconfigure later with: code-indexing-mcp configure"
    if result is None or result.launcher is None or not result.launcher.ok:
        return [
            "Reconfigure later with:",
            f"  {shell_path.launcher_path(shell_path.default_bin_directory())} configure",
            "  (no launcher was created, so run it by path or add it to PATH yourself)",
        ]
    if result.profiles_updated:
        return [
            f"PATH was updated in {len(result.profiles_updated)} shell profile(s). "
            "Start a new shell, or run:",
            f"  {shell_path.activation_hint(result.profiles_updated)}",
            reconfigure,
        ]
    return [reconfigure]


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
            if result.launcher is not None:
                launcher = result.launcher
                verb = "Launcher" if launcher.ok else "Launcher NOT created"
                lines.append(f"{verb}: {launcher.path}\n  {launcher.detail}")
            for profile in result.profiles_updated:
                lines.append(f"Added to PATH in {profile}")
            for slug, path in result.configured:
                lines.append(f"Configured {slug}: {path}")
            for slug, message in result.failures:
                lines.append(f"FAILED {slug}: {message}")
            for slug, message in result.skills:
                lines.append(f"Skills for {slug}: {message}")
            if result.checks:
                lines.append("")
                lines.append("Checks:")
                lines.extend(f"  {format_check(check)}" for check in result.checks)
            if result.failures:
                title = "Installation complete with failures"
            elif result.warnings:
                title = "Installation complete with warnings"
        lines.append("")
        lines.append("Restart configured clients to load the MCP server.")
        lines.extend(_next_step_lines(result))
        self.query_one("#done-title", Label).update(title)
        self.query_one("#done-body", Static).update("\n".join(lines))
