"""Wizard panels. Each panel owns its widgets and a commit() into WizardState."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, RichLog

from ..orchestrator import InstallResult
from ..wizard import WizardState
from .settings_form import SettingsPanel

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
        yield Label("Welcome")


class LocationPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Location")

    def commit(self) -> bool:
        return True


class AcceleratorPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Accelerator")

    def commit(self) -> bool:
        return True


class HarnessesPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Harnesses")

    def commit(self) -> bool:
        return True


class SummaryPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Summary")


class ProgressPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.cancelled = False

    def compose(self) -> ComposeResult:
        yield Label("Progress")
        yield RichLog(id="progress-log")
        yield Button("Cancel", id="progress-cancel", variant="error")

    def start(self) -> None:
        raise NotImplementedError


class DonePanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Label(id="done-title")
        yield Label("", id="done-body")
        yield Button("Exit", id="exit", variant="primary")

    def show_result(
        self,
        result: InstallResult | None,
        *,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        raise NotImplementedError
