"""Spec-driven settings forms for the wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical

from ..settings_spec import Setting
from ..wizard import WizardState


class SettingField(Vertical):
    """One labelled input for a catalog setting."""

    def __init__(self, setting: Setting, value: str = "") -> None:
        super().__init__(classes="field")
        self.setting = setting
        self.initial = value

    def compose(self) -> ComposeResult:
        yield from ()


class SettingsPanel(Vertical):
    """A group of SettingFields built from the catalog."""

    def __init__(self, state: WizardState, group: str, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.group = group

    def compose(self) -> ComposeResult:
        yield from ()

    def commit(self) -> bool:
        return True
