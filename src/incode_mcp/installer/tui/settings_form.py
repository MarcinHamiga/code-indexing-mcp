"""Spec-driven settings forms for the wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Checkbox, Input, Label, Select, Static

from ..settings_spec import SETTINGS, Setting, as_bool, default_value, validate
from ..wizard import WizardState


class SettingField(Vertical):
    """One labelled input for a catalog setting, generated from its spec."""

    def __init__(self, setting: Setting, value: str = "") -> None:
        super().__init__(classes="field")
        self.setting = setting
        self.initial = value

    def compose(self) -> ComposeResult:
        widget_id = f"f-{self.setting.name}"
        # A prefilled value comes from a configuration file a user may have
        # written by hand, so neither widget may assume a canonical spelling:
        # Select raises on a value outside its options, and a bool has more
        # spellings than "1".
        if self.setting.type == "bool":
            yield Checkbox(
                self.setting.label,
                value=as_bool(self.initial or self.setting.default),
                id=widget_id,
            )
        elif self.setting.type == "choice":
            options = [(choice, choice) for choice in self.setting.choices]
            chosen = self.initial.strip().lower()
            yield Select(
                options,
                value=chosen if chosen in self.setting.choices else self.setting.default,
                id=widget_id,
                allow_blank=False,
            )
        else:
            yield Label(self.setting.label)
            yield Input(
                value=self.initial,
                placeholder=default_value(self.setting),
                id=widget_id,
            )
        yield Static(self.setting.help, classes="help")

    def value(self) -> str:
        widget = self.query_one(f"#f-{self.setting.name}")
        if isinstance(widget, Checkbox):
            return "1" if widget.value else "0"
        if isinstance(widget, Select):
            return str(widget.value)
        if isinstance(widget, Input):
            return widget.value.strip() or default_value(self.setting)
        raise AssertionError(f"unexpected widget for {self.setting.name}")

    def raw_input(self) -> str:
        """The typed text for Input fields ("" means 'use the default')."""

        widget = self.query_one(f"#f-{self.setting.name}")
        if isinstance(widget, Input):
            return widget.value.strip()
        return self.value()


class SettingsPanel(Vertical):
    """A group of SettingFields built from the catalog."""

    def __init__(self, state: WizardState, group: str, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.group = group

    def compose(self) -> ComposeResult:
        yield Label(f"{self.group} settings")
        yield Static(
            "Fields left empty keep their default and are not written to any config.",
            classes="help",
        )
        for setting in SETTINGS:
            if setting.group == self.group:
                yield SettingField(setting, self.state.field_value(setting.name))
        yield Label("", id=f"{self.group.lower()}-error", classes="error")

    def commit(self) -> bool:
        error_label = self.query_one(f"#{self.group.lower()}-error", Label)
        for field in self.query(SettingField):
            raw = field.raw_input()
            if raw:  # empty means default; defaults are valid by construction
                error = validate(field.setting, raw)
                if error is not None:
                    error_label.update(error)
                    return False
            self.state.set_field(field.setting.name, raw)
        error_label.update("")
        return True
