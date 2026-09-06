"""Spec-driven settings forms for the wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Checkbox, Input, Label, Select, Static

from ..settings_spec import SETTINGS, Setting, as_bool, default_value, validate
from ..wizard import WizardState


class SettingField(Vertical):
    """One labelled input for a catalog setting, generated from its spec.

    Every field renders the same three rows: a bold header (label plus the
    effective default, or a ``modified`` marker when the widget no longer
    matches it), the input widget itself, and the muted help line. Choice
    fields get a real label this way instead of a bare dropdown.
    """

    def __init__(self, setting: Setting, value: str = "") -> None:
        super().__init__(classes="field")
        self.setting = setting
        self.initial = value

    def compose(self) -> ComposeResult:
        widget_id = f"f-{self.setting.name}"
        # markup=False: headers carry literal "[modified]" / "[default: ...]"
        # markers, which the Rich parser would otherwise swallow as style tags.
        yield Static("", id=f"h-{self.setting.name}", classes="field-header", markup=False)
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
            yield Input(
                value=self.initial,
                placeholder=default_value(self.setting),
                id=widget_id,
            )
        yield Static(self.setting.help, classes="help")

    def on_mount(self) -> None:
        self.refresh_header()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == f"f-{self.setting.name}":
            self.refresh_header()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == f"f-{self.setting.name}":
            self.refresh_header()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == f"f-{self.setting.name}":
            self.refresh_header()

    def is_modified(self) -> bool:
        """Whether the widget currently differs from the setting's default."""

        default = default_value(self.setting)
        widget_id = f"f-{self.setting.name}"
        try:
            if self.setting.type == "bool":
                return self.query_one(f"#{widget_id}", Checkbox).value != as_bool(default)
            if self.setting.type == "choice":
                return str(self.query_one(f"#{widget_id}", Select).value) != default
            raw = self.query_one(f"#{widget_id}", Input).value.strip()
            return (raw or default) != default
        except NoMatches:
            return False

    def refresh_header(self) -> None:
        """Rewrite the header line: label, default, and modified marker."""

        try:
            header = self.query_one(f"#h-{self.setting.name}", Static)
        except NoMatches:
            return
        if self.is_modified():
            header.update(f"{self.setting.label} [modified]")
        else:
            header.update(f"{self.setting.label} [default: {default_value(self.setting)}]")

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
            "Fields at their default are not written to any config. "
            "Headers show each default; [modified] marks what will be written.",
            classes="help",
        )
        for setting in SETTINGS:
            if setting.group == self.group:
                yield SettingField(setting, self.state.field_value(setting.name))
        yield Label("", id=f"{self.group.lower()}-error", classes="error")

    def on_became_visible(self) -> None:
        for field in self.query(SettingField):
            field.refresh_header()

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
