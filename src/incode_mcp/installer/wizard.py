"""Wizard state shared by the Textual UI; no Textual imports in this module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import accelerator, harnesses
from .env_blocks import env_from_entry
from .orchestrator import DEFAULT_REPOSITORY_URL, InstallPlan, default_install_directory
from .settings_spec import BY_NAME, SETTINGS, default_value, normalize


@dataclass(frozen=True)
class Prefill:
    values: Mapping[str, str]  # managed env values found in harness configs
    configured_slugs: tuple[str, ...]  # harnesses that already have a server entry
    disagreements: tuple[str, ...]  # env names whose values differ between harnesses


def load_prefill(
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Prefill:
    """Read every harness's current server entry for wizard prefill."""

    values: dict[str, str] = {}
    configured: list[str] = []
    disagreements: list[str] = []
    for choice in harnesses.HARNESS_CHOICES:
        entry = harnesses.read_server_entry(choice.slug, home=home, environment=environment)
        if entry is None:
            continue
        configured.append(choice.slug)
        for name, value in env_from_entry(choice.slug, entry).items():
            if name not in BY_NAME:
                continue
            if name in values and values[name] != value:
                if name not in disagreements:
                    disagreements.append(name)
            else:
                values[name] = value
    return Prefill(values, tuple(configured), tuple(disagreements))


@dataclass
class WizardState:
    mode: str  # "install" | "reconfigure"
    install_directory: Path = field(default_factory=default_install_directory)
    repo_url: str = DEFAULT_REPOSITORY_URL
    # None keeps the prepared backend (reconfigure default); install mode uses "auto".
    accelerator: str | None = "auto"
    prepared_accelerator: str | None = None
    harness_slugs: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)  # env name -> raw field value
    prefilled_names: set[str] = field(default_factory=set)  # names found in existing configs
    disagreements: list[str] = field(default_factory=list)
    offline: bool = False

    @classmethod
    def for_install(
        cls,
        install_directory: Path,
        repo_url: str,
        *,
        preset_values: Mapping[str, str] | None = None,
        preset_accelerator: str | None = None,
        home: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> WizardState:
        prefill = load_prefill(home=home, environment=environment)
        values = dict(prefill.values)
        values.update(preset_values or {})
        return cls(
            mode="install",
            install_directory=install_directory,
            repo_url=repo_url,
            accelerator=preset_accelerator or "auto",
            harness_slugs=list(prefill.configured_slugs),
            values=values,
            prefilled_names=set(prefill.values),
            disagreements=list(prefill.disagreements),
        )

    @classmethod
    def for_reconfigure(
        cls,
        install_directory: Path,
        *,
        home: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> WizardState:
        prefill = load_prefill(home=home, environment=environment)
        return cls(
            mode="reconfigure",
            install_directory=install_directory,
            accelerator=None,
            prepared_accelerator=accelerator.prepared_accelerator(install_directory),
            harness_slugs=list(prefill.configured_slugs),
            values=dict(prefill.values),
            prefilled_names=set(prefill.values),
            disagreements=list(prefill.disagreements),
        )

    def field_value(self, name: str) -> str:
        return self.values.get(name, "")

    def set_field(self, name: str, raw: str) -> None:
        self.values[name] = raw

    def env_updates(self) -> dict[str, str | None]:
        """Non-default values to write; prefilled values reset to default delete the key."""

        updates: dict[str, str | None] = {}
        for setting in SETTINGS:
            raw = self.values.get(setting.name, "").strip()
            if not raw or raw == default_value(setting):
                if self.mode == "reconfigure" and setting.name in self.prefilled_names:
                    updates[setting.name] = None
                continue
            updates[setting.name] = normalize(setting, raw)
        return updates

    def to_plan(self) -> InstallPlan:
        return InstallPlan(
            install_directory=self.install_directory,
            accelerator=self.accelerator,
            harness_slugs=tuple(self.harness_slugs),
            env_updates=self.env_updates(),
            offline=self.offline,
        )
