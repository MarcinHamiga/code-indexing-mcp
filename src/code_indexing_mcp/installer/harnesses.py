"""Harness detection, configuration merging, and bundled-skill installation."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from .config_files import (
    SERVER_NAME,
    InstallerError,
    merge_codex_server,
    merge_json_object_entry,
    merge_muse_code_entry,
    remove_codex_server,
    remove_json_object_entry,
)
from .env_blocks import OBJECT_KEYS, entry_from_text, env_from_entry, merge_env
from .links import is_under, link_destination, replace_link


class HarnessChoice(NamedTuple):
    slug: str
    label: str


HARNESS_CHOICES = [
    HarnessChoice("codex", "Codex (CLI + Desktop)"),
    HarnessChoice("claude-code", "Claude Code"),
    HarnessChoice("kimi-code", "Kimi Code"),
    HarnessChoice("claude-desktop", "Claude Desktop"),
    HarnessChoice("opencode", "OpenCode"),
    HarnessChoice("kilocode", "KiloCode"),
    HarnessChoice("antigravity", "Antigravity 2"),
    HarnessChoice("antigravity-cli", "Antigravity CLI"),
    HarnessChoice("muse-code", "Muse Code"),
]


def parse_harness_selection(selection: str) -> list[str]:
    """Parse interactive menu numbers or stable harness slugs."""

    value = selection.strip().lower()
    if not value:
        return []
    if value == "all":
        return [choice.slug for choice in HARNESS_CHOICES]

    by_slug = {choice.slug: choice.slug for choice in HARNESS_CHOICES}
    by_number = {str(index): choice.slug for index, choice in enumerate(HARNESS_CHOICES, start=1)}
    selected: list[str] = []
    for token in (part.strip().lower() for part in value.split(",")):
        slug = by_number.get(token, by_slug.get(token))
        if slug is None:
            options = ", ".join(choice.slug for choice in HARNESS_CHOICES)
            raise InstallerError(
                f"Unknown harness {token!r}; choose 1-{len(HARNESS_CHOICES)}, all, "
                f"or one of: {options}"
            )
        if slug not in selected:
            selected.append(slug)
    return selected


def _configured_directory(
    environment: Mapping[str, str],
    variable: str,
    default: Path,
) -> Path:
    configured = environment.get(variable)
    return Path(configured).expanduser() if configured else default


def _preferred_json_config(directory: Path, stem: str, *, default_suffix: str) -> Path:
    json_path = directory / f"{stem}.json"
    jsonc_path = directory / f"{stem}.jsonc"
    if json_path.exists():
        return json_path
    if jsonc_path.exists():
        return jsonc_path
    return directory / f"{stem}{default_suffix}"


def configuration_path(
    slug: str,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    """Return the user-wide configuration path for a supported harness."""

    home = home or Path.home()
    environment = os.environ if environment is None else environment
    platform_name = platform_name or sys.platform
    xdg_config = _configured_directory(
        environment,
        "XDG_CONFIG_HOME",
        home / ".config",
    )

    if slug == "codex":
        return _configured_directory(environment, "CODEX_HOME", home / ".codex") / "config.toml"
    if slug == "claude-code":
        return _configured_directory(environment, "CLAUDE_CONFIG_DIR", home) / ".claude.json"
    if slug == "kimi-code":
        return (
            _configured_directory(environment, "KIMI_CODE_HOME", home / ".kimi-code") / "mcp.json"
        )
    if slug == "claude-desktop":
        if platform_name == "darwin":
            return (
                home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
            )
        if platform_name.startswith("win"):
            app_data = environment.get("APPDATA")
            if not app_data:
                raise InstallerError("APPDATA is required to configure Claude Desktop on Windows")
            return Path(app_data).expanduser() / "Claude" / "claude_desktop_config.json"
        if platform_name.startswith("linux"):
            return xdg_config / "Claude" / "claude_desktop_config.json"
        raise InstallerError(f"Claude Desktop configuration is not supported on {platform_name}")
    if slug == "opencode":
        configured = environment.get("OPENCODE_CONFIG")
        if configured:
            return Path(configured).expanduser()
        directory = _configured_directory(
            environment,
            "OPENCODE_CONFIG_DIR",
            xdg_config / "opencode",
        )
        return _preferred_json_config(directory, "opencode", default_suffix=".json")
    if slug == "kilocode":
        configured = environment.get("KILO_CONFIG")
        if configured:
            return Path(configured).expanduser()
        directory = _configured_directory(
            environment,
            "KILO_CONFIG_DIR",
            xdg_config / "kilo",
        )
        return _preferred_json_config(directory, "kilo", default_suffix=".jsonc")
    if slug == "antigravity":
        # Antigravity 2 keeps its user-wide state under the shared Gemini home.
        return (
            _configured_directory(environment, "ANTIGRAVITY_HOME", home / ".gemini" / "config")
            / "mcp_config.json"
        )
    if slug == "antigravity-cli":
        configured = environment.get("ANTIGRAVITY_CLI_HOME") or environment.get("AGY_HOME")
        directory = (
            Path(configured).expanduser() if configured else home / ".gemini" / "antigravity-cli"
        )
        return directory / "mcp_config.json"
    if slug == "muse-code":
        directory = _configured_directory(environment, "XDG_CONFIG_HOME", home / ".config")
        return directory / "muse" / "settings.json"
    raise InstallerError(f"Unknown harness {slug!r}")


def read_server_entry(
    slug: str,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any] | None:
    """Return the current server entry in a harness config, or None."""

    path = configuration_path(
        slug,
        home=home,
        environment=environment,
        platform_name=platform_name,
    )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # An unreadable or non-UTF-8 config has nothing to tell us. Writing to it
        # still fails loudly later; reading it must not take the wizard down.
        return None
    return entry_from_text(slug, text)


def configure_harness(
    slug: str,
    command: Path,
    *,
    env: Mapping[str, str | None] | None = None,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    """Merge the Code Indexing MCP entry into one user-wide harness config.

    ``env`` maps managed setting names to values, or to None to delete a key;
    unrelated keys already in the entry's env block are preserved. When ``env``
    is None the legacy entries are written exactly as before.
    """

    path = configuration_path(
        slug,
        home=home,
        environment=environment,
        platform_name=platform_name,
    )
    merged_env: dict[str, str] = {}
    if env is not None:
        existing = read_server_entry(
            slug,
            home=home,
            environment=environment,
            platform_name=platform_name,
        )
        merged_env = merge_env(env_from_entry(slug, existing) if existing else {}, env)
    if slug == "codex":
        merge_codex_server(path, command, env=merged_env if env is not None else None)
        return path

    if slug == "claude-code":
        object_key = "mcpServers"
        entry: dict[str, Any] = {
            "type": "stdio",
            "command": str(command),
            "args": ["serve"],
        }
        if merged_env:
            entry["env"] = merged_env
    elif slug in {"kimi-code", "claude-desktop", "antigravity", "antigravity-cli", "muse-code"}:
        object_key = "mcpServers"
        entry = {"command": str(command), "args": ["serve"]}
        if merged_env:
            entry["env"] = merged_env
    elif slug in {"opencode", "kilocode"}:
        object_key = "mcp"
        entry = {
            "type": "local",
            "command": [str(command), "serve"],
            "enabled": True,
        }
        if merged_env:
            entry["environment"] = merged_env
    else:
        raise InstallerError(f"Unknown harness {slug!r}")

    if slug == "muse-code":
        # Muse Code rejects a settings document without schema_version; the
        # merge adds it when the file does not already carry one.
        merge_muse_code_entry(path, SERVER_NAME, entry)
        return path
    merge_json_object_entry(path, object_key, SERVER_NAME, entry)
    return path


def deconfigure_harness(
    slug: str,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> tuple[Path, bool]:
    """Remove the Code Indexing MCP entry from one harness config.

    Returns the config path and whether anything was actually removed. Only the
    server's own entry goes; every other key in the file is left untouched.
    """

    path = configuration_path(
        slug,
        home=home,
        environment=environment,
        platform_name=platform_name,
    )
    if not path.exists():
        return path, False
    if slug == "codex":
        return path, remove_codex_server(path)
    object_key = OBJECT_KEYS.get(slug)
    if object_key is None:
        raise InstallerError(f"Unknown harness {slug!r}")
    return path, remove_json_object_entry(path, object_key, SERVER_NAME)


def deconfigure_selected_harnesses(
    slugs: list[str],
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> tuple[list[tuple[str, Path, bool]], list[tuple[str, str]]]:
    """Remove the entry from every selection, keeping one client's failure isolated."""

    removed: list[tuple[str, Path, bool]] = []
    failures: list[tuple[str, str]] = []
    for slug in slugs:
        try:
            path, changed = deconfigure_harness(
                slug,
                home=home,
                environment=environment,
                platform_name=platform_name,
            )
        except (InstallerError, OSError) as exc:
            failures.append((slug, str(exc)))
        else:
            removed.append((slug, path, changed))
    return removed, failures


def remove_skills(
    slugs: list[str],
    install_directory: Path | None = None,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Unlink bundled skills, leaving anything the user owns exactly where it is.

    ``install_directory`` scopes removal to links pointing into the checkout
    being uninstalled, so a second installation elsewhere keeps its own links.
    """

    results: list[tuple[str, str]] = []
    for slug in slugs:
        directory = skill_directory(slug, home=home, environment=environment)
        if directory is None or not directory.is_dir():
            results.append((slug, "skipped: no skill directory"))
            continue
        removed = 0
        try:
            for entry in sorted(directory.iterdir()):
                # Only links this installer left, recognised by where they point.
                if is_bundled_skill_link(entry, install_directory):
                    entry.unlink()
                    removed += 1
        except OSError as exc:
            results.append((slug, f"skipped: {exc}"))
            continue
        results.append((slug, f"{removed} unlinked from {directory}"))
    return results


def configure_selected_harnesses(
    slugs: list[str],
    command: Path,
    *,
    env: Mapping[str, str | None] | None = None,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> tuple[list[tuple[str, Path]], list[tuple[str, str]]]:
    """Configure every selection while keeping one client's failure isolated."""

    successes: list[tuple[str, Path]] = []
    failures: list[tuple[str, str]] = []
    for slug in slugs:
        try:
            path = configure_harness(
                slug,
                command,
                env=env,
                home=home,
                environment=environment,
                platform_name=platform_name,
            )
        except (InstallerError, OSError) as exc:
            failures.append((slug, str(exc)))
        else:
            successes.append((slug, path))
    return successes, failures


def skill_directory(
    slug: str,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the user-level skill directory for a harness, or None if unsupported."""

    home = home or Path.home()
    environment = os.environ if environment is None else environment
    if slug == "claude-code":
        return _configured_directory(environment, "CLAUDE_CONFIG_DIR", home / ".claude") / "skills"
    if slug in {"codex", "kimi-code"}:
        return home / ".agents" / "skills"
    if slug == "opencode":
        xdg_config = _configured_directory(environment, "XDG_CONFIG_HOME", home / ".config")
        return xdg_config / "opencode" / "skills"
    if slug == "antigravity":
        return (
            _configured_directory(environment, "ANTIGRAVITY_HOME", home / ".gemini" / "config")
            / "skills"
        )
    if slug == "antigravity-cli":
        configured = environment.get("ANTIGRAVITY_CLI_HOME") or environment.get("AGY_HOME")
        directory = (
            Path(configured).expanduser() if configured else home / ".gemini" / "antigravity-cli"
        )
        return directory / "skills"
    if slug == "muse-code":
        xdg_config = _configured_directory(environment, "XDG_CONFIG_HOME", home / ".config")
        return xdg_config / "muse" / "skills"
    return None


def is_bundled_skill_link(target: Path, install_directory: Path | None = None) -> bool:
    """True when ``target`` is a link to a skill bundled with this project.

    Without ``install_directory`` the test is shape-only -- it points into some
    ``code_indexing_mcp/skills`` directory -- which is what installing needs:
    a link left by an *older* checkout is exactly the one to replace. Removal
    needs the opposite, so it passes the checkout and requires a link into it.
    """

    if not target.is_symlink():
        return False
    destination = link_destination(target)
    skills_dir = destination.parent
    if not (skills_dir.name == "skills" and skills_dir.parent.name == "code_indexing_mcp"):
        return False
    if install_directory is None:
        return True
    return is_under(destination, install_directory)


def _is_stale_bundled_link(target: Path) -> bool:
    """True when target is a link this installer left pointing at an older checkout."""

    return is_bundled_skill_link(target)


def _link_skill(source: Path, target: Path) -> bool:
    """Symlink one bundled skill folder, backing up any clashing entry.

    Returns True when a new link was created, False when it already existed.
    """

    return replace_link(
        source,
        target,
        is_directory=True,
        stale=_is_stale_bundled_link(target),
    )


def install_skills(
    slugs: list[str],
    install_directory: Path,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Symlink bundled skills into each selected harness's skill directory.

    Returns one (slug, status message) pair per harness; per-harness problems
    become "skipped" messages instead of raising.
    """

    skills_source = install_directory / "src" / "code_indexing_mcp" / "skills"
    if not skills_source.is_dir():
        return [(slug, f"skipped: bundled skills not found at {skills_source}") for slug in slugs]
    skills = sorted(entry for entry in skills_source.iterdir() if (entry / "SKILL.md").is_file())
    results: list[tuple[str, str]] = []
    for slug in slugs:
        directory = skill_directory(slug, home=home, environment=environment)
        if directory is None:
            results.append((slug, "skipped: harness has no skill-directory support"))
            continue
        try:
            created = [_link_skill(skill, directory / skill.name) for skill in skills]
        except OSError as exc:
            results.append((slug, f"skipped: {exc}"))
            continue
        linked = sum(created)
        results.append(
            (
                slug,
                f"{linked} linked, {len(created) - linked} already installed in {directory}",
            )
        )
    return results


def harness_label(slug: str) -> str:
    return next((choice.label for choice in HARNESS_CHOICES if choice.slug == slug), slug)
