"""Harness detection, configuration merging, and bundled-skill installation."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from .config_files import SERVER_NAME, InstallerError, merge_codex_server, merge_json_object_entry
from .env_blocks import entry_from_text, env_from_entry, merge_env


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
                f"Unknown harness {token!r}; choose 1-6, all, or one of: {options}"
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
    elif slug in {"kimi-code", "claude-desktop"}:
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

    merge_json_object_entry(path, object_key, SERVER_NAME, entry)
    return path


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
    return None


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _backup_path(target: Path) -> Path:
    """Pick a backup name that does not overwrite a backup from an earlier install."""

    candidate = target.with_name(f"{target.name}.bak")
    counter = 2
    while candidate.is_symlink() or candidate.exists():
        candidate = target.with_name(f"{target.name}.bak.{counter}")
        counter += 1
    return candidate


def _link_destination(link: Path) -> Path:
    """Where a symlink points, in a form that compares reliably.

    Raw os.readlink output is not comparable: Windows hands back an extended-length
    "\\\\?\\C:\\..." path that never equals the plain path the link was created from,
    which would make every re-install look like a first install.
    """

    return link.resolve()


def _is_stale_bundled_link(target: Path) -> bool:
    """True when target is a link this installer left pointing at an older checkout."""

    if not target.is_symlink():
        return False
    skills_dir = _link_destination(target).parent
    return skills_dir.name == "skills" and skills_dir.parent.name == "code_indexing_mcp"


def _link_skill(source: Path, target: Path) -> bool:
    """Symlink one bundled skill folder, backing up any clashing entry.

    Returns True when a new link was created, False when it already existed.
    """

    if target.is_symlink() and _link_destination(target) == source.resolve():
        return False
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Build the replacement link before disturbing what is already there, so a
    # platform that cannot create symlinks at all fails without having moved a
    # skill the user owns out from under its harness.
    staged = target.with_name(f"{target.name}.incoming")
    if staged.is_symlink() or staged.exists():
        _remove_path(staged)
    staged.symlink_to(source, target_is_directory=True)
    try:
        if _is_stale_bundled_link(target):
            target.unlink()
        elif target.is_symlink() or target.exists():
            target.rename(_backup_path(target))
        staged.rename(target)
    except OSError:
        staged.unlink(missing_ok=True)
        raise
    return True


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
