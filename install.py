#!/usr/bin/env python3
"""Install or update Code Indexing MCP and configure supported MCP clients."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

SERVER_NAME = "code-indexing-mcp"
DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git"


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


class InstallerError(RuntimeError):
    """An actionable installer failure."""


def _skip_jsonc_trivia(text: str, position: int) -> int:
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        if text.startswith("//", position):
            newline = text.find("\n", position + 2)
            return len(text) if newline == -1 else _skip_jsonc_trivia(text, newline + 1)
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end == -1:
                raise ValueError("unterminated block comment")
            position = end + 2
            continue
        break
    return position


def _parse_json_string(text: str, position: int) -> tuple[str, int]:
    if position >= len(text) or text[position] != '"':
        raise ValueError("object keys must be double-quoted strings")
    end = position + 1
    while end < len(text):
        if text[end] == "\\":
            end += 2
            continue
        if text[end] == '"':
            end += 1
            try:
                return str(json.loads(text[position:end])), end
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON string") from exc
        end += 1
    raise ValueError("unterminated JSON string")


def _scan_jsonc_value(text: str, position: int) -> int:
    position = _skip_jsonc_trivia(text, position)
    if position >= len(text):
        raise ValueError("missing value")
    if text[position] == '"':
        _, end = _parse_json_string(text, position)
        return end
    if text[position] in "[{":
        stack = ["]" if text[position] == "[" else "}"]
        current = position + 1
        while current < len(text):
            if text[current] == '"':
                _, current = _parse_json_string(text, current)
                continue
            if text.startswith("//", current) or text.startswith("/*", current):
                current = _skip_jsonc_trivia(text, current)
                continue
            if text[current] in "[{":
                stack.append("]" if text[current] == "[" else "}")
            elif text[current] in "]}":
                if text[current] != stack[-1]:
                    raise ValueError(f"unexpected {text[current]!r}")
                stack.pop()
                if not stack:
                    return current + 1
            current += 1
        raise ValueError("unterminated object or array")

    end = position
    while end < len(text) and text[end] not in ",}":
        if text.startswith("//", end) or text.startswith("/*", end):
            break
        end += 1
    token = text[position:end].strip()
    if not token:
        raise ValueError("missing value")
    try:
        json.loads(token)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid value {token!r}") from exc
    return position + len(text[position:end].rstrip())


JsonMember = tuple[str, int, int]


def _jsonc_object_members(text: str, object_start: int) -> tuple[list[JsonMember], int]:
    if object_start >= len(text) or text[object_start] != "{":
        raise ValueError("expected an object")
    members: list[JsonMember] = []
    position = object_start + 1
    while True:
        position = _skip_jsonc_trivia(text, position)
        if position >= len(text):
            raise ValueError("unterminated object")
        if text[position] == "}":
            return members, position
        key, position = _parse_json_string(text, position)
        position = _skip_jsonc_trivia(text, position)
        if position >= len(text) or text[position] != ":":
            raise ValueError(f"missing colon after {key!r}")
        value_start = _skip_jsonc_trivia(text, position + 1)
        value_end = _scan_jsonc_value(text, value_start)
        members.append((key, value_start, value_end))
        position = _skip_jsonc_trivia(text, value_end)
        if position >= len(text):
            raise ValueError("unterminated object")
        if text[position] == ",":
            position += 1
            continue
        if text[position] == "}":
            return members, position
        raise ValueError(f"expected a comma or closing brace after {key!r}")


def _line_indent(text: str, position: int) -> str:
    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position]
    return prefix[: len(prefix) - len(prefix.lstrip())]


def _format_json_value(value: Any, base_indent: str) -> str:
    lines = json.dumps(value, ensure_ascii=False, indent=2).splitlines()
    return lines[0] + "".join(f"\n{base_indent}{line}" for line in lines[1:])


def _jsonc_as_json(text: str) -> str:
    without_comments: list[str] = []
    position = 0
    while position < len(text):
        if text[position] == '"':
            _, end = _parse_json_string(text, position)
            without_comments.append(text[position:end])
            position = end
            continue
        if text.startswith("//", position):
            end = text.find("\n", position + 2)
            if end == -1:
                without_comments.append(" " * (len(text) - position))
                break
            without_comments.append(" " * (end - position))
            position = end
            continue
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end == -1:
                raise ValueError("unterminated block comment")
            comment = text[position : end + 2]
            without_comments.append("".join("\n" if char == "\n" else " " for char in comment))
            position = end + 2
            continue
        without_comments.append(text[position])
        position += 1

    cleaned = "".join(without_comments)
    without_trailing_commas: list[str] = []
    position = 0
    while position < len(cleaned):
        if cleaned[position] == '"':
            _, end = _parse_json_string(cleaned, position)
            without_trailing_commas.append(cleaned[position:end])
            position = end
            continue
        if cleaned[position] == ",":
            next_token = position + 1
            while next_token < len(cleaned) and cleaned[next_token].isspace():
                next_token += 1
            if next_token < len(cleaned) and cleaned[next_token] in "]}":
                position += 1
                continue
        without_trailing_commas.append(cleaned[position])
        position += 1
    return "".join(without_trailing_commas)


def _validate_jsonc(text: str) -> None:
    try:
        json.loads(_jsonc_as_json(text))
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc


def _insert_jsonc_member(
    text: str,
    object_start: int,
    object_end: int,
    members: list[JsonMember],
    key: str,
    value: Any,
) -> str:
    if members:
        _, _, last_value_end = members[-1]
        after_value = _skip_jsonc_trivia(text, last_value_end)
        if after_value >= len(text) or text[after_value] != ",":
            text = text[:last_value_end] + "," + text[last_value_end:]
            object_end += 1

    closing_line_start = text.rfind("\n", 0, object_end) + 1
    if text[closing_line_start:object_end].strip():
        insertion_point = object_end
        closing_indent = _line_indent(text, object_start)
        closing_suffix = f"\n{closing_indent}"
    else:
        insertion_point = closing_line_start
        closing_indent = text[closing_line_start:object_end]
        closing_suffix = ""

    member_indent = f"{closing_indent}  "
    encoded_key = json.dumps(key, ensure_ascii=False)
    encoded_value = _format_json_value(value, member_indent)
    leading_newline = "" if text[:insertion_point].endswith("\n") else "\n"
    addition = f"{leading_newline}{member_indent}{encoded_key}: {encoded_value}\n{closing_suffix}"
    return text[:insertion_point] + addition + text[insertion_point:]


def _merge_jsonc_text(text: str, object_key: str, entry_key: str, entry_value: Any) -> str:
    root_start = _skip_jsonc_trivia(text, 0)
    if root_start >= len(text) or text[root_start] != "{":
        raise ValueError("configuration root must be an object")
    root_members, root_end = _jsonc_object_members(text, root_start)
    if _skip_jsonc_trivia(text, root_end + 1) != len(text):
        raise ValueError("unexpected content after the root object")

    root_member = next((member for member in root_members if member[0] == object_key), None)
    if root_member is None:
        return _insert_jsonc_member(
            text,
            root_start,
            root_end,
            root_members,
            object_key,
            {entry_key: entry_value},
        )

    _, object_start, object_value_end = root_member
    if text[object_start] != "{":
        raise ValueError(f"{object_key!r} must contain an object")
    entries, object_end = _jsonc_object_members(text, object_start)
    if object_end + 1 != object_value_end:
        raise ValueError(f"{object_key!r} has an invalid object value")
    entry = next((member for member in entries if member[0] == entry_key), None)
    if entry is None:
        return _insert_jsonc_member(
            text,
            object_start,
            object_end,
            entries,
            entry_key,
            entry_value,
        )

    _, value_start, value_end = entry
    replacement = _format_json_value(entry_value, _line_indent(text, value_start))
    return text[:value_start] + replacement + text[value_end:]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
        if path.exists():
            temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_changed_configuration(path: Path, original: str | None, updated: str) -> bool:
    if original == updated:
        return False
    if original is not None:
        shutil.copy2(path, path.with_name(f"{path.name}.bak"))
    _atomic_write(path, updated)
    return True


def _read_configuration(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InstallerError(f"Configuration must be UTF-8: {path}") from exc


def merge_json_object_entry(
    path: Path,
    object_key: str,
    entry_key: str,
    entry_value: Any,
) -> bool:
    """Merge one entry into a top-level JSON/JSONC object without rewriting other text."""

    original = _read_configuration(path)
    source = original if original and original.strip() else "{}\n"
    try:
        _validate_jsonc(source)
        updated = _merge_jsonc_text(source, object_key, entry_key, entry_value)
        _validate_jsonc(updated)
    except ValueError as exc:
        raise InstallerError(f"Invalid JSON/JSONC configuration in {path}: {exc}") from exc
    return _write_changed_configuration(path, original, updated)


_TOML_TABLE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"\[\[\s*(?P<array_name>[^]\n]+?)\s*\]\]"
    r"|"
    r"\[\s*(?P<table_name>[^]\n]+?)\s*\]"
    r")[ \t]*(?:#.*)?$"
)


def _split_toml_dotted_key(value: str) -> list[str]:
    components: list[str] = []
    position = 0
    while position < len(value):
        while position < len(value) and value[position].isspace():
            position += 1
        if position >= len(value):
            raise ValueError("empty table component")
        if value[position] == '"':
            component, end = _parse_json_string(value, position)
            position = end
        elif value[position] == "'":
            end = value.find("'", position + 1)
            if end == -1:
                raise ValueError("unterminated literal table key")
            component = value[position + 1 : end]
            position = end + 1
        else:
            match = re.match(r"[A-Za-z0-9_-]+", value[position:])
            if match is None:
                raise ValueError("invalid bare table key")
            component = match.group()
            position += len(component)
        components.append(component)
        while position < len(value) and value[position].isspace():
            position += 1
        if position == len(value):
            return components
        if value[position] != ".":
            raise ValueError("expected a dot between table keys")
        position += 1
    raise ValueError("empty table component")


def _codex_server_block(command: Path) -> str:
    encoded_command = json.dumps(str(command), ensure_ascii=False)
    return f'[mcp_servers.{SERVER_NAME}]\ncommand = {encoded_command}\nargs = ["serve"]\n'


def _trailing_toml_trivia(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return "".join(lines[index + 1 :])
    return text


def merge_codex_server(path: Path, command: Path) -> bool:
    """Create or replace only the Code Indexing MCP table in a Codex config."""

    original = _read_configuration(path)
    source = original or ""
    parsed: dict[str, Any] = {}
    if source.strip():
        try:
            parsed = tomllib.loads(source)
        except tomllib.TOMLDecodeError as exc:
            raise InstallerError(f"Invalid TOML configuration in {path}: {exc}") from exc

    headings: list[tuple[re.Match[str], list[str], bool]] = []
    for match in _TOML_TABLE.finditer(source):
        try:
            array_name = match.group("array_name")
            name = array_name or match.group("table_name")
            headings.append((match, _split_toml_dotted_key(name), array_name is not None))
        except ValueError as exc:
            raise InstallerError(f"Invalid TOML table in {path}: {exc}") from exc

    target = ["mcp_servers", SERVER_NAME]
    target_index = next(
        (
            index
            for index, (_, components, is_array) in enumerate(headings)
            if components == target and not is_array
        ),
        None,
    )
    block = _codex_server_block(command)
    if target_index is None:
        mcp_servers = parsed.get("mcp_servers")
        if isinstance(mcp_servers, dict) and SERVER_NAME in mcp_servers:
            raise InstallerError(
                f"Codex server {SERVER_NAME!r} uses an inline or dotted TOML definition in "
                f"{path}; convert it to [mcp_servers.{SERVER_NAME}] before rerunning"
            )
        prefix = f"{source.rstrip()}\n\n" if source.strip() else ""
        updated = prefix + block
    else:
        start = headings[target_index][0].start()
        end = len(source)
        for match, components, _ in headings[target_index + 1 :]:
            if components[: len(target)] != target:
                end = match.start()
                break
        trailing_trivia = _trailing_toml_trivia(source[start:end])
        updated = source[:start] + block + trailing_trivia + source[end:]

    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise InstallerError(
            f"Refusing to write invalid TOML configuration to {path}: {exc}"
        ) from exc
    return _write_changed_configuration(path, original, updated)


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


def configure_harness(
    slug: str,
    command: Path,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    """Merge the Code Indexing MCP entry into one user-wide harness config."""

    path = configuration_path(
        slug,
        home=home,
        environment=environment,
        platform_name=platform_name,
    )
    if slug == "codex":
        merge_codex_server(path, command)
        return path

    if slug == "claude-code":
        object_key = "mcpServers"
        entry: dict[str, Any] = {
            "type": "stdio",
            "command": str(command),
            "args": ["serve"],
            "env": {},
        }
    elif slug in {"kimi-code", "claude-desktop"}:
        object_key = "mcpServers"
        entry = {"command": str(command), "args": ["serve"]}
    elif slug in {"opencode", "kilocode"}:
        object_key = "mcp"
        entry = {
            "type": "local",
            "command": [str(command), "serve"],
            "enabled": True,
        }
    else:
        raise InstallerError(f"Unknown harness {slug!r}")

    merge_json_object_entry(path, object_key, SERVER_NAME, entry)
    return path


def _run_command(
    arguments: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise InstallerError(f"Required command was not found: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        command = " ".join(arguments)
        message = f"Command failed: {command}"
        if detail:
            message = f"{message}\n{detail}"
        raise InstallerError(message) from exc


def _canonical_repository_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        return f"github.com/{value.removeprefix('git@github.com:').lower()}"
    for prefix in ("https://github.com/", "http://github.com/", "ssh://git@github.com/"):
        if value.startswith(prefix):
            return f"github.com/{value.removeprefix(prefix).lower()}"
    if "://" not in value:
        return str(Path(value).expanduser().resolve())
    return value


def clone_or_update_repository(repository_url: str, install_directory: Path) -> str:
    """Clone a fresh checkout or fast-forward an existing clean checkout."""

    git = shutil.which("git")
    if git is None:
        raise InstallerError("Git is required but was not found in PATH")
    install_directory = install_directory.expanduser().resolve()

    if not install_directory.exists():
        install_directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _run_command([git, "clone", "--", repository_url, str(install_directory)])
        return "installed"

    if not (install_directory / ".git").exists():
        raise InstallerError(
            f"Install target exists but is not a Git repository: {install_directory}"
        )

    origin = _run_command(
        [git, "remote", "get-url", "origin"],
        cwd=install_directory,
    ).stdout.strip()
    if _canonical_repository_url(origin) != _canonical_repository_url(repository_url):
        raise InstallerError(
            "Existing checkout origin does not match the requested repository: "
            f"{origin} != {repository_url}"
        )

    status = _run_command(
        [git, "status", "--porcelain"],
        cwd=install_directory,
    ).stdout
    if status.strip():
        raise InstallerError(
            f"Existing checkout has uncommitted changes; update it manually: {install_directory}"
        )

    _run_command([git, "pull", "--ff-only"], cwd=install_directory)
    return "updated"


def server_executable(
    install_directory: Path,
    *,
    platform_name: str | None = None,
) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return install_directory / ".venv" / "Scripts" / "code-indexing-mcp.exe"
    return install_directory / ".venv" / "bin" / "code-indexing-mcp"


def sync_environment(
    install_directory: Path,
    *,
    uv_executable: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Create or refresh the locked virtual environment and return its server command."""

    uv = uv_executable or shutil.which("uv")
    if uv is None:
        raise InstallerError(
            "uv is required but was not found in PATH. Install it from https://docs.astral.sh/uv/"
        )
    _run_command([uv, "sync", "--locked"], cwd=install_directory)
    command = server_executable(install_directory, platform_name=platform_name)
    if not command.is_file():
        raise InstallerError(f"uv sync completed but the MCP executable is missing: {command}")
    return command


def configure_selected_harnesses(
    slugs: list[str],
    command: Path,
    *,
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


def _is_stale_bundled_link(target: Path) -> bool:
    """True when target is a link this installer left pointing at an older checkout."""

    if not target.is_symlink():
        return False
    skills_dir = Path(os.readlink(target)).parent
    return skills_dir.name == "skills" and skills_dir.parent.name == "incode_mcp"


def _link_skill(source: Path, target: Path) -> bool:
    """Symlink one bundled skill folder, backing up any clashing entry.

    Returns True when a new link was created, False when it already existed.
    """

    if target.is_symlink() and Path(os.readlink(target)) == source:
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

    skills_source = install_directory / "src" / "incode_mcp" / "skills"
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


def _default_install_directory() -> Path:
    configured = os.environ.get("CODE_INDEXING_MCP_INSTALL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "code-indexing-mcp"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install or update Code Indexing MCP and configure it for selected MCP harnesses."
        )
    )
    parser.add_argument(
        "--install-dir",
        default=str(_default_install_directory()),
        help="checkout location (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("CODE_INDEXING_MCP_REPO_URL", DEFAULT_REPOSITORY_URL),
        help="Git repository to clone or update (default: %(default)s)",
    )
    parser.add_argument(
        "--harnesses",
        help=(
            "comma-separated harness numbers/slugs or 'all'; omit for the interactive menu "
            "(codex, claude-code, kimi-code, claude-desktop, opencode, kilocode)"
        ),
    )
    return parser


def _harness_label(slug: str) -> str:
    return next((choice.label for choice in HARNESS_CHOICES if choice.slug == slug), slug)


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)


def main(
    argv: list[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    error_fn: Callable[[str], None] = _print_error,
) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    install_directory = Path(arguments.install_dir).expanduser().resolve()

    try:
        action = clone_or_update_repository(arguments.repo_url, install_directory)
        output_fn(f"{action.title()} repository: {install_directory}")
        command = sync_environment(install_directory)
        output_fn(f"Prepared MCP executable: {command}")

        if arguments.harnesses is None:
            output_fn("Select the harnesses to configure:")
            for index, choice in enumerate(HARNESS_CHOICES, start=1):
                output_fn(f"  {index}. {choice.label}")
            selected = parse_harness_selection(
                input_fn("Enter comma-separated choices, 'all', or leave blank to skip: ")
            )
        else:
            selected = parse_harness_selection(arguments.harnesses)

        if not selected:
            output_fn("No harness configuration selected.")
            output_fn("Installation complete.")
            return 0

        successes, failures = configure_selected_harnesses(selected, command)
        for slug, path in successes:
            output_fn(f"Configured {_harness_label(slug)}: {path}")
        for slug, message in failures:
            error_fn(f"Failed to configure {_harness_label(slug)}: {message}")
        for slug, message in install_skills(selected, install_directory):
            output_fn(f"Skills for {_harness_label(slug)}: {message}")
        if failures:
            return 1
        output_fn("Installation complete. Restart configured clients to load the MCP server.")
        return 0
    except InstallerError as exc:
        error_fn(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        error_fn("Installation cancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
