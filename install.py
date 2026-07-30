#!/usr/bin/env python3
"""Install or update Code Indexing MCP and configure supported MCP clients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

SERVER_NAME = "code-indexing-mcp"
DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git"

# The extra the serving environment always gets. It embeds queries in-process
# and is the fallback every accelerator degrades to, so it is never optional.
SERVING_EXTRA = "cpu"
# Accelerators this release can prepare, and the runtime extra each installs
# into an environment of its own. Core ML is still reached by explicit override
# inside the serving environment, so it needs no separate locked installation.
ACCELERATOR_EXTRAS = {
    "cuda": "cuda",
    "webgpu": "webgpu",
    "migraphx": "migraphx",
}
ACCELERATOR_CHOICES = ("auto", "cpu", "cuda", "webgpu", "migraphx", "coreml")
ACCELERATOR_ENVIRONMENT_DIRECTORY = ".venv-accel"
# Bumped in lockstep with incode_mcp.accelerator_env.RECORD_SCHEMA_VERSION.
ACCELERATOR_RECORD_SCHEMA_VERSION = 1
# A cold probe downloads the embedding model before it can run an inference, so
# this has to cover a slow link as well as a slow device.
PROBE_TIMEOUT_SECONDS = 900

# The pinned CUDA support window for this release. onnxruntime-gpu 1.22-1.23
# builds against CUDA 12.x and cuDNN 9, and NVIDIA's minor-version compatibility
# makes the driver below the floor for every 12.x runtime. A driver under it is
# reported and left alone: the installer never touches system drivers.
MINIMUM_NVIDIA_DRIVER = {"linux": (525, 60), "win32": (527, 41)}
# Lower-cased `platform.machine()` values, matching the `platform_machine`
# markers on the cuda extra exactly. A machine name the markers would miss must
# be refused here rather than nominated: `uv sync --extra cuda` would resolve
# that extra to nothing and build an environment with no embedding runtime in
# it at all, which fails the probe for a reason that explains nothing.
CUDA_PLATFORMS = {"linux": {"x86_64"}, "win32": {"amd64"}}
# The native WebGPU plugin/core pair's published wheels. The plugin's macOS
# wheel is universal2, but ONNX Runtime 1.24.4 itself is arm64-only there and
# has a deployment target of 14.0. Linux and Windows publish x86-64 wheels.
WEBGPU_PLATFORMS = {
    "darwin": {"arm64"},
    "linux": {"x86_64"},
    "win32": {"amd64"},
}
MINIMUM_WEBGPU_MACOS = (14, 0)
# AMD publishes this ONNX Runtime/MIGraphX combination as a single wheel rather
# than on PyPI. Nomination stays exact so the installer never assembles an
# untested Python/ROCm pair around it.
MIGRAPHX_PLATFORM = ("linux", "x86_64")
MIGRAPHX_PYTHON_VERSION = "3.12"
MIGRAPHX_ROCM_VERSION = "7.2.1"


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
    arguments: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=None if environment is None else {**os.environ, **environment},
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


def environment_python(directory: Path, *, platform_name: str | None = None) -> Path:
    """Return the interpreter inside a virtual environment directory."""

    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _uv_executable(uv_executable: str | None) -> str:
    uv = uv_executable or shutil.which("uv")
    if uv is None:
        raise InstallerError(
            "uv is required but was not found in PATH. Install it from https://docs.astral.sh/uv/"
        )
    return uv


def sync_environment(
    install_directory: Path,
    *,
    uv_executable: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Create or refresh the locked virtual environment and return its server command."""

    uv = _uv_executable(uv_executable)
    # The serving environment is pinned to the CPU extra. It is where queries
    # are embedded and where every accelerator failure lands, so it must never
    # depend on an accelerator runtime resolving.
    _run_command([uv, "sync", "--locked", "--extra", SERVING_EXTRA], cwd=install_directory)
    command = server_executable(install_directory, platform_name=platform_name)
    if not command.is_file():
        raise InstallerError(f"uv sync completed but the MCP executable is missing: {command}")
    return command


class AcceleratorPlan(NamedTuple):
    """What the installer will prepare, and the reason it settled on that.

    ``accelerator`` is ``"cpu"`` when nothing will be prepared, which is an
    outcome rather than an error: every machine indexes on CPU.
    """

    accelerator: str
    reason: str
    driver_version: str = ""
    device_name: str = ""
    # False when the request could not be honoured, which is what decides
    # whether the outcome is reported as a problem. A CPU result is not by
    # itself a denial: ``--accelerator cpu`` asked for exactly this, ``auto``
    # finding no GPU is just what the machine is, and Core ML needs nothing
    # prepared at all.
    honored: bool = True
    # Hash of the lockfile and selected extra. A record without this exact
    # fingerprint describes an older resolved runtime and must be rebuilt.
    lock_fingerprint: str = ""

    @property
    def prepares_environment(self) -> bool:
        return self.accelerator in ACCELERATOR_EXTRAS


def _nvidia_smi_report() -> str | None:
    """Return nvidia-smi's driver/name line, or None when there is no driver."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        # A driver too broken to answer is a driver this installer will not
        # build on top of, and not a reason to fail the whole installation.
        return None
    return result.stdout if result.returncode == 0 else None


def _rocm_report() -> str | None:
    """Return the installed ROCm version and, when available, an AMD device."""

    version = ""
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/.info/version-dev")):
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"\d+\.\d+(?:\.\d+)?", contents)
        if match is not None:
            version = match.group()
            break
    if not version:
        return None

    device = ""
    executable = shutil.which("rocminfo")
    if executable is not None:
        try:
            result = subprocess.run(
                [executable],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            match = re.search(r"(?m)^\s*Marketing Name:\s*(.+?)\s*$", result.stdout)
            if match is not None and match.group(1).strip().lower() != "unknown":
                device = match.group(1).strip()
    return f"{version}, {device or 'AMD GPU'}"


def _driver_components(version: str) -> tuple[int, ...]:
    components: list[int] = []
    for part in version.strip().split("."):
        if not part.isdigit():
            break
        components.append(int(part))
    return tuple(components)


def _normalized_platform(platform_name: str) -> str:
    return "win32" if platform_name.startswith("win") else platform_name


def _webgpu_plan(
    *,
    platform_name: str,
    machine: str,
    platform_version: str,
    reason_prefix: str = "",
) -> AcceleratorPlan:
    supported = WEBGPU_PLATFORMS.get(platform_name)
    problem = ""
    if supported is None or machine not in supported:
        problem = f"no native WebGPU plugin wheel is published for {platform_name}/{machine}"
    elif platform_name == "darwin":
        components = _driver_components(platform_version)
        if not components or components < MINIMUM_WEBGPU_MACOS:
            problem = (
                f"the locked WebGPU plugin requires macOS "
                f"{'.'.join(str(part) for part in MINIMUM_WEBGPU_MACOS)} or newer"
            )

    if problem:
        prefix = f"{reason_prefix}; " if reason_prefix else "WebGPU was requested but "
        return AcceleratorPlan("cpu", f"{prefix}{problem}", honored=False)
    reason = (
        f"the locked WebGPU plugin is available for {platform_name}/{machine}"
        if not reason_prefix
        else f"{reason_prefix}; falling back to WebGPU with the locked plugin"
    )
    return AcceleratorPlan("webgpu", reason, honored=not reason_prefix)


def plan_accelerator(
    requested: str,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    nvidia_report: Callable[[], str | None] = _nvidia_smi_report,
    rocm_report: Callable[[], str | None] = _rocm_report,
    python_version: str | None = None,
    platform_version: str | None = None,
) -> AcceleratorPlan:
    """Decide which accelerator, if any, this machine should have prepared.

    Detection only nominates: the environment still has to build and pass a real
    inference probe before anything offers the backend to the server.
    """

    platform_name = _normalized_platform((platform_name or sys.platform).lower())
    machine = (machine or platform.machine()).lower()
    python_version = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    if platform_version is None:
        platform_version = platform.mac_ver()[0] if platform_name == "darwin" else ""
    requested = requested.strip().lower()

    if requested == "cpu":
        return AcceleratorPlan("cpu", "CPU was requested")
    if requested == "coreml":
        # Not a denial: Core ML runs in the serving environment's own runtime,
        # so there is genuinely nothing for this installer to prepare.
        return AcceleratorPlan(
            "cpu",
            "Core ML needs no separate environment and stays manual-only: it lost to "
            "CPU on this model. Set INCODE_EMBED_ACCELERATOR=coreml to measure it",
        )
    if requested == "webgpu":
        return _webgpu_plan(
            platform_name=platform_name,
            machine=machine,
            platform_version=platform_version,
        )
    if requested == "migraphx":
        problem = ""
        if (platform_name, machine) != MIGRAPHX_PLATFORM:
            problem = (
                f"the pinned MIGraphX wheel is published only for "
                f"{MIGRAPHX_PLATFORM[0]}/{MIGRAPHX_PLATFORM[1]}"
            )
        elif python_version != MIGRAPHX_PYTHON_VERSION:
            problem = (
                f"the pinned MIGraphX wheel requires Python {MIGRAPHX_PYTHON_VERSION}, "
                f"not {python_version}"
            )
        else:
            report = rocm_report()
            if not report or not report.strip():
                problem = "ROCm was not detected"
            else:
                first = report.strip().splitlines()[0]
                rocm_version, _, device_name = (part.strip() for part in first.partition(","))
                if rocm_version != MIGRAPHX_ROCM_VERSION:
                    problem = (
                        f"ROCm {rocm_version or 'unknown'} does not match the pinned "
                        f"{MIGRAPHX_ROCM_VERSION} runtime"
                    )
                else:
                    return AcceleratorPlan(
                        "migraphx",
                        f"ROCm {rocm_version} on {device_name or 'an AMD device'} matches "
                        "the pinned MIGraphX runtime",
                        driver_version=rocm_version,
                        device_name=device_name,
                    )
        return _webgpu_plan(
            platform_name=platform_name,
            machine=machine,
            platform_version=platform_version,
            reason_prefix=f"MIGraphX was requested but {problem}",
        )

    supported = CUDA_PLATFORMS.get(platform_name)
    # `auto` finding no CUDA is what most machines are, not a request denied.
    explicit = "CUDA was requested but " if requested == "cuda" else ""
    honored = not explicit
    if supported is None or machine not in supported:
        return AcceleratorPlan(
            "cpu",
            f"{explicit}no CUDA wheels are published for {platform_name}/{machine}",
            honored=honored,
        )
    report = nvidia_report()
    if not report or not report.strip():
        return AcceleratorPlan(
            "cpu",
            f"{explicit}no usable NVIDIA driver was detected (nvidia-smi reported nothing)",
            honored=honored,
        )
    first = report.strip().splitlines()[0]
    driver_version, _, device_name = (part.strip() for part in first.partition(","))
    floor = MINIMUM_NVIDIA_DRIVER[platform_name]
    components = _driver_components(driver_version)
    if not components or components < floor:
        return AcceleratorPlan(
            "cpu",
            f"{explicit}NVIDIA driver {driver_version or 'unknown'} is below the "
            f"{'.'.join(str(part) for part in floor)} this release's CUDA 12 runtime "
            "needs; the installer does not change drivers",
            driver_version=driver_version,
            device_name=device_name,
            honored=honored,
        )
    return AcceleratorPlan(
        "cuda",
        f"NVIDIA driver {driver_version} on {device_name or 'an NVIDIA device'} "
        "satisfies the pinned CUDA 12 runtime",
        driver_version=driver_version,
        device_name=device_name,
    )


def interpreter_version(python: Path) -> str:
    """Return the ``major.minor`` version of an interpreter."""

    result = _run_command([str(python), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"])
    return result.stdout.strip()


def accelerator_lock_fingerprint(install_directory: Path, accelerator: str) -> str:
    """Hash the selected extra and the lockfile that resolved its environment."""

    lockfile = install_directory / "uv.lock"
    try:
        locked = lockfile.read_bytes()
    except OSError as exc:
        raise InstallerError(
            f"The accelerator lockfile cannot be read at {lockfile}: {exc}"
        ) from exc
    digest = hashlib.sha256()
    digest.update(accelerator.encode())
    digest.update(b"\0")
    digest.update(locked)
    return digest.hexdigest()


def runtime_record_path(python: Path) -> Path:
    """Ask the installed package where the server reads its accelerator record.

    The package is asked rather than told: it owns the filename, and it honours
    an ``INCODE_ACCEL_ENV`` override that a path assembled here would write
    straight past, leaving the record somewhere the server never looks.
    """

    result = _run_command(
        [
            str(python),
            "-c",
            "from incode_mcp.accelerator_env import record_path;"
            "from incode_mcp.application import RuntimePaths;"
            "print(record_path(RuntimePaths.from_environment().data))",
        ]
    )
    return Path(result.stdout.strip())


def accelerator_record_path(install_directory: Path, *, platform_name: str | None = None) -> Path:
    python = environment_python(install_directory / ".venv", platform_name=platform_name)
    return runtime_record_path(python)


def sync_accelerator_environment(
    install_directory: Path,
    extra: str,
    *,
    python_version: str,
    uv_executable: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Build the accelerator's own locked environment and return its interpreter.

    Whenever this runs it builds from empty, never over what is already there:
    an environment carrying leftovers from an earlier extra would resolve its
    ONNX Runtime to whichever distribution landed last, which is the exact
    failure the extras are separated to avoid. Deciding whether it needs to run
    at all is the caller's job -- see ``reusable_accelerator_environment``.
    """

    uv = _uv_executable(uv_executable)
    directory = install_directory / ACCELERATOR_ENVIRONMENT_DIRECTORY
    if directory.exists():
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            # Not ignore_errors: building over a half-removed environment is the
            # ONNX Runtime collision this function exists to prevent, so the
            # removal has to succeed or the build has to stop. Stopping is said
            # in the installer's own vocabulary, though -- the caller degrades to
            # CPU on an InstallerError, and a raw OSError from a file the machine
            # merely had locked would take the whole installation down instead.
            raise InstallerError(
                f"Could not remove the existing accelerator environment at {directory}: {exc}"
            ) from exc
    _run_command(
        [
            uv,
            "sync",
            "--locked",
            "--no-default-groups",
            "--extra",
            extra,
            # Both ends of the worker channel speak multiprocessing's connection
            # protocol, so the accelerator interpreter has to match the server's.
            "--python",
            python_version,
        ],
        cwd=install_directory,
        environment={"UV_PROJECT_ENVIRONMENT": str(directory)},
    )
    python = environment_python(directory, platform_name=platform_name)
    if not python.is_file():
        raise InstallerError(f"The accelerator environment has no interpreter at {python}")
    return python


def probe_accelerator(python: Path, accelerator: str, *, offline: bool = False) -> dict[str, Any]:
    """Run a real inference in the accelerator environment and return its report."""

    arguments = [str(python), "-m", "incode_mcp.accelerator_probe", "--accelerator", accelerator]
    if offline:
        arguments.append("--offline")
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        # Generous, because a cold probe downloads the model before it can embed
        # anything -- but bounded, because a driver that wedges initialising a
        # device wedges there forever, and the output is captured, so an
        # unbounded wait would look exactly like an installer that had hung.
        raise InstallerError(
            f"The accelerator probe did not finish within {PROBE_TIMEOUT_SECONDS // 60} minutes"
        ) from exc
    except OSError as exc:
        raise InstallerError(f"Could not run the accelerator probe: {exc}") from exc
    payload: Any = None
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    if not isinstance(payload, dict):
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise InstallerError(
            "The accelerator probe returned no report"
            + (f": {detail[-1]}" if detail else f" (exit status {result.returncode})")
        )
    if not payload.get("ok"):
        raise InstallerError(f"The accelerator probe failed: {payload.get('error', 'unknown')}")
    return payload


def write_accelerator_record(path: Path, plan: AcceleratorPlan, probe: Mapping[str, Any]) -> None:
    """Record the verified environment where the server looks for one.

    The shape is read back by ``incode_mcp.accelerator_env``; the schema version
    is what keeps a record written here from being misread by a server that
    changed its mind about what these fields mean.
    """

    record = {
        "schema_version": ACCELERATOR_RECORD_SCHEMA_VERSION,
        "accelerator": plan.accelerator,
        "interpreter": str(probe["interpreter"]),
        "providers": list(probe["providers"]),
        "runtime_version": str(probe.get("runtime_version", "")),
        "lock_fingerprint": plan.lock_fingerprint,
        "driver_version": plan.driver_version,
        "device": str(probe.get("device", "")),
        "python_version": str(probe.get("python_version", "")),
        "recorded_at_ns": time.time_ns(),
        "detail": str(probe.get("detail", "")),
    }
    _atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")


def clear_accelerator_record(path: Path) -> bool:
    """Drop any record, so an installation that fell back stops offering more."""

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise InstallerError(
            f"Could not remove the stale accelerator record: {path}: {exc}"
        ) from exc
    return True


def reusable_accelerator_environment(
    path: Path, plan: AcceleratorPlan, *, python_version: str
) -> Path | None:
    """Return the interpreter an existing record still vouches for, if any.

    Rebuilding a multi-gigabyte environment and re-probing a device on every
    update is a lot of work to arrive back where the last run already was. The
    record is reused only when it describes this exact plan running on this
    exact interpreter; anything that moved -- the driver, the server's Python,
    the environment itself -- puts the full build and probe back.
    """

    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    interpreter = Path(str(record.get("interpreter", "")))
    matches = (
        record.get("schema_version") == ACCELERATOR_RECORD_SCHEMA_VERSION
        and record.get("accelerator") == plan.accelerator
        and str(record.get("lock_fingerprint", "")) == plan.lock_fingerprint
        and str(record.get("driver_version", "")) == plan.driver_version
        and str(record.get("python_version", "")) == python_version
        and interpreter.is_file()
    )
    return interpreter if matches else None


def configure_accelerator(
    install_directory: Path,
    requested: str,
    *,
    uv_executable: str | None = None,
    platform_name: str | None = None,
    machine: str | None = None,
    nvidia_report: Callable[[], str | None] = _nvidia_smi_report,
    rocm_report: Callable[[], str | None] = _rocm_report,
    python_version: str | None = None,
    platform_version: str | None = None,
    offline: bool = False,
) -> AcceleratorPlan:
    """Prepare the planned accelerator, or leave the installation on CPU.

    Every failure below is a fall back to CPU with the reason attached, not an
    installation failure: an accelerator that cannot be built or cannot pass its
    probe costs speed, and refusing to install over it would cost the server.
    """

    serving_python = environment_python(install_directory / ".venv", platform_name=platform_name)
    detected_python_version: str | None = None
    planning_python_version = python_version
    planning_error: InstallerError | None = None
    if requested.strip().lower() == "migraphx" and planning_python_version is None:
        try:
            detected_python_version = interpreter_version(serving_python)
            planning_python_version = detected_python_version
        except InstallerError as exc:
            planning_error = exc
    if planning_error is None:
        plan = plan_accelerator(
            requested,
            platform_name=platform_name,
            machine=machine,
            nvidia_report=nvidia_report,
            rocm_report=rocm_report,
            python_version=planning_python_version,
            platform_version=platform_version,
        )
    else:
        plan = AcceleratorPlan(
            "cpu",
            f"MIGraphX was requested but the serving Python version could not be resolved: "
            f"{planning_error}",
            honored=False,
        )
    try:
        record = accelerator_record_path(install_directory, platform_name=platform_name)
    except InstallerError as exc:
        # Without the server's data directory there is nowhere to offer an
        # accelerator from, and nowhere a stale offer could be retracted from
        # either. Reporting that is the whole of what can be done about it; the
        # installation itself is fine and indexes on CPU.
        return AcceleratorPlan(
            "cpu",
            f"the server's runtime data directory could not be resolved: {exc}",
            honored=False,
        )
    if not plan.prepares_environment:
        clear_accelerator_record(record)
        # Once no record points at it, the environment is several gigabytes of
        # dead weight. A machine reinstalled as CPU-only should not keep paying
        # the disk for the GPU it used to have.
        shutil.rmtree(install_directory / ACCELERATOR_ENVIRONMENT_DIRECTORY, ignore_errors=True)
        return plan

    try:
        serving_python_version = detected_python_version or interpreter_version(serving_python)
        plan = plan._replace(
            lock_fingerprint=accelerator_lock_fingerprint(
                install_directory,
                ACCELERATOR_EXTRAS[plan.accelerator],
            )
        )
        reused = reusable_accelerator_environment(
            record,
            plan,
            python_version=serving_python_version,
        )
        if reused is not None:
            return plan._replace(reason=f"{plan.reason}; reusing the environment at {reused}")
        python = sync_accelerator_environment(
            install_directory,
            ACCELERATOR_EXTRAS[plan.accelerator],
            python_version=serving_python_version,
            uv_executable=uv_executable,
            platform_name=platform_name,
        )
        probe = probe_accelerator(python, plan.accelerator, offline=offline)
    except InstallerError as exc:
        # Nothing half-built may be left where the server could find it: the
        # record is what makes an environment reachable at all.
        clear_accelerator_record(record)
        shutil.rmtree(install_directory / ACCELERATOR_ENVIRONMENT_DIRECTORY, ignore_errors=True)
        return AcceleratorPlan(
            "cpu",
            f"{plan.accelerator} was detected but could not be prepared: {exc}",
            driver_version=plan.driver_version,
            device_name=plan.device_name,
            # Detection said this machine has the hardware and it still did not
            # come up. That is worth reporting however it was requested.
            honored=False,
        )
    write_accelerator_record(record, plan, probe)
    return plan._replace(reason=f"{plan.reason}; {probe.get('detail', 'probe passed')}")


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
    return skills_dir.name == "skills" and skills_dir.parent.name == "incode_mcp"


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
        "--accelerator",
        choices=ACCELERATOR_CHOICES,
        default=os.environ.get("CODE_INDEXING_MCP_ACCELERATOR", "auto"),
        help=(
            "which accelerator to prepare for passage indexing (default: %(default)s). "
            "auto detects one; anything that cannot be detected, built, or probed "
            "falls back to CPU with the reason reported"
        ),
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

        accelerator = configure_accelerator(
            install_directory,
            arguments.accelerator,
            offline=os.environ.get("INCODE_OFFLINE", "").lower() in {"1", "true", "yes"},
        )
        report = f"Passage embedding accelerator: {accelerator.accelerator} ({accelerator.reason})"
        # A request that could not be honoured is reported as a problem even
        # though the installation succeeded on CPU. The plan decides which is
        # which: landing on CPU is not by itself a denial.
        if accelerator.honored:
            output_fn(report)
        else:
            error_fn(report)

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
