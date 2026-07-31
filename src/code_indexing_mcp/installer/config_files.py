"""Comment-preserving JSON/JSONC/TOML configuration merging for harness setup."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SERVER_NAME = "code-indexing-mcp"


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


def _codex_server_block(command: Path, env: Mapping[str, str] | None = None) -> str:
    encoded_command = json.dumps(str(command), ensure_ascii=False)
    lines = [
        f"[mcp_servers.{SERVER_NAME}]",
        f"command = {encoded_command}",
        'args = ["serve"]',
    ]
    if env:
        pairs = ", ".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in sorted(env.items())
        )
        lines.append(f"env = {{ {pairs} }}")
    return "\n".join(lines) + "\n"


def _trailing_toml_trivia(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return "".join(lines[index + 1 :])
    return text


def merge_codex_server(path: Path, command: Path, *, env: Mapping[str, str] | None = None) -> bool:
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
    block = _codex_server_block(command, env)
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
