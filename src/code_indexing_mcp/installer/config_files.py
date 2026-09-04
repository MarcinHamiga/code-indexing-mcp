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
from typing import Any, NamedTuple

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


class JsonMember(NamedTuple):
    """One object member's key and the byte span of its value.

    ``key_start`` is what removal needs and merging does not: to take a member
    out, the quote that opens its key is the left edge of the span to cut.
    """

    key: str
    key_start: int
    value_start: int
    value_end: int


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
        key_start = position
        key, position = _parse_json_string(text, position)
        position = _skip_jsonc_trivia(text, position)
        if position >= len(text) or text[position] != ":":
            raise ValueError(f"missing colon after {key!r}")
        value_start = _skip_jsonc_trivia(text, position + 1)
        value_end = _scan_jsonc_value(text, value_start)
        members.append(JsonMember(key, key_start, value_start, value_end))
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
        last_value_end = members[-1].value_end
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

    root_member = next((member for member in root_members if member.key == object_key), None)
    if root_member is None:
        return _insert_jsonc_member(
            text,
            root_start,
            root_end,
            root_members,
            object_key,
            {entry_key: entry_value},
        )

    object_start, object_value_end = root_member.value_start, root_member.value_end
    if text[object_start] != "{":
        raise ValueError(f"{object_key!r} must contain an object")
    entries, object_end = _jsonc_object_members(text, object_start)
    if object_end + 1 != object_value_end:
        raise ValueError(f"{object_key!r} has an invalid object value")
    entry = next((member for member in entries if member.key == entry_key), None)
    if entry is None:
        return _insert_jsonc_member(
            text,
            object_start,
            object_end,
            entries,
            entry_key,
            entry_value,
        )

    replacement = _format_json_value(entry_value, _line_indent(text, entry.value_start))
    return text[: entry.value_start] + replacement + text[entry.value_end :]


def _remove_jsonc_member(text: str, members: list[JsonMember], index: int) -> str:
    """Cut one member out along with the comma that joined it to its neighbours.

    Deleting the span alone would leave either a double comma or a dangling one,
    so exactly one separator has to go with it: the comma that follows, or -- for
    the last member -- the comma that precedes it.
    """

    member = members[index]
    start, end = member.key_start, member.value_end
    after = _skip_jsonc_trivia(text, end)
    if after < len(text) and text[after] == ",":
        end = after + 1
    elif index > 0:
        previous_end = members[index - 1].value_end
        comma = _skip_jsonc_trivia(text, previous_end)
        if comma < len(text) and text[comma] == ",":
            start = comma

    # Take the whole line when the member had one to itself, so removal does not
    # leave a blank line where an entry used to be. A comment sharing either end
    # of the line is left exactly where the user put it.
    line_start = text.rfind("\n", 0, start) + 1
    if not text[line_start:start].strip():
        newline = text.find("\n", end)
        # A member on the final line has no newline after it; the line still
        # belongs to it, and leaving it behind is the blank line this avoids.
        line_end = len(text) if newline == -1 else newline + 1
        if not text[end:line_end].strip():
            start, end = line_start, line_end
    return text[:start] + text[end:]


def _remove_jsonc_entry(text: str, object_key: str, entry_key: str) -> str | None:
    """Return the text without the named entry, or None when it was not there."""

    root_start = _skip_jsonc_trivia(text, 0)
    if root_start >= len(text) or text[root_start] != "{":
        raise ValueError("configuration root must be an object")
    root_members, _ = _jsonc_object_members(text, root_start)
    root_member = next((member for member in root_members if member.key == object_key), None)
    if root_member is None:
        return None
    if text[root_member.value_start] != "{":
        raise ValueError(f"{object_key!r} must contain an object")
    entries, _ = _jsonc_object_members(text, root_member.value_start)
    index = next(
        (position for position, member in enumerate(entries) if member.key == entry_key),
        None,
    )
    if index is None:
        return None
    updated = _remove_jsonc_member(text, entries, index)
    if len(entries) > 1:
        return updated
    # That was the only server, so the container is now empty. Take it too: the
    # merge is what created it, and putting the file back exactly as it was is
    # worth more than preserving a `"mcpServers": {}` that means nothing.
    root_start = _skip_jsonc_trivia(updated, 0)
    root_members, _ = _jsonc_object_members(updated, root_start)
    container = next(
        (position for position, member in enumerate(root_members) if member.key == object_key),
        None,
    )
    if container is None:  # pragma: no cover - it was there a moment ago
        return updated
    remaining, _ = _jsonc_object_members(updated, root_members[container].value_start)
    if remaining:
        return updated
    return _remove_jsonc_member(updated, root_members, container)


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


def _backup_configuration(path: Path) -> None:
    """Keep a copy of ``path`` before overwriting it, without losing the first one.

    ``.bak`` is the file as it was before this installer ever touched it, which
    is the copy worth keeping: a user restoring a shell profile wants what they
    wrote, not what a previous run of ours left. Later writes roll into a second
    slot instead, so the pair stays bounded however often configure is re-run.
    """

    pristine = path.with_name(f"{path.name}.bak")
    if pristine.is_symlink() or pristine.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.bak.prev"))
    else:
        shutil.copy2(path, pristine)


def write_changed_configuration(path: Path, original: str | None, updated: str) -> bool:
    """Write ``updated`` unless it matches ``original``. True when the file changed."""

    if original == updated:
        return False
    if original is not None:
        _backup_configuration(path)
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
    return write_changed_configuration(path, original, updated)


MUSE_CODE_SCHEMA_VERSION = 1


def merge_muse_code_entry(path: Path, entry_key: str, entry_value: Any) -> bool:
    """Merge one entry into a Muse Code settings document.

    Identical to ``merge_json_object_entry`` under the ``"mcpServers"`` key,
    except a missing ``schema_version`` is added alongside it: Muse Code
    rejects a settings document without one, so a file this installer creates
    from scratch must already carry it. A version already present is left
    exactly as it is.
    """

    original = _read_configuration(path)
    source = original if original and original.strip() else "{}\n"
    try:
        _validate_jsonc(source)
        updated = _merge_jsonc_text(source, "mcpServers", entry_key, entry_value)
        root_start = _skip_jsonc_trivia(updated, 0)
        members, root_end = _jsonc_object_members(updated, root_start)
        if not any(member.key == "schema_version" for member in members):
            updated = _insert_jsonc_member(
                updated,
                root_start,
                root_end,
                members,
                "schema_version",
                MUSE_CODE_SCHEMA_VERSION,
            )
        _validate_jsonc(updated)
    except ValueError as exc:
        raise InstallerError(f"Invalid JSON/JSONC configuration in {path}: {exc}") from exc
    return write_changed_configuration(path, original, updated)


def remove_json_object_entry(path: Path, object_key: str, entry_key: str) -> bool:
    """Remove one entry from a top-level JSON/JSONC object. False if it was absent."""

    original = _read_configuration(path)
    if original is None or not original.strip():
        return False
    try:
        _validate_jsonc(original)
        updated = _remove_jsonc_entry(original, object_key, entry_key)
        if updated is None:
            return False
        _validate_jsonc(updated)
    except ValueError as exc:
        raise InstallerError(f"Invalid JSON/JSONC configuration in {path}: {exc}") from exc
    return write_changed_configuration(path, original, updated)


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


TomlHeading = tuple["re.Match[str]", list[str], bool]


def _codex_headings(source: str, path: Path) -> list[TomlHeading]:
    headings: list[TomlHeading] = []
    for match in _TOML_TABLE.finditer(source):
        try:
            array_name = match.group("array_name")
            name = array_name or match.group("table_name")
            headings.append((match, _split_toml_dotted_key(name), array_name is not None))
        except ValueError as exc:
            raise InstallerError(f"Invalid TOML table in {path}: {exc}") from exc
    return headings


def _codex_target_span(source: str, headings: list[TomlHeading], index: int) -> tuple[int, int]:
    """The byte span of the server table and every subtable beneath it."""

    target = ["mcp_servers", SERVER_NAME]
    start = headings[index][0].start()
    end = len(source)
    for match, components, _ in headings[index + 1 :]:
        if components[: len(target)] != target:
            end = match.start()
            break
    return start, end


def _codex_target_index(headings: list[TomlHeading]) -> int | None:
    target = ["mcp_servers", SERVER_NAME]
    return next(
        (
            index
            for index, (_, components, is_array) in enumerate(headings)
            if components == target and not is_array
        ),
        None,
    )


def remove_codex_server(path: Path) -> bool:
    """Remove the Code Indexing MCP table from a Codex config. False if absent."""

    original = _read_configuration(path)
    if original is None or not original.strip():
        return False
    try:
        tomllib.loads(original)
    except tomllib.TOMLDecodeError as exc:
        raise InstallerError(f"Invalid TOML configuration in {path}: {exc}") from exc
    headings = _codex_headings(original, path)
    index = _codex_target_index(headings)
    if index is None:
        return False
    start, end = _codex_target_span(original, headings, index)
    # Comments trailing the table were not written by the installer, so they stay.
    trailing_trivia = _trailing_toml_trivia(original[start:end])
    prefix = original[:start]
    if not prefix.strip():
        # Nothing but whitespace came before the table, so there is nothing to keep.
        prefix = ""
    elif prefix.endswith("\n\n"):
        # The merge separated the table from what came before it with exactly one
        # blank line. Take that one line back and no more: collapsing the whole
        # run would eat the spacing a user put between their own tables when our
        # table sits in the middle of the file rather than at the end of it.
        prefix = prefix[:-1]
    updated = prefix + trailing_trivia + original[end:]
    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise InstallerError(
            f"Refusing to write invalid TOML configuration to {path}: {exc}"
        ) from exc
    return write_changed_configuration(path, original, updated)


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

    headings = _codex_headings(source, path)
    target_index = _codex_target_index(headings)
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
        start, end = _codex_target_span(source, headings, target_index)
        trailing_trivia = _trailing_toml_trivia(source[start:end])
        updated = source[:start] + block + trailing_trivia + source[end:]

    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as exc:
        raise InstallerError(
            f"Refusing to write invalid TOML configuration to {path}: {exc}"
        ) from exc
    return write_changed_configuration(path, original, updated)
