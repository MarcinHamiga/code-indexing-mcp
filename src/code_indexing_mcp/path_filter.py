"""Gitignore-style path globs for the ``paths`` filter of the search tools.

One pattern string is parsed once into one regular expression, and that single
expression is used both for the LanceDB pushdown predicate and for the Python
post-filter, so the two can never disagree. Pushing the filter into the scan
matters because applying it only to rows the database already truncated would
make a match that ranks below the fetch window indistinguishable from "no such
code exists".

Semantics (paths are project-relative and use ``/``):

* ``*`` matches any run of characters except ``/``; ``?`` matches one character
  other than ``/``; ``[abc]``, ``[a-z]`` and ``[!abc]`` are character classes. A
  ``[`` with no closing ``]`` is an ordinary character.
* ``**`` as a whole segment matches zero or more directories: ``**/*.py``
  includes a top-level ``top.py``, and ``a/**/b`` matches ``a/b`` and ``a/x/y/b``.
  A trailing ``/**`` matches everything under the directory (``src/**``).
  Inside a segment ``**`` is the same as ``*``.
* A pattern with no wildcard names a path: it matches a file with exactly that
  path and, if it names a directory, everything beneath it (``src/pkg`` matches
  ``src/pkg/a.py``). Patterns containing a wildcard match whole paths only.
* A pattern containing ``/`` is anchored at the project root; a leading ``**/``
  is the one way to lift that. A pattern with no ``/`` (``*.py``, ``conftest.py``,
  ``tests``) matches at any depth. A trailing ``/`` is ignored, as in gitignore.
* Empty patterns, absolute patterns (leading ``/``) and ``..`` segments are
  rejected with :class:`PathPatternError`.

The module is pure so the semantics are testable without a database. The SQL
predicate additionally carries a literal-prefix conjunct (``starts_with``) when
an anchored pattern begins with literal text, so the ``path`` index prunes before
the regular expression runs. The conjunct is derived from the same parse, and is
only ever a necessary condition of the regex, never narrower than it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


class PathPatternError(ValueError):
    """A ``paths`` pattern that cannot be used."""


# Characters that must be escaped outside a class, and inside one. Both Python's
# ``re`` and the Rust regex engine behind LanceDB accept a backslash before each.
_REGEX_META = frozenset("\\.+*?()|[]{}^$#&-~")
_CLASS_META = frozenset("\\[]^-&~|")
_ANY = r"[\s\S]"
_ANY_DIRECTORIES = r"(?:[^/]+/)*"


def _escape(character: str) -> str:
    return "\\" + character if character in _REGEX_META else character


def _escape_in_class(character: str) -> str:
    return "\\" + character if character in _CLASS_META else character


def _character_class(segment: str, cursor: int) -> tuple[str, int] | None:
    """Translate the class opening at ``segment[cursor]``, or ``None`` if unclosed."""
    search = cursor + 1
    negated = segment[search : search + 1] == "!"
    if negated:
        search += 1
    start = search
    # A leading ] is a class member, not the closing delimiter.
    if segment[search : search + 1] == "]":
        search += 1
    closing = segment.find("]", search)
    if closing == -1:
        return None
    body = segment[start:closing]
    members: list[str] = []
    index = 0
    while index < len(body):
        if index + 2 < len(body) and body[index + 1] == "-":
            low, high = body[index], body[index + 2]
            if low > high:
                raise PathPatternError(f"reversed range {low}-{high} in character class")
            members.append(f"{_escape_in_class(low)}-{_escape_in_class(high)}")
            index += 3
        else:
            members.append(_escape_in_class(body[index]))
            index += 1
    # Segments never contain "/", but a negated class must not be the way one appears.
    prefix = "^/" if negated else ""
    return f"[{prefix}{''.join(members)}]", closing + 1


def _segment(segment: str) -> tuple[str, str, bool]:
    """Return (regex, leading literal text, is_literal) for one non-``**`` segment."""
    parts: list[str] = []
    literal: list[str] = []
    leading = True
    wildcard = False
    cursor = 0
    while cursor < len(segment):
        character = segment[cursor]
        if character == "*":
            while segment[cursor : cursor + 1] == "*":
                cursor += 1
            parts.append("[^/]*")
            wildcard, leading = True, False
        elif character == "?":
            parts.append("[^/]")
            cursor += 1
            wildcard, leading = True, False
        elif character == "[" and (translated := _character_class(segment, cursor)):
            expression, cursor = translated
            parts.append(expression)
            wildcard, leading = True, False
        else:
            parts.append(_escape(character))
            if leading:
                literal.append(character)
            cursor += 1
    return "".join(parts), "".join(literal), not wildcard


@dataclass(frozen=True)
class PathPattern:
    """One parsed ``paths`` pattern.

    ``regex`` is written to match a whole path; use :meth:`matches` in Python and
    the same text under ``regexp_like`` in SQL.
    """

    source: str
    regex: str
    literal_prefix: str
    _compiled: re.Pattern[str]

    def matches(self, path: str) -> bool:
        return self._compiled.fullmatch(path) is not None


def compile_path_pattern(pattern: str) -> PathPattern:
    if not pattern:
        raise PathPatternError("path pattern must not be empty")
    if pattern.startswith("/"):
        raise PathPatternError(
            f"path pattern {pattern!r} must be relative to the project root, not absolute"
        )
    segments = [part for part in pattern.split("/") if part not in ("", ".")]
    if not segments:
        raise PathPatternError(f"path pattern {pattern!r} names no path")
    if ".." in segments:
        raise PathPatternError(f"path pattern {pattern!r} must not contain '..'")

    floating = len(segments) == 1
    while segments and segments[0] == "**":
        floating = True
        segments.pop(0)
    collapsed: list[str] = []
    for segment in segments:
        if segment != "**" or not collapsed or collapsed[-1] != "**":
            collapsed.append(segment)
    segments = collapsed

    if not segments:
        regex = f"^{_ANY}*$"
        return PathPattern(pattern, regex, "", re.compile(regex))

    body: list[str] = []
    prefix: list[str] = []
    prefix_open = not floating
    all_literal = True
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            body.append(f"{_ANY}+" if last else _ANY_DIRECTORIES)
            all_literal = False
            prefix_open = False
            continue
        expression, literal, is_literal = _segment(segment)
        body.append(expression if last else expression + "/")
        all_literal = all_literal and is_literal
        if prefix_open:
            prefix.append(literal)
            if is_literal and not last:
                prefix.append("/")
            else:
                prefix_open = False

    regex = "^" + ("" if not floating else f"(?:{_ANY}*/)?") + "".join(body)
    if all_literal:
        regex += f"(?:/{_ANY}*)?"
    regex += "$"
    return PathPattern(pattern, regex, "".join(prefix), re.compile(regex))


def compile_path_patterns(patterns: Iterable[str]) -> list[PathPattern]:
    return [compile_path_pattern(pattern) for pattern in patterns]


def path_matches(path: str, patterns: Sequence[PathPattern]) -> bool:
    """Whether *path* matches any pattern; no patterns means no filter."""
    return not patterns or any(pattern.matches(path) for pattern in patterns)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def path_condition(patterns: Sequence[PathPattern], *, column: str = "path") -> str | None:
    """Return a SQL predicate matching *column* against any of *patterns*.

    ``None`` means there is nothing to filter. The regex is always the full
    predicate; a literal prefix is added as a leading conjunct only so the scan
    can prune on the ``path`` index first.
    """
    if not patterns:
        return None
    expressions: list[str] = []
    for pattern in patterns:
        expression = f"regexp_like({column}, {_quote(pattern.regex)})"
        if pattern.literal_prefix:
            expression = (
                f"(starts_with({column}, {_quote(pattern.literal_prefix)}) AND {expression})"
            )
        expressions.append(expression)
    return "(" + " OR ".join(expressions) + ")"
