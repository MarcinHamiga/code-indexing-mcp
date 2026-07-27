"""Translate search path globs into a LanceDB pushdown predicate.

``search_code`` filters paths with ``PurePosixPath.match``, which is the authority
on what matches. Applying it only in Python means it runs on rows the database has
already truncated, so a match that ranks below the fetch window disappears and is
indistinguishable from "no such code exists". These translations let the same
semantics be pushed into the scan, where they narrow the rows instead of discarding
them.

Everything here is pure so the equivalence with ``PurePosixPath.match`` is testable
without a database. The one rule that matters: the predicate may be equivalent or
broader, never narrower. Broader only costs rows the post-filter then drops;
narrower loses results, which is the bug this exists to fix.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePosixPath


def _character_class(pattern: str, cursor: int) -> tuple[str, int] | None:
    """Translate one conservative fnmatch character class.

    Returning ``None`` is always safe: the caller then skips pushdown and lets
    ``PurePosixPath.match`` remain the authority. Ranges and regex set-operation
    characters are deliberately left to that fallback because Python's fnmatch
    rules and LanceDB's regex engine do not share identical class syntax.
    """
    search = cursor + 1
    if search < len(pattern) and pattern[search] == "!":
        search += 1
    # A leading ] is a class member in fnmatch, not the closing delimiter.
    if search < len(pattern) and pattern[search] == "]":
        search += 1
    closing = pattern.find("]", search)
    if closing == -1:
        return None

    body = pattern[cursor + 1 : closing]
    negated = body.startswith("!")
    members = body[1:] if negated else body
    if not members or any(character in members for character in r"-\/&~|"):
        return None

    escaped = "".join(
        f"\\{character}" if character in "[]^" else character for character in members
    )
    prefix = "^" if negated else ""
    return f"[{prefix}{escaped}]", closing + 1


def glob_to_regex(pattern: str) -> str | None:
    """Return a regex equivalent to ``PurePosixPath(path).match(pattern)``.

    Returns ``None`` when the pattern cannot be translated with confidence, which
    tells the caller to skip the pushdown rather than risk a narrower predicate.

    Two properties of ``PurePosixPath.match`` drive the output:

    * Relative patterns match **from the right**, so ``*.py`` matches ``a/b/c.py``.
      Hence the ``(^|/)`` prefix and ``$`` suffix rather than a leading ``^``.
    * On Python 3.12 ``**`` spans exactly one segment, the same as ``*``
      (``PurePosixPath("src/deep/b.py").match("src/**")`` is ``False``), so runs of
      asterisks collapse to a single ``[^/]*``. Recursive matching lives in 3.13's
      separate ``full_match`` and is not what this mirrors.
    """
    if not pattern or pattern.startswith("/"):
        return None
    # PurePosixPath.match normalizes redundant and trailing separators before
    # matching. Translate that canonical spelling so the database predicate is
    # never narrower than the authoritative post-filter.
    pattern = PurePosixPath(pattern).as_posix()
    parts: list[str] = []
    cursor = 0
    while cursor < len(pattern):
        character = pattern[cursor]
        if character == "*":
            while pattern[cursor : cursor + 1] == "*":
                cursor += 1
            parts.append("[^/]*")
        elif character == "?":
            parts.append("[^/]")
            cursor += 1
        elif character == "[":
            translated = _character_class(pattern, cursor)
            if translated is None:
                return None
            expression, cursor = translated
            parts.append(expression)
        else:
            parts.append(re.escape(character))
            cursor += 1
    return "(^|/)" + "".join(parts) + "$"


def path_condition(patterns: Sequence[str], *, column: str = "path") -> str | None:
    """Return a SQL predicate matching *column* against any of *patterns*.

    ``None`` means no pushdown: either there is nothing to filter, or at least one
    pattern is untranslatable. Because the patterns are OR-ed, dropping one would
    make the predicate narrower than the post-filter and lose rows, so a single
    untranslatable pattern disables the whole pushdown.
    """
    if not patterns:
        return None
    expressions: list[str] = []
    for pattern in patterns:
        translated = glob_to_regex(pattern)
        if translated is None:
            return None
        quoted = "'" + translated.replace("'", "''") + "'"
        expressions.append(f"regexp_like({column}, {quoted})")
    return "(" + " OR ".join(expressions) + ")"
