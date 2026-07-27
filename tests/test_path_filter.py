"""Glob-to-regex translation for the search path pushdown."""

from __future__ import annotations

import random
import re
from pathlib import PurePosixPath

import pytest

from incode_mcp.path_filter import glob_to_regex, path_condition

# Every shape the translation must handle, including the two that make it subtle:
# right-anchored relative matching, and ** behaving as a single segment in 3.12.
PATTERNS = [
    "*.py",
    "**/*.py",
    "src/**",
    "src/*",
    "**",
    "*",
    "a.py",
    "?.py",
    "[st]*/*.py",
    "[!s]*/*.py",
    "src/*/x.py",
    "**/**/*.py",
    "deep/**/*.py",
    "*_score.py",
    "50%off.py",
    "*.PY",
    "src/a.py",
    "**/deep/*",
    "*/*/*.py",
    "under_score.py",
    "my-file.py",
    "a+b.py",
    "f(x).py",
    "with space.py",
    "d$e.py",
    "g{1}.py",
    # PurePosixPath normalizes these before matching.
    "*/",
    "foo//bar",
]


def _corpus() -> list[str]:
    segments = [
        "a",
        "b",
        "src",
        "deep",
        "tests",
        "x.py",
        "a.py",
        "b.pyi",
        "under_score.py",
        "50%off.py",
        "A.PY",
        "my-file.py",
        "with space.py",
    ]
    generator = random.Random(0)
    paths = set()
    for depth in (1, 2, 3, 4):
        for _ in range(400):
            paths.add("/".join(generator.choice(segments) for _ in range(depth)))
    return sorted(paths)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_translation_is_equivalent_to_purposixpath_match(pattern: str) -> None:
    """The pushdown must never disagree with the post-filter.

    A narrower pushdown loses results, which is the bug this module exists to fix;
    a broader one only wastes rows. Assert exact equivalence so neither drifts.
    """
    expression = glob_to_regex(pattern)
    assert expression is not None, f"{pattern!r} should be translatable"
    compiled = re.compile(expression)

    for path in _corpus():
        assert bool(compiled.search(path)) is PurePosixPath(path).match(pattern), (
            f"{pattern!r} disagreed on {path!r}"
        )


def test_absolute_and_empty_patterns_are_not_translated() -> None:
    assert glob_to_regex("") is None
    assert glob_to_regex("/absolute/x.py") is None


@pytest.mark.parametrize("pattern", ["[abc", "*[]", "*[!]", "*[a-]", r"*[a\\]"])
def test_unsafe_character_class_is_not_translated(pattern: str) -> None:
    assert glob_to_regex(pattern) is None


def test_redundant_and_trailing_separators_are_normalized() -> None:
    assert glob_to_regex("*/") == glob_to_regex("*")
    assert glob_to_regex("foo//bar") == glob_to_regex("foo/bar")


def test_path_condition_ors_every_pattern() -> None:
    condition = path_condition(["rare/*", "tests/*"])

    assert condition == (
        "(regexp_like(path, '(^|/)rare/[^/]*$') OR regexp_like(path, '(^|/)tests/[^/]*$'))"
    )


def test_path_condition_is_none_when_any_pattern_is_untranslatable() -> None:
    # Patterns are OR-ed, so one pattern we cannot express means the whole
    # predicate would be too narrow. Skip the pushdown rather than lose rows.
    assert path_condition(["src/*", "/absolute"]) is None


def test_path_condition_is_none_for_no_patterns() -> None:
    assert path_condition([]) is None


def test_path_condition_escapes_single_quotes() -> None:
    condition = path_condition(["it's.py"])

    assert condition is not None
    assert "it''s" in condition
