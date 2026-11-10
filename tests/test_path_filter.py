"""Gitignore-style path globs: one regex for the SQL pushdown and the post-filter."""

from __future__ import annotations

import random
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

from code_indexing_mcp.path_filter import (
    PathPatternError,
    compile_path_pattern,
    compile_path_patterns,
    path_condition,
    path_matches,
)

TREE = [
    "top.py",
    "README.md",
    "conftest.py",
    "src/a.py",
    "src/pkg/b.py",
    "src/pkg/deep/c.py",
    "src/pkgx/d.py",
    "vendor/x/src/y.py",
    "tests/conftest.py",
    "tests/unit/test_a.py",
    "docs/a.b/c.md",
    "odd/a[].py",
    "odd/50%off.py",
    "odd/with space.py",
    "odd/it's.py",
    "odd/sub_dir/under_score.py",
]

# pattern -> the exact subset of TREE it must select.
CASES: list[tuple[str, set[str]]] = [
    # The review's five examples, now recursive.
    ("**/*.py", {p for p in TREE if p.endswith(".py")}),
    ("src/**", {"src/a.py", "src/pkg/b.py", "src/pkg/deep/c.py", "src/pkgx/d.py"}),
    ("src/**/*.py", {"src/a.py", "src/pkg/b.py", "src/pkg/deep/c.py", "src/pkgx/d.py"}),
    ("src/pkg", {"src/pkg/b.py", "src/pkg/deep/c.py"}),
    ("*.py", {p for p in TREE if p.endswith(".py")}),
    # Anchoring: a pattern with a slash is rooted, one without matches at any depth.
    ("src/*", {"src/a.py"}),
    ("src/*.py", {"src/a.py"}),
    ("src/*/y.py", set()),
    ("conftest.py", {"conftest.py", "tests/conftest.py"}),
    ("tests/conftest.py", {"tests/conftest.py"}),
    ("**/conftest.py", {"conftest.py", "tests/conftest.py"}),
    ("tests", {"tests/conftest.py", "tests/unit/test_a.py"}),
    ("x", {"vendor/x/src/y.py"}),
    (
        "**/src",
        {"src/a.py", "src/pkg/b.py", "src/pkg/deep/c.py", "src/pkgx/d.py"} | {"vendor/x/src/y.py"},
    ),
    ("src/pkg/", {"src/pkg/b.py", "src/pkg/deep/c.py"}),
    ("src//pkg/./b.py", {"src/pkg/b.py"}),
    # ** as a whole segment spans zero or more directories.
    ("src/**/b.py", {"src/pkg/b.py"}),
    ("src/**/deep/*.py", {"src/pkg/deep/c.py"}),
    ("**", set(TREE)),
    ("**/**", set(TREE)),
    ("**/*", set(TREE)),
    ("*", set(TREE)),
    ("top.py", {"top.py"}),
    ("src/a.py", {"src/a.py"}),
    # Inside a segment ** is just *, and * never crosses a slash.
    ("src/p**b.py", set()),
    ("src/pkg/b**", {"src/pkg/b.py"}),
    (
        "*/*.py",
        {
            "src/a.py",
            "tests/conftest.py",
            "odd/a[].py",
            "odd/50%off.py",
            "odd/with space.py",
            "odd/it's.py",
        },
    ),
    # Wildcard patterns select whole paths, not subtrees.
    ("src/pk*", set()),
    ("src/pkg*/*.py", {"src/pkg/b.py", "src/pkgx/d.py"}),
    ("src/pkg?/d.py", {"src/pkgx/d.py"}),
    # Character classes.
    ("src/pkg[x]/d.py", {"src/pkgx/d.py"}),
    ("src/pkg[!x]/b.py", set()),
    ("src/pkg[!y]/d.py", {"src/pkgx/d.py"}),
    ("test_[a-c].py", {"tests/unit/test_a.py"}),
    ("test_[!a-c].py", set()),
    ("odd/a[].py", {"odd/a[].py"}),
    ("*[]*.py", {"odd/a[].py"}),
    # Characters that are special in regexes stay literal.
    ("odd/50%off.py", {"odd/50%off.py"}),
    ("odd/with space.py", {"odd/with space.py"}),
    ("odd/it's.py", {"odd/it's.py"}),
    ("docs/a.b", {"docs/a.b/c.md"}),
    ("docs/axb", set()),
    ("under_score.py", {"odd/sub_dir/under_score.py"}),
    ("odd/sub_dir", {"odd/sub_dir/under_score.py"}),
    ("odd/sub?dir/*", {"odd/sub_dir/under_score.py"}),
    ("*.PY", set()),
]


@pytest.mark.parametrize(("pattern", "expected"), CASES, ids=[case[0] for case in CASES])
def test_python_post_filter_semantics(pattern: str, expected: set[str]) -> None:
    compiled = compile_path_patterns([pattern])

    assert {path for path in TREE if path_matches(path, compiled)} == expected


@pytest.mark.parametrize(("pattern", "expected"), CASES, ids=[case[0] for case in CASES])
def test_pushed_down_query_agrees_with_the_post_filter(
    tmp_path: Path, pattern: str, expected: set[str]
) -> None:
    """The same compiled pattern must select the same rows inside LanceDB."""
    table = _path_table(tmp_path)
    compiled = compile_path_patterns([pattern])
    condition = path_condition(compiled)
    assert condition is not None

    pushed = set(table.search().where(condition).select(["path"]).to_arrow()["path"].to_pylist())

    assert pushed == expected


def test_multiple_patterns_are_or_ed_in_both_engines(tmp_path: Path) -> None:
    table = _path_table(tmp_path)
    compiled = compile_path_patterns(["src/pkg", "*.md", "top.py"])
    condition = path_condition(compiled)

    assert condition is not None
    pushed = set(table.search().where(condition).select(["path"]).to_arrow()["path"].to_pylist())
    local = {path for path in TREE if path_matches(path, compiled)}
    assert (
        pushed
        == local
        == {
            "src/pkg/b.py",
            "src/pkg/deep/c.py",
            "README.md",
            "docs/a.b/c.md",
            "top.py",
        }
    )


def test_engines_agree_on_a_random_corpus(tmp_path: Path) -> None:
    segments = ["a", "b", "src", "deep", "x.py", "a.py", "b.pyi", "50%off.py", "A.PY", "my-file.py"]
    generator = random.Random(0)
    paths = sorted(
        {
            "/".join(generator.choice(segments) for _ in range(depth))
            for depth in (1, 2, 3, 4)
            for _ in range(250)
        }
    )
    table = lancedb.connect(str(tmp_path / "db")).create_table(
        "t", pa.table({"path": paths}), mode="overwrite"
    )
    for pattern in [
        "*.py",
        "**/*.py",
        "src/**",
        "src",
        "a/**/x.py",
        "*/a.py",
        "deep/*",
        "?",
        "[ab]/*",
    ]:
        compiled = compile_path_patterns([pattern])
        condition = path_condition(compiled)
        assert condition is not None
        pushed = set(
            table.search().where(condition).select(["path"]).to_arrow()["path"].to_pylist()
        )
        assert pushed == {path for path in paths if path_matches(path, compiled)}, pattern


@pytest.mark.parametrize(
    "pattern", ["", "/", "/src/a.py", "/abs", ".", "./", "..", "src/../x", "a/[z-a]"]
)
def test_invalid_patterns_are_rejected(pattern: str) -> None:
    with pytest.raises(PathPatternError):
        compile_path_pattern(pattern)


def test_no_patterns_means_no_filter() -> None:
    assert path_condition([]) is None
    assert path_matches("anything", [])


@pytest.mark.parametrize(
    ("pattern", "prefix"),
    [
        ("src/pkg7/*.py", "src/pkg7/"),
        ("src/pkg", "src/pkg"),
        ("src/pk*", "src/pk"),
        ("src/**/x.py", "src/"),
        ("*.py", ""),
        ("**/src/x.py", ""),
        ("conftest.py", ""),
        ("a[].py/x", "a[].py/x"),
        ("[ab]/x", ""),
        ("src/[/x", "src/[/x"),
    ],
)
def test_literal_prefix_is_taken_from_the_anchored_parse(pattern: str, prefix: str) -> None:
    parsed = compile_path_pattern(pattern)

    assert parsed.literal_prefix == prefix
    condition = path_condition([parsed])
    assert condition is not None
    assert (f"starts_with(path, '{prefix}')" in condition) is bool(prefix)


def test_literal_prefix_never_excludes_a_regex_match() -> None:
    generator = random.Random(1)
    alphabet = ["a", "b", "*", "?", "/", "**", "[ab]", "[", ".", "%"]
    for _ in range(400):
        pattern = "".join(generator.choice(alphabet) for _ in range(generator.randint(1, 6)))
        try:
            parsed = compile_path_pattern(pattern)
        except PathPatternError:
            continue
        for path in [*TREE, "a/b", "ab", "a/a/b", "%", "a.b/%"]:
            if parsed.matches(path):
                assert path.startswith(parsed.literal_prefix), (pattern, path)


def test_single_quotes_are_escaped_in_the_predicate() -> None:
    condition = path_condition(compile_path_patterns(["it's/*.py"]))

    assert condition is not None
    assert "it''s" in condition


def _path_table(tmp_path: Path):  # type: ignore[no-untyped-def]
    return lancedb.connect(str(tmp_path / "db")).create_table(
        "t", pa.table({"path": TREE}), mode="overwrite"
    )
