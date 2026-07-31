# Search `paths` Pushdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `search_code(paths=[...])` return the hits that actually match, instead of silently
returning fewer or none because the path filter runs after the database has already truncated the
result set.

**Architecture:** Add a pure, tokenizer-free module that translates a `PurePosixPath.match` glob
into an equivalent regex, push that into LanceDB's `regexp_match` prefilter alongside the existing
`languages` and `kinds` predicates, and keep the current Python post-filter as the authority on
semantics. The pushdown only narrows what the database scans; it can never decide a match on its
own. Untranslatable patterns fall back to today's behaviour with a debug log rather than guessing.

**Tech Stack:** `re` for translation, LanceDB `regexp_match` SQL predicate, `pathspec` untouched.

## Global Constraints

See [the plan index](2026-07-27-review-followups-index.md#global-constraints). Additionally:

- **`PurePosixPath.match` stays the definition of "matches".** The pushdown must be *equivalent or
  broader*, never narrower. A narrower pushdown loses results, which is the bug being fixed.
- **No behaviour change when `paths` is absent.** The predicate is only added when the caller passes
  patterns.
- **The new module must be importable without a database or a model**, matching how
  `token_batching.py` keeps its policy pure and testable.

---

## Problem

`languages` and `kinds` are pushed into the SQL prefilter (`search.py:43-47` builds them,
`storage.py:222` applies them with `prefilter=True`). **`paths` is not.** It is applied in Python at
`search.py:60`, over rows the database has already limited to `max(50, limit * 5)`
(`search.py:48-54`).

Reproduced against a synthetic project of 60 noisy modules plus one needle in `rare/needle.py`:

```
no path filter      -> 8 hits; paths seen: ['noise']
paths=['rare/*']    -> 0 hits   <-- needle exists at rare/needle.py
find_symbol confirms the chunk is indexed: 1 hit(s)
```

The needle is indexed and matches the query, but never enters the top 50 rows, so the post-filter
has nothing to keep. The failure is **rank-dependent** — querying the exact symbol name pulls the
needle into the top 50 and it returns fine — which makes it intermittent and indistinguishable from
"no such code exists". That is the failure mode most likely to make a caller stop looking.

### Why `regexp_match`, and why it is safe

Measured in this repo against LanceDB 0.34.0:

| Predicate | Result |
|---|---|
| `path LIKE 'src/%'` | works, but cannot express "one path segment" — matches `src/deep/b.py` |
| `regexp_match(path, '(^\|/)src/[^/]*$')` | works, matches exactly what `PurePosixPath.match('src/*')` matches |
| `path GLOB 'src/*'` | rejected — `The filter path does not return a boolean` |
| `path ~ '^src/'` | rejected — `Operator ~ is not supported` |

`regexp_match` works as a `prefilter=True` predicate on hybrid search, `OR`-combines, `AND`-combines
with the existing conditions, and passes backslashes through the SQL string literal unharmed.
`re.escape` output is accepted as valid Rust regex — verified for `- + ( ) space $ ^ { } \ % _ '`,
including the `''` single-quote escaping `_quoted` already performs.

The translation was checked against `PurePosixPath.match` over **870 paths × 20 patterns**:
**0 result-losing mismatches and 0 over-broad mismatches.** Two details make that exactness
possible:

- `PurePosixPath.match` is **right-anchored** for relative patterns: `*.py` matches `a/b/c.py`.
  The translation therefore anchors with `(^|/)` … `$`, not `^` … `$`.
- In Python 3.12 `PurePosixPath.match` treats `**` as a **single** segment, identically to `*`
  (`PurePosixPath("src/deep/b.py").match("src/**")` is `False`). So `**` collapses to `[^/]*`.
  Python 3.13's recursive `full_match` is a different method and is not used here.

## File Structure

| File | Responsibility |
|---|---|
| `src/code_indexing_mcp/path_filter.py` | **New.** Pure translation: `glob_to_regex(pattern)` and `path_condition(patterns)`. No imports from storage, search, or models. |
| `src/code_indexing_mcp/search.py` | Adds the path predicate to the pushdown conditions; keeps the Python post-filter unchanged as the authority. |
| `tests/test_path_filter.py` | **New.** Unit tests plus the differential equivalence test against `PurePosixPath.match`. |
| `tests/test_search.py` | Gains the end-to-end regression test for the reported bug. |
| `README.md` | Documents that `paths` is pushed down and how patterns anchor. |

---

### Task 1: The pure translation module

**Files:**
- Create: `src/code_indexing_mcp/path_filter.py`
- Test: `tests/test_path_filter.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `glob_to_regex(pattern: str) -> str | None` — an equivalent regex, or `None` when the pattern
    cannot be translated safely.
  - `path_condition(patterns: Sequence[str], *, column: str = "path") -> str | None` — a SQL
    predicate `OR`-ing one `regexp_match` per pattern, or `None` when **any** pattern is
    untranslatable. Task 2 calls this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_path_filter.py`:

```python
"""Glob-to-regex translation for the search path pushdown."""

from __future__ import annotations

import random
import re
from pathlib import PurePosixPath

import pytest

from code_indexing_mcp.path_filter import glob_to_regex, path_condition

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


def test_unterminated_character_class_is_not_translated() -> None:
    assert glob_to_regex("[abc") is None


def test_path_condition_ors_every_pattern() -> None:
    condition = path_condition(["rare/*", "tests/*"])

    assert condition == (
        "(regexp_match(path, '(^|/)rare/[^/]*$') OR regexp_match(path, '(^|/)tests/[^/]*$'))"
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_path_filter.py -q`

Expected: collection error — `ModuleNotFoundError: No module named 'code_indexing_mcp.path_filter'`.

- [ ] **Step 3: Write the implementation**

Create `src/code_indexing_mcp/path_filter.py`:

```python
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
            closing = pattern.find("]", cursor + 1)
            if closing == -1:
                return None
            body = pattern[cursor + 1 : closing]
            # fnmatch spells negation "[!abc]"; regex spells it "[^abc]".
            parts.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
            cursor = closing + 1
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
        expressions.append(f"regexp_match({column}, {quoted})")
    return "(" + " OR ".join(expressions) + ")"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_path_filter.py -q`

Expected: `32 passed` (26 parametrized equivalence cases plus 6 unit tests).

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/path_filter.py tests/test_path_filter.py
git commit -m "feat: add glob-to-regex translation for path pushdown"
```

---

### Task 2: Push the predicate into the search

**Files:**
- Modify: `src/code_indexing_mcp/search.py:27-70`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `path_condition` from Task 1.
- Produces: no new public surface. `SearchService.search_code`'s signature is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_search.py`. That file already has everything needed: the `SemanticEmbedder`
double at line 13, and the `indexed_projects` helper at line 33 showing the
`LanceStore` + `Indexer` + `initialize_project` construction pattern. Add one helper alongside it
that indexes a caller-supplied file tree, then the three tests:

```python
def _indexed_tree(tmp_path: Path, sources: dict[str, str]) -> tuple[SearchService, str]:
    """Index one project whose files are given as {relative path: source}."""
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    )
    root = tmp_path / "tree"
    root.mkdir()
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    project = initialize_project(root)
    indexer.index(project)
    return SearchService(store, embedder), project.id


def test_path_filter_finds_matches_below_the_fetch_window(tmp_path: Path) -> None:
    """A match that ranks outside the fetch window must survive a path filter.

    Before the pushdown, `paths` was applied in Python to rows the scan had already
    truncated, so a low-ranking match in a rare directory returned zero hits even
    though find_symbol proved it was indexed. The noise files repeat the query terms
    so they all outrank the needle deterministically, rather than by tie-break luck.
    """
    sources = {
        f"noise/m{index}.py": (
            f"def enforce_permissions_{index}(user):\n"
            "    'permission permission permission check'\n"
            "    return user.permission\n"
        )
        for index in range(120)
    }
    sources["rare/needle.py"] = (
        "def audit_gate(user):\n    'permission'\n    return user.allowed\n"
    )
    search, project = _indexed_tree(tmp_path, sources)

    unfiltered = search.search_code("permission check", [project], limit=8)
    filtered = search.search_code("permission check", [project], paths=["rare/*"], limit=8)

    assert {hit.path.split("/")[0] for hit in unfiltered.hits} == {"noise"}
    assert [hit.path for hit in filtered.hits] == ["rare/needle.py"]


def test_path_filter_respects_right_anchored_glob_semantics(tmp_path: Path) -> None:
    search, project = _indexed_tree(
        tmp_path,
        {
            "src/deep/b.py": "def alpha_one():\n    return 1\n",
            "src/a.py": "def alpha_two():\n    return 2\n",
            "tests/c.py": "def alpha_three():\n    return 3\n",
        },
    )

    # "src/*" spans one segment; "*.py" matches at any depth.
    assert {hit.path for hit in search.search_code("alpha", [project], paths=["src/*"]).hits} == {
        "src/a.py"
    }
    assert {hit.path for hit in search.search_code("alpha", [project], paths=["*.py"]).hits} == {
        "src/deep/b.py",
        "src/a.py",
        "tests/c.py",
    }


def test_untranslatable_path_pattern_still_filters_in_python(tmp_path: Path) -> None:
    search, project = _indexed_tree(
        tmp_path,
        {
            "src/a.py": "def alpha_two():\n    return 2\n",
            "tests/c.py": "def alpha_three():\n    return 3\n",
        },
    )

    # An absolute pattern disables the pushdown; the post-filter must still apply,
    # and PurePosixPath.match never matches an absolute pattern against these paths.
    assert search.search_code("alpha", [project], paths=["/src/a.py"]).hits == []
```

The existing assertion at `tests/test_search.py:67` —
`search.search_code("invoice", [projects[0]], paths=["billing/**"]).hits == []` — must keep passing.
It stays empty either way: the search is scoped to the `auth` project, whose only file is `auth.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search.py -k path_filter -v`

Expected: `test_path_filter_finds_matches_below_the_fetch_window` FAILS with
`assert [] == ['rare/needle.py']` — the exact bug. The other two may already pass.

- [ ] **Step 3: Add the pushdown**

In `src/code_indexing_mcp/search.py`, add the imports:

```python
import logging

from .path_filter import path_condition

logger = logging.getLogger(__name__)
```

Then in `search_code`, replace the condition-building block (currently `search.py:43-54`):

```python
        conditions: list[str] = []
        if languages:
            conditions.append(self._in_condition("language", languages))
        if kinds:
            conditions.append(self._in_condition("kind", kinds))
        if paths:
            pushed = path_condition(paths)
            if pushed is not None:
                conditions.append(pushed)
            else:
                # Without a pushdown the Python filter below runs on rows the scan
                # already truncated, so a low-ranking match can be missed. Widen the
                # window to make that less likely and say so, rather than reporting a
                # confident empty result.
                logger.debug(
                    "Path patterns %r could not be pushed down; filtering %d fetched rows in "
                    "Python, so low-ranking matches may be missed",
                    paths,
                    _FALLBACK_FETCH_ROWS,
                )
        fetch = (
            _FALLBACK_FETCH_ROWS
            if paths and not any(condition.startswith("(regexp_match") for condition in conditions)
            else max(50, limit * 5)
        )
        rows = self.store.hybrid_search(
            query,
            self.embedder.embed_query(query),
            project_ids,
            " AND ".join(conditions) if conditions else None,
            fetch,
        )
```

and add the constant near the top of the module:

```python
# Rows fetched when path patterns cannot be pushed into the scan. Ten times the
# ordinary window: enough that a moderately low-ranking match survives the Python
# filter, without materialising a whole project's chunks.
_FALLBACK_FETCH_ROWS = 500
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search.py -k path_filter -v`

Expected: 3 passed.

- [ ] **Step 5: Verify against the original reproduction**

Run:

```bash
.venv/bin/python - <<'PY'
import os, random, tempfile
from pathlib import Path
os.environ["CODE_INDEXING_OFFLINE"] = "1"
os.environ["CODE_INDEXING_INDEX_EXECUTION"] = "in-process"
from code_indexing_mcp.application import Application, RuntimePaths

random.seed(1)
class F:
    model_id = "f"; dimension = 8
    def embed_passages(self, texts): return [[random.random() for _ in range(8)] for _ in texts]
    def embed_query(self, text): return [random.random() for _ in range(8)]

tmp = Path(tempfile.mkdtemp()); proj = tmp / "p"; proj.mkdir()
(proj / "pyproject.toml").write_text("[project]\nname='x'\n")
(proj / "noise").mkdir()
for i in range(60):
    (proj / "noise" / f"m{i}.py").write_text(
        "\n".join(f"def helper_{i}_{j}():\n    'token window planning helper'\n    return {j}\n"
                  for j in range(6)))
(proj / "rare").mkdir()
(proj / "rare" / "needle.py").write_text(
    "def token_window_planner():\n    'token window planning'\n    return 1\n")

app = Application(RuntimePaths(data=tmp / "d", cache=tmp / "c"), embedder=F(), cwd=proj)
info = app.init_project(proj); app.index_project(info.id)
for patterns in (None, ["rare/*"], ["**/*.py"], ["noise/*"]):
    hits = app.search_code("token window planning helper", projects=[info.id],
                           paths=patterns, limit=8).hits
    print(f"paths={patterns!s:14} -> {len(hits):2} hits  {sorted({h.path.split('/')[0] for h in hits})}")
PY
```

Expected: `paths=['rare/*']` now returns 1 hit at `rare/needle.py` where it returned 0 before.
`paths=None` and `paths=['noise/*']` return 8, `paths=['**/*.py']` returns 8.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all pass with 3 new tests. `tests/test_search.py:76-78` asserts structural queries never
call `list_chunks` — this change does not touch that path, so it must still pass.

- [ ] **Step 7: Document the behaviour**

In `README.md`, after the existing paragraph about tool scoping (near line 164), add:

```markdown
`search_code`'s `paths` argument takes glob patterns relative to the project root. Patterns match
from the right, so `*.py` matches a Python file at any depth while `src/*` matches only direct
children of `src`. A single `*` and `**` both span one path segment. Patterns are translated into
the index scan itself, so a filtered search finds matches that rank below the unfiltered result
window instead of returning an empty result.
```

- [ ] **Step 8: Commit**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/search.py tests/test_search.py README.md
git commit -m "fix: push search path patterns into the scan so filtered hits are not lost"
```

---

## Self-Review

**Spec coverage.** Review item B is fully covered: Task 1 provides the equivalent translation with a
differential test, Task 2 wires it into the prefilter and adds the end-to-end regression, Step 5
re-runs the original reproduction.

**Type consistency.** `glob_to_regex` and `path_condition` are defined in Task 1 Step 3 with the
signatures declared in the Interfaces block, and consumed in Task 2 Step 3. `_FALLBACK_FETCH_ROWS`
is defined and used in the same step.

**Deliberate design choices worth a reviewer's attention.**

- The Python post-filter at `search.py:60` is **not** removed. It remains the authority, so an
  imperfect pushdown degrades to over-fetching rather than to wrong answers. The equivalence test in
  Task 1 is what makes the pushdown trustworthy; the post-filter is what makes it safe.
- One untranslatable pattern disables the pushdown for the whole call rather than for that pattern.
  Because patterns are OR-ed, filtering on the translatable subset would be *narrower* than the
  post-filter and would lose rows.
- The `fetch` expression in Step 3 detects "no pushdown happened" by looking for the `(regexp_match`
  prefix in the assembled conditions. If a reviewer finds that too implicit, hoist it to a local
  `pushed_paths: str | None` and branch on that instead — the behaviour is identical.

**Adjacent observation, deliberately out of scope.** While testing predicates, `path LIKE
'src/under\_score.py'` correctly matched only `under_score.py` on LanceDB 0.34.0, which contradicts
the comment at `storage.py:273-277` claiming the engine ignores LIKE escape sequences. The
over-fetch-and-recheck machinery in `find_symbol_chunks` may therefore be guarding against
behaviour that no longer exists. Do **not** remove it in this plan: the pinned range is
`lancedb>=0.25,<1`, older in-range versions may still over-match, and the recheck is cheap
insurance. Worth its own investigation.
