"""Regression tests for extraction cost on deep or long inputs (review track G)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_indexing_mcp import memory_budget
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.memory_budget import indexing_memory_budget


def _extract(language: str, source: str, extractor: TreeSitterExtractor | None = None):
    return (extractor or TreeSitterExtractor()).extract(Path("x"), language, source.encode())


def test_else_if_ladder_extraction_is_not_cubic() -> None:
    # EXT-1: every identifier climbed to the root through Node.parent.
    branches = 300
    source = "int f(const char *s) {\n"
    for index in range(branches):
        keyword = "if" if index == 0 else "else if"
        source += f'  {keyword} (strcmp(s, "k{index}") == 0) {{ return {index}; }}\n'
    source += "  return -1;\n}\n"
    extractor = TreeSitterExtractor()
    started = time.perf_counter()
    result = _extract("c", source, extractor)
    elapsed = time.perf_counter() - started
    assert len(result.references) >= branches
    assert elapsed < 1.5, f"else-if ladder took {elapsed:.2f}s"


def test_long_boolean_chain_extraction_is_not_cubic() -> None:
    terms = 1_200
    source = "def f(a):\n    return " + " or ".join(f"a == K{i}" for i in range(terms)) + "\n"
    started = time.perf_counter()
    result = _extract("python", source)
    elapsed = time.perf_counter() - started
    assert len(result.references) >= terms
    # About 2 s on a developer laptop and 5-6 s on a shared CI runner under
    # xdist; the cubic climb took 27 s on the laptop, so 12 s still catches it.
    assert elapsed < 12.0, f"boolean chain took {elapsed:.2f}s"


def test_char_limited_windows_keep_a_bounded_duplication_factor() -> None:
    # EXT-3: 200-char lines make every window ~20 lines, so a fixed 20-line
    # overlap advanced the cursor one line at a time (~20x duplication).
    lines = [f'  "key{i:04d}": "' + "x" * 190 + '",' for i in range(2_000)]
    source = "{\n" + "\n".join(lines) + "\n}\n"
    result = _extract("json", source)
    total = sum(len(chunk.content) for chunk in result.chunks)
    factor = total / len(source)
    assert factor < 1.6, f"duplication factor {factor:.2f}"


def test_short_line_overlap_is_unchanged() -> None:
    lines = [f"x{i} = {i}" for i in range(1_000)]
    result = _extract("python", "\n".join(lines) + "\n")
    chunks = result.chunks
    assert [(c.start_line, c.end_line) for c in chunks] == [
        (1, 200),
        (181, 380),
        (361, 560),
        (541, 740),
        (721, 920),
        (901, 1000),
    ]  # line-limited windows keep the full 20-line overlap


class _GrowingProcess:
    """Fake psutil handle whose RSS grows by 1 MiB on every read."""

    def __init__(self) -> None:
        self.reads = 0

    def memory_info(self) -> SimpleNamespace:
        self.reads += 1
        return SimpleNamespace(rss=self.reads * 1024 * 1024)


def _many_functions() -> str:
    return "\n".join(f"def f{i}(a):\n    return a + {i}\n" for i in range(600))


def test_memory_budget_is_still_enforced_inside_the_per_match_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EXT-7: the check is throttled, but sustained growth must still trip the ceiling.
    # A 20 MiB ceiling allows only the handful of once-per-file checks, so tripping
    # it proves the throttled loops keep checking.
    fake = _GrowingProcess()
    monkeypatch.setattr(memory_budget, "_process_cache", (os.getpid(), fake))
    token = memory_budget._current_budget.set(memory_budget.ParentMemoryBudget(0, 20 * 1024**2))
    try:
        with pytest.raises(CodeIndexingError) as raised:
            _extract("python", _many_functions())
    finally:
        memory_budget._current_budget.reset(token)
    assert raised.value.code == ErrorCode.INDEX_RESOURCE_LIMIT


def test_memory_budget_ceiling_of_zero_trips_immediately() -> None:
    with pytest.raises(CodeIndexingError) as raised, indexing_memory_budget(-1):
        _extract("python", _many_functions())
    assert raised.value.code == ErrorCode.INDEX_RESOURCE_LIMIT


def test_memory_budget_check_is_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _GrowingProcess()
    monkeypatch.setattr(memory_budget, "_process_cache", (os.getpid(), fake))
    token = memory_budget._current_budget.set(memory_budget.ParentMemoryBudget(0, 10**15))
    try:
        result = _extract("python", _many_functions())
    finally:
        memory_budget._current_budget.reset(token)
    # 600 definitions yield thousands of query matches; unthrottled this read RSS
    # once per match on top of the once-per-chunk check.
    assert len(result.chunks) < fake.reads <= len(result.chunks) + 150


def _name_chars(language: str, source: str) -> tuple[int, list]:
    result = _extract(language, source)
    total = sum(
        len(ref.target_name) + len(ref.written_name or "") + len(ref.receiver_text or "")
        for ref in result.references
    )
    return total, result.references


def test_fluent_chain_reference_names_stay_bounded() -> None:
    # EXT-6: each call stored its whole callee prefix (6.8 M chars for 500 calls).
    calls = 500
    source = "const r = api" + "".join(f".step{i}(x => x + {i})" for i in range(calls)) + ";\n"
    started = time.perf_counter()
    total, references = _name_chars("javascript", source)
    elapsed = time.perf_counter() - started
    assert total < calls * 3 * 300  # was ~6.8 M
    assert elapsed < 4.0, f"fluent chain took {elapsed:.2f}s"  # was ~4 s before EXT-1 too
    tails = {ref.target_name.rsplit(".", 1)[-1] for ref in references if ref.kind == "call"}
    assert {f"step{i}" for i in range(calls)} <= tails  # every member name survives


def test_java_chain_reference_names_stay_bounded() -> None:
    calls = 500
    source = "class A { void f() { b" + "".join(f".s{i}(1)" for i in range(calls)) + "; } }\n"
    total, _ = _name_chars("java", source)
    assert total < calls * 3 * 300


def test_plain_dotted_chains_are_stored_verbatim() -> None:
    result = _extract("python", "def f():\n    pkg.mod.sub.run(1)\n    obj.a.b.c = 2\n")
    calls = [ref for ref in result.references if ref.kind == "call"]
    assert [(ref.target_name, ref.written_name, ref.receiver_text) for ref in calls] == [
        ("pkg.mod.sub.run", "pkg.mod.sub.run", "pkg.mod.sub")
    ]


def test_callee_with_arguments_collapses_but_keeps_head_and_tail() -> None:
    result = _extract("python", 'def f():\n    make(")", [1, 2]).fetch(a=(1, 2)).run()\n')
    names = {ref.target_name for ref in result.references if ref.kind == "call"}
    assert "make().fetch().run" in names


def _index_for(paths: list[str]):
    from code_indexing_mcp.reference_service import ReferenceService

    rows = [
        {"path": path, "record_kind": "coverage", "kind": "coverage", "module_path": None}
        for path in paths
    ]
    return ReferenceService._build_module_index(rows, None)  # type: ignore[arg-type]


def test_basename_candidates_match_the_full_scan_and_skip_it() -> None:
    # EXT-12: C includes and Kotlin imports scanned every known path per call.
    from pathlib import PurePosixPath

    from code_indexing_mcp.language_rules import (
        _c_include_candidates,
        _kotlin_import_candidates,
    )

    paths = [f"pkg{i % 50}/mod{i}/util{i}.h" for i in range(20_000)]
    paths += ["include/util.h", "vendor/util.h", "a/b/util.c"]
    paths += ["src/com/acme/Foo.kt", "other/com/acme/Foo.kts", "x/acme/Foo.kt", "Foo.kt"]
    known = frozenset(paths)
    index = _index_for(paths)
    source = PurePosixPath("main.c")
    for module_path in ("util.h", "sub/util.h", "stdio.h", "util1.h"):
        assert _c_include_candidates(source, module_path, known, index) == _c_include_candidates(
            source, module_path, known, None
        )
    for module_path in ("com.acme.Foo", "acme.Foo", "Foo", "com.acme.Missing"):
        assert _kotlin_import_candidates(
            source, module_path, known, index
        ) == _kotlin_import_candidates(source, module_path, known, None)
    assert _c_include_candidates(source, "util.h", known, index) == {
        PurePosixPath("include/util.h"),
        PurePosixPath("vendor/util.h"),
    }
    started = time.perf_counter()
    for _ in range(200):
        _c_include_candidates(source, "util.h", known, index)
        _kotlin_import_candidates(source, "com.acme.Foo", known, index)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"200 indexed lookups took {elapsed:.2f}s"  # scan: ~4 s
