"""Test and benchmark chunk classification for C#, Java, Python, Rust, and TypeScript/TSX.

Combines repository-relative path conventions with Tree-sitter AST inspection so
whole test/benchmark files (including helpers, fixtures, and builders) and
inline test scopes (such as Rust ``#[cfg(test)] mod tests`` or Vitest
``if (import.meta.vitest)`` blocks) are tagged at extraction time without
altering chunk boundaries or embeddings on ordinary production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from tree_sitter import Node

TEST_DETECTION_LANGUAGES: Final = frozenset(
    {"csharp", "java", "python", "rust", "typescript", "tsx"}
)

_COMMON_TEST_DIRS: Final = frozenset(
    {
        "test",
        "tests",
        "bench",
        "benches",
        "benchmark",
        "benchmarks",
    }
)

_JAVA_EXTRA_TEST_DIRS: Final = frozenset(
    {
        "javatests",
        "testfixtures",
        "integrationtest",
        "integrationtests",
        "inttest",
        "inttests",
        "functionaltest",
        "functionaltests",
        "acceptancetest",
        "acceptancetests",
        "jmh",
    }
)

_TS_EXTRA_TEST_DIRS: Final = frozenset(
    {
        "__tests__",
        "__test__",
        "__specs__",
        "__spec__",
        "__benchmarks__",
        "__mocks__",
        "__fixtures__",
        "e2e",
        "e2e-tests",
        "cypress",
        "playwright",
    }
)

_TS_AMBIENT_DECLARATION_SUFFIXES: Final = (".d.ts", ".d.mts", ".d.cts")
_TS_FILE_EXTENSIONS: Final = (".tsx", ".ts", ".mts", ".cts")
_TS_UNAMBIGUOUS_STEM_SUFFIXES: Final = (
    ".test",
    ".tests",
    ".spec",
    ".specs",
    ".bench",
    ".benches",
    ".benchmark",
    ".benchmarks",
    ".e2e",
    ".e2e-spec",
    ".e2e-test",
    ".test-d",
    ".spec-d",
    "_test",
    "_tests",
    "_spec",
    "_specs",
    "_bench",
    "_benches",
    "_benchmark",
    "_benchmarks",
)
_TS_AMBIGUOUS_EXACT_STEMS: Final = frozenset(
    {
        "test",
        "tests",
        "spec",
        "specs",
        "bench",
        "benches",
        "benchmark",
        "benchmarks",
    }
)

_TS_TEST_PACKAGES: Final = frozenset(
    {
        "vitest",
        "@jest/globals",
        "jest",
        "node:test",
        "bun:test",
        "@playwright/test",
        "playwright/test",
        "cypress",
        "mocha",
        "jasmine",
        "ava",
        "tape",
        "uvu",
        "tsd",
        "expect-type",
        "tinybench",
        "@angular/core/testing",
        "@nestjs/testing",
        "@storybook/test",
    }
)
_TS_TEST_PACKAGE_PREFIXES: Final = ("@vitest/", "@testing-library/")

_TS_DEFAULT_SINGLE_FN_PACKAGES: Final = frozenset({"ava", "tape", "uvu", "node:test"})

_TS_UNAMBIGUOUS_CALL_ROOTS: Final = frozenset(
    {
        "describe",
        "fdescribe",
        "xdescribe",
        "suite",
        "it",
        "fit",
        "xit",
        "test",
        "specify",
        "xspecify",
        "bench",
        "expectTypeOf",
        "assertType",
    }
)
_TS_UNAMBIGUOUS_HOOK_ROOTS: Final = frozenset(
    {
        "beforeEach",
        "afterEach",
        "beforeAll",
        "afterAll",
        "suiteSetup",
        "suiteTeardown",
    }
)
_TS_CONDITIONAL_HOOK_ROOTS: Final = frozenset({"context", "before", "after"})
_TS_ALL_CALL_ROOTS: Final = (
    _TS_UNAMBIGUOUS_CALL_ROOTS | _TS_UNAMBIGUOUS_HOOK_ROOTS | _TS_CONDITIONAL_HOOK_ROOTS
)
_TS_RUNNER_SHAPE_GUARD_ROOTS: Final = frozenset(
    {
        "test",
        "suite",
        "it",
        "describe",
        "bench",
        "fdescribe",
        "xdescribe",
        "fit",
        "xit",
        "specify",
        "xspecify",
    }
)
_TS_RUNNER_MODIFIERS: Final = frozenset(
    {
        "only",
        "skip",
        "todo",
        "each",
        "for",
        "concurrent",
        "sequential",
        "shuffle",
        "fails",
        "serial",
        "cb",
        "describe",
        "parallel",
        "step",
        "fixme",
        "slow",
        "use",
        "beforeEach",
        "afterEach",
        "beforeAll",
        "afterAll",
        "configure",
        "skipIf",
        "runIf",
    }
)
_TS_CLASS_TEST_DECORATORS: Final = frozenset({"Suite", "suite", "TestSuite"})
_TS_METHOD_TEST_DECORATORS: Final = frozenset({"Test", "test", "Benchmark", "benchmark"})

_CSHARP_DOT_TEST_DIR_SUFFIXES: Final = (
    ".test",
    ".tests",
    ".unittests",
    ".integrationtests",
    ".functionaltests",
    ".acceptancetests",
    ".e2etests",
    ".specs",
    ".bench",
    ".benches",
    ".benchmark",
    ".benchmarks",
)

_CSHARP_PASCAL_TEST_DIR_SUFFIXES: Final = (
    "Tests",
    "UnitTests",
    "IntegrationTests",
    "FunctionalTests",
    "AcceptanceTests",
    "Benchmarks",
)

_RUST_TEST_FN_ATTR_TAILS: Final = frozenset(
    {
        "test",
        "bench",
        "rstest",
        "fixture",
        "test_case",
        "wasm_bindgen_test",
        "quickcheck",
    }
)

_RUST_BENCH_MACROS: Final = frozenset(
    {
        "criterion_group",
        "criterion_main",
        "proptest",
    }
)

_CSHARP_METHOD_TEST_ATTRS: Final = frozenset(
    {
        # xUnit
        "Fact",
        "Theory",
        "SkippableFact",
        "SkippableTheory",
        "StaFact",
        "StaTheory",
        "UIFact",
        "UITheory",
        # NUnit
        "Test",
        "TestCase",
        "TestCaseSource",
        "SetUp",
        "TearDown",
        "OneTimeSetUp",
        "OneTimeTearDown",
        # MSTest
        "TestMethod",
        "DataTestMethod",
        "TestInitialize",
        "TestCleanup",
        "ClassInitialize",
        "ClassCleanup",
        "AssemblyInitialize",
        "AssemblyCleanup",
        # BenchmarkDotNet
        "Benchmark",
        "GlobalSetup",
        "GlobalCleanup",
        "IterationSetup",
        "IterationCleanup",
    }
)

_CSHARP_TYPE_TEST_ATTRS: Final = frozenset(
    {
        "TestFixture",
        "TestFixtureSource",
        "SetUpFixture",
        "TestClass",
        "CollectionDefinition",
    }
)

_JAVA_METHOD_TEST_ANNOTATIONS: Final = frozenset(
    {
        # JUnit Jupiter / JUnit 4 / TestNG
        "Test",
        "ParameterizedTest",
        "RepeatedTest",
        "TestFactory",
        "TestTemplate",
        "BeforeEach",
        "AfterEach",
        "BeforeAll",
        "AfterAll",
        "BeforeClass",
        "AfterClass",
        "BeforeMethod",
        "AfterMethod",
        "BeforeSuite",
        "AfterSuite",
        "BeforeGroups",
        "AfterGroups",
        "DataProvider",
        # JMH
        "Benchmark",
        "Setup",
        "TearDown",
    }
)

_JAVA_TYPE_TEST_ANNOTATIONS: Final = frozenset(
    {
        "SpringBootTest",
        "WebMvcTest",
        "WebFluxTest",
        "DataJpaTest",
        "DataJdbcTest",
        "DataMongoTest",
        "DataRedisTest",
        "RestClientTest",
        "JsonTest",
        "GraphQlTest",
        "JdbcTest",
        "JooqTest",
        "MicronautTest",
        "QuarkusTest",
        "QuarkusIntegrationTest",
        "Nested",
        "Suite",
        "RunWith",
        "ExtendWith",
        "TestInstance",
        "BenchmarkMode",
        "Warmup",
        "Measurement",
        "Fork",
        "State",
    }
)

_CSHARP_TYPE_NODES: Final = frozenset(
    {
        "class_declaration",
        "struct_declaration",
        "record_declaration",
    }
)

_CSHARP_OTHER_DECLARATION_NODES: Final = frozenset(
    {
        "interface_declaration",
        "enum_declaration",
        "delegate_declaration",
        "global_statement",
    }
)

_JAVA_TYPE_NODES: Final = frozenset(
    {
        "class_declaration",
        "record_declaration",
        "interface_declaration",
        "enum_declaration",
        "annotation_type_declaration",
    }
)


@dataclass(frozen=True)
class TestClassification:
    """Per-file test/benchmark classification computed once during extraction."""

    file_is_test: bool = False
    inline_spans: tuple[tuple[int, int], ...] = ()

    def contains_range(self, start_byte: int, end_byte: int) -> bool:
        """Whether ``[start_byte, end_byte]`` belongs to a test or benchmark."""
        if self.file_is_test:
            return True
        for span_start, span_end in self.inline_spans:
            if span_start > start_byte:
                break
            if span_start <= start_byte and end_byte <= span_end:
                return True
        return False

    def split_uncovered_range(self, start_byte: int, end_byte: int) -> list[tuple[int, int, bool]]:
        """Partition an uncovered module byte range along inline test spans."""
        if self.file_is_test:
            return [(start_byte, end_byte, True)]
        if not self.inline_spans or start_byte >= end_byte:
            return [(start_byte, end_byte, False)]
        segments: list[tuple[int, int, bool]] = []
        cursor = start_byte
        for span_start, span_end in self.inline_spans:
            if span_end <= cursor:
                continue
            if span_start >= end_byte:
                break
            if cursor < span_start:
                segments.append((cursor, span_start, False))
            clipped_start = max(cursor, span_start)
            clipped_end = min(end_byte, span_end)
            if clipped_start < clipped_end:
                segments.append((clipped_start, clipped_end, True))
            cursor = clipped_end
        if cursor < end_byte:
            segments.append((cursor, end_byte, False))
        return segments


_NON_TEST: Final = TestClassification()
_WHOLE_FILE_TEST: Final = TestClassification(file_is_test=True)


def _merge_spans(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not spans:
        return ()
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _node_str(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _pascal_test_name(stem: str, *, allow_it: bool = False, allow_specs: bool = False) -> bool:
    """Match PascalCase test/benchmark type and file stems without false positives.

    Case-sensitive suffix/prefix boundaries ensure words like ``Contest``,
    ``Latest``, ``Testable``, and ``Testing`` stay classified as production code.
    """
    exact = {"Test", "Tests", "TestBase", "TestCase", "Benchmark", "Benchmarks"}
    suffixes: tuple[str, ...] = (
        "Test",
        "Tests",
        "TestBase",
        "TestCase",
        "Benchmark",
        "Benchmarks",
    )
    if allow_specs:
        exact = {*exact, "Specs"}
        suffixes = (*suffixes, "Specs")
    if stem in exact or stem.endswith(suffixes):
        return True
    if allow_it and len(stem) > 2 and stem.endswith("IT"):
        before = stem[-3]
        if before.islower() or before.isdigit():
            return True
    for prefix in ("Test", "Tests", "Benchmark", "Benchmarks"):
        if stem.startswith(prefix) and len(stem) > len(prefix):
            nxt = stem[len(prefix)]
            if nxt.isupper() or nxt.isdigit() or nxt == "_":
                return True
    return False


def _has_test_directory(language: str, directories: list[str]) -> bool:
    for directory in directories:
        lower_dir = directory.lower()
        if lower_dir in _COMMON_TEST_DIRS:
            return True
        if language == "java" and lower_dir in _JAVA_EXTRA_TEST_DIRS:
            return True
        if language in {"typescript", "tsx"} and lower_dir in _TS_EXTRA_TEST_DIRS:
            return True
        if language == "csharp" and (
            lower_dir.endswith(_CSHARP_DOT_TEST_DIR_SUFFIXES)
            or directory.endswith(_CSHARP_PASCAL_TEST_DIR_SUFFIXES)
        ):
            return True
    return False


def _ts_base_stem(filename: str) -> str | None:
    """Return lowercased stem without TS/TSX extension, or None for ambient ``.d.*``."""
    lower_name = filename.lower()
    if lower_name.endswith(_TS_AMBIENT_DECLARATION_SUFFIXES):
        return None
    for ext in _TS_FILE_EXTENSIONS:
        if lower_name.endswith(ext):
            return lower_name[: -len(ext)]
    return None


def _is_unambiguous_test_path(language: str, path: Path) -> bool:
    """Whether *path* alone guarantees a test/fixture/benchmark file without AST checks.

    Files inside dedicated test/benchmark source sets (``tests/``, ``benches/``,
    ``src/test/``, ``javatests/``, ``*.Tests/``, ``*.Specs/``, ``__tests__/``),
    Python ``conftest.py``, Rust ``tests.rs``/``*_test.rs``/``*_bench.rs``, and
    TypeScript/TSX ``*.test.*``/``*.spec.*``/``*.bench.*``/``*_test.*`` modules
    routinely hold shared builders and fixtures with zero test attributes of
    their own, so they classify whole-file immediately. Filename-only matches
    outside test directories (such as ``src/pkg/test_detection.py``,
    ``src/main/java/acme/TestRunner.java``, or ``src/orders/spec.ts``) require
    AST confirmation in :func:`classify_test_ranges`.
    """
    if language not in TEST_DETECTION_LANGUAGES:
        return False
    posix = PurePosixPath(path.as_posix())
    parts = [part for part in posix.parts if part not in {"", "."}]
    if not parts:
        return False
    if _has_test_directory(language, parts[:-1]):
        return True
    filename = parts[-1]
    if language == "python" and filename.lower() == "conftest.py":
        return True
    if language == "rust":
        lower_stem = posix.stem.lower()
        return lower_stem in {"tests", "benches"} or lower_stem.endswith(
            ("_test", "_tests", "_bench", "_benches", "_benchmark", "_benchmarks")
        )
    if language in {"typescript", "tsx"}:
        ts_stem = _ts_base_stem(filename)
        return ts_stem is not None and ts_stem.endswith(_TS_UNAMBIGUOUS_STEM_SUFFIXES)
    return False


def is_test_file_path(language: str, path: Path) -> bool:
    """Return whether *path* matches a test or benchmark directory or filename convention."""
    if language not in TEST_DETECTION_LANGUAGES:
        return False
    posix = PurePosixPath(path.as_posix())
    parts = [part for part in posix.parts if part not in {"", "."}]
    if not parts:
        return False
    if _has_test_directory(language, parts[:-1]):
        return True
    filename = parts[-1]
    stem = posix.stem

    if language == "python":
        lower_name = filename.lower()
        lower_stem = stem.lower()
        if lower_name == "conftest.py":
            return True
        return lower_stem.startswith(("test_", "bench_", "benchmark_")) or lower_stem.endswith(
            ("_test", "_tests", "_bench", "_benchmark", "_benchmarks")
        )

    if language == "rust":
        lower_stem = stem.lower()
        return lower_stem in {"tests", "benches"} or lower_stem.endswith(
            ("_test", "_tests", "_bench", "_benches", "_benchmark", "_benchmarks")
        )

    if language == "java":
        return _pascal_test_name(stem, allow_it=True)

    if language == "csharp":
        return _pascal_test_name(stem, allow_specs=True)

    if language in {"typescript", "tsx"}:
        ts_stem = _ts_base_stem(filename)
        if ts_stem is None:
            return False
        return ts_stem in _TS_AMBIGUOUS_EXACT_STEMS or ts_stem.endswith(
            _TS_UNAMBIGUOUS_STEM_SUFFIXES
        )

    return False


def classify_test_ranges(
    path: Path, language: str, root: Node, source: bytes
) -> TestClassification:
    """Classify whole-file and inline test/benchmark spans for one parsed file."""
    if language not in TEST_DETECTION_LANGUAGES:
        return _NON_TEST
    if _is_unambiguous_test_path(language, path):
        return _WHOLE_FILE_TEST

    if language == "rust":
        ast_result = _classify_rust(root, source)
    elif language == "python":
        ast_result = _classify_python(root, source)
    elif language == "java":
        ast_result = _classify_java(root, source)
    elif language == "csharp":
        ast_result = _classify_csharp(root, source)
    elif language in {"typescript", "tsx"}:
        ast_result = _classify_typescript(root, source)
    else:
        return _NON_TEST

    if is_test_file_path(language, path) and (
        ast_result.file_is_test or bool(ast_result.inline_spans)
    ):
        return _WHOLE_FILE_TEST
    return ast_result


# ---------------------------------------------------------------------------
# Rust AST rules
# ---------------------------------------------------------------------------


def _rust_cfg_tokens_is_test(tokens: list[Node], source: bytes) -> bool:
    """Evaluate one ``cfg(...)`` predicate clause conservatively for test/bench."""
    meaningful = [tok for tok in tokens if not tok.is_extra]
    if not meaningful:
        return False
    if len(meaningful) == 1 and meaningful[0].type == "identifier":
        return _node_str(meaningful[0], source) in {"test", "bench", "doctest"}
    if (
        len(meaningful) == 2
        and meaningful[0].type == "identifier"
        and meaningful[1].type == "token_tree"
    ):
        combinator = _node_str(meaningful[0], source)
        if combinator == "not":
            return False
        clauses = _split_token_tree_clauses(meaningful[1])
        if combinator == "all":
            return any(_rust_cfg_tokens_is_test(clause, source) for clause in clauses)
        if combinator == "any":
            return bool(clauses) and all(
                _rust_cfg_tokens_is_test(clause, source) for clause in clauses
            )
    return False


def _split_tokens_by(tokens: list[Node], delimiter: str) -> list[list[Node]]:
    clauses: list[list[Node]] = []
    current: list[Node] = []
    for tok in tokens:
        if tok.is_extra:
            continue
        if tok.type == delimiter:
            if current:
                clauses.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        clauses.append(current)
    return clauses


def _split_token_tree_clauses(tree: Node) -> list[list[Node]]:
    """Split ``(a, b, ...)`` children into top-level comma-delimited token lists."""
    inner = [
        child
        for child in tree.children
        if not child.is_extra and child.type not in {"(", ")", "[", "]", "{", "}"}
    ]
    return _split_tokens_by(inner, ",")


def _rust_attribute_is_cfg_test(attr_item: Node, source: bytes) -> bool:
    """Whether ``#[...]`` or ``#![...]`` is a ``cfg(...)`` gating on test/bench."""
    attr = next((c for c in attr_item.named_children if c.type == "attribute"), None)
    if attr is None:
        return False
    named = [c for c in attr.named_children if not c.is_extra]
    if len(named) != 2 or named[0].type != "identifier" or named[1].type != "token_tree":
        return False
    # Deliberately match `cfg` only -- `cfg_attr(test, ...)` decorates production
    # items (e.g. `#[cfg_attr(test, derive(PartialEq))] pub struct Foo`).
    if _node_str(named[0], source) != "cfg":
        return False
    clauses = _split_token_tree_clauses(named[1])
    return len(clauses) == 1 and _rust_cfg_tokens_is_test(clauses[0], source)


def _rust_attribute_is_test_fn(attr_item: Node, source: bytes) -> bool:
    """Whether ``#[...]`` is ``#[test]``, ``#[bench]``, ``#[tokio::test]``, etc."""
    attr = next((c for c in attr_item.named_children if c.type == "attribute"), None)
    if attr is None:
        return False
    named = [c for c in attr.named_children if not c.is_extra]
    if not named:
        return False
    head = named[0]
    if head.type == "identifier":
        return _node_str(head, source) in _RUST_TEST_FN_ATTR_TAILS
    if head.type == "scoped_identifier":
        name_node = head.child_by_field_name("name")
        if name_node is not None:
            return _node_str(name_node, source) in _RUST_TEST_FN_ATTR_TAILS
    return False


def _rust_mod_has_inner_cfg_test(node: Node, source: bytes) -> bool:
    if node.type != "mod_item":
        return False
    body = node.child_by_field_name("body")
    if body is None:
        return False
    return any(
        c.type == "inner_attribute_item" and _rust_attribute_is_cfg_test(c, source)
        for c in body.named_children
    )


def _rust_macro_invocation(node: Node) -> Node | None:
    if node.type == "macro_invocation":
        return node
    if node.type == "expression_statement":
        first = next((c for c in node.named_children if not c.is_extra), None)
        if first is not None and first.type == "macro_invocation":
            return first
    return None


def _rust_macro_tail_name(invocation: Node, source: bytes) -> str | None:
    macro = invocation.child_by_field_name("macro")
    if macro is None:
        macro = next((c for c in invocation.named_children if not c.is_extra), None)
    if macro is None:
        return None
    if macro.type == "identifier":
        return _node_str(macro, source)
    if macro.type == "scoped_identifier":
        name_node = macro.child_by_field_name("name")
        if name_node is not None:
            return _node_str(name_node, source)
    return None


_RUST_PATH_SEGMENT_TOKENS: Final = frozenset({"identifier", "self", "super", "crate", "::"})


def _criterion_clause_target(clause: list[Node], source: bytes) -> str | None:
    if not clause or any(tok.type == "=" for tok in clause):
        return None
    if (
        all(tok.type in _RUST_PATH_SEGMENT_TOKENS for tok in clause)
        and clause[-1].type == "identifier"
    ):
        return _node_str(clause[-1], source)
    return None


def _criterion_target_functions(invocation: Node, source: bytes) -> set[str]:
    """Extract benchmark function identifiers listed in ``criterion_group!(...)``."""
    tree = next((c for c in invocation.named_children if c.type == "token_tree"), None)
    if tree is None:
        return set()
    inner = [
        child
        for child in tree.children
        if not child.is_extra and child.type not in {"(", ")", "[", "]", "{", "}"}
    ]
    if not inner:
        return set()
    targets: set[str] = set()
    if any(tok.type == ";" for tok in inner):
        for stmt in _split_tokens_by(inner, ";"):
            if (
                len(stmt) >= 3
                and stmt[0].type == "identifier"
                and _node_str(stmt[0], source) == "targets"
                and stmt[1].type == "="
            ):
                for clause in _split_tokens_by(stmt[2:], ","):
                    if (name := _criterion_clause_target(clause, source)) is not None:
                        targets.add(name)
        return targets

    clauses = _split_tokens_by(inner, ",")
    if len(clauses) < 2:
        return set()
    for clause in clauses[1:]:
        if (name := _criterion_clause_target(clause, source)) is not None:
            targets.add(name)
    return targets


def _collect_criterion_targets(container: Node, source: bytes, out: set[str]) -> None:
    for child in container.named_children:
        invocation = _rust_macro_invocation(child)
        if (
            invocation is not None
            and _rust_macro_tail_name(invocation, source) == "criterion_group"
        ):
            out.update(_criterion_target_functions(invocation, source))
        elif child.type == "mod_item":
            body = child.child_by_field_name("body")
            if body is not None:
                _collect_criterion_targets(body, source, out)


def _classify_rust(root: Node, source: bytes) -> TestClassification:
    for child in root.named_children:
        if child.type == "inner_attribute_item" and _rust_attribute_is_cfg_test(child, source):
            return _WHOLE_FILE_TEST

    criterion_targets: set[str] = set()
    _collect_criterion_targets(root, source, criterion_targets)

    spans: list[tuple[int, int]] = []
    top_test_items = 0
    top_prod_items = 0

    def walk_block(container: Node, *, is_root: bool) -> None:
        nonlocal top_test_items, top_prod_items
        pending_attrs: list[Node] = []
        pending_doc_start: int | None = None

        for child in container.named_children:
            if child.type in {"line_comment", "block_comment"}:
                text = _node_str(child, source)
                if (
                    text.startswith(("///", "/**"))
                    and pending_doc_start is None
                    and not pending_attrs
                ):
                    pending_doc_start = child.start_byte
                elif not text.startswith(("///", "/**")) and not pending_attrs:
                    pending_doc_start = None
                continue
            if child.type == "attribute_item":
                pending_attrs.append(child)
                continue
            if child.type == "inner_attribute_item":
                continue

            span_start = (
                pending_doc_start
                if pending_doc_start is not None
                else (pending_attrs[0].start_byte if pending_attrs else child.start_byte)
            )
            attrs = pending_attrs
            pending_attrs = []
            pending_doc_start = None

            is_cfg_test = any(_rust_attribute_is_cfg_test(a, source) for a in attrs) or (
                _rust_mod_has_inner_cfg_test(child, source)
            )
            is_fn_test = child.type == "function_item" and (
                any(_rust_attribute_is_test_fn(a, source) for a in attrs)
                or (
                    bool(criterion_targets)
                    and (name_node := child.child_by_field_name("name")) is not None
                    and _node_str(name_node, source) in criterion_targets
                )
            )
            invocation = _rust_macro_invocation(child)
            is_bench_macro = (
                invocation is not None
                and _rust_macro_tail_name(invocation, source) in _RUST_BENCH_MACROS
            )

            if is_cfg_test or is_fn_test or is_bench_macro:
                spans.append((span_start, child.end_byte))
                if is_root:
                    top_test_items += 1
                continue

            is_private_import = child.type in {
                "use_declaration",
                "extern_crate_declaration",
            } and not any(c.type == "visibility_modifier" for c in child.named_children)
            if is_root and not is_private_import:
                top_prod_items += 1

            # Nested inline modules (`mod outer { #[cfg(test)] mod tests { ... } }`).
            if child.type in {"mod_item", "impl_item"}:
                body = child.child_by_field_name("body")
                if body is not None:
                    walk_block(body, is_root=False)

    walk_block(root, is_root=True)
    if top_test_items > 0 and top_prod_items == 0:
        return _WHOLE_FILE_TEST
    return TestClassification(file_is_test=False, inline_spans=_merge_spans(spans))


# ---------------------------------------------------------------------------
# Python AST rules
# ---------------------------------------------------------------------------


def _python_decorator_dotted_name(decorator: Node, source: bytes) -> str | None:
    expr = next((c for c in decorator.named_children if not c.is_extra), None)
    if expr is None:
        return None
    if expr.type == "call":
        expr = expr.child_by_field_name("function")
    if expr is None:
        return None
    if expr.type in {"identifier", "attribute"}:
        return _node_str(expr, source)
    return None


def _python_decorator_is_test(name: str, *, for_class: bool) -> bool:
    if name in {"fixture", "pytest.fixture", "pytest_asyncio.fixture", "benchmark"}:
        return not for_class
    if name.startswith(("pytest.mark.", "mark.")) or name in {"pytest.mark", "mark"}:
        return True
    return name.startswith("unittest.skip") or name in {
        "unittest.expectedFailure",
        "expectedFailure",
        "skipIf",
        "skipUnless",
    }


def _python_is_dataclass_decorator(name: str) -> bool:
    return name == "dataclass" or name.endswith(".dataclass")


def _python_class_has_init(body: Node | None, source: bytes) -> bool:
    if body is None:
        return False
    for child in body.named_children:
        if child.is_extra:
            continue
        inner = (
            child.child_by_field_name("definition")
            if child.type == "decorated_definition"
            else child
        )
        if inner is not None and inner.type == "function_definition":
            name_node = inner.child_by_field_name("name")
            if name_node is not None and _node_str(name_node, source) == "__init__":
                return True
    return False


def _python_class_is_test_name(name: str) -> bool:
    if name in {"Test", "Tests", "Benchmark", "Benchmarks"}:
        return True
    if name.endswith(("Tests", "TestCase", "Benchmark", "Benchmarks")):
        return True
    for prefix in ("Test", "Benchmark"):
        if name.startswith(prefix) and len(name) > len(prefix):
            nxt = name[len(prefix)]
            if nxt.isupper() or nxt.isdigit() or nxt == "_":
                return True
    return False


def _python_class_inherits_testcase(class_node: Node, source: bytes) -> bool:
    superclasses = class_node.child_by_field_name("superclasses")
    if superclasses is None:
        return False
    for arg in superclasses.named_children:
        if arg.is_extra:
            continue
        target = arg
        if target.type == "subscript":
            target = target.child_by_field_name("value") or target
        text = _node_str(target, source)
        tail = text.rsplit(".", 1)[-1]
        if tail.endswith("TestCase"):
            return True
    return False


def _python_is_pytestmark_assignment(stmt: Node, source: bytes) -> bool:
    if stmt.type != "expression_statement":
        return False
    expr_children = [c for c in stmt.named_children if not c.is_extra]
    if len(expr_children) != 1 or expr_children[0].type != "assignment":
        return False
    left = expr_children[0].child_by_field_name("left")
    return (
        left is not None and left.type == "identifier" and _node_str(left, source) == "pytestmark"
    )


def _python_is_module_trivia(stmt: Node, source: bytes) -> bool:
    if stmt.type in {
        "import_statement",
        "import_from_statement",
        "future_import_statement",
        "comment",
    }:
        return True
    if stmt.type == "expression_statement":
        expr_children = [c for c in stmt.named_children if not c.is_extra]
        if len(expr_children) == 1 and expr_children[0].type == "string":
            return True
    if stmt.type == "if_statement":
        cond = stmt.child_by_field_name("condition")
        if cond is not None:
            cond_text = _node_str(cond, source).strip()
            if cond_text == "TYPE_CHECKING" or cond_text.endswith(".TYPE_CHECKING"):
                return True
            if (
                cond.type == "comparison_operator"
                and "__name__" in cond_text
                and ('"__main__"' in cond_text or "'__main__'" in cond_text)
            ):
                return True
    return False


_PythonScope = Literal["module", "test_class", "prod_class"]


def _classify_python(root: Node, source: bytes) -> TestClassification:
    spans: list[tuple[int, int]] = []
    top_test_defs = 0
    top_prod_items = 0

    def inspect_statement(stmt: Node, *, scope: _PythonScope) -> bool:
        nonlocal top_test_defs, top_prod_items
        outer = stmt
        decorators: list[str] = []
        inner = stmt
        if stmt.type == "decorated_definition":
            for child in stmt.named_children:
                if child.type == "decorator":
                    dotted = _python_decorator_dotted_name(child, source)
                    if dotted is not None:
                        decorators.append(dotted)
            defn = stmt.child_by_field_name("definition")
            if defn is None:
                return False
            inner = defn

        if inner.type == "function_definition":
            name_node = inner.child_by_field_name("name")
            name = _node_str(name_node, source) if name_node is not None else ""
            allow_name_match = scope in {"module", "test_class"}
            is_test = (
                allow_name_match and name.startswith(("test_", "bench_", "benchmark_"))
            ) or any(_python_decorator_is_test(d, for_class=False) for d in decorators)
            if is_test:
                spans.append((outer.start_byte, outer.end_byte))
                if scope == "module":
                    top_test_defs += 1
                return True
            if scope == "module":
                top_prod_items += 1
            return False

        if inner.type == "class_definition":
            name_node = inner.child_by_field_name("name")
            name = _node_str(name_node, source) if name_node is not None else ""
            if _python_class_inherits_testcase(inner, source) or any(
                _python_decorator_is_test(d, for_class=True) for d in decorators
            ):
                spans.append((outer.start_byte, outer.end_byte))
                if scope == "module":
                    top_test_defs += 1
                return True

            body = inner.child_by_field_name("body")
            is_pytest_named_class = (
                _python_class_is_test_name(name)
                and not any(_python_is_dataclass_decorator(d) for d in decorators)
                and not _python_class_has_init(body, source)
            )
            child_scope: _PythonScope = "test_class" if is_pytest_named_class else "prod_class"
            member_spans_before = len(spans)
            has_direct_test_method = False
            has_non_test_method = False
            has_nested_test_class = False
            has_nested_prod_class = False
            if body is not None:
                for sub in body.named_children:
                    if sub.is_extra:
                        continue
                    sub_inner = (
                        sub.child_by_field_name("definition")
                        if sub.type == "decorated_definition"
                        else sub
                    )
                    if sub_inner is not None and sub_inner.type == "function_definition":
                        if inspect_statement(sub, scope=child_scope):
                            has_direct_test_method = True
                        else:
                            has_non_test_method = True
                    elif sub_inner is not None and sub_inner.type == "class_definition":
                        if inspect_statement(sub, scope=child_scope):
                            has_nested_test_class = True
                        else:
                            has_nested_prod_class = True

            promote = has_direct_test_method or (
                has_nested_test_class
                and (
                    is_pytest_named_class or (not has_non_test_method and not has_nested_prod_class)
                )
            )
            if promote:
                del spans[member_spans_before:]
                spans.append((outer.start_byte, outer.end_byte))
                if scope == "module":
                    top_test_defs += 1
                return True
            if scope == "module":
                top_prod_items += 1
            return False

        if scope == "module":
            if _python_is_pytestmark_assignment(stmt, source):
                spans.append((stmt.start_byte, stmt.end_byte))
                return True
            if not _python_is_module_trivia(stmt, source):
                top_prod_items += 1
        return False

    for child in root.named_children:
        if not child.is_extra:
            inspect_statement(child, scope="module")

    if top_test_defs > 0 and top_prod_items == 0:
        return _WHOLE_FILE_TEST
    return TestClassification(file_is_test=False, inline_spans=_merge_spans(spans))


# ---------------------------------------------------------------------------
# Java AST rules
# ---------------------------------------------------------------------------


def _java_annotation_names(node: Node, source: bytes) -> list[str]:
    modifiers = next((c for c in node.named_children if c.type == "modifiers"), None)
    if modifiers is None:
        return []
    names: list[str] = []
    for child in modifiers.named_children:
        if child.type not in {"marker_annotation", "annotation"}:
            continue
        name_node = child.child_by_field_name("name") or next(
            (c for c in child.named_children if not c.is_extra), None
        )
        if name_node is None:
            continue
        text = _node_str(name_node, source)
        names.append(text.rsplit(".", 1)[-1])
    return names


def _java_class_inherits_testcase(type_node: Node, source: bytes) -> bool:
    superclass = type_node.child_by_field_name("superclass")
    if superclass is None:
        return False
    target = next((c for c in superclass.named_children if not c.is_extra), None)
    if target is None:
        return False
    tail = _node_str(target, source).split("<", 1)[0].rsplit(".", 1)[-1]
    return tail.endswith("TestCase")


def _iter_java_type_members(body: Node) -> list[Node]:
    members: list[Node] = []
    for child in body.named_children:
        if child.is_extra:
            continue
        if child.type == "enum_body_declarations":
            members.extend(c for c in child.named_children if not c.is_extra)
        else:
            members.append(child)
    return members


def _classify_java(root: Node, source: bytes) -> TestClassification:
    spans: list[tuple[int, int]] = []
    top_test_types = 0
    top_prod_types = 0

    def inspect_type(type_node: Node, *, is_root: bool) -> bool:
        nonlocal top_test_types, top_prod_types
        annotations = _java_annotation_names(type_node, source)
        name_node = type_node.child_by_field_name("name")
        type_name = _node_str(name_node, source) if name_node is not None else ""
        if any(
            a in _JAVA_TYPE_TEST_ANNOTATIONS for a in annotations
        ) or _java_class_inherits_testcase(type_node, source):
            spans.append((type_node.start_byte, type_node.end_byte))
            if is_root:
                top_test_types += 1
            return True

        body = type_node.child_by_field_name("body")
        before = len(spans)
        has_direct_test_method = False
        has_non_test_method = False
        has_nested_test_type = False
        has_nested_prod_type = False
        if body is not None:
            for member in _iter_java_type_members(body):
                if member.type in {"method_declaration", "constructor_declaration"}:
                    member_ann = _java_annotation_names(member, source)
                    if any(a in _JAVA_METHOD_TEST_ANNOTATIONS for a in member_ann):
                        has_direct_test_method = True
                    elif member.type == "method_declaration":
                        has_non_test_method = True
                elif member.type in _JAVA_TYPE_NODES:
                    if inspect_type(member, is_root=False):
                        has_nested_test_type = True
                    else:
                        has_nested_prod_type = True

        promote = has_direct_test_method or (
            has_nested_test_type
            and (
                _pascal_test_name(type_name, allow_it=True)
                or (not has_non_test_method and not has_nested_prod_type)
            )
        )
        if promote:
            del spans[before:]
            spans.append((type_node.start_byte, type_node.end_byte))
            if is_root:
                top_test_types += 1
            return True

        if is_root:
            top_prod_types += 1
        return False

    for child in root.named_children:
        if child.is_extra:
            continue
        if child.type in _JAVA_TYPE_NODES:
            inspect_type(child, is_root=True)

    if top_test_types > 0 and top_prod_types == 0:
        return _WHOLE_FILE_TEST
    return TestClassification(file_is_test=False, inline_spans=_merge_spans(spans))


# ---------------------------------------------------------------------------
# C# AST rules
# ---------------------------------------------------------------------------


def _csharp_attribute_names(node: Node, source: bytes) -> list[str]:
    names: list[str] = []
    for attr_list in node.named_children:
        if attr_list.type != "attribute_list":
            continue
        for attr in attr_list.named_children:
            if attr.type != "attribute":
                continue
            name_node = attr.child_by_field_name("name") or next(
                (c for c in attr.named_children if not c.is_extra), None
            )
            if name_node is None:
                continue
            raw = _node_str(name_node, source)
            # Strip generic arguments (`Foo<T>` -> `Foo`), `.` and `::` qualifiers,
            # and optional `Attribute` suffix (`FactAttribute` -> `Fact`).
            head = raw.split("<", 1)[0].rsplit(".", 1)[-1].rsplit("::", 1)[-1]
            if head.endswith("Attribute") and len(head) > len("Attribute"):
                head = head[: -len("Attribute")]
            names.append(head)
    return names


def _classify_csharp(root: Node, source: bytes) -> TestClassification:
    spans: list[tuple[int, int]] = []
    top_test_types = 0
    top_prod_items = 0

    def inspect_type(type_node: Node, *, is_root: bool) -> bool:
        nonlocal top_test_types, top_prod_items
        attrs = _csharp_attribute_names(type_node, source)
        name_node = type_node.child_by_field_name("name")
        type_name = _node_str(name_node, source) if name_node is not None else ""
        if any(a in _CSHARP_TYPE_TEST_ATTRS for a in attrs):
            spans.append((type_node.start_byte, type_node.end_byte))
            if is_root:
                top_test_types += 1
            return True

        body = type_node.child_by_field_name("body")
        before = len(spans)
        has_direct_test_method = False
        has_non_test_method = False
        has_nested_test_type = False
        has_nested_prod_type = False
        if body is not None:
            for member in body.named_children:
                if member.is_extra:
                    continue
                if member.type == "method_declaration":
                    member_attrs = _csharp_attribute_names(member, source)
                    if any(a in _CSHARP_METHOD_TEST_ATTRS for a in member_attrs):
                        has_direct_test_method = True
                    else:
                        has_non_test_method = True
                elif member.type in _CSHARP_TYPE_NODES:
                    if inspect_type(member, is_root=False):
                        has_nested_test_type = True
                    else:
                        has_nested_prod_type = True

        promote = has_direct_test_method or (
            has_nested_test_type
            and (
                _pascal_test_name(type_name, allow_specs=True)
                or (not has_non_test_method and not has_nested_prod_type)
            )
        )
        if promote:
            del spans[before:]
            spans.append((type_node.start_byte, type_node.end_byte))
            if is_root:
                top_test_types += 1
            return True

        if is_root:
            top_prod_items += 1
        return False

    def walk_namespace_container(container: Node) -> None:
        nonlocal top_prod_items
        for child in container.named_children:
            if child.is_extra:
                continue
            if child.type in _CSHARP_TYPE_NODES:
                inspect_type(child, is_root=True)
            elif child.type == "namespace_declaration":
                body = child.child_by_field_name("body")
                if body is not None:
                    walk_namespace_container(body)
            elif child.type in _CSHARP_OTHER_DECLARATION_NODES:
                top_prod_items += 1

    walk_namespace_container(root)
    if top_test_types > 0 and top_prod_items == 0:
        return _WHOLE_FILE_TEST
    return TestClassification(file_is_test=False, inline_spans=_merge_spans(spans))


# ---------------------------------------------------------------------------
# TypeScript / TSX AST rules
# ---------------------------------------------------------------------------


def _ts_is_test_package(pkg: str) -> bool:
    return pkg in _TS_TEST_PACKAGES or pkg.startswith(_TS_TEST_PACKAGE_PREFIXES)


def _ts_string_value(node: Node | None, source: bytes) -> str | None:
    if node is None or node.type != "string":
        return None
    raw = _node_str(node, source)
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    return None


def _ts_unwrap_parens(node: Node | None) -> Node | None:
    while node is not None and node.type == "parenthesized_expression":
        node = next((c for c in node.named_children if not c.is_extra), None)
    return node


def _ts_unwrap_expr_statement(node: Node, source: bytes) -> Node | None:
    inner = next((c for c in node.named_children if not c.is_extra), None)
    while inner is not None:
        if inner.type in {"await_expression", "parenthesized_expression"}:
            inner = next((c for c in inner.named_children if not c.is_extra), None)
        elif inner.type == "unary_expression":
            op = inner.child_by_field_name("operator")
            if op is not None and _node_str(op, source) == "void":
                inner = inner.child_by_field_name("argument")
            else:
                break
        else:
            break
    return inner


def _ts_is_import_meta_vitest(node: Node | None, source: bytes) -> bool:
    node = _ts_unwrap_parens(node)
    if node is None or node.type != "member_expression":
        return False
    obj = node.child_by_field_name("object")
    prop = node.child_by_field_name("property")
    return (
        obj is not None
        and obj.type == "meta_property"
        and _node_str(obj, source) == "import.meta"
        and prop is not None
        and prop.type == "property_identifier"
        and _node_str(prop, source) == "vitest"
    )


def _ts_is_undefined_or_null(node: Node | None, source: bytes) -> bool:
    node = _ts_unwrap_parens(node)
    if node is None:
        return False
    if node.type in {"undefined", "null"}:
        return True
    return node.type == "identifier" and _node_str(node, source) == "undefined"


def _ts_is_typeof_vitest(node: Node | None, source: bytes) -> bool:
    node = _ts_unwrap_parens(node)
    if node is None or node.type != "unary_expression":
        return False
    op = node.child_by_field_name("operator")
    if op is None or _node_str(op, source) != "typeof":
        return False
    return _ts_is_import_meta_vitest(node.child_by_field_name("argument"), source)


def _ts_binary_operator(node: Node, source: bytes) -> str | None:
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
    if left is None or right is None:
        return None
    for child in node.children:
        if not child.is_named and left.end_byte <= child.start_byte <= right.start_byte:
            return _node_str(child, source)
    return None


_KleeneVal = bool | None


def _kleene_not(val: _KleeneVal) -> _KleeneVal:
    return None if val is None else (not val)


def _kleene_and(left: _KleeneVal, right: _KleeneVal) -> _KleeneVal:
    if left is False or right is False:
        return False
    if left is True and right is True:
        return True
    return None


def _kleene_or(left: _KleeneVal, right: _KleeneVal) -> _KleeneVal:
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def _ts_eval_vitest_condition(node: Node | None, source: bytes) -> tuple[_KleeneVal, _KleeneVal]:
    """Evaluate a condition in ``(vitest_world, prod_world)`` using 3-valued Kleene logic."""
    node = _ts_unwrap_parens(node)
    if node is None:
        return (None, None)
    if _ts_is_import_meta_vitest(node, source):
        return (True, False)
    if node.type == "call_expression":
        func = _ts_unwrap_parens(node.child_by_field_name("function"))
        args_node = node.child_by_field_name("arguments")
        if (
            func is not None
            and func.type == "identifier"
            and _node_str(func, source) == "Boolean"
            and args_node is not None
        ):
            args = [c for c in args_node.named_children if not c.is_extra]
            if len(args) == 1:
                return _ts_eval_vitest_condition(args[0], source)
        return (None, None)
    if node.type == "unary_expression":
        op = node.child_by_field_name("operator")
        if op is not None and _node_str(op, source) == "!":
            vt, pr = _ts_eval_vitest_condition(node.child_by_field_name("argument"), source)
            return (_kleene_not(vt), _kleene_not(pr))
        return (None, None)
    if node.type == "binary_expression":
        op_str = _ts_binary_operator(node, source)
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if op_str in {"===", "==", "!==", "!="}:
            is_neq = op_str in {"!==", "!="}
            if (
                _ts_is_import_meta_vitest(left, source) and _ts_is_undefined_or_null(right, source)
            ) or (
                _ts_is_import_meta_vitest(right, source) and _ts_is_undefined_or_null(left, source)
            ):
                return (True, False) if is_neq else (False, True)
            for typeof_side, str_side in ((left, right), (right, left)):
                if _ts_is_typeof_vitest(typeof_side, source):
                    val = _ts_string_value(_ts_unwrap_parens(str_side), source)
                    if val == "undefined":
                        return (True, False) if is_neq else (False, True)
                    if val == "object":
                        return (False, True) if is_neq else (True, False)
        elif op_str == "&&":
            l_vt, l_pr = _ts_eval_vitest_condition(left, source)
            r_vt, r_pr = _ts_eval_vitest_condition(right, source)
            return (_kleene_and(l_vt, r_vt), _kleene_and(l_pr, r_pr))
        elif op_str == "||":
            l_vt, l_pr = _ts_eval_vitest_condition(left, source)
            r_vt, r_pr = _ts_eval_vitest_condition(right, source)
            return (_kleene_or(l_vt, r_vt), _kleene_or(l_pr, r_pr))
    return (None, None)


def _ts_decorator_tail_name(decorator: Node, source: bytes) -> str | None:
    expr = next((c for c in decorator.named_children if not c.is_extra), None)
    expr = _ts_unwrap_parens(expr)
    if expr is not None and expr.type == "call_expression":
        expr = _ts_unwrap_parens(expr.child_by_field_name("function"))
    if expr is None:
        return None
    if expr.type == "identifier":
        return _node_str(expr, source)
    if expr.type == "member_expression":
        prop = expr.child_by_field_name("property")
        if prop is not None:
            return _node_str(prop, source)
    return None


def _ts_node_decorators(node: Node, source: bytes) -> list[str]:
    names: list[str] = []
    for child in node.named_children:
        if child.type == "decorator":
            name = _ts_decorator_tail_name(child, source)
            if name is not None:
                names.append(name)
    return names


def _ts_class_is_test_name(name: str) -> bool:
    if not name:
        return False
    if _pascal_test_name(name, allow_specs=True):
        return True
    return name in {"Suite", "Suites", "Spec"} or name.endswith(("Suite", "Suites", "Spec"))


def _ts_unwrap_call_chain(call_node: Node, source: bytes) -> tuple[str | None, list[str], bool]:
    """Unwrap chained member/curried calls on a TS/JS ``call_expression``.

    Returns ``(root_name, member_chain, is_type_assert_chain)``.
    """
    members_rev: list[str] = []
    cur = call_node.child_by_field_name("function")
    while cur is not None:
        cur = _ts_unwrap_parens(cur)
        if cur is None:
            break
        if cur.type == "member_expression":
            prop = cur.child_by_field_name("property")
            if prop is None:
                return None, [], False
            members_rev.append(_node_str(prop, source))
            cur = cur.child_by_field_name("object")
        elif cur.type == "call_expression":
            inner_func = _ts_unwrap_parens(cur.child_by_field_name("function"))
            if inner_func is not None and inner_func.type == "identifier":
                return _node_str(inner_func, source), list(reversed(members_rev)), True
            cur = inner_func
        elif cur.type == "identifier":
            return _node_str(cur, source), list(reversed(members_rev)), False
        else:
            return None, [], False
    return None, [], False


def _ts_object_has_name_and_fn(obj: Node, source: bytes) -> bool:
    has_name = False
    has_fn = False
    for child in obj.named_children:
        if child.is_extra:
            continue
        if child.type == "pair":
            key = child.child_by_field_name("key")
            if key is None:
                continue
            key_text = _node_str(key, source).strip("\"'")
            if key_text == "name":
                has_name = True
            elif key_text == "fn":
                has_fn = True
        elif child.type == "method_definition":
            name_node = child.child_by_field_name("name")
            if name_node is not None and _node_str(name_node, source) == "fn":
                has_fn = True
        elif child.type == "shorthand_property_identifier":
            text = _node_str(child, source)
            if text == "name":
                has_name = True
            elif text == "fn":
                has_fn = True
    return has_name and has_fn


def _ts_has_runner_arg_shape(call_node: Node, source: bytes) -> bool:
    args_node = call_node.child_by_field_name("arguments")
    if args_node is None:
        return False
    args = [c for c in args_node.named_children if not c.is_extra]
    if not args:
        return False
    if any(arg.type in {"arrow_function", "function_expression"} for arg in args):
        return True
    return any(arg.type == "object" and _ts_object_has_name_and_fn(arg, source) for arg in args)


def _ts_is_require_test_pkg(node: Node | None, source: bytes) -> str | None:
    node = _ts_unwrap_parens(node)
    if node is None or node.type != "call_expression":
        return None
    func = _ts_unwrap_parens(node.child_by_field_name("function"))
    if func is None or func.type != "identifier" or _node_str(func, source) != "require":
        return None
    args_node = node.child_by_field_name("arguments")
    if args_node is None:
        return None
    args = [c for c in args_node.named_children if not c.is_extra]
    if len(args) != 1:
        return None
    pkg = _ts_string_value(args[0], source)
    if pkg is not None and _ts_is_test_package(pkg):
        return pkg
    return None


def _ts_extract_pattern_names(target: Node | None, source: bytes) -> list[str]:
    if target is None:
        return []
    if target.type in {"identifier", "shorthand_property_identifier_pattern"}:
        return [_node_str(target, source)]
    if target.type in {"object_pattern", "array_pattern", "rest_pattern"}:
        names: list[str] = []
        for child in target.named_children:
            names.extend(_ts_extract_pattern_names(child, source))
        return names
    if target.type == "pair_pattern":
        return _ts_extract_pattern_names(target.child_by_field_name("value"), source)
    if target.type in {"object_assignment_pattern", "assignment_pattern"}:
        return _ts_extract_pattern_names(target.child_by_field_name("left"), source)
    return []


def _ts_record_pattern_binding(
    target: Node | None,
    pkg: str,
    source: bytes,
    imported_aliases: dict[str, str],
    imported_namespaces: set[str],
    test_bound_names: set[str] | None = None,
) -> None:
    if target is None:
        return
    if test_bound_names is not None:
        test_bound_names.update(_ts_extract_pattern_names(target, source))
    if target.type == "identifier":
        local = _node_str(target, source)
        imported_namespaces.add(local)
        if pkg in _TS_DEFAULT_SINGLE_FN_PACKAGES or local in _TS_ALL_CALL_ROOTS:
            imported_aliases[local] = local if local in _TS_ALL_CALL_ROOTS else "test"
    elif target.type == "object_pattern":
        for child in target.named_children:
            if child.type == "shorthand_property_identifier_pattern":
                name = _node_str(child, source)
                if name in _TS_ALL_CALL_ROOTS:
                    imported_aliases[name] = name
            elif child.type == "pair_pattern":
                key = child.child_by_field_name("key")
                value = child.child_by_field_name("value")
                if key is not None and value is not None and value.type == "identifier":
                    orig = _node_str(key, source)
                    if orig in _TS_ALL_CALL_ROOTS:
                        imported_aliases[_node_str(value, source)] = orig


def _ts_is_extend_call(
    value: Node | None,
    imported_aliases: dict[str, str],
    imported_namespaces: set[str],
    source: bytes,
    shadowed_globals: set[str] | None = None,
) -> bool:
    value = _ts_unwrap_parens(value)
    if value is None or value.type != "call_expression":
        return False
    root, members, is_type_assert = _ts_unwrap_call_chain(value, source)
    if is_type_assert or root is None or not members:
        return False
    if (
        shadowed_globals is not None
        and root in shadowed_globals
        and root not in imported_aliases
        and root not in imported_namespaces
    ):
        return False
    if (root == "test" or imported_aliases.get(root) == "test") and all(
        m == "extend" for m in members
    ):
        return True
    return bool(
        root in imported_namespaces
        and len(members) >= 2
        and members[0] == "test"
        and all(m == "extend" for m in members[1:])
    )


def _classify_typescript(root: Node, source: bytes) -> TestClassification:
    spans: list[tuple[int, int]] = []
    top_test_items = 0
    top_prod_items = 0
    top_private_decl_items = 0
    has_test_import = False
    has_unambiguous_test_item = False
    imported_aliases: dict[str, str] = {}
    imported_namespaces: set[str] = set()
    shadowed_globals: set[str] = set()
    test_bound_names: set[str] = set()
    conditional_hooks: list[tuple[int, int, bool]] = []

    def classify_call(call_node: Node) -> Literal["unambiguous", "conditional_hook", "none"]:
        call_root, members, is_type_assert = _ts_unwrap_call_chain(call_node, source)
        if call_root is None:
            return "none"
        if (
            call_root in shadowed_globals
            and call_root not in imported_aliases
            and call_root not in imported_namespaces
        ):
            return "none"
        if is_type_assert:
            canonical = imported_aliases.get(call_root, call_root)
            return "unambiguous" if canonical in {"expectTypeOf", "assertType"} else "none"

        if call_root in {"Deno", "Bun"}:
            allowed_heads = {"test", "bench"} if call_root == "Deno" else {"test"}
            if not members or members[0] not in allowed_heads:
                return "none"
            if not all(m in _TS_RUNNER_MODIFIERS for m in members[1:]):
                return "none"
            return "unambiguous"

        if call_root in imported_namespaces:
            if members and members[0] in _TS_ALL_CALL_ROOTS:
                if not all(m in _TS_RUNNER_MODIFIERS for m in members[1:]):
                    return "none"
                return (
                    "conditional_hook"
                    if members[0] in _TS_CONDITIONAL_HOOK_ROOTS
                    else "unambiguous"
                )
            if call_root not in imported_aliases:
                return "none"

        canonical = imported_aliases.get(call_root, call_root)
        if canonical not in _TS_ALL_CALL_ROOTS:
            return "none"
        if members and not all(m in _TS_RUNNER_MODIFIERS for m in members):
            return "none"
        if (
            not has_test_import
            and not members
            and canonical in (_TS_RUNNER_SHAPE_GUARD_ROOTS | _TS_CONDITIONAL_HOOK_ROOTS)
            and not _ts_has_runner_arg_shape(call_node, source)
        ):
            return "none"
        return "conditional_hook" if canonical in _TS_CONDITIONAL_HOOK_ROOTS else "unambiguous"

    def record_import(stmt: Node) -> None:
        nonlocal has_test_import
        is_type_stmt = any(not c.is_named and c.type == "type" for c in stmt.children)
        source_node = stmt.child_by_field_name("source")
        if source_node is None:
            req_clause = next(
                (c for c in stmt.named_children if c.type == "import_require_clause"),
                None,
            )
            if req_clause is None:
                return
            str_node = next((c for c in req_clause.named_children if c.type == "string"), None)
            pkg = _ts_string_value(str_node, source)
            ident = next((c for c in req_clause.named_children if c.type == "identifier"), None)
            if pkg is not None and _ts_is_test_package(pkg):
                if ident is not None:
                    test_bound_names.add(_node_str(ident, source))
                if is_type_stmt:
                    return
                has_test_import = True
                _ts_record_pattern_binding(
                    ident, pkg, source, imported_aliases, imported_namespaces, test_bound_names
                )
            elif not is_type_stmt and ident is not None:
                shadowed_globals.add(_node_str(ident, source))
            return

        pkg = _ts_string_value(source_node, source)
        clause = next((c for c in stmt.named_children if c.type == "import_clause"), None)
        if pkg is not None and _ts_is_test_package(pkg):
            if clause is None:
                if not is_type_stmt:
                    has_test_import = True
                return
            has_runtime_binding = False
            for child in clause.named_children:
                if child.type == "identifier":
                    test_bound_names.add(_node_str(child, source))
                    if not is_type_stmt:
                        has_runtime_binding = True
                        _ts_record_pattern_binding(
                            child,
                            pkg,
                            source,
                            imported_aliases,
                            imported_namespaces,
                            test_bound_names,
                        )
                elif child.type == "namespace_import":
                    ident = next((c for c in child.named_children if c.type == "identifier"), None)
                    if ident is not None:
                        local = _node_str(ident, source)
                        test_bound_names.add(local)
                        if not is_type_stmt:
                            has_runtime_binding = True
                            imported_namespaces.add(local)
                elif child.type == "named_imports":
                    for spec in child.named_children:
                        if spec.type != "import_specifier":
                            continue
                        orig_node = spec.child_by_field_name("name")
                        if orig_node is None:
                            continue
                        orig = _node_str(orig_node, source)
                        alias_node = spec.child_by_field_name("alias")
                        local = _node_str(alias_node, source) if alias_node is not None else orig
                        test_bound_names.add(local)
                        is_type_spec = is_type_stmt or any(
                            not c.is_named and c.type == "type" for c in spec.children
                        )
                        if not is_type_spec:
                            has_runtime_binding = True
                            if orig in _TS_ALL_CALL_ROOTS:
                                imported_aliases[local] = orig
            if not is_type_stmt and has_runtime_binding:
                has_test_import = True
            return

        if is_type_stmt or clause is None:
            return
        for child in clause.named_children:
            if child.type == "identifier":
                shadowed_globals.add(_node_str(child, source))
            elif child.type == "namespace_import":
                ident = next((c for c in child.named_children if c.type == "identifier"), None)
                if ident is not None:
                    shadowed_globals.add(_node_str(ident, source))
            elif child.type == "named_imports":
                for spec in child.named_children:
                    if spec.type != "import_specifier":
                        continue
                    if any(not c.is_named and c.type == "type" for c in spec.children):
                        continue
                    orig_node = spec.child_by_field_name("name")
                    if orig_node is None:
                        continue
                    alias_node = spec.child_by_field_name("alias")
                    local = (
                        _node_str(alias_node, source)
                        if alias_node is not None
                        else _node_str(orig_node, source)
                    )
                    shadowed_globals.add(local)

    def prescan_top_level_bindings() -> None:
        nonlocal has_test_import
        for child in root.named_children:
            if child.type == "import_statement":
                record_import(child)
        for child in root.named_children:
            if child.type in {"import_statement", "empty_statement", "comment"} or child.is_extra:
                continue
            target = child
            if child.type == "export_statement":
                decl = child.child_by_field_name("declaration") or child.child_by_field_name(
                    "value"
                )
                if decl is None:
                    continue
                target = decl
            if target.type in {
                "function_declaration",
                "generator_function_declaration",
                "class_declaration",
                "abstract_class_declaration",
                "enum_declaration",
                "module",
                "internal_module",
            }:
                name_node = target.child_by_field_name("name")
                if name_node is not None:
                    shadowed_globals.add(_node_str(name_node, source))
            elif target.type in {"lexical_declaration", "variable_declaration"}:
                for dtor in target.named_children:
                    if dtor.type != "variable_declarator":
                        continue
                    value = dtor.child_by_field_name("value")
                    name_node = dtor.child_by_field_name("name")
                    pkg = _ts_is_require_test_pkg(value, source)
                    if pkg is not None:
                        has_test_import = True
                        _ts_record_pattern_binding(
                            name_node,
                            pkg,
                            source,
                            imported_aliases,
                            imported_namespaces,
                            test_bound_names,
                        )
                        continue
                    if _ts_is_extend_call(
                        value,
                        imported_aliases,
                        imported_namespaces,
                        source,
                        shadowed_globals,
                    ):
                        if name_node is not None:
                            test_bound_names.update(_ts_extract_pattern_names(name_node, source))
                            if name_node.type == "identifier":
                                imported_aliases[_node_str(name_node, source)] = "test"
                        continue
                    shadowed_globals.update(_ts_extract_pattern_names(name_node, source))
            elif target.type == "expression_statement":
                inner = next((c for c in target.named_children if not c.is_extra), None)
                if inner is not None and inner.type == "internal_module":
                    name_node = inner.child_by_field_name("name")
                    if name_node is not None:
                        shadowed_globals.add(_node_str(name_node, source))

    def callback_contains_unambiguous_test(call_node: Node) -> bool:
        args_node = call_node.child_by_field_name("arguments")
        if args_node is None:
            return False
        for arg in args_node.named_children:
            if arg.type not in {"arrow_function", "function_expression"}:
                continue
            body = arg.child_by_field_name("body")
            if body is None or body.type != "statement_block":
                continue
            for stmt in body.named_children:
                if stmt.type == "expression_statement":
                    expr = _ts_unwrap_expr_statement(stmt, source)
                    if (
                        expr is not None
                        and expr.type == "call_expression"
                        and classify_call(expr) == "unambiguous"
                    ):
                        return True
        return False

    def inspect_class(
        class_node: Node,
        *,
        outer: Node,
        span_start: int,
        is_top_scope: bool,
        extra_decorators: list[str] | None = None,
    ) -> bool:
        nonlocal top_test_items, top_prod_items, has_unambiguous_test_item
        class_decs = list(extra_decorators) if extra_decorators else []
        if outer is not class_node:
            class_decs.extend(_ts_node_decorators(outer, source))
        class_decs.extend(_ts_node_decorators(class_node, source))

        name_node = class_node.child_by_field_name("name")
        class_name = _node_str(name_node, source) if name_node is not None else ""

        if any(dec in _TS_CLASS_TEST_DECORATORS for dec in class_decs):
            spans.append((span_start, outer.end_byte))
            has_unambiguous_test_item = True
            if is_top_scope:
                top_test_items += 1
            return True

        body = class_node.child_by_field_name("body")
        before = len(spans)
        has_direct_test_method = False
        has_non_test_method = False
        has_nested_test_class = False
        has_nested_prod_class = False

        if body is not None:
            pending_decorators: list[Node] = []
            pending_doc_start: int | None = None
            for member in body.named_children:
                if member.type == "comment":
                    text = _node_str(member, source)
                    if (
                        text.startswith("/**")
                        and pending_doc_start is None
                        and not pending_decorators
                    ):
                        pending_doc_start = member.start_byte
                    elif not text.startswith("/**") and not pending_decorators:
                        pending_doc_start = None
                    continue
                if member.is_extra:
                    continue
                if member.type == "decorator":
                    pending_decorators.append(member)
                    continue

                member_span_start = (
                    pending_doc_start
                    if pending_doc_start is not None
                    else (
                        pending_decorators[0].start_byte
                        if pending_decorators
                        else member.start_byte
                    )
                )
                sibling_decs = [
                    name
                    for dec in pending_decorators
                    if (name := _ts_decorator_tail_name(dec, source)) is not None
                ]
                pending_decorators = []
                pending_doc_start = None

                if member.type == "method_definition":
                    if any(dec in _TS_METHOD_TEST_DECORATORS for dec in sibling_decs):
                        has_direct_test_method = True
                        has_unambiguous_test_item = True
                        spans.append((member_span_start, member.end_byte))
                    else:
                        has_non_test_method = True
                        method_body = member.child_by_field_name("body")
                        if method_body is not None:
                            walk_block(method_body, is_top_scope=False)
                elif member.type == "public_field_definition":
                    field_decs = sibling_decs + _ts_node_decorators(member, source)
                    value = member.child_by_field_name("value")
                    if value is not None and value.type == "class":
                        if inspect_class(
                            value,
                            outer=member,
                            span_start=member_span_start,
                            is_top_scope=False,
                            extra_decorators=field_decs,
                        ):
                            has_nested_test_class = True
                        else:
                            has_nested_prod_class = True
                    elif any(dec in _TS_METHOD_TEST_DECORATORS for dec in field_decs):
                        has_direct_test_method = True
                        has_unambiguous_test_item = True
                        spans.append((member_span_start, member.end_byte))
                    elif value is not None and value.type in {
                        "arrow_function",
                        "function_expression",
                    }:
                        has_non_test_method = True

        promote = has_direct_test_method or (
            has_nested_test_class
            and (
                _ts_class_is_test_name(class_name)
                or (not has_non_test_method and not has_nested_prod_class)
            )
        )
        if promote:
            del spans[before:]
            spans.append((span_start, outer.end_byte))
            has_unambiguous_test_item = True
            if is_top_scope:
                top_test_items += 1
            return True
        if is_top_scope:
            top_prod_items += 1
        return False

    def inspect_var_declaration(
        decl: Node, *, outer: Node, span_start: int, is_top_scope: bool
    ) -> None:
        nonlocal top_test_items, top_prod_items, top_private_decl_items
        nonlocal has_test_import, has_unambiguous_test_item
        is_exported = outer is not decl
        declarators = [c for c in decl.named_children if c.type == "variable_declarator"]
        if not declarators:
            if is_top_scope:
                if is_exported:
                    top_prod_items += 1
                else:
                    top_private_decl_items += 1
            return
        all_require = True
        all_extend = True
        for dtor in declarators:
            value = dtor.child_by_field_name("value")
            name_node = dtor.child_by_field_name("name")
            pkg = _ts_is_require_test_pkg(value, source)
            if pkg is not None:
                has_test_import = True
                _ts_record_pattern_binding(
                    name_node,
                    pkg,
                    source,
                    imported_aliases,
                    imported_namespaces,
                    test_bound_names,
                )
            else:
                all_require = False
            if _ts_is_extend_call(
                value, imported_aliases, imported_namespaces, source, shadowed_globals
            ):
                if name_node is not None:
                    test_bound_names.update(_ts_extract_pattern_names(name_node, source))
                    if name_node.type == "identifier":
                        imported_aliases[_node_str(name_node, source)] = "test"
            else:
                all_extend = False
        if all_require:
            return
        if all_extend:
            spans.append((span_start, outer.end_byte))
            has_unambiguous_test_item = True
            if is_top_scope:
                top_test_items += 1
            return
        if is_top_scope:
            if is_exported:
                top_prod_items += 1
            else:
                top_private_decl_items += 1

    def walk_block(container: Node, *, is_top_scope: bool) -> None:
        nonlocal top_test_items, top_prod_items, top_private_decl_items
        nonlocal has_test_import, has_unambiguous_test_item
        pending_doc_start: int | None = None

        for child in container.named_children:
            if child.type == "comment":
                text = _node_str(child, source)
                if text.startswith("/**") and pending_doc_start is None:
                    pending_doc_start = child.start_byte
                elif not text.startswith("/**"):
                    pending_doc_start = None
                continue
            if child.is_extra:
                continue

            span_start = pending_doc_start if pending_doc_start is not None else child.start_byte
            pending_doc_start = None

            if child.type == "empty_statement":
                continue

            if child.type == "import_statement":
                record_import(child)
                continue

            if child.type == "export_statement":
                export_source = child.child_by_field_name("source")
                if export_source is not None:
                    pkg = _ts_string_value(export_source, source)
                    if pkg is not None and _ts_is_test_package(pkg):
                        continue
                decl = child.child_by_field_name("declaration") or child.child_by_field_name(
                    "value"
                )
                if decl is not None:
                    if decl.type in {"class_declaration", "abstract_class_declaration", "class"}:
                        inspect_class(
                            decl, outer=child, span_start=span_start, is_top_scope=is_top_scope
                        )
                        continue
                    if decl.type in {"internal_module", "module"}:
                        body = decl.child_by_field_name("body")
                        if body is not None:
                            walk_block(body, is_top_scope=is_top_scope)
                            continue
                    if decl.type in {"lexical_declaration", "variable_declaration"}:
                        inspect_var_declaration(
                            decl, outer=child, span_start=span_start, is_top_scope=is_top_scope
                        )
                        continue
                    if decl.type in {"function_declaration", "generator_function_declaration"}:
                        if is_top_scope:
                            top_prod_items += 1
                        body = decl.child_by_field_name("body")
                        if body is not None:
                            walk_block(body, is_top_scope=False)
                        continue
                elif export_source is None:
                    export_clause = next(
                        (c for c in child.named_children if c.type == "export_clause"),
                        None,
                    )
                    if export_clause is not None:
                        specifiers = [
                            c for c in export_clause.named_children if c.type == "export_specifier"
                        ]
                        if specifiers and all(
                            (name_node := spec.child_by_field_name("name")) is not None
                            and _node_str(name_node, source) in test_bound_names
                            for spec in specifiers
                        ):
                            continue
                if is_top_scope:
                    top_prod_items += 1
                continue

            if child.type in {"class_declaration", "abstract_class_declaration"}:
                inspect_class(child, outer=child, span_start=span_start, is_top_scope=is_top_scope)
                continue

            if child.type in {"lexical_declaration", "variable_declaration"}:
                inspect_var_declaration(
                    child, outer=child, span_start=span_start, is_top_scope=is_top_scope
                )
                continue

            if child.type == "if_statement":
                cond = child.child_by_field_name("condition")
                val_vt, val_pr = _ts_eval_vitest_condition(cond, source)
                consequence = child.child_by_field_name("consequence")
                alternative = child.child_by_field_name("alternative")
                if val_pr is False and val_vt is not False:
                    if alternative is None:
                        spans.append((span_start, child.end_byte))
                        has_unambiguous_test_item = True
                        if is_top_scope:
                            top_test_items += 1
                    else:
                        if consequence is not None:
                            spans.append((span_start, consequence.end_byte))
                            has_unambiguous_test_item = True
                        if is_top_scope:
                            top_prod_items += 1
                        walk_block(alternative, is_top_scope=False)
                    continue
                if alternative is not None and val_pr is True and val_vt is not True:
                    alt_inner = next(
                        (c for c in alternative.named_children if not c.is_extra),
                        None,
                    )
                    if alt_inner is not None and alt_inner.type == "statement_block":
                        if is_top_scope:
                            top_prod_items += 1
                        if consequence is not None:
                            walk_block(consequence, is_top_scope=False)
                        spans.append((alternative.start_byte, child.end_byte))
                        has_unambiguous_test_item = True
                        continue
                if is_top_scope:
                    top_prod_items += 1
                if consequence is not None:
                    walk_block(consequence, is_top_scope=False)
                if alternative is not None:
                    walk_block(alternative, is_top_scope=False)
                continue

            if child.type == "module":
                body = child.child_by_field_name("body")
                if body is not None:
                    walk_block(body, is_top_scope=is_top_scope)
                    continue

            if child.type == "expression_statement":
                inner = next((c for c in child.named_children if not c.is_extra), None)
                if inner is not None and inner.type == "internal_module":
                    body = inner.child_by_field_name("body")
                    if body is not None:
                        walk_block(body, is_top_scope=is_top_scope)
                        continue
                expr = _ts_unwrap_expr_statement(child, source)
                if expr is not None and expr.type == "call_expression":
                    if _ts_is_require_test_pkg(expr, source) is not None:
                        has_test_import = True
                        continue
                    kind = classify_call(expr)
                    if kind == "unambiguous":
                        spans.append((span_start, child.end_byte))
                        has_unambiguous_test_item = True
                        if is_top_scope:
                            top_test_items += 1
                        continue
                    if kind == "conditional_hook":
                        if callback_contains_unambiguous_test(expr):
                            has_unambiguous_test_item = True
                        conditional_hooks.append((span_start, child.end_byte, is_top_scope))
                        continue
                if is_top_scope:
                    top_prod_items += 1
                continue

            if child.type in {"function_declaration", "generator_function_declaration"}:
                if is_top_scope:
                    top_private_decl_items += 1
                body = child.child_by_field_name("body")
                if body is not None:
                    walk_block(body, is_top_scope=False)
                continue

            if child.type in {"type_alias_declaration", "interface_declaration"}:
                continue

            if child.type == "statement_block":
                walk_block(child, is_top_scope=False)
                continue

            if is_top_scope:
                top_prod_items += 1

    prescan_top_level_bindings()
    walk_block(root, is_top_scope=True)

    if has_test_import or has_unambiguous_test_item:
        for hook_start, hook_end, hook_is_top in conditional_hooks:
            spans.append((hook_start, hook_end))
            if hook_is_top:
                top_test_items += 1
    else:
        for _hook_start, _hook_end, hook_is_top in conditional_hooks:
            if hook_is_top:
                top_prod_items += 1

    if (
        top_test_items > 0
        and top_prod_items == 0
        and (top_private_decl_items == 0 or has_test_import)
    ):
        return _WHOLE_FILE_TEST
    return TestClassification(file_is_test=False, inline_spans=_merge_spans(spans))
