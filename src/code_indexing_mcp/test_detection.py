"""Test and benchmark chunk classification for C#, Java, Python, and Rust.

Combines repository-relative path conventions with Tree-sitter AST inspection so
whole test/benchmark files (including helpers, fixtures, and builders) and
inline test scopes (such as Rust ``#[cfg(test)] mod tests`` blocks) are tagged
at extraction time without altering chunk boundaries or embeddings on ordinary
production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from tree_sitter import Node

TEST_DETECTION_LANGUAGES: Final = frozenset({"csharp", "java", "python", "rust"})

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
        if language == "csharp" and (
            lower_dir.endswith(_CSHARP_DOT_TEST_DIR_SUFFIXES)
            or directory.endswith(_CSHARP_PASCAL_TEST_DIR_SUFFIXES)
        ):
            return True
    return False


def _is_unambiguous_test_path(language: str, path: Path) -> bool:
    """Whether *path* alone guarantees a test/fixture/benchmark file without AST checks.

    Files inside dedicated test/benchmark source sets (``tests/``, ``benches/``,
    ``src/test/``, ``javatests/``, ``*.Tests/``, ``*.Specs/``), Python
    ``conftest.py``, and Rust ``tests.rs``/``*_test.rs``/``*_bench.rs`` modules
    routinely hold shared builders and fixtures with zero test attributes of
    their own, so they classify whole-file immediately. Filename-only matches
    outside test directories (such as ``src/pkg/test_detection.py`` or
    ``src/main/java/acme/TestRunner.java``) require AST confirmation in
    :func:`classify_test_ranges`.
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
