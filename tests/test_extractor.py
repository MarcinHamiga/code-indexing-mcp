from itertools import pairwise
from pathlib import Path

import pytest

from incode_mcp.extractor import TreeSitterExtractor, normalize_identifier


def test_extracts_python_symbols_with_qualified_methods_and_module_code() -> None:
    source = b'''import os
VALUE = 1

@registered
class Greeter:
    """Greets people."""

    def hello(self, name: str) -> str:
        return f"Hello {name}"

def standalone() -> None:
    pass
'''

    result = TreeSitterExtractor().extract(Path("pkg/greet.py"), "python", source)

    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert ("class", "Greeter") in symbols
    assert ("method", "Greeter.hello") in symbols
    assert ("function", "standalone") in symbols
    class_chunk = next(chunk for chunk in result.chunks if chunk.kind == "class")
    assert class_chunk.content.startswith("@registered\nclass Greeter:")
    assert "def hello" not in class_chunk.content
    assert any(chunk.kind == "module" and "import os" in chunk.content for chunk in result.chunks)


@pytest.mark.parametrize(
    ("language", "path", "source", "expected"),
    [
        (
            "javascript",
            "web/app.js",
            b"const fetchUser = async (id) => id;\nclass Api { load() { return 1; } }\n",
            {("function", "fetchUser"), ("class", "Api"), ("method", "Api.load")},
        ),
        (
            "typescript",
            "web/types.ts",
            b"interface User { name: string }\ntype UserId = string;\nenum Role { Admin }\n",
            {("interface", "User"), ("type", "UserId"), ("enum", "Role")},
        ),
        (
            "tsx",
            "web/app.tsx",
            b"export const App = () => <main>Hello</main>;\n",
            {("function", "App")},
        ),
    ],
)
def test_extracts_javascript_and_typescript_symbols(
    language: str,
    path: str,
    source: bytes,
    expected: set[tuple[str, str]],
) -> None:
    result = TreeSitterExtractor().extract(Path(path), language, source)

    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert expected <= symbols


def test_extracts_java_symbols_with_precise_kinds_and_nested_qualification() -> None:
    source = b"""package demo;

@interface Flag {
    String value();
}

interface Service {
    void run();
}

enum State {
    ON;

    void reset() {}
}

record User(String name) {
    User {}

    String value() {
        return name;
    }
}

class Outer {
    Outer() {}

    class Inner {
        void work() {}
    }
}
"""

    result = TreeSitterExtractor().extract(Path("src/demo/Types.java"), "java", source)

    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert {
        ("annotation", "Flag"),
        ("method", "Flag.value"),
        ("interface", "Service"),
        ("method", "Service.run"),
        ("enum", "State"),
        ("method", "State.reset"),
        ("record", "User"),
        ("constructor", "User.User"),
        ("method", "User.value"),
        ("class", "Outer"),
        ("constructor", "Outer.Outer"),
        ("class", "Outer.Inner"),
        ("method", "Outer.Inner.work"),
    } <= symbols
    record_chunk = next(chunk for chunk in result.chunks if chunk.kind == "record")
    assert record_chunk.content.startswith("record User(String name)")
    assert "String value()" not in record_chunk.content


def test_splits_oversized_function_into_bounded_parts() -> None:
    body = "\n".join(f"    value_{index} = {index}" for index in range(20))
    source = f"def large():\n{body}\n".encode()

    result = TreeSitterExtractor(max_chars=120, max_lines=6, overlap_lines=1).extract(
        Path("large.py"), "python", source
    )

    parts = [chunk for chunk in result.chunks if chunk.symbol == "large"]
    assert len(parts) > 1
    assert all(chunk.kind == "function_part" for chunk in parts)
    assert [chunk.part_index for chunk in parts] == list(range(len(parts)))
    assert all(chunk.start_line <= chunk.end_line for chunk in parts)


def test_splits_a_single_oversized_line_into_bounded_chunks() -> None:
    source = ("payload = '" + ("x" * 10_000) + "'\n").encode()

    result = TreeSitterExtractor(max_chars=1024).extract(Path("payload.py"), "python", source)

    assert len(result.chunks) > 1
    assert all(len(chunk.content) <= 1024 for chunk in result.chunks)


def test_one_oversized_line_does_not_split_its_neighbours_per_line() -> None:
    """An oversized line is fragmented on its own; the rest keeps line windows."""
    body = "\n".join(f"    value_{index} = {index}" for index in range(400))
    ordinary = f"def big():\n{body}\n".encode()
    with_long_line = f"def big():\n{body}\n    blob = '{'x' * 5_000}'\n".encode()
    extractor = TreeSitterExtractor()

    baseline = extractor.extract(Path("ordinary.py"), "python", ordinary)
    mixed = extractor.extract(Path("mixed.py"), "python", with_long_line)

    # The long line only adds its own fragments; it must not force every
    # surrounding line onto a chunk of its own.
    assert len(mixed.chunks) < len(baseline.chunks) + 10
    assert all(len(chunk.content) <= extractor.max_chars for chunk in mixed.chunks)


def test_blank_runs_around_an_oversized_line_produce_no_empty_chunks() -> None:
    lines = []
    for index in range(50):
        lines.append(f"    value_{index} = {index}")
        lines.append("")
    lines.append(f"    blob = '{'x' * 5_000}'")
    source = ("def spaced():\n" + "\n".join(lines) + "\n").encode()

    result = TreeSitterExtractor().extract(Path("spaced.py"), "python", source)

    assert result.chunks
    assert all(chunk.content.strip() for chunk in result.chunks)


def test_oversized_line_fragments_carry_contiguous_byte_ranges() -> None:
    prefix = "value = 1\n"
    source = (prefix + "blob = '" + ("x" * 5_000) + "'\n").encode()

    result = TreeSitterExtractor(max_chars=1024).extract(Path("offsets.py"), "python", source)
    fragments = [chunk for chunk in result.chunks if chunk.start_byte >= len(prefix)]

    assert len(fragments) > 1
    for earlier, later in pairwise(fragments):
        assert earlier.end_byte == later.start_byte
    assert fragments[-1].end_byte <= len(source)


def test_syntax_errors_are_reported_but_valid_symbols_survive() -> None:
    source = b"def valid():\n    return 1\n\ndef broken(:\n"

    result = TreeSitterExtractor().extract(Path("broken.py"), "python", source)

    assert result.has_errors is True
    assert any(chunk.symbol == "valid" for chunk in result.chunks)


def test_java_syntax_errors_are_reported_but_valid_symbols_survive() -> None:
    source = b"class Valid {}\n\nclass Broken { void run( { }\n"

    result = TreeSitterExtractor().extract(Path("broken.java"), "java", source)

    assert result.has_errors is True
    assert any(chunk.symbol == "Valid" for chunk in result.chunks)


def test_java_declaration_only_file_does_not_create_a_module_chunk() -> None:
    result = TreeSitterExtractor().extract(
        Path("OnlyType.java"), "java", b"class OnlyType { void run() {} }"
    )

    assert not any(chunk.kind == "module" for chunk in result.chunks)


def test_identifier_normalization_splits_code_and_path_tokens() -> None:
    assert normalize_identifier("HTTPServer_v2/path-name.ts") == ("http server v2 path name ts")


def test_java_local_classes_are_qualified_through_the_enclosing_method() -> None:
    source = b"""class A {
    void m() {
        class Local {
            Local() {}

            void run() {}
        }
    }

    void n() {
        class Local {
            void run() {}
        }
    }
}
"""

    result = TreeSitterExtractor().extract(Path("A.java"), "java", source)

    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert ("class", "A.m.Local") in symbols
    assert ("constructor", "A.m.Local.Local") in symbols
    assert ("method", "A.m.Local.run") in symbols
    assert ("class", "A.n.Local") in symbols
    assert ("method", "A.n.Local.run") in symbols


def test_java_enum_constant_bodies_qualify_their_methods() -> None:
    source = b"""enum E {
    A(1) {
        void go() {}
    },
    B;

    void go() {}
}
"""

    result = TreeSitterExtractor().extract(Path("E.java"), "java", source)

    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert ("constant", "E.A") in symbols
    assert ("method", "E.A.go") in symbols
    assert ("method", "E.go") in symbols
    assert not any(chunk.symbol == "B" for chunk in result.chunks)


def test_container_chunks_stop_before_nested_type_declarations() -> None:
    source = b"class Outer {\n    class Inner {\n        void work() {}\n    }\n}\n"

    result = TreeSitterExtractor().extract(Path("Outer.java"), "java", source)

    outer = next(chunk for chunk in result.chunks if chunk.qualified_symbol == "Outer")
    inner = next(chunk for chunk in result.chunks if chunk.qualified_symbol == "Outer.Inner")
    assert outer.content == "class Outer {"
    assert inner.content == "class Inner {"


def test_python_closures_are_qualified_through_the_enclosing_callable() -> None:
    source = b"""def outer():
    def inner():
        pass


class A:
    def m(self):
        def helper():
            pass
"""

    result = TreeSitterExtractor().extract(Path("mod.py"), "python", source)

    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert ("function", "outer.inner") in symbols
    assert ("function", "A.m.helper") in symbols


def test_qualified_symbols_are_unique_within_a_file() -> None:
    sources = [
        ("python", "mod.py", b"def outer():\n    def inner():\n        pass\n"),
        ("javascript", "app.js", b"function outer() { function inner() {} }\n"),
        ("java", "E.java", b"enum E {\n    A { void go() {} }\n    void go() {}\n}\n"),
        (
            "java",
            "A.java",
            b"class A {\n    void m() { class Local {} }\n    void n() { class Local {} }\n}\n",
        ),
    ]

    for language, path, source in sources:
        result = TreeSitterExtractor().extract(Path(path), language, source)
        keys = [
            (chunk.kind, chunk.qualified_symbol)
            for chunk in result.chunks
            if chunk.symbol is not None and not chunk.kind.endswith("_part")
        ]
        assert len(keys) == len(set(keys)), f"duplicate symbols in {path}: {keys}"


def test_java_exotic_declarations_extract_surrounding_symbols() -> None:
    source = b'''sealed interface Shape permits Circle, Square {
}

final class Circle implements Shape {
    static final double PI = 3.14;

    static {
        int ignored = 1;
    }

    {
        int alsoIgnored = 2;
    }

    <T> T pick(T value) {
        return value;
    }

    String describe() {
        String text = """
                multi line
                """;
        java.util.function.Supplier<String> supplier = () -> text;
        return supplier.get();
    }
}

final class Square implements Shape {
}

@Deprecated
class Old {
}
'''

    result = TreeSitterExtractor().extract(Path("Shapes.java"), "java", source)

    assert result.has_errors is False
    symbols = {(chunk.kind, chunk.qualified_symbol) for chunk in result.chunks}
    assert ("interface", "Shape") in symbols
    assert ("class", "Circle") in symbols
    assert ("method", "Circle.pick") in symbols
    assert ("method", "Circle.describe") in symbols
    assert ("class", "Square") in symbols
    assert ("class", "Old") in symbols
    old_chunk = next(chunk for chunk in result.chunks if chunk.qualified_symbol == "Old")
    assert old_chunk.content.startswith("@Deprecated")


def test_chunk_kind_literal_covers_every_kind_the_queries_capture() -> None:
    from importlib.resources import files
    from typing import get_args

    from incode_mcp.models import ChunkKind

    declared = set(get_args(ChunkKind))
    captured = set()
    for language in ("python", "java", "javascript", "typescript", "tsx"):
        text = files("incode_mcp.queries").joinpath(f"{language}.scm").read_text()
        captured |= {
            line.split("@definition.", 1)[1].split()[0].strip(")")
            for line in text.splitlines()
            if "@definition." in line
        }

    missing = captured - declared
    assert not missing, f"ChunkKind is missing extractor kinds: {sorted(missing)}"
    assert {f"{kind}_part" for kind in captured if kind != "module"} <= declared


def test_compiled_query_is_built_once_per_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The .scm files are package data and Language objects are built in __init__.

    Re-reading and recompiling per file cost 44% of extraction time over 35 files.
    """
    import incode_mcp.extractor as extractor_module

    compiled: list[str] = []
    original = extractor_module.Query

    def counting_query(language: object, text: str) -> object:
        compiled.append(text[:40])
        return original(language, text)

    monkeypatch.setattr(extractor_module, "Query", counting_query)
    extractor = extractor_module.TreeSitterExtractor()
    source = b"def one():\n    return 1\n"

    for _ in range(5):
        extractor.extract(Path("a.py"), "python", source)
    extractor.extract(Path("b.ts"), "typescript", b"export const x = 1;\n")

    assert len(compiled) == 2, f"compiled {len(compiled)} times, expected one per language"


def test_line_index_matches_a_naive_newline_count() -> None:
    from incode_mcp.extractor import _LineIndex

    source = b"alpha\nbeta\n\ngamma\r\ndelta"
    index = _LineIndex(source)

    for offset in range(len(source) + 1):
        assert index.line_at(offset) == source[:offset].count(b"\n") + 1, f"offset {offset}"


def test_line_index_handles_empty_and_newline_only_sources() -> None:
    from incode_mcp.extractor import _LineIndex

    assert _LineIndex(b"").line_at(0) == 1
    assert _LineIndex(b"\n\n\n").line_at(3) == 4
