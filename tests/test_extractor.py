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
