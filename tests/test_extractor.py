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
