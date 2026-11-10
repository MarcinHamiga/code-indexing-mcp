"""EXT-4 and EXT-5: definition queries must name every definition exactly once."""

from __future__ import annotations

from pathlib import Path

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.models import ExtractedChunk


def _symbols(language: str, source: str) -> list[tuple[str, str | None]]:
    result = TreeSitterExtractor().extract(Path("a"), language, source.encode())
    return [(c.kind, c.qualified_symbol) for c in result.chunks if c.kind != "module"]


def _chunks(language: str, source: str) -> list[ExtractedChunk]:
    return TreeSitterExtractor().extract(Path("a"), language, source.encode()).chunks


def test_c_pointer_returning_functions_are_definitions() -> None:
    source = (
        "char *dup_name(const char *s) { return 0; }\n"
        "struct ops *get_ops(void) { return 0; }\n"
        "char **argv_copy(void) { return 0; }\n"
    )

    assert _symbols("c", source) == [
        ("function", "dup_name"),
        ("function", "get_ops"),
        ("function", "argv_copy"),
    ]


def test_c_struct_uses_are_not_definitions() -> None:
    source = "static struct node head;\nstruct node { int value; struct node *next; };\n"

    chunks = _chunks("c", source)

    assert [(c.kind, c.qualified_symbol) for c in chunks if c.kind != "module"] == [
        ("struct", "node")
    ]
    struct = next(c for c in chunks if c.kind == "struct")
    assert struct.content == "struct node { int value; struct node *next; }"


def test_cpp_inline_methods_are_individual_definitions() -> None:
    source = (
        "class Stack {\npublic:\n  void push(int v) { }\n  int pop() { return 0; }\n"
        "  bool empty() const { return true; }\n};\n"
    )

    chunks = _chunks("cpp", source)

    assert [(c.kind, c.qualified_symbol) for c in chunks if c.kind != "module"] == [
        ("class", "Stack"),
        ("method", "Stack.push"),
        ("method", "Stack.pop"),
        ("method", "Stack.empty"),
    ]
    pop = next(c for c in chunks if c.qualified_symbol == "Stack.pop")
    assert pop.content == "int pop() { return 0; }"


def test_cpp_qualified_pointer_reference_and_struct_definitions() -> None:
    source = (
        "void Stack::push() {}\n"
        "Widget* make() { return 0; }\n"
        "Widget& instance() { static Widget w; return w; }\n"
        "Widget* Stack::build() { return 0; }\n"
        "struct Point { int x; };\n"
        "class Forward;\n"
    )

    assert _symbols("cpp", source) == [
        ("function", "push"),
        ("function", "make"),
        ("function", "instance"),
        ("function", "build"),
        ("struct", "Point"),
    ]


def test_go_grouped_types_each_get_a_definition() -> None:
    source = (
        "package p\ntype (\n  Circle struct{ r int }\n  Square struct{ s int }\n"
        "  Shape interface{ Area() float64 }\n)\ntype One struct{}\n"
    )

    chunks = _chunks("go", source)

    assert [(c.kind, c.qualified_symbol) for c in chunks if c.kind != "module"] == [
        ("class", "Circle"),
        ("class", "Square"),
        ("class", "Shape"),
        ("class", "One"),
    ]
    assert next(c for c in chunks if c.symbol == "One").content == "type One struct{}"


def test_swift_multi_case_entries_define_every_name() -> None:
    source = "enum D {\n  case north, south, east\n  case west\n}\n"

    chunks = _chunks("swift", source)

    assert [c.qualified_symbol for c in chunks if c.kind == "constant"] == [
        "D.north",
        "D.south",
        "D.east",
        "D.west",
    ]
    assert next(c for c in chunks if c.symbol == "west").content == "case west"


def test_gdshader_multi_const_defines_every_name_and_ignores_values() -> None:
    source = "const float A = 1.0, B = 2.0;\nconst float C = A;\n"

    chunks = _chunks("gdshader", source)

    assert [c.qualified_symbol for c in chunks if c.kind == "constant"] == ["A", "B", "C"]
    assert next(c for c in chunks if c.symbol == "C").content == "const float C = A;"
