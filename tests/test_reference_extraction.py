from __future__ import annotations

from pathlib import Path

from code_indexing_mcp.extractor import TreeSitterExtractor


def _references(source: str, language: str = "python"):
    return (
        TreeSitterExtractor()
        .extract(Path(f"sample.{language}"), language, source.encode())
        .references
    )


def test_python_extracts_structural_references_and_exact_ranges() -> None:
    source = (
        "from pkg import Widget as LocalWidget\n"
        "import tools as util\n\n"
        "@trace(enabled=True)\n"
        "class Child(Base, protocol.Marker):\n"
        "    value: LocalWidget\n\n"
        "    def run(self, first, /, second: LocalWidget, *items, option: int, **kwargs)"
        " -> LocalWidget:\n"
        "        return util.make(first, *items, option=option, **kwargs)\n"
    )

    result = TreeSitterExtractor().extract(Path("sample.py"), "python", source.encode())
    refs = {
        reference.kind + ":" + reference.written_name: reference for reference in result.references
    }

    imported = refs["import:LocalWidget"]
    assert (imported.module_path, imported.imported_name, imported.alias) == (
        "pkg",
        "Widget",
        "LocalWidget",
    )
    assert (imported.start_byte, imported.end_byte, imported.start_line, imported.end_line) == (
        source.index("Widget as LocalWidget"),
        source.index("Widget as LocalWidget") + len("Widget as LocalWidget"),
        1,
        1,
    )
    assert refs["import:util"].module_path == "tools"
    assert refs["decorator:trace"].source_qualified_symbol == "Child"
    assert {
        refs["inheritance:Base"].written_name,
        refs["inheritance:protocol.Marker"].written_name,
    } == {
        "Base",
        "protocol.Marker",
    }
    assert refs["type_use:LocalWidget"].source_qualified_symbol in {"Child", "Child.run"}

    call = refs["call:util.make"]
    assert call.source_qualified_symbol == "Child.run"
    assert call.call_shape is not None
    assert call.call_shape.positional_count == 1
    assert call.call_shape.keywords == ["option"]
    assert call.call_shape.has_positional_spread is True
    assert call.call_shape.has_keyword_spread is True
    assert (call.start_byte, call.end_byte, call.start_line, call.end_line) == (
        source.index("util.make"),
        source.index("util.make") + len("util.make"),
        9,
        9,
    )

    declaration = next(item for item in result.declarations if item.qualified_symbol == "Child.run")
    assert [
        (parameter.name, parameter.kind, parameter.required, parameter.position)
        for parameter in declaration.parameters
    ] == [
        ("self", "positional_only", True, 0),
        ("first", "positional_only", True, 1),
        ("second", "positional", True, 2),
        ("items", "variadic", True, 3),
        ("option", "keyword_only", True, 4),
        ("kwargs", "keyword_variadic", True, 5),
    ]


def test_python_parameter_modes_import_targets_and_direct_call() -> None:
    source = (
        "from pkg.auth import enforce as check\n\n"
        "def run(self, user, *, strict=True):\n"
        "    return check(user, strict=strict)\n"
    )
    result = TreeSitterExtractor().extract(Path("sample.py"), "python", source.encode())
    by_name = {
        reference.kind + ":" + reference.written_name: reference for reference in result.references
    }

    imported = by_name["import:check"]
    assert (imported.target_name, imported.imported_name, imported.alias, imported.module_path) == (
        "enforce",
        "enforce",
        "check",
        "pkg.auth",
    )
    direct_call = by_name["call:check"]
    assert direct_call.target_name == "check"
    assert direct_call.source_qualified_symbol == "run"
    assert direct_call.call_shape is not None
    assert (direct_call.call_shape.positional_count, direct_call.call_shape.keywords) == (
        1,
        ["strict"],
    )
    assert (
        direct_call.start_byte,
        direct_call.end_byte,
        direct_call.start_line,
        direct_call.end_line,
    ) == (
        source.index("check(user"),
        source.index("check(user") + len("check"),
        4,
        4,
    )
    declaration = next(item for item in result.declarations if item.qualified_symbol == "run")
    assert [(item.name, item.kind, item.required) for item in declaration.parameters] == [
        ("self", "positional", True),
        ("user", "positional", True),
        ("strict", "keyword_only", False),
    ]


def test_javascript_typescript_and_tsx_extract_structural_syntax() -> None:
    javascript = (
        "import Default, { named as local } from 'pkg';\n"
        "import * as ns from 'space';\n"
        "export { local as exposed } from 'pkg';\n"
        "class Child extends Base {\n"
        "  method(first, ...rest) { this.run(first); ns.make(...rest, {ok: true}); }\n"
        "}\n"
    )
    js_refs = _references(javascript, "javascript")
    js_by_name = {reference.kind + ":" + reference.written_name: reference for reference in js_refs}
    assert js_by_name["import:Default"].imported_name == "default"
    assert js_by_name["import:local"].imported_name == "named"
    assert js_by_name["import:ns"].imported_name == "*"
    assert js_by_name["export:exposed"].module_path == "pkg"
    assert js_by_name["inheritance:Base"].source_qualified_symbol == "Child"
    assert js_by_name["call:this.run"].source_qualified_symbol == "Child.method"
    assert js_by_name["call:ns.make"].call_shape is not None
    assert js_by_name["call:ns.make"].call_shape.has_positional_spread is True

    typescript = (
        "interface Contract<T> extends Base<T> { value: T }\n"
        "type Alias = Contract<string>;\n"
        "function make<T>(value: T = undefined as T, ...rest: T[]): Contract<T> "
        "{ return build<T>(value, ...rest); }\n"
    )
    ts_result = TreeSitterExtractor().extract(Path("sample.ts"), "typescript", typescript.encode())
    ts_by_name = {
        reference.kind + ":" + reference.written_name: reference
        for reference in ts_result.references
    }
    assert ts_by_name["inheritance:Base<T>"].source_qualified_symbol == "Contract"
    assert ts_by_name["type_use:Contract<string>"].written_name == "Contract<string>"
    assert ts_by_name["call:build"].call_shape is not None
    assert ts_by_name["call:build"].call_shape.type_argument_count == 1
    declaration = next(item for item in ts_result.declarations if item.qualified_symbol == "make")
    assert [
        (parameter.name, parameter.kind, parameter.required) for parameter in declaration.parameters
    ] == [
        ("value", "positional", False),
        ("rest", "variadic", False),
    ]

    tsx = (
        "import View from './view';\n"
        "type Props = { item: Item };\n"
        "export function Screen({ item }: Props) { return <View item={item} />; }\n"
    )
    tsx_result = TreeSitterExtractor().extract(Path("sample.tsx"), "tsx", tsx.encode())
    assert any(
        reference.kind == "import" and reference.written_name == "View"
        for reference in tsx_result.references
    )
    assert any(
        reference.kind == "type_use" and reference.written_name == "Props"
        for reference in tsx_result.references
    )
    assert any(item.qualified_symbol == "Screen" for item in tsx_result.declarations)
