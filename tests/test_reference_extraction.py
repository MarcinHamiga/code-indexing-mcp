from __future__ import annotations

from pathlib import Path

import pytest

from code_indexing_mcp.extractor import TreeSitterExtractor


def _references(source: str, language: str = "python"):
    return (
        TreeSitterExtractor()
        .extract(Path(f"sample.{language}"), language, source.encode())
        .references
    )


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", "def answer():\n    return 42\n\ncallback = answer\n"),
        ("javascript", "function answer() { return 42; }\nconst callback = answer;\n"),
        ("typescript", "function answer(): number { return 42; }\nconst callback = answer;\n"),
        ("tsx", "function answer(): number { return 42; }\nconst callback = answer;\n"),
    ],
)
def test_extracts_identifier_value_reads(language: str, source: str) -> None:
    references = _references(source, language)

    reads = [reference for reference in references if reference.kind == "read"]
    assert [reference.written_name for reference in reads] == ["answer"]
    read = reads[0]
    expected_start = source.rindex("answer")
    assert (read.start_byte, read.end_byte) == (expected_start, expected_start + len("answer"))


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
    assert (refs["import:util"].module_path, refs["import:util"].imported_name) == ("tools", None)
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
    # `type Alias = Contract<string>;` -- the generic head descends to its own
    # type_use row instead of the whole expression verbatim (E2). `string` is a
    # predefined_type, not a type_identifier, so it stays out of scope here.
    assert any(
        reference.kind == "type_use"
        and reference.written_name == "Contract"
        and reference.source_qualified_symbol == "Alias"
        for reference in ts_result.references
    )
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


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "javascript",
            "const run = (first = 1, ...rest) => rest;\n"
            "const outer = function (value = 1, ...more) { return more; };\n",
        ),
        (
            "typescript",
            "const run = (first: number = 1, ...rest: number[]) => rest;\n"
            "const outer = function (value: number = 1, ...more: number[]) { return more; };\n",
        ),
        (
            "tsx",
            "const run = (first: number = 1, ...rest: number[]) => <>{rest}</>;\n"
            "const outer = function (value: number = 1, ...more: number[]) "
            "{ return <>{more}</>; };\n",
        ),
    ],
)
def test_variable_assigned_callables_include_default_and_rest_parameters(
    language: str, source: str
) -> None:
    result = TreeSitterExtractor().extract(Path(f"sample.{language}"), language, source.encode())
    declarations = {item.qualified_symbol: item for item in result.declarations}

    for qualified in ("run", "outer"):
        declaration = declarations[qualified]
        assert declaration.symbol == qualified
        assert [(item.name, item.kind, item.required) for item in declaration.parameters] == [
            ("first" if qualified == "run" else "value", "positional", False),
            ("rest" if qualified == "run" else "more", "variadic", False),
        ]


@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        (
            "javascript",
            "export { foo };\nexport default foo;\nexport function Screen() {}\n",
            {"foo": ("foo", None), "default": ("foo", None), "Screen": ("Screen", None)},
        ),
        (
            "typescript",
            "export { foo };\nexport default foo;\nexport function Screen() {}\n",
            {"foo": ("foo", None), "default": ("foo", None), "Screen": ("Screen", None)},
        ),
        (
            "tsx",
            "export { foo };\nexport default foo;\nexport function Screen() { return <div />; }\n",
            {"foo": ("foo", None), "default": ("foo", None), "Screen": ("Screen", None)},
        ),
    ],
)
def test_js_family_extracts_local_default_and_declaration_exports(
    language: str, source: str, expected: dict[str, tuple[str, str | None]]
) -> None:
    references = _references(source, language)
    exports = {
        reference.written_name: reference for reference in references if reference.kind == "export"
    }

    assert set(exports) == set(expected)
    for written_name, (target_name, module_path) in expected.items():
        assert (exports[written_name].target_name, exports[written_name].module_path) == (
            target_name,
            module_path,
        )


@pytest.mark.parametrize("language", ["typescript", "tsx"])
def test_typescript_optional_parameters_are_not_required(language: str) -> None:
    source = "function f(value?: string) { return value; }\n"
    result = TreeSitterExtractor().extract(Path(f"sample.{language}"), language, source.encode())
    declaration = next(item for item in result.declarations if item.qualified_symbol == "f")

    assert [(item.name, item.kind, item.required) for item in declaration.parameters] == [
        ("value", "positional", False)
    ]


def test_python_extracts_relative_and_wildcard_imports() -> None:
    source = "from . import x\nfrom ..pkg import y\nfrom pkg import *\n"
    references = _references(source)
    imports = {
        reference.written_name: reference for reference in references if reference.kind == "import"
    }

    assert (imports["x"].target_name, imports["x"].module_path, imports["x"].alias) == (
        "x",
        ".",
        None,
    )
    assert (imports["y"].target_name, imports["y"].module_path, imports["y"].alias) == (
        "y",
        "..pkg",
        None,
    )
    assert (imports["*"].target_name, imports["*"].module_path, imports["*"].alias) == (
        "*",
        "pkg",
        None,
    )


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "javascript",
            "export const alpha = 1, gamma = 2;\n"
            "export let beta;\n"
            "export default function named() {}\n",
        ),
        (
            "typescript",
            "export const alpha = 1, gamma = 2;\n"
            "export let beta: number;\n"
            "export default function named() {}\n",
        ),
        (
            "tsx",
            "export const alpha = 1, gamma = 2;\n"
            "export let beta: number;\n"
            "export default function named() { return <div />; }\n",
        ),
    ],
)
def test_js_family_extracts_lexical_and_named_default_exports(language: str, source: str) -> None:
    references = _references(source, language)
    exports = [reference for reference in references if reference.kind == "export"]
    by_name = {reference.written_name: reference for reference in exports}

    assert len(exports) == 4
    assert {name: by_name[name].target_name for name in ("alpha", "gamma", "beta", "default")} == {
        "alpha": "alpha",
        "gamma": "gamma",
        "beta": "beta",
        "default": "named",
    }


def test_python_extracts_aliased_relative_and_wildcard_imports() -> None:
    source = "from .pkg import x as y\nfrom ..pkg import a as b\nfrom ...pkg import *\n"
    references = _references(source)
    imports = {
        reference.written_name: reference for reference in references if reference.kind == "import"
    }

    assert (imports["y"].target_name, imports["y"].module_path, imports["y"].alias) == (
        "x",
        ".pkg",
        "y",
    )
    assert (imports["b"].target_name, imports["b"].module_path, imports["b"].alias) == (
        "a",
        "..pkg",
        "b",
    )
    assert (imports["*"].target_name, imports["*"].module_path, imports["*"].alias) == (
        "*",
        "...pkg",
        None,
    )


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "javascript",
            "export var legacy = 3;\n"
            "export const { first, renamed: local, nested: [second = 2, ...rest] } = obj;\n"
            "export /* comment */ default function Commented() {}\n",
        ),
        (
            "typescript",
            "export var legacy = 3;\n"
            "export const { first, renamed: local, nested: [second = 2, ...rest] } = obj;\n"
            "export /* comment */ default function Commented() {}\n",
        ),
        (
            "tsx",
            "export var legacy = 3;\n"
            "export const { first, renamed: local, nested: [second = 2, ...rest] } = obj;\n"
            "export /* comment */ default function Commented() { return <div />; }\n",
        ),
    ],
)
def test_js_family_exports_binding_identifiers_and_commented_defaults(
    language: str, source: str
) -> None:
    references = _references(source, language)
    exports = [reference for reference in references if reference.kind == "export"]
    by_name = {reference.written_name: reference for reference in exports}

    assert len(exports) == 6
    names = ("legacy", "first", "local", "second", "rest")
    assert {name: by_name[name].target_name for name in names} == {
        "legacy": "legacy",
        "first": "first",
        "local": "local",
        "second": "second",
        "rest": "rest",
    }
    assert (by_name["default"].target_name, by_name["default"].written_name) == (
        "Commented",
        "default",
    )


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_js_family_export_star_and_namespace_export_carry_module_path(language: str) -> None:
    """E3: barrel re-exports emit an `export` row instead of nothing (or a bogus `read`)."""
    source = "export * from './x';\nexport * as ns from './x';\n"

    references = _references(source, language)
    exports = [reference for reference in references if reference.kind == "export"]

    assert len(exports) == 2
    bare, namespaced = exports
    assert (bare.target_name, bare.written_name, bare.module_path, bare.alias) == (
        "*",
        "*",
        "./x",
        None,
    )
    assert (namespaced.target_name, namespaced.written_name, namespaced.module_path, namespaced.alias) == (
        "*",
        "ns",
        "./x",
        "ns",
    )
    # The namespace alias must not also surface as a bare `read`.
    assert not any(reference.kind == "read" and reference.written_name == "ns" for reference in references)


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_js_family_module_edges_stay_visible(language: str) -> None:
    """E9: side-effect imports and require()/dynamic import() keep their module path."""
    source = (
        "import './polyfill';\n"
        "const lazy = require('./lazy');\n"
        "const dynamic = import('./dynamic');\n"
    )

    references = _references(source, language)

    bare_import = next(reference for reference in references if reference.kind == "import")
    assert (bare_import.module_path, bare_import.imported_name) == ("./polyfill", None)

    calls = {reference.target_name: reference for reference in references if reference.kind == "call"}
    assert calls["require"].module_path == "./lazy"
    assert calls["import"].module_path == "./dynamic"


def test_python_member_access_read_and_write_carry_the_receiver() -> None:
    """E5: attribute assignment/read are no longer swallowed by the `left` exclusion."""
    source = "config.TIMEOUT = 10\nprint(config.TIMEOUT)\n"

    references = _references(source, "python")

    write = next(reference for reference in references if reference.kind == "write")
    read = next(
        reference
        for reference in references
        if reference.kind == "read" and reference.written_name == "config.TIMEOUT"
    )
    assert (write.target_name, write.written_name, write.receiver_text) == (
        "config.TIMEOUT",
        "config.TIMEOUT",
        "config",
    )
    assert (read.target_name, read.written_name, read.receiver_text) == (
        "config.TIMEOUT",
        "config.TIMEOUT",
        "config",
    )
    # The bare receiver identifier still surfaces as its own `read` on the read line.
    assert any(
        reference.kind == "read" and reference.written_name == "config" for reference in references
    )


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_js_family_member_access_read_and_write_carry_the_receiver(language: str) -> None:
    """E5: member-expression assignment targets and plain reads are recorded."""
    source = "target.TIMEOUT = 5;\ntarget.TIMEOUT;\n"

    references = _references(source, language)

    write = next(reference for reference in references if reference.kind == "write")
    read = next(
        reference
        for reference in references
        if reference.kind == "read" and reference.written_name == "target.TIMEOUT"
    )
    assert (write.target_name, write.written_name, write.receiver_text) == (
        "target.TIMEOUT",
        "target.TIMEOUT",
        "target",
    )
    assert (read.target_name, read.written_name, read.receiver_text) == (
        "target.TIMEOUT",
        "target.TIMEOUT",
        "target",
    )


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_js_family_decorators_produce_decorator_references(language: str) -> None:
    """E6: `@Name`, `@ns.Name`, and `@Factory()` all yield a `decorator` row; the
    factory call keeps its own additional `call` row."""
    source = (
        "@sealed\n"
        "class Plain {}\n\n"
        "@ns.sealed\n"
        "class Namespaced {}\n\n"
        "@factory()\n"
        "class Factored {\n"
        "  @readonly\n"
        "  handle() {}\n"
        "}\n"
    )

    references = _references(source, language)

    by_span = {
        reference.start_byte: reference
        for reference in references
        if reference.kind == "decorator"
    }
    plain = by_span[source.index("sealed")]
    assert (plain.target_name, plain.written_name, plain.source_qualified_symbol) == (
        "sealed",
        "sealed",
        "Plain",
    )
    namespaced = by_span[source.index("ns.sealed")]
    assert (namespaced.target_name, namespaced.source_qualified_symbol) == (
        "ns.sealed",
        "Namespaced",
    )
    factory_target = source.index("factory()")
    factory = by_span[factory_target]
    assert (factory.target_name, factory.source_qualified_symbol) == ("factory", "Factored")
    # The factory call keeps its own `call` row in addition to the decorator row.
    assert any(
        reference.kind == "call" and reference.written_name == "factory" for reference in references
    )
    method_decorator = by_span[source.index("readonly")]
    assert method_decorator.source_qualified_symbol == "Factored.handle"

    # No duplicate `read`/member_access row shares a decorator's span.
    decorator_spans = {(r.start_byte, r.end_byte) for r in references if r.kind == "decorator"}
    for reference in references:
        if reference.kind in {"read", "write"}:
            assert (reference.start_byte, reference.end_byte) not in decorator_spans


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_destructured_parameter_is_one_marked_positional_slot(language: str) -> None:
    """A multi-key destructured parameter (E7) stays one positional slot.

    Expanding to N flat params would corrupt positional matching for every
    caller, so the extractor instead marks the slot as `destructured` and
    gives it a synthesized, non-pattern name.
    """
    if language == "javascript":
        source = "function describe({ title, subtitle, footnote }) { return title; }\n"
    else:
        source = (
            "function describe({ title, subtitle, footnote }: "
            "{ title: string; subtitle: string; footnote: string }) { return title; }\n"
        )
    result = TreeSitterExtractor().extract(Path(f"sample.{language}"), language, source.encode())
    declaration = next(item for item in result.declarations if item.qualified_symbol == "describe")

    assert len(declaration.parameters) == 1
    parameter = declaration.parameters[0]
    assert parameter.kind == "positional"
    assert parameter.position == 0
    assert parameter.destructured is True
    assert "{" not in parameter.name and "}" not in parameter.name


@pytest.mark.parametrize("language", ["typescript", "tsx"])
def test_callback_typed_parameter_is_required_not_defaulted(language: str) -> None:
    """E8: a `=>` inside a parameter's callback type must not misfire the
    text-based default heuristic and mark the parameter optional."""
    source = "function bind(handler: (event: Event) => void, retries: number) { return retries; }\n"
    result = TreeSitterExtractor().extract(Path(f"sample.{language}"), language, source.encode())
    declaration = next(item for item in result.declarations if item.qualified_symbol == "bind")

    assert [(item.name, item.kind, item.required) for item in declaration.parameters] == [
        ("handler", "positional", True),
        ("retries", "positional", True),
    ]


def test_python_member_call_does_not_duplicate_as_member_access() -> None:
    """A member call keeps its single `call` row -- no extra `read` for the same span."""
    source = "widget.render()\n"

    references = _references(source, "python")

    matching = [
        reference
        for reference in references
        if reference.start_byte == source.index("widget.render")
        and reference.end_byte == source.index("widget.render") + len("widget.render")
    ]
    assert [reference.kind for reference in matching] == ["call"]
