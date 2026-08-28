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


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_qualified_class_heritage_is_an_inheritance_reference(language: str) -> None:
    source = "class Child extends ns.Base {}\n"

    references = _references(source, language)
    inheritance = [reference for reference in references if reference.kind == "inheritance"]

    assert [
        (reference.written_name, reference.source_qualified_symbol) for reference in inheritance
    ] == [("ns.Base", "Child")]
    assert not any(
        reference.kind == "read" and reference.written_name == "ns.Base" for reference in references
    )


@pytest.mark.parametrize("language", ["typescript", "tsx"])
def test_typescript_callable_class_members_include_parameter_shapes(language: str) -> None:
    source = (
        "abstract class Base {\n"
        "  abstract run(a: number, b: number): void;\n"
        "  callback = (first: number, second: number): number => first + second;\n"
        "}\n"
    )

    result = TreeSitterExtractor().extract(Path(f"sample.{language}"), language, source.encode())
    declarations = {item.qualified_symbol: item for item in result.declarations}

    for qualified in ("Base.run", "Base.callback"):
        assert [item.name for item in declarations[qualified].parameters] == [
            "a" if qualified == "Base.run" else "first",
            "b" if qualified == "Base.run" else "second",
        ]


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
    assert (
        namespaced.target_name,
        namespaced.written_name,
        namespaced.module_path,
        namespaced.alias,
    ) == (
        "*",
        "ns",
        "./x",
        "ns",
    )
    # The namespace alias must not also surface as a bare `read`.
    assert not any(
        reference.kind == "read" and reference.written_name == "ns" for reference in references
    )


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

    calls = {
        reference.target_name: reference for reference in references if reference.kind == "call"
    }
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
        reference.start_byte: reference for reference in references if reference.kind == "decorator"
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


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "javascript",
            "const LIMIT = 5;\nfunction f(a = LIMIT) {}\n",
        ),
        (
            "typescript",
            "const LIMIT = 5;\nfunction f(a = LIMIT) {}\n",
        ),
        (
            "javascript",
            "const LIMIT = 5;\nconst f = (a = LIMIT) => {};\n",
        ),
    ],
)
def test_js_family_default_parameter_value_is_a_read(language: str, source: str) -> None:
    """A plain `assignment_pattern` default (`a = LIMIT`) exposes its value under
    the `right` field, not `value` -- only TS's typed parameter wrapper uses
    `value`. Missing the `right` field silently dropped every identifier read
    inside an untyped JS/TS default parameter (finding 6)."""
    references = _references(source, language)

    reads = [reference for reference in references if reference.kind == "read"]
    assert [reference.written_name for reference in reads] == ["LIMIT"]
    read = reads[0]
    expected_start = source.rindex("LIMIT")
    assert (read.start_byte, read.end_byte) == (expected_start, expected_start + len("LIMIT"))
    # The parameter's own name must stay excluded -- it is a binding, not a read.
    assert not any(reference.written_name == "a" for reference in references)


def test_python_lambda_default_parameter_value_is_a_read() -> None:
    """Mirrors the JS/TS case above for Python's `lambda a=LIMIT: a` -- the
    outer `lambda`-level exclusion used to blanket-exclude its whole
    `parameters` field, undoing the correct decision the parameter-defaults
    walk already made for the default value."""
    source = "LIMIT = 5\nf = lambda a=LIMIT: a\n"

    references = _references(source, "python")

    reads = [reference for reference in references if reference.kind == "read"]
    read_names = [reference.written_name for reference in reads]
    # `LIMIT` (the default value) and `a` (the body's use of the parameter)
    # are both genuine reads; the parameter *binding* itself is not.
    assert read_names.count("LIMIT") == 1
    assert read_names.count("a") == 1
    limit_read = next(reference for reference in reads if reference.written_name == "LIMIT")
    expected_start = source.index("LIMIT", source.index("lambda"))
    assert (limit_read.start_byte, limit_read.end_byte) == (
        expected_start,
        expected_start + len("LIMIT"),
    )


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", "def g(a, b):\n    pass\n\ng(1,  # note\n  2)\n"),
        ("javascript", "function g(a, b) {}\ng(1, /* c */ 2);\n"),
        ("typescript", "function g(a: number, b: number) {}\ng(1, /* c */ 2);\n"),
    ],
)
def test_call_shape_does_not_count_comments_as_positional_arguments(
    language: str, source: str
) -> None:
    """A comment is a named "extra" node inside the argument list, not an
    argument -- `_call_shape` used to fall into the positional-argument
    else-branch for it, inflating `positional_count` by one per inline
    comment (finding 7)."""
    references = _references(source, language)
    call = next(reference for reference in references if reference.kind == "call")

    assert call.call_shape is not None
    assert call.call_shape.positional_count == 2
    assert call.call_shape.keywords == []


def test_call_shape_keyword_arguments_with_a_comment_stay_correct() -> None:
    """A comment between keyword arguments must not be miscounted as either a
    positional or a keyword argument."""
    source = "def g(a=None, b=None):\n    pass\n\ng(a=1,  # note\n  b=2)\n"

    references = _references(source, "python")
    call = next(reference for reference in references if reference.kind == "call")

    assert call.call_shape is not None
    assert call.call_shape.positional_count == 0
    assert call.call_shape.keywords == ["a", "b"]


def test_module_path_survives_a_leading_comment_in_the_call_arguments() -> None:
    """Same root cause as the comment/positional-argument bug (finding 7), one
    level up: `_string_literal_argument` used `named_child(0)` to find the
    module-path string, so a leading comment (a named "extra" node) was
    mistaken for it and the module edge was silently dropped."""
    source = "require(/* c */ './mod');\n"

    references = _references(source, "javascript")
    call = next(reference for reference in references if reference.kind == "call")

    assert call.module_path == "./mod"


@pytest.mark.parametrize(
    ("language", "source"),
    [
        (
            "python",
            "from pkg import (\n    a,\n    # comment\n    b,\n)\n",
        ),
    ],
)
def test_python_import_list_ignores_a_comment_between_names(language: str, source: str) -> None:
    """A comment among parenthesized `from`-import names is a named "extra"
    node too -- it used to fall through the `aliased_import` check and
    produce a bogus `import` reference for the comment text itself."""
    references = _references(source, language)

    imports = [reference for reference in references if reference.kind == "import"]
    assert [reference.written_name for reference in imports] == ["a", "b"]


def test_python_class_heritage_ignores_a_comment_between_base_classes() -> None:
    """Mirrors the import-list case for `class Child(Base, # note\\n Other):`
    -- a comment between base classes must not surface as a bogus
    `inheritance` reference."""
    source = "class Child(\n    Base,\n    # comment\n    Other,\n):\n    pass\n"

    references = _references(source, "python")

    inheritance = [reference for reference in references if reference.kind == "inheritance"]
    assert {reference.written_name for reference in inheritance} == {"Base", "Other"}


def test_js_namespace_import_alias_survives_a_leading_comment() -> None:
    """`import * as /* c */ ns from 'mod'` -- the alias used to resolve via
    `named_child(0)`, which picked the comment instead of `ns`, corrupting
    both the alias and the written name."""
    source = "import * as /* c */ ns from 'mod';\n"

    references = _references(source, "javascript")
    imported = next(reference for reference in references if reference.kind == "import")

    assert (imported.written_name, imported.alias) == ("ns", "ns")


def test_js_namespace_export_alias_survives_a_leading_comment() -> None:
    """Mirrors the namespace-import case for `export * as /* c */ ns from './x'`."""
    source = "export * as /* c */ ns from './x';\n"

    references = _references(source, "javascript")
    exported = next(reference for reference in references if reference.kind == "export")

    assert exported.written_name == "ns"


def test_js_decorator_target_survives_a_leading_comment() -> None:
    """`@/* c */ dec` -- the decorator target used to resolve via
    `named_child(0)`, which picked the comment instead of `dec`."""
    source = "class A {\n  @/* c */ dec\n  method() {}\n}\n"

    references = _references(source, "javascript")
    decorator = next(reference for reference in references if reference.kind == "decorator")

    assert decorator.written_name == "dec"


def test_js_default_import_is_not_fabricated_from_a_comment() -> None:
    """`import Default, /* c */ { a } from 'mod'` -- a comment between clause
    items used to fall into the "bare identifier" else-branch and produce a
    bogus second `default` import naming the comment text."""
    source = "import Default, /* c */ { a } from 'mod';\n"

    references = _references(source, "javascript")
    imports = [reference for reference in references if reference.kind == "import"]

    assert sorted(reference.written_name for reference in imports) == ["Default", "a"]


def test_ts_extends_type_clause_ignores_a_comment() -> None:
    """`interface I extends /* c */ Base {}` -- a comment must not surface as
    a bogus `inheritance` reference alongside the real one."""
    source = "interface I extends /* c */ Base {}\n"

    references = _references(source, "typescript")
    inheritance = [reference for reference in references if reference.kind == "inheritance"]

    assert [reference.written_name for reference in inheritance] == ["Base"]


def test_ts_type_annotation_survives_a_leading_comment() -> None:
    """`x: /* c */ Widget` -- the annotated type used to resolve via
    `named_child(0)`, which picked the comment and silently dropped the
    `type_use` row for `Widget` entirely."""
    source = "function f(x: /* c */ Widget) {}\n"

    references = _references(source, "typescript")
    type_uses = [reference for reference in references if reference.kind == "type_use"]

    assert [reference.written_name for reference in type_uses] == ["Widget"]


def test_call_shape_type_argument_count_ignores_a_comment() -> None:
    """A comment among explicit type arguments (`build</* c */ T>(value)`)
    must not inflate `type_argument_count`."""
    source = "function make<T>(value: T): Contract<T> { return build</* c */ T>(value); }\n"

    references = _references(source, "typescript")
    call = next(
        reference
        for reference in references
        if reference.kind == "call" and reference.target_name == "build"
    )

    assert call.call_shape is not None
    assert call.call_shape.type_argument_count == 1


def test_js_rest_parameter_name_survives_a_leading_comment() -> None:
    """`function f(.../* c */ rest) {}` -- the rest parameter's name used to
    resolve via `named_child(0)` in `_parameter_shapes`, which picked the
    comment instead of `rest`."""
    source = "function f(.../* c */ rest) {}\n"

    result = TreeSitterExtractor().extract(Path("sample.js"), "javascript", source.encode())
    declaration = next(item for item in result.declarations if item.qualified_symbol == "f")

    assert [(item.name, item.kind) for item in declaration.parameters] == [("rest", "variadic")]


def test_js_destructured_rest_binding_survives_a_leading_comment() -> None:
    """`[.../* c */ rest]` -- `_binding_identifiers`' `rest_pattern` case used
    `named_child(0)`, which picked a comment placed right after `...` instead
    of the bound identifier, losing the export's real name."""
    source = "export const [.../* c */ rest] = arr;\n"

    references = _references(source, "javascript")
    exported = next(reference for reference in references if reference.kind == "export")

    assert exported.written_name == "rest"


# ---------------------------------------------------------------------------
# Go structural references (language step 1)
# ---------------------------------------------------------------------------


def _go_result(source: str):
    return TreeSitterExtractor().extract(Path("sample.go"), "go", source.encode())


def test_go_extracts_import_shapes() -> None:
    """Plain, aliased, grouped, and dot imports produce one row each with
    namespace semantics: `module_path` is the import path and the bound local
    spelling is the alias or the conventional package name."""
    source = (
        "package sample\n\n"
        'import "fmt"\n'
        'import st "app/store"\n'
        "import (\n"
        '\t"bytes"\n'
        '\t. "app/util"\n'
        ")\n"
    )

    refs = {r.written_name: r for r in _go_result(source).references if r.kind == "import"}

    assert refs["fmt"].module_path == "fmt"
    assert refs["fmt"].imported_name is None and refs["fmt"].alias is None
    assert (refs["st"].module_path, refs["st"].alias) == ("app/store", "st")
    assert refs["bytes"].module_path == "bytes"
    dot = refs["*"]
    assert (dot.module_path, dot.imported_name, dot.alias) == ("app/util", "*", None)


def test_go_calls_and_call_shapes() -> None:
    source = "package sample\n\nfunc run() {\n\tbuild(1)\n\tw.Draw(true)\n}\n"

    calls = {r.target_name: r for r in _go_result(source).references if r.kind == "call"}

    assert calls["build"].receiver_text is None
    assert calls["build"].call_shape is not None
    assert calls["build"].call_shape.positional_count == 1
    assert calls["w.Draw"].receiver_text == "w"
    assert calls["w.Draw"].source_qualified_symbol == "run"
    assert calls["w.Draw"].call_shape is not None
    assert calls["w.Draw"].call_shape.positional_count == 1


def test_go_method_receiver_is_parameter_slot_zero() -> None:
    source = (
        "package sample\n\n"
        "func (s *Store) Get(key string) (int, error) {\n"
        "\treturn 0, nil\n"
        "}\n"
        "func Total(items ...string) int {\n"
        "\treturn len(items)\n"
        "}\n"
    )

    declarations = {d.qualified_symbol: d for d in _go_result(source).declarations}

    get_params = [(p.name, p.kind, p.required, p.position) for p in declarations["Get"].parameters]
    assert get_params == [("s", "positional", True, 0), ("key", "positional", True, 1)]
    total = [(p.name, p.kind, p.required, p.position) for p in declarations["Total"].parameters]
    assert total == [("items", "variadic", False, 0)]
    receiver_type_use = next(
        r
        for r in _go_result(source).references
        if r.kind == "type_use" and r.target_name == "Store"
    )
    assert receiver_type_use.source_qualified_symbol == "Get"


def test_go_member_access_reads_and_writes() -> None:
    source = "package sample\n\nfunc set(s *Store) {\n\ts.next = nil\n\t_ = s.items\n}\n"

    refs = _go_result(source).references
    write = next(r for r in refs if r.kind == "write")
    assert write.target_name == "s.next"
    read = next(r for r in refs if r.kind == "read" and "." in r.target_name)
    assert read.target_name == "s.items"


def test_go_short_var_and_assignment_lefts_are_not_reads() -> None:
    """`:=` declares, `=` writes -- neither side's bare identifiers may claim a
    bogus read; the selector on an assignment's left still becomes a write.
    A *later* use of the bound name (`if ok`) stays a genuine read."""
    source = (
        "package sample\n\n"
        "func go_around(counter Counter) {\n"
        "\ttotal, ok := counter.Count()\n"
        "\tcounter.Total = total\n"
        "\tif ok {\n"
        "\t\t_ = total\n"
        "\t}\n"
        "}\n"
    )

    refs = [r for r in _go_result(source).references if r.source_qualified_symbol == "go_around"]
    assert not any(r.kind == "read" and r.start_byte == source.index("total, ok") for r in refs)
    write = next(r for r in refs if r.kind == "write")
    assert write.target_name == "counter.Total"
    assert [r.start_byte for r in refs if r.kind == "read" and r.written_name == "ok"] == [
        source.index("if ok") + len("if ")
    ]


def test_go_embedded_fields_are_inheritance_edges() -> None:
    source = "package sample\n\ntype Store struct {\n\tReader\n\tname string\n}\n"

    inheritance = [r for r in _go_result(source).references if r.kind == "inheritance"]

    assert [r.written_name for r in inheritance] == ["Reader"]
    assert all(r.source_qualified_symbol == "Store" for r in inheritance)


def test_go_types_become_type_use_rows() -> None:
    """Parameter/result/field/var/composite-literal types emit one type_use per
    project type; predeclared types (`string`, `int`) stay out."""
    source = (
        "package sample\n\n"
        "var cache map[string]Widget\n\n"
        "func build(w Widget) *Widget {\n"
        "\tout := &Widget{}\n"
        "\treturn out\n"
        "}\n"
    )

    type_uses = [r for r in _go_result(source).references if r.kind == "type_use"]
    names = [r.written_name for r in type_uses]
    assert names.count("Widget") >= 3
    assert not any(name in {"string", "int"} for name in names)


def test_go_exports_capitalized_top_level_names_only() -> None:
    source = (
        "package sample\n\n"
        "const Limit = 10\n"
        "var total int\n"
        "type Widget struct{}\n"
        "type helper struct{}\n"
        "func Build() {}\n"
        "func internal() {}\n"
    )

    exports = [r for r in _go_result(source).references if r.kind == "export"]

    assert sorted(r.written_name for r in exports) == ["Build", "Limit", "Widget"]


def test_go_qualified_type_annotations_name_their_type_identifier() -> None:
    """`pkg.Type` must contribute its final identifier as a `type_use`.

    tree-sitter-go names the two `qualified_type` sides `package`/`name` --
    there is no `type` field, so descending it silently dropped every
    cross-package type annotation."""
    source = (
        "package sample\n\n"
        'import st "app/store"\n\n'
        "var view st.View\n\n"
        "func build(s st.Store) *st.Item {\n"
        "\treturn nil\n"
        "}\n"
    )

    type_uses = [r for r in _go_result(source).references if r.kind == "type_use"]

    assert sorted(r.written_name for r in type_uses) == ["Item", "Store", "View"]


def test_go_qualified_embedding_is_an_inheritance_edge() -> None:
    """A qualified embedded type (`st.Config` in a struct, `st.Closer` in an
    interface) is promoted just like an unqualified one."""
    source = (
        "package sample\n\n"
        'import st "app/store"\n\n'
        "type Local struct {\n"
        "\tst.Config\n"
        "}\n\n"
        "type Handler interface {\n"
        "\tst.Closer\n"
        "\tPlain() error\n"
        "}\n"
    )

    inheritance = [r for r in _go_result(source).references if r.kind == "inheritance"]

    assert sorted(r.written_name for r in inheritance) == ["Closer", "Config"]


def test_go_grouped_var_exports_like_the_flat_form() -> None:
    """`var ( ... )` wraps its specs in a `var_spec_list`, unlike const/type
    groups whose specs sit directly under the declaration -- the group must
    still export exactly its capitalized names."""
    source = "package sample\n\nvar (\n\tCounter = 1\n\thidden = 2\n)\n"

    exports = [r for r in _go_result(source).references if r.kind == "export"]

    assert [r.written_name for r in exports] == ["Counter"]


def test_go_const_spec_names_are_bindings_not_reads() -> None:
    """A const spec's own name must not leak a `read` row at the declaration
    site (the var_spec equivalent was always cut); a real use of the const
    stays a read."""
    source = "package sample\n\nconst Limit = 10\n\nfunc use() int {\n\treturn Limit\n}\n"

    refs = _go_result(source).references

    assert not any(r.kind == "read" and r.source_qualified_symbol is None for r in refs)
    use_reads = [r for r in refs if r.kind == "read" and r.target_name == "Limit"]
    assert [r.source_qualified_symbol for r in use_reads] == ["use"]


def test_go_inc_dec_statements_are_writes() -> None:
    """`item.count++` mutates through the selector, so it is a write; the bare
    operand of `counter--` is likewise a mutation site, not a read."""
    source = (
        "package sample\n\n"
        "func tick(item Item) {\n"
        "\titem.count++\n"
        "\tcounter := 0\n"
        "\tcounter--\n"
        "\t_ = counter\n"
        "}\n"
    )

    refs = [r for r in _go_result(source).references if r.source_qualified_symbol == "tick"]

    writes = [r for r in refs if r.kind == "write"]
    assert [r.target_name for r in writes] == ["item.count"]
    assert not any(r.kind == "read" and r.start_byte == source.index("counter--") for r in refs)


def test_go_bare_result_type_is_a_type_use() -> None:
    """`func Load() Item`: the unparenthesized result hangs directly off the
    declaration, so the handler owns its `type_use` -- and the identifier
    fallback must not double it as a plain read."""
    source = "package sample\n\ntype Item struct{}\n\nfunc Load() Item {\n\treturn Item{}\n}\n"

    refs = _go_result(source).references
    result_use = next(r for r in refs if r.kind == "type_use" and r.written_name == "Item")

    assert result_use.source_qualified_symbol == "Load"
    assert not any(
        r.kind == "read" and r.written_name == "Item" and r.start_byte == result_use.start_byte
        for r in refs
    )


def test_go_interface_method_elem_types_are_type_uses() -> None:
    """Interface method elements contribute their parameter and result types
    -- through the `parameter_list` wrappers, qualified names included --
    but never their method names."""
    source = (
        "package sample\n\n"
        'import st "app/store"\n\n'
        "type Handler interface {\n"
        "\tServe(st.Item) error\n"
        "\tNamed(item st.Item) st.Result\n"
        "\tPlain() error\n"
        "}\n"
    )

    type_uses = [r for r in _go_result(source).references if r.kind == "type_use"]

    assert sorted(r.written_name for r in type_uses) == ["Item", "Item", "Result"]


# ---------------------------------------------------------------------------
# Rust structural references (language step 2)
# ---------------------------------------------------------------------------


def _rust_result(source: str):
    return TreeSitterExtractor().extract(Path("sample.rs"), "rust", source.encode())


def test_rust_extracts_use_shapes() -> None:
    """Plain, grouped, aliased, glob, and crate/super/self uses produce one
    row per binding; `module_path` is the path minus its final segment and a
    glob carries `imported_name="*"` so the wildcard gate owns it."""
    source = (
        "use crate::app::store::Saver;\n"
        "use crate::app::{util::limit, fmt::show};\n"
        "use std::io::Result as IoResult;\n"
        "use crate::app::util::*;\n"
        "use super::helper::assist;\n"
        "use self::inner::Local;\n"
    )

    refs = {r.written_name: r for r in _rust_result(source).references if r.kind == "import"}

    assert refs["Saver"].module_path == "crate::app::store"
    assert refs["limit"].module_path == "crate::app::util"
    assert refs["show"].module_path == "crate::app::fmt"
    assert (refs["IoResult"].module_path, refs["IoResult"].alias) == ("std::io", "IoResult")
    assert refs["IoResult"].imported_name == "Result"
    glob = refs["*"]
    assert (glob.module_path, glob.imported_name, glob.alias) == ("crate::app::util", "*", None)
    assert refs["assist"].module_path == "super::helper"
    assert refs["Local"].module_path == "self::inner"


def test_rust_calls_and_call_shapes() -> None:
    """Plain, path-qualified, method, and turbofish calls all carry a shape;
    path segments join with dots (Python/Go convention) and the receiver
    keeps its `::` spelling."""
    source = (
        "fn run(input: u32) {\n"
        "\tbuild(input);\n"
        "\tSaver::save(&input);\n"
        "\tw.draw();\n"
        "\tlet xs = Vec::<u8>::new();\n"
        "}\n"
    )

    calls = {r.target_name: r for r in _rust_result(source).references if r.kind == "call"}

    assert calls["build"].receiver_text is None
    assert calls["build"].call_shape is not None
    assert calls["build"].call_shape.positional_count == 1
    assert calls["Saver.save"].receiver_text == "Saver"
    assert calls["Saver.save"].source_qualified_symbol == "run"
    assert calls["w.draw"].receiver_text == "w"
    turbofish = calls["Vec.new"]
    assert turbofish.receiver_text == "Vec"
    assert turbofish.call_shape is not None
    assert turbofish.call_shape.positional_count == 0


def test_rust_impl_methods_qualify_and_self_calls_carry_the_owner() -> None:
    """Methods inside `impl Widget` qualify `Widget.method`, so a
    `self.helper()` call carries the owner-qualified enclosing symbol that
    `_same_owner` needs for an exact verdict."""
    source = (
        "struct Widget;\n\n"
        "impl Widget {\n"
        "\tfn helper(&self) -> u32 {\n\t\t1\n\t}\n"
        "\tfn run(&self) -> u32 {\n"
        "\t\tself.helper()\n"
        "\t}\n"
        "}\n"
    )

    result = _rust_result(source)
    declarations = {d.qualified_symbol: d for d in result.declarations}

    assert set(declarations) == {"Widget", "Widget.helper", "Widget.run"}
    assert declarations["Widget.run"].kind == "method"
    helper_call = next(
        r for r in result.references if r.kind == "call" and r.target_name == "self.helper"
    )
    assert helper_call.source_qualified_symbol == "Widget.run"


def test_rust_method_parameters_and_self_slot_zero() -> None:
    source = (
        "struct Store;\n\n"
        "impl Store {\n"
        "\tfn get(&self, key: &str) -> u32 {\n\t\t0\n\t}\n"
        "}\n"
        "fn total(items: Vec<String>) -> usize {\n\titems.len()\n}\n"
    )

    declarations = {d.qualified_symbol: d for d in _rust_result(source).declarations}

    get_params = [
        (p.name, p.kind, p.required, p.position) for p in declarations["Store.get"].parameters
    ]
    assert get_params == [("self", "positional", True, 0), ("key", "positional", True, 1)]
    total = [(p.name, p.kind, p.required, p.position) for p in declarations["total"].parameters]
    assert total == [("items", "positional", True, 0)]


def test_rust_member_access_reads_and_writes() -> None:
    source = "fn set(s: &mut Store) {\n\ts.count = 0;\n\tlet n = s.count;\n}\n"

    refs = _rust_result(source).references
    write = next(r for r in refs if r.kind == "write")
    assert write.target_name == "s.count"
    read = next(r for r in refs if r.kind == "read" and "." in r.target_name)
    assert read.target_name == "s.count"


def test_rust_let_and_assignment_bindings_are_not_reads() -> None:
    """`let` declares, `=` and `+=` write -- pattern names may not claim
    reads, and a later use of the binding stays a genuine read."""
    source = (
        "fn go_around(counter: Counter) {\n"
        "\tlet total = counter.count();\n"
        "\tlet mut ok = false;\n"
        "\tok = true;\n"
        "\tok += counter.more();\n"
        "\tif ok {\n"
        "\t\tlet _ = total;\n"
        "\t}\n"
        "}\n"
    )

    refs = [r for r in _rust_result(source).references if r.source_qualified_symbol == "go_around"]
    ok_reads = [r.start_byte for r in refs if r.kind == "read" and r.written_name == "ok"]
    assert ok_reads == [source.index("if ok") + len("if ")]
    total_reads = [r.start_byte for r in refs if r.kind == "read" and r.written_name == "total"]
    assert total_reads == [source.index("= total") + len("= ")]
    # A bare LHS identifier is a pure binding, not a write row -- only member
    # writes become rows (the same rule Go and Python ship).
    assert not any(r.kind == "write" for r in refs)


def test_rust_trait_impl_is_an_inheritance_edge() -> None:
    source = (
        "trait Draw { fn draw(&self) -> String; }\n"
        "struct Widget;\n"
        "impl Draw for Widget {\n"
        "\tfn draw(&self) -> String {\n\t\tString::new()\n\t}\n"
        "}\n"
    )

    inheritance = [r for r in _rust_result(source).references if r.kind == "inheritance"]

    assert [r.written_name for r in inheritance] == ["Draw"]


def test_rust_exports_pub_items_only() -> None:
    source = (
        "pub struct Client;\n"
        "struct Hidden;\n"
        "pub enum Mode {\n\tOn\n}\n"
        "pub fn serve() {}\n"
        "fn helper() {}\n"
        "pub const LIMIT: u32 = 1;\n"
        "const PRIVATE: u32 = 2;\n"
    )

    exports = [r.written_name for r in _rust_result(source).references if r.kind == "export"]

    assert exports == ["Client", "Mode", "serve", "LIMIT"]


def test_rust_pub_use_export_carries_the_module_path() -> None:
    """A `pub use` re-export emits an export row with its module path, import
    name, and alias so the re-export chain walker can hop through it."""
    source = "pub use crate::app::api::publicate as publish;\n"

    refs = _rust_result(source).references
    export = next(r for r in refs if r.kind == "export")
    assert (export.written_name, export.alias) == ("publish", "publish")
    assert (export.module_path, export.imported_name) == ("crate::app::api", "publicate")


def test_rust_types_become_type_use_rows() -> None:
    """Types in fields, parameters, returns, `let` annotations, generic
    arguments, and trait objects become `type_use`; primitives never do."""
    source = (
        "struct Pair<T> {\n"
        "\tleft: T,\n"
        "\tright: Vec<Box<dyn Draw>>,\n"
        "}\n"
        "fn load(input: &str) -> Option<Pair<u32>> {\n"
        "\tlet local: Pair<u32> = load_pair();\n"
        "\tSome(local)\n"
        "}\n"
    )

    type_uses = [r.written_name for r in _rust_result(source).references if r.kind == "type_use"]

    for expected in ("T", "Vec", "Box", "Draw", "Pair", "Option"):
        assert expected in type_uses
    assert "str" not in type_uses
    assert "u32" not in type_uses
