"""Regression tests for the Track E review findings (REF-1 .. REF-12).

Each test rebuilds a trigger from `docs/plans/2026-11-10-performance-and-bug-review.md`
and asserts on precision (what is `exact`) or on rename-patch honesty (a patch
that reports `complete` must leave working code).
"""

from pathlib import Path

import pyarrow as pa
import pytest

from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import (
    DeclarationSelector,
    ParameterShape,
    RefactorPatch,
    RenameOperation,
    SignatureChangeOperation,
)
from code_indexing_mcp.patching import (
    ByteEdit,
    PatchEncodingError,
    apply_edits,
    render_unified_diff,
)
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


class TinyEmbedder:
    model_id = "test/binding-precision"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def _indexed(tmp_path: Path, files: dict[str, str | bytes]) -> tuple[ReferenceService, str, Path]:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    for path, source in files.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source if isinstance(source, bytes) else source.encode("utf-8"))
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=TinyEmbedder(),
        lock_directory=tmp_path / "locks",
    ).index(project)
    return ReferenceService(store), project.id, root


def _selector(project_id: str, path: str, symbol: str) -> DeclarationSelector:
    return DeclarationSelector(project=project_id, path=path, qualified_symbol=symbol)


def _apply(root: Path, patch: RefactorPatch) -> dict[str, str]:
    """Apply the patch edits to the on-disk sources and return the resulting texts."""
    out: dict[str, str] = {}
    by_path: dict[str, list[tuple[int, int, str]]] = {}
    for edit in patch.edits:
        by_path.setdefault(edit.path, []).append(
            (edit.edit_start_byte, edit.edit_end_byte, edit.new_text)
        )
    for path, edits in by_path.items():
        data = (root / path).read_bytes()
        for start, end, text in sorted(edits, reverse=True):
            data = data[:start] + text.encode() + data[end:]
        out[path] = data.decode()
    return out


# --- REF-1 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("lib_path", "lib", "main_path", "main"),
    [
        (
            "lib.py",
            "def helper(): pass\ndef other(): pass\ndef unused(): pass\nCONST = 1\n",
            "main.py",
            "import lib\n\ndef go():\n    return lib.other() + lib.CONST\n",
        ),
        (
            "lib.ts",
            "export function helper() {}\nexport function other() {}\n"
            "export function unused() {}\n",
            "main.ts",
            "import * as lib from './lib';\nexport function go() { return lib.other(); }\n",
        ),
    ],
)
def test_namespace_import_binds_only_the_named_member(
    tmp_path: Path, lib_path: str, lib: str, main_path: str, main: str
) -> None:
    service, project_id, _root = _indexed(tmp_path, {lib_path: lib, main_path: main})

    unused = service.find_references(_selector(project_id, lib_path, "unused"))
    assert [hit for hit in unused.hits if hit.path == main_path] == []

    other = service.find_references(_selector(project_id, lib_path, "other"))
    call = next(hit for hit in other.hits if hit.path == main_path and hit.kind == "call")
    assert (call.resolution, call.reason_code) == ("exact", "known_namespace_member")

    report = service.dead_code_report(project_id)
    flagged = {item.declaration.qualified_symbol for item in report.review}
    assert "unused" in flagged
    assert "other" not in flagged

    radius = service.impact_radius(_selector(project_id, lib_path, "unused"))
    assert [
        edge for layer in radius.layers for edge in layer.edges if edge.source.path == main_path
    ] == []


# --- REF-2 -----------------------------------------------------------------


def _member_hit(service: ReferenceService, project_id: str, path: str, symbol: str, hit_path: str):
    response = service.find_references(_selector(project_id, path, symbol))
    return next(hit for hit in response.hits if hit.path == hit_path and hit.kind == "call")


def test_self_call_in_a_same_named_class_in_another_file_is_not_exact(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "a.py": "class Service:\n    def start(self):\n        return 1\n",
            "b.py": (
                "class Service:\n    def start(self):\n        return 2\n\n"
                "    def run(self):\n        return self.start()\n"
            ),
        },
    )
    call = _member_hit(service, project_id, "a.py", "Service.start", "b.py")
    assert call.resolution == "likely"
    assert call.reason_code == "unknown_receiver"

    patch = service.emit_refactor_patch(
        _selector(project_id, "a.py", "Service.start"), RenameOperation(new_name="begin")
    )
    assert all(edit.path == "a.py" for edit in patch.edits)
    assert "b.py" not in patch.patch
    # The unproven call is surfaced for review, so the patch cannot read as finished.
    assert patch.completeness.state != "complete"
    assert any(item.path == "b.py" for item in patch.unapplied)


def test_self_call_in_a_nested_class_is_not_the_outer_classs_member(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "n.py": (
                "class Outer:\n    def helper(self):\n        return 1\n\n"
                "    class Inner:\n        def run(self):\n            return self.helper()\n"
            )
        },
    )
    call = _member_hit(service, project_id, "n.py", "Outer.helper", "n.py")
    assert call.resolution == "likely"


def test_self_call_in_the_owning_class_stays_exact(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "a.py": (
                "class Service:\n    def start(self):\n        return 1\n\n"
                "    def run(self):\n        def inner():\n            return self.start()\n"
                "        return self.start()\n"
            )
        },
    )
    response = service.find_references(_selector(project_id, "a.py", "Service.start"))
    calls = [hit for hit in response.hits if hit.kind == "call"]
    assert len(calls) == 2
    assert {hit.reason_code for hit in calls} == {"known_owner_member"}


def test_typescript_this_call_in_a_same_named_class_in_another_file_is_not_exact(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "a.ts": "export class Service { start() { return 1; } }\n",
            "b.ts": "class Service { start() { return 2; } run() { return this.start(); } }\n",
        },
    )
    call = _member_hit(service, project_id, "a.ts", "Service.start", "b.ts")
    assert call.resolution == "likely"


_CSHARP_PARTIAL = {
    "Widget.cs": "namespace Demo { public partial class Widget { public void Start() {} } }\n",
    "Widget.Run.cs": (
        "namespace Demo { public partial class Widget { public void Run() { this.Start(); } } }\n"
    ),
}


def test_csharp_partial_class_this_call_stays_exact_in_the_same_namespace(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(tmp_path, _CSHARP_PARTIAL)
    call = _member_hit(service, project_id, "Widget.cs", "Widget.Start", "Widget.Run.cs")
    assert (call.resolution, call.reason_code) == ("exact", "known_owner_member")


_CSHARP_SERVICE_START = "namespace Demo { public %s Service%s { public void Start() {} } }\n"
_CSHARP_SERVICE_RUN = (
    "namespace Demo { public %s Service%s { public void Run() { this.Start(); } } }\n"
)


@pytest.mark.parametrize(
    ("start_file", "run_file"),
    [
        # generic arity differs: `Service` and `Service<T>` are distinct types
        (
            _CSHARP_SERVICE_START % ("partial class", ""),
            _CSHARP_SERVICE_RUN % ("partial class", "<T>"),
        ),
        # one project's non-partial class vs another's in the same namespace
        (
            _CSHARP_SERVICE_START % ("class", ""),
            _CSHARP_SERVICE_RUN % ("class", ""),
        ),
        # only one side is partial
        (
            _CSHARP_SERVICE_START % ("partial class", ""),
            _CSHARP_SERVICE_RUN % ("class", ""),
        ),
    ],
)
def test_csharp_this_call_across_distinct_same_named_types_is_not_exact(
    tmp_path: Path, start_file: str, run_file: str
) -> None:
    service, project_id, _root = _indexed(
        tmp_path, {"Service.cs": start_file, "Other/Service.cs": run_file}
    )
    call = _member_hit(service, project_id, "Service.cs", "Service.Start", "Other/Service.cs")
    assert call.resolution == "likely"

    selector = _selector(project_id, "Service.cs", "Service.Start")
    patch = service.emit_refactor_patch(selector, RenameOperation(new_name="Begin"))
    assert all(edit.path == "Service.cs" for edit in patch.edits)
    assert "Other/Service.cs" not in patch.patch
    assert patch.completeness.state != "complete"
    assert any(item.path == "Other/Service.cs" for item in patch.unapplied)


def test_csharp_partial_class_with_matching_generic_arity_stays_exact(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "Service.cs": _CSHARP_SERVICE_START % ("partial class", "<T>"),
            "Service.Run.cs": _CSHARP_SERVICE_RUN % ("sealed partial class", "<T>"),
        },
    )
    call = _member_hit(service, project_id, "Service.cs", "Service.Start", "Service.Run.cs")
    assert (call.resolution, call.reason_code) == ("exact", "known_owner_member")


def test_csharp_third_non_partial_declaration_breaks_a_partial_pair(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "Widget.cs": _CSHARP_PARTIAL["Widget.cs"],
            "Widget.Run.cs": _CSHARP_PARTIAL["Widget.Run.cs"],
            "Elsewhere/Widget.cs": "namespace Demo { public class Widget { } }\n",
        },
    )
    call = _member_hit(service, project_id, "Widget.cs", "Widget.Start", "Widget.Run.cs")
    assert call.resolution == "likely"


def test_csharp_same_named_class_in_another_namespace_is_not_exact(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "Widget.cs": _CSHARP_PARTIAL["Widget.cs"],
            "Other.cs": (
                "namespace Elsewhere { public class Widget { public void Run() { this.Start(); }"
                " public void Start() {} } }\n"
            ),
        },
    )
    call = _member_hit(service, project_id, "Widget.cs", "Widget.Start", "Other.cs")
    assert call.resolution == "likely"


def test_rust_impl_in_another_file_is_exact_only_when_the_type_is_unique(tmp_path: Path) -> None:
    files = {
        "a.rs": "pub struct S;\nimpl S { pub fn helper(&self) {} }\n",
        "b.rs": "use crate::a::S;\nimpl S { fn run(&self) { self.helper(); } }\n",
    }
    service, project_id, _root = _indexed(tmp_path / "unique", files)
    call = _member_hit(service, project_id, "a.rs", "S.helper", "b.rs")
    assert (call.resolution, call.reason_code) == ("exact", "known_owner_member")

    service, project_id, _root = _indexed(
        tmp_path / "duplicate", {**files, "c.rs": "pub struct S;\n"}
    )
    call = _member_hit(service, project_id, "a.rs", "S.helper", "b.rs")
    assert call.resolution == "likely"


# --- REF-3 -----------------------------------------------------------------


def test_class_rename_edits_the_receiver_of_a_static_member_call(tmp_path: Path) -> None:
    files = {
        "lib.py": ("class Factory:\n    @staticmethod\n    def make():\n        return 1\n"),
        "main.py": "from lib import Factory\n\n\ndef go():\n    return Factory.make()\n",
    }
    service, project_id, root = _indexed(tmp_path, files)

    patch = service.emit_refactor_patch(
        _selector(project_id, "lib.py", "Factory"), RenameOperation(new_name="Maker")
    )

    edited = _apply(root, patch)
    assert edited["main.py"] == ("from lib import Maker\n\n\ndef go():\n    return Maker.make()\n")
    assert edited["lib.py"].startswith("class Maker:")
    assert patch.completeness.state == "complete"
    assert patch.unapplied == [] and patch.conflicted == []


# --- REF-4 -----------------------------------------------------------------

_OVERLOADED = (
    "from typing import overload\n\n\n"
    "@overload\ndef conv(x: int) -> int: ...\n\n\n"
    "@overload\ndef conv(x: str) -> str: ...\n\n\n"
    "def conv(x):\n    return x\n"
)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_rename_of_an_overload_set_renames_every_declaration(tmp_path: Path, which: int) -> None:
    service, project_id, root = _indexed(
        tmp_path,
        {
            "m.py": _OVERLOADED,
            "use.py": "from m import conv\n\n\ndef go():\n    return conv(1)\n",
        },
    )
    chunks = sorted(
        (chunk for chunk in service.store.find_declarations(project_id, "m.py", "conv")),
        key=lambda chunk: chunk.start_line,
    )
    assert len(chunks) == 3
    selector = DeclarationSelector(chunk_id=chunks[which].chunk_id)

    patch = service.emit_refactor_patch(selector, RenameOperation(new_name="convert"))

    edited = _apply(root, patch)
    assert edited["m.py"].count("def convert(") == 3
    assert "def conv(" not in edited["m.py"]
    assert edited["use.py"] == "from m import convert\n\n\ndef go():\n    return convert(1)\n"
    assert patch.unapplied == [] and patch.conflicted == []
    assert patch.completeness.state == "complete"
    analysis = service.analyze_refactor(selector, RenameOperation(new_name="convert"))
    ids = [finding.reference_id for finding in analysis.must_change]
    assert len(ids) == len(set(ids))
    assert analysis.counts.must_change == len(analysis.must_change)


def test_rename_of_conditional_definitions_renames_both_branches(tmp_path: Path) -> None:
    service, project_id, root = _indexed(
        tmp_path,
        {
            "sep.py": (
                "import sys\n\nif sys.platform == 'win32':\n    def sep():\n        return '\\\\'\n"
                "else:\n    def sep():\n        return '/'\n"
            )
        },
    )
    chunk = service.store.find_declarations(project_id, "sep.py", "sep")[0]
    patch = service.emit_refactor_patch(
        DeclarationSelector(chunk_id=chunk.chunk_id),
        RenameOperation(new_name="separator"),
    )
    edited = _apply(root, patch)
    assert edited["sep.py"].count("def separator()") == 2
    assert "def sep()" not in edited["sep.py"]
    assert patch.completeness.state == "complete"


def test_rename_of_a_property_renames_getter_setter_and_the_accessor_decorator(
    tmp_path: Path,
) -> None:
    source = (
        "class C:\n    @property\n    def value(self):\n        return 1\n\n"
        "    @value.setter\n    def value(self, new):\n        pass\n"
    )
    service, project_id, root = _indexed(tmp_path, {"c.py": source})
    chunk = service.store.find_declarations(project_id, "c.py", "C.value")[0]

    patch = service.emit_refactor_patch(
        DeclarationSelector(chunk_id=chunk.chunk_id), RenameOperation(new_name="amount")
    )

    assert _apply(root, patch)["c.py"] == source.replace("value", "amount")
    assert patch.completeness.state == "complete"


# --- REF-6 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("lib_path", "source", "symbol"),
    [
        ("lib.py", "def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\n", "fact"),
        (
            "lib.py",
            'def fact(n):\n    """fact computes n! ; fact(3) == 6"""\n    return n\n',
            "fact",
        ),
        (
            "lib.py",
            "class Node:\n    def clone(self):\n        return Node()\n",
            "Node",
        ),
    ],
)
def test_declaration_edit_survives_a_body_that_mentions_its_own_name(
    tmp_path: Path, lib_path: str, source: str, symbol: str
) -> None:
    service, project_id, root = _indexed(tmp_path, {lib_path: source})

    patch = service.emit_refactor_patch(
        _selector(project_id, lib_path, symbol), RenameOperation(new_name="renamed")
    )

    edited = _apply(root, patch)[lib_path]
    declaration_line = edited.splitlines()[0]
    assert "renamed" in declaration_line
    assert symbol not in declaration_line
    assert not any(item.reason_code == "declaration" for item in patch.unapplied)


# --- REF-5 -----------------------------------------------------------------


def _long_function(name: str, lines: int = 90) -> str:
    body = "".join(
        f"    total_{index} = {index} * 2  # padding to force several chunks\n"
        for index in range(lines)
    )
    return f"def {name}(a, b):\n{body}    return a + b\n"


def test_a_declaration_split_into_parts_is_selectable_by_qualified_symbol(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "big.py": _long_function("big"),
            "use.py": "from big import big\n\n\ndef go():\n    return big(1, 2)\n",
        },
    )
    parts = service.store.find_declarations(project_id, "big.py", "big")
    assert len(parts) > 1 and any(part.kind.endswith("_part") for part in parts)

    by_symbol = service.find_references(_selector(project_id, "big.py", "big"))
    assert by_symbol.selected.kind == "function"
    assert by_symbol.selected.start_line == 1
    assert by_symbol.selected.end_line == max(part.end_line for part in parts)
    assert any(hit.path == "use.py" and hit.kind == "call" for hit in by_symbol.hits)
    assert service.impact_radius(_selector(project_id, "big.py", "big")).selected.kind == "function"

    # A later part selected by chunk_id normalises to the same whole declaration.
    last = max(parts, key=lambda part: part.start_line)
    by_chunk = service.find_references(DeclarationSelector(chunk_id=last.chunk_id))
    assert by_chunk.selected == by_symbol.selected


def test_an_unchanged_signature_does_not_flag_calls_of_a_split_method(tmp_path: Path) -> None:
    body = "".join(
        f"        total_{index} = {index} * 2  # padding to force several chunks\n"
        for index in range(90)
    )
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "mail.py": (
                f"class Mailer:\n    def send(self, message):\n{body}        return message\n\n"
                "    def again(self):\n        return self.send('hi')\n"
            )
        },
    )
    parts = service.store.find_declarations(project_id, "mail.py", "Mailer.send")
    assert len(parts) > 1
    last = max(parts, key=lambda part: part.start_line)
    analysis = service.analyze_refactor(
        DeclarationSelector(chunk_id=last.chunk_id),
        SignatureChangeOperation(
            parameters=[
                ParameterShape(name="self", kind="positional", required=True, position=0),
                ParameterShape(name="message", kind="positional", required=True, position=1),
            ]
        ),
    )
    assert analysis.selected.kind == "method"
    assert analysis.must_change == []


def test_overloads_still_need_a_chunk_id(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(tmp_path, {"m.py": _OVERLOADED})
    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(_selector(project_id, "m.py", "conv"))
    assert excinfo.value.code is ErrorCode.AMBIGUOUS_SYMBOL


# --- REF-9 -----------------------------------------------------------------


def test_typescript_global_fetch_call_inside_a_method_is_not_a_reference_to_it(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {"api.ts": "export class Api { fetch(url: string) { return fetch(url); } }\n"},
    )
    selector = _selector(project_id, "api.ts", "Api.fetch")

    response = service.find_references(selector)

    assert [hit for hit in response.hits if hit.kind == "call"] == []
    radius = service.impact_radius(selector)
    assert [edge for layer in radius.layers for edge in layer.edges if edge.source.path] == []


def test_go_builtin_close_is_not_a_reference_to_a_close_method(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "conn.go": (
                "package p\n\ntype Conn struct{ done chan int }\n\n"
                "func (c *Conn) close() { close(c.done) }\n"
            ),
            "other.go": "package p\n\nfunc shut(ch chan int) { close(ch) }\n",
        },
    )

    response = service.find_references(_selector(project_id, "conn.go", "close"))

    assert [hit for hit in response.hits if hit.kind == "call"] == []


def test_rust_bare_helper_call_binds_the_free_function_not_the_method(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "lib.rs": (
                "pub fn helper() {}\n\npub struct S;\n\n"
                "impl S {\n    pub fn helper(&self) {}\n    pub fn run(&self) { helper(); }\n}\n"
            )
        },
    )

    on_method = service.find_references(_selector(project_id, "lib.rs", "S.helper"))
    assert [hit for hit in on_method.hits if hit.kind == "call"] == []

    on_function = service.find_references(_selector(project_id, "lib.rs", "helper"))
    call = next(hit for hit in on_function.hits if hit.kind == "call")
    assert call.resolution == "exact"


def test_python_class_body_name_still_reaches_a_method(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "c.py": (
                "class C:\n    def getter(self):\n        return 1\n\n    prop = property(getter)\n"
            )
        },
    )
    response = service.find_references(_selector(project_id, "c.py", "C.getter"))
    assert any(hit.resolution == "exact" and hit.kind == "read" for hit in response.hits)


# --- REF-10 ----------------------------------------------------------------


def _overrides(analysis):
    return [
        item for item in analysis.likely_change if item.reason_code == "override_of_renamed_method"
    ]


def test_java_interface_method_rename_surfaces_the_implementing_class(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "shapes/Shape.java": "package shapes;\npublic interface Shape { double area(); }\n",
            "impl/Circle.java": (
                "package impl;\nimport shapes.Shape;\n"
                "public class Circle implements Shape { public double area() { return 1.0; } }\n"
            ),
        },
    )
    analysis = service.analyze_refactor(
        _selector(project_id, "shapes/Shape.java", "Shape.area"), RenameOperation(new_name="size")
    )
    assert [item.path for item in _overrides(analysis)] == ["impl/Circle.java"]
    assert analysis.completeness.state != "complete"


@pytest.mark.parametrize(
    ("base_path", "base", "child_path", "child"),
    [
        (
            "p/Base.java",
            "package p;\npublic abstract class Base { abstract void run(); }\n",
            "p/Child.java",
            "package p;\npublic class Child extends Base { void run() {} }\n",
        ),
        (
            "p/Base.java",
            "package p;\npublic interface Base { void run(); }\n",
            "p/Child.java",
            "package p;\npublic class Child implements Base { public void run() {} }\n",
        ),
        (
            "Base.cs",
            "namespace N { public interface Base { void Run(); } }\n",
            "Child.cs",
            "namespace N { public class Child : Base { public void Run() {} } }\n",
        ),
    ],
)
def test_same_package_subclass_override_is_surfaced(
    tmp_path: Path, base_path: str, base: str, child_path: str, child: str
) -> None:
    service, project_id, _root = _indexed(tmp_path, {base_path: base, child_path: child})
    symbol = "Base.Run" if base_path.endswith(".cs") else "Base.run"
    new_name = "Go" if base_path.endswith(".cs") else "go"
    analysis = service.analyze_refactor(
        _selector(project_id, base_path, symbol), RenameOperation(new_name=new_name)
    )
    assert [item.path for item in _overrides(analysis)] == [child_path]
    assert analysis.completeness.state != "complete"


def test_an_unbindable_heritage_row_naming_the_owner_degrades_completeness(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "a/Base.java": "package a;\npublic interface Base { void run(); }\n",
            # Same simple name, different package, no import: not provably this Base.
            "z/Child.java": (
                "package z;\npublic class Child implements Base { public void run() {} }\n"
            ),
        },
    )
    analysis = service.analyze_refactor(
        _selector(project_id, "a/Base.java", "Base.run"), RenameOperation(new_name="go")
    )
    assert _overrides(analysis) == []
    assert any(item.code == "unbound_heritage" for item in analysis.limitations)
    assert analysis.completeness.state != "complete"


# --- REF-12 ----------------------------------------------------------------


def test_impact_cursor_for_an_expired_snapshot_is_a_stale_cursor_error(tmp_path: Path) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "graph.py": (
                "def base():\n    return 1\n\n"
                "def a():\n    return base()\n\n"
                "def b():\n    return base()\n"
            )
        },
    )
    selector = _selector(project_id, "graph.py", "base")
    first = service.impact_radius(selector, limit=1)
    assert first.cursor is not None
    payload = service._decode_impact_cursor(first.cursor)
    payload["version"] = int(payload["version"]) + 1000
    service._impact_traversal_cache.clear()

    with pytest.raises(CodeIndexingError) as excinfo:
        service.impact_radius(selector, limit=1, cursor=service._encode_cursor(payload))

    assert excinfo.value.code is ErrorCode.STALE_CURSOR


def test_a_cursor_that_outlives_a_reindex_reports_the_superseded_snapshot(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": (
                "from lib import answer\n\ndef a(): return answer()\ndef b(): return answer()\n"
            ),
        },
    )
    selector = _selector(project_id, "lib.py", "answer")
    radius_selector = _selector(project_id, "lib.py", "answer")
    first = service.find_references(selector, limit=1)
    first_radius = service.impact_radius(radius_selector, limit=1)
    assert first.cursor is not None and first_radius.cursor is not None
    assert not [item for item in first.limitations if item.code == "superseded_snapshot"]

    file_id = service.store.list_files(project_id)[0].file_id
    service.store.replace_files_from_arrow(
        project_id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[
            ([file_id], pa.Table.from_batches([], schema=LanceStore.reference_arrow_schema()))
        ],
    )

    second = service.find_references(selector, limit=1, cursor=first.cursor)
    assert second.snapshot_version == first.snapshot_version
    assert any(item.code == "superseded_snapshot" for item in second.limitations)
    second_radius = service.impact_radius(radius_selector, limit=1, cursor=first_radius.cursor)
    assert any(item.code == "superseded_snapshot" for item in second_radius.limitations)
    assert second_radius.completeness.state != "complete"


def test_a_lone_carriage_return_is_not_a_line_break_in_the_patch() -> None:
    original = b"a\rb\ndef authorize():\n    pass\n"
    edited = apply_edits(original, [ByteEdit(8, 17, b"validate")])

    patch = render_unified_diff("m.py", original, edited)

    assert patch is not None
    assert "No newline at end of file" not in patch
    assert "-def authorize():\n+def validate():\n" in patch
    assert " a\rb\n" in patch


def test_non_utf8_bytes_in_a_hunk_are_refused_not_replaced() -> None:
    original = b"# caf\xe9\ndef authorize():\n    pass\n"
    edited = apply_edits(original, [ByteEdit(11, 20, b"validate")])

    with pytest.raises(PatchEncodingError):
        render_unified_diff("m.py", original, edited)
    # Outside every hunk the same byte is harmless.
    assert render_unified_diff("m.py", original, edited, context_lines=0) is not None


def test_a_file_whose_hunk_cannot_be_text_is_withheld_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scanner never indexes a non-UTF-8 file, so the renderer's refusal is
    # injected: the service must report the file instead of emitting a patch.
    from code_indexing_mcp import reference_service as module

    def refuse(*_args: object, **_kwargs: object) -> str:
        raise PatchEncodingError("not utf-8")

    service, project_id, _root = _indexed(
        tmp_path, {"m.py": "def authorize():\n    pass\n", "u.py": "pass\n"}
    )
    monkeypatch.setattr(module, "render_unified_diff", refuse)

    patch = service.emit_refactor_patch(
        _selector(project_id, "m.py", "authorize"), RenameOperation(new_name="validate")
    )

    assert patch.patch == "" and patch.applied == 0
    assert [item.reason_code for item in patch.conflicted] == ["non_utf8_content"]
    assert patch.completeness.state == "complete_with_dynamic_limitations"


@pytest.mark.parametrize(
    ("files", "path", "symbol"),
    [
        (
            {
                "tools.py": "def run():\n    return 1\n",
                "use.py": "from subprocess import run\n\n\ndef go():\n    return run(['ls'])\n",
            },
            "tools.py",
            "run",
        ),
        (
            {
                "codec.py": "class T:\n    def dumps(self):\n        return ''\n",
                "use.py": (
                    "import json\nfrom json import dumps\n\n\n"
                    "def go():\n    return dumps({}) + json.dumps({})\n"
                ),
            },
            "codec.py",
            "T.dumps",
        ),
        (
            {
                "tools.ts": "export function run() {}\n",
                "use.ts": "import { run } from 'child-proc';\nexport function go() { run(); }\n",
            },
            "tools.ts",
            "run",
        ),
    ],
)
def test_library_imports_of_a_same_named_symbol_are_not_references(
    tmp_path: Path, files: dict[str, str], path: str, symbol: str
) -> None:
    service, project_id, _root = _indexed(tmp_path, files)

    response = service.find_references(_selector(project_id, path, symbol))

    assert [hit for hit in response.hits if hit.path.startswith("use")] == []


def test_a_project_module_the_resolver_cannot_place_stays_a_likely_reference(
    tmp_path: Path,
) -> None:
    # `from pkg.lib import run` from a non-package directory: the import
    # resolver may miss `src/pkg/lib.py`, but the path shows it is project code.
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "src/pkg/lib.py": "def run():\n    return 1\n",
            "scripts/use.py": "from pkg.lib import run\n\n\ndef go():\n    return run()\n",
        },
    )
    response = service.find_references(_selector(project_id, "src/pkg/lib.py", "run"))
    assert [hit for hit in response.hits if hit.path == "scripts/use.py"]


def test_collision_review_count_matches_the_findings_listed_across_pages(
    tmp_path: Path,
) -> None:
    service, project_id, _root = _indexed(
        tmp_path,
        {
            "lib.py": "def helper():\n    return 1\n",
            "main.py": (
                "from lib import helper\n\n\ndef go():\n    convert = 2\n"
                "    return helper() + helper() + convert\n"
            ),
        },
    )
    selector = _selector(project_id, "lib.py", "helper")
    operation = RenameOperation(new_name="convert")

    listed: list[str] = []
    cursor = None
    counts = None
    while True:
        page = service.analyze_refactor(selector, operation, limit=1, cursor=cursor)
        counts = counts or page.counts
        listed.extend(item.reference_id for item in page.review)
        if page.cursor is None:
            break
        cursor = page.cursor

    assert counts is not None and counts.review > 0
    assert len(listed) == counts.review
    assert len(set(listed)) == len(listed)
    full = service.analyze_refactor(selector, operation, limit=500)
    assert full.completeness.state != "complete"
    assert full.must_change == []


# --- Track F x Track E: gap `class_part` chunks compose with part grouping --------


def _class_with_gap_parts() -> str:
    pad = "".join(f"    # padding line {i}\n" for i in range(3))
    return (
        "class Config:\n"
        "    HEAD = 1\n"
        f"{pad}"
        "    def first(self):\n        return self.HEAD\n\n"
        "    BETWEEN = 2\n"
        f"{pad}"
        "    def second(self):\n        return self.BETWEEN\n\n"
        "    AFTER = 3\n"
        f"{pad}"
    )


def test_a_class_with_gap_parts_is_selectable_by_qualified_symbol(tmp_path: Path) -> None:
    from code_indexing_mcp.search import SearchService

    service, project_id, _root = _indexed(
        tmp_path,
        {
            "config.py": _class_with_gap_parts(),
            "use.py": "from config import Config\n\n\ndef go():\n    return Config()\n",
        },
    )
    parts = service.store.find_declarations(project_id, "config.py", "Config")
    assert any(part.kind == "class_part" for part in parts), [p.kind for p in parts]
    assert len(parts) > 1
    whole_end = len(_class_with_gap_parts().splitlines())
    selector = _selector(project_id, "config.py", "Config")

    found = service.find_references(selector)
    assert found.selected.kind == "class"
    assert (found.selected.start_line, found.selected.end_line) == (
        1,
        max(part.end_line for part in parts),
    )
    assert found.selected.end_line == whole_end
    assert any(hit.path == "use.py" for hit in found.hits)
    assert service.impact_radius(selector).selected.kind == "class"
    refactor = service.analyze_refactor(selector, RenameOperation(new_name="Settings"))
    assert refactor.selected.kind == "class"
    assert refactor.selected.end_line == whole_end

    outline = SearchService(service.store, TinyEmbedder()).file_outline("config.py", project_id)
    assert [
        item.qualified_symbol for item in outline.items if item.qualified_symbol == "Config"
    ] == ["Config"]
