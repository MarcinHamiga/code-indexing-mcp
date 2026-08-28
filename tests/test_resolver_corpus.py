"""Resolver corpus: small multi-file repos with per-defect expected outcomes.

Each case under `tests/fixtures/resolver_corpus/<language>/<case>/` is indexed
through the real pipeline (`_indexed_service`, mirroring
`tests/test_references.py`), then queried through the public
`ReferenceService` API (or, where the defect has no selectable declaration,
through the live reference-record rows the service itself reads).

Known gaps are pinned as `xfail(strict=True)`, tagged with the defect ID from
`docs/plans/2026-08-07-reference-index-hardening.md` /
`2026-08-07-reference-index-hardening-plan.md`. When a later phase's fix makes
one of these pass, remove its xfail marker as part of that fix — `strict=True`
turns an accidental early fix (or a regression) into a hard failure, so this
file is the corpus gate the backlog says never existed.

Hard gate (not a defect, never xfailed): zero false positives in the `exact`
resolution category. A same-named symbol reachable only through an unrelated
re-export chain must never bind `exact`.
"""

from __future__ import annotations

import json
from pathlib import Path

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import (
    DeclarationSelector,
    ParameterShape,
    RenameOperation,
    SignatureChangeOperation,
)
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore

CORPUS_ROOT = Path(__file__).parent / "fixtures" / "resolver_corpus"


class TinyEmbedder:
    model_id = "test/resolver-corpus"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def _load_repo(case_dir: Path) -> dict[str, str]:
    """Read every file under a corpus case directory into a path->source map."""
    files = {
        path.relative_to(case_dir).as_posix(): path.read_text()
        for path in sorted(case_dir.rglob("*"))
        if path.is_file()
    }
    assert files, f"resolver corpus case {case_dir} has no fixture files"
    return files


def _indexed_service(tmp_path: Path, case_dir: Path) -> tuple[ReferenceService, str]:
    root = tmp_path / "repo"
    root.mkdir()
    for path, source in _load_repo(case_dir).items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=TinyEmbedder(),
        lock_directory=tmp_path / "locks",
    )
    indexer.index(project)
    return ReferenceService(store), project.id


# ---------------------------------------------------------------------------
# E1: TS/TSX class heritage captures the raw clause text, not the identifier.
# ---------------------------------------------------------------------------


def test_e1_ts_class_heritage_finds_the_base_class_reference(tmp_path: Path) -> None:
    """Verbatim backlog repro: renaming a TS base class must surface `extends Base`.

    Applying only the reported edits today (declaration + import) leaves
    `extends Base` dangling in child.ts.
    """

    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "typescript" / "e1_inheritance_base_foundation"
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.ts", qualified_symbol="Base"),
        RenameOperation(new_name="Foundation"),
    )

    findings = analysis.must_change + analysis.likely_change
    inheritance_hit = next(
        item for item in findings if item.path == "child.ts" and item.kind == "inheritance"
    )
    assert inheritance_hit.resolution in {"exact", "likely"}


# ---------------------------------------------------------------------------
# E2: TS generic/union type inner names are excluded, only the whole
# type expression is captured verbatim.
# ---------------------------------------------------------------------------


def test_e2_ts_generic_argument_is_a_separate_type_use(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "typescript" / "e2_generic_type")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="box.ts", qualified_symbol="Item")
    )

    type_use = next(hit for hit in response.hits if hit.kind == "type_use")
    assert type_use.path == "main.ts"


# ---------------------------------------------------------------------------
# E3: `export * from './x'` produces no reference row at all; `export * as ns`
# discards the module path onto a bogus `read`.
# ---------------------------------------------------------------------------


def test_e3_bare_export_star_emits_a_barrel_export_row(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "typescript" / "e3_export_star")

    rows = service.store.list_reference_records(project_id)
    row = next(
        item for item in rows if item["path"] == "index.ts" and item["record_kind"] == "reference"
    )
    assert row["kind"] == "export"
    assert row["module_path"] == "./lib"


def test_e3_namespace_export_star_keeps_its_module_path(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "typescript" / "e3_export_star")

    rows = service.store.list_reference_records(project_id)
    row = next(
        item
        for item in rows
        if item["path"] == "index_namespace.ts" and item["record_kind"] == "reference"
    )
    assert row["kind"] == "export"
    assert row["alias"] == "ns"
    assert row["module_path"] == "./lib"


# ---------------------------------------------------------------------------
# E4: the mandatory `arguments:` field drops whole call forms.
# ---------------------------------------------------------------------------


def test_e4_python_generator_sole_argument_is_a_call(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "python" / "e4_generator_argument"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="summarize")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.path == "main.py"
    assert call.resolution == "exact"


def test_e4_js_tagged_template_is_a_call(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "javascript" / "e4_tagged_template"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.js", qualified_symbol="tag")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.path == "main.js"
    assert call.resolution == "exact"


def test_e4_js_new_without_parens_is_a_call(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "javascript" / "e4_new_expression"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.js", qualified_symbol="Widget")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.path == "main.js"
    assert call.resolution == "exact"


# ---------------------------------------------------------------------------
# E5: no `write` kind is ever emitted, and non-call member access is dropped.
# ---------------------------------------------------------------------------


def test_e5_python_member_write_and_read_are_recorded(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "python" / "e5_member_write")

    rows = service.store.list_reference_records(project_id)
    main_rows = [
        row for row in rows if row["path"] == "main.py" and row["record_kind"] == "reference"
    ]
    write_row = next(
        row
        for row in main_rows
        if row["kind"] == "write" and (row["target_name"] or "").rsplit(".", 1)[-1] == "TIMEOUT"
    )
    read_row = next(
        row
        for row in main_rows
        if row["kind"] == "read" and (row["target_name"] or "").rsplit(".", 1)[-1] == "TIMEOUT"
    )
    assert write_row["start_line"] == 3
    assert read_row["start_line"] == 4


def test_e5_js_shorthand_property_is_a_read(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "javascript" / "e5_shorthand")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="widget.js", qualified_symbol="onSave")
    )

    shorthand = next(hit for hit in response.hits if hit.kind == "read" and hit.path == "main.js")
    assert shorthand.start_line == 3


# ---------------------------------------------------------------------------
# E6: JS/TS decorators produce nothing.
# ---------------------------------------------------------------------------


def test_e6_ts_decorator_is_a_reference(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "typescript" / "e6_decorator")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.ts", qualified_symbol="Sealed")
    )

    decorator_hit = next(hit for hit in response.hits if hit.kind == "decorator")
    assert decorator_hit.path == "main.ts"


# ---------------------------------------------------------------------------
# E7: destructured JS/TS parameters produce a wrong, lossy shape.
# ---------------------------------------------------------------------------


def test_e7_tsx_destructured_parameter_name_is_not_raw_pattern_text(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "tsx" / "e7_destructured_parameter"
    )

    rows = service.store.list_reference_records(project_id)
    declaration = next(
        row
        for row in rows
        if row["record_kind"] == "declaration" and row["target_name"] == "Widget"
    )
    parameters = json.loads(declaration["shape_json"])
    assert len(parameters) == 1
    name = parameters[0]["name"]
    assert "{" not in name and "}" not in name


# ---------------------------------------------------------------------------
# E9: bare and dynamic module edges are dropped.
# ---------------------------------------------------------------------------


def test_e9_module_edges_stay_visible(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "javascript" / "e9_module_edges")

    rows = service.store.list_reference_records(project_id)
    main_rows = [
        row for row in rows if row["path"] == "main.js" and row["record_kind"] == "reference"
    ]
    bare_import = next(
        row for row in main_rows if row["kind"] == "import" and row["module_path"] == "./polyfill"
    )
    require_call = next(
        row for row in main_rows if row["kind"] == "call" and row["target_name"] == "require"
    )
    assert bare_import["imported_name"] is None
    assert require_call["module_path"] == "./lib"


# ---------------------------------------------------------------------------
# R1: no override analysis. Renaming Base.handle never mentions Child.handle.
# ---------------------------------------------------------------------------


def test_r1_python_override_is_a_likely_change(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "python" / "r1_override")

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.py", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    override = next(
        item
        for item in analysis.likely_change
        if item.path == "child.py" and item.reason_code == "override_of_renamed_method"
    )
    assert override.resolution == "likely"


# ---------------------------------------------------------------------------
# R2: re-export chains degrade to likely/unresolved rather than resolving.
# ---------------------------------------------------------------------------


def test_r2_reexport_chain_resolves_exactly(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "python" / "r2_reexport_chain")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="pkg/impl.py", qualified_symbol="b")
    )

    call = next(hit for hit in response.hits if hit.path == "main.py" and hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "reexport_chain"


def test_r2_hard_gate_unrelated_reexport_chain_never_binds_exact(tmp_path: Path) -> None:
    """Zero false positives in `exact`: a same-named symbol reachable only
    through a *different* re-export chain must never bind."""

    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "python" / "r2_reexport_chain_false_positive"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="pkg_a/impl.py", qualified_symbol="shared")
    )

    calls = [hit for hit in response.hits if hit.path == "main.py" and hit.kind == "call"]
    assert calls
    assert all(call.resolution != "exact" for call in calls)


# ---------------------------------------------------------------------------
# R3: rename of an exported TS symbol double-counts one identifier.
# ---------------------------------------------------------------------------


def test_r3_declaration_and_export_hit_are_deduped(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "typescript" / "r3_duplicate_declaration_export"
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="answer.ts", qualified_symbol="answer"),
        RenameOperation(new_name="result"),
    )

    assert analysis.counts.must_change == 1


def test_r3_declaration_and_export_hit_are_deduped_for_an_arrow_const(tmp_path: Path) -> None:
    """The second of the three duplicate shapes the verification identified:
    `export const answer = () => {}` narrows the export edit span to the
    same identifier as the synthetic declaration finding."""

    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "typescript" / "r3_duplicate_declaration_export_const"
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="answer.ts", qualified_symbol="answer"),
        RenameOperation(new_name="result"),
    )

    assert analysis.counts.must_change == 1


def test_r3_declaration_and_export_hit_are_deduped_for_a_default_class(tmp_path: Path) -> None:
    """The third of the three duplicate shapes the verification identified:
    `export default class Foo` narrows the export edit span to the same
    identifier as the synthetic declaration finding."""

    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "typescript" / "r3_duplicate_declaration_export_default_class"
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="foo.ts", qualified_symbol="Foo"),
        RenameOperation(new_name="Foundation"),
    )

    assert analysis.counts.must_change == 1


# ---------------------------------------------------------------------------
# R4: page-independent completeness and counts.
# ---------------------------------------------------------------------------


def test_r4_last_page_completeness_accounts_for_earlier_pages(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "python" / "r4_pagination_review"
    )
    selector = DeclarationSelector(project=project_id, path="mod.py", qualified_symbol="send")
    operation = SignatureChangeOperation(
        parameters=[ParameterShape(name="message", kind="positional", required=True, position=0)]
    )

    first = service.analyze_refactor(selector, operation, limit=2)
    assert first.cursor is not None
    assert any(item.reason_code == "spread_uncertainty" for item in first.review)

    second = service.analyze_refactor(selector, operation, limit=2, cursor=first.cursor)
    assert second.cursor is None
    assert second.completeness.state != "complete"


# ---------------------------------------------------------------------------
# Go language step: package imports, intra-package calls, receivers, dot
# imports, and embedded interfaces resolve with the shared
# exact/likely/unresolved contract.
# ---------------------------------------------------------------------------


def test_go_package_import_resolves_exactly(tmp_path: Path) -> None:
    """`store.Save()` through a plain import binds the suffixed directory."""

    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "go" / "package_import_exact")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="app/store/save.go", qualified_symbol="Save")
    )

    call = next(hit for hit in response.hits if hit.path == "main.go" and hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "known_namespace_member"


def test_go_same_package_call_resolves_exactly(tmp_path: Path) -> None:
    """A bare intra-package call needs no import: one directory IS one package."""

    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "go" / "same_package_exact")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="auth.go", qualified_symbol="Authorize")
    )

    call = next(hit for hit in response.hits if hit.path == "main.go" and hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "same_package_symbol"


def test_go_aliased_package_receiver_resolves_exactly(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "go" / "aliased_import")

    response = service.find_references(
        DeclarationSelector(
            project=project_id, path="app/users/user.go", qualified_symbol="GetByName"
        )
    )

    call = next(hit for hit in response.hits if hit.path == "main.go" and hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "known_namespace_member"


def test_go_a_foreign_receiver_stays_likely(tmp_path: Path) -> None:
    """`s.Handle()` inside `(*Worker).Run`: the local's spelling matches no
    receiver parameter of the enclosing method, so the name alone cannot bind
    even though it is unique -- the plan's honest receiver cap."""
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "go" / "unknown_receiver_likely")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="store.go", qualified_symbol="Handle")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "likely"
    assert call.reason_code == "unknown_receiver"
    assert any(item.code == "unknown_receiver" for item in response.limitations)


def test_go_dot_import_is_unresolved_with_a_wildcard_reason(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "go" / "dot_import_unresolved")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="app/util/util.go", qualified_symbol="Stamp")
    )

    call = next(hit for hit in response.hits if hit.path == "main.go" and hit.kind == "call")
    assert call.resolution == "unresolved"
    assert call.reason_code == "wildcard_import"


def test_go_embedded_interface_is_an_inheritance_edge(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "go" / "embedded_interface")

    rows = service.store.list_reference_records(project_id)
    embedded = [
        row
        for row in rows
        if row["record_kind"] == "reference"
        and row["kind"] == "inheritance"
        and row["target_name"] == "Reader"
    ]

    assert len(embedded) == 1
    assert embedded[0]["source_qualified_symbol"] == "LogStore"


def test_rust_crate_relative_use_resolves_exactly(tmp_path: Path) -> None:
    """`Save()` through `use crate::app::store::save::Save` anchors at the
    known crate root and binds the module file exactly."""
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "rust" / "crate_relative_use")

    response = service.find_references(
        DeclarationSelector(
            project=project_id, path="src/app/store/save.rs", qualified_symbol="Save"
        )
    )

    call = next(hit for hit in response.hits if hit.path == "src/lib.rs" and hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "direct_import_alias"


def test_rust_self_and_super_paths_resolve_exactly(tmp_path: Path) -> None:
    """`use self::inner::ping` anchors at the file's own directory and
    `use super::outer::pong` pops one directory from a `mod.rs` file."""
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "rust" / "self_super_paths")

    ping = service.find_references(
        DeclarationSelector(project=project_id, path="src/app/inner.rs", qualified_symbol="ping")
    )
    ping_hit = next(hit for hit in ping.hits if hit.path == "src/app/mod.rs" and hit.kind == "call")
    assert ping_hit.resolution == "exact"
    assert ping_hit.reason_code == "direct_import_alias"

    pong = service.find_references(
        DeclarationSelector(project=project_id, path="src/app/outer.rs", qualified_symbol="pong")
    )
    pong_hit = next(
        hit for hit in pong.hits if hit.path == "src/app/wrap/mod.rs" and hit.kind == "call"
    )
    assert pong_hit.resolution == "exact"
    assert pong_hit.reason_code == "direct_import_alias"


def test_rust_pub_use_reexport_chain_resolves_exactly(tmp_path: Path) -> None:
    """An importer reaches `Kick` through a `pub use` barrel: the chain walker
    follows the export row's module path to the defining file."""
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "rust" / "pub_use_reexport_chain"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="src/core/kick.rs", qualified_symbol="Kick")
    )

    call = next(hit for hit in response.hits if hit.path == "src/lib.rs" and hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "reexport_chain"


def test_rust_glob_use_is_unresolved_with_a_wildcard_reason(tmp_path: Path) -> None:
    """`use crate::util::*` binds every exported name dynamically, so the
    wildcard gate holds the call unresolved with a reason -- the same honest
    answer Go's dot imports and Python's star imports ship."""
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "rust" / "glob_use_unresolved")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="src/util.rs", qualified_symbol="limit")
    )

    call = next(hit for hit in response.hits if hit.path == "src/lib.rs" and hit.kind == "call")
    assert call.resolution == "unresolved"
    assert call.reason_code == "wildcard_import"


def test_rust_unprefixed_use_with_divergent_anchors_stays_likely(tmp_path: Path) -> None:
    """A plain first segment is edition-ambiguous: both the current-directory
    and crate-root readings exist here and disagree, so the binding cannot be
    proven and stays `likely` (unproven re-export) rather than gambling."""
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "rust" / "unprefixed_ambiguous")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="src/shared.rs", qualified_symbol="Tool")
    )

    call = next(hit for hit in response.hits if hit.path == "src/app/mod.rs" and hit.kind == "call")
    assert call.resolution == "likely"
    assert call.reason_code == "unproven_reexport"


def test_rust_self_call_resolves_exactly_through_the_impl_owner(tmp_path: Path) -> None:
    """`self.helper()` inside `impl Widget` carries the `Widget.run` enclosing
    symbol, so the receiver matches the declaration's owner exactly."""
    service, project_id = _indexed_service(tmp_path, CORPUS_ROOT / "rust" / "self_call_exact")

    response = service.find_references(
        DeclarationSelector(project=project_id, path="src/lib.rs", qualified_symbol="Widget.helper")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "known_owner_member"


def test_rust_trait_impl_is_an_inheritance_edge(tmp_path: Path) -> None:
    """`impl Draw for Widget` records the trait as an inheritance reference
    and binds it exactly (same file, same name)."""
    service, project_id = _indexed_service(
        tmp_path, CORPUS_ROOT / "rust" / "trait_impl_inheritance"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="src/lib.rs", qualified_symbol="Draw")
    )

    inheritance = next(hit for hit in response.hits if hit.kind == "inheritance")
    assert inheritance.resolution == "exact"
    assert inheritance.reason_code == "same_file_symbol"
