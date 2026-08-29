import shutil
import subprocess
from pathlib import Path

import pytest

from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import (
    DeclarationSelector,
    ParameterShape,
    RefactorPatch,
    ReferenceHit,
    RenameOperation,
    SelectedDeclaration,
    SignatureChangeOperation,
)
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


class TinyEmbedder:
    model_id = "test/refactor"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def _indexed_service(tmp_path: Path, files: dict[str, str]) -> tuple[ReferenceService, str]:
    root = tmp_path / "repo"
    root.mkdir()
    for path, source in files.items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.encode("utf-8"))
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=TinyEmbedder(),
        lock_directory=tmp_path / "locks",
    ).index(project)
    return ReferenceService(store), project.id


def test_rename_marks_the_imported_name_but_not_an_alias_call(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize as check\n\ndef run(user):\n    return check(user)\n"
            ),
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="validate"),
    )

    assert {(item.path, item.kind) for item in analysis.must_change} == {
        ("auth.py", "write"),
        ("consumer.py", "import"),
    }
    assert analysis.counts.must_change == 2
    assert analysis.counts.evidence == 1
    alias_call = next(item for item in analysis.findings if item.kind == "call")
    assert alias_call.resolution == "exact"
    assert not alias_call.edit_required

    # The import's own range covers "authorize as check"; only the imported
    # name may be rewritten or the alias is destroyed along with it.
    imported = next(item for item in analysis.must_change if item.kind == "import")
    source = (tmp_path / "repo" / "consumer.py").read_bytes()
    assert source[imported.edit_start_byte : imported.edit_end_byte] == b"authorize"


def test_signature_renamed_keyword_marks_the_exact_call_with_a_stable_reason(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "mail.py": "def send(message):\n    return message\n",
            "consumer.py": "from mail import send\n\nsend(message='hi')\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="mail.py", qualified_symbol="send"),
        SignatureChangeOperation(
            parameters=[ParameterShape(name="body", kind="positional", required=True, position=0)]
        ),
    )

    call = next(item for item in analysis.must_change if item.kind == "call")
    assert call.reason_code == "invalid_keyword"


def test_signature_spread_calls_are_reviewed_not_silently_ignored(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "mail.py": "def send(message):\n    return message\n",
            "consumer.py": "from mail import send\n\nargs = ('hi',)\nsend(*args)\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="mail.py", qualified_symbol="send"),
        SignatureChangeOperation(
            parameters=[
                ParameterShape(name="message", kind="positional", required=True, position=0),
                ParameterShape(name="timeout", kind="positional", required=True, position=1),
            ]
        ),
    )

    call = next(item for item in analysis.review if item.kind == "call")
    assert call.reason_code == "spread_uncertainty"


def test_rename_marks_exact_qualified_member_calls_for_edit(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {"auth.py": ("class Gate:\n    def authorize(self):\n        return self.authorize()\n")},
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="Gate.authorize"),
        RenameOperation(new_name="validate"),
    )

    call = next(item for item in analysis.must_change if item.kind == "call")
    assert call.written_name == "self.authorize"
    assert call.edit_required is True


def test_rename_marks_identifier_reads_for_edit(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {"lib.py": "def answer():\n    return 42\n\ncallback = answer\n"},
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer"),
        RenameOperation(new_name="result"),
    )

    read = next(item for item in analysis.must_change if item.kind == "read")
    assert read.written_name == "answer"


@pytest.mark.parametrize("new_name", ["$answer", "class"])
def test_python_rename_rejects_invalid_identifiers(tmp_path: Path, new_name: str) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {"lib.py": "def answer():\n    return 42\n"},
    )

    with pytest.raises(CodeIndexingError) as raised:
        service.analyze_refactor(
            DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer"),
            RenameOperation(new_name=new_name),
        )

    assert raised.value.code is ErrorCode.INVALID_REFACTOR


def test_rename_validation_uses_the_selected_language(tmp_path: Path) -> None:
    (tmp_path / "python").mkdir()
    (tmp_path / "javascript").mkdir()
    python_service, python_project = _indexed_service(
        tmp_path / "python",
        {"lib.py": "def answer():\n    return 42\n"},
    )
    javascript_service, javascript_project = _indexed_service(
        tmp_path / "javascript",
        {"lib.js": "function answer() { return 42; }\n"},
    )

    python_analysis = python_service.analyze_refactor(
        DeclarationSelector(project=python_project, path="lib.py", qualified_symbol="answer"),
        RenameOperation(new_name="_answer"),
    )
    javascript_analysis = javascript_service.analyze_refactor(
        DeclarationSelector(project=javascript_project, path="lib.js", qualified_symbol="answer"),
        RenameOperation(new_name="$answer"),
    )

    assert python_analysis.operation.new_name == "_answer"
    assert javascript_analysis.operation.new_name == "$answer"


@pytest.mark.parametrize(
    ("language", "new_name"),
    [
        ("javascript", "class"),
        ("typescript", "import"),
        ("tsx", "await"),
        ("go", "func"),
        ("rust", "match"),
        ("java", "class"),
        ("csharp", "namespace"),
    ],
)
def test_rename_rejects_reserved_words_for_the_selected_language(
    language: str, new_name: str
) -> None:
    selected = SelectedDeclaration(
        project_id="project",
        file_id="file",
        path="source",
        language=language,
        symbol="answer",
        qualified_symbol="answer",
        kind="function",
        start_line=1,
        end_line=1,
    )

    with pytest.raises(CodeIndexingError) as raised:
        ReferenceService._validate_rename(selected, RenameOperation(new_name=new_name))

    assert raised.value.code is ErrorCode.INVALID_REFACTOR


def test_rename_rejects_a_noop() -> None:
    selected = SelectedDeclaration(
        project_id="project",
        file_id="file",
        path="source.py",
        language="python",
        symbol="answer",
        qualified_symbol="answer",
        kind="function",
        start_line=1,
        end_line=1,
    )

    with pytest.raises(CodeIndexingError) as raised:
        ReferenceService._validate_rename(selected, RenameOperation(new_name="answer"))

    assert raised.value.code is ErrorCode.INVALID_REFACTOR


def test_package_import_that_does_not_spell_the_symbol_is_evidence(tmp_path: Path) -> None:
    service = ReferenceService(LanceStore(tmp_path / "data", vector_dimension=4))
    selected = SelectedDeclaration(
        project_id="project",
        file_id="target",
        path="auth.py",
        language="python",
        symbol="authorize",
        qualified_symbol="authorize",
        kind="function",
        start_line=1,
        end_line=1,
    )
    package_import = ReferenceHit(
        reference_id="import:auth",
        project_id="project",
        path="consumer.py",
        language="python",
        kind="import",
        start_line=1,
        end_line=1,
        start_byte=0,
        end_byte=11,
        snippet="import auth",
        written_name="auth",
        resolution="exact",
        reason_code="known_namespace_import",
        explanation="The import binds the target module.",
    )

    findings = service._classify_refactor_hits(
        selected,
        RenameOperation(new_name="validate"),
        [package_import],
        {},
        [],
        None,
        {},
    )

    assert findings.must_change == []
    assert [item.reference_id for item in findings.evidence] == ["import:auth"]


def test_refactor_analysis_pagination_is_independent_of_completeness_and_counts(
    tmp_path: Path,
) -> None:
    """A mid-stream page is not a coverage gap: `cursor` alone signals more
    pages remain, while `completeness.state` and `counts` are computed from
    the full, unsliced result set and so are identical on every page (R4).
    """

    callers = "".join(f"def caller_{index}():\n    return answer()\n\n" for index in range(501))
    service, project_id = _indexed_service(
        tmp_path,
        {"lib.py": f"def answer():\n    return 42\n\n{callers}"},
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    operation = RenameOperation(new_name="result")

    first = service.analyze_refactor(selector, operation)
    assert first.cursor is not None
    # Nothing here is a coverage gap or an unproven candidate, so the
    # first (mid-stream) page reports the same honest "complete" state as
    # the last page — the cursor, not completeness, carries the pagination
    # signal.
    assert first.completeness.state == "complete"
    assert first.counts.must_change == 502

    second = service.analyze_refactor(selector, operation, cursor=first.cursor)
    assert second.cursor is None
    assert second.completeness.state == "complete"
    # Counts are page-independent: the last page reports the same total as
    # the first, not just the count of what happens to be on this page.
    assert second.counts.must_change == 502


def test_refactor_cursor_is_bound_to_the_operation_and_page_limit(tmp_path: Path) -> None:
    """A page-2 cursor is rejected if the caller silently changes the
    refactor operation or the page size between calls (T2 new gap): neither
    dimension was bound into the cursor payload, so page 2 could otherwise
    accept a different `new_name` (or apply a rename's edits under a
    signature-change operation) or a different page size than page 1 used.
    """

    callers = "".join(f"def caller_{index}():\n    return answer()\n\n" for index in range(501))
    service, project_id = _indexed_service(
        tmp_path,
        {"lib.py": f"def answer():\n    return 42\n\n{callers}"},
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")

    first = service.analyze_refactor(selector, RenameOperation(new_name="result"))
    assert first.cursor is not None

    with pytest.raises(CodeIndexingError) as excinfo:
        service.analyze_refactor(
            selector, RenameOperation(new_name="different"), cursor=first.cursor
        )
    assert excinfo.value.code == ErrorCode.INVALID_CURSOR

    with pytest.raises(CodeIndexingError) as excinfo:
        service.analyze_refactor(
            selector, RenameOperation(new_name="result"), cursor=first.cursor, limit=10
        )
    assert excinfo.value.code == ErrorCode.INVALID_CURSOR

    # The identical operation and limit are accepted, unaffected by binding.
    second = service.analyze_refactor(
        selector, RenameOperation(new_name="result"), cursor=first.cursor
    )
    assert second.cursor is None

    calls = [
        item for analysis in (first, second) for item in analysis.must_change if item.kind == "call"
    ]
    assert len(calls) == 501


def test_signature_keyword_satisfies_required_positional_parameter(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "mail.py": "def send(message):\n    return message\n",
            "consumer.py": "from mail import send\n\nsend(message='hi')\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="mail.py", qualified_symbol="send"),
        SignatureChangeOperation(
            parameters=[
                ParameterShape(name="message", kind="positional", required=True, position=0)
            ]
        ),
    )

    call = next(item for item in analysis.evidence if item.kind == "call")
    assert call.reason_code == "direct_import_alias"


def test_signature_bound_receiver_does_not_consume_a_call_argument(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "mail.py": (
                "class Mailer:\n    def send(self, message):\n        return self.send(message)\n"
            )
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="mail.py", qualified_symbol="Mailer.send"),
        SignatureChangeOperation(
            parameters=[
                ParameterShape(name="self", kind="positional", required=True, position=0),
                ParameterShape(name="message", kind="positional", required=True, position=1),
            ]
        ),
    )

    call = next(item for item in analysis.evidence if item.kind == "call")
    assert call.reason_code == "known_owner_member"


@pytest.mark.parametrize(
    ("source", "qualified_symbol"),
    [
        (
            "abstract class Base {\n"
            "  abstract run(a: number, b: number): void;\n"
            "  invoke(): void { this.run(1, 2); }\n"
            "}\n",
            "Base.run",
        ),
        (
            "class Service {\n"
            "  run = (a: number, b: number): number => a + b;\n"
            "  invoke(): number { return this.run(1, 2); }\n"
            "}\n",
            "Service.run",
        ),
    ],
)
def test_typescript_callable_class_members_keep_parameters_for_signature_analysis(
    tmp_path: Path, source: str, qualified_symbol: str
) -> None:
    service, project_id = _indexed_service(tmp_path, {"service.ts": source})

    analysis = service.analyze_refactor(
        DeclarationSelector(
            project=project_id,
            path="service.ts",
            qualified_symbol=qualified_symbol,
        ),
        SignatureChangeOperation(
            parameters=[
                ParameterShape(name="b", kind="positional", required=True, position=0),
                ParameterShape(name="a", kind="positional", required=True, position=1),
            ]
        ),
    )

    call = next(item for item in analysis.must_change if item.kind == "call")
    assert call.reason_code == "positional_order_change"


def test_a_qualified_call_edits_only_the_member_name(tmp_path: Path) -> None:
    """The reference range is wider than the identifier to rewrite.

    `auth.authorize(u)` spans `auth.authorize`. Replacing that whole range with
    the new name drops the module qualifier and breaks the call.
    """

    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "caller.py": "import auth\n\ndef run(user):\n    return auth.authorize(user)\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="permit"),
    )

    call = next(item for item in analysis.must_change if item.path == "caller.py")
    source = (tmp_path / "repo" / "caller.py").read_bytes()
    assert source[call.start_byte : call.end_byte] == b"auth.authorize"
    assert source[call.edit_start_byte : call.edit_end_byte] == b"authorize"
    edited = source[: call.edit_start_byte] + b"permit" + source[call.edit_end_byte :]
    assert b"return auth.permit(user)" in edited


def test_the_declaration_finding_points_at_its_own_name(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, {"auth.py": "def authorize(user):\n    return user\n"}
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="permit"),
    )

    declaration = next(item for item in analysis.must_change if item.reason_code == "declaration")
    source = (tmp_path / "repo" / "auth.py").read_bytes()
    assert source[declaration.edit_start_byte : declaration.edit_end_byte] == b"authorize"


def test_required_rename_edits_are_deduplicated_by_span(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {"models.py": "class Model:\n    pass\n\nclass FrozenModel(Model):\n    pass\n"},
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="models.py", qualified_symbol="Model"),
        RenameOperation(new_name="Entity"),
    )

    spans = [
        (item.path, item.edit_start_byte, item.edit_end_byte)
        for item in analysis.must_change
        if item.edit_start_byte is not None and item.edit_end_byte is not None
    ]
    assert len(spans) == len(set(spans))
    assert analysis.counts.must_change == 2


def test_analyze_refactor_fetches_the_reference_table_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """analyze_refactor must reuse find_references' fetch, not re-scan (S4)."""
    service, project_id = _indexed_service(
        tmp_path, {"auth.py": "def authorize(user):\n    return user\n"}
    )
    calls: list[int | None] = []
    real_list_reference_records = service.store.list_reference_records

    def counting_list_reference_records(
        project: str,
        *,
        version: int | None = None,
        schema_version: int | None = None,
        record_kinds: object = None,
        partition_id: str | None = None,
    ) -> list[object]:
        calls.append(version)
        return real_list_reference_records(
            project,
            version=version,
            schema_version=schema_version,
            record_kinds=record_kinds,
            partition_id=partition_id,
        )

    monkeypatch.setattr(service.store, "list_reference_records", counting_list_reference_records)

    service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="permit"),
    )

    assert len(calls) == 1, f"expected exactly one full-table fetch, got {calls}"


def test_analyze_refactor_classifies_the_full_hit_list_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """analyze_refactor must reuse find_references' classification pass, not repeat it (E1).

    The full-table fetch was already de-duplicated (see the test above); the
    remaining, more expensive duplication was the classification pass itself
    (`_hits_and_limitations`, which walks every reference row and reads every
    referenced file) running once inside `find_references` and a second time
    inside `analyze_refactor` to get an unpaginated hit list.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "main.py": (
                "from auth import authorize\n\n\n"
                "def run():\n    return authorize(1)\n\n\n"
                "def run_again():\n    return authorize(2)\n"
            ),
        },
    )
    calls = 0
    real_hits_and_limitations = service._hits_and_limitations

    def counting_hits_and_limitations(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_hits_and_limitations(*args, **kwargs)

    monkeypatch.setattr(service, "_hits_and_limitations", counting_hits_and_limitations)

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="permit"),
    )

    assert calls == 1, f"expected exactly one classification pass, got {calls}"
    # The single pass must still be a correct, consistent result: every call
    # site is a must_change rename edit, and completeness reflects the whole
    # (unpaginated) result set, not just whichever page happened to be asked for.
    assert {item.path for item in analysis.must_change} == {"auth.py", "main.py"}
    assert analysis.completeness.state == "complete"


def test_rename_of_a_base_method_surfaces_the_subclass_override(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": "class Base:\n    def handle(self):\n        return 1\n",
            "child.py": (
                "from base import Base\n\n\n"
                "class Child(Base):\n    def handle(self):\n        return 2\n"
            ),
        },
    )

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
    assert not any(
        item.path == "child.py" and item.reason_code == "override_of_renamed_method"
        for item in analysis.must_change
    )


def test_rename_of_an_aliased_base_method_surfaces_the_subclass_override(tmp_path: Path) -> None:
    """Same as the unaliased control above, but the base class is imported
    under an alias (`from base import Base as B; class C(B): ...`).

    `_inheritance_targets` must consult the same alias-to-imported-name
    mapping the direct-import path already uses, not just the base class's
    real name, or the override is silently dropped and completeness lies
    about having fully accounted for the rename (finding 5).
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": "class Base:\n    def handle(self):\n        return 1\n",
            "child.py": (
                "from base import Base as B\n\n\n"
                "class C(B):\n    def handle(self):\n        return 2\n"
            ),
        },
    )

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
    assert not any(
        item.path == "child.py" and item.reason_code == "override_of_renamed_method"
        for item in analysis.must_change
    )
    assert analysis.completeness.state == "complete_with_dynamic_limitations"


def test_an_unrelated_aliased_import_is_not_treated_as_the_base_class(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": (
                "class Base:\n    def handle(self):\n        return 1\n\n"
                "class Other:\n    def handle(self):\n        return 2\n"
            ),
            "child.py": (
                "from base import Other as B\n\n"
                "class Child(B):\n    def handle(self):\n        return 3\n"
            ),
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.py", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    assert not any(
        item.reason_code == "override_of_renamed_method" for item in analysis.likely_change
    )


def test_same_file_namespace_heritage_is_not_bound_to_a_local_namesake(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": (
                "import other\n\n"
                "class Base:\n    def handle(self):\n        return 1\n\n"
                "class Child(other.Base):\n    def handle(self):\n        return 2\n"
            ),
            "other.py": "class Base:\n    def handle(self):\n        return 3\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.py", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    assert not any(
        item.reason_code == "override_of_renamed_method" for item in analysis.likely_change
    )


def test_a_namespace_imported_base_class_surfaces_the_subclass_override(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": "class Base:\n    def handle(self):\n        return 1\n",
            "child.py": (
                "import base\n\nclass Child(base.Base):\n    def handle(self):\n        return 2\n"
            ),
        },
    )

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


def test_rename_of_a_base_method_surfaces_a_transitive_subclass_override(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": "class Base:\n    def handle(self):\n        return 1\n",
            "mid.py": (
                "from base import Base\n\n\n"
                "class Mid(Base):\n    def handle(self):\n        return 2\n"
            ),
            "leaf.py": (
                "from mid import Mid\n\n\n"
                "class Leaf(Mid):\n    def handle(self):\n        return 3\n"
            ),
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.py", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    override_paths = {
        item.path
        for item in analysis.likely_change
        if item.reason_code == "override_of_renamed_method"
    }
    assert override_paths == {"mid.py", "leaf.py"}


def test_javascript_rename_of_a_base_method_surfaces_the_subclass_override(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.js": "export class Base {\n  handle() {\n    return 1;\n  }\n}\n",
            "child.js": (
                "import { Base } from './base';\n\n"
                "export class Child extends Base {\n  handle() {\n    return 2;\n  }\n}\n"
            ),
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.js", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    override = next(
        item
        for item in analysis.likely_change
        if item.path == "child.js" and item.reason_code == "override_of_renamed_method"
    )
    assert override.resolution == "likely"


def test_typescript_namespace_heritage_surfaces_the_subclass_override(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.ts": "export class Base { handle(): void {} }\n",
            "child.ts": (
                "import * as ns from './base';\n"
                "export class Child extends ns.Base { handle(): void {} }\n"
            ),
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.ts", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    override = next(
        item
        for item in analysis.likely_change
        if item.path == "child.ts" and item.reason_code == "override_of_renamed_method"
    )
    assert override.resolution == "likely"


def test_an_unanalyzable_language_makes_the_analysis_incomplete(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "client.c": "int Run(void) {\n\treturn 1;\n}\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="permit"),
    )

    assert analysis.completeness.state == "incomplete"
    assert any(item.code == "unsupported_language" for item in analysis.limitations)


def test_a_declaration_without_reference_extraction_is_refused(tmp_path: Path) -> None:
    """Answering at all would mean reporting "rename one line" for a C function
    whose callers this index never looked at."""

    service, project_id = _indexed_service(
        tmp_path,
        {
            "svc.c": "int Authorize(const char *u) {\n\treturn 1;\n}\n",
            "use.c": 'int Run(void) {\n\treturn Authorize("a");\n}\n',
        },
    )

    with pytest.raises(CodeIndexingError) as raised:
        service.analyze_refactor(
            DeclarationSelector(project=project_id, path="svc.c", qualified_symbol="Authorize"),
            RenameOperation(new_name="Permit"),
        )

    assert raised.value.code is ErrorCode.UNSUPPORTED_LANGUAGE


def test_a_go_rename_analysis_covers_its_same_package_caller(tmp_path: Path) -> None:
    """Go joined the structural languages, so a rename answers instead of
    refusing -- and the intra-package call site binds exactly."""

    service, project_id = _indexed_service(
        tmp_path,
        {
            "svc.go": ("package main\n\nfunc Authorize(u string) string {\n\treturn u\n}\n"),
            "use.go": ('package main\n\nfunc Run() string {\n\treturn Authorize("a")\n}\n'),
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="svc.go", qualified_symbol="Authorize"),
        RenameOperation(new_name="Permit"),
    )

    assert analysis.completeness.state == "complete"
    edited_paths = {
        (item.path, item.written_name)
        for item in analysis.must_change
        if item.edit_required and item.written_name == "Authorize"
    }
    assert ("svc.go", "Authorize") in edited_paths
    assert ("use.go", "Authorize") in edited_paths


def test_a_rust_rename_analysis_covers_its_imported_caller(tmp_path: Path) -> None:
    """Rust joined the structural languages, so a rename answers instead of
    refusing -- and the crate-relative import binds exactly."""

    service, project_id = _indexed_service(
        tmp_path,
        {
            "src/lib.rs": (
                "mod auth;\n\nuse crate::auth::Authorize;\n\n"
                "pub fn run() -> u32 {\n    Authorize(1)\n}\n"
            ),
            "src/auth.rs": "pub fn Authorize(user: u32) -> u32 {\n    user\n}\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="src/auth.rs", qualified_symbol="Authorize"),
        RenameOperation(new_name="Permit"),
    )

    assert analysis.completeness.state == "complete"
    edited_paths = {
        (item.path, item.written_name)
        for item in analysis.must_change
        if item.edit_required and item.written_name == "Authorize"
    }
    assert ("src/auth.rs", "Authorize") in edited_paths
    assert ("src/lib.rs", "Authorize") in edited_paths


def test_an_unproven_call_keeps_the_analysis_out_of_the_complete_state(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "def caller(thing):\n    return thing.answer()\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer"),
        RenameOperation(new_name="result"),
    )

    assert analysis.counts.likely_change >= 1
    assert analysis.completeness.state == "complete_with_dynamic_limitations"


def test_an_ambiguous_selector_names_its_candidates(tmp_path: Path) -> None:
    service, project_id = _indexed_service(tmp_path, {"lib.py": "def answer():\n    return 1\n"})

    with pytest.raises(CodeIndexingError) as raised:
        service.analyze_refactor(
            DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="missing"),
            RenameOperation(new_name="result"),
        )

    assert raised.value.code is ErrorCode.AMBIGUOUS_SYMBOL


def test_a_ts_scope_no_longer_carries_the_blanket_extraction_gaps_limitation(
    tmp_path: Path,
) -> None:
    """Phase 2 covered heritage, generic types, `export *`, member access, and
    decorators (E1/E2/E3/E5/E6/E9/E10/E11/E12); the corpus-gated cap over
    those constructs is retired (Task 2.7). A plain TS scope with nothing left
    uncovered must not claim otherwise."""

    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.ts": "export function answer(): number { return 42; }\n",
            "main.ts": "import { answer } from './lib';\nanswer();\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.ts", qualified_symbol="answer"),
    )

    assert not any(item.code == "extraction_gaps" for item in response.limitations)


def test_a_ts_scope_reaches_the_complete_state_for_a_function_rename(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.ts": "export function answer(): number { return 42; }\n",
            "main.ts": "import { answer } from './lib';\nanswer();\n",
        },
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="lib.ts", qualified_symbol="answer"),
        RenameOperation(new_name="result"),
    )

    assert not any(item.code == "extraction_gaps" for item in analysis.limitations)
    assert analysis.completeness.state == "complete"


def test_a_ts_class_rename_finds_the_heritage_reference_and_reaches_complete(
    tmp_path: Path,
) -> None:
    """E1 is fixed: class-heritage extraction now surfaces `extends Base`, so
    renaming a base class is no longer a known-wrong-answer case."""

    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.ts": "export class Base {\n  run(): number { return 1; }\n}\n",
            "child.ts": ("import { Base } from './base';\n\nexport class Child extends Base {}\n"),
        },
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
    assert not any(item.code == "extraction_gaps" for item in analysis.limitations)
    assert analysis.completeness.state != "incomplete"


def test_a_tsx_jsx_component_use_resolves_without_a_standing_limitation(
    tmp_path: Path,
) -> None:
    """E14 (JSX component tag references) is fixed: a `<Widget />` use is its

    own `type_use` reference row, so it resolves exactly and a TSX scope no
    longer needs a narrow limitation naming that gap.
    """

    service, project_id = _indexed_service(
        tmp_path,
        {
            "widget.tsx": ("export function Widget(): JSX.Element {\n  return <div />;\n}\n"),
            "main.tsx": (
                "import { Widget } from './widget';\n"
                "export function App(): JSX.Element {\n  return <Widget />;\n}\n"
            ),
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="widget.tsx", qualified_symbol="Widget"),
    )

    assert not any(item.code == "extraction_gaps" for item in response.limitations)
    component_hit = next(
        item for item in response.hits if item.path == "main.tsx" and item.kind == "type_use"
    )
    assert component_hit.resolution == "exact"


def test_a_python_only_scope_is_unaffected_by_the_standing_limitation(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {"lib.py": "class Base:\n    def run(self):\n        return 1\n"},
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="Base"),
        RenameOperation(new_name="Foundation"),
    )

    assert not any(item.code == "extraction_gaps" for item in analysis.limitations)
    assert analysis.completeness.state == "complete"


def test_analyze_refactor_fetches_declarations_narrowly_not_from_the_full_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`analyze_refactor` looks up declarations by exact qualified symbol (S4/E3).

    The declaration-finding lookup, the transitive-override BFS, and
    (for a signature change) the old-shape comparison all used to scan
    `records` for `record_kind == "declaration"` -- `records` no longer
    carries those rows at all, so this also proves the rename path does not
    silently fall back to an empty declaration set: the declaration/override
    findings below still come out correct.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": "class Base:\n    def handle(self):\n        return 1\n",
            "mid.py": (
                "from base import Base\n\n\n"
                "class Mid(Base):\n    def handle(self):\n        return 2\n"
            ),
        },
    )
    calls: list[str] = []
    real_declaration_shapes = service.store.declaration_shapes

    def spy_declaration_shapes(
        project: str, qualified_symbol: str, **kwargs: object
    ) -> list[object]:
        calls.append(qualified_symbol)
        return real_declaration_shapes(project, qualified_symbol, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "declaration_shapes", spy_declaration_shapes)

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="base.py", qualified_symbol="Base.handle"),
        RenameOperation(new_name="process"),
    )

    # Exactly the qualified symbols this rename actually needed -- the
    # renamed declaration itself, the owner class for the override walk, and
    # the one override found while walking it -- never a project-wide fetch.
    assert set(calls) == {"Base.handle", "Base", "Mid.handle"}

    declaration = next(item for item in analysis.must_change if item.reason_code == "declaration")
    assert declaration.path == "base.py"
    override = next(
        item for item in analysis.likely_change if item.reason_code == "override_of_renamed_method"
    )
    assert override.path == "mid.py"


def test_analyze_refactor_signature_change_fetches_the_old_shape_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_signature_issue`'s old-shape lookup is a single pinned-version query
    (S4/E3), not a rescan of the live table per hit -- see the docstring on
    `_signature_issue` for the regression this must not reintroduce.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user, level):\n    return user\n",
            "main.py": (
                "from auth import authorize\n\n\n"
                "def run():\n    return authorize(1, 2)\n\n\n"
                "def run_again():\n    return authorize(3, 4)\n"
            ),
        },
    )
    calls: list[str] = []
    real_declaration_shapes = service.store.declaration_shapes

    def spy_declaration_shapes(
        project: str, qualified_symbol: str, **kwargs: object
    ) -> list[object]:
        calls.append(qualified_symbol)
        return real_declaration_shapes(project, qualified_symbol, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "declaration_shapes", spy_declaration_shapes)

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        SignatureChangeOperation(
            parameters=[
                ParameterShape(name="user", kind="positional", required=True, position=0),
                ParameterShape(name="level", kind="positional", required=True, position=1),
                ParameterShape(name="extra", kind="positional", required=True, position=2),
            ]
        ),
    )

    # One fetch total, not one per call site -- two call sites are classified
    # below, so a per-hit rescan would show up as more than one call here.
    assert calls == ["authorize"]
    assert len(analysis.must_change) == 2


def test_analyze_refactor_suppresses_edit_spans_from_a_stale_file(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
            ),
        },
    )
    # consumer.py changed on disk without a reindex: its stored offsets
    # describe bytes that no longer exist, so no edit may be derived from
    # them -- the wrong-edit hazard the serve-time hash gate exists for.
    (tmp_path / "repo" / "consumer.py").write_text(
        "from auth import authorize\n\n\ndef run(user):\n    return authorize(user)\n"
    )

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="validate"),
    )

    assert all(item.path != "consumer.py" for item in analysis.must_change)
    assert any(
        item.code == "stale_file" and "consumer.py" in item.explanation
        for item in analysis.limitations
    )
    assert analysis.completeness.state == "incomplete"


def _emit(
    service: ReferenceService, project_id: str, path: str, symbol: str, new_name: str
) -> RefactorPatch:
    return service.emit_refactor_patch(
        DeclarationSelector(project=project_id, path=path, qualified_symbol=symbol),
        RenameOperation(new_name=new_name),
    )


def test_emit_refactor_patch_renders_a_multi_file_rename(tmp_path: Path) -> None:
    files = {
        "auth.py": "def authorize(user):\n    return user\n",
        "consumer.py": (
            "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
        ),
    }
    service, project_id = _indexed_service(tmp_path, files)

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    assert result.applied == 3
    # Files render in sorted path order, each under a/ and b/ headers.
    assert result.patch == (
        "diff --git a/auth.py b/auth.py\n"
        "--- a/auth.py\n"
        "+++ b/auth.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def authorize(user):\n"
        "+def validate(user):\n"
        "     return user\n"
        "diff --git a/consumer.py b/consumer.py\n"
        "--- a/consumer.py\n"
        "+++ b/consumer.py\n"
        "@@ -1,4 +1,4 @@\n"
        "-from auth import authorize\n"
        "+from auth import validate\n"
        " \n"
        " def run(user):\n"
        "-    return authorize(user)\n"
        "+    return validate(user)\n"
    )
    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize"),
        RenameOperation(new_name="validate"),
    )
    assert sorted(
        (edit.path, edit.edit_start_byte, edit.edit_end_byte) for edit in result.edits
    ) == sorted(
        (item.path, item.edit_start_byte, item.edit_end_byte) for item in analysis.must_change
    )
    assert result.unapplied == []
    assert result.conflicted == []
    assert result.completeness.state == "complete"


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_emitted_patch_applies_with_git(tmp_path: Path) -> None:
    files = {
        "auth.py": "def authorize(user):\n    return user\n",
        "consumer.py": (
            "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
        ),
    }
    service, project_id = _indexed_service(tmp_path, files)

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    worktree = tmp_path / "apply"
    worktree.mkdir()
    for name, source in files.items():
        (worktree / name).write_bytes(source.encode("utf-8"))

    def git(*arguments: str) -> None:
        run = subprocess.run(
            ("git", *arguments), cwd=worktree, check=False, capture_output=True, text=True
        )
        assert run.returncode == 0, run.stderr

    git("init", "-q")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    (worktree / "rename.diff").write_bytes(result.patch.encode("utf-8"))
    git("apply", "rename.diff")

    assert (worktree / "auth.py").read_bytes() == b"def validate(user):\n    return user\n"
    assert (
        worktree / "consumer.py"
    ).read_bytes() == b"from auth import validate\n\ndef run(user):\n    return validate(user)\n"


def test_emit_refactor_patch_replaces_only_the_imported_name_not_the_alias(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize as check\n\ndef run(user):\n    return check(user)\n"
            ),
        },
    )

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    assert result.applied == 2
    assert "+from auth import validate as check\n" in result.patch
    # The alias call binds exactly but needs no spelling change, so it stays
    # out of the patch entirely.
    assert "    return check(user)\n" in result.patch
    assert "+    return validate" not in result.patch


def test_emit_refactor_patch_replaces_only_the_member_of_a_qualified_call(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": "import auth\n\ndef run(u):\n    return auth.authorize(u)\n",
        },
    )

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    assert result.applied == 2
    assert "+    return auth.validate(u)\n" in result.patch
    assert "auth.authorize" not in result.patch.replace("-    return auth.authorize(u)", "")


def test_likely_findings_never_reach_the_patch_but_always_reach_unapplied(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "base.py": "class Base:\n    def handle(self):\n        return 1\n",
            "child.py": (
                "from base import Base\n\n\nclass Child(Base):\n    def handle(self):\n"
                "        return 2\n"
            ),
        },
    )

    result = _emit(service, project_id, "base.py", "Base.handle", "process")

    assert result.applied >= 1
    override = next(
        item
        for item in result.unapplied
        if item.path == "child.py" and item.reason_code == "override_of_renamed_method"
    )
    # Overrides need human judgement even when their offsets are known, so
    # they are reported verbatim instead of being rendered.
    assert override.resolution == "likely"
    assert "child.py" not in result.patch
    assert result.completeness.state == "complete_with_dynamic_limitations"


def test_a_stale_touched_file_is_conflicted_and_omitted_until_reindexed(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
            ),
        },
    )
    # consumer.py changed on disk without a reindex, exactly as in the
    # analyze_refactor suppression test above.
    (tmp_path / "repo" / "consumer.py").write_text(
        "from auth import authorize\n\n\ndef run(user):\n    return authorize(user)\n"
    )

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    stale = [item for item in result.conflicted if item.path == "consumer.py"]
    assert stale and all(item.reason_code == "stale_file" for item in stale)
    assert result.patch == (
        "diff --git a/auth.py b/auth.py\n"
        "--- a/auth.py\n"
        "+++ b/auth.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def authorize(user):\n"
        "+def validate(user):\n"
        "     return user\n"
    )
    assert result.applied == 1
    assert any(item.code == "stale_file" for item in result.limitations)
    assert result.completeness.state == "incomplete"

    # After a reindex the fresh offsets are served and the hunks appear.
    store = LanceStore(tmp_path / "data2", vector_dimension=4)
    Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=TinyEmbedder(),
        lock_directory=tmp_path / "locks2",
    ).index(initialize_project(tmp_path / "repo"))
    fresh = ReferenceService(store)
    reindexed = _emit(
        fresh, initialize_project(tmp_path / "repo").id, "auth.py", "authorize", "validate"
    )
    assert reindexed.conflicted == []
    assert reindexed.applied == 3
    assert "consumer.py" in reindexed.patch


def test_a_stale_selected_file_conflicts_its_declaration_edit(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
            ),
        },
    )
    (tmp_path / "repo" / "auth.py").write_text("def authorize(user, level):\n    return user\n")

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    declaration = [item for item in result.conflicted if item.kind == "write"]
    assert declaration and all(item.reason_code == "stale_file" for item in declaration)
    assert "diff --git a/auth.py" not in result.patch
    # consumer.py is fresh, so its hunks still render.
    assert "diff --git a/consumer.py" in result.patch
    assert result.completeness.state == "incomplete"


def test_a_slice_mismatch_conflicts_the_finding(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
            ),
        },
    )
    operation = RenameOperation(new_name="validate")
    selector = DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize")
    analysis, query = service._rename_analysis(
        selector,
        operation,
        limit=500,
        cursor=None,
        backfill=None,
        partition=None,
        paginate=False,
    )
    # Shift one finding's span by one byte: the slice no longer spells the
    # identifier the resolver matched, so the edit must be refused.
    shifted_source = "consumer.py"
    victim = next(item for item in analysis.must_change if item.path == shifted_source)
    assert victim.edit_start_byte is not None and victim.edit_end_byte is not None
    mutated = analysis.model_copy(
        update={
            "must_change": [
                item.model_copy(
                    update={
                        "edit_start_byte": item.edit_start_byte + 1,
                        "edit_end_byte": item.edit_end_byte + 1,
                    }
                )
                if item.path == shifted_source
                else item
                for item in analysis.must_change
            ]
        }
    )

    result = service._render_patch(mutated, query, operation, context_lines=3)

    conflicted = [item for item in result.conflicted if item.path == shifted_source]
    # Both consumer.py findings (import and call) carried the shifted span.
    assert len(conflicted) == 2
    assert all("no longer spell" in item.explanation for item in conflicted)
    assert result.applied == 1
    assert result.edits[0].path == "auth.py"


def test_emission_rereads_files_after_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
            ),
        },
    )
    consumer = tmp_path / "repo" / "consumer.py"
    analyze = service._rename_analysis

    def mutate_after_analysis(*args: object, **kwargs: object) -> object:
        result = analyze(*args, **kwargs)
        consumer.write_text(
            "from auth import authorize\n\n\ndef run(user):\n    return authorize(user)\n"
        )
        return result

    monkeypatch.setattr(service, "_rename_analysis", mutate_after_analysis)

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    assert "diff --git a/consumer.py" not in result.patch
    assert any(
        item.path == "consumer.py" and item.reason_code == "stale_file"
        for item in result.conflicted
    )


def test_an_overlapping_edit_is_omitted_not_merged(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "main.py": (
                "from auth import authorize\n\n\n"
                "def run(u):\n    return authorize(u) + authorize(u)\n"
            ),
        },
    )
    operation = RenameOperation(new_name="validate")
    selector = DeclarationSelector(project=project_id, path="auth.py", qualified_symbol="authorize")
    analysis, query = service._rename_analysis(
        selector,
        operation,
        limit=500,
        cursor=None,
        backfill=None,
        partition=None,
        paginate=False,
    )
    main_calls = [
        item for item in analysis.must_change if item.path == "main.py" and item.kind == "call"
    ]
    assert len(main_calls) == 2
    first, second = main_calls
    assert first.edit_start_byte is not None and first.edit_end_byte is not None
    assert second.edit_start_byte is not None and second.edit_end_byte is not None
    # Point the second call's span exactly at the first's: a duplicate the
    # span dedupe missed. The slice still spells the identifier, so the
    # overlap defense is what must refuse it -- omitted, never merged.
    mutated = analysis.model_copy(
        update={
            "must_change": [
                item.model_copy(
                    update={
                        "edit_start_byte": first.edit_start_byte,
                        "edit_end_byte": first.edit_end_byte,
                    }
                )
                if item is second
                else item
                for item in analysis.must_change
            ]
        }
    )

    result = service._render_patch(mutated, query, operation, context_lines=3)

    overlap = [
        item for item in result.conflicted if "overlaps an already accepted" in item.explanation
    ]
    assert len(overlap) == 1
    # Declaration (auth.py), import (main.py), and the first call render; the
    # overlapping second call is left out entirely.
    assert result.applied == 3
    assert "+    return validate(u) + validate(u)\n" not in result.patch


def test_signature_change_emission_is_rejected(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, {"mail.py": "def send(message):\n    return message\n"}
    )

    with pytest.raises(CodeIndexingError) as excinfo:
        service.emit_refactor_patch(
            DeclarationSelector(project=project_id, path="mail.py", qualified_symbol="send"),
            SignatureChangeOperation(
                parameters=[
                    ParameterShape(name="body", kind="positional", required=True, position=0)
                ]
            ),
        )
    assert excinfo.value.code == ErrorCode.UNSUPPORTED_OPERATION


def test_a_crlf_file_emits_a_crlf_hunk(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, {"auth.py": "def authorize(user):\r\n    return user\r\n"}
    )

    result = _emit(service, project_id, "auth.py", "authorize", "validate")

    assert result.patch == (
        "diff --git a/auth.py b/auth.py\n"
        "--- a/auth.py\n"
        "+++ b/auth.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def authorize(user):\r\n"
        "+def validate(user):\r\n"
        "     return user\r\n"
    )


def test_a_non_ascii_identifier_emits_a_byte_exact_hunk(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path, {" café.py": "def café_x(user):\n    return user\n"}
    )

    result = _emit(service, project_id, " café.py", "café_x", "cafe_x")

    assert result.patch == (
        "diff --git a/ café.py b/ café.py\n"
        "--- a/ café.py\n"
        "+++ b/ café.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def café_x(user):\n"
        "+def cafe_x(user):\n"
        "     return user\n"
    )


def test_emission_is_deterministic(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "auth.py": "def authorize(user):\n    return user\n",
            "consumer.py": (
                "from auth import authorize\n\ndef run(user):\n    return authorize(user)\n"
            ),
        },
    )

    first = _emit(service, project_id, "auth.py", "authorize", "validate")
    second = _emit(service, project_id, "auth.py", "authorize", "validate")

    assert first == second


def test_emission_returns_every_finding_regardless_of_page_limit(tmp_path: Path) -> None:
    callers = "".join(f"def caller_{index}():\n    return authorize()\n\n" for index in range(11))
    service, project_id = _indexed_service(
        tmp_path, {"lib.py": f"def authorize():\n    return 42\n\n{callers}"}
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="authorize")
    operation = RenameOperation(new_name="validate")

    first_page = service.analyze_refactor(selector, operation, limit=1)
    total = service.analyze_refactor(selector, operation).counts.must_change
    result = service.emit_refactor_patch(selector, operation)

    assert len(first_page.must_change) < total
    assert result.applied == total
    assert result.unapplied == []
    assert result.completeness.state == "complete"
