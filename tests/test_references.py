from pathlib import Path

import pyarrow as pa
import pytest

from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import REFERENCE_SCHEMA_VERSION, Indexer
from code_indexing_mcp.models import DeclarationSelector
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore, PartitionRef


class TinyEmbedder:
    model_id = "test/reference"
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


def test_direct_python_import_alias_resolves_exactly(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "from lib import answer as local\n\ndef caller():\n    return local()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "direct_import_alias"
    assert call.snippet == "local"


def test_absolute_import_within_a_package_resolves_exactly(tmp_path: Path) -> None:
    """A same-package absolute import (`from mypkg.lib import answer`) written
    inside `mypkg/main.py` must anchor at the project root, not at `mypkg/`
    itself -- otherwise the generated candidate is `mypkg/mypkg/lib.py`,
    which never matches (finding 2).
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/lib.py": "def answer():\n    return 42\n",
            "mypkg/main.py": (
                "from mypkg.lib import answer\n\ndef caller():\n    return answer()\n"
            ),
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="mypkg/lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "direct_import_alias"


def test_absolute_import_under_a_src_layout_resolves_exactly(tmp_path: Path) -> None:
    """Same as above, but the package sits under a `src/` sub-root (this
    repo's own layout): `src/mypkg/main.py` importing `mypkg.lib`
    absolutely must anchor at `src/`, not at the true project root and not
    at `src/mypkg/` -- found by walking the unbroken `__init__.py` chain
    from the importing file up to (but not past) `src/`, which has none
    (finding 2 closure).
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "src/mypkg/__init__.py": "",
            "src/mypkg/lib.py": "def answer():\n    return 42\n",
            "src/mypkg/main.py": (
                "from mypkg.lib import answer\n\ndef caller():\n    return answer()\n"
            ),
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="src/mypkg/lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "direct_import_alias"


def test_absolute_import_under_src_layout_does_not_bind_the_wrong_directory(
    tmp_path: Path,
) -> None:
    """Negative control for the src-layout case above: a same-named module
    that happens to sit at the (wrong) project root, rather than under
    `src/`, must never be treated as an *exact* match just because some
    candidate set happened to include it -- proving false positives stay
    closed off, not just that the true positive works.

    `main.py`'s call site is unambiguous (it binds `src/mypkg/lib.py`), but
    the resolver classifies per selected declaration: asked about this
    decoy, the same call textually matches on name, so it is conservatively
    reported as `likely`/`unproven_reexport` rather than silently dropped
    -- it must never be upgraded to `exact`.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "src/mypkg/__init__.py": "",
            "src/mypkg/lib.py": "def answer():\n    return 42\n",
            "src/mypkg/main.py": (
                "from mypkg.lib import answer\n\ndef caller():\n    return answer()\n"
            ),
            # A coincidentally-named, unrelated module at the wrong anchor.
            "mypkg/lib.py": "def answer():\n    return 0\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="mypkg/lib.py", qualified_symbol="answer")
    )

    assert response.hits
    assert all(hit.resolution != "exact" for hit in response.hits)


@pytest.mark.parametrize(
    ("target_path", "target_source", "source_path", "source"),
    [
        (
            "lib.py",
            "def answer():\n    return 42\n",
            "main.py",
            "import lib as ns\n\nns.answer()\n",
        ),
        (
            "lib.py",
            "def answer():\n    return 42\n",
            "main.py",
            "import lib\n\nlib.answer()\n",
        ),
        (
            "lib.js",
            "export function answer() { return 42; }\n",
            "main.js",
            "import * as ns from './lib';\nns.answer();\n",
        ),
        (
            "lib.ts",
            "export function answer(): number { return 42; }\n",
            "main.ts",
            "import * as ns from './lib';\nns.answer();\n",
        ),
        (
            "lib.tsx",
            "export function answer(): number { return 42; }\n",
            "main.tsx",
            "import * as ns from './lib';\nns.answer();\n",
        ),
    ],
)
def test_namespace_member_imports_resolve_exactly(
    tmp_path: Path,
    target_path: str,
    target_source: str,
    source_path: str,
    source: str,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {target_path: target_source, source_path: source},
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path=target_path, qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "exact"
    assert call.reason_code == "known_namespace_member"
    assert not any(hit.kind == "import" for hit in response.hits)
    assert not any(limitation.code == "wildcard_import" for limitation in response.limitations)


def test_unimported_bare_call_with_one_project_wide_candidate_is_name_only(
    tmp_path: Path,
) -> None:
    """A bare call with no local declaration and exactly one same-named declaration anywhere.

    Exercises `_classify`'s project-wide ambiguity fallback (E2), which now
    looks candidates up through a precomputed `target_name` index instead of
    scanning every declaration per reference row -- this must still find
    exactly the declarations that share the bare name, nothing more or less.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "def caller():\n    return answer()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "likely"
    assert call.reason_code == "name_only_candidate"


def test_unimported_bare_call_with_two_project_wide_candidates_is_ambiguous(
    tmp_path: Path,
) -> None:
    """The same shape as above, but a second file also declares the same bare name.

    Two declarations sharing `target_name` project-wide must downgrade the
    call from `name_only_candidate` to `ambiguous_symbol` -- proving the E2
    index groups by name across every file, not just the one being queried.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "other.py": "def answer():\n    return 0\n",
            "main.py": "def caller():\n    return answer()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "unresolved"
    assert call.reason_code == "ambiguous_symbol"


def test_python_wildcard_import_remains_unresolved(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "from lib import *\n\nanswer()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "unresolved"
    assert call.reason_code == "wildcard_import"


def test_direct_import_named_like_its_module_is_not_a_namespace(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "from lib import lib as ns\n\nns.answer()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "likely"
    assert call.reason_code == "unknown_receiver"


def test_unknown_member_receiver_is_never_exact(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "def caller(thing):\n    return thing.answer()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call = next(hit for hit in response.hits if hit.kind == "call")
    assert call.resolution == "likely"
    assert call.reason_code == "unknown_receiver"


def test_same_file_shadowed_call_does_not_bind_the_selected_declaration(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": (
                "def target():\n"
                "    return 1\n\n"
                "def direct():\n"
                "    return target()\n\n"
                "def outer():\n"
                "    def target():\n"
                "        return 2\n"
                "    return target()\n"
            )
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target")
    )

    calls = [hit for hit in response.hits if hit.kind == "call"]
    assert [(call.start_line, call.resolution) for call in calls] == [(5, "exact")]


def test_find_references_reports_a_missing_reference_table_distinctly(tmp_path: Path) -> None:
    """A legacy/never-built reference index must not read as "no references" (S5)."""
    service, project_id = _indexed_service(
        tmp_path,
        {"lib.py": "def answer():\n    return 42\n"},
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    # Sanity: the freshly indexed project answers normally first.
    assert service.find_references(selector).hits == []

    store = service.store
    store._partitions.pop(project_id, None)
    (store.directory / "projects" / project_id / "references.lance").rename(
        tmp_path / "references.lance.bak"
    )

    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(selector)
    assert excinfo.value.code == ErrorCode.REFERENCE_INDEX_UNAVAILABLE


def test_cursor_is_filter_bound_and_reads_its_original_snapshot(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": (
                "from lib import answer\n\ndef a(): return answer()\ndef b(): return answer()\n"
            ),
        },
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")

    first = service.find_references(selector, limit=1)
    assert first.cursor is not None
    second = service.find_references(selector, limit=1, cursor=first.cursor)

    assert second.hits
    # `CodeIndexingError` does not subclass `ValueError` (see errors.py) --
    # a cursor/filter mismatch is a structured, machine-readable error, not
    # a bare `ValueError` that would bypass `_with_error_details` and reach
    # the client as a raw exception message (T2).
    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(selector, kinds={"call"}, cursor=first.cursor)
    assert excinfo.value.code == ErrorCode.INVALID_CURSOR

    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(selector, limit=2, cursor=first.cursor)
    assert excinfo.value.code == ErrorCode.INVALID_CURSOR

    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(selector, limit=1, cursor="not-a-real-cursor")
    assert excinfo.value.code == ErrorCode.INVALID_CURSOR

    cursor_payload = service._decode_cursor(first.cursor)
    cursor_payload["slot_id"] = "inactive-slot"
    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(
            selector,
            limit=1,
            cursor=service._encode_cursor(cursor_payload),
        )
    assert excinfo.value.code == ErrorCode.STALE_CURSOR

    file_id = service.store.list_files(project_id)[0].file_id
    service.store.replace_files_from_arrow(
        project_id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[
            ([file_id], pa.Table.from_batches([], schema=LanceStore.reference_arrow_schema()))
        ],
    )
    original_snapshot = service.find_references(selector, limit=1, cursor=first.cursor)
    assert original_snapshot.hits
    assert original_snapshot.snapshot_version == first.snapshot_version


def test_cursor_switch_is_stale_before_declaration_lookup(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": (
                "from lib import answer\n\ndef a(): return answer()\ndef b(): return answer()\n"
            ),
        },
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    first = service.find_references(selector, limit=1)
    assert first.cursor is not None
    active = service.store.active_partition(project_id)
    active_slot = service.store.get_slot(active.slot_id)
    assert active_slot is not None
    pending = active_slot.model_copy(
        update={
            "slot_id": "other-slot",
            "partition_id": "slot-other-slot",
            "selector_kind": "ref",
            "selector_value": "refs/heads/other",
            "state": "pending",
        }
    )
    service.store.upsert_slot(pending)
    epoch = service.store.activate_slot(project_id, pending.slot_id)
    partition = PartitionRef(project_id, pending.slot_id, pending.partition_id, epoch)

    with pytest.raises(CodeIndexingError) as excinfo:
        service.find_references(selector, limit=1, cursor=first.cursor, partition=partition)

    assert excinfo.value.code is ErrorCode.STALE_CURSOR


def test_stale_schema_version_rows_are_not_served(tmp_path: Path) -> None:
    """A row left behind under an old `REFERENCE_SCHEMA_VERSION` must never surface.

    The version bump to the current schema was explicitly meant to discard
    the previous generation's id scheme; a row that survives a partial
    reindex under the old version carries stale offsets and a colliding id
    shape, so it must be excluded on the read path even though
    `list_reference_records` itself applies no filter (finding 9).
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
        },
    )
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")

    baseline = service.find_references(selector)
    call_hits_before = [hit for hit in baseline.hits if hit.kind == "call"]
    assert len(call_hits_before) == 1

    store = service.store
    main_file_id = next(
        item.file_id for item in store.list_files(project_id) if item.path == "main.py"
    )
    current_rows = [
        row for row in store.list_reference_records(project_id) if row["file_id"] == main_file_id
    ]
    stale_row = dict(next(row for row in current_rows if row["kind"] == "call"))
    stale_row["reference_id"] = "stale-v3-call"
    stale_row["schema_version"] = REFERENCE_SCHEMA_VERSION - 1

    table = pa.Table.from_pylist(
        [*current_rows, stale_row], schema=LanceStore.reference_arrow_schema()
    )
    store.replace_files_from_arrow(
        project_id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[([main_file_id], table)],
    )

    after = service.find_references(selector)
    call_hits_after = [hit for hit in after.hits if hit.kind == "call"]
    assert len(call_hits_after) == 1
    assert {hit.reference_id for hit in call_hits_after} == {
        hit.reference_id for hit in call_hits_before
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"chunk_id": "chunk", "project": "project", "path": "x.py", "qualified_symbol": "x"},
        {"project": "project", "path": "x.py"},
    ],
)
def test_declaration_selector_requires_one_complete_mode(payload: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        DeclarationSelector(**payload)


def test_a_class_body_does_not_shadow_a_module_level_function(tmp_path: Path) -> None:
    """A method is not in a sibling method's scope chain.

    Python and JS/TS both resolve a bare `helper()` inside `Gate.run` to the
    module-level `helper`. Treating the class body as an enclosing scope made
    the resolver discard that call site, so a rename reported no callers at all.
    """

    service, project_id = _indexed_service(
        tmp_path,
        {
            "app.py": (
                "def helper():\n"
                "    return 1\n"
                "\n"
                "class Gate:\n"
                "    def helper(self):\n"
                "        return 2\n"
                "\n"
                "    def run(self):\n"
                "        return helper()\n"
            )
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="app.py", qualified_symbol="helper")
    )

    calls = [hit for hit in response.hits if hit.kind == "call"]
    assert [(call.start_line, call.resolution) for call in calls] == [(9, "exact")]


def test_a_method_still_shadows_a_reference_made_through_its_own_receiver(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "app.py": (
                "def helper():\n"
                "    return 1\n"
                "\n"
                "class Gate:\n"
                "    def helper(self):\n"
                "        return 2\n"
                "\n"
                "    def run(self):\n"
                "        return self.helper()\n"
            )
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="app.py", qualified_symbol="helper")
    )

    # `self.helper()` names the method, so the module function must not claim it
    # as an exact use.
    assert all(hit.resolution != "exact" for hit in response.hits if hit.kind == "call")


def test_an_unproven_receiver_is_reported_as_a_limitation(tmp_path: Path) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "def caller(thing):\n    return thing.answer()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    assert any(item.code == "unknown_receiver" for item in response.limitations)


def test_files_without_reference_extraction_are_reported_as_a_coverage_gap(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "svc.c": "int Run(void) {\n\treturn 1;\n}\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    limitation = next(item for item in response.limitations if item.code == "unsupported_language")
    assert "svc.c" in limitation.explanation
    assert "1 c file(s)" in limitation.explanation


def test_go_files_stop_being_a_coverage_gap_once_structural(tmp_path: Path) -> None:
    """Coverage flips only for the newly supported language.

    Indexing a mixed Python + Go + C project must stop reporting
    `unsupported_language` for the Go files while the C files stay reported,
    proving the flip is scoped to the language that gained extraction.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.go": 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("hi")\n}\n',
            "svc.c": "int Run(void) {\n\treturn 1;\n}\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    go_gaps = [
        item
        for item in response.limitations
        if item.code == "unsupported_language" and "main.go" in item.explanation
    ]
    assert not go_gaps
    c_gaps = [
        item
        for item in response.limitations
        if item.code == "unsupported_language" and "svc.c" in item.explanation
    ]
    assert c_gaps


def test_find_references_narrows_the_declaration_fetch_to_files_with_a_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declarations are fetched narrowly, not project-wide, per page (S4/E3).

    `unrelated.py` holds only its own declaration (no calls, no imports, no
    reads) -- nothing in `_lexical_declaration`/class-scope resolution ever
    looks it up, so `declarations_for_files` must never be asked for it. A
    caller passing "every known file" here would just be a redundant round
    trip back to the same data (E3 finding 2's own reasoning) -- this proves
    the file set actually passed is a real, proper subset.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
            "unrelated.py": "def other():\n    pass\n",
        },
    )
    declaration_calls: list[frozenset[str]] = []
    real_declarations_for_files = service.store.declarations_for_files

    def spy_declarations_for_files(
        project: str, file_ids: object, **kwargs: object
    ) -> list[object]:
        declaration_calls.append(frozenset(file_ids))  # type: ignore[arg-type]
        return real_declarations_for_files(project, file_ids, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "declarations_for_files", spy_declarations_for_files)

    target_calls: list[tuple[str, str | None]] = []
    real_target_name_candidates = service.store.target_name_candidates

    def spy_target_name_candidates(
        project: str, target_name: str, **kwargs: object
    ) -> list[object]:
        target_calls.append((target_name, kwargs.get("record_kind")))
        return real_target_name_candidates(project, target_name, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "target_name_candidates", spy_target_name_candidates)

    record_kinds_seen: list[object] = []
    real_list_reference_records = service.store.list_reference_records

    def spy_list_reference_records(project: str, **kwargs: object) -> list[object]:
        record_kinds_seen.append(kwargs.get("record_kinds"))
        return real_list_reference_records(project, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(service.store, "list_reference_records", spy_list_reference_records)

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    # The pushdowns were actually used, not just left available.
    assert len(declaration_calls) == 1
    assert target_calls == [("answer", "declaration")]
    assert record_kinds_seen == [("reference", "coverage")]

    # And the narrowing is real: the fetched file set excludes the file with
    # no candidate reference, while still covering the one that matters.
    files_by_id = {item.file_id: item.path for item in service.store.list_files(project_id)}
    fetched_paths = {files_by_id[file_id] for file_id in declaration_calls[0]}
    assert "unrelated.py" not in fetched_paths
    assert "main.py" in fetched_paths

    # Still the correct answer.
    call_hits = [hit for hit in response.hits if hit.kind == "call"]
    assert len(call_hits) == 1
    assert call_hits[0].path == "main.py"
    assert call_hits[0].resolution == "exact"


def test_find_references_follows_a_renaming_two_hop_barrel_alias(tmp_path: Path) -> None:
    """A reference row's own `target_name` can be an arbitrary local alias.

    `pkg/__init__.py` re-exports `answer` under a *different* local name
    (`ans_alias`), and `importer.py` imports that renamed binding under yet
    another alias (`x2`) -- so the final call site's own `target_name` is
    `"x2"`, a spelling that cannot be predicted from `"answer"` by any single
    SQL predicate, and not even by a static candidate set computed in one
    query (E3 finding 1): `importer.py`'s import row's own `target_name` is
    `"ans_alias"`, which only becomes knowable after `pkg/__init__.py`'s row
    has already been read. This is the concrete case backing the decision to
    leave the reference-side scan unfiltered while narrowing declarations
    (S4/E3); it must keep working exactly as before.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "pkg/__init__.py": "from lib import answer as ans_alias\n",
            "importer.py": "from pkg import ans_alias as x2\n\ndef use():\n    return x2()\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    call_hits = [hit for hit in response.hits if hit.kind == "call" and hit.path == "importer.py"]
    assert len(call_hits) == 1
    assert call_hits[0].written_name == "x2"
    assert call_hits[0].resolution == "exact"
    assert call_hits[0].reason_code == "reexport_chain"


def test_known_paths_completeness_lets_absolute_imports_skip_a_sibling_decoy(
    tmp_path: Path,
) -> None:
    """Narrowing the *declaration* fetch (S4/E3) must not narrow `known_paths`.

    `mypkg/__init__.py` is empty -- zero declarations, zero references -- so
    the only way its path reaches `known_paths` is through its coverage row.
    `_python_package_root` needs that path to recognize `mypkg` as a real
    package and anchor `mypkg/other.py`'s *absolute* import at the project
    root rather than at `mypkg` itself; without it, the decoy
    `mypkg/utils.py` (same name, wrong module) would be indistinguishable
    from the real top-level `utils.py`.
    """
    service, project_id = _indexed_service(
        tmp_path,
        {
            "mypkg/__init__.py": "",
            "mypkg/utils.py": "def f():\n    return 'decoy'\n",
            "mypkg/other.py": "from utils import f\n\ndef use():\n    return f()\n",
            "utils.py": "def f():\n    return 'real'\n",
        },
    )

    real_response = service.find_references(
        DeclarationSelector(project=project_id, path="utils.py", qualified_symbol="f")
    )
    call_hits = [hit for hit in real_response.hits if hit.kind == "call"]
    assert len(call_hits) == 1
    assert call_hits[0].path == "mypkg/other.py"
    assert call_hits[0].resolution == "exact"

    # The bare call `f()` is still a name-only *candidate* against the decoy
    # (any same-named declaration is a candidate -- see `_may_refer`), but it
    # must not be graded `exact`: `known_paths` correctly told
    # `_python_package_root` that `mypkg` is a real package, so the import's
    # module resolves to the top-level `utils.py`, not `mypkg/utils.py`.
    decoy_response = service.find_references(
        DeclarationSelector(project=project_id, path="mypkg/utils.py", qualified_symbol="f")
    )
    decoy_hit = next(hit for hit in decoy_response.hits if hit.path == "mypkg/other.py")
    assert decoy_hit.resolution != "exact"


def test_references_from_a_file_changed_since_extraction_are_suppressed_as_stale(
    tmp_path: Path,
) -> None:
    service, project_id = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    return 42\n",
            "main.py": "from lib import answer\n\ndef caller():\n    return answer()\n",
        },
    )
    # The file changes on disk without a reindex: its stored rows now describe
    # bytes that no longer exist, so their offsets must not be served against
    # the new content.
    (tmp_path / "repo" / "main.py").write_text(
        "from lib import answer\n\n\ndef caller():\n    return answer()\n"
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    assert response.hits == []
    limitation = next(item for item in response.limitations if item.code == "stale_file")
    assert "main.py" in limitation.explanation


def test_references_return_once_a_stale_file_is_reindexed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "lib.py").write_text("def answer():\n    return 42\n")
    (root / "main.py").write_text("from lib import answer\n\ndef caller():\n    return answer()\n")
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
    # Same divergence as above, then a real index run heals it.
    (root / "main.py").write_text(
        "from lib import answer\n\n\ndef caller():\n    return answer()\n"
    )
    service = ReferenceService(store)
    selector = DeclarationSelector(project=project.id, path="lib.py", qualified_symbol="answer")
    assert service.find_references(selector).hits == []

    indexer.index(project)

    healed = service.find_references(selector)
    assert any(hit.path == "main.py" and hit.kind == "call" for hit in healed.hits)
    assert all(item.code != "stale_file" for item in healed.limitations)
