from pathlib import Path

import pyarrow as pa
import pytest

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import DeclarationSelector
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


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
    with pytest.raises(ValueError):
        service.find_references(selector, kinds={"call"}, cursor=first.cursor)
    file_id = service.store.list_files(project_id)[0].file_id
    service.store.replace_files_from_arrow(
        project_id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_groups=(),
        reference_groups=[
            (file_id, pa.Table.from_batches([], schema=LanceStore.reference_arrow_schema()))
        ],
        replace_reference_file_ids=[file_id],
    )
    original_snapshot = service.find_references(selector, limit=1, cursor=first.cursor)
    assert original_snapshot.hits
    assert original_snapshot.snapshot_version == first.snapshot_version


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
            "svc.go": "package main\n\nfunc Run() int {\n\treturn 1\n}\n",
        },
    )

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="answer")
    )

    limitation = next(item for item in response.limitations if item.code == "unsupported_language")
    assert "svc.go" in limitation.explanation
    assert "go" in limitation.explanation
