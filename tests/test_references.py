from pathlib import Path

import pyarrow as pa
import pytest

from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
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


def test_cursor_is_filter_bound_and_rejects_a_changed_snapshot(tmp_path: Path) -> None:
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
    with pytest.raises(CodeIndexingError) as raised:
        service.find_references(selector, limit=1, cursor=first.cursor)
    assert raised.value.code is ErrorCode.STALE_CURSOR


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
