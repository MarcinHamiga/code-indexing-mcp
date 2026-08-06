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
        destination.write_text(source)
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
    alias_call = next(item for item in analysis.findings if item.written_name == "check")
    assert alias_call.resolution == "exact"
    assert not alias_call.edit_required


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
