from pathlib import Path

import pytest

from incode_mcp.extractor import TreeSitterExtractor
from incode_mcp.indexing import Indexer
from incode_mcp.projects import initialize_project
from incode_mcp.scanner import SourceScanner
from incode_mcp.search import SearchService
from incode_mcp.storage import LanceStore


class SemanticEmbedder:
    model_id = "test/semantic-code"
    dimension = 4

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "auth" in lowered or "permission" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        if "invoice" in lowered or "billing" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def indexed_projects(tmp_path: Path) -> tuple[LanceStore, SearchService, list[str]]:
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    )
    project_ids = []
    sources = {
        "auth": "def enforce_permissions(user):\n    return user.is_admin\n",
        "billing": "def create_invoice(order):\n    return order.total\n",
    }
    for name, source in sources.items():
        root = tmp_path / name
        root.mkdir()
        (root / f"{name}.py").write_text(source)
        project = initialize_project(root)
        project_ids.append(project.id)
        indexer.index(project)
    return store, SearchService(store, embedder), project_ids


def test_hybrid_search_respects_project_scope_and_filters(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)

    auth = search.search_code("where are permissions enforced", [projects[0]])
    billing = search.search_code("create billing invoice", [projects[1]], languages=["python"])

    assert auth.hits[0].symbol == "enforce_permissions"
    assert {hit.project_id for hit in auth.hits} == {projects[0]}
    assert billing.hits[0].symbol == "create_invoice"
    assert search.search_code("invoice", [projects[0]], paths=["billing/**"]).hits == []


def test_structural_queries_do_not_materialize_full_chunk_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, search, projects = indexed_projects(tmp_path)

    def reject_full_rows(*_: object, **__: object) -> object:
        raise AssertionError("structural queries must not call list_chunks")

    monkeypatch.setattr(store, "list_chunks", reject_full_rows)

    symbols = search.find_symbol("enforce_permissions", projects[0])
    outline = search.file_outline("auth.py", projects[0])

    assert symbols.hits[0].symbol == "enforce_permissions"
    assert outline.items[0].symbol == "enforce_permissions"


def test_hybrid_search_projects_result_columns(tmp_path: Path) -> None:
    store, _, projects = indexed_projects(tmp_path)

    rows = store.hybrid_search(
        "permissions",
        [1.0, 0.0, 0.0, 0.0],
        [projects[0]],
        None,
        5,
    )

    assert rows
    assert "vector" not in rows[0]
    assert "embedding_text" not in rows[0]
    assert "search_text" not in rows[0]


def test_symbol_lookup_and_outline_use_indexed_metadata(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)

    exact = search.find_symbol("enforce_permissions", projects[0])
    prefix = search.find_symbol("create_", projects[1], match="prefix")
    outline = search.file_outline("auth.py", projects[0])

    assert [hit.qualified_symbol for hit in exact.hits] == ["enforce_permissions"]
    assert [hit.qualified_symbol for hit in prefix.hits] == ["create_invoice"]
    assert [item.qualified_symbol for item in outline.items] == ["enforce_permissions"]


def test_java_symbols_are_indexed_and_filterable(tmp_path: Path) -> None:
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "User.java").write_text(
        "record User(String name) {\n    User {}\n    String display() { return name; }\n}\n"
    )
    project = initialize_project(root)
    Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    ).index(project)
    search = SearchService(store, embedder)

    outline = search.file_outline("User.java", project.id)
    constructors = search.find_symbol("User.User", project.id, kinds=["constructor"])
    methods = search.search_code("display", [project.id], languages=["java"], kinds=["method"])

    assert {(item.kind, item.qualified_symbol) for item in outline.items} >= {
        ("record", "User"),
        ("constructor", "User.User"),
        ("method", "User.display"),
    }
    assert [hit.qualified_symbol for hit in constructors.hits] == ["User.User"]
    assert methods.hits[0].language == "java"
    assert methods.hits[0].kind == "method"


def test_search_truncates_snippet_and_get_chunk_returns_full_content(tmp_path: Path) -> None:
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    root = tmp_path / "repo"
    root.mkdir()
    payload = "a" * 5_000
    (root / "auth.py").write_text(
        f"def authenticate():\n    token = '{payload}'\n    return token\n"
    )
    project = initialize_project(root)
    Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(max_chars=8_000),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    ).index(project)
    search = SearchService(store, embedder)

    result = search.search_code("authenticate", [project.id])
    full = search.get_chunk(result.hits[0].chunk_id)

    assert len(result.hits[0].snippet) == 4_000
    assert result.hits[0].truncated is True
    assert len(full.content) > len(result.hits[0].snippet)
