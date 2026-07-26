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


def _indexed_source(tmp_path: Path, source: str) -> tuple[SearchService, str]:
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    )
    root = tmp_path / "repo"
    root.mkdir()
    (root / "module.py").write_text(source)
    project = initialize_project(root)
    indexer.index(project)
    return SearchService(store, embedder), project.id


def test_symbol_match_does_not_treat_underscores_as_wildcards(tmp_path: Path) -> None:
    """The LIKE pushdown over-matches; exact semantics must be re-applied."""
    search, project_id = _indexed_source(
        tmp_path,
        "def load_user():\n    return 1\n\n\ndef loadXuser():\n    return 2\n",
    )

    prefix = search.find_symbol("load_user", project_id, match="prefix")
    contains = search.find_symbol("load_user", project_id, match="contains")

    assert {hit.symbol for hit in prefix.hits} == {"load_user"}
    assert {hit.symbol for hit in contains.hits} == {"load_user"}


def test_symbol_results_are_ordered_before_the_limit_applies(tmp_path: Path) -> None:
    source = "".join(f"def handler_{index}():\n    return {index}\n\n\n" for index in range(12))
    search, project_id = _indexed_source(tmp_path, source)

    hits = search.find_symbol("handler_", project_id, match="prefix", limit=3)

    assert [hit.symbol for hit in hits.hits] == ["handler_0", "handler_1", "handler_2"]


def _indexed_tree(tmp_path: Path, sources: dict[str, str]) -> tuple[SearchService, str]:
    """Index one project whose files are given as {relative path: source}."""
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    )
    root = tmp_path / "tree"
    root.mkdir()
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    project = initialize_project(root)
    indexer.index(project)
    return SearchService(store, embedder), project.id


def test_path_filter_finds_matches_below_the_fetch_window(tmp_path: Path) -> None:
    """A match that ranks outside the fetch window must survive a path filter.

    Before the pushdown, `paths` was applied in Python to rows the scan had already
    truncated, so a low-ranking match in a rare directory returned zero hits even
    though find_symbol proved it was indexed. The noise files repeat the query terms
    so they all outrank the needle deterministically, rather than by tie-break luck.
    """
    sources = {
        f"noise/m{index}.py": (
            f"def enforce_permissions_{index}(user):\n"
            "    'permission permission permission check'\n"
            "    return user.permission\n"
        )
        for index in range(120)
    }
    sources["rare/needle.py"] = "def audit_gate(user):\n    'permission'\n    return user.allowed\n"
    search, project = _indexed_tree(tmp_path, sources)

    unfiltered = search.search_code("permission check", [project], limit=8)
    filtered = search.search_code("permission check", [project], paths=["rare/*"], limit=8)

    assert {hit.path.split("/")[0] for hit in unfiltered.hits} == {"noise"}
    assert [hit.path for hit in filtered.hits] == ["rare/needle.py"]


def test_path_filter_respects_right_anchored_glob_semantics(tmp_path: Path) -> None:
    search, project = _indexed_tree(
        tmp_path,
        {
            "src/deep/b.py": "def alpha_one():\n    return 1\n",
            "src/a.py": "def alpha_two():\n    return 2\n",
            "tests/c.py": "def alpha_three():\n    return 3\n",
        },
    )

    # "src/*" spans one segment; "*.py" matches at any depth.
    assert {hit.path for hit in search.search_code("alpha", [project], paths=["src/*"]).hits} == {
        "src/a.py"
    }
    assert {hit.path for hit in search.search_code("alpha", [project], paths=["*.py"]).hits} == {
        "src/deep/b.py",
        "src/a.py",
        "tests/c.py",
    }


def test_untranslatable_path_pattern_still_filters_in_python(tmp_path: Path) -> None:
    search, project = _indexed_tree(
        tmp_path,
        {
            "src/a.py": "def alpha_two():\n    return 2\n",
            "tests/c.py": "def alpha_three():\n    return 3\n",
        },
    )

    # An absolute pattern disables the pushdown; the post-filter must still apply,
    # and PurePosixPath.match never matches an absolute pattern against these paths.
    assert search.search_code("alpha", [project], paths=["/src/a.py"]).hits == []
