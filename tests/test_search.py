from pathlib import Path
from typing import Any

import pytest

from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import ExampleSearchResponse, SearchResponse
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.search import (
    _EXAMPLE_SUFFIX,
    MAX_EXAMPLE_LENGTH,
    SearchService,
    _example_passages,
    detect_example_language,
)
from code_indexing_mcp.storage import LanceStore, PartitionRef


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


def test_search_code_hits_carry_ranking_explanation(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)

    response = search.search_code("permissions", [projects[0]])

    assert response.hits
    top = response.hits[0]
    assert top.symbol == "enforce_permissions"
    assert top.explanation is not None
    assert top.explanation.fts_score is not None and top.explanation.fts_score > 0
    assert top.explanation.vector_score is not None
    assert top.explanation.vector_score == pytest.approx(1.0)
    assert top.explanation.fts_rank == 1
    assert top.explanation.vector_rank == 1
    # The field is additive to the response schema.
    reloaded = SearchResponse.model_validate(response.model_dump(mode="json"))
    assert reloaded == response


def test_search_code_explanation_distinguishes_signals(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)

    response = search.search_code("permissions", projects)

    by_symbol = {hit.symbol: hit for hit in response.hits}
    assert set(by_symbol) == {"enforce_permissions", "create_invoice"}
    matched = by_symbol["enforce_permissions"].explanation
    unmatched = by_symbol["create_invoice"].explanation
    assert matched is not None and unmatched is not None
    assert matched.fts_score is not None
    assert unmatched.fts_score is None
    assert matched.vector_score is not None and unmatched.vector_score is not None
    assert matched.vector_score > unmatched.vector_score
    assert matched.vector_rank == 1
    assert unmatched.vector_rank == 2


def test_search_code_explanation_absent_when_probes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, search, projects = indexed_projects(tmp_path)

    def reject_probes(*_: object, **__: object) -> object:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(store, "explain_hits", reject_probes)

    response = search.search_code("permissions", [projects[0]])

    assert response.hits
    assert response.hits[0].symbol == "enforce_permissions"
    assert all(hit.explanation is None for hit in response.hits)


def test_multi_project_search_is_deterministic_across_concurrent_runs(
    tmp_path: Path,
) -> None:
    """Five partitions run through the bounded pool and merge identically every run."""
    embedder = SemanticEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    )
    project_ids: list[str] = []
    for index in range(5):
        root = tmp_path / f"p{index}"
        root.mkdir()
        (root / "mod.py").write_text(f"def feature_{index}(user):\n    return user.is_admin\n")
        project = initialize_project(root)
        project_ids.append(project.id)
        indexer.index(project)
    search = SearchService(store, embedder)

    baseline = search.search_code("permissions", project_ids)
    for order in (project_ids, list(reversed(project_ids)), project_ids[2:] + project_ids[:2]):
        repeated = search.search_code("permissions", order)
        assert [hit.chunk_id for hit in repeated.hits] == [hit.chunk_id for hit in baseline.hits]
        assert [hit.score for hit in repeated.hits] == [hit.score for hit in baseline.hits]

    assert len(baseline.hits) == 5
    assert {hit.project_id for hit in baseline.hits} == set(project_ids)
    assert [hit.score for hit in baseline.hits] == pytest.approx([1 / 61] * 5)


def test_multi_project_vector_relevance_is_independent_of_project_order(tmp_path: Path) -> None:
    store, search, projects = indexed_projects(tmp_path)
    # "authorization" embeds onto the auth axis but has no lexical match.
    for order in (projects, list(reversed(projects))):
        hits = search.search_code("authorization", order, limit=1).hits
        assert [hit.symbol for hit in hits] == ["enforce_permissions"]
        rows = store.hybrid_search("authorization", [1.0, 0.0, 0.0, 0.0], order, None, 2)
        assert [row["symbol"] for row in rows] == ["enforce_permissions", "create_invoice"]
        assert [row["_relevance_score"] for row in rows] == pytest.approx([1 / 61, 1 / 62])


def test_partition_lexical_winners_share_rank_before_global_fusion(tmp_path: Path) -> None:
    store, _, projects = indexed_projects(tmp_path)
    # Both singleton partitions match "return" lexically. Their top lexical
    # evidence contributes equally; their distinct vector relevance does not.
    for order in (projects, list(reversed(projects))):
        rows = store.hybrid_search("return", [1.0, 0.0, 0.0, 0.0], order, None, 2)
        assert [row["symbol"] for row in rows] == ["enforce_permissions", "create_invoice"]
        assert [row["_relevance_score"] for row in rows] == pytest.approx([2 / 61, 1 / 61 + 1 / 62])


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


def test_explicit_active_slot_never_reads_an_inactive_chunk_partition(tmp_path: Path) -> None:
    store, search, projects = indexed_projects(tmp_path)
    project_id = projects[0]
    active = store.active_partition(project_id)
    old_chunk_id = search.find_symbol("enforce_permissions", project_id).hits[0].chunk_id
    active_slot = store.get_slot(active.slot_id)
    assert active_slot is not None
    pending = active_slot.model_copy(
        update={
            "slot_id": "pending-slot",
            "partition_id": "slot-pending-slot",
            "selector_kind": "ref",
            "selector_value": "refs/heads/pending",
            "state": "pending",
        }
    )
    store.upsert_slot(pending)
    epoch = store.activate_slot(project_id, pending.slot_id, checkout_key="checkout")
    partition = PartitionRef(project_id, pending.slot_id, pending.partition_id, epoch)

    response = search.search_code("permissions", [project_id], partitions={project_id: partition})

    assert response.hits == []
    with pytest.raises(CodeIndexingError) as excinfo:
        search.get_chunk(old_chunk_id, partition=partition)
    assert excinfo.value.code is ErrorCode.CHUNK_NOT_FOUND


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


@pytest.mark.parametrize("match", ["prefix", "contains"])
def test_symbol_lookup_survives_more_than_a_window_of_wildcard_false_positives(
    tmp_path: Path, match: str
) -> None:
    source = "".join(f"def loadXuser{i}():\n    return {i}\n\n" for i in range(205))
    source += "def load_user():\n    return 1\n"
    search, project_id = _indexed_source(tmp_path, source)

    hits = search.find_symbol("load_user", project_id, match=match).hits

    assert [hit.symbol for hit in hits] == ["load_user"]


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


def test_path_filter_uses_gitignore_style_recursive_globs(tmp_path: Path) -> None:
    search, project = _indexed_tree(
        tmp_path,
        {
            "top.py": "def alpha_zero():\n    return 0\n",
            "src/deep/b.py": "def alpha_one():\n    return 1\n",
            "src/a.py": "def alpha_two():\n    return 2\n",
            "src/pkg/deep/c.py": "def alpha_four():\n    return 4\n",
            "tests/c.py": "def alpha_three():\n    return 3\n",
            "vendor/x/src/y.py": "def alpha_five():\n    return 5\n",
        },
    )

    def found(*patterns: str) -> set[str]:
        response = search.search_code(
            "alpha", [project], paths=list(patterns), scope="all", limit=50
        )
        return {hit.path for hit in response.hits}

    everything = {
        "top.py",
        "src/deep/b.py",
        "src/a.py",
        "src/pkg/deep/c.py",
        "tests/c.py",
        "vendor/x/src/y.py",
    }
    # Before: "**/*.py" dropped top.py, "src/**" matched only src/a.py,
    # "src/**/*.py" only src/deep/b.py, "src/pkg" nothing.
    assert found("**/*.py") == everything
    assert found("*.py") == everything
    assert found("src/**") == {"src/deep/b.py", "src/a.py", "src/pkg/deep/c.py"}
    assert found("src/**/*.py") == {"src/deep/b.py", "src/a.py", "src/pkg/deep/c.py"}
    assert found("src/pkg") == {"src/pkg/deep/c.py"}
    # Before: "src/*" also matched vendor/x/src/y.py (right-anchored).
    assert found("src/*") == {"src/a.py"}
    assert found("src//a.py") == {"src/a.py"}
    assert found("*/") == everything


def test_path_filter_rejects_empty_and_absolute_patterns(tmp_path: Path) -> None:
    search, project = _indexed_tree(tmp_path, {"src/a.py": "def alpha_two():\n    return 2\n"})

    for bad in ("", "/src/a.py"):
        with pytest.raises(CodeIndexingError) as raised:
            search.search_code("alpha", [project], paths=[bad])
        assert raised.value.code is ErrorCode.INVALID_FILTER
        with pytest.raises(CodeIndexingError) as raised:
            search.search_by_example("def alpha_two():\n    return 2\n", [project], paths=[bad])
        assert raised.value.code is ErrorCode.INVALID_FILTER


def test_literal_empty_character_class_falls_back_without_a_regex_error(tmp_path: Path) -> None:
    search, project = _indexed_tree(
        tmp_path,
        {"literal/a[].py": "def alpha_literal():\n    return 1\n"},
    )

    assert [hit.path for hit in search.search_code("alpha", [project], paths=["*[]*.py"]).hits] == [
        "literal/a[].py"
    ]


CHUNK_PAYLOAD_FIELDS = {
    "chunk_id",
    "file_id",
    "project_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "content_hash",
    "part_index",
}


def test_get_chunk_excludes_embedding_and_duplicated_text(tmp_path: Path) -> None:
    """get_chunk is the full-text counterpart to a snippet, not a storage row dump.

    The vector was 72% of the old payload and the code text appeared three times
    over, in content, embedding_text, and search_text.
    """
    search, project = _indexed_source(tmp_path, "def authenticate(user):\n    return user.token\n")
    hit = search.search_code("authenticate", [project]).hits[0]

    payload = search.get_chunk(hit.chunk_id).model_dump()

    assert set(payload) == CHUNK_PAYLOAD_FIELDS
    assert "vector" not in payload
    assert "embedding_text" not in payload
    assert "search_text" not in payload
    assert "return user.token" in payload["content"]


def test_get_chunk_payload_is_dominated_by_content(tmp_path: Path) -> None:
    import json

    source = "def authenticate(user):\n" + "".join(
        f"    step_{index} = user.token\n" for index in range(80)
    )
    search, project = _indexed_source(tmp_path, source)
    hit = search.search_code("authenticate", [project]).hits[0]

    chunk = search.get_chunk(hit.chunk_id)
    encoded = json.dumps(chunk.model_dump(mode="json"))

    # Metadata is ~290 chars of ids and offsets; anything beyond a small multiple of
    # the content means a payload field crept back in.
    assert len(encoded) < len(chunk.content) * 2


def test_example_search_response_round_trip() -> None:
    response = ExampleSearchResponse(language=None, segments=0, hits=[])
    dumped = response.model_dump(mode="json")
    assert dumped == {"language": None, "segments": 0, "hits": []}
    loaded = ExampleSearchResponse.model_validate(dumped)
    assert loaded == response


def test_example_passages_python_snippet() -> None:
    extractor = TreeSitterExtractor()
    snippet = "def calculate_total(items):\n    return sum(items)\n"
    lang, passages = _example_passages(extractor, snippet, language="python")
    assert lang == "python"
    assert len(passages) >= 1
    assert passages[0].startswith("language: python")
    assert "return sum(items)" in passages[0]


def test_detect_example_language() -> None:
    extractor = TreeSitterExtractor()
    assert detect_example_language(extractor, "def foo():\n    return 1\n") == "python"
    assert detect_example_language(extractor, "func answer() int { return 42 }") == "go"
    assert (
        detect_example_language(
            extractor, "interface UserProfile {\n    id: string;\n    name: string;\n}\n"
        )
        is None
    )
    assert detect_example_language(extractor, "function greet() { return 'hello'; }") is None
    assert detect_example_language(extractor, "int add(int a, int b) { return a + b; }") is None
    assert (
        detect_example_language(extractor, "This is plain prose explaining how something works.")
        is None
    )


def test_example_language_tables_cover_every_advertised_language() -> None:
    """`language=` must accept exactly what the tool schema's LanguageName lists."""
    from typing import get_args

    from code_indexing_mcp.models import LanguageName
    from code_indexing_mcp.scanner import LANGUAGES
    from code_indexing_mcp.search import _DETECTION_ORDER, _NOT_AUTO_DETECTED

    advertised = set(get_args(LanguageName))
    assert set(_EXAMPLE_SUFFIX) == advertised
    # Each representative suffix is one the scanner maps back to that language.
    assert {LANGUAGES[suffix] for suffix in _EXAMPLE_SUFFIX.values()} == advertised
    assert all(LANGUAGES[suffix] == language for language, suffix in _EXAMPLE_SUFFIX.items())
    assert set(_DETECTION_ORDER) | _NOT_AUTO_DETECTED == advertised
    assert len(set(_DETECTION_ORDER)) == len(_DETECTION_ORDER)


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("kotlin", "fun main() { println(1) }\n"),
        ("swift", "struct S { var a: Int }\nfunc g() {}\n"),
        ("zig", 'const std = @import("std");\npub fn main() void {}\n'),
        ("prisma", "model User {\n  id Int @id\n}\n"),
        ("xml", "<a><b>1</b></a>\n"),
    ],
)
def test_example_passages_accept_every_advertised_language(language: str, source: str) -> None:
    detected, passages = _example_passages(TreeSitterExtractor(), source, language=language)

    assert detected == language
    assert passages


def test_auto_detection_covers_newer_languages_without_claiming_prose() -> None:
    extractor = TreeSitterExtractor()

    assert detect_example_language(extractor, "fun main() { println(1) }") == "kotlin"
    assert detect_example_language(extractor, "model User {\n  id Int @id\n}\n") == "prisma"
    assert detect_example_language(extractor, "Note: fix this later\n") is None


def test_example_passages_prose_fallback() -> None:
    extractor = TreeSitterExtractor()
    prose = "Just plain text without code declarations."
    lang, passages = _example_passages(extractor, prose)
    assert lang is None
    assert passages == [prose]


def test_example_suffix_entries_parse_via_extractor() -> None:
    extractor = TreeSitterExtractor()
    for lang, suffix in _EXAMPLE_SUFFIX.items():
        result = extractor.extract(Path(f"example{suffix}"), lang, b"")
        assert result.chunks == []


def test_example_passages_validation_errors() -> None:
    extractor = TreeSitterExtractor()
    with pytest.raises(CodeIndexingError) as empty_err:
        _example_passages(extractor, "   \n  ")
    assert empty_err.value.code == ErrorCode.INVALID_FILTER

    with pytest.raises(CodeIndexingError) as oversized_err:
        _example_passages(extractor, "x" * (MAX_EXAMPLE_LENGTH + 1))
    assert oversized_err.value.code == ErrorCode.INVALID_FILTER
    assert str(MAX_EXAMPLE_LENGTH) in str(oversized_err.value)

    with pytest.raises(CodeIndexingError) as multibyte_err:
        _example_passages(extractor, "é" * (MAX_EXAMPLE_LENGTH // 2 + 1))
    assert multibyte_err.value.code == ErrorCode.INVALID_FILTER
    assert "bytes" in str(multibyte_err.value)

    with pytest.raises(CodeIndexingError) as unsupported_err:
        _example_passages(extractor, "some code", language="unsupported_lang")
    assert unsupported_err.value.code == ErrorCode.UNSUPPORTED_LANGUAGE


def test_search_by_example_exact_chunk_and_renamed_similarity(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)

    # 1. Pasting a function from an indexed file ranks that chunk first
    exact_snippet = "def enforce_permissions(user):\n    return user.is_admin\n"
    res = search.search_by_example(exact_snippet, projects)
    assert res.language == "python"
    assert res.segments == 1
    assert len(res.hits) >= 1
    assert res.hits[0].symbol == "enforce_permissions"
    assert res.hits[0].project_id == projects[0]
    assert pytest.approx(res.hits[0].score, abs=1e-4) == 1.0

    # 2. A structurally similar-but-renamed function outranks unrelated code
    similar_snippet = "def check_permissions(subject):\n    return subject.is_admin\n"
    res_similar = search.search_by_example(similar_snippet, projects)
    assert res_similar.hits[0].symbol == "enforce_permissions"
    if len(res_similar.hits) > 1:
        assert res_similar.hits[0].score > res_similar.hits[1].score


def test_search_by_example_filters_compose(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)
    snippet = "def enforce_permissions(user):\n    return user.is_admin\n"

    # Matching filters
    matched = search.search_by_example(
        snippet,
        projects,
        languages=["python"],
        kinds=["function"],
        paths=["auth.py"],
    )
    assert len(matched.hits) == 1
    assert matched.hits[0].symbol == "enforce_permissions"

    # Conflicting language
    assert (
        search.search_by_example(
            snippet,
            projects,
            languages=["rust"],
        ).hits
        == []
    )

    # Conflicting kind
    assert (
        search.search_by_example(
            snippet,
            projects,
            kinds=["class"],
        ).hits
        == []
    )

    # Conflicting path
    assert (
        search.search_by_example(
            snippet,
            projects,
            paths=["billing/**"],
        ).hits
        == []
    )


def test_search_by_example_limit_clamping(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)
    snippet = "def enforce_permissions(user):\n    return user.is_admin\n"

    # limit <= 0 clamped to 1
    res_min = search.search_by_example(snippet, projects, limit=0)
    assert len(res_min.hits) == 1

    # limit > 50 clamped to 50
    res_max = search.search_by_example(snippet, projects, limit=100)
    assert len(res_max.hits) <= 50


def test_search_by_example_partitions_validation_errors(tmp_path: Path) -> None:
    _, search, projects = indexed_projects(tmp_path)
    snippet = "def enforce_permissions(user):\n    return user.is_admin\n"

    # Empty project_ids
    with pytest.raises(CodeIndexingError) as err:
        search.search_by_example(snippet, [])
    assert err.value.code == ErrorCode.INVALID_FILTER

    # Partitions mapping mismatch
    with pytest.raises(ValueError, match="search partitions do not match the requested projects"):
        search.search_by_example(
            snippet,
            projects,
            partitions={
                projects[0]: PartitionRef(
                    project_id="other", slot_id="slot", partition_id="part", activation_epoch=1
                )
            },
        )


def test_example_passages_are_windowed_by_tokens_when_the_embedder_can(tmp_path: Path) -> None:
    """EMB-2: an over-budget example becomes bounded windows, not one sequence."""
    import numpy as np

    from code_indexing_mcp.embedding import EmbeddedSegment, SegmentPlan
    from code_indexing_mcp.search import _embed_example_passages

    seen: list[SegmentPlan] = []

    class Windowing:
        model_id = "x"
        dimension = 2
        max_input_tokens = 7

        def plan_and_embed(self, candidates, plan):  # type: ignore[no-untyped-def]
            seen.append(plan)
            vector = np.asarray([1.0, 2.0], dtype="<f4").tobytes()
            return [
                [EmbeddedSegment(0, 1, 1, vector), EmbeddedSegment(1, 2, 1, vector)]
                for _ in candidates
            ]

    vectors = _embed_example_passages(Windowing(), ["a", "b"])  # type: ignore[arg-type]

    assert seen[0].max_tokens == 7
    assert vectors == [[1.0, 2.0]] * 4


def test_find_symbol_wildcard_characters_are_literal_without_paging(tmp_path: Path) -> None:
    """LIKE wildcards in a name must not make the prefilter match every row.

    The literal post-filter then rejected all of them and the loop paged through
    the whole project. The names here exceed one page so a regression shows as a
    second storage scan, not just a slow call.
    """
    source = "".join(f"def fn{index}():\n    return {index}\n\n" for index in range(230))
    source += (
        "def _private():\n    return 1\n\ndef a_b():\n    return 2\n\ndef axb():\n    return 3\n"
    )
    search, project = _indexed_tree(tmp_path, {"mod.py": source})

    scans = 0
    original = search.store._projected_chunks

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal scans
        scans += 1
        return original(*args, **kwargs)

    search.store._projected_chunks = counting  # type: ignore[method-assign]
    try:
        assert search.find_symbol("%", project, match="contains").hits == []
        assert scans == 1
        scans = 0
        assert search.find_symbol("zz_", project, match="prefix").hits == []
        assert scans == 1
    finally:
        search.store._projected_chunks = original  # type: ignore[method-assign]

    assert [hit.symbol for hit in search.find_symbol("_", project, match="prefix").hits] == [
        "_private"
    ]
    assert [hit.symbol for hit in search.find_symbol("a_", project, match="contains").hits] == [
        "a_b"
    ]
    assert [hit.symbol for hit in search.find_symbol("a_b", project, match="prefix").hits] == [
        "a_b"
    ]


def test_search_code_holds_the_partition_lock_once_for_hits_and_explanations(
    tmp_path: Path,
) -> None:
    """Hits and their ranking explanations come from one locked view of the partition.

    Re-acquiring the lock for the explanation probes cost a second lock round trip
    and let a commit land between the search and its explanation.
    """
    search, project = _indexed_tree(tmp_path, {"src/a.py": "def alpha_two():\n    return 2\n"})
    acquisitions = 0
    original = search.store.partitions_access

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal acquisitions
        acquisitions += 1
        return original(*args, **kwargs)

    search.store.partitions_access = counting  # type: ignore[method-assign]
    try:
        response = search.search_code("alpha", [project])
    finally:
        search.store.partitions_access = original  # type: ignore[method-assign]

    assert response.hits
    assert response.hits[0].explanation is not None
    assert acquisitions == 1


def test_path_pushdown_adds_the_literal_prefix_conjunct(tmp_path: Path) -> None:
    search, project = _indexed_tree(tmp_path, {"src/pkg7/a.py": "def alpha():\n    return 1\n"})
    conditions: list[str | None] = []
    original = search.store.hybrid_search

    def spy(*args: Any, **kwargs: Any) -> Any:
        conditions.append(args[3])
        return original(*args, **kwargs)

    search.store.hybrid_search = spy  # type: ignore[method-assign]
    try:
        hits = search.search_code("alpha", [project], paths=["src/pkg7/*.py"]).hits
    finally:
        search.store.hybrid_search = original  # type: ignore[method-assign]

    assert [hit.path for hit in hits] == ["src/pkg7/a.py"]
    assert conditions[0] is not None
    assert "starts_with(path, 'src/pkg7/')" in conditions[0]
    assert "regexp_like(path," in conditions[0]
