"""Regression guard for Track 3 (reference query pushdown).

Pins the two things the plan cares about: the storage-layer superset
condition (D1) is provably sound -- `find_references` output is identical
whether or not it actually narrows anything -- and the pushdown machinery is
actually being used rather than merely available, for every caller of
`ReferenceService._find_references_with_records`: `find_references` and
`analyze_refactor`'s own single-lookup fetch, `impact_radius`'s
context-sharing across a whole traversal and its cursor page cache, and
`_select`'s indexed lookup.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_resolver_corpus import CORPUS_ROOT
from test_resolver_corpus import _load_repo as _load_corpus_repo

from code_indexing_mcp import reference_service as reference_service_module
from code_indexing_mcp.errors import CodeIndexingError
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import DeclarationSelector, RenameOperation
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


class TinyEmbedder:
    model_id = "test/pushdown"
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


def _large_fixture() -> dict[str, str]:
    """~50 files, ~800 `reference`-kind rows, only 5 files that can refer to `target`.

    44 "noise" files each define a 20-function call chain (19 calls none of
    which can ever spell `target`); 5 "signal" files import and call it.
    """
    files: dict[str, str] = {"lib.py": "def target():\n    return 1\n"}
    for index in range(5):
        files[f"caller_{index}.py"] = (
            f"from lib import target\n\n\ndef use_{index}():\n    return target()\n"
        )
    for index in range(44):
        functions = [f"def n{index}_0():\n    return 0\n"]
        functions.extend(
            f"def n{index}_{step}():\n    return n{index}_{step - 1}()\n" for step in range(1, 20)
        )
        files[f"noise_{index}.py"] = "\n\n".join(functions) + "\n"
    return files


def _fanout_fixture() -> dict[str, str]:
    """A 2-level fan-out reaching 20+ nodes, all resolved exactly via direct imports."""
    files: dict[str, str] = {"lib.py": "def target():\n    return 1\n"}
    depth1 = [f"d1_{index}" for index in range(6)]
    for name in depth1:
        files[f"{name}.py"] = "from lib import target\n\n\ndef use():\n    return target()\n"
    for index, parent in enumerate(depth1):
        for step in range(3):
            files[f"d2_{index}_{step}.py"] = (
                f"from {parent} import use\n\n\ndef call_use():\n    return use()\n"
            )
    return files


def test_impact_radius_candidate_fetch_pulls_far_fewer_rows_than_the_full_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The storage `where` condition (D1/D2) actually narrows what leaves LanceDB.

    Only 5 of the ~800 `reference`-kind rows in this fixture can ever refer
    to `target` (`_may_refer`'s tail/import-alias checks are the arbiter;
    this only checks how many rows the SQL layer handed to them).
    """
    service, project_id = _indexed_service(tmp_path, _large_fixture())
    total_reference_rows = len(
        service.store.list_reference_records(project_id, record_kinds=("reference",))
    )
    assert total_reference_rows > 500, "fixture is not large enough to be a meaningful guard"

    row_counts: list[int] = []
    real_reference_rows = LanceStore._reference_rows

    def counting_reference_rows(self: LanceStore, *args: object, **kwargs: object) -> list[object]:
        rows = real_reference_rows(self, *args, **kwargs)  # type: ignore[arg-type]
        row_counts.append(len(rows))
        return rows

    monkeypatch.setattr(LanceStore, "_reference_rows", counting_reference_rows)

    result = service.impact_radius(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target"),
        max_depth=1,
    )

    assert len(result.layers[0].edges) == 5
    # Coverage (~50) + import/export (~5) + candidate rows (a handful) --
    # nowhere near the ~800 `reference` rows the unfiltered fetch pulled.
    assert sum(row_counts) < 200


def test_find_references_candidate_fetch_pulls_far_fewer_rows_than_the_full_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1-D3 apply to the plain `find_references` lookup itself, not just `impact_radius`."""
    service, project_id = _indexed_service(tmp_path, _large_fixture())

    row_counts: list[int] = []
    real_reference_rows = LanceStore._reference_rows

    def counting_reference_rows(self: LanceStore, *args: object, **kwargs: object) -> list[object]:
        rows = real_reference_rows(self, *args, **kwargs)  # type: ignore[arg-type]
        row_counts.append(len(rows))
        return rows

    monkeypatch.setattr(LanceStore, "_reference_rows", counting_reference_rows)

    response = service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target")
    )

    # 5 callers, each with an `import` hit and a `call` hit.
    assert len(response.hits) == 10
    assert sum(row_counts) < 200


def test_analyze_refactor_candidate_fetch_pulls_far_fewer_rows_than_the_full_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1-D3 apply to `analyze_refactor`'s reused fetch too, not just `find_references`."""
    service, project_id = _indexed_service(tmp_path, _large_fixture())

    row_counts: list[int] = []
    real_reference_rows = LanceStore._reference_rows

    def counting_reference_rows(self: LanceStore, *args: object, **kwargs: object) -> list[object]:
        rows = real_reference_rows(self, *args, **kwargs)  # type: ignore[arg-type]
        row_counts.append(len(rows))
        return rows

    monkeypatch.setattr(LanceStore, "_reference_rows", counting_reference_rows)

    analysis = service.analyze_refactor(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target"),
        RenameOperation(new_name="renamed"),
    )

    assert len(analysis.must_change) + len(analysis.likely_change) >= 5
    assert sum(row_counts) < 200


def test_impact_radius_loads_the_context_once_per_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4: one `_ReferenceContext` is shared across every frontier node, not reloaded per node."""
    service, project_id = _indexed_service(tmp_path, _fanout_fixture())
    calls: list[int] = []
    real_load_context = ReferenceService._load_reference_context

    def counting_load_context(self: ReferenceService, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return real_load_context(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ReferenceService, "_load_reference_context", counting_load_context)

    result = service.impact_radius(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target"),
        max_depth=3,
        max_nodes=200,
        limit=500,
    )

    assert result.visited >= 20
    assert len(calls) == 1


def test_impact_radius_second_page_reuses_the_cached_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4: a cursor's later page is served from the traversal cache, not re-run."""
    service, project_id = _indexed_service(tmp_path, _fanout_fixture())
    selector = DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target")

    first = service.impact_radius(selector, max_depth=3, max_nodes=200, limit=1)
    assert first.cursor is not None

    calls: list[int] = []
    real_find = ReferenceService._find_references_with_records

    def counting_find(self: ReferenceService, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return real_find(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ReferenceService, "_find_references_with_records", counting_find)

    second = service.impact_radius(
        selector, max_depth=3, max_nodes=200, limit=1, cursor=first.cursor
    )

    assert len(calls) == 0
    assert second.selected == first.selected
    assert second.visited == first.visited


def test_select_by_path_and_symbol_never_calls_list_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5: both the exact-match and "did you mean" branches use indexed lookups."""
    service, project_id = _indexed_service(
        tmp_path, {"lib.py": "def target():\n    return 1\n\n\ndef other_target():\n    return 2\n"}
    )
    calls: list[int] = []
    real_list_chunks = LanceStore.list_chunks

    def counting_list_chunks(self: LanceStore, *args: object, **kwargs: object) -> list[object]:
        calls.append(1)
        return real_list_chunks(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(LanceStore, "list_chunks", counting_list_chunks)

    service.find_references(
        DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="target")
    )

    with pytest.raises(CodeIndexingError):
        service.find_references(
            DeclarationSelector(project=project_id, path="lib.py", qualified_symbol="missing")
        )

    assert len(calls) == 0


def _corpus_declarations(service: ReferenceService, project_id: str) -> list[DeclarationSelector]:
    """Every declaration chunk in an indexed corpus case, as a selector."""
    return [
        DeclarationSelector(
            project=project_id, path=chunk.path, qualified_symbol=chunk.qualified_symbol
        )
        for chunk in service.store.list_chunks([project_id])
        if chunk.symbol is not None and chunk.qualified_symbol is not None
    ]


@pytest.mark.parametrize(
    "case_dir",
    sorted(path for path in CORPUS_ROOT.glob("*/*") if path.is_dir()),
    ids=lambda path: f"{path.parent.name}/{path.name}",
)
def test_pushdown_matches_unfiltered_for_every_resolver_corpus_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case_dir: Path
) -> None:
    """The one principle, pinned: pushdown on vs off must never disagree.

    For every declaration the resolver corpus can select, the classified hit
    list and limitations from `_candidate_records`'s SQL-pushed-down superset
    fetch (D1, `_PUSHDOWN_ENABLED=True`) must equal the same fetch with the
    storage condition disabled (`_PUSHDOWN_ENABLED=False`, falling back to an
    unfiltered `reference`-row fetch) -- proving the condition excludes only
    rows `_may_refer` could never have accepted anyway.
    """
    root = tmp_path / "repo"
    root.mkdir()
    for path, source in _load_corpus_repo(case_dir).items():
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
    service = ReferenceService(store)

    declarations = _corpus_declarations(service, project.id)
    if not declarations:
        # Some corpus cases exist to pin a defect visible only in the raw
        # reference rows (test_resolver_corpus.py's own docstring), with no
        # declaration a selector could ever name -- nothing for this guard
        # to compare in that case.
        pytest.skip(f"{case_dir} has no selectable declaration")

    for selector in declarations:
        selected = service._select(selector)
        partition = store.active_partition(selected.project_id)
        version = store.reference_version(selected.project_id, partition_id=partition.partition_id)
        context = service._load_reference_context(
            selected.project_id, partition_id=partition.partition_id, version=version
        )

        monkeypatch.setattr(reference_service_module, "_PUSHDOWN_ENABLED", True)
        enabled = service._find_references_with_records(
            selector, limit=500, partition=partition, context=context, preselected=selected
        )
        monkeypatch.setattr(reference_service_module, "_PUSHDOWN_ENABLED", False)
        disabled = service._find_references_with_records(
            selector, limit=500, partition=partition, context=context, preselected=selected
        )
        monkeypatch.setattr(reference_service_module, "_PUSHDOWN_ENABLED", True)

        assert enabled.response.hits == disabled.response.hits, selector
        assert enabled.response.limitations == disabled.response.limitations, selector
