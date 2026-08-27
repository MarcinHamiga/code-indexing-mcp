import shutil
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from conftest import run_git
from test_indexing import _remove_reference_generation, _write_with_pinned_mtime

from code_indexing_mcp.accelerator_env import (
    RECORD_FILENAME,
    AcceleratorEnvironment,
    running_python_version,
    write_environment,
)
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.backends import CPU_BACKEND, Accelerator
from code_indexing_mcp.embedding_worker import default_launcher
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.git_state import GitProbeOutcome, GitState, SelectorKind
from code_indexing_mcp.models import (
    DeclarationSelector,
    ProjectInfo,
    ReferenceBackfillReport,
    RenameOperation,
    SearchHit,
    SearchResponse,
)
from code_indexing_mcp.projects import existing_marker_path, initialize_project
from code_indexing_mcp.settings import IndexSettings
from code_indexing_mcp.token_batching import DEFAULT_MAX_TOKEN_PRODUCT, REFERENCE_MEMORY_BYTES
from code_indexing_mcp.worker_launcher import ExternalInterpreterLauncher


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


class OtherModelTinyEmbedder:
    """Same vector dimension as TinyEmbedder but a different model_id, to
    exercise LanceStore's incompatible-model detection."""

    model_id = "test/other-tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def test_concurrent_init_project_registers_one_project(tmp_path: Path) -> None:
    """The daemon serves each client on its own thread; one root, one project id."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    callers = 8
    barrier = threading.Barrier(callers)
    identifiers: list[str] = []
    lock = threading.Lock()

    def initialize() -> None:
        barrier.wait(timeout=10)
        project = app.init_project(root)
        with lock:
            identifiers.append(project.id)

    threads = [threading.Thread(target=initialize) for _ in range(callers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert len(set(identifiers)) == 1
    assert len(app.list_projects()) == 1


def test_application_orchestrates_default_project_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def locate_feature():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )

    project = app.init_project(root)
    assert app.project_status(roots=[root]).state == "pending"
    report = app.index_project(roots=[root])
    status = app.project_status(roots=[root])

    assert status.state == "ready"
    search = app.search_code("locate feature", roots=[root])
    removal = app.remove_project(project.id)

    assert report.project_id == project.id
    assert status.file_count == 1
    assert status.chunk_count >= 1
    assert search.hits[0].symbol == "locate_feature"
    assert removal.removed is True
    assert app.list_projects() == []
    assert (root / ".ci-mcp" / "project.toml").exists()


def test_git_checkout_switches_active_slots_without_leaking_branch_rows(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text("def main_branch():\n    return 1\n")
    run_git("add", "main.py", cwd=root)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "main",
        cwd=root,
    )
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    main_slot = app.project_status(project.id).active_slot_id
    main_chunk_id = app.find_symbol("main_branch", project.id).hits[0].chunk_id

    run_git("checkout", "-qb", "feature", cwd=root)
    (root / "feature.py").write_text("def feature_branch():\n    return 2\n")
    run_git("add", "feature.py", cwd=root)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "feature",
        cwd=root,
    )

    pending = app.project_status(project.id)
    assert pending.active_slot_id != main_slot
    assert pending.branch_build_pending is True
    assert pending.file_count == 0
    assert app.find_symbol("main_branch", project.id).hits == []
    assert app.file_outline("main.py", project.id).items == []
    with pytest.raises(CodeIndexingError) as excinfo:
        app.get_chunk(main_chunk_id)
    assert excinfo.value.code is ErrorCode.CHUNK_NOT_FOUND
    app.index_project(project.id)
    feature = app.search_code("feature branch", projects=[project.id], paths=["feature.py"])
    assert [hit.symbol for hit in feature.hits] == ["feature_branch"]
    feature_chunk_id = feature.hits[0].chunk_id

    run_git("checkout", "-q", "main", cwd=root)
    restored = app.project_status(project.id)
    assert restored.active_slot_id == main_slot
    assert restored.file_count == 1
    assert restored.branch_build_pending is False
    assert app.search_code("feature branch", projects=[project.id], paths=["feature.py"]).hits == []
    with pytest.raises(CodeIndexingError) as excinfo:
        app.get_chunk(feature_chunk_id)
    assert excinfo.value.code is ErrorCode.CHUNK_NOT_FOUND


def test_degraded_git_probe_activates_an_empty_workspace_slot(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text("def main_branch():\n    return 1\n")
    run_git("add", "main.py", cwd=root)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "main",
        cwd=root,
    )
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    branch_slot = app.project_status(project.id).active_slot_id
    degraded = GitState(
        probe=GitProbeOutcome.TIMEOUT,
        selector_kind=SelectorKind.WORKSPACE,
        selector_value=str(root.resolve()),
    )

    with patch("code_indexing_mcp.application.probe_git_state", return_value=degraded):
        status = app.project_status(project.id)

    assert status.active_slot_id != branch_slot
    assert status.git_probe == GitProbeOutcome.TIMEOUT.value
    assert status.file_count == 0
    assert status.branch_build_pending is True
    assert branch_slot in {slot.slot_id for slot in app.store.list_slots(project.id)}


def test_search_retries_once_when_the_repository_changes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    first = SearchResponse(query="answer", hits=[])
    second = SearchResponse(query="answer after switch", hits=[])

    with (
        patch.object(app.search, "search_code", side_effect=[first, second]) as search_code,
        patch.object(app, "_target_changed", side_effect=[True, False]),
    ):
        response = app.search_code("answer", projects=[project.id])

    assert response is second
    assert search_code.call_count == 2


def test_search_rejects_a_second_repository_transition(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    response = SearchResponse(query="answer", hits=[])

    with (
        patch.object(app.search, "search_code", return_value=response),
        patch.object(app, "_target_changed", side_effect=[True, True]),
        pytest.raises(CodeIndexingError) as excinfo,
    ):
        app.search_code("answer", projects=[project.id])

    assert excinfo.value.code is ErrorCode.REPOSITORY_CHANGED


def test_application_can_ensure_the_structural_index_without_a_semantic_search(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    report = app.ensure_reference_index(project.id)

    assert report.complete is True
    assert report.files_current == 1


def test_reference_query_reports_an_incomplete_structural_index(tmp_path: Path) -> None:
    """An uncoverable file degrades the answer instead of disabling the tool.

    Refusing outright meant one unparseable file anywhere in a repository made
    both reference tools permanently unusable, because such a file never gains
    coverage and so fails the same way on every later call.
    """

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    with patch.object(
        app.indexer,
        "backfill_references",
        return_value=ReferenceBackfillReport(project_id=project.id, incomplete_paths=["broken.py"]),
    ):
        response = app.find_references(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            )
        )
        analysis = app.analyze_refactor(
            DeclarationSelector(
                project=project.id,
                path="main.py",
                qualified_symbol="answer",
            ),
            RenameOperation(new_name="result"),
        )

    limitation = next(item for item in response.limitations if item.code == "parse_error")
    assert "broken.py" in limitation.explanation
    assert analysis.completeness.state == "incomplete"


def test_reference_queries_prepare_selectors_once(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    chunk_id = app.find_symbol("answer", project.id).hits[0].chunk_id
    selectors = [
        DeclarationSelector(chunk_id=chunk_id),
        DeclarationSelector(
            project=str(root),
            path="main.py",
            qualified_symbol="answer",
        ),
    ]

    with patch.object(
        app, "ensure_reference_index", wraps=app.ensure_reference_index
    ) as ensure_reference_index:
        for selector in selectors:
            response = app.find_references(selector)
            analysis = app.analyze_refactor(selector, RenameOperation(new_name="result"))

            assert response.selected.qualified_symbol == "answer"
            assert analysis.selected.qualified_symbol == "answer"
    assert ensure_reference_index.call_count == 4
    assert {call.args[0] for call in ensure_reference_index.call_args_list} == {project.id}


def test_modified_source_marks_an_index_stale(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    assert app.project_is_stale(project.id) is False

    source.write_text("value = 200\n")

    assert app.project_is_stale(project.id) is True
    assert app.project_status(project.id).state == "stale"


def test_created_source_marks_an_index_stale(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    (root / "added.py").write_text("added = True\n")

    assert app.project_is_stale(project.id) is True


def test_deleted_source_marks_an_index_stale(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    source.unlink()

    assert app.project_is_stale(project.id) is True


def test_a_rejected_file_does_not_make_the_project_permanently_stale(tmp_path: Path) -> None:
    """A NUL-byte file used to vanish from storage while the scanner kept

    yielding it, so `current.keys() != existing.keys()` was true forever and
    every reference query triggered a full re-index under the global lock
    (S3). The file must instead persist as a tombstone row, so freshness
    settles once its size/mtime stop changing.
    """

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "garbage.py").write_bytes(b"def broken(\x00):\n    pass\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    assert app.project_is_stale(project.id) is False
    assert app.project_status(project.id).state in {"ready", "partial"}

    with patch.object(app.indexer, "index", wraps=app.indexer.index) as index_spy:
        first = app.ensure_reference_index(project.id)
        second = app.ensure_reference_index(project.id)

    assert index_spy.call_count == 0
    assert first.files_current == second.files_current


def test_init_project_defaults_to_the_single_client_root(tmp_path: Path) -> None:
    root = tmp_path / "client-root"
    root.mkdir()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )

    project = app.init_project(roots=[root])

    assert project.root == root.resolve()


def _overlap_application(tmp_path: Path) -> Application:
    return Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )


def test_init_project_rejects_a_root_nested_inside_a_registered_project(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)
    app = _overlap_application(tmp_path)
    parent = app.init_project(root)

    with pytest.raises(CodeIndexingError) as raised:
        app.init_project(nested)

    assert raised.value.code is ErrorCode.OVERLAPPING_PROJECT
    assert app.list_projects() == [parent]


def test_init_project_rejecting_an_overlap_writes_no_marker(tmp_path: Path) -> None:
    """A rejected registration must leave nothing for discovery to pick up.

    If the overlap check ran after marker creation, the orphaned marker would
    let a later discover_project call register exactly the registration the
    user just rejected.
    """
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)
    app = _overlap_application(tmp_path)
    app.init_project(root)

    with pytest.raises(CodeIndexingError):
        app.init_project(nested)

    assert existing_marker_path(nested) is None
    # The rejected root still initializes cleanly once overlap is allowed.
    child = app.init_project(nested, allow_overlap=True)
    assert existing_marker_path(nested) is not None
    assert child.root == nested.resolve()


def test_init_project_rejects_a_root_containing_a_registered_project(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)
    app = _overlap_application(tmp_path)
    child = app.init_project(nested)

    with pytest.raises(CodeIndexingError) as raised:
        app.init_project(root)

    assert raised.value.code is ErrorCode.OVERLAPPING_PROJECT
    assert app.list_projects() == [child]


def test_init_project_allows_a_nested_registration_when_allow_overlap_is_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)
    app = _overlap_application(tmp_path)
    parent = app.init_project(root)

    child = app.init_project(nested, allow_overlap=True)

    assert {project.id for project in app.list_projects()} == {parent.id, child.id}
    warnings = app.storage_status().overlap_warnings
    assert len(warnings) == 1
    assert "contains the root" in warnings[0] or "nested inside" in warnings[0]


def test_reinitializing_the_same_root_keeps_one_registration(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    app = _overlap_application(tmp_path)

    project = app.init_project(root)
    again = app.init_project(root)

    assert again.id == project.id
    assert app.list_projects() == [project]


def test_force_new_id_replaces_a_registration_without_allow_overlap(tmp_path: Path) -> None:
    """force_new_id is itself explicit intent to replace this directory's project."""
    root = tmp_path / "repo"
    root.mkdir()
    app = _overlap_application(tmp_path)

    original = app.init_project(root)
    replacement = app.init_project(root, force_new_id=True)

    assert replacement.id != original.id
    assert replacement.root == original.root


def test_discovery_keeps_registering_roots_that_overlap(tmp_path: Path) -> None:
    """Automatic MCP-root discovery has no allow_overlap flag, so it registers.

    Existing overlapping registrations must remain usable; the resulting
    overlap is advisory and surfaced through storage_status warnings.
    """
    root = tmp_path / "repo"
    nested = root / "nested"
    nested.mkdir(parents=True)
    for directory in (root, nested):
        (directory / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        (directory / "main.py").write_text("value = 1\n")
    app = _overlap_application(tmp_path)

    # Discovering the nested root first registers it while no marker exists
    # above it; the parent root has no marker of its own, so it registers
    # separately and the overlap stands.
    first = app.discover_project(nested)
    second = app.discover_project(root)

    assert first is not None
    assert second is not None
    assert len(app.list_projects()) == 2
    warnings = app.storage_status().overlap_warnings
    assert len(warnings) == 1
    assert "contains the root" in warnings[0] or "nested inside" in warnings[0]


def test_case_insensitive_root_alias_is_one_registration_and_one_lock(
    tmp_path: Path, case_insensitive_path_alias: Callable[[Path], Path]
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    alias = case_insensitive_path_alias(root)

    from_alias = app.init_project(alias)
    from_duplicate_roots = app.init_project(roots=[root, alias])

    assert from_alias.id == project.id
    assert from_duplicate_roots.id == project.id
    assert app._root_lock(root).lock_file == app._root_lock(alias).lock_file
    assert app.list_projects() == [project]


def test_discover_project_requires_marker_and_supported_source(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    source_only = tmp_path / "source-only"
    source_only.mkdir()
    (source_only / "main.py").write_text("value = 1\n")

    assert app.discover_project(source_only) is None
    assert not (source_only / ".ci-mcp").exists()

    (source_only / "pyproject.toml").write_text("[project]\nname = 'source-only'\n")

    project = app.discover_project(source_only)

    assert project is not None
    assert project.root == source_only.resolve()
    assert app.project_status(project.id).state == "pending"
    assert (source_only / ".ci-mcp" / "project.toml").exists()


def test_discover_project_accepts_javascript_manifest_and_existing_marker(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    javascript = tmp_path / "javascript"
    javascript.mkdir()
    (javascript / "package.json").write_text('{"name": "javascript"}\n')
    (javascript / "main.ts").write_text("export const value = 1\n")

    project = app.discover_project(javascript)

    assert project is not None
    empty = tmp_path / "empty"
    empty.mkdir()
    existing = app.init_project(empty)

    assert app.discover_project(empty) == existing


def test_application_supports_explicit_cross_project_search(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    ids = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "main.py").write_text(f"def {name}_feature():\n    return True\n")
        project = app.init_project(root)
        app.index_project(project.id)
        ids.append(project.id)

    selected = app.search_code("feature", projects=ids)
    all_projects = app.search_code("feature", all_projects=True)

    assert {hit.project_id for hit in selected.hits} == set(ids)
    assert {hit.project_id for hit in all_projects.hits} == set(ids)


def test_duplicate_live_project_marker_is_rejected_but_moved_checkout_is_adopted(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    duplicate = tmp_path / "duplicate"
    original.mkdir()
    (original / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(original)
    shutil.copytree(original, duplicate)

    with pytest.raises(CodeIndexingError) as raised:
        app.index_project(str(duplicate))
    assert raised.value.code is ErrorCode.PROJECT_ID_CONFLICT

    shutil.rmtree(original)
    report = app.index_project(str(duplicate))
    assert report.project_id == project.id
    assert app.list_projects()[0].root == duplicate.resolve()


def test_incompatible_duplicate_project_marker_is_rejected_before_rebuild(tmp_path: Path) -> None:
    original = tmp_path / "original"
    duplicate = tmp_path / "duplicate"
    original.mkdir()
    (original / "main.py").write_text("def answer():\n    return 42\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    first = Application(paths, embedder=TinyEmbedder(), cwd=original)
    project = first.init_project(original)
    first.index_project(project.id)
    shutil.copytree(original, duplicate)

    conflicting = Application(paths, embedder=OtherModelTinyEmbedder(), cwd=duplicate)

    with pytest.raises(CodeIndexingError) as raised:
        conflicting.init_project(duplicate)
    assert raised.value.code is ErrorCode.PROJECT_ID_CONFLICT
    assert first.search_code("answer", projects=[project.id]).hits


def test_duplicate_legacy_project_marker_is_still_rejected(tmp_path: Path) -> None:
    original = tmp_path / "original"
    duplicate = tmp_path / "duplicate"
    original.mkdir()
    (original / "main.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    app.init_project(original)
    (original / ".ci-mcp").rename(original / ".code-indexing-mcp")
    shutil.copytree(original, duplicate)

    with pytest.raises(CodeIndexingError) as raised:
        app.index_project(str(duplicate))

    assert raised.value.code is ErrorCode.PROJECT_ID_CONFLICT


def test_reregistering_a_known_project_preserves_state_and_still_validates_compatibility(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def locate_feature():\n    return True\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = app.init_project(root)
    app.index_project(project.id)
    assert app.project_status(project.id).state == "ready"

    # Re-initializing (or re-discovering) an already-known, ready project
    # must not reset its state back to "pending".
    app.init_project(root)
    assert app.project_status(project.id).state == "ready"
    app.discover_project(root)
    assert app.project_status(project.id).state == "ready"

    # A different embedding model makes the stored generation incompatible.
    # That is reconstructable, so registration marks the project for rebuild
    # instead of failing, and the next index run rebuilds it back to ready.
    other_app = Application(paths, embedder=OtherModelTinyEmbedder(), cwd=root)
    other_project = other_app.init_project(root)
    assert other_project.id == project.id
    assert other_app.project_status(project.id).state == "rebuild_required"

    rebuilt = other_app.index_project(project.id)
    assert rebuilt.indexed_files >= 1
    assert other_app.project_status(project.id).state == "ready"


def test_query_rebuilds_an_incompatible_project_before_serving_results(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    original = Application(paths, embedder=TinyEmbedder(), cwd=root)
    project = original.init_project(root)
    original.index_project(project.id)

    replacement = Application(paths, embedder=OtherModelTinyEmbedder(), cwd=root)
    replacement.init_project(root)
    assert replacement.project_status(project.id).state == "rebuild_required"

    response = replacement.search_code("answer", projects=[project.id])

    assert response.hits
    assert replacement.project_status(project.id).state == "ready"


def test_the_application_resolves_a_backend_and_reports_it(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(IndexSettings.from_environment({}), embedding_accelerator=Accelerator.CPU)

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    status = app.model_status()
    assert app.backend_selection.accelerator is Accelerator.CPU
    assert status.embedding_model == "test/tiny"
    assert status.batch_calibration == "default"
    assert status.probe_cache_state == "not-applicable"


def test_an_explicit_batch_size_is_not_overridden_by_calibration(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}),
        embedding_batch_size=12,
        embedding_batch_auto=False,
    )

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    assert app.model_status().batch_size == 12
    assert app.model_status().batch_calibration == "explicit"


def test_the_microbatch_token_budget_follows_the_memory_ceiling(tmp_path: Path) -> None:
    """The padded matrix a microbatch materializes is charged to the same
    ceiling the operator configured, so it has to move with it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}), index_memory_bytes=REFERENCE_MEMORY_BYTES * 2
    )

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    assert app.indexer.segment_plan.max_token_product == DEFAULT_MAX_TOKEN_PRODUCT * 2


def test_the_query_model_stays_in_process_regardless_of_the_backend(tmp_path: Path) -> None:
    """Acceleration targets passage indexing; a query must never wait on it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    embedder = TinyEmbedder()

    app = Application(paths, embedder=embedder, cwd=tmp_path)

    assert app.search.embedder is embedder
    assert app.embedder is embedder


def test_a_backend_that_failed_once_is_not_attempted_again(tmp_path: Path) -> None:
    """Only successful probes are cached, so the failure has to be remembered.

    Without this the daemon would spawn a worker, load the model onto a dead
    device, and terminate it again before every single index run.
    """
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(IndexSettings.from_environment({}), embedding_accelerator=Accelerator.CPU)
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)
    resolved = app.backend_selection
    assert app.effective_backend_selection is resolved

    app._remember_fallback(resolved.fell_back_to(CPU_BACKEND, "the device fell off the bus"))

    assert app.effective_backend_selection is not resolved
    assert app.effective_backend_selection.accelerator is Accelerator.CPU
    # backend_selection still records what capability alone resolved to; only
    # the effective selection carries the verdict a real run reached.
    assert app.backend_selection is resolved


def test_model_status_reports_a_runtime_fallback_rather_than_the_original_choice(
    tmp_path: Path,
) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    app._remember_fallback(
        app.backend_selection.fell_back_to(CPU_BACKEND, "the accelerator died on load")
    )

    status = app.model_status()
    assert status.resolved_accelerator == "cpu"
    assert status.fallback_reason == "the accelerator died on load"


def _prepared_cuda_environment(tmp_path: Path, **overrides: object) -> Path:
    """Write the record an installer leaves behind for a prepared CUDA machine."""
    interpreter = tmp_path / "venv-accel" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")
    settings: dict[str, object] = {
        "accelerator": Accelerator.CUDA,
        "interpreter": interpreter,
        "providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        "runtime_version": "1.23.2",
        "driver_version": "550.54.14",
        "device": "cuda:0",
        "python_version": running_python_version(),
    }
    settings.update(overrides)
    write_environment(
        tmp_path / "data" / RECORD_FILENAME,
        AcceleratorEnvironment(**settings),  # type: ignore[arg-type]
    )
    return interpreter


def test_auto_selects_a_prepared_accelerator_this_process_cannot_execute_itself(
    tmp_path: Path,
) -> None:
    """The serving environment is CPU-only; the record is what makes CUDA reachable."""
    interpreter = _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    assert app.backend_selection.accelerator is Accelerator.CUDA
    status = app.model_status()
    assert status.resolved_accelerator == "cuda"
    assert status.accelerator_environment == str(interpreter)
    assert status.accelerator_prepared == "cuda"
    # Diagnostics the probe cache is keyed on come from the environment that was
    # probed, not from this process's own CPU runtime.
    assert status.driver_version == "550.54.14"
    assert status.runtime_version == "1.23.2"
    assert status.device == "cuda:0"


def _measure(app: Application, *, cpu: float, accelerator: float, load_ns: int) -> None:
    """Record the calibration a first run would have left behind."""
    app.probe_cache.store(
        app._build_probe_key(app.embedder),
        batch_size=8,
        dimension=app.embedder.dimension,
        characters_per_second=accelerator,
        load_ns=load_ns,
    )
    app.probe_cache.store(
        app._cpu_probe_key(),
        batch_size=1,
        dimension=app.embedder.dimension,
        characters_per_second=cpu,
        load_ns=0,
    )


def test_the_crossover_is_computed_from_both_measured_backends(tmp_path: Path) -> None:
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    _measure(app, cpu=1_000.0, accelerator=2_000.0, load_ns=2_000_000_000)

    status = app.model_status()
    assert app.crossover_characters() == 4_000
    assert status.crossover_characters == 4_000
    assert status.accelerator_characters_per_second == 2_000.0
    assert status.cpu_characters_per_second == 1_000.0
    assert status.accelerator_load_ms == 2_000
    # The measured batch size is adopted by the next process to start, which is
    # where the segment plan for a run is built.
    assert (
        Application(paths, embedder=TinyEmbedder(), cwd=tmp_path).model_status().batch_calibration
        == "measured"
    )


def test_an_unmeasured_accelerator_has_no_crossover_and_starts_at_once(
    tmp_path: Path,
) -> None:
    """Nothing has been measured, so there is no size to defer below and the
    accelerator is used exactly as it was before any of this existed."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    assert app.crossover_characters() == 0
    assert app.model_status().crossover_characters is None


def test_an_explicit_crossover_wins_over_the_measured_one(tmp_path: Path) -> None:
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}),
        embedding_crossover_characters=99,
        embedding_crossover_auto=False,
    )
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    _measure(app, cpu=1_000.0, accelerator=2_000.0, load_ns=2_000_000_000)

    assert app.crossover_characters() == 99


def test_strict_mode_refuses_to_defer_to_cpu_at_all(tmp_path: Path) -> None:
    """Strict mode is for a caller who would rather fail than index quietly on
    CPU, and a deferral is quiet CPU indexing no degradation reports."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(IndexSettings.from_environment({}), embedding_strict=True)
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    _measure(app, cpu=1_000.0, accelerator=2_000.0, load_ns=2_000_000_000)

    # The measurement still stands and is still reported; only the deferral it
    # would otherwise drive is refused.
    assert app.model_status().crossover_characters == 4_000
    assert app.crossover_characters() == 0


def test_an_accelerator_that_lost_to_cpu_recommends_the_override(tmp_path: Path) -> None:
    """There is no run size at which it wins, so the useful thing to report is
    not a threshold but that this machine should stop preparing it."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    _measure(app, cpu=2_000.0, accelerator=1_000.0, load_ns=2_000_000_000)

    status = app.model_status()
    assert status.crossover_characters is None
    assert status.recommended_override is not None
    assert "CODE_INDEXING_EMBED_ACCELERATOR=cpu" in status.recommended_override
    # The same None reaches the session, which is what stops it starting a
    # backend no run is large enough to justify. Reporting the largest
    # admissible run instead would name a threshold and defer against it.
    assert app.crossover_characters() is None


def test_a_batch_size_a_ceiling_overrun_reduced_is_reported_as_reduced(
    tmp_path: Path,
) -> None:
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    app.probe_cache.store(
        app._build_probe_key(app.embedder),
        batch_size=1,
        dimension=app.embedder.dimension,
        characters_per_second=500.0,
        load_ns=1,
        limited_by="memory",
    )

    rebuilt = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    status = rebuilt.model_status()
    assert status.batch_size == 1
    assert status.batch_calibration == "reduced"
    assert status.recommended_override is not None
    assert "CODE_INDEXING_EMBED_MEMORY_MB" in status.recommended_override


def test_a_prepared_accelerator_runs_in_its_own_interpreter(tmp_path: Path) -> None:
    interpreter = _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    launcher = app._accelerator_launcher(app.backend_selection.descriptor)

    assert isinstance(launcher, ExternalInterpreterLauncher)
    assert launcher.executable == interpreter
    # The fallback is what a failed accelerator falls back *to*, so it must not
    # depend on the environment that just failed.
    assert not isinstance(default_launcher(), ExternalInterpreterLauncher)


def test_a_backend_this_process_already_offers_needs_no_second_environment(
    tmp_path: Path,
) -> None:
    """An explicit Core ML override runs in the serving environment's own runtime."""
    _prepared_cuda_environment(tmp_path)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    in_process = replace(app.backend_selection.descriptor, provider=app.serving_providers[0])

    assert not isinstance(app._accelerator_launcher(in_process), ExternalInterpreterLauncher)


def test_a_refused_record_explains_the_cpu_outcome(tmp_path: Path) -> None:
    """ "No accelerator is prepared" is the wrong diagnosis when one nearly was."""
    _prepared_cuda_environment(tmp_path, python_version="3.7")
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)

    assert app.backend_selection.accelerator is Accelerator.CPU
    status = app.model_status()
    assert "built for Python 3.7" in (status.fallback_reason or "")
    assert status.accelerator_environment is None
    assert status.accelerator_prepared is None


def test_a_record_offers_only_the_accelerator_it_was_probed_for(tmp_path: Path) -> None:
    """A provider the prepared runtime happens to ship is not evidence of anything.

    A CUDA environment's ONNX Runtime lists more providers than CUDA. Offering
    all of them would select a backend no probe ever exercised there, and then
    describe it with a record that cannot say which device or driver it ran on.
    """
    _prepared_cuda_environment(
        tmp_path, providers=("CUDAExecutionProvider", "MIGraphXExecutionProvider")
    )
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    settings = replace(
        IndexSettings.from_environment({}), embedding_accelerator=Accelerator.MIGRAPHX
    )

    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path, settings=settings)

    assert app.backend_selection.accelerator is Accelerator.CPU
    assert app.backend_selection.honored is False
    assert "not among the execution providers" in (app.model_status().fallback_reason or "")
    # The record's own accelerator is still reachable; only the rest is not.
    assert Application(
        paths, embedder=TinyEmbedder(), cwd=tmp_path
    ).backend_selection.accelerator is (Accelerator.CUDA)


def test_storage_status_reports_registry_project_and_totals(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def locate_feature():\n    return True\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    status = app.storage_status()

    assert status.schema_version == 2
    assert status.consistent is True
    assert status.overlap_warnings == []
    assert status.worktree_warnings == []
    assert status.registry.name == "projects"
    assert status.registry.row_count == 1
    assert status.registry.logical_bytes > 0
    assert len(status.projects) == 1
    project_stats = status.projects[0]
    assert project_stats.project.id == project.id
    assert {table.name for table in project_stats.tables} == {"files", "chunks", "references"}
    assert project_stats.partition_physical_bytes > 0
    assert project_stats.consistent is True
    assert status.physical_bytes_total >= (
        status.registry.physical_bytes + project_stats.partition_physical_bytes
    )

    scoped = app.storage_status(project.id)

    assert [entry.project.id for entry in scoped.projects] == [project.id]

    active = app.store.active_slot(project.id)
    assert active is not None
    alternate = active.model_copy(
        update={"slot_id": "slot-alternate", "partition_id": "partition-alternate"}
    )
    app.store.upsert_slot(alternate)
    app.store._tables(alternate.partition_id)

    with_alternate = app.storage_status()
    with_alternate_stats = with_alternate.projects[0]
    assert {slot.slot_id for slot in with_alternate_stats.slots} == {
        active.slot_id,
        "slot-alternate",
    }
    assert with_alternate.physical_bytes_total == (
        with_alternate.registry.physical_bytes
        + sum(slot.physical_bytes for slot in with_alternate_stats.slots)
    )


def test_storage_status_reports_registered_root_overlaps(tmp_path: Path) -> None:
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app = Application(paths, embedder=TinyEmbedder(), cwd=tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    app.init_project(first)
    nested = first / "nested"
    nested.mkdir()
    app.store.upsert_project(
        ProjectInfo(id="nested-id", name="nested", root=nested), model_id="test/model"
    )

    status = app.storage_status()

    assert any("nested" in warning for warning in status.overlap_warnings)
    assert len(status.projects) == 2


def test_storage_status_raises_for_an_unknown_project(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )

    with pytest.raises(CodeIndexingError) as raised:
        app.storage_status("no-such-project")

    assert raised.value.code is ErrorCode.PROJECT_NOT_FOUND


def _indexed_app(tmp_path: Path, name: str = "repo") -> tuple[Application, ProjectInfo]:
    root = tmp_path / name
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    return app, project


def test_maintenance_preserves_data_through_the_full_application_path(tmp_path: Path) -> None:
    app, project = _indexed_app(tmp_path)
    (tmp_path / "repo" / "main.py").write_text("def renamed_answer():\n    return 43\n")
    app.index_project(project.id)
    before_search = app.search_code("answer", projects=[project.id])

    report = app.maintain_storage(wait_for_lock=True)

    assert report.trigger == "manual"
    assert report.dry_run is False
    assert report.duration_ms >= 0
    entry = next(result for result in report.projects if result.project.id == project.id)
    assert entry.status == "ok"
    assert entry.after is not None
    assert entry.versions_removed >= 0
    assert entry.bytes_reclaimed >= 0
    assert entry.reclaimable_bytes_estimate >= 0
    assert report.registry_after is not None
    assert report.registry_status == "ok"

    # Maintenance must not change what searches return. Relevance scores can
    # shift fractionally when optimize merges the FTS index, so compare the
    # hit identities, not the scores.
    def identity(hits: list[SearchHit]) -> list[tuple[str | None, str, int, int]]:
        return [(hit.symbol, hit.path, hit.start_line, hit.end_line) for hit in hits]

    after_search = app.search_code("answer", projects=[project.id])
    assert identity(after_search.hits) == identity(before_search.hits)
    assert app.project_status(project.id).state == "ready"


def test_maintenance_dry_run_mutates_nothing(tmp_path: Path) -> None:
    from unittest.mock import patch as mock_patch

    from lancedb.table import LanceTable

    app, project = _indexed_app(tmp_path)
    optimized: list[str] = []
    original_optimize = LanceTable.optimize

    def counting_optimize(self: LanceTable, **kwargs: object) -> None:
        optimized.append(self.name)
        original_optimize(self, **kwargs)

    with mock_patch.object(LanceTable, "optimize", counting_optimize):
        report = app.maintain_storage(dry_run=True)

    assert report.dry_run is True
    assert optimized == []
    entry = next(result for result in report.projects if result.project.id == project.id)
    assert entry.status == "skipped"
    assert entry.skip_reason == "dry-run"
    assert entry.before is not None
    assert entry.after is None
    assert entry.reclaimable_bytes_estimate >= 0
    assert report.registry_status == "skipped"
    assert report.registry_skip_reason == "dry-run"
    assert app.project_status(project.id).state == "ready"


def test_maintenance_preserves_versions_named_by_a_recovery_journal(tmp_path: Path) -> None:
    import json

    from code_indexing_mcp.staging import (
        JOURNAL_FORMAT_VERSION,
        JOURNAL_NAME,
        PHASE_COMMITTING,
        recover_staged_commits,
    )

    app, project = _indexed_app(tmp_path)
    partition = app.store.active_partition(project.id)
    versions = app.store.table_versions(project.id)
    directory = app.paths.data / "staging" / project.id / partition.slot_id / "job-1"
    directory.mkdir(parents=True)
    (directory / JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": JOURNAL_FORMAT_VERSION,
                "job_id": directory.name,
                "project_id": project.id,
                "slot_id": partition.slot_id,
                "partition_id": partition.partition_id,
                "activation_epoch": partition.activation_epoch,
                "phase": PHASE_COMMITTING,
                "files_version": versions.files,
                "chunks_version": versions.chunks,
                "references_version": versions.references,
                "replace_file_ids": [],
                "replace_reference_file_ids": [],
                "removed_file_ids": [],
            }
        )
    )

    with patch.object(
        app.store,
        "maintain_project",
        wraps=app.store.maintain_project,
    ) as maintain_project:
        report = app.maintain_storage(wait_for_lock=True)

    entry = next(result for result in report.projects if result.project.id == project.id)
    assert entry.status == "ok"
    assert maintain_project.call_args.kwargs["protected_slot_ids"] == frozenset({partition.slot_id})
    assert recover_staged_commits(app.paths.data / "staging", app.store) == 1


def test_busy_projects_are_skipped_in_automatic_maintenance(tmp_path: Path) -> None:
    from filelock import FileLock

    app, first = _indexed_app(tmp_path, "first")
    _, second = _indexed_app(tmp_path, "second")
    competing = FileLock(app.paths.data / "locks" / f"{first.id}.lock")
    competing.acquire()
    try:
        report = app.maintain_storage(wait_for_lock=False)
    finally:
        competing.release()

    assert first.id in report.busy_projects
    by_id = {result.project.id: result for result in report.projects}
    assert by_id[first.id].status == "skipped"
    assert by_id[first.id].skip_reason == "busy"
    assert by_id[second.id].status == "ok"
    # The registry was reachable even though one project was busy.
    assert report.registry_after is not None
    assert report.registry_status == "ok"


def test_busy_automatic_maintenance_does_not_walk_partition_storage(tmp_path: Path) -> None:
    from filelock import FileLock

    from code_indexing_mcp import storage as storage_module

    app, project = _indexed_app(tmp_path)
    competing = FileLock(app.paths.data / "locks" / "index-global.lock")
    with (
        competing,
        patch.object(
            storage_module,
            "_directory_bytes",
            wraps=storage_module._directory_bytes,
        ) as walk,
    ):
        report = app.maintain_storage(wait_for_lock=False)

    assert project.id in report.busy_projects
    assert report.registry_status == "skipped"
    assert report.registry_skip_reason == "busy"
    assert walk.call_count == 0


def test_maintenance_releases_the_global_lock_when_a_project_is_busy(tmp_path: Path) -> None:
    """A busy project must not leave the global writer lock held behind it.

    The scheduled pass acquires the global lock before the per-project lock; if
    the project is busy the global lock is released before skipping, so later
    maintenance and indexing in this process can re-acquire it immediately.
    (CPython's refcounting masks a leaked FileLock at function exit, which is
    why the probe and the follow-up pass assert the observable contract.)
    """
    from filelock import FileLock

    app, project = _indexed_app(tmp_path)
    competing = FileLock(app.paths.data / "locks" / f"{project.id}.lock")
    competing.acquire()
    try:
        report = app.maintain_storage(wait_for_lock=False)
    finally:
        competing.release()

    assert project.id in report.busy_projects
    probe = FileLock(app.paths.data / "locks" / "index-global.lock")
    assert probe.acquire(timeout=0)
    probe.release()

    rerun = app.maintain_storage(wait_for_lock=False)
    by_id = {result.project.id: result for result in rerun.projects}
    assert by_id[project.id].status == "ok"
    assert rerun.registry_after is not None


def test_maybe_run_maintenance_retries_an_all_busy_pass(tmp_path: Path) -> None:
    """A pass that maintained nothing must leave the timestamp stale for retry."""
    from filelock import FileLock

    app, project = _indexed_app(tmp_path)
    competing = FileLock(app.paths.data / "locks" / "index-global.lock")
    competing.acquire()
    try:
        report = app.maybe_run_maintenance()
    finally:
        competing.release()

    assert report is not None
    assert project.id in report.busy_projects
    timestamp_path = app.paths.data / "maintenance.json"
    assert not timestamp_path.exists()
    # Still due: the next startup retries instead of waiting out 24 hours.
    retried = app.maybe_run_maintenance()
    assert retried is not None
    assert any(result.status == "ok" for result in retried.projects)
    assert timestamp_path.exists()


def test_maybe_run_maintenance_retries_a_partially_busy_pass(tmp_path: Path) -> None:
    from filelock import FileLock

    app, first = _indexed_app(tmp_path, "first")
    _indexed_app(tmp_path, "second")
    competing = FileLock(app.paths.data / "locks" / f"{first.id}.lock")
    with competing:
        report = app.maybe_run_maintenance()

    assert report is not None
    assert first.id in report.busy_projects
    assert any(result.status == "ok" for result in report.projects)
    assert report.registry_status == "ok"
    assert not (app.paths.data / "maintenance.json").exists()


class _ExplodingEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("maintenance must never embed passages")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("maintenance must never embed a query")


def test_automatic_maintenance_never_loads_the_embedding_model(tmp_path: Path) -> None:
    app, _ = _indexed_app(tmp_path)
    app.embedder = _ExplodingEmbedder()

    report = app.maybe_run_maintenance()

    assert report is not None
    assert report.trigger == "scheduled"
    assert any(result.status == "ok" for result in report.projects)


def test_maybe_run_maintenance_is_disabled_by_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODE_INDEXING_AUTO_MAINTENANCE", "0")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )

    assert app.maybe_run_maintenance() is None
    assert not (app.paths.data / "maintenance.json").exists()


def test_maybe_run_maintenance_respects_the_24h_cadence_and_persists(
    tmp_path: Path,
) -> None:
    import json
    from datetime import UTC, datetime, timedelta

    app, _ = _indexed_app(tmp_path)
    timestamp_path = app.paths.data / "maintenance.json"

    report = app.maybe_run_maintenance()

    assert report is not None and report.trigger == "scheduled"
    assert timestamp_path.exists()
    first = json.loads(timestamp_path.read_text())["last_maintenance_at"]

    # A fresh timestamp means the next check is not due.
    assert app.maybe_run_maintenance() is None

    # An overdue timestamp (or a missing one) makes the next check run again.
    overdue = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    timestamp_path.write_text(json.dumps({"schema_version": 1, "last_maintenance_at": overdue}))
    again = app.maybe_run_maintenance()
    assert again is not None
    assert json.loads(timestamp_path.read_text())["last_maintenance_at"] != first


def test_maybe_run_maintenance_treats_a_naive_timestamp_as_overdue(tmp_path: Path) -> None:
    import json
    from datetime import datetime

    app, _ = _indexed_app(tmp_path)
    timestamp_path = app.paths.data / "maintenance.json"
    timestamp_path.write_text(json.dumps({"last_maintenance_at": datetime.now().isoformat()}))

    report = app.maybe_run_maintenance()

    assert report is not None
    assert report.registry_status == "ok"


def test_scheduled_maintenance_is_serialized_across_applications(tmp_path: Path) -> None:
    app, _ = _indexed_app(tmp_path)
    other = Application(app.paths, embedder=TinyEmbedder(), cwd=tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = app.maintain_storage
    outcomes: list[object] = []

    def slow_maintenance(
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        dry_run: bool = False,
        wait_for_lock: bool = False,
        trigger: str = "manual",
    ) -> object:
        entered.set()
        assert release.wait(timeout=5)
        return original(
            project,
            roots=roots,
            dry_run=dry_run,
            wait_for_lock=wait_for_lock,
            trigger=trigger,
        )

    with (
        patch.object(app, "maintain_storage", side_effect=slow_maintenance),
        patch.object(
            other,
            "maintain_storage",
            side_effect=AssertionError("the cadence lock must suppress a duplicate pass"),
        ),
    ):
        first = threading.Thread(target=lambda: outcomes.append(app.maybe_run_maintenance()))
        first.start()
        assert entered.wait(timeout=5)
        second = threading.Thread(target=lambda: outcomes.append(other.maybe_run_maintenance()))
        second.start()
        second.join(timeout=5)
        assert not second.is_alive()
        release.set()
        first.join(timeout=10)

    assert not first.is_alive()
    assert len(outcomes) == 2
    assert sum(outcome is None for outcome in outcomes) == 1
    assert (app.paths.data / "maintenance.json").exists()


def test_registry_only_maintenance_runs_and_persists_the_cadence(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )

    with patch.object(
        app.store, "maintain_registry", wraps=app.store.maintain_registry
    ) as maintain_registry:
        report = app.maybe_run_maintenance()

    assert report is not None
    maintain_registry.assert_called_once()
    assert report.projects == []
    assert report.registry_status == "ok"
    assert report.registry_after is not None
    assert (app.paths.data / "maintenance.json").exists()


def test_damaged_partition_fails_scheduled_maintenance_without_persisting(
    tmp_path: Path,
) -> None:
    app, project = _indexed_app(tmp_path)
    app.store._partitions.pop(project.id, None)
    partition = app.store.directory / "projects" / project.id
    (partition / "files.lance").rename(tmp_path / "files.lance.bak")

    report = app.maybe_run_maintenance()

    assert report is not None
    entry = next(result for result in report.projects if result.project.id == project.id)
    assert entry.status == "error"
    assert entry.before is not None and entry.before.partition_open_failed is True
    assert project.id in report.failed_projects
    assert not (app.paths.data / "maintenance.json").exists()


def test_registry_failure_prevents_a_successful_cadence_timestamp(tmp_path: Path) -> None:
    app, _ = _indexed_app(tmp_path)

    with patch.object(app.store, "maintain_registry", side_effect=RuntimeError("boom")):
        report = app.maybe_run_maintenance()

    assert report is not None
    assert report.registry_status == "error"
    assert report.registry_error == "RuntimeError: boom"
    assert not (app.paths.data / "maintenance.json").exists()


def test_maybe_run_maintenance_does_not_persist_after_errors(
    tmp_path: Path,
) -> None:
    app, project = _indexed_app(tmp_path)
    timestamp_path = app.paths.data / "maintenance.json"

    with patch.object(app.store, "maintain_project", side_effect=RuntimeError("boom")):
        report = app.maybe_run_maintenance()

    assert report is not None
    assert project.id in report.failed_projects
    assert not timestamp_path.exists()


def test_project_status_includes_the_last_run_summary_and_live_progress(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)

    before = app.project_status(project.id)
    assert before.last_run is None
    assert before.progress is None

    app.index_project(project.id)

    status = app.project_status(project.id)
    assert status.last_run is not None
    assert status.last_run.state == "completed"
    assert status.last_run.trigger == "manual"
    assert status.last_run.eligible_files == 1
    assert status.last_run.changed_files == 1
    assert status.progress is None
    assert status.chunk_count > 0


def test_index_history_is_paginated_and_never_returns_more_than_asked(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    for _ in range(3):
        app.index_project(project.id)

    first = app.index_history(project.id, limit=2)
    assert first.project is not None
    assert first.project.id == project.id
    assert len(first.runs) == 2
    assert first.next_cursor is not None

    second = app.index_history(project.id, limit=2, cursor=first.next_cursor)
    assert len(second.runs) == 1
    assert second.next_cursor is None
    assert first.runs[0].run_id != second.runs[0].run_id


def test_history_rejects_bad_limits_and_cursors_with_structured_errors(
    tmp_path: Path,
) -> None:
    """Cursors are opaque user-supplied tokens and limits arrive from the CLI
    unguarded; both must surface as CodeIndexingError, never a traceback."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    with pytest.raises(CodeIndexingError) as caught:
        app.index_history(project.id, cursor="garbage")
    assert caught.value.code is ErrorCode.INVALID_CURSOR

    with pytest.raises(CodeIndexingError) as caught:
        app.index_history(project.id, limit=0)
    assert caught.value.code is ErrorCode.INVALID_FILTER


def test_reference_tool_path_uses_the_lazy_query_and_backfill_triggers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    # Drop the reference generation so the backfill has real work to do: a
    # no-op backfill deliberately records nothing (reference tools run one on
    # every query, and durable no-op rows would evict genuine index runs from
    # the bounded history window).
    _remove_reference_generation(app.store, project.id)

    report = app.ensure_reference_index(project.id)
    assert report.files_current == 1
    assert report.files_backfilled == 1

    page = app.index_history(project.id, limit=10)
    triggers = {run.trigger for run in page.runs}
    assert "reference-backfill" in triggers

    # The converged retry finds nothing missing and leaves no new row behind.
    app.ensure_reference_index(project.id)
    page = app.index_history(project.id, limit=10)
    assert [run.trigger for run in page.runs].count("reference-backfill") == 1


def _counted_scanner_scans(app: Application, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap the application scanner's iter_scan, returning a shared counter."""
    counter: list[int] = [0]
    original = app.indexer.scanner.iter_scan

    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(app.indexer.scanner, "iter_scan", counted)
    return counter


def test_project_status_caches_a_clean_freshness_answer_briefly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated status calls in one interaction must not walk a clean project
    once per call; a brief negative cache covers them."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    scans = _counted_scanner_scans(app, monkeypatch)
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 60.0)

    assert app.project_status(project.id).state == "ready"
    assert app.project_status(project.id).state == "ready"
    assert app.project_status(project.id).state == "ready"

    assert scans[0] == 1


def test_project_status_rechecks_after_the_cache_expires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    scans = _counted_scanner_scans(app, monkeypatch)
    # A zero-length window means every status call re-checks the tree.
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 0.0)

    assert app.project_status(project.id).state == "ready"
    assert app.project_status(project.id).state == "ready"

    assert scans[0] == 2


def test_indexing_invalidates_the_cached_freshness_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    scans = _counted_scanner_scans(app, monkeypatch)
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 60.0)

    assert app.project_status(project.id).state == "ready"
    assert scans[0] == 1

    # Whatever invalidates the entry -- an index run, a reference backfill,
    # registration, removal -- the next status check must re-scan.
    app.invalidate_freshness(project.id)
    assert app.project_status(project.id).state == "ready"
    assert scans[0] == 2


def test_scan_config_changes_invalidate_the_cached_freshness_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "excluded.py").write_text("value = 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    scans = _counted_scanner_scans(app, monkeypatch)
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 60.0)

    assert app.project_status(project.id).state == "ready"
    assert scans[0] == 1

    # A scan-config change makes the cached answer inapplicable; the next
    # status must re-check the tree instead of trusting the old fingerprint.
    # Excluding a file that the index still holds is a genuine difference, so
    # the re-check truthfully reports the project stale.
    changed = project.model_copy(
        update={"scan": project.scan.model_copy(update={"exclude": ["excluded.py"]})}
    )
    app.store.upsert_project(changed, model_id="test/tiny", state="ready")

    assert app.project_status(project.id).state == "stale"
    assert scans[0] == 2


def test_edits_are_detected_on_the_next_status_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative cache never outlives an edit: it expires within its brief
    window, and the honest primitive (project_is_stale) is never cached."""
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def before():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 60.0)

    assert app.project_status(project.id).state == "ready"

    source.write_text("def after():\n    return 2\n")

    # project_is_stale is never cached: the edit is visible immediately.
    assert app.project_is_stale(project.id) is True

    # The status answer comes from the cache until it expires -- that is the
    # accepted brief staleness of the negative cache -- and once the entry is
    # gone the next check re-scans and reports the edit.
    assert app.project_status(project.id).state == "ready"
    app.invalidate_freshness(project.id)
    assert app.project_status(project.id).state == "stale"


def test_inspect_scan_paginates_and_filters(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("ignored.py\n")
    (root / "ignored.py").write_text("value = 0\n")
    (root / "notes.md").write_text("not source\n")
    for name in ("a.py", "b.py", "c.py"):
        (root / name).write_text("def symbol():\n    return 1\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    # Deterministic per-directory order: unsupported files are yielded at
    # their sorted position, eligible candidates when their batch flushes.
    first = app.inspect_scan(project.id, limit=2)
    assert first.project is not None and first.project.id == project.id
    assert [item.path.as_posix() for item in first.items] == [".gitignore", "notes.md"]
    assert first.items[0].outcome == "skipped"
    assert first.items[0].reason == "unsupported"
    assert first.next_cursor is not None

    second = app.inspect_scan(project.id, limit=2, cursor=first.next_cursor)
    assert [item.path.as_posix() for item in second.items] == ["a.py", "b.py"]
    assert second.items[0].outcome == "eligible"
    assert second.items[0].language == "python"
    assert second.items[0].size is not None
    assert second.next_cursor is not None

    third = app.inspect_scan(project.id, limit=2, cursor=second.next_cursor)
    assert [item.path.as_posix() for item in third.items] == ["c.py", "ignored.py"]
    assert third.items[1].reason == "ignored"
    assert third.next_cursor is None

    eligible = app.inspect_scan(project.id, outcome="eligible")
    assert [item.path.as_posix() for item in eligible.items] == ["a.py", "b.py", "c.py"]

    skipped = app.inspect_scan(project.id, outcome="skipped")
    assert {item.path.as_posix() for item in skipped.items} == {
        ".gitignore",
        "ignored.py",
        "notes.md",
    }

    ignored_only = app.inspect_scan(project.id, reason="ignored")
    assert [item.path.as_posix() for item in ignored_only.items] == ["ignored.py"]


def test_inspect_scan_rejects_unknown_filters_and_cursors(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    with pytest.raises(CodeIndexingError) as outcome_error:
        app.inspect_scan(project.id, outcome="maybe")
    assert outcome_error.value.code is ErrorCode.INVALID_FILTER

    with pytest.raises(CodeIndexingError) as reason_error:
        app.inspect_scan(project.id, reason="exploded")
    assert reason_error.value.code is ErrorCode.INVALID_FILTER

    with pytest.raises(CodeIndexingError) as cursor_error:
        app.inspect_scan(project.id, cursor="not-a-number")
    assert cursor_error.value.code is ErrorCode.INVALID_CURSOR

    with pytest.raises(CodeIndexingError) as limit_error:
        app.inspect_scan(project.id, limit=0)
    assert limit_error.value.code is ErrorCode.INVALID_FILTER


def test_returning_to_a_clean_indexed_branch_skips_the_source_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching back to an unchanged clean cached branch must cost no scan.

    The acceptance gate for branch caching is stronger than "no parsing or
    embedding": an exact clean cache hit performs no scanner, parser, or
    embedder work at all, because the slot already proves every generation
    identity matches.
    """
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text("def main_branch():\n    return 1\n")
    run_git("add", "main.py", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "main", cwd=root)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    run_git("checkout", "-qb", "feature", cwd=root)
    (root / "feature.py").write_text("def feature_branch():\n    return 2\n")
    run_git("add", "feature.py", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "feature", cwd=root)
    app.index_project(project.id)
    assert app.project_status(project.id).active_slot_id is not None

    run_git("checkout", "-q", "main", cwd=root)
    scans = _counted_scanner_scans(app, monkeypatch)
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 0.0)

    status = app.project_status(project.id)

    assert status.state == "ready"
    assert status.branch_build_pending is False
    assert scans[0] == 0


def test_a_commit_with_a_hidden_content_change_marks_the_slot_stale(
    tmp_path: Path,
) -> None:
    """A same-size, same-mtime commit must not keep a slot looking current.

    Metadata cannot distinguish the reset or fast-forward case, so a slot
    indexed at a different HEAD of the same branch reports stale until an
    index run validates the commit-to-commit diff.
    """
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text("def one():\n    return 1\n")
    run_git("add", "main.py", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "first", cwd=root)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    assert app.project_status(project.id).state == "ready"

    _write_with_pinned_mtime(root / "main.py", "def one():\n    return 2\n")
    run_git("add", "main.py", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "second", cwd=root)

    stale = app.project_status(project.id)

    assert stale.state == "stale"
    assert stale.branch_build_pending is True

    report = app.index_project(project.id)

    assert report.indexed_files == 1
    assert app.project_status(project.id).state == "ready"
    hits = app.search_code("return 2", projects=[project.id]).hits
    assert hits and "return 2" in hits[0].snippet


def _git_repo_with_main(tmp_path: Path, name: str = "repo") -> tuple[Path, ProjectInfo]:
    root = tmp_path / name
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text("def main_branch():\n    return 1\n")
    run_git("add", "main.py", cwd=root)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "main",
        cwd=root,
    )
    return root, ProjectInfo(id="pending", name=name, root=root)


def test_a_worktree_joins_the_registration_and_keeps_its_own_slot(
    tmp_path: Path,
) -> None:
    """Registering a linked worktree must not mint a second project."""
    root, _ = _git_repo_with_main(tmp_path)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    main_status = app.project_status(project.id)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "--detach", str(worktree), cwd=root)
    joined = app.init_project(worktree)

    assert joined.id == project.id
    assert existing_marker_path(worktree) is not None
    assert len(app.list_projects()) == 1

    app.index_project(project.id, roots=[worktree])
    wt_status = app.project_status(project.id, roots=[worktree])
    assert wt_status.checkout_root == str(worktree.resolve())
    assert wt_status.active_slot_id != main_status.active_slot_id
    # The canonical checkout keeps its own pointer and its own rows.
    assert app.project_status(project.id).active_slot_id == main_status.active_slot_id
    hits = app.search_code("main branch", projects=[project.id], roots=[root])
    assert [hit.symbol for hit in hits.hits] == ["main_branch"]
    merged = app.search_code("branch", roots=[root, worktree])
    assert {hit.project_id for hit in merged.hits} == {project.id}


def test_search_merges_slots_of_all_requested_checkouts(tmp_path: Path) -> None:
    root, _ = _git_repo_with_main(tmp_path)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    # Branch off main, then diverge both histories so each checkout holds a
    # symbol the other has never seen.
    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "--detach", str(worktree), cwd=root)
    (worktree / "wt.py").write_text("def worktree_branch():\n    return 2\n")
    run_git("add", "wt.py", cwd=worktree)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "worktree commit",
        cwd=worktree,
    )
    (root / "extra.py").write_text("def main_only():\n    return 3\n")
    run_git("add", "extra.py", cwd=root)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "diverge main",
        cwd=root,
    )
    app.init_project(worktree)
    app.index_project(project.id, roots=[worktree])
    app.index_project(project.id, roots=[root])

    def symbols(response: SearchResponse) -> list[str]:
        return sorted(hit.symbol for hit in response.hits)

    # Each checkout's slot answers only from its own branch.
    scoped_main = app.search_code("return", projects=[project.id], roots=[root])
    assert symbols(scoped_main) == ["main_branch", "main_only"]
    scoped_wt = app.search_code("return", projects=[project.id], roots=[worktree])
    assert symbols(scoped_wt) == ["main_branch", "worktree_branch"]

    # One request across both checkouts merges every slot into one ranking.
    merged = app.search_code("return", projects=[project.id], roots=[root, worktree])
    assert symbols(merged) == ["main_branch", "main_only", "worktree_branch"]


def test_a_branch_slot_survives_relocation_between_worktrees(tmp_path: Path) -> None:
    """Checking a branch out elsewhere reuses the slot; no second index run."""
    root, _ = _git_repo_with_main(tmp_path)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "-b", "feature", str(worktree), cwd=root)
    (worktree / "feature.py").write_text("def feature_branch():\n    return 3\n")
    run_git("add", "feature.py", cwd=worktree)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        "feature",
        cwd=worktree,
    )
    app.init_project(worktree)
    app.index_project(project.id, roots=[worktree])
    feature_slot = app.project_status(project.id, roots=[worktree]).active_slot_id
    feature_chunk_id = (
        app.find_symbol("feature_branch", project.id, roots=[worktree]).hits[0].chunk_id
    )

    # Tear the worktree down out of band; pruning unregisters it so the
    # branch becomes available to the main checkout again.
    shutil.rmtree(worktree)
    run_git("worktree", "prune", cwd=root)
    run_git("checkout", "-q", "feature", cwd=root)

    relocated = app.project_status(project.id, roots=[root])
    assert relocated.checkout_root == str(root.resolve())
    assert relocated.active_slot_id == feature_slot
    assert relocated.branch_build_pending is False
    # The worktree's branch tree (main.py + feature.py) is now served from
    # the main checkout through the very same slot.
    assert relocated.file_count == 2
    # And the cached chunk resolves through the shared registration.
    assert app.get_chunk(feature_chunk_id).content is not None


def test_init_project_unifies_a_legacy_duplicate_worktree_registration(
    tmp_path: Path,
) -> None:
    """Pre-worktree registrations surface as warnings until init unifies them."""
    root, _ = _git_repo_with_main(tmp_path)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    project = app.init_project(root)
    app.index_project(project.id)

    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "--detach", str(worktree), cwd=root)
    # Simulate the old behavior: the worktree was initialized as its own
    # project before registrations were shared across checkouts.
    legacy_marker = initialize_project(worktree)
    app.store.upsert_project(legacy_marker, model_id=app.embedder.model_id, state="pending")

    status = app.storage_status()
    assert len(status.worktree_warnings) == 1

    unified = app.init_project(worktree)

    assert unified.id == project.id
    assert len(app.list_projects()) == 1
    assert app.storage_status().worktree_warnings == []
    assert app.index_project(project.id, roots=[worktree]).indexed_files >= 1
    wt_status = app.project_status(project.id, roots=[worktree])
    assert wt_status.state == "ready"
    assert wt_status.file_count == 1
