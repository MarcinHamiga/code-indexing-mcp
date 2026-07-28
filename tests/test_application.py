import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from incode_mcp.application import Application, RuntimePaths
from incode_mcp.backends import Accelerator
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.settings import IndexSettings


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

    with pytest.raises(IncodeError) as raised:
        app.index_project(str(duplicate))
    assert raised.value.code is ErrorCode.PROJECT_ID_CONFLICT

    shutil.rmtree(original)
    report = app.index_project(str(duplicate))
    assert report.project_id == project.id
    assert app.list_projects()[0].root == duplicate.resolve()


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
    (original / ".ci-mcp").rename(original / ".incode")
    shutil.copytree(original, duplicate)

    with pytest.raises(IncodeError) as raised:
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

    other_app = Application(paths, embedder=OtherModelTinyEmbedder(), cwd=root)
    with pytest.raises(IncodeError) as raised_init:
        other_app.init_project(root)
    assert raised_init.value.code is ErrorCode.INDEX_INCOMPATIBLE

    with pytest.raises(IncodeError) as raised_discover:
        other_app.discover_project(root)
    assert raised_discover.value.code is ErrorCode.INDEX_INCOMPATIBLE


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


def test_the_query_model_stays_in_process_regardless_of_the_backend(tmp_path: Path) -> None:
    """Acceleration targets passage indexing; a query must never wait on it."""
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    embedder = TinyEmbedder()

    app = Application(paths, embedder=embedder, cwd=tmp_path)

    assert app.search.embedder is embedder
    assert app.embedder is embedder
