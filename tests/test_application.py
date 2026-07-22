import shutil
from pathlib import Path

import pytest

from incode_mcp.application import Application, RuntimePaths
from incode_mcp.errors import ErrorCode, IncodeError


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


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
    report = app.index_project(roots=[root])
    status = app.project_status(roots=[root])
    search = app.search_code("locate feature", roots=[root])
    removal = app.remove_project(project.id)

    assert report.project_id == project.id
    assert status.file_count == 1
    assert status.chunk_count >= 1
    assert search.hits[0].symbol == "locate_feature"
    assert removal.removed is True
    assert app.list_projects() == []
    assert (root / ".incode" / "project.toml").exists()


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
