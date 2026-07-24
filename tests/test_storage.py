from __future__ import annotations

import time
from pathlib import Path

import lancedb

from incode_mcp.projects import initialize_project
from incode_mcp.storage import LanceStore


def test_storage_uses_one_partition_per_project(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)

    store.upsert_project(project, model_id="test/model", state="pending")
    store.list_files(project.id)

    assert (store.directory / "registry" / "projects.lance").exists()
    assert (store.directory / "projects" / project.id / "files.lance").exists()
    assert (store.directory / "projects" / project.id / "chunks.lance").exists()


def test_v1_store_is_backed_up_and_registered_for_lazy_rebuild(tmp_path: Path) -> None:
    directory = tmp_path / "lancedb"
    directory.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    legacy = lancedb.connect(directory)
    projects = legacy.create_table("projects", schema=LanceStore._project_schema())
    projects.add(
        [
            {
                "id": project.id,
                "name": project.name,
                "root": str(project.root),
                "payload": project.model_dump_json(),
                "model_id": "test/model",
                "vector_dimension": 4,
                "schema_version": 1,
                "state": "ready",
                "updated_at": time.time_ns(),
            }
        ]
    )
    del projects
    del legacy

    store = LanceStore(directory, vector_dimension=4)

    assert store.list_projects() == [project]
    assert store.project_state(project.id) == "pending"
    assert list(tmp_path.glob("lancedb-v1-backup-*"))
