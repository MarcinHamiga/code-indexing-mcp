from __future__ import annotations

import time
from pathlib import Path

import lancedb

from incode_mcp.models import ProjectInfo, StoredChunk, StoredFile
from incode_mcp.projects import initialize_project
from incode_mcp.storage import LanceStore


def stored_file(project_id: str, *, file_id: str = "file-1") -> StoredFile:
    return StoredFile(
        file_id=file_id,
        project_id=project_id,
        path="module.py",
        language="python",
        size=4,
        mtime_ns=1,
        content_hash="hash",
        indexed_at=time.time_ns(),
    )


def test_storage_uses_one_partition_per_project(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)

    store.upsert_project(project, model_id="test/model", state="pending")
    store.upsert_file(stored_file(project.id))

    assert (store.directory / "registry" / "projects.lance").exists()
    assert (store.directory / "projects" / project.id / "files.lance").exists()
    assert (store.directory / "projects" / project.id / "chunks.lance").exists()


def test_reads_never_materialize_a_partition(tmp_path: Path) -> None:
    """get_chunk scans every project; a registered-but-unindexed one stays empty."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model", state="pending")

    assert store.get_chunk("no-such-chunk") is None
    assert store.list_files(project.id) == []
    assert store.list_chunks([project.id]) == []
    assert store.count_chunks([project.id]) == 0
    assert store.outline_chunks("module.py", project.id) == []
    assert store.find_symbol_chunks("answer", project.id, match="exact", kinds=None, limit=5) == []

    assert not (store.directory / "projects").exists()


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


def _stored_chunks(project_id: str, count: int) -> list[StoredChunk]:
    return [
        StoredChunk(
            chunk_id=f"chunk-{index}",
            project_id=project_id,
            file_id="file-1",
            path="module.py",
            language="python",
            kind="function",
            symbol=f"symbol_{index}",
            qualified_symbol=f"symbol_{index}",
            parent_symbol=None,
            start_byte=0,
            end_byte=1,
            start_line=index + 1,
            end_line=index + 1,
            content="pass",
            embedding_text="pass",
            search_text="pass",
            content_hash="hash",
            part_index=0,
            vector=[0.0, 0.0, 0.0, 1.0],
        )
        for index in range(count)
    ]


def _store_with_one_chunk(tmp_path: Path) -> tuple[LanceStore, str, str]:
    """A store holding a single committed chunk, with its project and chunk ids."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    chunks = _stored_chunks(project.id, 1)
    store.replace_file(stored_file(project.id), chunks)
    return store, project.id, chunks[0].chunk_id


def test_compaction_keeps_recent_versions_readable(tmp_path: Path) -> None:
    """Concurrent readers must survive the maintenance pass after deletions."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    chunks = _stored_chunks(project.id, 4)
    record = StoredFile(
        file_id="file-1",
        project_id=project.id,
        path="module.py",
        language="python",
        size=4,
        mtime_ns=1,
        content_hash="hash",
        indexed_at=time.time_ns(),
    )
    store.replace_file(record, chunks)
    store.ensure_indexes(project.id)

    store.remove_file(project.id, "file-1")
    store.ensure_indexes(project.id, compact=True)

    # A version reaped mid-read would surface here as a Lance IO failure.
    assert store.count_chunks([project.id]) == 0
    assert store.list_files(project.id) == []


def test_get_chunk_does_not_read_the_vector_column(tmp_path: Path) -> None:
    from incode_mcp.models import CodeChunk

    store, _project, chunk_id = _store_with_one_chunk(tmp_path)

    chunk = store.get_chunk(chunk_id)

    assert isinstance(chunk, CodeChunk)
    assert not hasattr(chunk, "vector")
    assert store.get_chunk("no-such-chunk") is None


def test_partition_cache_evicts_least_recently_used(tmp_path: Path) -> None:
    """The daemon is long-lived and get_chunk faults in every project's partition.

    Without a bound, two open LanceTable handles per project accumulate for the life
    of the process.
    """
    from incode_mcp import storage as storage_module

    store = LanceStore(tmp_path / "data", vector_dimension=4)
    projects = []
    for index in range(storage_module.MAX_CACHED_PARTITIONS + 3):
        root = tmp_path / f"p{index}"
        root.mkdir()
        project = ProjectInfo(id=f"id-{index:02d}", name=f"p{index}", root=root)
        store.upsert_project(project, model_id="test")
        store._tables(project.id)  # fault the partition in
        projects.append(project)

    assert len(store._partitions) == storage_module.MAX_CACHED_PARTITIONS
    # The oldest three were evicted; the most recent are still resident.
    assert projects[0].id not in store._partitions
    assert projects[1].id not in store._partitions
    assert projects[2].id not in store._partitions
    assert projects[-1].id in store._partitions


def test_partition_cache_keeps_recently_used_entries(tmp_path: Path) -> None:
    from incode_mcp import storage as storage_module

    store = LanceStore(tmp_path / "data", vector_dimension=4)
    ids = []
    for index in range(storage_module.MAX_CACHED_PARTITIONS):
        root = tmp_path / f"p{index}"
        root.mkdir()
        project = ProjectInfo(id=f"id-{index:02d}", name=f"p{index}", root=root)
        store.upsert_project(project, model_id="test")
        store._tables(project.id)
        ids.append(project.id)

    # Touch the oldest so it is no longer the eviction candidate, then overflow by one.
    store._tables(ids[0])
    overflow_root = tmp_path / "overflow"
    overflow_root.mkdir()
    overflow = ProjectInfo(id="id-overflow", name="overflow", root=overflow_root)
    store.upsert_project(overflow, model_id="test")
    store._tables(overflow.id)

    assert ids[0] in store._partitions, "a freshly used partition must not be evicted"
    assert ids[1] not in store._partitions


def test_evicted_partition_reopens_with_its_data(tmp_path: Path) -> None:
    """Eviction is a cache decision, never a data decision."""
    from incode_mcp import storage as storage_module

    store, project, chunk_id = _store_with_one_chunk(tmp_path)
    for index in range(storage_module.MAX_CACHED_PARTITIONS + 1):
        root = tmp_path / f"filler{index}"
        root.mkdir()
        filler = ProjectInfo(id=f"filler-{index:02d}", name=f"f{index}", root=root)
        store.upsert_project(filler, model_id="test")
        store._tables(filler.id)

    assert project not in store._partitions
    assert store.count_chunks([project]) == 1
    assert store.get_chunk(chunk_id) is not None
