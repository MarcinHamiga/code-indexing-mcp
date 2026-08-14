from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import lancedb
import pyarrow as pa
import pytest
from lancedb.expr import Expr
from lancedb.merge import LanceMergeInsertBuilder
from lancedb.table import LanceTable

from code_indexing_mcp import storage as storage_module
from code_indexing_mcp.models import ProjectInfo, StoredChunk, StoredFile
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.storage import (
    LanceStore,
    overlap_warnings,
    overlapping_registration,
    worktree_warnings,
)


def reference_record(
    project_id: str,
    file_id: str,
    *,
    reference_id: str = "reference-1",
    target_name: str = "answer",
    **updates: object,
) -> dict[str, object]:
    row = {
        "reference_id": reference_id,
        "record_kind": "reference",
        "file_id": file_id,
        "project_id": project_id,
        "path": "module.py",
        "language": "python",
        "kind": "call",
        "source_qualified_symbol": "caller",
        "written_name": target_name,
        "target_name": target_name,
        "module_path": None,
        "imported_name": None,
        "alias": None,
        "receiver_text": None,
        "start_byte": 0,
        "end_byte": len(target_name or ""),
        "start_line": 1,
        "end_line": 1,
        "shape_json": '{"positional_count": 0}',
        "content_hash": "hash",
        "schema_version": 1,
    }
    return {**row, **updates}


def reference_table(*rows: dict[str, object]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=LanceStore.reference_arrow_schema())


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
    assert (store.directory / "projects" / project.id / "references.lance").exists()


def test_reads_never_materialize_a_partition(tmp_path: Path) -> None:
    """Reads on a registered-but-unindexed project leave it empty on disk."""
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
    assert store.list_reference_records(project.id) == []
    assert store.list_reference_records(project.id, record_kinds=("reference",)) == []
    assert store.reference_coverage(project.id) == []
    assert store.declaration_shapes(project.id, "answer") == []
    assert store.target_name_candidates(project.id, "answer") == []
    assert store.declarations_for_files(project.id, ["file-1"]) == []

    assert not (store.directory / "projects").exists()


def test_reference_rows_are_replaced_independently_from_chunk_rows(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    record = stored_file(project.id)
    store.upsert_project(project, model_id="test/model")

    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist([record.model_dump()], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[(["file-1"], reference_table(reference_record(project.id, "file-1")))],
    )
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist([record.model_dump()], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[
            (
                ["file-1"],
                reference_table(
                    reference_record(
                        project.id,
                        "file-1",
                        reference_id="reference-2",
                        target_name="renamed",
                    )
                ),
            )
        ],
    )

    records = store.list_reference_records(project.id)
    assert [row["reference_id"] for row in records] == ["reference-2"]
    assert store.target_name_candidates(project.id, "renamed") == records


def test_removing_a_file_deletes_its_reference_rows(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    record = stored_file(project.id)
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist([record.model_dump()], schema=LanceStore.file_arrow_schema()),
        chunk_batches=[
            (
                [record.file_id],
                pa.Table.from_pylist(
                    [_stored_chunks(project.id, 1)[0].model_dump()],
                    schema=LanceStore.chunk_arrow_schema(store.vector_dimension),
                ),
            )
        ],
        reference_batches=[(["file-1"], reference_table(reference_record(project.id, "file-1")))],
    )

    assert store.count_chunks([project.id]) == 1

    store.remove_file(project.id, "file-1")

    assert store.list_files(project.id) == []
    assert store.count_chunks([project.id]) == 0
    assert store.list_reference_records(project.id) == []


def test_reference_read_methods_apply_exact_structural_filters(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    record = stored_file(project.id)
    coverage = reference_record(
        project.id,
        record.file_id,
        reference_id="coverage",
        record_kind="coverage",
        kind=None,
        source_qualified_symbol=None,
        written_name=None,
        target_name=None,
        shape_json=None,
    )
    declaration = reference_record(
        project.id,
        record.file_id,
        reference_id="declaration",
        record_kind="declaration",
        kind="function",
        source_qualified_symbol="package.answer",
        target_name="answer",
        shape_json="[]",
    )
    imported = reference_record(
        project.id,
        record.file_id,
        reference_id="import",
        kind="import",
        module_path="package",
        target_name="answer",
    )
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist([record.model_dump()], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[(["file-1"], reference_table(coverage, declaration, imported))],
    )

    assert store.coverage_for_file(project.id, "file-1", 1) == [coverage]
    assert store.declaration_shapes(project.id, "package.answer") == [declaration]
    assert store.target_name_candidates(project.id, "answer") == [declaration, imported]
    # record_kind narrows target_name_candidates to one shape (S4).
    assert store.target_name_candidates(project.id, "answer", record_kind="declaration") == [
        declaration
    ]
    assert store.target_name_candidates(project.id, "answer", record_kind="reference") == [imported]
    # declarations restricted to a candidate file set (S4).
    assert store.declarations_for_files(project.id, ["file-1"]) == [declaration]
    assert store.declarations_for_files(project.id, ["no-such-file"]) == []
    assert store.declarations_for_files(project.id, []) == []
    # list_reference_records' record_kinds narrows the same way (S4/E3): a
    # query-time caller can drop declaration rows from the fetch entirely.
    assert store.list_reference_records(project.id, record_kinds=("declaration",)) == [declaration]
    assert store.list_reference_records(project.id, record_kinds=("reference", "coverage")) == [
        coverage,
        imported,
    ]
    assert store.list_reference_records(project.id, record_kinds=()) == []
    # version kwarg is honored on the same pinned snapshot as list_reference_records.
    version = store.reference_version(project.id)
    assert store.declaration_shapes(project.id, "package.answer", version=version) == [declaration]
    assert store.target_name_candidates(project.id, "answer", version=version) == [
        declaration,
        imported,
    ]
    assert store.declarations_for_files(project.id, ["file-1"], version=version) == [declaration]
    assert store.reference_coverage(project.id, version=version) == [coverage]
    # schema_version pushes the same filter as list_reference_records' (S4/E3):
    # a stale-schema row must not survive any of these three narrower reads
    # either, or the regression `list_reference_records`' own schema_version
    # test guards against would resurface through these instead.
    assert store.declaration_shapes(
        project.id, "package.answer", schema_version=declaration["schema_version"]
    ) == [declaration]
    assert store.declaration_shapes(project.id, "package.answer", schema_version=999) == []
    assert store.target_name_candidates(
        project.id, "answer", schema_version=imported["schema_version"]
    ) == [declaration, imported]
    assert store.target_name_candidates(project.id, "answer", schema_version=999) == []
    assert store.declarations_for_files(
        project.id, ["file-1"], schema_version=declaration["schema_version"]
    ) == [declaration]
    assert store.declarations_for_files(project.id, ["file-1"], schema_version=999) == []


def test_list_reference_records_schema_version_pushes_the_filter_into_sql(
    tmp_path: Path,
) -> None:
    """`schema_version` narrows via the storage-layer `WHERE` clause (E3/S4).

    Omitting it must still return every row unfiltered -- `find_references`'s
    stale-schema-version regression test (finding 9) and several other
    callers pin their assertions to `list_reference_records(project_id)`
    returning the *whole* table regardless of schema generation.
    """
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    record = stored_file(project.id)
    current = reference_record(project.id, record.file_id, reference_id="current", schema_version=4)
    stale = reference_record(project.id, record.file_id, reference_id="stale", schema_version=3)
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist([record.model_dump()], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[(["file-1"], reference_table(current, stale))],
    )

    assert {row["reference_id"] for row in store.list_reference_records(project.id)} == {
        "current",
        "stale",
    }
    assert store.list_reference_records(project.id, schema_version=4) == [current]
    assert store.list_reference_records(project.id, schema_version=3) == [stale]
    assert store.list_reference_records(project.id, schema_version=5) == []


@pytest.mark.parametrize("schema_version", [True, "1"])
def test_list_reference_records_rejects_non_integer_schema_versions(
    tmp_path: Path, schema_version: object
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)

    with pytest.raises(ValueError, match="schema_version"):
        store.list_reference_records("project-1", schema_version=schema_version)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_version", [True, "1"])
def test_coverage_for_file_rejects_non_integer_schema_versions(
    tmp_path: Path, schema_version: object
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)

    with pytest.raises(ValueError, match="schema_version"):
        store.coverage_for_file("project-1", "file-1", schema_version)  # type: ignore[arg-type]


@pytest.mark.parametrize("schema_version", [True, "1"])
def test_declaration_side_pushdowns_reject_non_integer_schema_versions(
    tmp_path: Path, schema_version: object
) -> None:
    """`declaration_shapes`/`target_name_candidates`/`declarations_for_files` share
    `list_reference_records`'s schema_version validation (S4/E3)."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)

    with pytest.raises(ValueError, match="schema_version"):
        store.declaration_shapes("project-1", "answer", schema_version=schema_version)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema_version"):
        store.target_name_candidates("project-1", "answer", schema_version=schema_version)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema_version"):
        store.declarations_for_files("project-1", ["file-1"], schema_version=schema_version)  # type: ignore[arg-type]


def test_reference_indexes_cover_every_exact_filter(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    store.ensure_indexes(project.id)
    store.ensure_indexes(project.id)
    tables = store._existing_tables(project.id)
    assert tables is not None and tables.references is not None

    indexed_columns = {
        column for index in tables.references.list_indices() for column in index.columns
    }

    assert {
        "file_id",
        "record_kind",
        "target_name",
        "module_path",
        "kind",
        "source_qualified_symbol",
        "schema_version",
    } <= indexed_columns


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
            chunk_id=f"{project_id}:chunk-{index}",
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
            identifier_terms="symbol module py",
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


def test_maintenance_keeps_recent_versions_readable(tmp_path: Path) -> None:
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
    store.maintain_project(project.id, cleanup_older_than=timedelta(hours=24))

    # A version reaped mid-read would surface here as a Lance IO failure.
    assert store.count_chunks([project.id]) == 0
    assert store.list_files(project.id) == []


def test_maintenance_preserves_rows_and_search_results(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id).model_dump()], schema=LanceStore.file_arrow_schema()
        ),
        chunk_batches=[(["file-1"], _chunk_table(project.id, "file-1", 2))],
        reference_batches=[(["file-1"], reference_table())],
    )
    store.ensure_indexes(project.id)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])
    references_before = store.list_reference_records(project.id)

    store.maintain_project(project.id, cleanup_older_than=timedelta(hours=24))

    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    assert store.list_reference_records(project.id) == references_before
    rows = store.hybrid_search("pass", [1.0, 0.0, 0.0, 0.0], [project.id], None, 5)
    assert {row["chunk_id"] for row in rows} == {chunk.chunk_id for chunk in chunks_before}


def test_hybrid_search_spans_content_and_identifier_terms(tmp_path: Path) -> None:
    """Keyword search must match both FTS indexes.

    Native FTS is single-column, so content and identifier_terms get one index
    each and hybrid search spans them with a MultiMatchQuery. A query that
    appears only in the normalized identifier terms (never in the code text)
    must still hit; so must one that appears only in the content.
    """
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    chunk_terms_only = StoredChunk(
        chunk_id=f"{project.id}:terms-only",
        file_id="file-terms",
        path="loader.py",
        language="python",
        kind="function",
        symbol="load",
        qualified_symbol="load",
        parent_symbol=None,
        start_byte=0,
        end_byte=1,
        start_line=1,
        end_line=1,
        content="def x(): pass",
        identifier_terms="zoo keeper config",
        part_index=0,
        vector=[0.0, 0.0, 0.0, 1.0],
    )
    chunk_content_only = StoredChunk(
        chunk_id=f"{project.id}:content-only",
        file_id="file-content",
        path="other.py",
        language="python",
        kind="function",
        symbol="y",
        qualified_symbol="y",
        parent_symbol=None,
        start_byte=0,
        end_byte=1,
        start_line=1,
        end_line=1,
        content="def get_http(): pass",
        identifier_terms="other py",
        part_index=0,
        vector=[1.0, 1.0, 1.0, 1.0],
    )
    record = stored_file(project.id, file_id="file-terms")
    record2 = stored_file(project.id, file_id="file-content")
    store.replace_file(record, [chunk_terms_only])
    store.replace_file(record2, [chunk_content_only])
    store.ensure_indexes(project.id)

    terms_hit = store.hybrid_search("keeper", [0.0, 0.0, 0.0, 0.0], [project.id], None, 5)
    content_hit = store.hybrid_search("http", [0.9, 0.9, 0.9, 0.9], [project.id], None, 5)

    # "keeper" appears only in the identifier-terms index, and "http" only in
    # the content index; the query vectors favor the *other* chunk, so an FTS
    # miss would leave the FTS-matched chunk behind. It must rank first.
    assert terms_hit[0]["chunk_id"] == chunk_terms_only.chunk_id
    assert content_hit[0]["chunk_id"] == chunk_content_only.chunk_id


def test_maintenance_keeps_versions_inside_the_retention_window(tmp_path: Path) -> None:
    """Fresh versions survive a pass: nothing older than the window is removed."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    for index in range(3):
        file_id = f"file-{index}"
        store.replace_files_from_arrow(
            project.id,
            files=pa.Table.from_pylist(
                [stored_file(project.id, file_id=file_id).model_dump()],
                schema=LanceStore.file_arrow_schema(),
            ),
            chunk_batches=[([file_id], _chunk_table(project.id, file_id, 1))],
        )
    before = store.storage_stats(project.id)
    chunks_before = next(table for table in before.tables if table.name == "chunks")

    store.maintain_project(project.id, cleanup_older_than=timedelta(hours=24))

    after = store.storage_stats(project.id)
    chunks_after = next(table for table in after.tables if table.name == "chunks")
    # Compaction writes a fresh merged version and keeps every younger version,
    # so the retained set only grows; nothing inside the window was removed.
    assert chunks_after.retained_version_count >= chunks_before.retained_version_count
    assert store.count_chunks([project.id]) == 3


def test_maintenance_removes_verified_versions_outside_the_retention_window(
    tmp_path: Path,
) -> None:
    """Verified versions older than the window are removed; current rows survive."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    for index in range(3):
        file_id = f"file-{index}"
        store.replace_files_from_arrow(
            project.id,
            files=pa.Table.from_pylist(
                [stored_file(project.id, file_id=file_id).model_dump()],
                schema=LanceStore.file_arrow_schema(),
            ),
            chunk_batches=[([file_id], _chunk_table(project.id, file_id, 1))],
        )
    before = store.storage_stats(project.id)
    chunks_before = next(table for table in before.tables if table.name == "chunks")
    assert chunks_before.retained_version_count >= 3

    store.maintain_project(project.id, cleanup_older_than=timedelta(seconds=0))

    after = store.storage_stats(project.id)
    chunks_after = next(table for table in after.tables if table.name == "chunks")
    # The zero window is the deterministic test stand-in for "older than the
    # window": every version is older than now, so only the freshly compacted
    # current version remains.
    assert chunks_after.retained_version_count == 1
    assert store.count_chunks([project.id]) == 3
    assert {file.file_id for file in store.list_files(project.id)} == {
        "file-0",
        "file-1",
        "file-2",
    }


def test_maintenance_covers_every_partition_table_and_the_registry(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id).model_dump()], schema=LanceStore.file_arrow_schema()
        ),
        chunk_batches=[(["file-1"], _chunk_table(project.id, "file-1", 1))],
        reference_batches=[(["file-1"], reference_table())],
    )

    optimized: list[str] = []
    original_optimize = LanceTable.optimize

    def counting_optimize(self: LanceTable, **kwargs: object) -> None:
        optimized.append(self.name)
        original_optimize(self, **kwargs)

    with patch.object(LanceTable, "optimize", counting_optimize):
        store.maintain_project(project.id, cleanup_older_than=timedelta(hours=24))
        store.maintain_registry(cleanup_older_than=timedelta(hours=24))

    assert optimized == ["files", "chunks", "references", "projects"]


def test_maintenance_never_passes_delete_unverified(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id).model_dump()], schema=LanceStore.file_arrow_schema()
        ),
        chunk_batches=[(["file-1"], _chunk_table(project.id, "file-1", 1))],
    )

    seen: list[dict[str, object]] = []
    original_optimize = LanceTable.optimize

    def recording_optimize(self: LanceTable, **kwargs: object) -> None:
        seen.append(kwargs)
        original_optimize(self, **kwargs)

    with patch.object(LanceTable, "optimize", recording_optimize):
        store.maintain_project(project.id, cleanup_older_than=timedelta(hours=24))

    assert seen
    # delete_unverified may be omitted (False) but must never be passed True.
    assert all(kwargs.get("delete_unverified") is not True for kwargs in seen)


def test_maintenance_skips_a_registered_project_without_a_partition(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model", state="pending")

    assert store.maintain_project(project.id, cleanup_older_than=timedelta(hours=24)) is False
    # Maintenance is not a write path: it must not materialize a partition.
    assert not (store.directory / "projects").exists()


def test_ensure_indexes_no_longer_optimizes(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    optimized: list[str] = []
    original_optimize = LanceTable.optimize

    def counting_optimize(self: LanceTable, **kwargs: object) -> None:
        optimized.append(self.name)
        original_optimize(self, **kwargs)

    with patch.object(LanceTable, "optimize", counting_optimize):
        store.ensure_indexes(project_id)

    assert optimized == []


def test_get_chunk_does_not_read_the_vector_column(tmp_path: Path) -> None:
    from code_indexing_mcp.models import CodeChunk

    store, _project, chunk_id = _store_with_one_chunk(tmp_path)

    chunk = store.get_chunk(chunk_id)

    assert isinstance(chunk, CodeChunk)
    assert not hasattr(chunk, "vector")
    assert store.get_chunk("no-such-chunk") is None


def test_get_chunk_resolves_the_routing_prefix_to_the_owning_partition(
    tmp_path: Path,
) -> None:
    """The project-id prefix routes get_chunk to one partition and back.

    The response's project_id is injected from the partition and its
    content_hash comes from the files table, not the chunk row.
    """
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = initialize_project(first_root)
    store.upsert_project(first, model_id="test/model")
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = initialize_project(second_root)
    store.upsert_project(second, model_id="test/model")
    first_chunk = _stored_chunks(first.id, 1)[0]
    second_chunk = _stored_chunks(second.id, 1)[0]
    store.replace_file(stored_file(first.id), [first_chunk])
    store.replace_file(stored_file(second.id), [second_chunk])

    from_first = store.get_chunk(first_chunk.chunk_id)
    from_second = store.get_chunk(second_chunk.chunk_id)

    assert from_first is not None and from_first.project_id == first.id
    assert from_second is not None and from_second.project_id == second.id
    assert from_first.content_hash == "hash"
    assert from_second.content_hash == "hash"


def test_get_chunk_reads_its_own_generation_content_hash(tmp_path: Path) -> None:
    store, project_id, chunk_id = _store_with_one_chunk(tmp_path)
    tables = store._existing_tables(project_id)
    assert tables is not None
    tables.files.delete("file_id = 'file-1'")
    tables.files.add(
        [stored_file(project_id).model_copy(update={"content_hash": "new-hash"}).model_dump()]
    )

    chunk = store.get_chunk(chunk_id)

    assert chunk is not None
    assert chunk.content_hash == "hash"


def test_partition_deletion_waits_for_active_reader(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    finished = threading.Event()

    def delete() -> None:
        store.delete_partition(project_id, model_id="test/other")
        finished.set()

    with store.partition_access(project_id):
        thread = threading.Thread(target=delete)
        thread.start()
        assert not finished.wait(timeout=0.1)
    thread.join(timeout=5)

    assert finished.is_set()


def test_get_chunk_rejects_malformed_unknown_and_pre_migration_ids(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)

    # A pre-migration id has no routing prefix; a malformed one has an empty
    # prefix or digest; an unknown prefix names no partition. All are
    # deliberately indistinguishable from "not found", and none of them
    # materialize storage.
    assert store.get_chunk("chunk-1") is None
    assert store.get_chunk(":digest") is None
    assert store.get_chunk(f"{project_id}:") is None
    assert store.get_chunk("no-such-project:digest") is None
    assert not (store.directory / "projects" / "no-such-project").exists()


def test_partition_cache_evicts_least_recently_used(tmp_path: Path) -> None:
    """The daemon is long-lived; partition handles must not accumulate without bound.

    Without a bound, two open LanceTable handles per project accumulate for the life
    of the process.
    """
    from code_indexing_mcp import storage as storage_module

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
    from code_indexing_mcp import storage as storage_module

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
    from code_indexing_mcp import storage as storage_module

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


def test_list_chunks_does_not_materialize_vectors(tmp_path: Path) -> None:
    """Nothing in production calls list_chunks; the vectors were read for no one."""
    from code_indexing_mcp.models import IndexedChunk

    store, project, _ = _store_with_one_chunk(tmp_path)

    chunks = store.list_chunks([project])

    assert chunks
    assert all(isinstance(chunk, IndexedChunk) for chunk in chunks)
    assert not any(hasattr(chunk, "vector") for chunk in chunks)
    # The fields the tests actually read must survive the projection.
    assert chunks[0].identifier_terms
    assert chunks[0].content
    assert chunks[0].chunk_id


def test_stored_chunk_still_carries_its_vector(tmp_path: Path) -> None:
    """The write path is unaffected: StoredChunk keeps the vector it commits."""
    from code_indexing_mcp.models import IndexedChunk, StoredChunk

    assert issubclass(StoredChunk, IndexedChunk)
    assert "vector" in StoredChunk.model_fields
    assert "vector" not in IndexedChunk.model_fields
    # Field order matters to nothing in LanceDB, but the schema lists vector last
    # and keeping it there makes the inheritance a pure refactor.
    assert list(StoredChunk.model_fields)[-1] == "vector"


def _break_references_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every future `_tables()` call build a partition with no references table.

    `_tables()` normally creates the references table on demand
    (`exist_ok=True`), so the only way to observe a `None` reference table
    downstream -- the interrupted-transaction/legacy-partition state the
    bare asserts guarded against -- is to make table creation itself fail
    to produce one, the same way an older on-disk partition would.
    """
    real_table = LanceStore._table

    def fake_table(database: object, name: str, schema: object) -> object:
        if name == "references":
            return None
        return real_table(database, name, schema)  # type: ignore[arg-type]

    monkeypatch.setattr(LanceStore, "_table", staticmethod(fake_table))


def test_remove_file_raises_instead_of_asserting_on_a_missing_reference_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    _break_references_table(monkeypatch)

    with pytest.raises(RuntimeError, match="Reference table is missing"):
        store.remove_file(project.id, "file-1")


def test_table_versions_raises_instead_of_asserting_on_a_missing_reference_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    _break_references_table(monkeypatch)

    with pytest.raises(RuntimeError, match="Reference table is missing"):
        store.table_versions(project.id)


def test_restore_versions_checkout_raises_instead_of_asserting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    versions = store.table_versions(project.id)
    store._partitions.pop(project.id, None)
    (store.directory / "projects" / project.id / "references.lance").rename(
        tmp_path / "references.lance.bak"
    )

    with pytest.raises(RuntimeError, match="Reference table is missing"):
        store.restore_versions(project.id, versions)


def test_ensure_indexes_raises_instead_of_asserting_on_a_missing_reference_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    _break_references_table(monkeypatch)

    with pytest.raises(RuntimeError, match="Reference table is missing"):
        store.ensure_indexes(project.id)


def test_replace_files_from_arrow_raises_instead_of_asserting_on_a_missing_reference_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    _break_references_table(monkeypatch)

    with pytest.raises(RuntimeError, match="Reference table is missing"):
        store.replace_files_from_arrow(
            project.id,
            files=pa.Table.from_pylist([], schema=LanceStore.file_arrow_schema()),
            chunk_batches=(),
        )


def test_has_reference_table_distinguishes_missing_from_empty(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")

    # Never indexed: no partition at all, but not what this guard is for --
    # find_symbol/get_chunk already report "not found" long before a
    # reference query is reached.
    assert store.has_reference_table(project.id) is False

    store.upsert_file(stored_file(project.id))
    assert store.has_reference_table(project.id) is True

    store._partitions.pop(project.id, None)
    (store.directory / "projects" / project.id / "references.lance").rename(
        tmp_path / "references.lance.bak"
    )
    assert store.has_reference_table(project.id) is False


def test_has_file_errors(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)

    # Never indexed: no partition at all.
    assert store.has_file_errors(project.id) is False

    store.upsert_project(project, model_id="test/model")
    store.upsert_file(stored_file(project.id))
    assert store.has_file_errors(project.id) is False

    # A rejection tombstone is a deliberate skip, not an error.
    store.upsert_file(
        stored_file(project.id).model_copy(update={"has_errors": True, "error": "rejected: binary"})
    )
    assert store.has_file_errors(project.id) is False

    store.upsert_file(
        stored_file(project.id).model_copy(update={"has_errors": True, "error": "embedding failed"})
    )
    assert store.has_file_errors(project.id) is True


def test_storage_stats_reports_table_level_metrics(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)

    report = store.storage_stats(project_id)

    assert report.project.id == project_id
    assert report.consistent is True
    by_name = {table.name: table for table in report.tables}
    assert set(by_name) == {"files", "chunks", "references"}
    assert by_name["files"].row_count == 1
    assert by_name["chunks"].row_count == 1
    assert by_name["chunks"].current_version >= 1
    assert by_name["chunks"].logical_bytes > 0
    assert by_name["chunks"].physical_bytes > 0
    assert by_name["chunks"].retained_version_count >= 1
    assert by_name["chunks"].oldest_version_at is not None
    assert by_name["chunks"].newest_version_at is not None
    assert by_name["chunks"].newest_version_at >= by_name["chunks"].oldest_version_at
    assert report.partition_physical_bytes > 0
    # No indexes are created by default, so nothing claims rows are indexed.
    assert by_name["chunks"].indexes == []
    # Every table reports non-negative counts and at least its own version.
    # Deliberately no logical-vs-physical inequality: Lance's total_bytes is
    # uncompressed-logical, so relating it to on-disk bytes would encode this
    # release's storage layout as if it were an invariant.
    for table in report.tables:
        assert table.row_count >= 0
        assert table.logical_bytes >= 0
        assert table.physical_bytes >= 0
        assert table.retained_version_count >= 1


def test_storage_stats_reports_indexes_after_ensure_indexes(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    store.ensure_indexes(project_id)

    report = store.storage_stats(project_id)

    chunks = next(table for table in report.tables if table.name == "chunks")
    assert {index.index_type for index in chunks.indexes} == {"FTS", "BTree"}
    fts = [index for index in chunks.indexes if index.index_type == "FTS"]
    # Native FTS indexes one field each; content and the identifier terms are
    # searched together with a MultiMatchQuery.
    assert {column for index in fts for column in index.columns} == {
        "content",
        "identifier_terms",
    }
    for index in fts:
        assert index.indexed_rows == 1
        assert index.unindexed_rows == 0
        assert index.size_bytes > 0


def test_storage_stats_works_without_the_references_table(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    store._partitions.pop(project_id, None)
    (store.directory / "projects" / project_id / "references.lance").rename(
        tmp_path / "references.lance.bak"
    )

    report = store.storage_stats(project_id)

    assert {table.name for table in report.tables} == {"files", "chunks"}
    assert report.consistent is True


def test_storage_stats_works_for_a_registered_project_without_a_partition(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model", state="pending")

    report = store.storage_stats(project.id)

    assert report.tables == []
    assert report.partition_physical_bytes == 0
    assert report.consistent is True
    assert report.partition_open_failed is False
    # Statistics are read-only: they must not materialize a partition.
    assert not (store.directory / "projects").exists()


def test_storage_stats_flags_a_partition_that_fails_to_open(tmp_path: Path) -> None:
    """A partition that exists but cannot be opened is not an unindexed project.

    The failure must be reported explicitly and the snapshot marked unusable,
    so a damaged store is distinguishable from one that was never indexed.
    """
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    store._partitions.pop(project_id, None)
    (store.directory / "projects" / project_id / "files.lance").rename(tmp_path / "files.lance.bak")

    report = store.storage_stats(project_id)

    assert report.tables == []
    assert report.partition_open_failed is True
    assert report.consistent is False
    assert report.partition_physical_bytes > 0


def test_physical_byte_accounting_does_not_follow_symlinks(tmp_path: Path) -> None:
    """Adding a symlink must change the reported bytes by exactly nothing.

    Comparing against the link target's size instead would only prove the
    partition is smaller than the target -- true here by accident of the
    fixture, and silently weaker as the partition grows.
    """
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    partition = store.directory / "projects" / project_id
    before = store.storage_stats(project_id)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x" * 1_000_000)
    (partition / "linked-file").symlink_to(outside)
    (partition / "chunks.lance" / "linked-dir").symlink_to(tmp_path, target_is_directory=True)

    after = store.storage_stats(project_id)

    assert after.partition_physical_bytes == before.partition_physical_bytes
    chunks_before = next(table for table in before.tables if table.name == "chunks")
    chunks_after = next(table for table in after.tables if table.name == "chunks")
    assert chunks_after.physical_bytes == chunks_before.physical_bytes


def test_concurrent_mutation_marks_storage_stats_inconsistent(tmp_path: Path) -> None:
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    original = LanceStore._table_storage_stats

    def mutate_mid_collection(
        self: LanceStore,
        table: object,
        name: str,
        *,
        physical_directory: Path,
    ) -> object:
        stats = original(table, name, physical_directory=physical_directory)
        if name == "chunks":
            store.upsert_file(stored_file(project_id, file_id="file-2"))
        return stats

    with patch.object(LanceStore, "_table_storage_stats", mutate_mid_collection):
        report = store.storage_stats(project_id)

    assert report.consistent is False


def test_mutation_during_the_physical_byte_walk_marks_stats_inconsistent(
    tmp_path: Path,
) -> None:
    """The byte walk must sit inside the consistency window too.

    A commit landing while the partition is being measured yields byte counts
    that disagree with the table statistics collected just before them. If the
    walk ran after the closing version snapshot, that disagreement would be
    reported as a consistent observation.
    """
    store, project_id, _ = _store_with_one_chunk(tmp_path)
    original = storage_module._directory_bytes
    partition = store.directory / "projects" / project_id

    def mutate_mid_walk(directory: Path) -> int:
        total = original(directory)
        if directory == partition:
            store.upsert_file(stored_file(project_id, file_id="file-2"))
        return total

    with patch.object(storage_module, "_directory_bytes", mutate_mid_walk):
        report = store.storage_stats(project_id)

    assert report.consistent is False


def test_registry_stats_report_the_project_registry(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model", state="pending")

    stats = store.registry_stats()

    assert stats.name == "projects"
    assert stats.row_count == 1
    assert stats.current_version >= 1
    assert stats.logical_bytes > 0
    assert stats.physical_bytes > 0
    assert stats.retained_version_count >= 1
    assert stats.oldest_version_at is not None
    assert stats.newest_version_at is not None


def test_overlap_warnings_detect_duplicate_and_nested_roots(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "src"
    other = tmp_path / "other"
    root.mkdir(parents=True)
    other.mkdir()

    def project(project_id: str, path: Path) -> ProjectInfo:
        return ProjectInfo(id=project_id, name=project_id, root=path)

    assert overlap_warnings([project("a", root), project("b", root)]) == [
        f"Projects 'a' and 'b' register the same root: {root}"
    ]
    nested_warnings = overlap_warnings([project("a", root), project("b", nested)])
    assert len(nested_warnings) == 1
    assert "nested inside" in nested_warnings[0] or "contains the root" in nested_warnings[0]
    assert overlap_warnings([project("a", root), project("b", other)]) == []


def test_overlapping_registration_detects_exact_nested_and_parent_roots(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "src"
    sibling = tmp_path / "sibling"
    root.mkdir(parents=True)
    sibling.mkdir()

    def project(project_id: str, path: Path) -> ProjectInfo:
        return ProjectInfo(id=project_id, name=project_id, root=path)

    parent = project("parent", root)
    child = project("child", nested)

    assert overlapping_registration([parent], root) == parent
    assert overlapping_registration([parent], nested) == parent
    assert overlapping_registration([child], root) == child
    assert overlapping_registration([parent, child], sibling) is None


def test_overlapping_registration_matches_case_insensitive_aliases(
    tmp_path: Path, case_insensitive_path_alias: Callable[[Path], Path]
) -> None:
    """Containment must agree with same_project_root on case-insensitive filesystems.

    same_project_root already treats /Repo and /repo as one directory; without
    a samefile-based containment check, a nested registration under a
    differently-cased spelling would silently skip overlap detection.
    """
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)
    alias = case_insensitive_path_alias(root)

    def project(project_id: str, path: Path) -> ProjectInfo:
        return ProjectInfo(id=project_id, name=project_id, root=path)

    registered = project("parent", alias)
    assert overlapping_registration([registered], nested) is registered
    assert overlapping_registration([project("child", nested)], alias) is not None


def test_worktree_warnings_share_a_git_common_directory(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    common = tmp_path / ".git"
    projects = [
        ProjectInfo(id="a", name="a", root=first),
        ProjectInfo(id="b", name="b", root=second),
    ]

    def fake_git(command: list[str], cwd: Path) -> str | None:
        assert command[:2] == ["git", "rev-parse"]
        if command[-1] == "--show-toplevel":
            return str(cwd.resolve())
        return str(common)

    warnings = worktree_warnings(projects, _run=fake_git)

    assert len(warnings) == 1
    assert "common directory" in warnings[0]
    assert "a" in warnings[0] and "b" in warnings[0]


def test_worktree_warnings_stay_silent_without_a_shared_common_directory(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    projects = [
        ProjectInfo(id="a", name="a", root=first),
        ProjectInfo(id="b", name="b", root=second),
    ]

    def distinct_commons(command: list[str], cwd: Path) -> str | None:
        if command[-1] == "--show-toplevel":
            return str(cwd.resolve())
        return str(cwd / ".git")

    assert worktree_warnings(projects, _run=distinct_commons) == []
    # Git being unavailable is a swallowed failure, not a hard error.
    assert worktree_warnings(projects, _run=lambda command, cwd: None) == []


def test_relative_git_common_directories_resolve_against_the_toplevel(
    tmp_path: Path,
) -> None:
    """A main checkout reports a relative --git-common-dir ('.git').

    A linked worktree reports the same directory in absolute form; the two must
    be recognized as one common directory.
    """
    main_root = tmp_path / "repo"
    worktree_root = tmp_path / "worktree"
    main_root.mkdir()
    worktree_root.mkdir()
    common = main_root / ".git"
    projects = [
        ProjectInfo(id="a", name="a", root=main_root),
        ProjectInfo(id="b", name="b", root=worktree_root),
    ]

    def fake_git(command: list[str], cwd: Path) -> str | None:
        if cwd == main_root and command[-1] == "--show-toplevel":
            return str(main_root)
        if cwd == worktree_root and command[-1] == "--show-toplevel":
            return str(worktree_root)
        if cwd == main_root:
            return ".git"
        return str(common)

    warnings = worktree_warnings(projects, _run=fake_git)

    assert len(warnings) == 1
    assert str(common) in warnings[0]


def test_relative_git_common_directories_resolve_against_the_registered_root(
    tmp_path: Path,
) -> None:
    """--git-common-dir is relative to the query cwd, which is the registered root.

    A registered root that is a subdirectory of a checkout reports '../.git';
    it must be joined against that root, not against the repository toplevel,
    or two subdirectory roots of one repository would look like different
    common directories.
    """
    main_root = tmp_path / "repo"
    worktree_root = tmp_path / "worktree"
    (main_root / "sub").mkdir(parents=True)
    (worktree_root / "sub").mkdir(parents=True)
    common = main_root / ".git"
    projects = [
        ProjectInfo(id="a", name="a", root=main_root / "sub"),
        ProjectInfo(id="b", name="b", root=worktree_root / "sub"),
    ]

    def fake_git(command: list[str], cwd: Path) -> str | None:
        if command[-1] == "--show-toplevel":
            return str(main_root) if cwd == projects[0].root else str(worktree_root)
        if cwd == projects[0].root:
            return "../.git"
        return str(common)

    warnings = worktree_warnings(projects, _run=fake_git)

    assert len(warnings) == 1
    assert str(common) in warnings[0]


def _chunk_table(project_id: str, file_id: str, count: int) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "chunk_id": f"{project_id}:{file_id}:chunk-{index}",
                "file_id": file_id,
                "path": "module.py",
                "language": "python",
                "kind": "function",
                "symbol": f"symbol_{index}",
                "qualified_symbol": f"symbol_{index}",
                "parent_symbol": None,
                "start_byte": 0,
                "end_byte": 1,
                "start_line": index + 1,
                "end_line": index + 1,
                "content": "pass",
                "identifier_terms": "symbol module py",
                "content_hash": "hash",
                "part_index": 0,
                "vector": [0.0, 0.0, 0.0, 1.0],
            }
            for index in range(count)
        ],
        schema=LanceStore.chunk_arrow_schema(4),
    )


def test_batched_chunk_merge_commits_every_file_in_the_predicate(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    files = pa.Table.from_pylist(
        [
            stored_file(project.id, file_id="file-a").model_dump(),
            stored_file(project.id, file_id="file-b").model_dump(),
        ],
        schema=LanceStore.file_arrow_schema(),
    )

    store.replace_files_from_arrow(
        project.id,
        files=files,
        chunk_batches=[
            (
                ["file-a", "file-b"],
                pa.concat_tables(
                    [_chunk_table(project.id, "file-a", 2), _chunk_table(project.id, "file-b", 2)]
                ),
            )
        ],
    )

    chunks = store.list_chunks([project.id])
    assert len(chunks) == 4
    assert {chunk.file_id for chunk in chunks} == {"file-a", "file-b"}


def test_an_entirely_empty_batch_deletes_its_files_previous_chunks(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    file_id = "file-1"
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id).model_dump()], schema=LanceStore.file_arrow_schema()
        ),
        chunk_batches=[([file_id], _chunk_table(project.id, file_id, 2))],
    )
    assert store.count_chunks([project.id]) == 2

    empty = pa.Table.from_batches([], schema=LanceStore.chunk_arrow_schema(4))
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id).model_dump()], schema=LanceStore.file_arrow_schema()
        ),
        chunk_batches=[([file_id], empty)],
    )

    assert store.count_chunks([project.id]) == 0


def test_a_zero_chunk_file_is_covered_by_a_sibling_batch_predicate(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    for file_id in ("file-a", "file-b"):
        store.replace_files_from_arrow(
            project.id,
            files=pa.Table.from_pylist(
                [stored_file(project.id, file_id=file_id).model_dump()],
                schema=LanceStore.file_arrow_schema(),
            ),
            chunk_batches=[([file_id], _chunk_table(project.id, file_id, 1))],
        )
    assert store.count_chunks([project.id]) == 2

    # file-b now extracts to nothing while file-a still has one chunk; they
    # share one batch, so the merge's predicate must retire file-b's rows.
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id, file_id="file-a").model_dump()],
            schema=LanceStore.file_arrow_schema(),
        ),
        chunk_batches=[
            (
                ["file-a", "file-b"],
                pa.concat_tables([_chunk_table(project.id, "file-a", 1)]),
            )
        ],
    )

    chunks = store.list_chunks([project.id])
    assert [chunk.file_id for chunk in chunks] == ["file-a"]


def test_removed_files_are_deleted_in_one_batched_predicate_per_table(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    for index in range(3):
        file_id = f"file-{index}"
        store.replace_files_from_arrow(
            project.id,
            files=pa.Table.from_pylist(
                [stored_file(project.id, file_id=file_id).model_dump()],
                schema=LanceStore.file_arrow_schema(),
            ),
            chunk_batches=[([file_id], _chunk_table(project.id, file_id, 1))],
        )
    assert store.count_chunks([project.id]) == 3

    original_delete = LanceTable.delete
    calls: dict[str, int] = {}

    def counting_delete(self: LanceTable, predicate: str) -> None:
        calls[self.name] = calls.get(self.name, 0) + 1
        original_delete(self, predicate)

    with patch.object(LanceTable, "delete", counting_delete):
        store.replace_files_from_arrow(
            project.id,
            files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
            chunk_batches=(),
            removed_file_ids=["file-0", "file-1", "file-2"],
        )

    assert calls == {"chunks": 1, "references": 1, "files": 1}
    assert store.count_chunks([project.id]) == 0
    assert store.list_files(project.id) == []


def test_replacement_ids_win_over_removal_ids(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    file_id = "file-a"
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id, file_id=file_id).model_dump()],
            schema=LanceStore.file_arrow_schema(),
        ),
        chunk_batches=[([file_id], _chunk_table(project.id, file_id, 1))],
    )

    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_pylist(
            [stored_file(project.id, file_id=file_id).model_dump()],
            schema=LanceStore.file_arrow_schema(),
        ),
        chunk_batches=[([file_id], _chunk_table(project.id, file_id, 2))],
        removed_file_ids=[file_id],
    )

    assert store.count_chunks([project.id]) == 2
    assert [record.file_id for record in store.list_files(project.id)] == [file_id]


def test_five_hundred_files_commit_in_eight_data_batches_per_table(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    file_ids = [f"file-{index}" for index in range(500)]
    chunk_batches = []
    reference_batches = []
    for offset in range(0, 500, 64):
        batch_ids = file_ids[offset : offset + 64]
        chunk_batches.append(
            (
                batch_ids,
                pa.concat_tables([_chunk_table(project.id, file_id, 1) for file_id in batch_ids]),
            )
        )
        reference_batches.append(
            (
                batch_ids,
                pa.concat_tables(
                    [
                        reference_table(
                            reference_record(
                                project.id,
                                file_id,
                                reference_id=f"{file_id}:reference",
                            )
                        )
                        for file_id in batch_ids
                    ]
                ),
            )
        )
    files = pa.Table.from_pylist(
        [stored_file(project.id, file_id=file_id).model_dump() for file_id in file_ids],
        schema=LanceStore.file_arrow_schema(),
    )

    original_merge = LanceTable.merge_insert
    merges: dict[str, int] = {}

    def counting_merge(self: LanceTable, key: str) -> object:
        merges[self.name] = merges.get(self.name, 0) + 1
        return original_merge(self, key)

    with patch.object(LanceTable, "merge_insert", counting_merge):
        store.replace_files_from_arrow(
            project.id,
            files=files,
            chunk_batches=chunk_batches,
            reference_batches=reference_batches,
        )

    assert merges["chunks"] == 8
    assert merges["references"] == 8
    assert merges["files"] == 1
    assert store.count_chunks([project.id]) == 500
    assert len(store.list_reference_records(project.id)) == 500


def test_upsert_project_skips_a_noop_registry_write(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    version_after_first = store.registry_stats().current_version

    store.upsert_project(project, model_id="test/model")

    assert store.registry_stats().current_version == version_after_first


def test_upsert_project_writes_when_the_state_changes(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model", state="pending")
    version_after_first = store.registry_stats().current_version

    store.upsert_project(project, model_id="test/model", state="ready")

    assert store.registry_stats().current_version == version_after_first + 1


def test_upsert_project_does_not_mistake_a_null_for_the_string_none(
    tmp_path: Path,
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    row = {
        "id": project.id,
        "name": project.name,
        "root": str(project.root.resolve()),
        "payload": project.model_dump_json(),
        "model_id": "test/model",
        "vector_dimension": 4,
        "schema_version": storage_module.SCHEMA_VERSION,
        "state": None,
        "updated_at": 0,
    }
    store._merge(store._projects, "id", [row])
    version_before = store.registry_stats().current_version

    store.upsert_project(project, model_id="test/model", state="None")

    # A stored null differs from the literal string "None": the typed
    # comparison must not treat it as a no-op and drop the state update.
    assert store.registry_stats().current_version == version_before + 1


def test_mark_project_state_skips_when_the_state_is_unchanged(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model", state="pending")
    store.mark_project_state(project.id, "error")
    version_after_error = store.registry_stats().current_version

    assert store.mark_project_state(project.id, "error") is True

    assert store.registry_stats().current_version == version_after_error


def test_an_incompatible_generation_is_marked_for_rebuild_not_rejected(
    tmp_path: Path,
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    store.replace_file(stored_file(project.id), _stored_chunks(project.id, 1))

    # A different embedding model is reconstructable: registration must mark
    # the project for rebuild, not raise INDEX_INCOMPATIBLE, and must leave
    # the stored generation fields describing the still-live partition.
    store.upsert_project(project, model_id="test/other")
    assert store.project_state(project.id) == "rebuild_required"
    assert store.incompatibility_reason(project.id, "test/other") is not None
    assert store.incompatibility_reason(project.id, "test/model") is None
    assert store.count_chunks([project.id]) == 1

    # An incompatible upsert cannot clobber rebuild_required, whatever state
    # it asks for: the partition still holds the old generation.
    store.upsert_project(project, model_id="test/other", state="indexing")
    assert store.project_state(project.id) == "rebuild_required"

    # A compatible upsert (same stored model) proceeds normally.
    store.upsert_project(project, model_id="test/model", state="indexing")
    assert store.project_state(project.id) == "indexing"


def test_incompatibility_reason_detects_model_dimension_and_schema_mismatches(
    tmp_path: Path,
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")

    assert store.incompatibility_reason(project.id, "test/model") is None
    assert "embedding model" in (store.incompatibility_reason(project.id, "other") or "")

    different_dimension = LanceStore(tmp_path / "lancedb-other", vector_dimension=8)
    row = {
        "id": project.id,
        "name": project.name,
        "root": str(project.root.resolve()),
        "payload": project.model_dump_json(),
        "model_id": "test/model",
        "vector_dimension": 4,
        "schema_version": storage_module.SCHEMA_VERSION,
        "state": "ready",
        "updated_at": 0,
    }
    different_dimension._merge(different_dimension._projects, "id", [row])
    assert "vector dimension" in (
        different_dimension.incompatibility_reason(project.id, "test/model") or ""
    )

    different_schema = LanceStore(tmp_path / "lancedb-other-schema", vector_dimension=4)
    row = {
        "id": project.id,
        "name": project.name,
        "root": str(project.root.resolve()),
        "payload": project.model_dump_json(),
        "model_id": "test/model",
        "vector_dimension": 4,
        "schema_version": storage_module.SCHEMA_VERSION - 1,
        "state": "ready",
        "updated_at": 0,
    }
    different_schema._merge(different_schema._projects, "id", [row])
    assert "index schema version" in (
        different_schema.incompatibility_reason(project.id, "test/model") or ""
    )

    # A pending registration has no live rows and is never incompatible.
    store.upsert_project(project, model_id="test/model", state="pending")
    assert store.incompatibility_reason(project.id, "other") is None


def test_delete_partition_preserves_registration_and_re_stamps_the_generation(
    tmp_path: Path,
) -> None:
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    store.replace_file(stored_file(project.id), _stored_chunks(project.id, 1))
    chunk_id = store.list_chunks([project.id])[0].chunk_id
    assert store._existing_tables(project.id) is not None

    removed = store.delete_partition(project.id, model_id="test/other")

    assert removed is True
    assert store.list_projects() == [project]
    assert store.incompatibility_reason(project.id, "test/other") is None
    assert store.project_state(project.id) == "indexing"
    assert store.count_chunks([project.id]) == 0
    assert store.get_chunk(chunk_id) is None
    assert store._existing_tables(project.id) is None
    assert (root / ".ci-mcp" / "project.toml").exists()
    assert not (store.directory / "projects" / project.id).exists()


def test_merge_semantics_probe_passes_on_the_installed_lancedb() -> None:
    # A future lancedb that regresses when_not_matched_by_source_delete to the
    # all-or-nothing gate behavior fails this probe, and with it every batched
    # commit, instead of silently deleting unrelated rows.
    assert storage_module._probe_batched_merge_semantics() is True
    assert storage_module._batched_merge_semantics_ok() is True


def test_merge_semantics_probe_rejects_all_or_nothing_gate_semantics() -> None:
    original_merge = LanceTable.merge_insert

    def gated_merge(self: LanceTable, key: str) -> LanceMergeInsertBuilder:
        builder = original_merge(self, key)

        def gate_only(condition: str | Expr | None = None) -> LanceMergeInsertBuilder:
            builder._when_not_matched_by_source_delete = True
            builder._when_not_matched_by_source_condition = None
            builder._when_not_matched_by_source_condition_expr = None
            return builder

        builder.when_not_matched_by_source_delete = gate_only  # type: ignore[method-assign]
        return builder

    with patch.object(LanceTable, "merge_insert", gated_merge):
        assert storage_module._probe_batched_merge_semantics() is False


def test_a_failed_probe_refuses_the_batched_commit(tmp_path: Path) -> None:
    """The semantics probe gates the commit, not store construction."""
    store = LanceStore(tmp_path / "lancedb", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    files = pa.Table.from_pylist([], schema=LanceStore.file_arrow_schema())

    with (
        patch.object(storage_module, "_batched_merge_semantics_ok", return_value=False),
        pytest.raises(storage_module.CodeIndexingError) as excinfo,
    ):
        store.replace_files_from_arrow(
            project.id,
            files=files,
            chunk_batches=[],
        )

    assert excinfo.value.code == storage_module.ErrorCode.UNSUPPORTED_RUNTIME
    assert store.list_chunks([project.id]) == []
