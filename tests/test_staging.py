"""Staging, commit-rollback, and startup-recovery coverage for Task 4."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest

from incode_mcp.embedding import pack_vector
from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.extractor import TreeSitterExtractor
from incode_mcp.indexing import Indexer
from incode_mcp.models import StoredChunk, StoredFile
from incode_mcp.projects import initialize_project
from incode_mcp.scanner import SourceScanner
from incode_mcp.staging import (
    CHUNKS_NAME,
    JOURNAL_NAME,
    PHASE_COMMITTING,
    PHASE_STAGING,
    ChunkRow,
    StagingJob,
    recover_staged_commits,
)
from incode_mcp.storage import LanceStore, TableVersions


class RecordingEmbedder:
    model_id = "test/code"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 7), 1.0, 2.0, 3.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 2.0, 3.0]


def make_indexer(tmp_path: Path, embedder: RecordingEmbedder) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    return (
        Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=tmp_path / "locks",
        ),
        store,
    )


def make_project(root: Path, source: str = "def answer():\n    return 42\n"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(source)
    return initialize_project(root)


def chunk_row(project_id: str, file_id: str, vector: list[float]) -> ChunkRow:
    return ChunkRow(
        chunk_id="chunk-1",
        file_id=file_id,
        project_id=project_id,
        path="main.py",
        language="python",
        kind="function",
        symbol="answer",
        qualified_symbol="answer",
        parent_symbol=None,
        start_byte=0,
        end_byte=26,
        start_line=1,
        end_line=2,
        content="def answer():\n    return 42\n",
        embedding_text="embedding",
        search_text="search",
        content_hash="hash",
        part_index=0,
        vector=pack_vector(vector),
    )


def make_job(
    tmp_path: Path, store: LanceStore, project_id: str, job_id: str = "job-1"
) -> StagingJob:
    job = StagingJob(
        tmp_path / "staging",
        project_id,
        file_schema=LanceStore.file_arrow_schema(),
        chunk_schema=LanceStore.chunk_arrow_schema(store.vector_dimension),
        job_id=job_id,
    )
    job.begin()
    return job


def test_staged_vectors_stay_packed_float32_arrow_arrays(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks([chunk_row("project-1", "file-1", [1.5, 2.5, 3.5, 4.5])])
    job.begin_commit(TableVersions(files=1, chunks=1))

    table = pa.ipc.open_file(job.directory / CHUNKS_NAME).read_all()

    vector_field = table.schema.field("vector")
    assert vector_field.type == pa.list_(pa.float32(), 4)
    # 1.5/2.5/3.5/4.5 are exact in float32, so the round trip is lossless.
    assert table.column("vector")[0].as_py() == [1.5, 2.5, 3.5, 4.5]

    journal = json.loads((job.directory / JOURNAL_NAME).read_text())
    assert journal["phase"] == PHASE_COMMITTING
    assert journal["files_version"] == 1
    assert journal["chunks_version"] == 1
    job.discard()


def test_the_write_path_never_dumps_a_stored_chunk(tmp_path: Path) -> None:
    """The triple materialization this task removes was model_dump() driven."""
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    with patch.object(
        StoredChunk, "model_dump", side_effect=AssertionError("list-of-floats path used")
    ):
        report = indexer.index(project)

    assert report.errors == []
    assert len(store.list_chunks([project.id])) == 1


def test_staged_chunks_round_trip_through_the_store(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    indexer.index(project)
    chunks = store.list_chunks([project.id])

    assert len(chunks) == 1
    expected = len(chunks[0].embedding_text) % 7
    assert chunks[0].vector == [float(expected), 1.0, 2.0, 3.0]
    assert store.get_chunk(chunks[0].chunk_id) == chunks[0]


def test_a_file_that_now_extracts_no_chunks_has_its_old_chunks_deleted(
    tmp_path: Path,
) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    record = StoredFile(
        file_id="file-1",
        project_id=project.id,
        path="main.py",
        language="python",
        size=4,
        mtime_ns=1,
        content_hash="hash",
        indexed_at=time.time_ns(),
    )
    store.replace_file(
        record,
        [
            StoredChunk(
                chunk_id="chunk-1",
                file_id="file-1",
                project_id=project.id,
                path="main.py",
                language="python",
                kind="function",
                symbol="answer",
                qualified_symbol="answer",
                parent_symbol=None,
                start_byte=0,
                end_byte=26,
                start_line=1,
                end_line=2,
                content="def answer():\n    return 42\n",
                embedding_text="embedding",
                search_text="search",
                content_hash="hash",
                part_index=0,
                vector=[0.0, 0.0, 0.0, 1.0],
            )
        ],
    )
    assert store.count_chunks([project.id]) == 1

    files = pa.RecordBatch.from_pylist([record.model_dump()], schema=LanceStore.file_arrow_schema())
    empty = pa.Table.from_batches([], schema=LanceStore.chunk_arrow_schema(4))
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_batches([files]),
        chunk_groups=[("file-1", empty)],
    )

    assert store.count_chunks([project.id]) == 0
    assert store.list_files(project.id) == [record]


def test_cancellation_during_staging_leaves_the_live_tables_unchanged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    (root / "other.py").write_text("value = 1\n")
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])

    class FailingEmbedder(RecordingEmbedder):
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            if any("changed" in text for text in texts):
                raise IncodeError(ErrorCode.MODEL_UNAVAILABLE, "model went away")
            return super().embed_passages(texts)

    (root / "main.py").write_text("def changed():\n    return 1\n")
    failing, _ = make_indexer(tmp_path, FailingEmbedder())

    with pytest.raises(IncodeError) as caught:
        failing.index(project)

    assert caught.value.code is ErrorCode.MODEL_UNAVAILABLE
    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    # The abandoned job leaves nothing behind for startup recovery to find.
    assert list((tmp_path / "staging").glob("*/*/")) == []


def test_a_failed_commit_restores_both_table_versions(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])

    (root / "main.py").write_text("def renamed():\n    return 43\n")
    original = LanceStore.replace_files_from_arrow

    def apply_then_crash(self: LanceStore, project_id: str, **kwargs: object) -> None:
        original(self, project_id, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("simulated crash after the live writes")

    with (
        patch.object(LanceStore, "replace_files_from_arrow", apply_then_crash),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        indexer.index(project)

    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    # The rollback consumed the staged directory and the project is marked.
    assert list((tmp_path / "staging").glob("*/*/")) == []
    assert store.project_state(project.id) == "error"

    recovered = indexer.index(project)
    assert recovered.errors == []
    assert {chunk.qualified_symbol for chunk in store.list_chunks([project.id])} == {"renamed"}


def test_a_crash_mid_commit_is_rolled_back_by_startup_recovery(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])
    file_id = files_before[0].file_id

    # Stage a replacement, record the versions, and then "crash": the chunks
    # merge lands but the run dies before the files merge and before any
    # rollback code runs.
    job = make_job(tmp_path, store, project.id)
    record = files_before[0].model_copy(update={"mtime_ns": 2})
    job.stage_file(record)
    job.stage_chunks([chunk_row(project.id, file_id, [9.0, 9.0, 9.0, 9.0])])
    job.mark_replaced(file_id)
    versions = store.table_versions(project.id)
    job.begin_commit(versions)
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_groups=job.iter_chunk_groups(),
    )
    assert store.count_chunks([project.id]) == 1
    assert store.list_chunks([project.id])[0].content != chunks_before[0].content

    recovered = recover_staged_commits(tmp_path / "staging", store)

    assert recovered == 1
    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    assert list((tmp_path / "staging").glob("*/*/")) == []


def test_repeated_recovery_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])
    versions = store.table_versions(project.id)

    journal_dir = tmp_path / "staging" / project.id / "job-1"
    journal_dir.mkdir(parents=True)
    journal = {
        "version": 1,
        "job_id": "job-1",
        "project_id": project.id,
        "phase": PHASE_COMMITTING,
        "files_version": versions.files,
        "chunks_version": versions.chunks,
        "replace_file_ids": [],
        "removed_file_ids": [],
    }
    (journal_dir / JOURNAL_NAME).write_text(json.dumps(journal))

    for _ in range(2):
        journal_dir.mkdir(parents=True, exist_ok=True)
        (journal_dir / JOURNAL_NAME).write_text(json.dumps(journal))
        assert recover_staged_commits(tmp_path / "staging", store) == 1
        assert store.list_files(project.id) == files_before
        assert store.list_chunks([project.id]) == chunks_before
        assert list((tmp_path / "staging").glob("*/*/")) == []

    assert recover_staged_commits(tmp_path / "staging", store) == 0


def test_staging_phase_leftovers_are_removed_without_touching_live_tables(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    chunks_before = store.list_chunks([project.id])

    # A job abandoned while still staging never reached the live tables.
    job = make_job(tmp_path, store, project.id)
    job.stage_chunks([chunk_row(project.id, "file-1", [1.0, 1.0, 1.0, 1.0])])
    journal = json.loads((job.directory / JOURNAL_NAME).read_text())
    assert journal["phase"] == PHASE_STAGING

    recovered = recover_staged_commits(tmp_path / "staging", store)

    assert recovered == 0
    assert store.list_chunks([project.id]) == chunks_before
    assert list((tmp_path / "staging").glob("*/*/")) == []
