"""Staging, commit-rollback, and startup-recovery coverage for Task 4."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lancedb
import pyarrow as pa
import pytest
from filelock import FileLock

from code_indexing_mcp import application
from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.embedding import pack_vector
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import StoredChunk, StoredFile
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.staging import (
    CHUNKS_NAME,
    JOURNAL_FORMAT_VERSION,
    JOURNAL_NAME,
    MAX_RECOVERY_ATTEMPTS,
    PHASE_COMMITTING,
    PHASE_STAGING,
    REFERENCES_NAME,
    ChunkRow,
    ReferenceRow,
    StagingJob,
    recover_staged_commits,
)
from code_indexing_mcp.storage import LanceStore, TableVersions


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


def chunk_row(
    project_id: str, file_id: str, vector: list[float], *, chunk_id: str = "chunk-1"
) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk_id,
        file_id=file_id,
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
        identifier_terms="answer main py",
        part_index=0,
        vector=pack_vector(vector),
    )


def reference_row(
    project_id: str,
    file_id: str,
    *,
    reference_id: str = "reference-1",
    target_name: str = "answer",
) -> ReferenceRow:
    return ReferenceRow(
        reference_id=reference_id,
        record_kind="reference",
        file_id=file_id,
        project_id=project_id,
        path="main.py",
        language="python",
        kind="call",
        source_qualified_symbol="caller",
        written_name=target_name,
        target_name=target_name,
        module_path=None,
        imported_name=None,
        alias=None,
        receiver_text=None,
        start_byte=0,
        end_byte=len(target_name),
        start_line=1,
        end_line=1,
        shape_json='{"positional_count": 0}',
        content_hash="hash",
        schema_version=1,
    )


def make_job(
    tmp_path: Path, store: LanceStore, project_id: str, job_id: str = "job-1"
) -> StagingJob:
    job = StagingJob(
        tmp_path / "staging",
        project_id,
        file_schema=LanceStore.file_arrow_schema(),
        chunk_schema=LanceStore.chunk_arrow_schema(store.vector_dimension, store.vector_dtype),
        reference_schema=LanceStore.reference_arrow_schema(),
        job_id=job_id,
    )
    job.begin()
    return job


def test_staged_vectors_carry_the_store_storage_dtype(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks([chunk_row("project-1", "file-1", [1.5, 2.5, 3.5, 4.5])])
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    table = pa.ipc.open_file(job.directory / CHUNKS_NAME).read_all()

    vector_field = table.schema.field("vector")
    # float16 is the storage default: staged float32 worker bytes are cast
    # while building the Arrow batch, never materialized as Python floats.
    assert vector_field.type == pa.list_(pa.float16(), 4)
    # 1.5/2.5/3.5/4.5 are exact in float16, so the round trip is lossless.
    assert table.column("vector")[0].as_py() == [1.5, 2.5, 3.5, 4.5]

    journal = json.loads((job.directory / JOURNAL_NAME).read_text())
    assert journal["phase"] == PHASE_COMMITTING
    assert journal["files_version"] == 1
    assert journal["chunks_version"] == 1
    assert journal["references_version"] == 1
    job.discard()


def test_staged_vectors_stay_float32_with_the_explicit_opt_out(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4, vector_storage="float32")
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks([chunk_row("project-1", "file-1", [1.5, 2.5, 3.5, 4.5])])
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    table = pa.ipc.open_file(job.directory / CHUNKS_NAME).read_all()

    assert table.schema.field("vector").type == pa.list_(pa.float32(), 4)
    assert table.column("vector")[0].as_py() == [1.5, 2.5, 3.5, 4.5]
    job.discard()


def test_staged_references_stream_to_arrow_without_object_row_materialization(
    tmp_path: Path,
) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")

    arrow = SimpleNamespace(
        array=pa.array,
        record_batch=pa.record_batch,
        RecordBatch=SimpleNamespace(
            from_pylist=lambda *_: (_ for _ in ()).throw(
                AssertionError("reference rows must stream as Arrow columns")
            )
        ),
    )
    with patch("code_indexing_mcp.staging.pa", arrow):
        job.stage_references([reference_row("project-1", "file-1")])
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    table = pa.ipc.open_file(job.directory / REFERENCES_NAME).read_all()

    assert table.column("reference_id")[0].as_py() == "reference-1"
    assert table.column("shape_json")[0].as_py() == '{"positional_count": 0}'
    job.discard()


def test_staging_references_rejects_mixed_file_batches(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")

    with pytest.raises(ValueError, match="one file"):
        job.stage_references(
            [
                reference_row("project-1", "file-a", reference_id="a"),
                reference_row("project-1", "file-b", reference_id="b"),
            ]
        )


def test_reference_batches_combine_complete_file_groups(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_references([reference_row("project-1", "file-a", reference_id="a-1")])
    job.stage_references([reference_row("project-1", "file-a", reference_id="a-2")])
    job.stage_references([reference_row("project-1", "file-b", reference_id="b-1")])
    job.mark_references_replaced("file-a")
    job.mark_references_replaced("file-b")
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    batches = list(job.iter_reference_batches())

    assert [file_ids for file_ids, _ in batches] == [["file-a", "file-b"]]
    assert batches[0][1].column("reference_id").to_pylist() == ["a-1", "a-2", "b-1"]


def test_chunk_batches_never_split_a_file_across_batches(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks([chunk_row("project-1", "file-a", [1.0, 2.0, 3.0, 4.0], chunk_id="a-1")])
    job.stage_chunks([chunk_row("project-1", "file-a", [2.0, 2.0, 3.0, 4.0], chunk_id="a-2")])
    job.stage_chunks([chunk_row("project-1", "file-b", [3.0, 2.0, 3.0, 4.0], chunk_id="b-1")])
    job.mark_replaced("file-a")
    job.mark_replaced("file-b")
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    batches = list(job.iter_chunk_batches(max_files=1))

    # The file-count bound splits the batch, but never between a file's rows.
    assert [file_ids for file_ids, _ in batches] == [["file-a"], ["file-b"]]
    assert batches[0][1].column("chunk_id").to_pylist() == ["a-1", "a-2"]


def test_commit_batches_respect_the_file_and_byte_limits(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    for index in range(5):
        job.stage_chunks(
            [
                chunk_row(
                    "project-1",
                    f"file-{index}",
                    [1.0, 2.0, 3.0, 4.0],
                    chunk_id=f"chunk-{index}",
                )
            ]
        )
        job.mark_replaced(f"file-{index}")
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    batches = list(job.iter_chunk_batches(max_files=2))

    assert [file_ids for file_ids, _ in batches] == [
        ["file-0", "file-1"],
        ["file-2", "file-3"],
        ["file-4"],
    ]
    assert [table.num_rows for _, table in batches] == [2, 2, 1]


def test_commit_batches_respect_the_byte_bound(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    for index in range(3):
        row = chunk_row(
            "project-1", f"file-{index}", [1.0, 2.0, 3.0, 4.0], chunk_id=f"chunk-{index}"
        )
        job.stage_chunks([replace(row, content="x" * 2048)])
        job.mark_replaced(f"file-{index}")
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    batches = list(job.iter_chunk_batches(max_files=10, max_rows=100_000, max_bytes=4000))

    # The file-count bound would allow one batch of three; the byte bound
    # binds instead and gives each file its own batch, never splitting a
    # file's rows.
    assert [file_ids for file_ids, _ in batches] == [["file-0"], ["file-1"], ["file-2"]]
    assert all(table.nbytes <= 4000 for _, table in batches)


def test_commit_batches_cover_zero_row_files_in_the_affected_predicate(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks([chunk_row("project-1", "file-a", [1.0, 2.0, 3.0, 4.0], chunk_id="a-1")])
    job.mark_replaced("file-a")
    job.mark_replaced("file-b")
    job.mark_replaced("file-c")
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    batches = list(job.iter_chunk_batches())

    # Every replaced file lands in exactly one batch's predicate, whether or
    # not it has staged rows, so the commit removes its previous chunks.
    assert sorted(file_id for file_ids, _ in batches for file_id in file_ids) == [
        "file-a",
        "file-b",
        "file-c",
    ]
    assert sum(table.num_rows for _, table in batches) == 1


def test_a_single_file_may_exceed_the_row_and_byte_bounds(tmp_path: Path) -> None:
    """A file's rows are never split, so one file alone may exceed the bounds."""
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks(
        [
            chunk_row("project-1", "file-a", [1.0, 2.0, 3.0, 4.0], chunk_id=f"a-{index}")
            for index in range(3)
        ]
    )
    job.mark_replaced("file-a")
    job.begin_commit(TableVersions(files=1, chunks=1, references=1))

    batches = list(job.iter_chunk_batches(max_rows=2, max_bytes=1))

    assert [file_ids for file_ids, _ in batches] == [["file-a"]]
    assert batches[0][1].num_rows == 3


def test_staging_references_rejects_non_contiguous_file_batches(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_references([reference_row("project-1", "file-a", reference_id="a-1")])
    job.stage_references([reference_row("project-1", "file-b", reference_id="b-1")])

    with pytest.raises(ValueError, match="contiguous"):
        job.stage_references([reference_row("project-1", "file-a", reference_id="a-2")])


def test_staging_chunks_rejects_mixed_file_batches(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")

    with pytest.raises(ValueError, match="one file"):
        job.stage_chunks(
            [
                chunk_row("project-1", "file-a", [1.0, 2.0, 3.0, 4.0], chunk_id="chunk-a"),
                chunk_row("project-1", "file-b", [4.0, 3.0, 2.0, 1.0], chunk_id="chunk-b"),
            ]
        )


def test_staging_chunks_rejects_non_contiguous_file_batches(tmp_path: Path) -> None:
    """The commit's batch grouping needs one file's rows contiguous.

    A non-contiguous repeat would split the file across two batches whose
    later predicate would silently delete the earlier group's rows, so it
    must be refused at staging time rather than corrupt the commit.
    """
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    job = make_job(tmp_path, store, "project-1")
    job.stage_chunks([chunk_row("project-1", "file-a", [1.0, 2.0, 3.0, 4.0], chunk_id="a-1")])
    job.stage_chunks([chunk_row("project-1", "file-b", [4.0, 3.0, 2.0, 1.0], chunk_id="b-1")])

    with pytest.raises(ValueError, match="contiguous"):
        job.stage_chunks([chunk_row("project-1", "file-a", [1.0, 2.0, 3.0, 4.0], chunk_id="a-2")])


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
    tables = store._existing_tables(project.id)
    assert tables is not None
    stored = tables.chunks.search().select(["identifier_terms", "vector"]).to_list()[0]
    assert stored["identifier_terms"]
    assert len(stored["vector"]) == 4

    fetched = store.get_chunk(chunks[0].chunk_id)
    assert fetched is not None
    assert fetched.chunk_id == chunks[0].chunk_id
    assert fetched.content == chunks[0].content


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
                chunk_id=f"{project.id}:chunk-1",
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
                identifier_terms="answer main py",
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
        chunk_batches=[(["file-1"], empty)],
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
                raise CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "model went away")
            return super().embed_passages(texts)

    (root / "main.py").write_text("def changed():\n    return 1\n")
    failing, _ = make_indexer(tmp_path, FailingEmbedder())

    with pytest.raises(CodeIndexingError) as caught:
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


def test_a_failed_commit_restores_reference_rows_with_files_and_chunks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    file_id = store.list_files(project.id)[0].file_id
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[
            (
                [file_id],
                pa.Table.from_pylist(
                    [reference_row(project.id, file_id).__dict__],
                    schema=LanceStore.reference_arrow_schema(),
                ),
            )
        ],
    )
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])
    references_before = store.list_reference_records(project.id)
    job = make_job(tmp_path, store, project.id)
    job.stage_references(
        [reference_row(project.id, file_id, reference_id="replacement", target_name="renamed")]
    )
    job.mark_references_replaced(file_id)
    original = LanceStore.replace_files_from_arrow

    def apply_then_crash(self: LanceStore, project_id: str, **kwargs: object) -> None:
        original(self, project_id, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("simulated crash after the live writes")

    with (
        patch.object(LanceStore, "replace_files_from_arrow", apply_then_crash),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        indexer._commit_staged(project, job, errors=[])

    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    assert store.list_reference_records(project.id) == references_before


def test_a_commit_whose_rollback_fails_keeps_its_journal_for_recovery(
    tmp_path: Path,
) -> None:
    """Commit fails, then the rollback fails too: recovery must still be possible.

    The staged directory is the only record of the versions the live tables
    have to return to, so discarding it here would strand a half-committed
    project. It has to survive in ``committing`` for the next startup.
    """
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

    def restore_fails(self: LanceStore, *args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated rollback failure")

    with (
        patch.object(LanceStore, "replace_files_from_arrow", apply_then_crash),
        patch.object(LanceStore, "restore_versions", restore_fails),
        # The original commit failure is reported, not the rollback failure.
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        indexer.index(project)

    journals = sorted((tmp_path / "staging").glob(f"*/*/{JOURNAL_NAME}"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["phase"] == PHASE_COMMITTING

    # With the journal intact, startup recovery still returns both tables.
    assert recover_staged_commits(tmp_path / "staging", store) == 1
    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    assert list((tmp_path / "staging").glob("*/*/")) == []


def test_a_crash_mid_commit_is_rolled_back_by_startup_recovery(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])
    file_id = files_before[0].file_id
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[
            (
                [file_id],
                pa.Table.from_pylist(
                    [reference_row(project.id, file_id).__dict__],
                    schema=LanceStore.reference_arrow_schema(),
                ),
            )
        ],
    )
    references_before = store.list_reference_records(project.id)

    # Stage a replacement, record the versions, and then "crash" after all
    # three live tables mutate but before rollback code runs.
    job = make_job(tmp_path, store, project.id)
    record = files_before[0].model_copy(update={"mtime_ns": 2})
    job.stage_file(record)
    job.stage_chunks([chunk_row(project.id, file_id, [9.0, 9.0, 9.0, 9.0])])
    job.stage_references(
        [reference_row(project.id, file_id, reference_id="replacement", target_name="renamed")]
    )
    job.mark_replaced(file_id)
    job.mark_references_replaced(file_id)
    versions = store.table_versions(project.id)
    job.begin_commit(versions)
    store.replace_files_from_arrow(
        project.id,
        files=job.files_table(),
        chunk_batches=job.iter_chunk_batches(),
        reference_batches=job.iter_reference_batches(),
    )
    assert store.list_files(project.id) != files_before
    assert store.count_chunks([project.id]) == 1
    assert store.list_chunks([project.id])[0].content != chunks_before[0].content
    assert store.list_reference_records(project.id) != references_before

    recovered = recover_staged_commits(tmp_path / "staging", store)

    assert recovered == 1
    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    assert store.list_reference_records(project.id) == references_before
    assert list((tmp_path / "staging").glob("*/*/")) == []


def test_recovery_of_a_rebuild_journal_discards_a_deleted_partition(
    tmp_path: Path,
) -> None:
    """A crash mid-rebuild names versions of a partition the rebuild deleted.

    Restoring those versions is impossible by design (the old generation was
    intentionally discarded), so recovery must retire the journal and leave
    the registered project empty and re-indexable -- never error or spin.
    """
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    file_id = "file-1"
    record = StoredFile(
        file_id=file_id,
        project_id=project.id,
        path="main.py",
        language="python",
        size=4,
        mtime_ns=1,
        content_hash="hash",
        indexed_at=1,
    )
    store.replace_file(record, _staged_chunks_for_recovery(project.id, file_id))

    # "Crash" after the rebuild deleted the partition but before the new
    # generation committed: the journal names versions that no longer exist.
    store.delete_partition(project.id, model_id="test/model")
    directory = tmp_path / "staging" / project.id / "rebuild-job"
    directory.mkdir(parents=True)
    (directory / JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": JOURNAL_FORMAT_VERSION,
                "job_id": "rebuild-job",
                "project_id": project.id,
                "phase": PHASE_COMMITTING,
                "files_version": 7,
                "chunks_version": 9,
                "references_version": 11,
                "replace_file_ids": [],
                "removed_file_ids": [],
            }
        )
    )

    recovered = recover_staged_commits(tmp_path / "staging", store)

    assert recovered == 0
    assert store.list_projects() == [project]
    assert store.list_chunks([project.id]) == []
    assert store.count_chunks([project.id]) == 0
    assert list((tmp_path / "staging").glob("*/*/")) == []


def _staged_chunks_for_recovery(project_id: str, file_id: str) -> list[StoredChunk]:
    return [
        StoredChunk(
            chunk_id=f"{project_id}:chunk-1",
            file_id=file_id,
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
            identifier_terms="answer main py",
            part_index=0,
            vector=[0.0, 0.0, 0.0, 1.0],
        )
    ]


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
        "version": JOURNAL_FORMAT_VERSION,
        "job_id": "job-1",
        "project_id": project.id,
        "phase": PHASE_COMMITTING,
        "files_version": versions.files,
        "chunks_version": versions.chunks,
        "references_version": versions.references,
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


def _committing_journal(directory: Path, project_id: str, versions: TableVersions) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / JOURNAL_NAME
    path.write_text(
        json.dumps(
            {
                "version": JOURNAL_FORMAT_VERSION,
                "job_id": directory.name,
                "project_id": project_id,
                "phase": PHASE_COMMITTING,
                "files_version": versions.files,
                "chunks_version": versions.chunks,
                "references_version": versions.references,
                "replace_file_ids": [],
                "removed_file_ids": [],
            }
        )
    )
    return path


def test_upgrade_recovery_rolls_back_a_legacy_two_table_journal(tmp_path: Path) -> None:
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    store.upsert_project(project, model_id="test/model")
    database = lancedb.connect(store.directory / "projects" / project.id)
    files = database.create_table("files", schema=LanceStore.file_arrow_schema())
    chunks = database.create_table("chunks", schema=LanceStore.chunk_arrow_schema(4))
    original_file = StoredFile(
        file_id="file-1",
        project_id=project.id,
        path="main.py",
        language="python",
        size=4,
        mtime_ns=1,
        content_hash="old",
        indexed_at=1,
    )
    original_chunk = {
        **chunk_row(project.id, "file-1", [1.0, 2.0, 3.0, 4.0]).__dict__,
        "vector": [1.0, 2.0, 3.0, 4.0],
    }
    files.add([original_file.model_dump()])
    chunks.add([original_chunk])
    versions = TableVersions(files=files.version, chunks=chunks.version, references=0)
    files.delete("file_id = 'file-1'")
    files.add([original_file.model_copy(update={"mtime_ns": 2}).model_dump()])
    chunks.delete("file_id = 'file-1'")
    chunks.add([{**original_chunk, "content": "new"}])
    directory = tmp_path / "staging" / project.id / "legacy-job"
    directory.mkdir(parents=True)
    (directory / JOURNAL_NAME).write_text(
        json.dumps(
            {
                "version": 1,
                "job_id": "legacy-job",
                "project_id": project.id,
                "phase": PHASE_COMMITTING,
                "files_version": versions.files,
                "chunks_version": versions.chunks,
            }
        )
    )

    assert recover_staged_commits(tmp_path / "staging", store) == 1
    assert store.list_files(project.id) == [original_file]
    assert store.list_chunks([project.id])[0].content == original_chunk["content"]
    assert not (store.directory / "projects" / project.id / "references.lance").exists()
    assert not directory.exists()


def test_recovery_for_a_removed_project_does_not_recreate_its_partition(
    tmp_path: Path,
) -> None:
    """A journal outliving its project must not materialise empty tables.

    Going through the create-on-write path would build a fresh partition whose
    versions can never match the journal, so the rollback would fail on every
    startup for a project that no longer exists.
    """
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    directory = tmp_path / "staging" / "ghost-project" / "job-1"
    _committing_journal(directory, "ghost-project", TableVersions(files=3, chunks=3, references=3))

    assert recover_staged_commits(tmp_path / "staging", store) == 0

    assert not (tmp_path / "data" / "projects" / "ghost-project").exists()
    assert list((tmp_path / "staging").glob("*/*/")) == []


def test_a_rollback_that_never_succeeds_is_abandoned_after_bounded_retries(
    tmp_path: Path,
) -> None:
    """Recovery runs on the startup path, so it must not retry a lost cause forever."""
    root = tmp_path / "repo"
    project = make_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    versions = store.table_versions(project.id)
    directory = tmp_path / "staging" / project.id / "job-1"
    journal_path = _committing_journal(directory, project.id, versions)

    def restore_fails(self: LanceStore, *args: object, **kwargs: object) -> bool:
        raise RuntimeError("simulated permanent rollback failure")

    with patch.object(LanceStore, "restore_versions", restore_fails):
        for attempt in range(1, MAX_RECOVERY_ATTEMPTS):
            assert recover_staged_commits(tmp_path / "staging", store) == 0
            # Still retryable: the journal survives with the attempt recorded.
            assert json.loads(journal_path.read_text())["recovery_attempts"] == attempt

        assert recover_staged_commits(tmp_path / "staging", store) == 0

    # Given up on: the journal is gone and the project is flagged for re-index.
    assert list((tmp_path / "staging").glob("*/*/")) == []
    assert store.project_state(project.id) == "error"
    # A later startup has nothing left to retry.
    assert recover_staged_commits(tmp_path / "staging", store) == 0


def test_startup_recovery_does_not_wait_out_an_active_index_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building an Application must not block for the length of an index run.

    The global index lock is held from scan to commit, and every CLI call and
    daemon start builds an Application. Recovery yields instead of waiting.
    """
    monkeypatch.setattr(application, "RECOVERY_LOCK_TIMEOUT_SECONDS", 0.05)
    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    root = tmp_path / "repo"
    make_project(root)
    directory = paths.data / "staging" / "ghost-project" / "job-1"
    journal_path = _committing_journal(
        directory, "ghost-project", TableVersions(files=1, chunks=1, references=1)
    )
    lock_directory = paths.data / "locks"
    lock_directory.mkdir(parents=True, exist_ok=True)

    with FileLock(lock_directory / "index-global.lock"):
        started = time.monotonic()
        Application(paths, embedder=RecordingEmbedder(), cwd=root)
        elapsed = time.monotonic() - started

    assert elapsed < 2
    # Recovery was skipped rather than run against a live commit.
    assert journal_path.exists()

    # Once the run finishes, the next start picks the journal up.
    Application(paths, embedder=RecordingEmbedder(), cwd=root)
    assert not journal_path.exists()


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
    # The staging process died, so its handles are closed and only the journal
    # and the unfinished .tmp payloads survive. Leaving the writers open here
    # would not model any real crash, and Windows would refuse to unlink them.
    job._close_writers(finalize=False)

    recovered = recover_staged_commits(tmp_path / "staging", store)

    assert recovered == 0
    assert store.list_chunks([project.id]) == chunks_before
    assert list((tmp_path / "staging").glob("*/*/")) == []
