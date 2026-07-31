"""Staging, commit-rollback, and startup-recovery coverage for Task 4."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

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
    JOURNAL_NAME,
    MAX_RECOVERY_ATTEMPTS,
    PHASE_COMMITTING,
    PHASE_STAGING,
    ChunkRow,
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
    tables = store._existing_tables(project.id)
    assert tables is not None
    stored = tables.chunks.search().select(["embedding_text", "vector"]).to_list()[0]
    expected = len(stored["embedding_text"]) % 7
    assert stored["vector"] == [float(expected), 1.0, 2.0, 3.0]

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


def _committing_journal(directory: Path, project_id: str, versions: TableVersions) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / JOURNAL_NAME
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "job_id": directory.name,
                "project_id": project_id,
                "phase": PHASE_COMMITTING,
                "files_version": versions.files,
                "chunks_version": versions.chunks,
                "replace_file_ids": [],
                "removed_file_ids": [],
            }
        )
    )
    return path


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
    _committing_journal(directory, "ghost-project", TableVersions(files=3, chunks=3))

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
    journal_path = _committing_journal(directory, "ghost-project", TableVersions(files=1, chunks=1))
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
