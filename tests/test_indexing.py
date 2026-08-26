import itertools
import os
import sqlite3
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest
from conftest import run_git
from filelock import FileLock
from lancedb.table import LanceTable
from test_token_batching import fake_encode

from code_indexing_mcp import indexing as indexing_module
from code_indexing_mcp import staging as staging_module
from code_indexing_mcp.embedding import (
    EmbeddedSegment,
    PassageCandidate,
    SegmentPlan,
    embed_planned_segments,
    pack_vector,
)
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.history import HistoryStore
from code_indexing_mcp.indexing import REFERENCE_SCHEMA_VERSION, Indexer
from code_indexing_mcp.models import (
    ExtractedChunk,
    ExtractionResult,
    IndexProgress,
    ProjectInfo,
    StoredFile,
)
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import SourceScanner, _GitEnumerationError
from code_indexing_mcp.storage import LanceStore, _quoted


class RecordingEmbedder:
    model_id = "test/code"
    dimension = 4

    def __init__(self) -> None:
        self.passage_batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if any("RAISE_EMBEDDING" in text for text in texts):
            raise RuntimeError("embedding failed")
        self.passage_batches.append(texts)
        return [[float(len(text) % 7), 1.0, 2.0, 3.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [float(len(text) % 7), 1.0, 2.0, 3.0]


class SessionEmbedder(RecordingEmbedder):
    def __enter__(self) -> "SessionEmbedder":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def make_indexer(
    tmp_path: Path,
    embedder: RecordingEmbedder,
    *,
    batch_size: int = 1,
    history: HistoryStore | None = None,
) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    return (
        Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=tmp_path / "locks",
            batch_size=batch_size,
            history=history,
        ),
        store,
    )


def test_indexer_skips_unchanged_and_metadata_only_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    first = indexer.index(project)
    original_read_bytes = Path.read_bytes

    def fail_if_source_is_read(path: Path) -> bytes:
        if path == source:
            raise AssertionError("unchanged source was read")
        return original_read_bytes(path)

    with patch.object(Path, "read_bytes", fail_if_source_is_read):
        second = indexer.index(project)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    metadata_only = indexer.index(project)

    assert first.indexed_files == 1
    assert first.parsed_files == 1
    assert len(embedder.passage_batches) == 1
    assert second.unchanged_files == 1
    assert second.parsed_files == 0
    assert second.embedded_chunks == 0
    assert metadata_only.metadata_only_files == 1
    assert metadata_only.parsed_files == 0
    assert len(embedder.passage_batches) == 1
    assert len(store.list_files(project.id)) == 1


def test_walk_fallback_after_partial_git_enumeration_indexes_each_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a git enumeration that failed after streaming some batches
    let the walk fallback yield those files again. The repeat queued the same
    file_id under a second pending owner, so the flush staged its chunk rows
    non-contiguously and crashed with "Staged chunk batches for a file must be
    contiguous" (or duplicated the rows when the copies landed adjacently)."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "a.py").write_text("def a():\n    return 1\n")
    (root / "b.py").write_text("def b():\n    return 2\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    def partially_failing_enumeration(_: Path):
        yield [root / "a.py"]
        raise _GitEnumerationError("simulated mid-stream git failure")

    monkeypatch.setattr(indexer.scanner, "_iter_git_batches", partially_failing_enumeration)

    report = indexer.index(project)

    assert report.indexed_files == 2
    assert len(store.list_files(project.id)) == 2
    assert sorted(chunk.path for chunk in store.list_chunks([project.id])) == ["a.py", "b.py"]


def test_successful_index_stages_references_declarations_and_coverage(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text(
        "def callee(value):\n    return value\n\ndef caller():\n    return callee(1)\n"
    )
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    indexer.index(project)

    rows = store.list_reference_records(project.id)
    assert any(
        row["record_kind"] == "reference"
        and row["kind"] == "call"
        and row["target_name"] == "callee"
        for row in rows
    )
    assert any(
        row["record_kind"] == "declaration" and row["source_qualified_symbol"] == "caller"
        for row in rows
    )
    assert [row["schema_version"] for row in rows if row["record_kind"] == "coverage"] == [
        REFERENCE_SCHEMA_VERSION
    ]


def test_a_file_with_no_structural_occurrences_still_gets_coverage(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "empty.py").write_text("# nothing to resolve\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    indexer.index(project)

    rows = store.list_reference_records(project.id)
    assert [row["record_kind"] for row in rows] == ["coverage"]


def test_changed_file_replaces_its_structural_generation_atomically(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def old():\n    return 1\n\ndef caller():\n    return old()\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)

    source.write_text("def new():\n    return 2\n\ndef caller():\n    return new()\n")
    indexer.index(project)

    rows = store.list_reference_records(project.id)
    assert {row["target_name"] for row in rows if row["record_kind"] == "reference"} == {"new"}
    assert len([row for row in rows if row["record_kind"] == "coverage"]) == 1


def test_index_report_splits_duration_into_phases(tmp_path: Path) -> None:
    """Embedding cost must be separable from scan/parse/commit cost in the report."""

    class SlowEmbedder(RecordingEmbedder):
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            time.sleep(0.05)
            return super().embed_passages(texts)

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, SlowEmbedder())

    report = indexer.index(project)

    assert report.embed_duration_ms is not None
    assert report.embed_duration_ms >= 45
    assert report.embedding_backend == "cpu"
    assert report.embedding_batch_size == 1
    assert report.scan_ms == report.scan_duration_ms
    assert report.parse_ms == report.parse_duration_ms
    assert report.embed_ms == report.embed_duration_ms
    assert report.commit_ms == report.commit_duration_ms
    assert report.fallback_count == 0
    assert report.peak_memory_bytes is not None
    assert report.peak_memory_bytes > 0
    phases = [
        report.scan_duration_ms,
        report.parse_duration_ms,
        report.embed_duration_ms,
        report.commit_duration_ms,
    ]
    assert all(phase is not None and phase >= 0 for phase in phases)
    # The phases partition the indexing work, so together they cannot exceed the
    # total, which also covers lock acquisition and project bookkeeping.
    assert sum(phase or 0 for phase in phases) <= report.duration_ms


def test_reference_extraction_duration_is_its_own_phase_not_the_whole_parse(
    tmp_path: Path,
) -> None:
    """T1: `reference_extraction_duration_ms` must time only structural

    reference extraction, and `staged_reference_rows` must count what this
    run staged, not the whole project's stored total -- otherwise every
    scenario in a benchmark that touches the same project reports identical
    numbers regardless of how much work it actually did.
    """

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text(
        "class Base:\n    pass\n\nclass Child(Base):\n    def run(self):\n        return Base()\n"
    )
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())

    first = indexer.index(project)

    assert first.reference_extraction_duration_ms is not None
    assert first.reference_extraction_duration_ms >= 0
    assert first.reference_extraction_duration_ms <= (first.parse_duration_ms or 0)
    assert first.staged_reference_rows > 0

    (root / "b.py").write_text("def only_one_call():\n    return len([])\n")
    second = indexer.index(project, force=False)

    # An incremental run that only touches one small new file must not report
    # the same staged-row count as the first full-project run.
    assert second.staged_reference_rows != first.staged_reference_rows
    assert second.staged_reference_rows > 0


def test_indexer_batches_chunks_across_changed_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return 1\n")
    (root / "b.py").write_text("def beta():\n    return 2\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder, batch_size=8)

    report = indexer.index(project)

    assert report.errors == []
    assert len(embedder.passage_batches) == 1
    assert len(embedder.passage_batches[0]) == 2
    tables = store._existing_tables(project.id)
    assert tables is not None
    rows = tables.chunks.search().select(["path", "identifier_terms", "vector"]).to_list()
    assert {row["path"] for row in rows} == {"a.py", "b.py"}
    assert all(row["identifier_terms"] for row in rows)
    # The embedding text is transient (never persisted); the committed vectors
    # must still be the embedder's own output for the texts it was given.
    embedded_first = {float(len(text) % 7) for batch in embedder.passage_batches for text in batch}
    stored_first = {row["vector"][0] for row in rows}
    assert stored_first == embedded_first


def test_cross_file_batch_failure_only_rejects_its_own_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    good = root / "good.py"
    bad = root / "bad.py"
    good.write_text("def old_good():\n    return 1\n")
    bad.write_text("def old_bad():\n    return 2\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder, batch_size=8)
    indexer.index(project)
    old_bad_ids = {
        chunk.chunk_id for chunk in store.list_chunks([project.id]) if chunk.path == "bad.py"
    }

    good.write_text("def new_good():\n    return 3\n")
    bad.write_text("def RAISE_EMBEDDING():\n    return 4\n")
    report = indexer.index(project)

    chunks = store.list_chunks([project.id])
    assert [issue.path for issue in report.errors] == ["bad.py"]
    assert report.indexed_files == 1
    assert report.fallback_count >= 1
    assert {chunk.path for chunk in chunks} == {"good.py", "bad.py"}
    assert any(chunk.symbol == "new_good" for chunk in chunks)
    assert {chunk.chunk_id for chunk in chunks if chunk.path == "bad.py"} == old_bad_ids


def test_flush_failure_aborts_run_instead_of_poisoning_pending(tmp_path: Path) -> None:
    """A failure inside the pending flush must abort the run, not queue poison.

    The flush stages chunk rows for its whole pending batch before it writes
    file records. When one of those final writes fails, the pending files have
    chunks already in the staged stream: swallowing that failure as the
    currently-scanned file's error leaves them queued, and the next flush
    re-stages them after other files. That either splits one file across
    batches and crashes the staging contiguity invariant, or silently
    duplicates its rows.
    """
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a.py", "b.py", "c.py", "d.py"):
        (root / name).write_text(f"def {name[0]}_value():\n    return 1\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    calls = itertools.count()
    real_stage_file = staging_module.StagingJob.stage_file

    def flaky_stage_file(job: staging_module.StagingJob, record: StoredFile) -> None:
        # a.py's record from the first flush succeeds; b.py's from the second
        # flush's final loop explodes after its chunks were already staged.
        if next(calls) == 1:
            raise RuntimeError("staging exploded")
        real_stage_file(job, record)

    with (
        patch.object(indexing_module, "CANDIDATE_GROUP_CHARS", 64),
        patch.object(indexing_module, "CANDIDATE_GROUP_COUNT", 3),
        patch.object(staging_module.StagingJob, "stage_file", flaky_stage_file),
        pytest.raises(RuntimeError, match="staging exploded"),
    ):
        indexer.index(project)

    # The aborted run committed nothing; a clean re-run heals the project.
    assert store.list_files(project.id) == []
    healed = indexer.index(project)
    assert healed.errors == []
    assert {record.path for record in store.list_files(project.id)} == {
        "a.py",
        "b.py",
        "c.py",
        "d.py",
    }


def test_indexer_replaces_changed_files_and_removes_deleted_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 41\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    original_ids = {chunk.chunk_id for chunk in store.list_chunks([project.id])}

    source.write_text("def renamed_answer():\n    return 42\n")
    changed = indexer.index(project)
    changed_ids = {chunk.chunk_id for chunk in store.list_chunks([project.id])}
    source.unlink()
    removed = indexer.index(project)

    assert changed.indexed_files == 1
    assert changed_ids != original_ids
    assert removed.removed_files == 1
    assert store.list_chunks([project.id]) == []
    assert store.list_reference_records(project.id) == []


def test_failed_changed_file_preserves_previous_generation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def stable():\n    return 1\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    original_chunks = store.list_chunks([project.id])
    original_references = store.list_reference_records(project.id)
    original_hash = store.list_files(project.id)[0].content_hash
    assert original_references

    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    report = indexer.index(project)

    assert len(report.errors) == 1
    # A failed replacement keeps the previous generation live: the retained
    # chunks, references, and file row all describe the same (old) content
    # hash, so nothing is served against bytes it was never extracted from.
    assert store.list_chunks([project.id]) == original_chunks
    assert store.list_reference_records(project.id) == original_references
    failed = store.list_files(project.id)[0]
    assert failed.content_hash == original_hash
    assert failed.has_errors is True
    assert store.project_state(project.id) == "partial"
    # A failed file is recorded with its current size/mtime, so subsequent
    # runs skip it instead of re-reading, re-parsing, and re-embedding it.
    batches = len(embedder.passage_batches)
    second = indexer.index(project)
    assert len(embedder.passage_batches) == batches
    assert second.unchanged_files == 1
    assert second.errors == []
    assert store.list_chunks([project.id]) == original_chunks
    assert store.list_reference_records(project.id) == original_references
    # The no-op run must not clear the error it did not heal.
    assert store.list_files(project.id)[0].has_errors is True


def test_reindexing_changed_content_retires_the_previous_chunk_id(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    previous = store.list_chunks([project.id])[0]

    # Keep offsets stable so content identity is the only reason the stale id
    # must be retired.
    source.write_text("def answer():\n    return 43\n")
    indexer.index(project)

    current = store.list_chunks([project.id])[0]
    assert current.chunk_id != previous.chunk_id
    assert store.get_chunk(previous.chunk_id) is None
    fetched = store.get_chunk(current.chunk_id)
    assert fetched is not None and fetched.content.endswith("43")


def test_failed_changed_file_retains_previous_references_on_extraction_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def stable():\n    return 1\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    original_references = store.list_reference_records(project.id)
    original_hash = store.list_files(project.id)[0].content_hash
    assert original_references
    source.write_text("def changed():\n    return 2\n")

    with patch.object(indexer.extractor, "extract", side_effect=RuntimeError("query failed")):
        report = indexer.index(project)

    assert [issue.path for issue in report.errors] == ["main.py"]
    # An extraction failure is a failed replacement too: the previous
    # generation stays live and internally consistent, so its references are
    # retained rather than retired.
    assert store.list_reference_records(project.id) == original_references
    assert store.list_files(project.id)[0].content_hash == original_hash


def test_a_reindex_that_gains_a_syntax_error_retires_its_stale_references(tmp_path: Path) -> None:
    """A syntax error must not leave references from the *previous* content

    Regression for finding 4: a file that still extracts -- just with
    tree-sitter errors -- gets its chunks and content_hash replaced by the
    new generation. Leaving its old reference rows in place would serve them
    at byte offsets from content that no longer exists on disk, which is worse
    than reporting no references at all.
    """
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def foo():\n    return 1\n\ndef caller():\n    return foo()\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    original_references = store.list_reference_records(project.id)
    assert any(row["record_kind"] == "reference" for row in original_references)

    source.write_text("def foo(:\n    return 1\n")  # moved call, syntax error
    with patch.object(
        indexer.extractor,
        "extract",
        return_value=ExtractionResult(chunks=[], has_errors=True),
    ):
        report = indexer.index(project)

    assert report.errors == []
    record = store.list_files(project.id)[0]
    assert record.has_errors is True
    # The stale generation is gone rather than served against new bytes; the
    # file is honestly uncovered until a later parse succeeds cleanly.
    assert store.list_reference_records(project.id) == []


def _remove_reference_generation(store: LanceStore, project_id: str) -> None:
    for record in store.list_files(project_id):
        store.replace_files_from_arrow(
            project_id,
            files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
            chunk_batches=(),
            reference_batches=[
                (
                    [record.file_id],
                    pa.Table.from_batches([], schema=LanceStore.reference_arrow_schema()),
                )
            ],
        )


def _remove_reference_coverage(store: LanceStore, project_id: str) -> None:
    records = store.list_reference_records(project_id)
    for record in store.list_files(project_id):
        remaining = [
            row
            for row in records
            if row["file_id"] == record.file_id and row["record_kind"] != "coverage"
        ]
        store.replace_files_from_arrow(
            project_id,
            files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
            chunk_batches=(),
            reference_batches=[
                (
                    [record.file_id],
                    pa.Table.from_pylist(remaining, schema=LanceStore.reference_arrow_schema()),
                )
            ],
        )


def test_reference_backfill_parses_unchanged_files_without_embedding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    _remove_reference_generation(store, project.id)
    batches_before = len(embedder.passage_batches)

    report = indexer.backfill_references(project)

    assert report.files_backfilled == 1
    assert report.complete is True
    assert len(embedder.passage_batches) == batches_before
    assert store.coverage_for_file(
        project.id,
        store.list_files(project.id)[0].file_id,
        REFERENCE_SCHEMA_VERSION,
    )


def test_files_current_agrees_between_the_backfilling_call_and_the_converged_one(
    tmp_path: Path,
) -> None:
    """`files_current` means "files with current coverage after this report",

    not "files that already had coverage before it ran" -- so a call that
    itself does the backfilling and a later, fully-converged call that finds
    nothing left to do must report the same total for the same persisted
    state. This is a general property of the counter, not specific to
    rejected files (see the rejected-tombstone regression above).
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    _remove_reference_generation(store, project.id)

    first = indexer.backfill_references(project)
    second = indexer.backfill_references(project)

    assert first.files_backfilled == 1
    assert second.files_checked == 0
    assert first.files_current == second.files_current == 1


def test_reference_backfill_reports_a_source_hash_mismatch_without_committing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    _remove_reference_generation(store, project.id)
    source.write_text("def changed():\n    return 43\n")
    batches_before = len(embedder.passage_batches)

    report = indexer.backfill_references(project)

    assert report.stale_paths == ["main.py"]
    assert store.list_reference_records(project.id) == []
    assert len(embedder.passage_batches) == batches_before


def test_reference_backfill_interruption_keeps_the_previous_generation(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    _remove_reference_coverage(store, project.id)
    before = store.list_reference_records(project.id)

    with (
        patch.object(
            indexer.extractor,
            "extract",
            side_effect=CodeIndexingError(ErrorCode.INDEX_CANCELLED, "interrupted"),
        ),
        pytest.raises(CodeIndexingError) as raised,
    ):
        indexer.backfill_references(project)

    assert raised.value.code is ErrorCode.INDEX_CANCELLED
    assert store.list_reference_records(project.id) == before


def test_reference_backfill_covers_a_file_with_no_occurrences(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "empty.py").write_text("# no occurrences\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    _remove_reference_generation(store, project.id)

    report = indexer.backfill_references(project)

    assert report.complete is True
    assert [row["record_kind"] for row in store.list_reference_records(project.id)] == ["coverage"]


def test_reference_backfill_retries_an_incomplete_parse(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    _remove_reference_generation(store, project.id)

    with patch.object(
        indexer.extractor,
        "extract",
        return_value=ExtractionResult(chunks=[], has_errors=True),
    ):
        incomplete = indexer.backfill_references(project)
    retried = indexer.backfill_references(project)

    assert incomplete.incomplete_paths == ["main.py"]
    assert retried.files_backfilled == 1
    assert retried.complete is True


def _assert_chunk_content_hashes_match_files(store: LanceStore, project_id: str) -> None:
    """Divergence tripwire (S1), post-slim-schema form.

    Chunk rows no longer carry a content hash (it lives on the files table),
    but structural coverage rows do. A failed file legitimately keeps its
    previous generation's coverage, and the files row keeps that same
    generation's hash, so the comparison holds for failed files too -- a
    mismatch is exactly the silent corruption S1 caused, caught without
    relying on coverage bookkeeping to self-report it.
    """
    files_by_id = {record.file_id: record for record in store.list_files(project_id)}
    for coverage in store.reference_coverage(project_id):
        record = files_by_id.get(coverage["file_id"])
        if record is None:
            continue
        assert coverage["content_hash"] == record.content_hash, (
            f"coverage for {record.path} was staged for content_hash "
            f"{coverage['content_hash']!r} but the files row now reports "
            f"{record.content_hash!r}"
        )


def test_reference_backfill_does_not_launder_an_embed_failed_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    original_hash = store.list_files(project.id)[0].content_hash
    original_references = store.list_reference_records(project.id)
    assert original_references

    # Change the file's content in a way that fails embedding. The previous
    # generation -- chunks, references, and the file row -- stays live and
    # internally consistent: the row keeps the old content hash rather than
    # advancing to the (never-embedded) new one.
    source.write_text("def RAISE_EMBEDDING():\n    return 43\n")
    failed = indexer.index(project)
    assert len(failed.errors) == 1
    failed_record = store.list_files(project.id)[0]
    assert failed_record.has_errors is True
    assert failed_record.content_hash == original_hash
    assert store.list_reference_records(project.id) == original_references

    report = indexer.backfill_references(project)

    # The file is still honestly covered by its retained generation, so
    # backfill has nothing to write for it -- it neither launders references
    # for content the chunk table does not contain nor re-parses a file it
    # already describes.
    assert report.incomplete_paths == []
    assert report.files_backfilled == 0
    assert report.complete is True
    coverage = store.coverage_for_file(project.id, failed_record.file_id, REFERENCE_SCHEMA_VERSION)
    assert coverage and coverage[0]["content_hash"] == failed_record.content_hash
    assert store.list_reference_records(project.id) == original_references
    _assert_chunk_content_hashes_match_files(store, project.id)

    # The failed file heals once it indexes successfully again.
    source.write_text("def stable():\n    return 44\n")
    healed = indexer.index(project)
    assert healed.errors == []
    healed_record = store.list_files(project.id)[0]
    assert healed_record.has_errors is False

    # A successful index stages references itself, so the file is already
    # covered and backfill has nothing left to do for it.
    healed_report = indexer.backfill_references(project)
    assert healed_report.files_backfilled == 0
    assert healed_report.complete is True
    coverage = store.coverage_for_file(project.id, healed_record.file_id, REFERENCE_SCHEMA_VERSION)
    assert coverage and coverage[0]["content_hash"] == healed_record.content_hash
    _assert_chunk_content_hashes_match_files(store, project.id)


def test_reference_backfill_preserves_a_partial_project_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "good.py").write_text("value = 1\n")
    (root / "bad.py").write_text("def RAISE_EMBEDDING():\n    return 2\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    failed = indexer.index(project)
    assert [issue.path for issue in failed.errors] == ["bad.py"]
    assert store.project_state(project.id) == "partial"
    _remove_reference_generation(store, project.id)

    # Backfilling references for the file that already succeeded must not
    # promote the project past what the failed file still says about it.
    report = indexer.backfill_references(project)

    assert report.files_backfilled == 1
    assert store.project_state(project.id) == "partial"


def test_reference_backfill_publishes_committing_before_it_commits(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    _remove_reference_generation(store, project.id)

    last_phase: list[str | None] = [None]

    def on_progress(progress: object) -> None:
        last_phase[0] = progress.phase  # type: ignore[attr-defined]

    phase_when_committed: list[str | None] = []
    original_commit_staged = indexer._commit_staged

    def recording_commit_staged(*args: object, **kwargs: object) -> None:
        phase_when_committed.append(last_phase[0])
        return original_commit_staged(*args, **kwargs)

    with patch.object(indexer, "_commit_staged", side_effect=recording_commit_staged):
        indexer.backfill_references(project, on_progress=on_progress)

    # The "committing" phase must be visible to a watcher *before* the commit
    # runs, not published (and immediately erased by progress.clear()) after
    # the fact (S6).
    assert phase_when_committed == ["committing"]


def test_reference_backfill_stale_report_includes_files_backfilled(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "good.py").write_text("value = 1\n")
    (root / "stale.py").write_text("value = 2\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    _remove_reference_generation(store, project.id)

    original_read_bytes = Path.read_bytes

    def flaky_read_bytes(path: Path) -> bytes:
        if path.name == "stale.py":
            raise OSError("vanished mid-scan")
        return original_read_bytes(path)

    with patch.object(Path, "read_bytes", flaky_read_bytes):
        report = indexer.backfill_references(project)

    # The stale-path early return still discards the whole generation (a
    # partial commit would be dishonest), but the retry loop's cost -- work
    # parsed for good.py and then thrown away -- must stay visible (S6).
    assert report.stale_paths == ["stale.py"]
    assert report.files_backfilled == 1
    assert store.list_reference_records(project.id) == []


def test_backfill_stops_retrying_a_rejected_tombstone_and_reports_it_correctly(
    tmp_path: Path,
) -> None:
    """Regression for finding 10.

    A deliberately rejected file (binary/undecodable content) is not a parse
    failure -- it will never parse, by design. Backfill must mark it covered
    (a coverage-only row) instead of leaving it in the missing set forever,
    where it would be re-walked on every future call and misreported as
    `parse_error`.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_bytes(b"\x00binary garbage")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    index_report = indexer.index(project)
    assert index_report.skipped_files == 1
    rejected_record = store.list_files(project.id)[0]
    assert rejected_record.has_errors is True
    assert rejected_record.error == "rejected: binary"

    report = indexer.backfill_references(project)

    assert report.incomplete_paths == []
    assert report.files_backfilled == 1
    assert report.complete is True
    coverage = store.coverage_for_file(
        project.id, rejected_record.file_id, REFERENCE_SCHEMA_VERSION
    )
    assert coverage and coverage[0]["content_hash"] == rejected_record.content_hash

    # Converged: the file is now known at the current schema, so a later
    # call does not even need to walk it again.
    again = indexer.backfill_references(project)
    assert again.files_checked == 0
    assert again.files_current == 1
    # The two reports describe the same converged state, so a counter that
    # means "files with current coverage" must agree between them regardless
    # of which call happened to do the covering.
    assert report.files_current == again.files_current


def test_backfill_creates_the_reference_table_even_when_every_file_fails_to_parse(
    tmp_path: Path,
) -> None:
    """Regression for finding 8.

    A legacy pre-feature partition (files/chunks exist, references never
    did) whose only structural file fails to parse used to leave `job` at
    None forever, so `_commit_staged` never ran and the references table was
    never created -- `find_references`/`analyze_refactor` would then raise
    REFERENCE_INDEX_UNAVAILABLE and tell the caller to run the exact backfill
    that just ran and can never succeed.
    """
    import lancedb

    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    assert store.has_reference_table(project.id) is True

    # Simulate a legacy pre-feature partition: files/chunks exist, but the
    # references table itself was never created. A second LanceStore avoids
    # the first store's cached table handles.
    lancedb.connect(tmp_path / "data" / "projects" / project.id).drop_table("references")
    indexer2, store2 = make_indexer(tmp_path, RecordingEmbedder())
    assert store2.has_reference_table(project.id) is False

    with patch.object(indexer2.extractor, "extract", side_effect=RuntimeError("broken parser")):
        report = indexer2.backfill_references(project)

    assert report.incomplete_paths == ["main.py"]
    # Nothing could be backfilled, but the table itself must still exist.
    assert store2.has_reference_table(project.id) is True


def test_backfill_retires_reference_rows_from_a_retired_schema_version(tmp_path: Path) -> None:
    """Regression for finding 9 (write side).

    The version bump's comment claims it "discards any generation written by
    version 3", but nothing walked existing rows to make that true for a file
    that cannot currently be re-covered. A file stuck in incomplete_paths
    must not keep serving colliding-id rows from a retired schema forever.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    file_id = store.list_files(project.id)[0].file_id

    stale_version = REFERENCE_SCHEMA_VERSION - 1
    downgraded = [
        {**row, "schema_version": stale_version}
        for row in store.list_reference_records(project.id)
        if row["file_id"] == file_id
    ]
    store.replace_files_from_arrow(
        project.id,
        files=pa.Table.from_batches([], schema=LanceStore.file_arrow_schema()),
        chunk_batches=(),
        reference_batches=[
            (
                [file_id],
                pa.Table.from_pylist(downgraded, schema=LanceStore.reference_arrow_schema()),
            )
        ],
    )
    assert all(
        row["schema_version"] == stale_version for row in store.list_reference_records(project.id)
    )

    with patch.object(indexer.extractor, "extract", side_effect=RuntimeError("broken parser")):
        report = indexer.backfill_references(project)

    assert report.incomplete_paths == ["main.py"]
    # The stale schema-3 generation must not survive: serving it collides
    # with the current schema's id scheme, exactly what the bump intended to
    # prevent.
    assert store.list_reference_records(project.id) == []


def test_backfill_does_not_crash_for_an_unregistered_marker_resolved_project(
    tmp_path: Path,
) -> None:
    """Regression for finding 11.

    A marker-resolved project that was never registered in the store (zero
    eligible source files, so `index()` was never called to register it; or
    a data directory wiped while the on-disk marker survived) must not crash
    `project_state` lookup -- it should behave like any other project with
    nothing to report.
    """
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)  # writes the on-disk marker only
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())

    report = indexer.backfill_references(project)

    assert report.files_current == 0
    assert report.complete is True


def test_global_index_lock_serializes_different_projects(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())

    lock = FileLock(tmp_path / "locks" / "index-global.lock")
    with lock, pytest.raises(CodeIndexingError) as caught:
        indexer.index(project)

    assert caught.value.code is ErrorCode.INDEX_BUSY


def test_indexer_uses_passage_worker_session_when_configured(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    parent_embedder = RecordingEmbedder()
    worker_embedder = SessionEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=parent_embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=parent_embedder,
        lock_directory=tmp_path / "locks",
        passage_session_factory=lambda: worker_embedder,
    )

    indexer.index(project)

    assert worker_embedder.passage_batches
    assert parent_embedder.passage_batches == []


def test_force_reindexes_previously_failed_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    failed = indexer.index(project)
    assert len(failed.errors) == 1

    source.write_text("def stable():\n    return 3\n")
    forced = indexer.index(project, force=True)

    assert forced.errors == []
    assert forced.indexed_files == 1
    assert store.list_files(project.id)[0].has_errors is False


def test_a_later_source_edit_retries_and_clears_the_error_after_success(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    assert store.list_files(project.id)[0].has_errors is True

    # A later source edit changes size/mtime, so the failed file is retried
    # without force; once the edit succeeds, its stored error is cleared.
    source.write_text("def stable():\n    return 3\n")
    healed = indexer.index(project)

    assert healed.errors == []
    assert healed.unchanged_files == 0
    assert store.list_files(project.id)[0].has_errors is False
    assert store.project_state(project.id) == "ready"


def test_a_noop_run_cannot_promote_a_partial_project_with_stored_errors_to_ready(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    assert store.project_state(project.id) == "partial"
    assert store.list_files(project.id)[0].has_errors is True

    # Nothing changed, so this is a no-op run with no fresh errors -- but the
    # stored file error still says the project is partial.
    noop = indexer.index(project)

    assert noop.errors == []
    assert noop.unchanged_files == 1
    assert store.project_state(project.id) == "partial"


def test_unexpected_index_failure_marks_project_error(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("x = 1\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    with (
        patch.object(SourceScanner, "iter_scan", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        indexer.index(project)

    assert store.project_state(project.id) == "error"


def test_model_unavailable_marks_project_error(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n")
    project = initialize_project(root)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    with (
        patch.object(
            embedder,
            "embed_passages",
            side_effect=CodeIndexingError(ErrorCode.MODEL_UNAVAILABLE, "model missing"),
        ),
        pytest.raises(CodeIndexingError) as raised,
    ):
        indexer.index(project)

    assert raised.value.code is ErrorCode.MODEL_UNAVAILABLE
    assert store.project_state(project.id) == "error"


def test_store_keeps_projects_isolated_and_removal_does_not_touch_markers(
    tmp_path: Path,
) -> None:
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    projects = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        (root / "main.py").write_text(f"def {name}():\n    return '{name}'\n")
        project = initialize_project(root)
        projects.append(project)
        indexer.index(project)

    store.remove_project(projects[0].id)

    assert {project.id for project in store.list_projects()} == {projects[1].id}
    # list_chunks no longer carries project_id on chunk rows; the surviving
    # project's chunks are identifiable by their routing prefix.
    assert len(store.list_chunks()) == 1
    assert store.list_chunks()[0].chunk_id.startswith(f"{projects[1].id}:")
    assert (projects[0].root / ".ci-mcp" / "project.toml").exists()


class ResourceLimitEmbedder(RecordingEmbedder):
    """Trips the memory ceiling on the second file, as a busy machine would."""

    def __init__(self) -> None:
        super().__init__()
        self.files_seen = 0

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.files_seen += 1
        if self.files_seen == 2:
            raise CodeIndexingError(
                ErrorCode.INDEX_RESOURCE_LIMIT, "Indexing exceeded its memory ceiling"
            )
        return super().embed_passages(texts)


def test_resource_limit_aborts_instead_of_poisoning_the_file_record(tmp_path: Path) -> None:
    """A transient ceiling trip must not mark the file as up to date."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("value = 1\n")
    (root / "b.py").write_text("value = 2\n")
    project = initialize_project(root)
    embedder = ResourceLimitEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)

    with pytest.raises(CodeIndexingError) as caught:
        indexer.index(project)

    assert caught.value.code is ErrorCode.INDEX_RESOURCE_LIMIT
    # The file that failed was never recorded, so a later run retries it rather
    # than skipping it as unchanged forever.
    recorded = {record.path for record in store.list_files(project.id)}
    assert len(recorded) < 2

    healthy, _ = make_indexer(tmp_path, RecordingEmbedder())
    report = healthy.index(project)

    assert report.errors == []
    assert len(store.list_files(project.id)) == 2
    assert store.project_state(project.id) == "ready"


class WindowingEmbedder(RecordingEmbedder):
    """A double that windows candidates the way the real worker does.

    It runs the production planner over the deterministic tokenizer from
    ``test_token_batching``, so the offsets under test are the ones the real
    path produces, without loading a model.
    """

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        planned = embed_planned_segments(fake_encode, self.embed_passages, candidates, plan)
        return [
            [
                EmbeddedSegment(
                    window.start_char, window.end_char, window.token_count, pack_vector(vector)
                )
                for window, vector in segments
            ]
            for segments in planned
        ]


def make_windowing_indexer(
    tmp_path: Path, embedder: RecordingEmbedder, plan: SegmentPlan
) -> tuple[Indexer, LanceStore]:
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    return (
        Indexer(
            store=store,
            scanner=SourceScanner(),
            extractor=TreeSitterExtractor(),
            embedder=embedder,
            lock_directory=tmp_path / "locks",
            segment_plan=plan,
        ),
        store,
    )


DENSE_SOURCE = (
    "def answer():\n"
    "    total = 0\n"
    "    for index in range(10):\n"
    "        total = total + index\n"
    "    return total\n"
)


def test_a_token_dense_chunk_is_split_into_several_stored_chunks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    # Bytes, not text: write_text translates "\n" to "\r\n" on Windows, and
    # these tests compare on-disk offsets against DENSE_SOURCE itself.
    (root / "main.py").write_bytes(DENSE_SOURCE.encode("utf-8"))
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    report = indexer.index(project)
    chunks = store.list_chunks([project.id])

    assert report.errors == []
    assert len(chunks) > 1
    assert {chunk.qualified_symbol for chunk in chunks} == {"answer"}


def test_windowed_chunk_offsets_still_slice_the_original_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    # Bytes, not text: write_text translates "\n" to "\r\n" on Windows, and
    # these tests compare on-disk offsets against DENSE_SOURCE itself.
    (root / "main.py").write_bytes(DENSE_SOURCE.encode("utf-8"))
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    indexer.index(project)
    source = (root / "main.py").read_bytes()

    for chunk in store.list_chunks([project.id]):
        assert source[chunk.start_byte : chunk.end_byte].decode("utf-8") == chunk.content
        assert source[: chunk.start_byte].decode("utf-8").count("\n") + 1 == chunk.start_line
        assert chunk.end_line == chunk.start_line + chunk.content.count("\n")


def test_every_window_keeps_the_identifier_tail(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    # Bytes, not text: write_text translates "\n" to "\r\n" on Windows, and
    # these tests compare on-disk offsets against DENSE_SOURCE itself.
    (root / "main.py").write_bytes(DENSE_SOURCE.encode("utf-8"))
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    indexer.index(project)
    chunks = store.list_chunks([project.id])

    # The identifier terms are not recomputed per window: every window of the
    # chunk carries the same normalized path and symbol tail, so camelCase and
    # snake_case symbol queries keep matching every part of a windowed chunk.
    # The embedding context header is transient (it feeds the embedder, not
    # storage), so only the persisted tail is observable here.
    assert chunks
    assert all(chunk.identifier_terms == "main py answer answer" for chunk in chunks)
    assert all(chunk.content for chunk in chunks)


def test_windows_cover_the_symbol_without_dropping_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    # Bytes, not text: write_text translates "\n" to "\r\n" on Windows, and
    # these tests compare on-disk offsets against DENSE_SOURCE itself.
    (root / "main.py").write_bytes(DENSE_SOURCE.encode("utf-8"))
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, WindowingEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    indexer.index(project)
    chunks = sorted(store.list_chunks([project.id]), key=lambda chunk: chunk.start_byte)
    source = (root / "main.py").read_bytes()

    covered = source[chunks[0].start_byte : chunks[0].end_byte]
    for previous, chunk in itertools.pairwise(chunks):
        assert chunk.start_byte <= previous.end_byte
        covered += source[max(chunk.start_byte, previous.end_byte) : chunk.end_byte]
    assert covered == source[chunks[0].start_byte : chunks[-1].end_byte]
    assert covered.decode("utf-8").strip() == DENSE_SOURCE.strip()


def test_a_chunk_within_the_budget_is_stored_unchanged(tmp_path: Path) -> None:
    """Windowing must be invisible to files that never exceed the budget."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    windowing, windowed_store = make_windowing_indexer(
        tmp_path / "windowed", WindowingEmbedder(), SegmentPlan(max_tokens=1_024)
    )
    plain, plain_store = make_indexer(tmp_path / "plain", RecordingEmbedder())

    windowing.index(project)
    plain.index(project)

    def comparable(store: LanceStore) -> list[tuple[object, ...]]:
        return sorted(
            (
                chunk.chunk_id,
                chunk.start_byte,
                chunk.end_byte,
                chunk.content,
                chunk.identifier_terms,
            )
            for chunk in store.list_chunks([project.id])
        )

    assert comparable(windowed_store) == comparable(plain_store)


class UnplannableEmbedder(WindowingEmbedder):
    """Rejects one file's candidates the way an exploding window plan would."""

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        if any("REJECT" in candidate.content for candidate in candidates):
            raise ValueError("Token planning exceeded 16 windows")
        return super().plan_and_embed(candidates, plan)


def test_an_unplannable_file_is_charged_to_the_file_not_the_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "good.py").write_text("value = 1\n")
    (root / "bad.py").write_text("REJECT = 'x'\n")
    project = initialize_project(root)
    indexer, store = make_windowing_indexer(
        tmp_path, UnplannableEmbedder(), SegmentPlan(max_tokens=8, overlap_tokens=2)
    )

    report = indexer.index(project)

    # The run completes and the healthy file is indexed; only the file that
    # could not be planned is recorded as an issue.
    assert [issue.path for issue in report.errors] == ["bad.py"]
    assert store.project_state(project.id) == "partial"
    assert {chunk.path for chunk in store.list_chunks([project.id])} == {"good.py"}


def test_binary_and_undecodable_files_are_skipped_by_the_indexer(tmp_path: Path) -> None:
    """Content rejection moved to where the bytes are already read.

    The file must not be indexed, must not be committed as a stored file, and any
    chunks from an earlier text version of it must be dropped.
    """
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    (root / "good.py").write_text("def good():\n    return 1\n")
    (root / "turned_binary.py").write_text("def old():\n    return 0\n")
    indexer.index(project)
    assert {chunk.path for chunk in store.list_chunks([project.id])} >= {"turned_binary.py"}

    (root / "turned_binary.py").write_bytes(b"def old():\x00\n")
    (root / "latin.py").write_bytes(b"# caf\xe9\ndef latin():\n    return 2\n")
    report = indexer.index(project)

    indexed = {chunk.path for chunk in store.list_chunks([project.id])}
    assert indexed == {"good.py"}
    assert report.skipped_files >= 2
    assert {issue.path for issue in report.errors} == set()


def test_each_changed_file_is_read_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder())
    for index in range(5):
        (root / f"m{index}.py").write_text(f"def f{index}():\n    return {index}\n")

    reads: dict[str, int] = {}
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        if self.suffix == ".py":
            reads[str(self)] = reads.get(str(self), 0) + 1
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    indexer.index(project)

    assert len(reads) == 5
    assert set(reads.values()) == {1}, f"expected one read per file, got {reads}"


def test_two_references_over_one_range_survive_a_reindex(tmp_path: Path) -> None:
    """One byte range can carry two references, and both must be storable.

    A superclass is both `inheritance` and a `read`; a decorator call is both
    `decorator` and `call`. When the row identity omitted the kind, the pair
    collided and merge_insert rejected the whole commit, so every incremental
    index after the first failed permanently and left the project in `error`.
    """

    root = tmp_path / "repo"
    root.mkdir()
    source = root / "guard.py"
    source.write_text(
        "import functools\n"
        "\n"
        "class BaseGuard:\n"
        "    pass\n"
        "\n"
        "class Guard(BaseGuard):\n"
        "    @functools.cache\n"
        "    def check(self):\n"
        "        return 1\n"
    )
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)

    rows = [
        row for row in store.list_reference_records(project.id) if row["record_kind"] == "reference"
    ]
    inherited = next(row for row in rows if row["kind"] == "inheritance")
    span = (inherited["start_byte"], inherited["end_byte"])
    overlapping = [row for row in rows if (row["start_byte"], row["end_byte"]) == span]
    assert {row["kind"] for row in overlapping} == {"inheritance", "read"}
    assert len({row["reference_id"] for row in rows}) == len(rows)

    source.write_text(source.read_text() + "\n\ndef extra():\n    return 2\n")
    report = indexer.index(project)

    assert report.errors == []
    assert report.indexed_files == 1
    assert store.project_state(project.id) == "ready"


def test_a_noop_run_creates_zero_partition_mutations(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def stable():\n    return 1\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    versions_before = store.table_versions(project.id)

    noop = indexer.index(project)

    assert noop.errors == []
    assert noop.unchanged_files == 1
    assert store.table_versions(project.id) == versions_before


def test_five_hundred_changed_files_commit_in_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    for index in range(500):
        (root / f"file_{index}.py").write_text(f"def function_{index}():\n    return {index}\n")
    monkeypatch.setattr(staging_module, "COMMIT_BATCH_MAX_FILES", 64)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())

    original_merge = LanceTable.merge_insert
    merges: dict[str, int] = {}

    def counting_merge(self: LanceTable, key: str) -> object:
        merges[self.name] = merges.get(self.name, 0) + 1
        return original_merge(self, key)

    with patch.object(LanceTable, "merge_insert", counting_merge):
        report = indexer.index(project)

    assert report.errors == []
    assert merges["chunks"] == 8
    assert merges["references"] == 8
    assert merges["files"] == 1
    assert store.count_chunks([project.id]) == 500


def test_a_failure_mid_commit_restores_all_three_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def first():\n    return 1\n")
    project = initialize_project(root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    indexer.index(project)
    files_before = store.list_files(project.id)
    chunks_before = store.list_chunks([project.id])
    references_before = store.list_reference_records(project.id)

    (root / "main.py").write_text("def renamed():\n    return 43\n")
    (root / "second.py").write_text("def other():\n    return 2\n")
    # One file per batch, so the second chunk batch fails after the first
    # batch already mutated the live tables.
    monkeypatch.setattr(staging_module, "COMMIT_BATCH_MAX_FILES", 1)
    original_merge = LanceTable.merge_insert
    calls = {"count": 0}

    def failing_merge(self: LanceTable, key: str) -> object:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated failure on the second chunk batch")
        return original_merge(self, key)

    with (
        patch.object(LanceTable, "merge_insert", failing_merge),
        pytest.raises(RuntimeError, match="second chunk batch"),
    ):
        indexer.index(project)

    assert store.list_files(project.id) == files_before
    assert store.list_chunks([project.id]) == chunks_before
    assert store.list_reference_records(project.id) == references_before
    assert store.project_state(project.id) == "error"

    healed = indexer.index(project)
    assert healed.errors == []
    assert {chunk.qualified_symbol for chunk in store.list_chunks([project.id])} == {
        "renamed",
        "other",
    }


def test_a_completed_run_is_recorded_in_the_audit_history(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    (root / "notes.txt").write_text("unsupported")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder(), history=history)

    report = indexer.index(project, trigger="watcher")

    assert report.run_id is not None
    page = history.list_runs(project.id)
    assert len(page.runs) == 1
    run = page.runs[0]
    assert run.run_id == report.run_id
    assert run.trigger == "watcher"
    assert run.state == "completed"
    assert run.model_id == indexer.embedder.model_id
    assert run.schema_version >= 1
    assert run.scan_config_hash
    assert run.server_version
    assert run.force is False
    assert run.eligible_files == 1
    assert run.changed_files == 1
    assert run.parsed_files == 1
    assert run.skipped_total == 1
    assert run.skip_reasons == {"unsupported": 1}
    assert run.chunks_embedded == report.embedded_chunks
    assert run.phase_durations
    assert run.finished_at is not None
    assert run.storage_after


def test_a_failed_run_is_recorded_as_failed_and_reports_its_reason(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("def stable():\n    return 1\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    indexer, store = make_indexer(tmp_path, RecordingEmbedder(), history=history)
    indexer.index(project)

    source.write_text("def RAISE_EMBEDDING():\n    return 2\n")
    report = indexer.index(project)

    assert report.failed_files == 1
    # A failure is not a skip: it must not leak into the skip pair, which
    # would break skipped_total == sum(skip_reasons).
    assert report.skip_reasons == {}
    assert report.skipped_files == 0
    assert report.skipped_samples == []
    page = history.list_runs(project.id)
    failed = next(run for run in page.runs if run.run_id == report.run_id)
    assert failed.state == "completed"
    assert failed.failed_files == 1
    assert failed.skip_reasons == {}
    assert failed.skipped_total == 0
    assert failed.errors and failed.errors[0].path == "main.py"
    assert store.project_state(project.id) == "partial"


def test_an_unexpected_run_failure_is_recorded_as_failed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder(), history=history)

    with (
        patch.object(indexer.scanner, "iter_scan", side_effect=RuntimeError("simulated crash")),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        indexer.index(project)

    page = history.list_runs(project.id)
    assert len(page.runs) == 1
    assert page.runs[0].state == "failed"
    assert page.runs[0].finished_at is not None


def test_an_audit_write_failure_never_fails_a_run_that_committed(tmp_path: Path) -> None:
    """The audit trail follows the progress-publishing rule: a full disk or a
    runs.sqlite locked past its busy timeout costs the audit row, not the run."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    indexer, store = make_indexer(tmp_path, RecordingEmbedder(), history=history)

    locked = sqlite3.OperationalError("database is locked")
    with (
        patch.object(history, "begin", side_effect=locked),
        patch.object(history, "finish", side_effect=locked),
    ):
        report = indexer.index(project)

    assert report.indexed_files == 1
    assert store.project_state(project.id) == "ready"
    assert history.list_runs(project.id).runs == []


def test_a_reference_backfill_is_audited_with_its_own_trigger(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    indexer, store = make_indexer(tmp_path, RecordingEmbedder(), history=history)
    indexer.index(project)
    _remove_reference_generation(store, project.id)

    history.mark_interrupted()
    indexer.backfill_references(project)

    page = history.list_runs(project.id)
    assert len(page.runs) == 2
    backfill = page.runs[0]
    assert backfill.trigger == "reference-backfill"
    assert backfill.state == "completed"
    assert backfill.changed_files == 1
    assert backfill.eligible_files == 1


def test_a_backfill_with_nothing_missing_leaves_no_audit_row(tmp_path: Path) -> None:
    """Reference tools run a backfill on every query; a no-op must not write a
    durable row, or ordinary reference queries would evict genuine index runs
    from the bounded per-project history window."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    indexer, _ = make_indexer(tmp_path, RecordingEmbedder(), history=history)
    indexer.index(project)

    report = indexer.backfill_references(project)

    assert report.files_backfilled == 0
    page = history.list_runs(project.id)
    assert [run.trigger for run in page.runs] == ["manual"]


class OtherModelEmbedder(RecordingEmbedder):
    """The same vector shape under a different model id and different values.

    The first component is offset so a rebuild is observable: an index that
    failed to re-embed would keep the old generation's vectors.
    """

    model_id = "test/other"

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 7) + 1.0, 1.0, 2.0, 3.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7) + 1.0, 1.0, 2.0, 3.0]


def make_indexer_on(
    tmp_path: Path,
    store: LanceStore,
    embedder: RecordingEmbedder,
    *,
    history: HistoryStore | None = None,
) -> Indexer:
    return Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
        history=history,
    )


def test_index_rebuilds_a_partition_written_by_an_incompatible_model(
    tmp_path: Path,
) -> None:
    """A different embedding model rebuilds the partition instead of failing."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    first = make_indexer_on(tmp_path, store, RecordingEmbedder())
    second = make_indexer_on(tmp_path, store, OtherModelEmbedder())

    first.index(project)
    assert store.project_state(project.id) == "ready"
    old_chunk = store.list_chunks([project.id])[0]
    tables = store._existing_tables(project.id)
    assert tables is not None
    old_vector = (
        tables.chunks.search()
        .where(f"chunk_id = {_quoted(old_chunk.chunk_id)}")
        .select(["vector"])
        .to_list()[0]["vector"]
    )
    assert store.incompatibility_reason(project.id, "test/other") is not None

    report = second.index(project)

    assert report.trigger == "schema-rebuild"
    assert store.incompatibility_reason(project.id, "test/other") is None
    assert store.project_state(project.id) == "ready"
    rebuilt = store.list_chunks([project.id])
    assert len(rebuilt) == 1
    # Chunk ids are content-derived, so they survive a model rebuild; the
    # vectors must not, or the rebuild never re-embedded anything.
    assert rebuilt[0].chunk_id == old_chunk.chunk_id
    tables = store._existing_tables(project.id)
    assert tables is not None
    new_vector = (
        tables.chunks.search()
        .where(f"chunk_id = {_quoted(old_chunk.chunk_id)}")
        .select(["vector"])
        .to_list()[0]["vector"]
    )
    assert new_vector[0] == old_vector[0] + 1.0


def test_chunk_identity_includes_the_physical_slot() -> None:
    file = StoredFile(
        file_id="file-1",
        project_id="project-1",
        path="module.py",
        language="python",
        size=1,
        mtime_ns=1,
        content_hash="content-hash",
        indexed_at=1,
    )
    chunk = ExtractedChunk(
        kind="function",
        symbol="answer",
        qualified_symbol="answer",
        start_byte=0,
        end_byte=1,
        start_line=1,
        end_line=1,
        content="pass",
        embedding_text="pass",
        search_text="answer",
    )

    first = Indexer._chunk_row("project-1", file, chunk, b"vector", slot_id="slot-a")
    second = Indexer._chunk_row("project-1", file, chunk, b"vector", slot_id="slot-b")
    repeat = Indexer._chunk_row("project-1", file, chunk, b"vector", slot_id="slot-a")

    assert first.chunk_id != second.chunk_id
    assert first.chunk_id == repeat.chunk_id


def test_a_rebuild_records_its_reason_and_trigger_in_the_audit_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    first = make_indexer_on(tmp_path, store, RecordingEmbedder())
    second = make_indexer_on(tmp_path, store, OtherModelEmbedder(), history=history)
    first.index(project)

    second.index(project)

    page = history.list_runs(project.id)
    rebuild = next(run for run in page.runs if run.trigger == "schema-rebuild")
    assert rebuild.rebuild_reason is not None
    assert "embedding model" in rebuild.rebuild_reason
    assert rebuild.state == "completed"


def test_a_failed_rebuild_preserves_registration_and_recovers(tmp_path: Path) -> None:
    """A rebuild failure must never drop the registration or the marker."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    first = make_indexer_on(tmp_path, store, RecordingEmbedder())
    second = make_indexer_on(tmp_path, store, OtherModelEmbedder())
    first.index(project)

    with (
        patch.object(
            SourceScanner,
            "iter_scan",
            side_effect=RuntimeError("rebuild boom"),
        ),
        pytest.raises(RuntimeError, match="rebuild boom"),
    ):
        second.index(project)

    assert store.list_projects() == [project]
    assert (root / ".ci-mcp" / "project.toml").exists()
    # The partition was deleted before the failed run; the project must be
    # recoverable by a plain re-run, not stuck behind a permanent error.
    assert store.project_state(project.id) == "error"
    assert store.list_chunks([project.id]) == []

    healed = second.index(project)
    assert healed.errors == []
    assert len(store.list_chunks([project.id])) == 1
    assert store.project_state(project.id) == "ready"


def test_backfill_delegates_a_rebuild_for_an_incompatible_partition(
    tmp_path: Path,
) -> None:
    """A parse-only backfill cannot heal an incompatible partition; it must
    rebuild it with a full embedding run first."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    project = initialize_project(root)
    history = HistoryStore(tmp_path / "history")
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    first = make_indexer_on(tmp_path, store, RecordingEmbedder())
    second = make_indexer_on(tmp_path, store, OtherModelEmbedder(), history=history)
    first.index(project)

    report = second.backfill_references(project)

    assert report.complete is True
    assert report.files_current == 1
    assert report.files_backfilled == 0
    assert store.incompatibility_reason(project.id, "test/other") is None
    page = history.list_runs(project.id)
    assert any(run.trigger == "schema-rebuild" for run in page.runs)


def test_rebuild_backfill_reports_unparseable_files_as_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "valid.py").write_text("def answer():\n    return 42\n")
    (root / "broken.py").write_text("def broken(:\n    pass\n")
    project = initialize_project(root)
    store = LanceStore(tmp_path / "data", vector_dimension=4)
    first = make_indexer_on(tmp_path, store, RecordingEmbedder())
    second = make_indexer_on(tmp_path, store, OtherModelEmbedder())
    first.index(project)

    report = second.backfill_references(project)

    assert report.files_current == 1
    assert report.incomplete_paths == ["broken.py"]


def _git_commit(root: Path, message: str) -> None:
    run_git("add", "-A", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message, cwd=root)


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _write_with_pinned_mtime(path: Path, content: str) -> None:
    """Overwrite *path* while restoring its previous timestamps.

    Captures size and mtime *before* the write so a same-length edit is
    invisible to a metadata walk and only the validation plan can catch it.
    """
    stat = path.stat()
    path.write_text(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def _git_project(tmp_path: Path) -> tuple[Path, ProjectInfo]:
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text("def one():\n    return 1\n")
    (root / "other.py").write_text("def two():\n    return 2\n")
    _git_commit(root, "init")
    return root, initialize_project(root)


def test_a_commit_revalidates_only_the_paths_the_diff_names(tmp_path: Path) -> None:
    root, project = _git_project(tmp_path)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    batches_after_first = len(embedder.passage_batches)

    # Same-length content change plus a pinned mtime: neither size nor mtime
    # can reveal it, so only the commit-to-commit diff names main.py.
    _write_with_pinned_mtime(root / "main.py", "def one():\n    return 2\n")
    _git_commit(root, "second")

    report = indexer.index(project)

    assert report.indexed_files == 1
    assert report.unchanged_files == 1
    assert len(embedder.passage_batches) == batches_after_first + 1
    assert "return 2" in embedder.passage_batches[-1][-1]
    partition = store.active_partition(project.id)
    slot = store.get_slot(partition.slot_id)
    assert slot is not None
    assert slot.indexed_head == _git_head(root)


def test_a_dirty_tracked_file_with_a_pinned_mtime_is_revalidated(tmp_path: Path) -> None:
    root, project = _git_project(tmp_path)
    embedder = RecordingEmbedder()
    indexer, store = make_indexer(tmp_path, embedder)
    indexer.index(project)
    first_head = _git_head(root)

    _write_with_pinned_mtime(root / "main.py", "def one():\n    return 3\n")

    report = indexer.index(project)

    assert report.indexed_files == 1
    assert report.unchanged_files == 1
    slot = store.get_slot(store.active_partition(project.id).slot_id)
    assert slot is not None
    assert slot.indexed_head == first_head
    assert slot.indexed_clean is False


def test_an_untracked_file_with_a_pinned_mtime_is_revalidated(tmp_path: Path) -> None:
    root, project = _git_project(tmp_path)
    embedder = RecordingEmbedder()
    indexer, _store = make_indexer(tmp_path, embedder)
    (root / "extra.py").write_text("def three():\n    return 3\n")
    indexer.index(project)

    _write_with_pinned_mtime(root / "extra.py", "def three():\n    return 4\n")

    report = indexer.index(project)

    assert report.indexed_files == 1
    assert report.unchanged_files == 2
    assert "return 4" in embedder.passage_batches[-1][-1]


def test_a_branch_moved_before_indexing_raises_without_scanning(tmp_path: Path) -> None:
    root, project = _git_project(tmp_path)
    run_git("checkout", "-qb", "feature", cwd=root)
    run_git("checkout", "-q", "main", cwd=root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    store.upsert_project(project, model_id="test/code", state="pending")
    partition = store.active_partition(project.id)

    run_git("checkout", "-q", "feature", cwd=root)
    with (
        patch.object(
            indexer.scanner, "iter_scan", side_effect=AssertionError("the scan must not start")
        ) as scan,
        pytest.raises(CodeIndexingError) as excinfo,
    ):
        indexer.index(project, partition=partition)

    scan.assert_not_called()
    assert excinfo.value.code is ErrorCode.REPOSITORY_CHANGED
    assert excinfo.value.details["project"] == project.id


def test_a_branch_switch_during_the_scan_discards_the_staged_run(tmp_path: Path) -> None:
    root, project = _git_project(tmp_path)
    run_git("checkout", "-qb", "feature", cwd=root)
    run_git("checkout", "-q", "main", cwd=root)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    store.upsert_project(project, model_id="test/code", state="pending")
    partition = store.active_partition(project.id)
    original_scan = indexer.scanner.iter_scan

    def switching_scan(
        scan_project: ProjectInfo,
        known_files: dict[str, StoredFile] | None = None,
        **kwargs: bool,
    ):
        stream = original_scan(scan_project, known_files, **kwargs)
        for index, item in enumerate(stream):
            if index == 0:
                run_git("checkout", "-q", "feature", cwd=root)
            yield item

    with (
        patch.object(indexer.scanner, "iter_scan", switching_scan),
        pytest.raises(CodeIndexingError) as excinfo,
    ):
        indexer.index(project, partition=partition)

    assert excinfo.value.code is ErrorCode.REPOSITORY_CHANGED
    # No staged row survived into the moved-from slot, the slot is not stuck
    # in "indexing", and no journal remains for recovery to trip over.
    assert store.list_files(project.id, partition_id=partition.partition_id) == []
    slot = store.get_slot(partition.slot_id)
    assert slot is not None
    assert slot.state == "pending"
    assert list((tmp_path / "staging").rglob("job-*")) == []
    # The moved-to branch keeps its own untouched pending slot.
    active = store.active_partition(project.id)
    assert slot.partition_id != active.partition_id


def test_progress_publishes_the_slot_identity(tmp_path: Path) -> None:
    root, project = _git_project(tmp_path)
    indexer, store = make_indexer(tmp_path, RecordingEmbedder())
    snapshots: list[IndexProgress] = []

    indexer.index(project, on_progress=snapshots.append)

    partition = store.active_partition(project.id)
    assert snapshots
    assert {item.slot_id for item in snapshots} == {partition.slot_id}
    assert {item.activation_epoch for item in snapshots} == {partition.activation_epoch}
    assert snapshots[0].selector == "ref:refs/heads/main"
    assert snapshots[0].expected_head == _git_head(root)
