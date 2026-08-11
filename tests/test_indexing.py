import itertools
import os
import time
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pytest
from filelock import FileLock
from lancedb.table import LanceTable
from test_token_batching import fake_encode

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
from code_indexing_mcp.indexing import REFERENCE_SCHEMA_VERSION, Indexer
from code_indexing_mcp.models import ExtractionResult
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


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
    tmp_path: Path, embedder: RecordingEmbedder, *, batch_size: int = 1
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
    rows = tables.chunks.search().select(["path", "embedding_text", "vector"]).to_list()
    assert {row["path"] for row in rows} == {"a.py", "b.py"}
    assert all(row["vector"][0] == float(len(row["embedding_text"]) % 7) for row in rows)


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
                    pa.Table.from_pylist(
                        remaining, schema=LanceStore.reference_arrow_schema()
                    ),
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
    """Divergence tripwire (S1): for every file that is *not* in a known-failed
    state, its committed chunks' content_hash must equal the files row's
    content_hash. A failed file legitimately keeps stale chunks under an
    advanced hash (the non-destructive failure path), but a healthy file with
    mismatched chunk/file hashes is exactly the silent corruption S1 caused --
    catch it here instead of relying on coverage bookkeeping to self-report it.
    """
    files_by_id = {record.file_id: record for record in store.list_files(project_id)}
    for chunk in store.list_chunks([project_id]):
        record = files_by_id.get(chunk.file_id)
        if record is None or record.has_errors:
            continue
        assert chunk.content_hash == record.content_hash, (
            f"chunk {chunk.chunk_id} for {record.path} was staged for content_hash "
            f"{chunk.content_hash!r} but the files row now reports "
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
    assert {chunk.project_id for chunk in store.list_chunks()} == {projects[1].id}
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


def test_every_window_keeps_the_context_header_and_identifier_tail(tmp_path: Path) -> None:
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

    for chunk in chunks:
        assert chunk.embedding_text.startswith("language: python\npath: main.py\n")
        assert "symbol: answer" in chunk.embedding_text
        assert chunk.embedding_text.endswith(chunk.content)
        assert chunk.search_text.startswith(chunk.embedding_text)
        assert chunk.search_text.endswith("main py answer answer")


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
            (chunk.chunk_id, chunk.start_byte, chunk.end_byte, chunk.content, chunk.search_text)
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
