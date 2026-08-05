"""Journalled Arrow staging for crash-recoverable index commits.

An index run never touches the live Lance tables while it scans, parses, and
embeds. It streams file, chunk, and structural-reference rows into Arrow IPC files under
``<data>/staging/<project-id>/<job-id>/`` and records its progress in
``journal.json``. Only once staging finishes does the run record the live
tables' versions, switch the journal to ``committing``, and apply the staged
batches. A crash before that switch leaves the live tables untouched; a crash
after it is rolled back to the recorded versions on the next startup, so the
previously searchable generation of the project survives either way.

Every file write goes through a temporary sibling, an ``fsync``, and an atomic
rename, so a journal or Arrow payload is never observed half-written.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa

from .models import StoredFile
from .storage import LanceStore, TableVersions

logger = logging.getLogger(__name__)

JOURNAL_NAME = "journal.json"
FILES_NAME = "files.arrow"
CHUNKS_NAME = "chunks.arrow"
REFERENCES_NAME = "references.arrow"

PHASE_STAGING = "staging"
PHASE_COMMITTING = "committing"
PHASE_COMPLETE = "complete"
PHASE_ROLLED_BACK = "rolled_back"

LEGACY_JOURNAL_FORMAT_VERSION = 1
JOURNAL_FORMAT_VERSION = 2

# A rollback that keeps failing must not be retried forever: every startup
# would pay for it, under the global index lock, with no prospect of success.
# Lance prunes versions older than a day during compaction, so a journal that
# survives long enough can name a version that no longer exists. Retry a few
# times to ride out transient I/O, then give up loudly.
MAX_RECOVERY_ATTEMPTS = 3


@dataclass(frozen=True)
class ChunkRow:
    """One staged chunk: the StoredChunk fields, with the vector still packed.

    ``vector`` is contiguous little-endian float32 bytes exactly as the
    embedding worker returned them, so the write path never materializes a
    list of Python floats per chunk.
    """

    chunk_id: str
    file_id: str
    project_id: str
    path: str
    language: str
    kind: str
    symbol: str | None
    qualified_symbol: str | None
    parent_symbol: str | None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    content: str
    embedding_text: str
    search_text: str
    content_hash: str
    part_index: int
    vector: bytes


@dataclass(frozen=True)
class ReferenceRow:
    """One structural reference, declaration shape, or coverage record."""

    reference_id: str
    record_kind: str
    file_id: str
    project_id: str
    path: str
    language: str
    kind: str | None
    source_qualified_symbol: str | None
    written_name: str | None
    target_name: str | None
    module_path: str | None
    imported_name: str | None
    alias: str | None
    receiver_text: str | None
    start_byte: int | None
    end_byte: int | None
    start_line: int | None
    end_line: int | None
    shape_json: str | None
    content_hash: str
    schema_version: int


def _write_atomically(path: Path, payload: bytes) -> None:
    """Write *payload* to *path* through a temporary sibling and fsync."""
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as sink:
        sink.write(payload)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory so renames inside it are durable."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class StagingJob:
    """One index run's staged output and its journal.

    Writers open in :meth:`begin` and stay open while the run stages rows;
    :meth:`begin_commit` finalizes the Arrow payloads and records the live
    table versions the commit must be able to roll back to. Batches are
    written per embedding group of a single file, so a record batch never
    spans two files -- the commit can stream them back one file at a time.
    """

    def __init__(
        self,
        staging_root: Path,
        project_id: str,
        *,
        file_schema: pa.Schema,
        chunk_schema: pa.Schema,
        reference_schema: pa.Schema,
        job_id: str | None = None,
    ) -> None:
        self.staging_root = staging_root
        self.project_id = project_id
        self.job_id = job_id or f"{time.time_ns():x}-{os.getpid()}"
        self._file_schema = file_schema
        self._chunk_schema = chunk_schema
        self._reference_schema = reference_schema
        self.replace_file_ids: list[str] = []
        self.replace_reference_file_ids: list[str] = []
        self._reference_file_ids: set[str] = set()
        self._last_reference_file_id: str | None = None
        self.removed_file_ids: list[str] = []
        self._journal: dict[str, Any] = {}
        self._files_sink: Any = None
        self._chunks_sink: Any = None
        self._references_sink: Any = None
        self._files_writer: pa.RecordBatchWriter | None = None
        self._chunks_writer: pa.RecordBatchWriter | None = None
        self._references_writer: pa.RecordBatchWriter | None = None

    @property
    def directory(self) -> Path:
        return self.staging_root / self.project_id / self.job_id

    def begin(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._journal = {
            "version": JOURNAL_FORMAT_VERSION,
            "job_id": self.job_id,
            "project_id": self.project_id,
            "phase": PHASE_STAGING,
            "created_at_ns": time.time_ns(),
        }
        self._write_journal()
        self._files_sink, self._files_writer = self._open_writer(FILES_NAME, self._file_schema)
        self._chunks_sink, self._chunks_writer = self._open_writer(CHUNKS_NAME, self._chunk_schema)
        self._references_sink, self._references_writer = self._open_writer(
            REFERENCES_NAME, self._reference_schema
        )

    def stage_file(self, record: StoredFile) -> None:
        """Stage one file record. Files are few and carry no vectors."""
        assert self._files_writer is not None
        batch = pa.RecordBatch.from_pylist([record.model_dump()], schema=self._file_schema)
        self._files_writer.write_batch(batch)

    def stage_chunks(self, rows: list[ChunkRow]) -> None:
        """Stage one embedding group's chunk rows as a single record batch."""
        assert self._chunks_writer is not None
        if not rows:
            return
        if any(row.file_id != rows[0].file_id for row in rows[1:]):
            raise ValueError("A staged chunk batch must contain rows from one file")
        vector_type = self._chunk_schema.field("vector").type
        dimension = vector_type.list_size
        columns: list[pa.Array] = []
        for field in self._chunk_schema:
            if field.name == "vector":
                packed = b"".join(row.vector for row in rows)
                flat = np.frombuffer(packed, dtype="<f4")
                values = pa.array(flat, type=pa.float32())
                columns.append(pa.FixedSizeListArray.from_arrays(values, dimension))
            else:
                columns.append(
                    pa.array([getattr(row, field.name) for row in rows], type=field.type)
                )
        self._chunks_writer.write_batch(pa.record_batch(columns, schema=self._chunk_schema))

    def stage_references(self, rows: list[ReferenceRow]) -> None:
        """Stage one file's structural rows as one ordered Arrow batch.

        Calls for a file may be repeated while contiguous, but once another
        file is staged that file is closed for this job. This lets commit stream
        one complete file group at a time without retaining the whole staged
        reference dataset in memory.
        """
        assert self._references_writer is not None
        if not rows:
            return
        file_id = rows[0].file_id
        if any(row.file_id != file_id for row in rows[1:]):
            raise ValueError("A staged reference batch must contain rows from one file")
        if file_id in self._reference_file_ids and file_id != self._last_reference_file_id:
            raise ValueError("Staged reference batches for a file must be contiguous")
        self._reference_file_ids.add(file_id)
        self._last_reference_file_id = file_id
        columns = [
            pa.array([getattr(row, field.name) for row in rows], type=field.type)
            for field in self._reference_schema
        ]
        self._references_writer.write_batch(
            pa.record_batch(columns, schema=self._reference_schema)
        )

    def mark_replaced(self, file_id: str) -> None:
        if file_id not in self.replace_file_ids:
            self.replace_file_ids.append(file_id)

    def mark_references_replaced(self, file_id: str) -> None:
        """Mark one file's structural rows for replacement, independently of chunks."""
        if file_id not in self.replace_reference_file_ids:
            self.replace_reference_file_ids.append(file_id)

    def mark_removed(self, file_id: str) -> None:
        self.removed_file_ids.append(file_id)

    def begin_commit(self, versions: TableVersions) -> None:
        """Finalize the payloads and record the versions a rollback restores."""
        self._close_writers()
        self._journal.update(
            {
                "phase": PHASE_COMMITTING,
                "files_version": versions.files,
                "chunks_version": versions.chunks,
                "references_version": versions.references,
                "replace_file_ids": self.replace_file_ids,
                "replace_reference_file_ids": self.replace_reference_file_ids,
                "removed_file_ids": self.removed_file_ids,
            }
        )
        self._write_journal()

    def files_table(self) -> pa.Table:
        """Read back every staged file record. One row per file; small."""
        return pa.ipc.open_file(self.directory / FILES_NAME).read_all()

    def iter_chunk_groups(self) -> Iterator[tuple[str, pa.Table]]:
        """Yield ``(file_id, table)`` for each replaced file, one at a time.

        Staged rows for files absent from ``replace_file_ids`` -- files that
        failed mid-run -- are skipped, and a replaced file with no staged rows
        yields an empty table so the commit deletes its previous chunks.
        """
        wanted = set(self.replace_file_ids)
        seen: set[str] = set()
        current_file_id: str | None = None
        batches: list[pa.RecordBatch] = []
        reader = pa.ipc.open_file(self.directory / CHUNKS_NAME)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            file_id = cast(str, batch.column("file_id")[0].as_py())
            if file_id != current_file_id:
                if batches:
                    assert current_file_id is not None
                    seen.add(current_file_id)
                    yield current_file_id, pa.Table.from_batches(batches, schema=self._chunk_schema)
                    batches = []
                current_file_id = file_id
            if file_id in wanted:
                batches.append(batch)
        if batches:
            assert current_file_id is not None
            seen.add(current_file_id)
            yield current_file_id, pa.Table.from_batches(batches, schema=self._chunk_schema)
        empty = pa.Table.from_batches([], schema=self._chunk_schema)
        for file_id in self.replace_file_ids:
            if file_id not in seen:
                yield file_id, empty

    def iter_reference_groups(self) -> Iterator[tuple[str, pa.Table]]:
        """Yield one complete staged structural table per replaced file.

        :meth:`stage_references` enforces one file per contiguous batch, so
        only the current file's Arrow batches are retained while streaming the
        staged IPC file back into the commit.
        """
        wanted = set(self.replace_reference_file_ids)
        seen: set[str] = set()
        current_file_id: str | None = None
        batches: list[pa.RecordBatch] = []
        reader = pa.ipc.open_file(self.directory / REFERENCES_NAME)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            file_id = cast(str, batch.column("file_id")[0].as_py())
            if file_id != current_file_id:
                if batches:
                    assert current_file_id is not None
                    seen.add(current_file_id)
                    yield current_file_id, pa.Table.from_batches(
                        batches, schema=self._reference_schema
                    )
                    batches = []
                current_file_id = file_id
            if file_id in wanted:
                batches.append(batch)
        if batches:
            assert current_file_id is not None
            seen.add(current_file_id)
            yield current_file_id, pa.Table.from_batches(batches, schema=self._reference_schema)
        empty = pa.Table.from_batches([], schema=self._reference_schema)
        for file_id in self.replace_reference_file_ids:
            if file_id not in seen:
                yield file_id, empty

    def complete(self) -> None:
        """Mark the commit successful and remove the staged directory."""
        self._journal["phase"] = PHASE_COMPLETE
        self._write_journal()
        shutil.rmtree(self.directory)

    def rolled_back(self) -> None:
        """Mark the commit rolled back and remove the staged directory."""
        self._journal["phase"] = PHASE_ROLLED_BACK
        self._write_journal()
        shutil.rmtree(self.directory)

    def discard(self) -> None:
        """Abandon a job that never started committing; live tables untouched.

        Once :meth:`begin_commit` has switched the journal to ``committing``
        the directory is the only record of which versions the live tables
        must return to. A commit whose own rollback failed leaves the journal
        in that phase deliberately, so discarding it here would strand a
        half-committed project with no way back -- keep it for recovery.
        """
        if self._journal.get("phase") == PHASE_COMMITTING:
            logger.warning(
                "Keeping staged commit %s: its rollback did not finish, so startup "
                "recovery still needs the journal",
                self.directory,
            )
            return
        try:
            self._close_writers(finalize=False)
        except Exception:
            logger.debug("Discarding staging writers failed", exc_info=True)
        shutil.rmtree(self.directory, ignore_errors=True)

    def _open_writer(self, name: str, schema: pa.Schema) -> tuple[Any, pa.RecordBatchWriter]:
        temporary = self.directory / f"{name}.tmp"
        sink = temporary.open("wb")
        return sink, pa.ipc.new_file(sink, schema)

    def _close_writers(self, *, finalize: bool = True) -> None:
        for name, sink, writer in (
            (FILES_NAME, self._files_sink, self._files_writer),
            (CHUNKS_NAME, self._chunks_sink, self._chunks_writer),
            (REFERENCES_NAME, self._references_sink, self._references_writer),
        ):
            if writer is None:
                continue
            writer.close()
            sink.flush()
            os.fsync(sink.fileno())
            sink.close()
            if finalize:
                os.replace(
                    self.directory / f"{name}.tmp",
                    self.directory / name,
                )
        self._files_sink = self._chunks_sink = self._references_sink = None
        self._files_writer = self._chunks_writer = self._references_writer = None
        if finalize:
            _sync_directory(self.directory)

    def _write_journal(self) -> None:
        payload = json.dumps(self._journal, indent=2, sort_keys=True).encode()
        _write_atomically(self.directory / JOURNAL_NAME, payload)


def recover_staged_commits(staging_root: Path, store: LanceStore) -> int:
    """Roll back every interrupted commit before queries are accepted.

    A journal still in ``staging`` never reached the live tables, and one in a
    terminal phase only failed to clean up, so both are simply removed. A
    journal in ``committing`` means a crash after live writes began: restore
    the tables recorded by that journal generation, mark the journal ``rolled_back``,
    and remove the directory. Restoring an already-restored table is safe, so
    repeated recovery over the same journal is idempotent.

    A rollback that fails is retried on later startups, but only up to
    :data:`MAX_RECOVERY_ATTEMPTS`. Past that the journal is retired and its
    project marked ``error``: the rollback is never going to succeed, and
    retrying it forever would block every startup behind a lost cause.
    """
    if not staging_root.is_dir():
        return 0
    recovered = 0
    for journal_path in sorted(staging_root.glob(f"*/*/{JOURNAL_NAME}")):
        directory = journal_path.parent
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable staging journal: %s", journal_path)
            continue
        if journal.get("phase") != PHASE_COMMITTING:
            shutil.rmtree(directory, ignore_errors=True)
            continue
        project_id = str(journal["project_id"])
        try:
            journal_version = int(journal.get("version", LEGACY_JOURNAL_FORMAT_VERSION))
            if journal_version == LEGACY_JOURNAL_FORMAT_VERSION:
                versions = TableVersions(
                    files=int(journal["files_version"]),
                    chunks=int(journal["chunks_version"]),
                    references=0,
                )
            elif journal_version == JOURNAL_FORMAT_VERSION:
                versions = TableVersions(
                    files=int(journal["files_version"]),
                    chunks=int(journal["chunks_version"]),
                    references=int(journal["references_version"]),
                )
            else:
                raise ValueError(f"Unsupported staging journal version: {journal_version}")
            restored = store.restore_versions(
                project_id,
                versions,
                restore_references=journal_version != LEGACY_JOURNAL_FORMAT_VERSION,
            )
        except Exception:
            attempts = int(journal.get("recovery_attempts", 0)) + 1
            if attempts < MAX_RECOVERY_ATTEMPTS:
                # Could be transient. Record the attempt so the retries are
                # bounded even though the journal survives this startup.
                logger.exception(
                    "Could not roll back interrupted commit in %s (attempt %d of %d)",
                    directory,
                    attempts,
                    MAX_RECOVERY_ATTEMPTS,
                )
                journal["recovery_attempts"] = attempts
                _write_atomically(
                    journal_path, json.dumps(journal, indent=2, sort_keys=True).encode()
                )
                continue
            logger.exception(
                "Giving up on the interrupted commit in %s after %d attempts. Project "
                "%s may hold partially committed data; re-index it to rebuild.",
                directory,
                attempts,
                project_id,
            )
            # Flagging the project is a courtesy to the operator; recovery runs
            # during startup, so it must not fail construction if this does.
            with contextlib.suppress(Exception):
                store.mark_project_state(project_id, "error")
            shutil.rmtree(directory, ignore_errors=True)
            continue
        if not restored:
            # The project was removed after the journal was written, so there
            # is no longer anything to roll back.
            logger.info(
                "Discarding staged commit for removed project %s in %s", project_id, directory
            )
            shutil.rmtree(directory, ignore_errors=True)
            continue
        journal["phase"] = PHASE_ROLLED_BACK
        _write_atomically(journal_path, json.dumps(journal, indent=2, sort_keys=True).encode())
        shutil.rmtree(directory, ignore_errors=True)
        recovered += 1
    return recovered
