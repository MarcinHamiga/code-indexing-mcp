"""Journalled Arrow staging for crash-recoverable index commits.

An index run never touches the live Lance tables while it scans, parses, and
embeds. It streams file and chunk rows into Arrow IPC files under
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

PHASE_STAGING = "staging"
PHASE_COMMITTING = "committing"
PHASE_COMPLETE = "complete"
PHASE_ROLLED_BACK = "rolled_back"

JOURNAL_FORMAT_VERSION = 1


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
        job_id: str | None = None,
    ) -> None:
        self.staging_root = staging_root
        self.project_id = project_id
        self.job_id = job_id or f"{time.time_ns():x}-{os.getpid()}"
        self._file_schema = file_schema
        self._chunk_schema = chunk_schema
        self.replace_file_ids: list[str] = []
        self.removed_file_ids: list[str] = []
        self._journal: dict[str, Any] = {}
        self._files_sink: Any = None
        self._chunks_sink: Any = None
        self._files_writer: pa.RecordBatchWriter | None = None
        self._chunks_writer: pa.RecordBatchWriter | None = None

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

    def mark_replaced(self, file_id: str) -> None:
        self.replace_file_ids.append(file_id)

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
                "replace_file_ids": self.replace_file_ids,
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
        """Abandon a job that never started committing; live tables untouched."""
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
        self._files_sink = self._chunks_sink = None
        self._files_writer = self._chunks_writer = None
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
    both tables to the recorded versions, mark the journal ``rolled_back``,
    and remove the directory. Restoring an already-restored table is safe, so
    repeated recovery over the same journal is idempotent.
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
        try:
            store.restore_versions(
                str(journal["project_id"]),
                TableVersions(
                    files=int(journal["files_version"]),
                    chunks=int(journal["chunks_version"]),
                ),
            )
        except Exception:
            # Leave the journal in place so the next startup retries instead of
            # silently serving a half-committed project.
            logger.exception("Could not roll back interrupted commit in %s", directory)
            continue
        journal["phase"] = PHASE_ROLLED_BACK
        _write_atomically(journal_path, json.dumps(journal, indent=2, sort_keys=True).encode())
        shutil.rmtree(directory, ignore_errors=True)
        recovered += 1
    return recovered
