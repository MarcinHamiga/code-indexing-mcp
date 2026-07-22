"""Explicit incremental indexing orchestration."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from filelock import FileLock, Timeout

from .embedding import Embedder
from .errors import ErrorCode, IncodeError
from .extractor import TreeSitterExtractor
from .models import IndexIssue, IndexReport, ProjectInfo, StoredChunk, StoredFile
from .scanner import SourceScanner
from .storage import LanceStore


def _digest(value: str | bytes) -> str:
    data = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


class Indexer:
    def __init__(
        self,
        *,
        store: LanceStore,
        scanner: SourceScanner,
        extractor: TreeSitterExtractor,
        embedder: Embedder,
        lock_directory: Path,
    ) -> None:
        self.store = store
        self.scanner = scanner
        self.extractor = extractor
        self.embedder = embedder
        self.lock_directory = lock_directory

    def index(self, project: ProjectInfo, *, force: bool = False) -> IndexReport:
        started = time.monotonic_ns()
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.lock_directory / f"{project.id}.lock")
        try:
            with lock.acquire(timeout=0):
                report = self._index_locked(project, force=force)
        except Timeout as exc:
            raise IncodeError(
                ErrorCode.INDEX_BUSY,
                f"Project is already being indexed: {project.name}",
                project=project.id,
            ) from exc
        duration_ms = (time.monotonic_ns() - started) // 1_000_000
        return report.model_copy(update={"duration_ms": duration_ms})

    def _index_locked(self, project: ProjectInfo, *, force: bool) -> IndexReport:
        self.store.upsert_project(project, model_id=self.embedder.model_id, state="indexing")
        scan = self.scanner.scan(project)
        existing = {record.path: record for record in self.store.list_files(project.id)}
        current_paths = {item.path.as_posix() for item in scan.files}
        indexed = parsed = embedded = unchanged = metadata_only = removed = 0
        errors: list[IndexIssue] = []

        for item in scan.files:
            path = item.path.as_posix()
            previous = existing.get(path)
            if (
                not force
                and previous is not None
                and previous.size == item.size
                and previous.mtime_ns == item.mtime_ns
            ):
                unchanged += 1
                continue
            try:
                source = item.absolute_path.read_bytes()
                content_hash = _digest(source)
                if not force and previous is not None and previous.content_hash == content_hash:
                    self.store.upsert_file(
                        previous.model_copy(update={"size": item.size, "mtime_ns": item.mtime_ns})
                    )
                    metadata_only += 1
                    continue
                extraction = self.extractor.extract(item.path, item.language, source)
                parsed += 1
                vectors: list[list[float]] = []
                texts = [chunk.embedding_text for chunk in extraction.chunks]
                for offset in range(0, len(texts), 32):
                    vectors.extend(self.embedder.embed_passages(texts[offset : offset + 32]))
                stored_file = StoredFile(
                    file_id=_digest(f"{project.id}\0{path}"),
                    project_id=project.id,
                    path=path,
                    language=item.language,
                    size=item.size,
                    mtime_ns=item.mtime_ns,
                    content_hash=content_hash,
                    has_errors=extraction.has_errors,
                    indexed_at=time.time_ns(),
                )
                chunks = [
                    self._stored_chunk(project.id, stored_file, chunk, vector)
                    for chunk, vector in zip(extraction.chunks, vectors, strict=True)
                ]
                self.store.replace_file(stored_file, chunks)
                indexed += 1
                embedded += len(chunks)
            except Exception as exc:
                errors.append(IndexIssue(path=path, message=str(exc)))

        for path, record in existing.items():
            if path not in current_paths:
                self.store.remove_file(project.id, record.file_id)
                removed += 1

        self.store.upsert_project(
            project,
            model_id=self.embedder.model_id,
            state="partial" if errors else "ready",
        )
        return IndexReport(
            project_id=project.id,
            discovered_files=len(scan.files),
            indexed_files=indexed,
            parsed_files=parsed,
            embedded_chunks=embedded,
            unchanged_files=unchanged,
            metadata_only_files=metadata_only,
            removed_files=removed,
            skipped_files=len(scan.skipped),
            errors=errors,
        )

    @staticmethod
    def _stored_chunk(
        project_id: str, file: StoredFile, chunk: object, vector: list[float]
    ) -> StoredChunk:
        from .models import ExtractedChunk

        extracted = ExtractedChunk.model_validate(chunk)
        identity = "\0".join(
            [
                file.file_id,
                extracted.kind,
                extracted.qualified_symbol or "",
                str(extracted.start_byte),
                str(extracted.end_byte),
                str(extracted.part_index),
            ]
        )
        return StoredChunk(
            chunk_id=_digest(identity),
            file_id=file.file_id,
            project_id=project_id,
            path=file.path,
            language=file.language,
            kind=extracted.kind,
            symbol=extracted.symbol,
            qualified_symbol=extracted.qualified_symbol,
            parent_symbol=extracted.parent_symbol,
            start_byte=extracted.start_byte,
            end_byte=extracted.end_byte,
            start_line=extracted.start_line,
            end_line=extracted.end_line,
            content=extracted.content,
            embedding_text=extracted.embedding_text,
            search_text=extracted.search_text,
            content_hash=file.content_hash,
            part_index=extracted.part_index,
            vector=vector,
        )
