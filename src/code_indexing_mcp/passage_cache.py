"""Bounded, disposable cache for complete passage embedding results.

The cache stores only the output of one candidate's token-window plan. Source
metadata is intentionally absent: callers rebuild file hashes, offsets, chunk
ids, and references from the current checkout before staging a result.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .embedding import EmbeddedSegment, PassageCandidate, SegmentPlan
from .token_batching import MAX_WINDOWS_PER_CANDIDATE

logger = logging.getLogger(__name__)

PASSAGE_CACHE_FORMAT: Final = 1
DEFAULT_MAX_PAYLOAD_BYTES: Final = 256 * 1024**2
DEFAULT_MAX_DATABASE_BYTES: Final = 512 * 1024**2
DEFAULT_MAX_ENTRY_BYTES: Final = 64 * 1024
DEFAULT_BUSY_TIMEOUT_MS: Final = 50
MAX_SQL_KEYS: Final = 100
MAX_WRITE_ENTRIES: Final = 256


@dataclass(frozen=True)
class PassageCacheNamespace:
    """The output contract that must match before a vector can be reused."""

    project_id: str
    artifact_digest: str
    tokenizer_digest: str
    producer: str
    runtime_version: str
    dimension: int
    precision: str
    # Version 2 reserves prefix separators and model special tokens when
    # planning windows; version 1 cached different content boundaries.
    embedding_contract_version: int = 2


@dataclass(frozen=True)
class EncodedSegments:
    """The bounded SQLite payload for one candidate result."""

    metadata: str
    vectors: bytes
    checksum: str
    payload_bytes: int


@dataclass(frozen=True)
class PassageCacheEntry:
    """A complete candidate result ready for durable storage."""

    key: str
    segments: tuple[EmbeddedSegment, ...]


def candidate_key(
    namespace: PassageCacheNamespace,
    candidate: PassageCandidate,
    plan: SegmentPlan,
) -> str:
    """Return a stable key for the candidate and its output contract.

    Batch size and padded-token limits affect scheduling only. The token-window
    policy affects offsets and therefore remains part of the key.
    """
    payload = {
        "namespace": asdict(namespace),
        "prefix": candidate.prefix,
        "content": candidate.content,
        "windowing": [plan.max_tokens, plan.overlap_tokens, plan.max_windows],
        "cache_format": PASSAGE_CACHE_FORMAT,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_segments(
    segments: Sequence[EmbeddedSegment], *, content_length: int, plan: SegmentPlan
) -> None:
    """Validate current candidate coverage before a cached result is staged."""
    if content_length < 0:
        raise ValueError("content_length must not be negative")
    if len(segments) > plan.max_windows:
        raise ValueError("segment count exceeds the current window limit")
    if not segments:
        if content_length:
            raise ValueError("segments do not cover the candidate")
        return

    previous_end = 0
    for index, segment in enumerate(segments):
        if segment.start_char < 0 or segment.end_char < segment.start_char:
            raise ValueError("segment offsets are invalid")
        if segment.end_char > content_length:
            raise ValueError("segment offsets exceed the candidate")
        if index == 0 and segment.start_char != 0:
            raise ValueError("segments do not cover the candidate from the start")
        if index and segment.start_char > previous_end:
            raise ValueError("segments do not cover the candidate")
        if segment.token_count < 0 or segment.token_count > plan.max_tokens:
            raise ValueError("segment token count exceeds the current window policy")
        previous_end = segment.end_char
    if previous_end != content_length:
        raise ValueError("segments do not cover the candidate")


def _vector_bytes(segment: EmbeddedSegment, *, dimension: int) -> bytes:
    vector = bytes(segment.vector)
    expected = dimension * 4
    if len(vector) != expected:
        raise ValueError("vector byte length does not match the cache dimension")
    values = np.frombuffer(vector, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("vectors must contain only finite values")
    if not np.any(values):
        raise ValueError("vectors must be nonzero")
    return vector


def encode_segments(
    segments: Sequence[EmbeddedSegment],
    *,
    dimension: int,
    max_windows: int = MAX_WINDOWS_PER_CANDIDATE,
    max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
) -> EncodedSegments:
    """Encode and validate one complete candidate result."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if len(segments) > max_windows:
        raise ValueError("segment count exceeds the cache window limit")
    metadata_values: list[list[int]] = []
    vectors = bytearray()
    for segment in segments:
        if segment.start_char < 0 or segment.end_char < segment.start_char:
            raise ValueError("segment offsets are invalid")
        if segment.token_count < 0:
            raise ValueError("segment token count must be nonnegative")
        metadata_values.append([segment.start_char, segment.end_char, segment.token_count])
        vectors.extend(_vector_bytes(segment, dimension=dimension))

    metadata_bytes = json.dumps(metadata_values, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    vector_bytes = bytes(vectors)
    payload_bytes = len(metadata_bytes) + len(vector_bytes)
    if payload_bytes > max_entry_bytes:
        raise ValueError("encoded cache entry exceeds the entry-size limit")
    checksum = hashlib.sha256(metadata_bytes + vector_bytes).hexdigest()
    return EncodedSegments(
        metadata=metadata_bytes.decode("utf-8"),
        vectors=vector_bytes,
        checksum=checksum,
        payload_bytes=payload_bytes,
    )


def decode_segments(
    metadata: str,
    vectors: bytes,
    checksum: str,
    *,
    dimension: int,
    max_windows: int = MAX_WINDOWS_PER_CANDIDATE,
    max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
) -> list[EmbeddedSegment]:
    """Decode a row only after checking its bounded, self-authenticated data."""
    metadata_bytes = metadata.encode("utf-8")
    if len(metadata_bytes) + len(vectors) > max_entry_bytes:
        raise ValueError("encoded cache entry exceeds the entry-size limit")
    expected_checksum = hashlib.sha256(metadata_bytes + vectors).hexdigest()
    if not hmac.compare_digest(expected_checksum, checksum):
        raise ValueError("cache entry checksum does not match")
    try:
        values = json.loads(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache entry metadata is not valid JSON") from exc
    if not isinstance(values, list):
        raise ValueError("cache entry metadata must be an array")
    if len(values) > max_windows:
        raise ValueError("segment count exceeds the cache window limit")
    expected_bytes = len(values) * dimension * 4
    if len(vectors) != expected_bytes:
        raise ValueError("vector byte length does not match the segment count")
    if dimension < 1:
        raise ValueError("dimension must be positive")
    vector_values = np.frombuffer(vectors, dtype="<f4")
    if not np.isfinite(vector_values).all():
        raise ValueError("vectors must contain only finite values")
    if values and not np.any(vector_values.reshape((len(values), dimension)), axis=1).all():
        raise ValueError("vectors must be nonzero")

    segments: list[EmbeddedSegment] = []
    for index, raw in enumerate(values):
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        ):
            raise ValueError("cache entry metadata has an invalid segment")
        start_char, end_char, token_count = raw
        if start_char < 0 or end_char < start_char or token_count < 0:
            raise ValueError("cache entry metadata has invalid offsets")
        start = index * dimension * 4
        end = start + dimension * 4
        segments.append(EmbeddedSegment(start_char, end_char, token_count, vectors[start:end]))
    return segments


_DIGEST_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
_DIGEST_LOCK = threading.Lock()


def _stat_key(path: Path, stat_result: os.stat_result) -> tuple[str, int, int, int, int, int]:
    return (
        str(path),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _file_digest(path: Path) -> str | None:
    try:
        before = path.stat()
        key = _stat_key(path, before)
        with _DIGEST_LOCK:
            cached = _DIGEST_CACHE.get(key)
        if cached is not None:
            after = path.stat()
            return cached if _stat_key(path, after) == key else None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if _stat_key(path, after) != key:
            return None
        value = digest.hexdigest()
        with _DIGEST_LOCK:
            _DIGEST_CACHE[key] = value
        return value
    except OSError:
        return None


def artifact_digest(path: Path) -> str | None:
    """Return a content digest, or ``None`` when the artifact is changing."""
    try:
        resolved = path.resolve()
        if resolved.is_file():
            return _file_digest(resolved)
        if not resolved.is_dir():
            return None
        root_stat = resolved.stat()
        files = [child for child in resolved.rglob("*") if child.is_file()]
        parts: list[tuple[str, str]] = []
        for child in sorted(files):
            digest = _file_digest(child)
            if digest is None:
                return None
            parts.append((child.relative_to(resolved).as_posix(), digest))
        if _stat_key(resolved, resolved.stat()) != _stat_key(resolved, root_stat):
            return None
        encoded = json.dumps(parts, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except OSError:
        return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    cache_key TEXT PRIMARY KEY,
    metadata TEXT NOT NULL,
    vectors BLOB NOT NULL,
    checksum TEXT NOT NULL,
    payload_bytes INTEGER NOT NULL,
    written_at_ns INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_age ON entries(written_at_ns, cache_key);
CREATE TABLE IF NOT EXISTS cache_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL,
    payload_bytes INTEGER NOT NULL
);
"""


class PassageEmbeddingCache:
    """A run-scoped adapter around one bounded SQLite cache file."""

    def __init__(
        self,
        path: Path,
        *,
        dimension: int,
        max_windows: int = MAX_WINDOWS_PER_CANDIDATE,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_database_bytes: int = DEFAULT_MAX_DATABASE_BYTES,
        max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if min(max_windows, max_payload_bytes, max_database_bytes, max_entry_bytes) < 1:
            raise ValueError("cache limits must be positive")
        if busy_timeout_ms < 0:
            raise ValueError("busy timeout must not be negative")
        self.path = path
        self.dimension = dimension
        self.max_windows = max_windows
        self.max_payload_bytes = max_payload_bytes
        self.max_database_bytes = max_database_bytes
        self.max_entry_bytes = max_entry_bytes
        self.busy_timeout_ms = busy_timeout_ms
        self.payload_bytes = 0
        self.disabled = False
        self._connection: sqlite3.Connection | None = None
        self._reported_failure = False

    def __enter__(self) -> PassageEmbeddingCache:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                self._disable("could not close passage embedding cache")

    def _disable(self, message: str) -> None:
        self.disabled = True
        if not self._reported_failure:
            self._reported_failure = True
            logger.warning("%s at %s; continuing without passage reuse", message, self.path)
        connection = self._connection
        self._connection = None
        if connection is not None:
            with suppress(sqlite3.Error):
                connection.close()

    def _ensure_open(self) -> sqlite3.Connection | None:
        if self.disabled:
            return None
        if self._connection is not None:
            return self._connection
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1000,
                isolation_level=None,
            )
            connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
            connection.execute("PRAGMA journal_mode = DELETE")
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            max_pages = max(1, self.max_database_bytes // page_size)
            connection.execute(f"PRAGMA max_page_count = {max_pages}")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO cache_meta(singleton, format_version, payload_bytes) "
                "VALUES (1, ?, 0)",
                (PASSAGE_CACHE_FORMAT,),
            )
            meta = connection.execute(
                "SELECT format_version, payload_bytes FROM cache_meta WHERE singleton = 1"
            ).fetchone()
            if meta is None or int(meta[0]) != PASSAGE_CACHE_FORMAT:
                raise sqlite3.DatabaseError("unsupported passage cache format")
            self.payload_bytes = int(meta[1])
            self._connection = connection
            return connection
        except (OSError, sqlite3.Error):
            self._disable("could not open passage embedding cache")
            return None

    def get_many(self, keys: Sequence[str]) -> dict[str, list[EmbeddedSegment]]:
        """Return validated hits for a bounded list of keys."""
        connection = self._ensure_open()
        if connection is None:
            return {}
        hits: dict[str, list[EmbeddedSegment]] = {}
        unique_keys = list(dict.fromkeys(keys))
        try:
            for start in range(0, len(unique_keys), MAX_SQL_KEYS):
                group = unique_keys[start : start + MAX_SQL_KEYS]
                placeholders = ",".join("?" for _ in group)
                rows = connection.execute(
                    "SELECT cache_key, metadata, vectors, checksum, payload_bytes "
                    f"FROM entries WHERE cache_key IN ({placeholders}) "
                    "AND payload_bytes <= ? "
                    "AND length(metadata) + length(vectors) <= ?",
                    [*group, self.max_entry_bytes, self.max_entry_bytes],
                )
                for key, metadata, vectors, checksum, payload_bytes in rows:
                    if int(payload_bytes) > self.max_entry_bytes:
                        continue
                    try:
                        hits[str(key)] = decode_segments(
                            str(metadata),
                            bytes(vectors),
                            str(checksum),
                            dimension=self.dimension,
                            max_windows=self.max_windows,
                            max_entry_bytes=self.max_entry_bytes,
                        )
                    except ValueError:
                        continue
        except (OSError, sqlite3.Error):
            self._disable("could not read passage embedding cache")
            return {}
        return hits

    def put_many(
        self,
        entries: Mapping[str, Sequence[EmbeddedSegment]] | Sequence[PassageCacheEntry],
    ) -> None:
        """Persist complete entries in bounded transactions."""
        connection = self._ensure_open()
        if connection is None:
            return
        if isinstance(entries, Mapping):
            values = [
                PassageCacheEntry(str(key), tuple(segments)) for key, segments in entries.items()
            ]
        else:
            values = list(entries)
        for start in range(0, len(values), MAX_WRITE_ENTRIES):
            self._put_group(connection, values[start : start + MAX_WRITE_ENTRIES])

    def _put_group(
        self, connection: sqlite3.Connection, entries: Sequence[PassageCacheEntry]
    ) -> None:
        encoded: list[tuple[PassageCacheEntry, EncodedSegments]] = []
        for entry in entries:
            try:
                value = encode_segments(
                    entry.segments,
                    dimension=self.dimension,
                    max_windows=self.max_windows,
                    max_entry_bytes=self.max_entry_bytes,
                )
            except ValueError:
                continue
            encoded.append((entry, value))
        if not encoded:
            return
        try:
            connection.execute("BEGIN IMMEDIATE")
            # Another run can share this cache with a different data directory.
            # Read its committed total only after acquiring the writer lock.
            payload_bytes = int(
                connection.execute(
                    "SELECT payload_bytes FROM cache_meta WHERE singleton = 1"
                ).fetchone()[0]
            )
            for entry, value in encoded:
                if value.payload_bytes > self.max_payload_bytes:
                    continue
                previous = connection.execute(
                    "SELECT payload_bytes FROM entries WHERE cache_key = ?", (entry.key,)
                ).fetchone()
                if previous is not None:
                    connection.execute("DELETE FROM entries WHERE cache_key = ?", (entry.key,))
                    payload_bytes -= int(previous[0])
                while payload_bytes + value.payload_bytes > self.max_payload_bytes:
                    old_rows = connection.execute(
                        "SELECT cache_key, payload_bytes FROM entries "
                        "ORDER BY written_at_ns, cache_key LIMIT ?",
                        (MAX_SQL_KEYS,),
                    ).fetchall()
                    if not old_rows:
                        break
                    for old_key, old_size in old_rows:
                        connection.execute("DELETE FROM entries WHERE cache_key = ?", (old_key,))
                        payload_bytes -= int(old_size)
                        if payload_bytes + value.payload_bytes <= self.max_payload_bytes:
                            break
                if payload_bytes + value.payload_bytes > self.max_payload_bytes:
                    continue
                connection.execute(
                    "INSERT INTO entries "
                    "(cache_key, metadata, vectors, checksum, payload_bytes, written_at_ns) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        entry.key,
                        value.metadata,
                        value.vectors,
                        value.checksum,
                        value.payload_bytes,
                        time.time_ns(),
                    ),
                )
                payload_bytes += value.payload_bytes
            connection.execute(
                "UPDATE cache_meta SET payload_bytes = ? WHERE singleton = 1", (payload_bytes,)
            )
            connection.execute("COMMIT")
            self.payload_bytes = payload_bytes
        except (OSError, sqlite3.Error):
            with suppress(sqlite3.Error):
                connection.execute("ROLLBACK")
            self._disable("could not write passage embedding cache")


class PassageReuseContext:
    """Resolve complete candidate results for one indexing run.

    The context owns the cache connection and all counters for one run. It is
    intentionally separate from ``PassageEmbeddingCache`` so a caller can
    bypass reuse without opening the durable file, and so cache failures stay
    an optimization concern while embedding errors continue through the
    indexer's normal error handling.
    """

    def __init__(
        self,
        path: Path,
        namespace: PassageCacheNamespace | None,
        *,
        force: bool = False,
        strict: bool = False,
        producer_matches: Callable[[object], bool] | None = None,
        max_windows: int = MAX_WINDOWS_PER_CANDIDATE,
    ) -> None:
        self.path = path
        self.namespace = namespace
        self.force = force
        self.strict = strict
        self.producer_matches = producer_matches
        self.max_windows = max_windows
        self.status = (
            "bypassed" if force or strict else "disabled" if namespace is None else "pending"
        )
        self.reused_candidates = 0
        self.reused_segments = 0
        self.lookup_duration_ns = 0
        self.write_duration_ns = 0
        self._cache: PassageEmbeddingCache | None = None

    def __enter__(self) -> PassageReuseContext:
        if self.status != "pending" or self.namespace is None:
            return self
        cache = PassageEmbeddingCache(
            self.path,
            dimension=self.namespace.dimension,
            max_windows=self.max_windows,
        )
        cache.__enter__()
        if cache.disabled:
            self.status = "error"
            cache.close()
        else:
            self._cache = cache
            self.status = "active"
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    @property
    def lookup_duration_ms(self) -> int:
        return self.lookup_duration_ns // 1_000_000

    @property
    def write_duration_ms(self) -> int:
        return self.write_duration_ns // 1_000_000

    def lookup(
        self,
        candidates: Sequence[PassageCandidate],
        plan: SegmentPlan,
        *,
        producer: object | None = None,
    ) -> tuple[dict[int, list[EmbeddedSegment]], list[int]]:
        """Return validated hits and the original positions of misses."""
        misses = list(range(len(candidates)))
        cache = self._cache
        namespace = self.namespace
        if self.status != "active" or cache is None or namespace is None or not candidates:
            return {}, misses
        if self.producer_matches is not None and not self.producer_matches(producer):
            return {}, misses
        if getattr(producer, "tokenizer_available", None) is not True:
            return {}, misses
        keys = [candidate_key(namespace, candidate, plan) for candidate in candidates]
        started = time.monotonic_ns()
        values = cache.get_many(keys)
        self.lookup_duration_ns += time.monotonic_ns() - started
        if cache.disabled:
            self.status = "error"
            return {}, misses
        hits: dict[int, list[EmbeddedSegment]] = {}
        misses = []
        for index, candidate in enumerate(candidates):
            segments = values.get(keys[index])
            if segments is None:
                misses.append(index)
                continue
            try:
                validate_segments(segments, content_length=len(candidate.content), plan=plan)
            except ValueError:
                misses.append(index)
                continue
            hits[index] = segments
        self.reused_candidates += len(hits)
        self.reused_segments += sum(len(segments) for segments in hits.values())
        return hits, misses

    def store(
        self,
        candidates: Sequence[PassageCandidate],
        results: Mapping[int, Sequence[EmbeddedSegment]],
        plan: SegmentPlan,
        *,
        producer: object | None = None,
    ) -> None:
        """Store complete successful misses when the producer is still known."""
        cache = self._cache
        namespace = self.namespace
        if self.status != "active" or cache is None or namespace is None or not results:
            return
        if self.producer_matches is not None and not self.producer_matches(producer):
            return
        if getattr(producer, "tokenizer_available", None) is not True:
            return
        entries: dict[str, Sequence[EmbeddedSegment]] = {}
        for index, segments in results.items():
            if index < 0 or index >= len(candidates):
                continue
            try:
                validate_segments(
                    segments, content_length=len(candidates[index].content), plan=plan
                )
            except ValueError:
                continue
            entries[candidate_key(namespace, candidates[index], plan)] = segments
        if not entries:
            return
        started = time.monotonic_ns()
        cache.put_many(entries)
        self.write_duration_ns += time.monotonic_ns() - started
        if cache.disabled:
            self.status = "error"
