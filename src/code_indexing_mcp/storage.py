"""Partitioned LanceDB persistence for projects, files, chunks, and references."""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict, cast

import lancedb
import pyarrow as pa
from filelock import FileLock
from lancedb.index import FTS, BTree, HnswSq
from lancedb.query import ColumnOrdering, FullTextOperator, MultiMatchQuery
from lancedb.table import LanceTable

from .errors import CodeIndexingError, ErrorCode
from .git_state import (
    GitProbeOutcome,
    GitState,
    SelectorKind,
    probe_git_state,
)
from .git_state import (
    checkout_key as _checkout_key,
)
from .git_state import (
    partition_id as _git_partition_id,
)
from .git_state import (
    slot_id as _git_slot_id,
)
from .models import (
    ChunkPreview,
    CodeChunk,
    FragmentLengthStats,
    FragmentStats,
    IndexedChunk,
    IndexStorageStats,
    ProjectInfo,
    ProjectSlot,
    ProjectStorageStats,
    SlotStorageStats,
    StoredChunk,
    StoredFile,
    TableStorageStats,
)
from .projects import rooted_under, same_project_root

logger = logging.getLogger(__name__)

# Version 5 stores chunk vectors as float16. The precision benchmark measured
# no recall or rank loss against a float32-exact reference while roughly
# halving vector bytes, so float16 is the write default; float32 remains
# available through CODE_INDEXING_VECTOR_STORAGE. The bump keeps a pre-float16
# binary from serving a float16 partition: it sees version 5 against its own 4
# and marks the partition for rebuild instead of mixing generations.
SCHEMA_VERSION = 5

# Symbol lookups over-fetch because the LIKE pushdown over-matches; these bound
# how many rows are scanned before the exact filter and the caller's limit apply.
OVERFETCH_FACTOR = 10
MINIMUM_OVERFETCH = 200

# Open partitions kept resident. Each entry holds two LanceTable handles and their
# caches, and nothing evicted them before: the daemon is long-lived and get_chunk
# walks every registered project, so one call could fault in every project a user
# has ever indexed. Sixteen covers the projects one developer works across while
# keeping the ceiling independent of how many they have registered.
MAX_CACHED_PARTITIONS = 16

# Upper bound on how many partitions one hybrid query reads at once. More
# concurrency than this has not shown a latency win in benchmarks and raises
# contention on the shared partition cache, so the pool stays small and bounded.
_SEARCH_CONCURRENCY = 8

# touch_slot buffers last-use timestamps in memory instead of writing on every
# call (a read-heavy workload touches its active slot on every query, and a
# write there would bump the project_slots table version once per query). A
# touch is flushed inline once the oldest pending one has waited this long, so
# a long-idle-but-still-running process does not lose LRU history indefinitely
# if nothing else ever flushes it.
SLOT_TOUCH_FLUSH_SECONDS = 300.0

# Columns get_chunk reads. The vector and the identifier-terms column are
# excluded: nothing outside indexing and ranking can use them, and reading them
# made a single-chunk fetch an order of magnitude larger than the code it
# returned. project_id is injected from the owning partition. content_hash
# remains on the chunk row so a read cannot combine separate commit generations.
CHUNK_PAYLOAD_COLUMNS = [
    "chunk_id",
    "file_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "content_hash",
    "part_index",
]

# Every chunk column except the vector. list_chunks has no production caller and its
# test callers read text and offsets, so decoding vectors was pure waste.
INDEXED_CHUNK_COLUMNS = [
    "chunk_id",
    "file_id",
    "path",
    "language",
    "kind",
    "symbol",
    "qualified_symbol",
    "parent_symbol",
    "start_byte",
    "end_byte",
    "start_line",
    "end_line",
    "content",
    "identifier_terms",
    "content_hash",
    "part_index",
]


def _quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


# A registry row past this many dirty/untracked paths would bloat every
# registry read for a rare, very large changeset; beyond the cap the slot's
# status-paths column is stored as null and the freshness fast path (see
# Application._project_status_for_target) falls back to a full scan instead.
MAX_PERSISTED_STATUS_PATHS = 2000


def encode_status_paths(paths: Iterable[str]) -> str | None:
    """JSON-encode the sorted, de-duplicated path list, or None past the cap."""
    ordered = sorted(set(paths))
    if len(ordered) > MAX_PERSISTED_STATUS_PATHS:
        return None
    return json.dumps(ordered, separators=(",", ":"), ensure_ascii=True)


def decode_status_paths(value: str | None) -> frozenset[str] | None:
    """Decode a slot's stored status-paths column, or None if absent or unusable."""
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except ValueError:
        return None
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        return None
    return frozenset(decoded)


def _legacy_slot_id(project_id: str) -> str:
    """Deterministic identifier of a project's pre-slot legacy partition."""
    payload = json.dumps(["legacy-slot-v1", project_id], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_ids_condition(file_ids: Iterable[str]) -> str:
    """A predicate matching every file in *file_ids* with one IN list."""
    return f"file_id IN ({', '.join(_quoted(file_id) for file_id in file_ids)})"


# The batched commit replaces each file's rows with one merge_insert whose
# ``when_not_matched_by_source_delete("file_id IN (...)")`` predicate must
# delete only the predicate's unmatched target rows, never rows of untouched
# files. Every supported lancedb release (0.34.0 and up) filters per row, but a
# regression to the older all-or-nothing gate behavior would silently delete
# every untouched file's rows on a multi-file project's second run, so the
# store refuses to commit unless a cheap scratch-table probe confirms the
# semantics. The probe is checked on the first batched commit, not on store
# construction: read-only processes (status checks, searches, metrics) never
# run the merge and must not pay for it. See
# docs/plans/2026-07-27-review-followups-index.md.
_batched_merge_semantics_cache: bool | None = None
_batched_merge_semantics_lock = threading.Lock()


def _probe_batched_merge_semantics() -> bool:
    """Return whether the installed lancedb filters source-deletes per row.

    Mirrors the real commit shape: one merge on the key with a predicate that
    also covers a sibling whose rows are absent from the source. Per-row
    semantics leave the untouched file's row and delete the predicate's
    unmatched rows; gate semantics leave only the matched replacement.
    """
    try:
        with tempfile.TemporaryDirectory() as directory:
            database = lancedb.connect(directory)
            table = database.create_table(
                "probe",
                pa.table(
                    {
                        "file_id": ["a", "b", "c"],
                        "chunk_id": ["a1", "b1", "c1"],
                        "vector": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    }
                ),
            )
            source = pa.table({"file_id": ["a"], "chunk_id": ["a2"], "vector": [[0.0, 0.0]]})
            (
                table.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete("file_id IN ('a', 'b')")
                .execute(source)
            )
            surviving = sorted(row["chunk_id"] for row in table.search().to_list())
            return surviving == ["a2", "c1"]
    except Exception as exc:
        logger.warning(
            "Merge-insert semantics probe failed (%s); batched commits are refused",
            exc,
        )
        logger.debug("Merge-insert semantics probe failure details", exc_info=True)
        return False


def _batched_merge_semantics_ok() -> bool:
    """Cached :func:`_probe_batched_merge_semantics`, once per process."""
    global _batched_merge_semantics_cache
    if _batched_merge_semantics_cache is None:
        with _batched_merge_semantics_lock:
            if _batched_merge_semantics_cache is None:
                _batched_merge_semantics_cache = _probe_batched_merge_semantics()
                logger.debug(
                    "Batched merge-insert semantics verified on lancedb %s",
                    getattr(lancedb, "__version__", "unknown"),
                )
    return _batched_merge_semantics_cache


def _nullable_int(value: Any) -> int | None:
    """Coerce a fragment-length statistic, tolerating None and non-finite noise."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_timestamp(value: str) -> str:
    """Normalize a Lance version timestamp string to ISO-8601."""
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


# Git inspection for storage-status worktree warnings is best-effort and must
# never make the status command fail. A short timeout keeps a hung repository
# from holding the command hostage.
_GIT_TIMEOUT_SECONDS = 5.0
_GitRunner = Callable[[list[str], Path], str | None]


def _run_git_quietly(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _directory_bytes(directory: Path) -> int:
    """Sum the file bytes under *directory* without following symlinks.

    A symlinked directory or file inside a partition is a deliberate escape
    hatch, not storage the index owns, so its target's bytes must not be
    counted as physical index storage.
    """
    total = 0
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                total += _directory_bytes(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def overlap_warnings(projects: Iterable[ProjectInfo]) -> list[str]:
    """Warn about registered roots that duplicate or nest one another.

    Read-only detection: rejecting new overlaps is a later-phase registration
    concern, but status must already say when two registrations point at the
    same or a contained directory.
    """
    warnings: list[str] = []
    for left, right in itertools.combinations(list(projects), 2):
        if same_project_root(left.root, right.root):
            warnings.append(
                f"Projects {left.id!r} and {right.id!r} register the same root: {left.root}"
            )
            continue
        left_resolved = left.root.expanduser().resolve()
        right_resolved = right.root.expanduser().resolve()
        try:
            left_resolved.relative_to(right_resolved)
        except ValueError:
            try:
                right_resolved.relative_to(left_resolved)
            except ValueError:
                continue
            warnings.append(
                f"Project {left.id!r} root {left_resolved} contains the root of project "
                f"{right.id!r} ({right_resolved})"
            )
        else:
            warnings.append(
                f"Project {left.id!r} root {left_resolved} is nested inside the root of "
                f"project {right.id!r} ({right_resolved})"
            )
    return warnings


def overlapping_registration(projects: Iterable[ProjectInfo], root: Path) -> ProjectInfo | None:
    """Return an existing project whose root equals, contains, or nests in *root*.

    Every overlap kind indexes the same sources twice, so registration checks
    all three. Containment goes through rooted_under, whose samefile boundary
    check makes a differently-cased spelling of one directory count as overlap
    on case-insensitive filesystems, matching how same_project_root treats
    equality. Detection only: rejecting a new overlap is the registration
    layer's decision, and existing overlapping registrations stay valid.
    """
    root = root.expanduser().resolve()
    for project in projects:
        existing = project.root.expanduser().resolve()
        if same_project_root(existing, root):
            return project
        if rooted_under(root, existing) or rooted_under(existing, root):
            return project
    return None


def worktree_warnings(
    projects: Iterable[ProjectInfo], *, _run: _GitRunner | None = None
) -> list[str]:
    """Warn about registered roots that are checkouts of one Git repository.

    Two roots whose ``--show-toplevel`` differs but whose Git common directory
    is the same are worktrees (or a main checkout and a worktree) of one
    repository. Since registrations are now shared across a repository's
    checkouts, such a pair means two pre-worktree-support registrations that
    were never unified. All failures are swallowed: this is advisory only.
    """
    runner = _run or _run_git_quietly
    repositories: list[tuple[ProjectInfo, Path, Path]] = []
    for project in projects:
        toplevel = runner(["git", "rev-parse", "--show-toplevel"], project.root)
        common = runner(["git", "rev-parse", "--git-common-dir"], project.root)
        if toplevel is None or common is None:
            continue
        common_path = Path(common)
        if not common_path.is_absolute():
            # --git-common-dir is relative to the query cwd, which is the
            # registered root itself, not the repository toplevel: a root that
            # is a subdirectory of a checkout reports '../.git'.
            common_path = (Path(project.root) / common_path).resolve()
        repositories.append((project, Path(toplevel), common_path))
    warnings: list[str] = []
    for (left, left_top, left_common), (right, right_top, right_common) in itertools.combinations(
        repositories, 2
    ):
        if same_project_root(left_common, right_common) and not same_project_root(
            left_top, right_top
        ):
            warnings.append(
                f"Projects {left.id!r} and {right.id!r} share Git common directory "
                f"{left_common} from different checkouts. Linked worktrees of one "
                "repository now share one project registration; remove one of these "
                f"registrations and re-run init_project on its root ({left_top} or "
                f"{right_top}) to unify them."
            )
    return warnings


def _symbol_matches(chunk: ChunkPreview, name: str, match: str) -> bool:
    """Apply exact symbol-match semantics that the SQL pre-filter cannot."""
    candidate = chunk.qualified_symbol or chunk.symbol or ""
    symbol = chunk.symbol or ""
    if match == "exact":
        return candidate == name or symbol == name
    if match == "prefix":
        return candidate.startswith(name) or symbol.startswith(name)
    return name in candidate or name in symbol


@dataclass(frozen=True)
class _ProjectTables:
    files: LanceTable
    chunks: LanceTable
    references: LanceTable | None
    generation: int


class ReferenceRecord(TypedDict):
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


@dataclass(frozen=True)
class TableVersions:
    """A point-in-time snapshot of a project partition's three tables."""

    files: int
    chunks: int
    references: int


@dataclass(frozen=True)
class PartitionRef:
    """Immutable logical-to-physical index selection for one operation."""

    project_id: str
    slot_id: str
    partition_id: str
    activation_epoch: int


@dataclass(frozen=True)
class ActiveIndexTarget:
    """Immutable application-boundary selection for one index operation.

    ``project.root`` names the checkout the request was resolved from -- the
    worktree root, not the registered canonical root, whenever the request
    arrived through a worktree's marker -- so every scan, probe, and guard
    derived from the target observes that same checkout.
    """

    project: ProjectInfo
    slot: ProjectSlot
    partition_id: str
    activation_epoch: int
    git_state: GitState

    @property
    def partition(self) -> PartitionRef:
        return PartitionRef(
            project_id=self.project.id,
            slot_id=self.slot.slot_id,
            partition_id=self.partition_id,
            activation_epoch=self.activation_epoch,
        )


class LanceStore:
    def __init__(
        self,
        directory: Path,
        *,
        vector_dimension: int = 768,
        vector_index: str = "exact",
        vector_storage: str = "float16",
        branch_cache_limit: int = 4,
    ) -> None:
        if vector_storage not in {"float32", "float16"}:
            raise ValueError(f"vector_storage must be float32 or float16, got {vector_storage!r}")
        self.directory = directory
        self.vector_dimension = vector_dimension
        self.vector_index = vector_index
        self.vector_storage = vector_storage
        self.branch_cache_limit = branch_cache_limit
        self.vector_dtype = pa.float16() if vector_storage == "float16" else pa.float32()
        legacy_rows = self._migrate_v1(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(directory / "registry", read_consistency_interval=timedelta(0))
        self._migrate_active_checkouts(directory / "registry")
        self._migrate_project_slot_status_columns(directory / "registry")
        self._projects = self._table(self._db, "projects", self._project_schema())
        self._project_slots = self._table(self._db, "project_slots", self._project_slot_schema())
        self._active_slots = self._table(self._db, "active_slots", self._active_slot_schema())
        self._partitions: OrderedDict[str, _ProjectTables] = OrderedDict()
        self._partitions_lock = threading.Lock()
        # Serializes legacy adoption inside this process; see _ensure_adopted
        # for why it is not the cross-process project writer lock.
        self._adoption_lock = threading.Lock()
        # Buffered touch_slot timestamps, keyed by slot_id; see touch_slot and
        # flush_slot_touches.
        self._pending_slot_touches: dict[str, int] = {}
        self._touch_lock = threading.Lock()
        for row in legacy_rows:
            row = {
                **row,
                "vector_dimension": vector_dimension,
                "schema_version": SCHEMA_VERSION,
                "state": "pending",
                "updated_at": time.time_ns(),
            }
            self._merge(self._projects, "id", [row])

    def upsert_project(self, project: ProjectInfo, *, model_id: str, state: str = "ready") -> None:
        existing = self._rows(self._projects, f"id = {_quoted(project.id)}")
        if existing:
            registered_root = Path(str(existing[0]["root"])).resolve()
            incoming_root = project.root.resolve()
            same_root = same_project_root(registered_root, incoming_root)
            if same_root:
                project = project.model_copy(update={"root": registered_root})
            elif registered_root.exists():
                # The registered directory is still there -- whether or not its
                # marker survives in it -- so a different, unrelated root
                # claiming this id is ambiguous rather than a move: it could be
                # a stale marker, a directory copy, or someone reusing an id.
                # Only a directory that has vanished entirely (checked below by
                # falling through) is the legitimate "the user moved it" case.
                if not self._shares_repository(registered_root, incoming_root):
                    raise CodeIndexingError(
                        ErrorCode.PROJECT_ID_CONFLICT,
                        "The project ID is already active at another path. Run "
                        "remove_project on the registered root, or init_project "
                        "with force_new_id here.",
                        project=project.id,
                        registered_root=str(registered_root),
                        incoming_root=str(incoming_root),
                    )
                # A checkout of the same repository -- a linked worktree --
                # legally carries this registration id in its local marker.
                # The canonical root stays whichever checkout registered
                # first; only the marker's mutable payload flows through.
                project = project.model_copy(update={"root": registered_root})
        if (
            existing
            and str(existing[0]["state"]) != "pending"
            and self.incompatibility_reason(project.id, model_id) is not None
        ):
            # A reconstructable generation mismatch (different embedding
            # model, vector dimension, or index schema version) marks the
            # partition for automatic rebuild instead of raising
            # INDEX_INCOMPATIBLE. The stored generation fields are left
            # untouched: they describe the rows still live in the
            # partition until the rebuild deletes and re-stamps them.
            self.mark_rebuild_required(project.id)
            return
        row = {
            "id": project.id,
            "name": project.name,
            "root": str(project.root),
            "payload": project.model_dump_json(),
            "model_id": model_id,
            "vector_dimension": self.vector_dimension,
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": time.time_ns(),
        }
        # A no-op upsert -- project discovery, a status check, a state that did
        # not change -- must not churn registry versions. The comparison
        # excludes updated_at so a real mutation still stamps it fresh. It is
        # a typed comparison, not a string coercion, so a stored null can
        # never be mistaken for the literal string "None".
        if existing and all(
            existing[0][column] == row[column] for column in row if column != "updated_at"
        ):
            return
        self._merge(self._projects, "id", [row])

    def _shares_repository(self, left: Path, right: Path) -> bool:
        """Return whether both paths are checkouts of one Git repository.

        This is what makes a linked worktree's marker with the same project
        id legal, while a plain directory copy (a different repository
        identity) still conflicts. A degraded probe cannot verify anything
        and deliberately counts as "not shared": fail closed.
        """
        if same_project_root(left, right):
            return True
        left_state = probe_git_state(left)
        right_state = probe_git_state(right)
        return (
            left_state.probe is GitProbeOutcome.GIT
            and right_state.probe is GitProbeOutcome.GIT
            and bool(left_state.repository_identity)
            and left_state.repository_identity == right_state.repository_identity
        )

    def upsert_slot(self, slot: ProjectSlot) -> None:
        """Create or update one slot row, idempotently.

        Like ``upsert_project``, a no-op write must not churn registry table
        versions, so timestamps (``created_at``, ``last_used_at``) are excluded
        from the equality check.
        """
        row = self._slot_row(slot)
        existing = self._rows(self._project_slots, f"slot_id = {_quoted(slot.slot_id)}")
        if existing and all(
            existing[0][column] == row[column]
            for column in row
            if column not in {"created_at", "last_used_at"}
        ):
            return
        self._merge(self._project_slots, "slot_id", [row])

    def get_slot(self, slot_id: str) -> ProjectSlot | None:
        rows = self._rows(self._project_slots, f"slot_id = {_quoted(slot_id)}")
        return self._overlay_pending_touch(self._slot_from_row(rows[0])) if rows else None

    def list_slots(self, project_id: str) -> list[ProjectSlot]:
        rows = self._rows(self._project_slots, f"project_id = {_quoted(project_id)}")
        return [self._overlay_pending_touch(self._slot_from_row(row)) for row in rows]

    def _overlay_pending_touch(self, slot: ProjectSlot) -> ProjectSlot:
        """Apply a buffered touch_slot timestamp not yet written to disk.

        Every LRU-retention reader goes through get_slot or list_slots, so
        overlaying here (rather than in each reader) is enough to make
        eviction order correct even when nothing has explicitly called
        flush_slot_touches yet.
        """
        with self._touch_lock:
            pending = self._pending_slot_touches.get(slot.slot_id)
        if pending is None or pending <= slot.last_used_at:
            return slot
        return slot.model_copy(update={"last_used_at": pending})

    def touch_slot(self, slot_id: str) -> None:
        """Stamp a slot's last-use time for least-recently-used retention.

        Buffered in memory rather than written immediately: a read-heavy
        workload touches its active slot on every query, and a write on every
        touch would bump the project_slots table version -- and its on-disk
        footprint -- once per query (see the query-path-overhead remediation
        plan). flush_slot_touches merges every buffered touch in one write;
        this also flushes inline once the oldest pending touch has waited
        past SLOT_TOUCH_FLUSH_SECONDS, so a long-lived process without an
        explicit flush call still bounds how much history a crash could lose.
        """
        now = time.time_ns()
        with self._touch_lock:
            self._pending_slot_touches[slot_id] = now
            oldest = min(self._pending_slot_touches.values())
        if (now - oldest) / 1_000_000_000 > SLOT_TOUCH_FLUSH_SECONDS:
            self.flush_slot_touches()

    def flush_slot_touches(self) -> None:
        """Write every buffered touch_slot timestamp in one merge.

        The buffer is drained before the write, not after: a touch that
        arrives while this is running is left pending for the next flush
        rather than lost or double-counted.
        """
        with self._touch_lock:
            pending = dict(self._pending_slot_touches)
            self._pending_slot_touches.clear()
        if not pending:
            return
        rows: list[dict[str, Any]] = []
        for slot_id, touched_at in pending.items():
            slot = self.get_slot(slot_id)
            if slot is None:
                continue
            rows.append(self._slot_row(slot.model_copy(update={"last_used_at": touched_at})))
        if rows:
            self._merge(self._project_slots, "slot_id", rows)

    def close(self) -> None:
        """Release in-memory state that must not outlive the process.

        Flushes every buffered slot touch so a later process's LRU decision
        never mistakes an unflushed touch for genuine inactivity.
        """
        self.flush_slot_touches()

    def set_slot_state(
        self,
        partition: PartitionRef,
        state: str,
        *,
        project: ProjectInfo | None = None,
    ) -> None:
        """Update one pinned slot without changing any active pointer.

        ``project`` carries the checkout whose state is being stamped: a
        shared slot can be indexed from any worktree of the repository, and
        its HEAD/cleanliness stamp must describe the checkout that ran.
        """
        slot = self.get_slot(partition.slot_id)
        if slot is None or slot.partition_id != partition.partition_id:
            return
        updates: dict[str, Any] = {"state": state, "last_used_at": time.time_ns()}
        if project is not None:
            git = probe_git_state(project.root, include_status=True)
            if git.probe is GitProbeOutcome.GIT:
                updates["indexed_head"] = git.head_oid
                updates["indexed_clean"] = (
                    None if git.worktree.value == "unknown" else git.worktree.value == "clean"
                )
                updates["indexed_status_fingerprint"] = git.status_fingerprint
                updates["indexed_status_paths"] = encode_status_paths(
                    {*git.dirty_paths, *git.untracked_paths}
                )
            else:
                updates["indexed_head"] = None
                updates["indexed_clean"] = None
                updates["indexed_status_fingerprint"] = None
                updates["indexed_status_paths"] = None
            # The generation identity the next freshness check compares against:
            # scan configuration, model, dimension, and schema as of this
            # commit. The caller's upsert_project has already refreshed the
            # logical row these are mirrored from.
            rows = self._rows(self._projects, f"id = {_quoted(project.id)}")
            if rows:
                updates["scan_config_hash"] = self._scan_config_hash(project)
                updates["model_id"] = str(rows[0]["model_id"])
                updates["vector_dimension"] = int(rows[0]["vector_dimension"])
                updates["schema_version"] = int(rows[0]["schema_version"])
        self.upsert_slot(slot.model_copy(update=updates))

    def activate_slot(self, project_id: str, slot_id: str, *, checkout_key: str) -> int:
        """Point *checkout_key*'s active pointer at *slot_id*.

        Pointers are keyed per checkout, not per project: two worktrees of
        one project must be able to keep two different slots active at the
        same time. A switch is a single atomic upsert rather than two
        coordinated flag updates. It is published only after the slot row
        exists, and every actual change increments ``activation_epoch`` so
        long-lived handles can detect that the selection moved under them.
        Returns the resulting epoch.
        """
        lock_directory = self.directory.parent / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_directory / f"active-{project_id}.lock"):
            slot = self.get_slot(slot_id)
            if slot is None or slot.project_id != project_id:
                raise ValueError(f"cannot activate unknown slot {slot_id!r} for {project_id!r}")
            rows = self._rows(
                self._active_slots,
                f"project_id = {_quoted(project_id)} AND checkout_key = {_quoted(checkout_key)}",
            )
            if rows and str(rows[0]["slot_id"]) == slot_id:
                return int(rows[0]["activation_epoch"])
            epoch = (int(rows[0]["activation_epoch"]) if rows else 0) + 1
            self._merge(
                self._active_slots,
                ["project_id", "checkout_key"],
                [
                    {
                        "project_id": project_id,
                        "slot_id": slot_id,
                        "checkout_key": checkout_key,
                        "activation_epoch": epoch,
                        "updated_at": time.time_ns(),
                    }
                ],
            )
            return epoch

    def active_slots_for(self, project_id: str) -> list[ProjectSlot]:
        """Return every checkout pointer's slot, one entry per live checkout."""
        rows = self._rows(self._active_slots, f"project_id = {_quoted(project_id)}")
        slots: list[ProjectSlot] = []
        seen: set[str] = set()
        for row in rows:
            slot_id = str(row["slot_id"])
            if slot_id in seen:
                continue
            seen.add(slot_id)
            slot = self.get_slot(slot_id)
            # A pointer left behind by a removed slot row selects nothing.
            if slot is not None and slot.project_id == project_id:
                slots.append(slot)
        return slots

    def active_slot(
        self, project_id: str, *, checkout_key: str | None = None
    ) -> ProjectSlot | None:
        """Return the slot *checkout_key*'s active pointer selects, or None.

        Without an explicit key the project's freshest checkout-scoped pointer
        wins, falling back to a pre-worktree empty-key pointer only when no
        real one exists; callers that care about one checkout always pass its
        key.
        """
        slot_row = self._selected_pointer_row(project_id, checkout_key)
        if slot_row is None:
            return None
        slot = self.get_slot(str(slot_row["slot_id"]))
        # A pointer left behind by a removed slot row selects nothing; the
        # caller's fallback (adoption, then the legacy identity) takes over.
        if slot is None or slot.project_id != project_id:
            return None
        return slot

    def _selected_pointer_row(
        self, project_id: str, checkout_key: str | None
    ) -> dict[str, Any] | None:
        condition = f"project_id = {_quoted(project_id)}"
        if checkout_key is not None:
            condition += f" AND checkout_key = {_quoted(checkout_key)}"
        rows = self._rows(self._active_slots, condition)
        if not rows:
            return None
        if checkout_key is not None:
            return rows[0]
        keyed = [row for row in rows if str(row["checkout_key"])]
        return max(keyed or rows, key=lambda row: int(row["updated_at"]))

    def active_partition(self, project_id: str, *, checkout_key: str | None = None) -> PartitionRef:
        """Resolve and pin an active physical partition for the project."""
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        project = ProjectInfo.model_validate_json(str(rows[0]["payload"]))
        state = probe_git_state(project.root)
        resolved_key = checkout_key if checkout_key is not None else _checkout_key(state)
        return self.resolve_partition(project, state, resolved_key)

    def resolve_partition(
        self, project: ProjectInfo, state: GitState, checkout_key: str | None = None
    ) -> PartitionRef:
        """Activate the slot selected by an already-probed Git state.

        The application owns probing so every lower layer uses the same
        immutable selector and HEAD snapshot. Workspace selectors are resolved
        for degraded probes too; a transient Git failure must never leave a
        known branch partition active. The pointer that is moved belongs to
        this checkout alone (*checkout_key*, derived from *state* when
        omitted), so sibling worktrees keep their own selections.
        """
        self._ensure_adopted(project.id)
        selected_key = checkout_key if checkout_key is not None else _checkout_key(state)
        rows = self._rows(self._projects, f"id = {_quoted(project.id)}")
        if not rows:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project.id}")
        desired = self._slot_for_git_state(rows[0], state)
        active = self._active_partition_ref(project.id, checkout_key=selected_key)
        existing = self.get_slot(desired.slot_id)
        if existing is None:
            desired = desired.model_copy(update={"state": "pending"})
            selected = desired
        else:
            selected = existing.model_copy(
                update={
                    "repository_identity": desired.repository_identity,
                    "checkout_identity": desired.checkout_identity,
                    "project_prefix": desired.project_prefix,
                    "selector_kind": desired.selector_kind,
                    "selector_value": desired.selector_value,
                }
            )
        self.upsert_slot(selected)
        if active is None or active.slot_id != selected.slot_id:
            self.activate_slot(project.id, selected.slot_id, checkout_key=selected_key)
        active = self._active_partition_ref(project.id, checkout_key=selected_key)
        if active is None:
            raise CodeIndexingError(
                ErrorCode.PROJECT_NOT_FOUND,
                f"Project {project.id} has no active index slot",
            )
        self.touch_slot(active.slot_id)
        # This path already writes (upsert_slot, possibly activate_slot), so
        # flushing the touch here costs nothing extra and keeps activation
        # durable rather than leaving it in the in-memory buffer.
        self.flush_slot_touches()
        return active

    def _active_partition_ref(
        self, project_id: str, *, checkout_key: str | None = None
    ) -> PartitionRef | None:
        slot_row = self._selected_pointer_row(project_id, checkout_key)
        if slot_row is None:
            return None
        slot = self.get_slot(str(slot_row["slot_id"]))
        if slot is None or slot.project_id != project_id:
            return None
        return PartitionRef(
            project_id=project_id,
            slot_id=slot.slot_id,
            partition_id=slot.partition_id,
            activation_epoch=int(slot_row["activation_epoch"]),
        )

    def _slot_for_git_state(self, row: dict[str, Any], state: GitState) -> ProjectSlot:
        project = ProjectInfo.model_validate_json(str(row["payload"]))
        slot = _git_slot_id(project.id, state)
        return ProjectSlot(
            slot_id=slot,
            project_id=project.id,
            partition_id=_git_partition_id(slot),
            selector_kind=state.selector_kind.value,
            selector_value=state.selector_value,
            repository_identity=state.repository_identity,
            checkout_identity=state.checkout_identity,
            project_prefix=state.project_prefix,
            scan_config_hash=self._scan_config_hash(project),
            model_id=str(row["model_id"]),
            vector_dimension=int(row["vector_dimension"]),
            schema_version=int(row["schema_version"]),
            state=str(row["state"]),
            created_at=time.time_ns(),
            last_used_at=time.time_ns(),
        )

    def remove_slot(self, slot_id: str) -> bool:
        """Remove a slot row and any pointer that still selects it.

        Registry-level removal only. Durable physical eviction (deleting
        states, generation advancement, partition deletion) is the retention
        pass's job, so the partition directory is deliberately left alone.
        """
        slot = self.get_slot(slot_id)
        if slot is None:
            return False
        self._active_slots.delete(f"slot_id = {_quoted(slot_id)}")
        self._project_slots.delete(f"slot_id = {_quoted(slot_id)}")
        with self._partitions_lock:
            self._partitions.pop(slot.partition_id, None)
        return True

    def _partition_id_for(self, project_id: str) -> str:
        """Resolve the physical partition a logical project currently reads and writes.

        The active pointer selects it; a project without one is adopted on
        first touch. Until the application boundary resolves explicit
        ``ActiveIndexTarget`` values, this is the single place the
        logical-to-physical mapping happens, and for adopted legacy and
        workspace slots it deliberately still yields ``project_id``.
        """
        slot = self.active_slot(project_id)
        if slot is not None:
            return slot.partition_id
        self._ensure_adopted(project_id)
        slot = self.active_slot(project_id)
        return slot.partition_id if slot is not None else project_id

    def _ensure_adopted(self, project_id: str) -> None:
        """Map a pre-slot registration onto the slot registry, once and idempotently.

        Runs under an in-process mutex with a re-check inside it. It must not
        take the logical project writer lock: the indexer and maintenance
        already hold it while calling into storage, and re-acquiring it here
        deadlocks. Adoption needs no cross-process mutual exclusion anyway --
        slot identifiers are derived deterministically from stable inputs, so
        two processes racing an adoption write the same row and the pointer
        converges (the losing epoch bump is harmless because nothing has
        captured an epoch yet).

        Non-Git projects adopt their existing partition as the workspace slot.
        Git (or temporarily unreadable) checkouts record the old partition as
        an unscoped legacy slot that no branch selector claims. The slot row
        mirrors the project row's lifecycle state, so a project that is still
        pending adopts a pending slot and reads through it never materialise
        a partition.
        """
        if self._active_pointer_rows(project_id):
            # Adopted already, or at least owned by a checkout pointer; either
            # way there is nothing pre-slot left to adopt.
            return
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return
        row = rows[0]
        project = ProjectInfo.model_validate_json(str(row["payload"]))
        with self._adoption_lock:
            if self._active_pointer_rows(project.id):
                return
            state = probe_git_state(Path(project.root))
            now = time.time_ns()
            partition_state = str(row["state"])
            scan_hash = self._scan_config_hash(project)
            model_id = str(row["model_id"])
            vector_dimension = int(row["vector_dimension"])
            schema_version = int(row["schema_version"])
            if state.probe is GitProbeOutcome.NOT_GIT:
                slot = ProjectSlot(
                    slot_id=_git_slot_id(project.id, state),
                    project_id=project.id,
                    partition_id=project.id,
                    selector_kind=state.selector_kind.value,
                    selector_value=state.selector_value,
                    repository_identity=None,
                    checkout_identity=None,
                    project_prefix="",
                    indexed_head=None,
                    indexed_clean=None,
                    scan_config_hash=scan_hash,
                    model_id=model_id,
                    vector_dimension=vector_dimension,
                    schema_version=schema_version,
                    state=partition_state,
                    created_at=now,
                    last_used_at=now,
                )
            else:
                slot = ProjectSlot(
                    slot_id=_legacy_slot_id(project.id),
                    project_id=project.id,
                    partition_id=project.id,
                    selector_kind="legacy",
                    selector_value=project.id,
                    repository_identity=state.repository_identity,
                    checkout_identity=state.checkout_identity,
                    project_prefix=state.project_prefix,
                    indexed_head=None,
                    indexed_clean=None,
                    scan_config_hash=scan_hash,
                    model_id=model_id,
                    vector_dimension=vector_dimension,
                    schema_version=schema_version,
                    state=partition_state,
                    created_at=now,
                    last_used_at=now,
                )
            self.upsert_slot(slot)
            self.activate_slot(project.id, slot.slot_id, checkout_key=_checkout_key(state))

    @staticmethod
    def _scan_config_hash(project: ProjectInfo) -> str:
        payload = json.dumps(
            project.scan.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _slot_row(slot: ProjectSlot) -> dict[str, Any]:
        return {
            "slot_id": slot.slot_id,
            "project_id": slot.project_id,
            "partition_id": slot.partition_id,
            "selector_kind": slot.selector_kind,
            "selector_value": slot.selector_value,
            "repository_identity": slot.repository_identity,
            "checkout_identity": slot.checkout_identity,
            "project_prefix": slot.project_prefix,
            "indexed_head": slot.indexed_head,
            "indexed_clean": slot.indexed_clean,
            "indexed_status_fingerprint": slot.indexed_status_fingerprint,
            "indexed_status_paths": slot.indexed_status_paths,
            "scan_config_hash": slot.scan_config_hash,
            "model_id": slot.model_id,
            "vector_dimension": slot.vector_dimension,
            "schema_version": slot.schema_version,
            "state": slot.state,
            "created_at": slot.created_at,
            "last_used_at": slot.last_used_at,
        }

    @staticmethod
    def _slot_from_row(row: dict[str, Any]) -> ProjectSlot:
        return ProjectSlot(
            slot_id=str(row["slot_id"]),
            project_id=str(row["project_id"]),
            partition_id=str(row["partition_id"]),
            selector_kind=str(row["selector_kind"]),
            selector_value=str(row["selector_value"]),
            repository_identity=_optional_str(row.get("repository_identity")),
            checkout_identity=_optional_str(row.get("checkout_identity")),
            project_prefix=str(row.get("project_prefix") or ""),
            indexed_head=_optional_str(row.get("indexed_head")),
            indexed_clean=(
                None if row.get("indexed_clean") is None else bool(row["indexed_clean"])
            ),
            indexed_status_fingerprint=_optional_str(row.get("indexed_status_fingerprint")),
            indexed_status_paths=_optional_str(row.get("indexed_status_paths")),
            scan_config_hash=str(row.get("scan_config_hash") or ""),
            model_id=str(row.get("model_id") or ""),
            vector_dimension=int(row.get("vector_dimension") or 0),
            schema_version=int(row.get("schema_version") or 0),
            state=str(row.get("state") or "pending"),
            created_at=int(row.get("created_at") or 0),
            last_used_at=int(row.get("last_used_at") or 0),
        )

    def incompatibility_reason(
        self, project_id: str, model_id: str, *, partition_id: str | None = None
    ) -> str | None:
        """Return why *project_id*'s stored generation must be rebuilt, or None.

        A stored embedding model, vector dimension, index schema version, or
        on-disk vector storage dtype that differs from this store's current
        values describes rows this build cannot serve coherently. Rebuilding
        -- deleting the partition and re-indexing -- recovers it, so the
        mismatch is returned as a reason rather than raised as the hard
        ``INDEX_INCOMPATIBLE`` failure it used to be. A pending registration
        has no live rows to describe, so it is never incompatible.

        The model, dimension, and schema-version comparisons are registry-only;
        the dtype comparison is authoritative only on the partition's own
        chunk-table schema, so it opens the partition on disk when one exists.
        """
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return None
        row = rows[0]
        if str(row["state"]) == "pending":
            return None
        differences: list[str] = []
        if str(row["model_id"]) != model_id:
            differences.append(f"embedding model {row['model_id']!r} -> {model_id!r}")
        if int(row["vector_dimension"]) != self.vector_dimension:
            differences.append(
                f"vector dimension {row['vector_dimension']} -> {self.vector_dimension}"
            )
        if int(row["schema_version"]) != SCHEMA_VERSION:
            differences.append(f"index schema version {row['schema_version']} -> {SCHEMA_VERSION}")
        # The registry row alone cannot describe a dtype flip between two
        # builds of the same schema version, so the partition's own vector
        # column is the authority: a stored float32 partition under a float16
        # store (or the reverse) must rebuild rather than mix generations.
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is not None:
            stored_dtype = tables.chunks.schema.field("vector").type.value_type
            if stored_dtype != self.vector_dtype:
                differences.append(f"vector storage {stored_dtype} -> {self.vector_dtype}")
        return "; ".join(differences) if differences else None

    def mark_rebuild_required(self, project_id: str) -> None:
        """Stamp *project_id*'s registry state ``rebuild_required``.

        Read-only callers keep working -- the state surfaces through status
        and queries -- but nothing serves the partition again until an index
        run rebuilds it. The stored generation fields are not touched: they
        describe the rows still live in the partition.
        """
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return
        if str(rows[0]["state"]) == "rebuild_required":
            return
        row = dict(rows[0])
        row["state"] = "rebuild_required"
        row["updated_at"] = time.time_ns()
        self._merge(self._projects, "id", [row])
        active = self._active_partition_ref(project_id)
        if active is not None:
            slot = self.get_slot(active.slot_id)
            if slot is not None and slot.state != "rebuild_required":
                self.upsert_slot(slot.model_copy(update={"state": "rebuild_required"}))

    def delete_partition(
        self, project_id: str, *, model_id: str, partition_id: str | None = None
    ) -> bool:
        """Delete *project_id*'s partition while preserving its registration.

        Evicts the cached table handles before removing the partition
        directory. The registry row is re-stamped to the current generation
        and ``indexing`` state: with the partition gone, no rows remain for
        the old generation's claim to describe, and the replacement is about
        to be written by the calling run. Returns False when the project is
        not registered.
        """
        physical = self._partition_id_for(project_id) if partition_id is None else partition_id
        with self._partition_access_physical(physical):
            self._advance_partition_generation(physical)
            with self._partitions_lock:
                self._partitions.pop(physical, None)
            partition = self.directory / "projects" / physical
            if partition.exists():
                shutil.rmtree(partition)
            rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
            if not rows:
                return False
            row = dict(rows[0])
            row["model_id"] = model_id
            row["vector_dimension"] = self.vector_dimension
            row["schema_version"] = SCHEMA_VERSION
            row["state"] = "indexing"
            row["updated_at"] = time.time_ns()
            self._merge(self._projects, "id", [row])
            return True

    def list_projects(self) -> list[ProjectInfo]:
        return [
            ProjectInfo.model_validate_json(row["payload"]) for row in self._rows(self._projects)
        ]

    def project_state(self, project_id: str) -> str:
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        return str(rows[0]["state"])

    def list_files(self, project_id: str, *, partition_id: str | None = None) -> list[StoredFile]:
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return []
        return [StoredFile.model_validate(row) for row in self._rows(tables.files)]

    def has_file_errors(self, project_id: str, *, partition_id: str | None = None) -> bool:
        """Whether any stored file row records a genuine indexing error.

        Rejection tombstones ("rejected: ...") are deliberate, permanent skips,
        not errors, so they do not count. Reads only the error rows instead of
        materializing every file in the project.
        """
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return False
        return any(
            not str(row["error"] or "").startswith("rejected:")
            for row in self._rows(tables.files, "has_errors = true")
        )

    def upsert_file(self, record: StoredFile, *, partition_id: str | None = None) -> None:
        self._merge(
            self._project_tables(record.project_id, partition_id=partition_id).files,
            "file_id",
            [record.model_dump()],
        )

    def replace_file(
        self,
        record: StoredFile,
        chunks: list[StoredChunk],
        *,
        partition_id: str | None = None,
    ) -> None:
        tables = self._project_tables(record.project_id, partition_id=partition_id)
        condition = f"file_id = {_quoted(record.file_id)}"
        chunks = [
            chunk
            if chunk.content_hash
            else chunk.model_copy(update={"content_hash": record.content_hash})
            for chunk in chunks
        ]
        chunks = [
            chunk
            if chunk.content_hash
            else chunk.model_copy(update={"content_hash": record.content_hash})
            for chunk in chunks
        ]
        if chunks:
            (
                tables.chunks.merge_insert("chunk_id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(condition)
                .execute([chunk.model_dump() for chunk in chunks])
            )
        else:
            tables.chunks.delete(condition)
        self.upsert_file(record, partition_id=partition_id)

    def remove_file(
        self, project_id: str, file_id: str, *, partition_id: str | None = None
    ) -> None:
        tables = self._project_tables(project_id, partition_id=partition_id)
        condition = f"file_id = {_quoted(file_id)}"
        tables.chunks.delete(condition)
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        tables.references.delete(condition)
        tables.files.delete(condition)

    def table_versions(self, project_id: str, *, partition_id: str | None = None) -> TableVersions:
        """Snapshot every partition table's version before a commit begins."""
        tables = self._project_tables(project_id, partition_id=partition_id)
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        return TableVersions(
            files=tables.files.version,
            chunks=tables.chunks.version,
            references=tables.references.version,
        )

    def restore_versions(
        self,
        project_id: str,
        versions: TableVersions,
        *,
        restore_references: bool = True,
        partition_id: str,
    ) -> bool:
        """Return every partition table to *versions*' data.

        ``restore`` followed by ``checkout_latest`` makes the recorded version
        the live one; restoring a table that is already at that version's data
        is a no-op, so repeated recovery over the same journal is idempotent.

        Returns False when the partition no longer exists -- a project removed
        since the journal was written has nothing left to roll back. Recovery
        must not go through the create-on-write path here: materialising an
        empty partition would leave a version the journal can never name.
        """
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return False
        tables.files.restore(versions.files)
        tables.chunks.restore(versions.chunks)
        if restore_references:
            if tables.references is None:
                raise RuntimeError("Reference table is missing from an interrupted transaction")
            tables.references.restore(versions.references)
        tables.files.checkout_latest()
        tables.chunks.checkout_latest()
        if restore_references:
            if tables.references is None:
                raise RuntimeError("Reference table is missing from an interrupted transaction")
            tables.references.checkout_latest()
        return True

    def owns_recovery_partition(
        self, project_id: str, partition_id: str, *, slot_id: str | None
    ) -> bool:
        """Whether a journal still names storage owned by its registered project."""
        if not self._rows(self._projects, f"id = {_quoted(project_id)}"):
            return False
        if slot_id is None:
            return partition_id == project_id
        slot = self.get_slot(slot_id)
        return (
            slot is not None and slot.project_id == project_id and slot.partition_id == partition_id
        )

    def mark_project_state(self, project_id: str, state: str) -> bool:
        """Set a registered project's state, leaving its other columns alone.

        Returns False when the project is not registered. Recovery uses this
        to flag a project whose rollback could not be completed, since it only
        has the ID from the journal rather than a full ProjectInfo.
        """
        rows = self._rows(self._projects, f"id = {_quoted(project_id)}")
        if not rows:
            return False
        if str(rows[0]["state"]) == state:
            return True
        row = dict(rows[0])
        row["state"] = state
        row["updated_at"] = time.time_ns()
        self._merge(self._projects, "id", [row])
        return True

    def replace_files_from_arrow(
        self,
        project_id: str,
        *,
        files: pa.Table,
        chunk_batches: Iterable[tuple[list[str], pa.Table]],
        reference_batches: Iterable[tuple[list[str], pa.Table]] = (),
        removed_file_ids: Iterable[str] = (),
        partition_id: str | None = None,
    ) -> None:
        """Commit staged Arrow batches without materializing chunk objects.

        Each batch carries its full affected ``file_ids`` predicate: one
        ``merge_insert`` runs per non-empty batch (one batched ``delete`` when
        the whole batch has no rows), so table versions scale with O(batches)
        rather than O(files). ``when_not_matched_by_source_delete`` removes a
        file's previous rows both when its ids changed and when the file now
        extracts to nothing, since every file in the predicate is either
        present in the source or intentionally empty. Replacement ids win over
        removal ids, and removed files are deleted in one predicate per table.
        The vector columns stay fixed-size-list arrays of the store's storage
        dtype (float16 by default) end to end.

        Refuses to commit when the installed lancedb fails the batched
        merge-insert semantics probe, because a regression would delete rows
        of untouched files rather than fail loudly.
        """
        if not _batched_merge_semantics_ok():
            raise CodeIndexingError(
                ErrorCode.UNSUPPORTED_RUNTIME,
                "The installed lancedb version does not filter "
                "when_not_matched_by_source_delete rows the way batched commits "
                "require; refusing to commit because it could delete rows of "
                "untouched files. Upgrade lancedb and retry.",
            )
        tables = self._project_tables(project_id, partition_id=partition_id)
        replacement_ids: list[str] = []
        chunk_iter = iter(chunk_batches)
        try:
            for file_ids, chunks in chunk_iter:
                replacement_ids.extend(file_ids)
                condition = _file_ids_condition(file_ids)
                if chunks.num_rows:
                    (
                        tables.chunks.merge_insert("chunk_id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .when_not_matched_by_source_delete(condition)
                        .execute(chunks)
                    )
                else:
                    tables.chunks.delete(condition)
        finally:
            close = getattr(chunk_iter, "close", None)
            if close is not None:
                close()
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        reference_iter = iter(reference_batches)
        try:
            for file_ids, references in reference_iter:
                replacement_ids.extend(file_ids)
                condition = _file_ids_condition(file_ids)
                if references.num_rows:
                    (
                        tables.references.merge_insert("reference_id")
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .when_not_matched_by_source_delete(condition)
                        .execute(references)
                    )
                else:
                    tables.references.delete(condition)
        finally:
            close = getattr(reference_iter, "close", None)
            if close is not None:
                close()
        if files.num_rows:
            self._merge(tables.files, "file_id", files)
        replaced = set(replacement_ids)
        removed = [file_id for file_id in removed_file_ids if file_id not in replaced]
        if removed:
            condition = _file_ids_condition(removed)
            tables.chunks.delete(condition)
            tables.references.delete(condition)
            tables.files.delete(condition)

    def list_reference_records(
        self,
        project_id: str,
        *,
        version: int | None = None,
        schema_version: int | None = None,
        record_kinds: Iterable[str] | None = None,
        partition_id: str | None = None,
    ) -> list[ReferenceRecord]:
        """Return structural rows from the requested immutable table version.

        Optional schema and record-kind filters are pushed into the table query.
        Omitting them deliberately returns every row for recovery and raw-storage
        callers. Reference classification requests reference and coverage rows;
        declarations are fetched separately through narrower methods below.
        """
        self._validate_schema_version(schema_version)
        conditions: list[str] = []
        if schema_version is not None:
            conditions.append(f"schema_version = {schema_version}")
        if record_kinds is not None:
            kinds = sorted(set(record_kinds))
            if not kinds:
                return []
            values = ", ".join(_quoted(kind) for kind in kinds)
            conditions.append(f"record_kind IN ({values})")
        condition = " AND ".join(conditions) if conditions else None
        return self._reference_rows(
            project_id, condition, version=version, partition_id=partition_id
        )

    def reference_coverage(
        self, project_id: str, *, version: int | None = None, partition_id: str | None = None
    ) -> list[ReferenceRecord]:
        return self._reference_rows(
            project_id,
            "record_kind = 'coverage'",
            version=version,
            partition_id=partition_id,
        )

    def reference_version(self, project_id: str, *, partition_id: str | None = None) -> int:
        """Return the current structural snapshot without creating a partition."""
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None or tables.references is None:
            return 0
        return int(tables.references.version)

    def has_reference_table(self, project_id: str, *, partition_id: str | None = None) -> bool:
        """True when the references table exists for *project_id*.

        Distinguishes a legitimately empty reference index (the table
        exists, `ensure_reference_index` has run, there is simply nothing
        to report) from one that was never built at all -- a legacy
        partition indexed before this feature existed, or one whose
        `ensure_reference_index` was skipped. `_reference_rows` and
        `reference_version` collapse both cases to `[]`/`0`, so callers
        that need the distinction must ask this directly rather than trust an
        empty result.
        """
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        return tables is not None and tables.references is not None

    def coverage_for_file(
        self,
        project_id: str,
        file_id: str,
        schema_version: int,
        *,
        partition_id: str | None = None,
    ) -> list[ReferenceRecord]:
        self._validate_schema_version(schema_version)
        return self._reference_rows(
            project_id,
            "record_kind = 'coverage' "
            f"AND file_id = {_quoted(file_id)} AND schema_version = {schema_version}",
            partition_id=partition_id,
        )

    # Declarations can be narrowed by exact symbol, target name, or candidate
    # file. References cannot be narrowed the same way: aliases and renamed
    # re-exports give downstream rows arbitrary local target names, so finding a
    # conservative subset requires an iterative module-graph walk.
    def declaration_shapes(
        self,
        project_id: str,
        qualified_symbol: str,
        *,
        schema_version: int | None = None,
        version: int | None = None,
        partition_id: str | None = None,
    ) -> list[ReferenceRecord]:
        self._validate_schema_version(schema_version)
        condition = (
            f"record_kind = 'declaration' AND source_qualified_symbol = {_quoted(qualified_symbol)}"
        )
        if schema_version is not None:
            condition = f"{condition} AND schema_version = {schema_version}"
        return self._reference_rows(
            project_id, condition, version=version, partition_id=partition_id
        )

    def target_name_candidates(
        self,
        project_id: str,
        target_name: str,
        *,
        record_kind: str | None = None,
        schema_version: int | None = None,
        version: int | None = None,
        partition_id: str | None = None,
    ) -> list[ReferenceRecord]:
        self._validate_schema_version(schema_version)
        condition = f"target_name = {_quoted(target_name)}"
        if record_kind is not None:
            condition = f"record_kind = {_quoted(record_kind)} AND {condition}"
        if schema_version is not None:
            condition = f"{condition} AND schema_version = {schema_version}"
        return self._reference_rows(
            project_id, condition, version=version, partition_id=partition_id
        )

    def declarations_for_files(
        self,
        project_id: str,
        file_ids: Iterable[str],
        *,
        schema_version: int | None = None,
        version: int | None = None,
        partition_id: str | None = None,
    ) -> list[ReferenceRecord]:
        """Return declaration rows for exactly the given files.

        `_lexical_declaration`/class-scope resolution only ever compares a
        declaration against a reference row in the *same* file, so callers
        that already narrowed to a candidate file set never need the whole
        project's declaration table.
        """
        self._validate_schema_version(schema_version)
        ids = sorted(set(file_ids))
        if not ids:
            return []
        values = ", ".join(_quoted(file_id) for file_id in ids)
        condition = f"record_kind = 'declaration' AND file_id IN ({values})"
        if schema_version is not None:
            condition = f"{condition} AND schema_version = {schema_version}"
        return self._reference_rows(
            project_id, condition, version=version, partition_id=partition_id
        )

    def list_chunks(
        self,
        project_ids: Iterable[str] | None = None,
        *,
        partition_ids: Mapping[str, str] | None = None,
    ) -> list[IndexedChunk]:
        ids = list(project_ids or [project.id for project in self.list_projects()])
        self._validate_partition_mapping(ids, partition_ids)
        chunks: list[IndexedChunk] = []
        for project_id in ids:
            partition_id = None if partition_ids is None else partition_ids[project_id]
            tables = self._project_existing_tables(project_id, partition_id=partition_id)
            if tables is None:
                continue
            chunks.extend(
                IndexedChunk.model_validate(row)
                for row in cast(
                    list[dict[str, Any]],
                    tables.chunks.search().select(INDEXED_CHUNK_COLUMNS).to_list(),
                )
            )
        return chunks

    def get_chunk(self, chunk_id: str, *, partition_id: str | None = None) -> CodeChunk | None:
        # New-format ids carry a project-routing prefix
        # ("<project-id>:<digest>"). The registry resolves that logical owner,
        # then the active pointer selects the physical partition. A malformed
        # id, one whose prefix names no project, or one from a pre-migration
        # generation is treated as unknown -- consistent with the existing
        # contract that chunk ids change when a file is re-indexed.
        project_id = self._chunk_project_id(chunk_id)
        if project_id is None:
            return None
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return None
        rows = cast(
            list[dict[str, Any]],
            tables.chunks.search()
            .where(f"chunk_id = {_quoted(chunk_id)}")
            .select(CHUNK_PAYLOAD_COLUMNS)
            .to_list(),
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["project_id"] = project_id
        return CodeChunk.model_validate(row)

    @staticmethod
    def _chunk_id_prefix(chunk_id: str) -> str | None:
        """The project-routing prefix of a chunk id, or None when malformed."""
        if ":" not in chunk_id:
            return None
        prefix, _, remainder = chunk_id.partition(":")
        if not prefix or not remainder:
            return None
        return prefix

    def _chunk_project_id(self, chunk_id: str) -> str | None:
        """Resolve a chunk id's routing prefix through the logical registry."""
        prefix = self._chunk_id_prefix(chunk_id)
        if prefix is None:
            return None
        if self._rows(self._projects, f"id = {_quoted(prefix)}"):
            return prefix
        return None

    def count_chunks(
        self,
        project_ids: Iterable[str] | None = None,
        *,
        partition_ids: Mapping[str, str] | None = None,
    ) -> int:
        ids = list(project_ids or [project.id for project in self.list_projects()])
        self._validate_partition_mapping(ids, partition_ids)
        tables = (
            self._project_existing_tables(
                project_id,
                partition_id=None if partition_ids is None else partition_ids[project_id],
            )
            for project_id in ids
        )
        return sum(table.chunks.count_rows() for table in tables if table is not None)

    def hybrid_search(
        self,
        query_text: str,
        vector: list[float],
        project_ids: Iterable[str],
        condition: str | None,
        limit: int,
        *,
        partition_ids: Mapping[str, str | Sequence[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the hybrid query across every pinned physical partition.

        A project may pin several partitions -- one per live checkout of a
        shared registration -- so each project's entry may be one partition id
        or a sequence of them.
        """
        ids = list(project_ids)
        if not ids:
            return []
        self._validate_partition_mapping(ids, partition_ids)
        # Independent partitions are read concurrently through a small bounded
        # pool, so a multi-project query costs the slowest partition plus the
        # merge rather than the sum of every partition. Results are reassembled
        # in request order so relevance-score ties break exactly as the
        # sequential implementation did.
        tasks: list[tuple[str, str | None]] = []
        for project_id in ids:
            pinned = None if partition_ids is None else partition_ids[project_id]
            if pinned is None or isinstance(pinned, str):
                tasks.append((project_id, pinned))
            else:
                tasks.extend((project_id, partition_id) for partition_id in pinned)
        results: dict[int, list[dict[str, Any]]] = {}
        if len(tasks) == 1:
            project_id, partition_id = tasks[0]
            results[0] = self._hybrid_search_rows(
                project_id,
                query_text,
                vector,
                condition,
                limit,
                partition_id=partition_id,
            )
        else:
            workers = min(len(tasks), _SEARCH_CONCURRENCY)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    index: pool.submit(
                        self._hybrid_search_rows,
                        project_id,
                        query_text,
                        vector,
                        condition,
                        limit,
                        partition_id=partition_id,
                    )
                    for index, (project_id, partition_id) in enumerate(tasks)
                }
                for index, future in futures.items():
                    results[index] = future.result()
        rows = [row for index in range(len(tasks)) for row in results.get(index, [])]
        rows.sort(key=lambda row: float(row.get("_relevance_score", 0.0)), reverse=True)
        return rows[:limit]

    @staticmethod
    def _validate_partition_mapping(
        project_ids: Iterable[str], partition_ids: Mapping[str, Any] | None
    ) -> None:
        if partition_ids is None:
            return
        missing = sorted(set(project_ids) - set(partition_ids))
        if missing:
            raise ValueError(f"Missing physical partitions for projects: {', '.join(missing)}")

    def _hybrid_search_rows(
        self,
        project_id: str,
        query_text: str,
        vector: list[float],
        condition: str | None,
        limit: int,
        *,
        partition_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the hybrid query for one partition, injecting its project id."""
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return []
        # MultiMatchQuery spans the two single-column FTS indexes (content
        # and identifier_terms); a plain string would silently search only
        # one of them whenever both exist.
        query = (
            tables.chunks.search(query_type="hybrid", vector_column_name="vector")
            .vector(vector)
            .text(
                MultiMatchQuery(
                    query_text,
                    ["content", "identifier_terms"],
                    boosts=None,
                    operator=FullTextOperator.OR,
                )
            )
        )
        if condition:
            query = query.where(condition, prefilter=True)
        query = (
            query.limit(limit)
            .select(
                [
                    "chunk_id",
                    "path",
                    "language",
                    "kind",
                    "symbol",
                    "qualified_symbol",
                    "parent_symbol",
                    "start_line",
                    "end_line",
                    "content",
                ]
            )
            .rerank()
        )
        if self.vector_index == "exact":
            query = query.bypass_vector_index()
        # project_id is not stored on chunk rows; it belongs to the
        # partition being searched, so it is injected per project.
        query_rows = cast(list[dict[str, Any]], query.to_list())
        for row in query_rows:
            row["project_id"] = project_id
        return query_rows

    def find_symbol_chunks(
        self,
        name: str,
        project_id: str,
        *,
        match: str,
        kinds: list[str] | None,
        limit: int,
        partition_id: str | None = None,
    ) -> list[ChunkPreview]:
        escaped = _quoted(name)
        if match == "exact":
            symbol = f"(qualified_symbol = {escaped} OR symbol = {escaped})"
        elif match == "prefix":
            prefix = _quoted(name + "%")
            symbol = f"(qualified_symbol LIKE {prefix} OR symbol LIKE {prefix})"
        else:
            contains = _quoted("%" + name + "%")
            symbol = f"(qualified_symbol LIKE {contains} OR symbol LIKE {contains})"
        conditions = [symbol]
        if kinds:
            values = ", ".join(_quoted(kind) for kind in kinds)
            conditions.append(f"kind IN ({values})")
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return []
        # LIKE is only a pushdown pre-filter. The query engine ignores escape
        # sequences, so `_` and `%` inside an identifier stay wildcards and the
        # predicate over-matches (`load_user` also matches `loadXuser`). It never
        # under-matches, so exact semantics are re-applied below. Over-fetch so
        # the caller's limit is applied to real matches in a stable order.
        scan_limit = max(limit * OVERFETCH_FACTOR, MINIMUM_OVERFETCH)
        rows = self._projected_chunks(
            tables.chunks,
            " AND ".join(conditions),
            limit=scan_limit,
            content=True,
            order_by=["path", "start_line", "kind"],
        )
        for row in rows:
            row["project_id"] = project_id
        if len(rows) == scan_limit:
            # The pre-filter filled the scan window, so real matches sorting
            # after it were never seen. Silent truncation is otherwise
            # indistinguishable from "no more matches exist".
            logger.debug(
                "Symbol pre-filter for %r in project %s hit the %d-row scan cap; "
                "later exact matches may be missing",
                name,
                project_id,
                scan_limit,
            )
        matches = [
            preview
            for preview in (ChunkPreview.model_validate(row) for row in rows)
            if _symbol_matches(preview, name, match)
        ]
        return matches[:limit]

    def outline_chunks(
        self, path: str, project_id: str, *, partition_id: str | None = None
    ) -> list[ChunkPreview]:
        tables = self._project_existing_tables(project_id, partition_id=partition_id)
        if tables is None:
            return []
        condition = (
            f"path = {_quoted(path)} AND symbol IS NOT NULL AND qualified_symbol IS NOT NULL"
        )
        rows = self._projected_chunks(
            tables.chunks,
            condition,
            limit=None,
            content=False,
        )
        for row in rows:
            row["project_id"] = project_id
        return [ChunkPreview.model_validate(row) for row in rows]

    def ensure_indexes(self, project_id: str, *, partition_id: str | None = None) -> None:
        """Create missing search indexes on the write path.

        Index *existence* is a correctness requirement for search, so missing
        FTS and BTree indexes are created after a commit. Compaction, index
        optimization, and old-version cleanup are deliberately not part of this
        method: they run only in ``maintain_project`` on a schedule, so an
        incremental commit stops paying for a full optimize pass. Searches
        combine indexed rows with the unindexed tail until maintenance folds
        that tail into the indexes.
        """
        tables = self._project_tables(project_id, partition_id=partition_id)
        chunks = tables.chunks
        indices = list(chunks.list_indices())
        indexed_columns = {column for index in indices for column in index.columns}
        # Native FTS in the supported lancedb range indexes one field per
        # index, so the persisted source content and the compact normalized
        # identifier terms get one FTS index each and searches use a
        # MultiMatchQuery spanning both. That covers code-text keywords and
        # camelCase/snake_case symbol and path names without storing a second
        # full copy of the text.
        for fts_column in ("content", "identifier_terms"):
            if fts_column not in indexed_columns:
                chunks.create_index(
                    fts_column,
                    config=FTS(lower_case=True, stem=False, remove_stop_words=False),
                    replace=False,
                )
        for column in ("file_id", "language", "path", "symbol"):
            if column not in indexed_columns:
                chunks.create_index(column, config=BTree(), replace=False)
        vector_indices = [index for index in indices if "vector" in index.columns]
        if self.vector_index == "exact":
            for index in vector_indices:
                chunks.drop_index(index.name)
        elif not vector_indices and chunks.count_rows() >= 20_000:
            chunks.create_index(
                "vector",
                config=HnswSq(distance_type="cosine"),
                replace=False,
            )
        if tables.references is None:
            raise RuntimeError("Reference table is missing from an interrupted transaction")
        reference_indices = list(tables.references.list_indices())
        indexed_reference_columns = {
            column for index in reference_indices for column in index.columns
        }
        for column in (
            "file_id",
            "record_kind",
            "target_name",
            "module_path",
            "kind",
            "source_qualified_symbol",
            "schema_version",
        ):
            if column not in indexed_reference_columns:
                tables.references.create_index(column, config=BTree(), replace=False)

    def maintain_project(
        self,
        project_id: str,
        *,
        cleanup_older_than: timedelta,
        branch_cache_limit: int | None = None,
        protected_slot_ids: Iterable[str] = (),
    ) -> bool:
        """Compact and clean *project_id*'s partition, returning whether it ran.

        Optimizes the files, chunks, and references tables, reclaiming space
        after deletions. Versions older than ``cleanup_older_than`` are removed
        once verified; ``delete_unverified`` is never set, and a zero age must
        never be passed by an automatic path. A registered project with no
        partition is left untouched (``False``) rather than materialized.
        """
        slots = self.list_slots(project_id)
        limit = self.branch_cache_limit if branch_cache_limit is None else branch_cache_limit
        recovery_protected = set(protected_slot_ids)
        # Every checkout pointer protects its slot: a worktree of this project
        # may be holding another branch live while maintenance runs.
        pointed = {str(row["slot_id"]) for row in self._active_pointer_rows(project_id)}
        protected = recovery_protected | pointed

        def evict(slot: ProjectSlot) -> None:
            with self._partition_access_physical(slot.partition_id):
                self.remove_slot(slot.slot_id)
                self._advance_partition_generation(slot.partition_id)
                with self._partitions_lock:
                    self._partitions.pop(slot.partition_id, None)
                partition = self.directory / "projects" / slot.partition_id
                if partition.exists():
                    shutil.rmtree(partition)
                generation = self.directory / "partition-generations" / slot.partition_id
                if generation.exists():
                    generation.unlink()

        # Branch slots keyed by an obsolete slot-key formula can never be
        # claimed by any checkout again (they only survive from before an
        # upgrade), so they are reclaimed first rather than competing with
        # live slots for the retention budget.
        stale_ids = self._stale_slot_ids(project_id, slots) - protected
        stale = sorted(
            (slot for slot in slots if slot.slot_id in stale_ids and slot.state != "indexing"),
            key=lambda slot: (slot.last_used_at, slot.slot_id),
        )
        for slot in stale:
            evict(slot)

        remaining = [slot for slot in self.list_slots(project_id) if slot.slot_id not in protected]
        candidates = sorted(
            (slot for slot in remaining if slot.state != "indexing"),
            key=lambda slot: (slot.last_used_at, slot.slot_id),
        )
        remove_count = max(0, len(remaining) - limit)
        for slot in candidates[:remove_count]:
            evict(slot)

        ran = bool(stale) or bool(candidates[:remove_count])
        for slot in self.list_slots(project_id):
            if slot.slot_id in recovery_protected:
                continue
            tables = self._project_existing_tables(project_id, partition_id=slot.partition_id)
            if tables is None:
                continue
            with self._partition_access_physical(slot.partition_id):
                tables.files.optimize(cleanup_older_than=cleanup_older_than)
                tables.chunks.optimize(cleanup_older_than=cleanup_older_than)
                if tables.references is not None:
                    tables.references.optimize(cleanup_older_than=cleanup_older_than)
            ran = True
        return ran or bool(slots)

    def _active_pointer_rows(self, project_id: str) -> list[dict[str, Any]]:
        return self._rows(self._active_slots, f"project_id = {_quoted(project_id)}")

    def _stale_slot_ids(self, project_id: str, slots: list[ProjectSlot]) -> set[str]:
        """Return branch/commit slots no current slot key can ever claim.

        The slot-key formula is versioned; rows written before a formula
        change keep their old identifiers. Selector kind, value, repository
        identity, and prefix are all stored on the row, so the identity each
        row *would* claim today is recomputable -- and a mismatch means the
        row is unreachable by design, never merely out of date.
        """
        stale: set[str] = set()
        branch_kinds = {SelectorKind.REF.value, SelectorKind.COMMIT.value}
        for slot in slots:
            if slot.selector_kind not in branch_kinds:
                continue
            synthetic = GitState(
                probe=GitProbeOutcome.GIT,
                selector_kind=SelectorKind(slot.selector_kind),
                selector_value=slot.selector_value,
                repository_identity=slot.repository_identity,
                project_prefix=slot.project_prefix,
            )
            if _git_slot_id(project_id, synthetic) != slot.slot_id:
                stale.add(slot.slot_id)
        return stale

    def maintain_registry(self, *, cleanup_older_than: timedelta) -> None:
        """Compact and clean the project registry table."""
        self._projects.optimize(cleanup_older_than=cleanup_older_than)

    def remove_project(self, project_id: str, *, locks_held: bool = False) -> bool:
        """Remove a registration, every slot row, every owned partition, and its pointer."""
        if locks_held:
            return self._remove_project_unlocked(project_id)
        lock_directory = self.directory.parent / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        with (
            FileLock(lock_directory / "index-global.lock"),
            FileLock(lock_directory / f"{project_id}.lock"),
            FileLock(lock_directory / f"active-{project_id}.lock"),
        ):
            return self._remove_project_unlocked(project_id)

    def _remove_project_unlocked(self, project_id: str) -> bool:
        existed = bool(self._rows(self._projects, f"id = {_quoted(project_id)}"))
        partition_ids = {slot.partition_id for slot in self.list_slots(project_id)}
        # A never-adopted project still owns its legacy partition directory.
        partition_ids.add(project_id)
        with ExitStack() as stack:
            for partition_id in sorted(partition_ids):
                stack.enter_context(self._partition_access_physical(partition_id))
            self._projects.delete(f"id = {_quoted(project_id)}")
            self._project_slots.delete(f"project_id = {_quoted(project_id)}")
            self._active_slots.delete(f"project_id = {_quoted(project_id)}")
            for partition_id in sorted(partition_ids):
                with self._partitions_lock:
                    self._partitions.pop(partition_id, None)
                partition = self.directory / "projects" / partition_id
                if partition.exists():
                    shutil.rmtree(partition)
                generation = self.directory / "partition-generations" / partition_id
                if generation.exists():
                    generation.unlink()
        return existed

    @staticmethod
    def _project_schema() -> pa.Schema:
        return pa.schema(
            [
                ("id", pa.string()),
                ("name", pa.string()),
                ("root", pa.string()),
                ("payload", pa.string()),
                ("model_id", pa.string()),
                ("vector_dimension", pa.int32()),
                ("schema_version", pa.int32()),
                ("state", pa.string()),
                ("updated_at", pa.int64()),
            ]
        )

    @staticmethod
    def _project_slot_schema() -> pa.Schema:
        return pa.schema(
            [
                ("slot_id", pa.string()),
                ("project_id", pa.string()),
                ("partition_id", pa.string()),
                ("selector_kind", pa.string()),
                ("selector_value", pa.string()),
                ("repository_identity", pa.string()),
                ("checkout_identity", pa.string()),
                ("project_prefix", pa.string()),
                ("indexed_head", pa.string()),
                ("indexed_clean", pa.bool_()),
                ("indexed_status_fingerprint", pa.string()),
                ("indexed_status_paths", pa.string()),
                ("scan_config_hash", pa.string()),
                ("model_id", pa.string()),
                ("vector_dimension", pa.int32()),
                ("schema_version", pa.int32()),
                ("state", pa.string()),
                ("created_at", pa.int64()),
                ("last_used_at", pa.int64()),
            ]
        )

    @staticmethod
    def _active_slot_schema() -> pa.Schema:
        return pa.schema(
            [
                ("project_id", pa.string()),
                ("slot_id", pa.string()),
                # One pointer per live checkout, not per project: two
                # worktrees of one project keep their different slots active
                # at the same time. Empty on rows written before worktree
                # support; such rows stay addressable but no lookup reuses
                # the empty key, so every checkout converges onto its own.
                ("checkout_key", pa.string()),
                ("activation_epoch", pa.int64()),
                ("updated_at", pa.int64()),
            ]
        )

    @staticmethod
    def _file_schema() -> pa.Schema:
        return pa.schema(
            [
                ("file_id", pa.string()),
                ("project_id", pa.string()),
                ("path", pa.string()),
                ("language", pa.string()),
                ("size", pa.int64()),
                ("mtime_ns", pa.int64()),
                ("content_hash", pa.string()),
                ("has_errors", pa.bool_()),
                ("error", pa.string()),
                ("indexed_at", pa.int64()),
            ]
        )

    @staticmethod
    def _chunk_schema(vector_dimension: int, vector_dtype: pa.DataType) -> pa.Schema:
        return pa.schema(
            [
                ("chunk_id", pa.string()),
                ("file_id", pa.string()),
                ("path", pa.string()),
                ("language", pa.string()),
                ("kind", pa.string()),
                ("symbol", pa.string()),
                ("qualified_symbol", pa.string()),
                ("parent_symbol", pa.string()),
                ("start_byte", pa.int64()),
                ("end_byte", pa.int64()),
                ("start_line", pa.int32()),
                ("end_line", pa.int32()),
                ("content", pa.string()),
                # Compact normalized identifier terms (camelCase and snake_case
                # split apart) that FTS searches alongside ``content``, so
                # symbol, path, and qualified-name queries match without
                # storing a second full copy of the source text.
                ("identifier_terms", pa.string()),
                ("content_hash", pa.string()),
                ("part_index", pa.int32()),
                (
                    "vector",
                    pa.list_(vector_dtype, vector_dimension),
                ),
            ]
        )

    @staticmethod
    def _reference_schema() -> pa.Schema:
        return pa.schema(
            [
                ("reference_id", pa.string()),
                ("record_kind", pa.string()),
                ("file_id", pa.string()),
                ("project_id", pa.string()),
                ("path", pa.string()),
                ("language", pa.string()),
                ("kind", pa.string()),
                ("source_qualified_symbol", pa.string()),
                ("written_name", pa.string()),
                ("target_name", pa.string()),
                ("module_path", pa.string()),
                ("imported_name", pa.string()),
                ("alias", pa.string()),
                ("receiver_text", pa.string()),
                ("start_byte", pa.int64()),
                ("end_byte", pa.int64()),
                ("start_line", pa.int32()),
                ("end_line", pa.int32()),
                ("shape_json", pa.string()),
                ("content_hash", pa.string()),
                ("schema_version", pa.int32()),
            ]
        )

    def _cached(self, partition_id: str) -> _ProjectTables | None:
        """Return the cached partition, marking it recently used."""
        with self._partitions_lock:
            cached = self._partitions.get(partition_id)
            if cached is not None:
                self._partitions.move_to_end(partition_id)
            return cached

    @contextmanager
    def partition_access(
        self, project_id: str, *, partition_id: str | None = None
    ) -> Iterator[None]:
        """Prevent a destructive rebuild from invalidating an active query."""
        physical = self._partition_id_for(project_id) if partition_id is None else partition_id
        with self._partition_access_physical(physical):
            yield

    @contextmanager
    def _partition_access_physical(self, partition_id: str) -> Iterator[None]:
        """Lock one already-resolved physical partition."""
        directory = self.directory.parent / "locks"
        directory.mkdir(parents=True, exist_ok=True)
        with FileLock(directory / f"partition-{partition_id}.lock"):
            yield

    @contextmanager
    def partitions_access(
        self,
        project_ids: Iterable[str]
        | Mapping[str, str | PartitionRef | Sequence[PartitionRef]]
        | Iterable[PartitionRef],
    ) -> Iterator[None]:
        """Hold the pinned partitions of logical projects in stable order.

        A project's mapping entry may be one partition or a sequence of them
        (one per requested checkout of a shared registration); every distinct
        physical partition is locked exactly once.
        """
        partition_ids: set[str] = set()

        def collect(value: str | PartitionRef | Sequence[PartitionRef]) -> None:
            if isinstance(value, PartitionRef):
                partition_ids.add(value.partition_id)
            elif isinstance(value, str):
                partition_ids.add(value)
            else:
                for ref in value:
                    partition_ids.add(ref.partition_id)

        if isinstance(project_ids, Mapping):
            for value in project_ids.values():
                collect(value)
        else:
            for value in project_ids:
                if isinstance(value, PartitionRef):
                    partition_ids.add(value.partition_id)
                else:
                    partition_ids.add(self._partition_id_for(value))
        with ExitStack() as stack:
            for partition_id in sorted(partition_ids):
                stack.enter_context(self._partition_access_physical(partition_id))
            yield

    def _remember(self, partition_id: str, tables: _ProjectTables) -> _ProjectTables:
        """Cache *tables*, evicting the least recently used partition past the bound.

        Eviction only drops this dictionary's reference. A caller mid-query holds its
        own reference to the tables, so the underlying dataset stays open until that
        caller is done — the daemon serves each client on its own thread and must not
        have a table closed underneath it.

        A caller that just built a complete partition (write path, references table
        created) must win the cache slot even if a concurrent read of the same
        partition cached a partial object while the tables were still being created;
        returning the partial object would make the write path hand its own commit a
        partition whose references table it can no longer see.
        """
        with self._partitions_lock:
            existing = self._partitions.get(partition_id)
            if existing is not None:
                # Another thread opened it first; keep one instance so both callers
                # share a single set of handles.
                self._partitions.move_to_end(partition_id)
                if existing.references is None and tables.references is not None:
                    self._partitions[partition_id] = tables
                    self._partitions.move_to_end(partition_id)
                    return tables
                return existing
            self._partitions[partition_id] = tables
            while len(self._partitions) > MAX_CACHED_PARTITIONS:
                self._partitions.popitem(last=False)
            return tables

    def _project_tables(
        self, project_id: str, *, partition_id: str | None = None
    ) -> _ProjectTables:
        """Open *project_id*'s active partition, creating it. For write paths."""
        physical = self._partition_id_for(project_id) if partition_id is None else partition_id
        return self._tables(physical)

    def _project_existing_tables(
        self, project_id: str, *, partition_id: str | None = None
    ) -> _ProjectTables | None:
        """Open *project_id*'s active partition without creating it, or None."""
        physical = self._partition_id_for(project_id) if partition_id is None else partition_id
        return self._existing_tables(physical)

    def _tables(self, partition_id: str) -> _ProjectTables:
        """Open one physical partition, creating it. For write paths only."""
        cached = self._cached(partition_id)
        if (
            cached is not None
            and cached.references is not None
            and cached.generation == self._partition_generation(partition_id)
        ):
            return cached
        database = lancedb.connect(
            self.directory / "projects" / partition_id,
            read_consistency_interval=timedelta(0),
        )
        tables = _ProjectTables(
            files=self._table(database, "files", self._file_schema()),
            chunks=self._table(
                database,
                "chunks",
                self._chunk_schema(self.vector_dimension, self.vector_dtype),
            ),
            references=self._table(database, "references", self._reference_schema()),
            generation=self._partition_generation(partition_id),
        )
        if cached is not None:
            return self._replace_cached(partition_id, tables)
        return self._remember(partition_id, tables)

    def _replace_cached(self, partition_id: str, tables: _ProjectTables) -> _ProjectTables:
        """Replace a cached legacy partition after adding its references table."""
        with self._partitions_lock:
            self._partitions[partition_id] = tables
            self._partitions.move_to_end(partition_id)
        return tables

    def _existing_tables(self, partition_id: str) -> _ProjectTables | None:
        """Open one physical partition without creating it, or return None.

        Reads must not materialise storage for a partition they are only
        looking at. get_chunk in particular scans every registered project, so
        going through the create-on-write _tables() would leave an empty
        partition directory behind for each project that has never been
        indexed.
        """
        cached = self._cached(partition_id)
        directory = self.directory / "projects" / partition_id
        generation = self._partition_generation(partition_id)
        if cached is not None:
            if directory.is_dir() and cached.generation == generation:
                return cached
            with self._partitions_lock:
                self._partitions.pop(partition_id, None)
        if not directory.is_dir():
            return None
        database = lancedb.connect(directory, read_consistency_interval=timedelta(0))
        try:
            tables = _ProjectTables(
                files=cast(LanceTable, database.open_table("files")),
                chunks=cast(LanceTable, database.open_table("chunks")),
                references=self._open_optional_table(database, "references"),
                generation=generation,
            )
        except (ValueError, FileNotFoundError):
            return None
        return self._remember(partition_id, tables)

    def _partition_generation(self, partition_id: str) -> int:
        path = self.directory / "partition-generations" / partition_id
        try:
            return int(path.read_text())
        except (OSError, ValueError):
            return 0

    def _advance_partition_generation(self, partition_id: str) -> None:
        path = self.directory / "partition-generations" / partition_id
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(str(self._partition_generation(partition_id) + 1))
        os.replace(temporary, path)

    @staticmethod
    def _table(database: Any, name: str, schema: pa.Schema) -> LanceTable:
        return cast(
            LanceTable,
            database.create_table(name, schema=schema, exist_ok=True),
        )

    @staticmethod
    def file_arrow_schema() -> pa.Schema:
        return LanceStore._file_schema()

    @staticmethod
    def chunk_arrow_schema(vector_dimension: int, vector_dtype: pa.DataType) -> pa.Schema:
        return LanceStore._chunk_schema(vector_dimension, vector_dtype)

    @staticmethod
    def reference_arrow_schema() -> pa.Schema:
        return LanceStore._reference_schema()

    @staticmethod
    def _merge(
        table: LanceTable,
        key: str | Sequence[str],
        rows: list[dict[str, Any]] | pa.Table,
    ) -> None:
        (
            table.merge_insert(key)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    @staticmethod
    def _validate_schema_version(schema_version: int | None) -> None:
        if schema_version is not None and (
            isinstance(schema_version, bool) or not isinstance(schema_version, int)
        ):
            raise ValueError("schema_version must be a non-boolean integer")

    def registry_stats(self) -> TableStorageStats:
        """Snapshot the project registry table's storage statistics."""
        return self._table_storage_stats(
            self._projects, "projects", physical_directory=self.directory / "registry"
        )

    def storage_stats(self, project_id: str) -> ProjectStorageStats:
        """Collect read-only storage statistics for one project partition.

        A registered project with no partition yields zeroed tables rather than
        materializing storage (reads never create partitions). ``consistent``
        is False when any table's version changed between the initial snapshot
        and the end of collection, meaning the observations do not form one
        atomic snapshot.
        """
        project = next((p for p in self.list_projects() if p.id == project_id), None)
        if project is None:
            raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {project_id}")
        return self.storage_stats_for(project)

    def storage_stats_for(
        self, project: ProjectInfo, *, partition_ref: PartitionRef | None = None
    ) -> ProjectStorageStats:
        """Collect read-only storage statistics for an already-resolved project.

        Unlike ``storage_stats``, this does not re-scan the registry, so an
        installation-wide report can resolve every project once and then
        collect each partition without N+1 registry reads.
        """
        active = partition_ref or self._active_partition_ref(project.id)
        if active is not None and active.project_id != project.id:
            raise ValueError("partition does not belong to project")
        partition_id = project.id if active is None else active.partition_id
        tables = self._existing_tables(partition_id)
        partition_path = self.directory / "projects" / partition_id
        before = self._partition_versions(tables)
        collected: list[TableStorageStats] = []
        if tables is not None:
            collected.append(
                self._table_storage_stats(
                    tables.files, "files", physical_directory=partition_path / "files.lance"
                )
            )
            collected.append(
                self._table_storage_stats(
                    tables.chunks, "chunks", physical_directory=partition_path / "chunks.lance"
                )
            )
            if tables.references is not None:
                collected.append(
                    self._table_storage_stats(
                        tables.references,
                        "references",
                        physical_directory=partition_path / "references.lance",
                    )
                )
        # Walked before the closing version snapshot so it falls inside the
        # consistency window: a commit landing during the walk must make the
        # report inconsistent, not yield byte counts that silently disagree
        # with the table statistics collected above.
        partition_physical_bytes = _directory_bytes(partition_path)
        after = self._partition_versions(tables)
        # The partition directory exists but its tables could not be opened
        # (a damaged or mid-mutation store is exactly what status is for); a
        # project that was never indexed has no directory at all. The two must
        # not be conflated: report the failure explicitly and treat the
        # snapshot as unusable.
        open_failed = tables is None and partition_path.is_dir()
        # With per-checkout pointers a project can have several live slots at
        # once (one worktree per branch), so "active" flags every pointed slot.
        pointed = {str(row["slot_id"]) for row in self._active_pointer_rows(project.id)}
        slots = [
            SlotStorageStats(
                slot_id=slot.slot_id,
                partition_id=slot.partition_id,
                selector_kind=slot.selector_kind,
                selector_value=slot.selector_value,
                active=slot.slot_id in pointed,
                state=slot.state,
                indexed_head=slot.indexed_head,
                indexed_clean=slot.indexed_clean,
                last_used_at=slot.last_used_at,
                physical_bytes=(
                    partition_physical_bytes
                    if slot.partition_id == partition_id
                    else _directory_bytes(self.directory / "projects" / slot.partition_id)
                ),
            )
            for slot in self.list_slots(project.id)
        ]
        return ProjectStorageStats(
            project=project,
            snapshot_at=datetime.now(UTC).isoformat(),
            tables=collected,
            slots=slots,
            partition_physical_bytes=partition_physical_bytes,
            consistent=before == after and not open_failed,
            partition_open_failed=open_failed,
        )

    @staticmethod
    def _partition_versions(tables: _ProjectTables | None) -> tuple[int, int, int]:
        """Snapshot the three partition table versions, 0 for a missing references table."""
        if tables is None:
            return (0, 0, 0)
        references = 0 if tables.references is None else int(tables.references.version)
        return (int(tables.files.version), int(tables.chunks.version), references)

    @staticmethod
    def _table_storage_stats(
        table: LanceTable,
        name: str,
        *,
        physical_directory: Path,
    ) -> TableStorageStats:
        """Collect one Lance table's storage snapshot.

        Introspection is best-effort: a table whose statistics cannot be read
        (a damaged or mid-mutation store is exactly what status is for) reports
        its version and nothing else rather than failing the whole report.
        """
        try:
            stats = cast(dict[str, Any], cast(object, table.stats()))
        except (ValueError, RuntimeError):
            stats = {}
        try:
            num_rows = max(0, int(stats.get("num_rows", 0)))
            logical_bytes = max(0, int(stats.get("total_bytes", 0)))
        except (TypeError, ValueError):
            num_rows = 0
            logical_bytes = 0
        fragments: dict[str, Any] = stats.get("fragment_stats") or {}
        lengths: dict[str, Any] | None = fragments.get("lengths")
        try:
            fragment_stats = FragmentStats(
                num_fragments=max(0, int(fragments.get("num_fragments", 0))),
                num_small_fragments=max(0, int(fragments.get("num_small_fragments", 0))),
                lengths=(
                    FragmentLengthStats(
                        **{key: _nullable_int(value) for key, value in lengths.items()}
                    )
                    if lengths
                    else None
                ),
            )
        except (TypeError, ValueError):
            fragment_stats = FragmentStats()
        try:
            versions = cast(list[dict[str, Any]], table.list_versions())
        except (ValueError, RuntimeError):
            versions = []
        timestamps = [
            str(version.get("timestamp", "")) for version in versions if version.get("timestamp")
        ]
        indexes: list[IndexStorageStats] = []
        try:
            for index in table.list_indices():
                indexes.append(
                    IndexStorageStats(
                        name=index.name,
                        index_type=index.index_type,
                        columns=list(index.columns),
                        indexed_rows=max(0, int(getattr(index, "num_indexed_rows", 0) or 0)),
                        unindexed_rows=max(0, int(getattr(index, "num_unindexed_rows", 0) or 0)),
                        size_bytes=max(0, int(getattr(index, "size_bytes", 0) or 0)),
                    )
                )
        except (ValueError, RuntimeError):
            indexes = []
        return TableStorageStats(
            name=name,
            current_version=int(table.version),
            row_count=num_rows,
            logical_bytes=logical_bytes,
            physical_bytes=_directory_bytes(physical_directory),
            fragment_stats=fragment_stats,
            retained_version_count=len(versions),
            oldest_version_at=_normalized_timestamp(timestamps[0]) if timestamps else None,
            newest_version_at=_normalized_timestamp(timestamps[-1]) if timestamps else None,
            indexes=indexes,
        )

    @staticmethod
    def _rows(table: LanceTable, condition: str | None = None) -> list[dict[str, Any]]:
        query = table.search()
        if condition:
            query = query.where(condition)
        return cast(list[dict[str, Any]], query.to_list())

    def _reference_rows(
        self,
        project_id: str,
        condition: str | None,
        *,
        version: int | None = None,
        partition_id: str | None = None,
    ) -> list[ReferenceRecord]:
        # Keep the physical partition fixed for the whole read. The active
        # pointer may switch while a historical version is being reopened.
        physical = self._partition_id_for(project_id) if partition_id is None else partition_id
        tables = self._existing_tables(physical)
        if tables is None or tables.references is None:
            return []
        references = tables.references
        if version is not None and version != int(references.version):
            database = lancedb.connect(
                self.directory / "projects" / physical,
                read_consistency_interval=timedelta(0),
            )
            references = cast(LanceTable, database.open_table("references", version=version))
        query = references.search()
        if condition:
            query = query.where(condition)
        query = query.order_by(
            [
                ColumnOrdering(column_name="path"),
                ColumnOrdering(column_name="start_line"),
                ColumnOrdering(column_name="reference_id"),
            ]
        )
        return cast(list[ReferenceRecord], query.to_list())

    @staticmethod
    def _open_optional_table(database: Any, name: str) -> LanceTable | None:
        try:
            return cast(LanceTable, database.open_table(name))
        except (ValueError, FileNotFoundError):
            return None

    @staticmethod
    def _projected_chunks(
        table: LanceTable,
        condition: str,
        *,
        limit: int | None,
        content: bool,
        order_by: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # project_id is deliberately absent: it is not stored on chunk rows
        # and belongs to the owning partition, so callers inject it.
        columns = [
            "chunk_id",
            "path",
            "language",
            "kind",
            "symbol",
            "qualified_symbol",
            "parent_symbol",
            "start_line",
            "end_line",
        ]
        if content:
            columns.append("content")
        query = table.search().where(condition).select(columns)
        if order_by is not None:
            # A stable scan order makes a truncated result set deterministic
            # rather than dependent on physical row layout.
            query = query.order_by([ColumnOrdering(column_name=name) for name in order_by])
        if limit is not None:
            query = query.limit(limit)
        return cast(list[dict[str, Any]], query.to_list())

    @classmethod
    def _migrate_v1(cls, directory: Path) -> list[dict[str, Any]]:
        legacy_projects = directory / "projects.lance"
        if not legacy_projects.exists():
            return []
        directory.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(directory.parent / f".{directory.name}-migrate.lock"):
            if not legacy_projects.exists():
                return []
            database = lancedb.connect(directory, read_consistency_interval=timedelta(0))
            table = database.open_table("projects")
            rows = cast(list[dict[str, Any]], table.search().to_list())
            del table
            del database
            gc.collect()
            backup = directory.with_name(f"{directory.name}-v1-backup-{time.time_ns()}")
            shutil.move(str(directory), str(backup))
            return rows

    def _migrate_active_checkouts(self, registry_directory: Path) -> None:
        """Add the per-checkout pointer key to a pre-worktree active_slots table.

        The pre-worktree layout kept exactly one active pointer per project
        and no ``checkout_key`` column. Existing rows are preserved verbatim
        under the empty key: they keep such legacy installations readable,
        while every real lookup filters on a concrete checkout identity (an
        absolute git-directory or root path) and therefore re-activates its
        own pointer on first touch.
        """
        lock_directory = registry_directory.parent.parent / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_directory / ".active-slots-migrate.lock"):
            try:
                existing = self._db.open_table("active_slots")
            except (ValueError, FileNotFoundError):
                return
            if "checkout_key" in {field.name for field in existing.schema}:
                return
            rows = self._rows(cast(LanceTable, existing))
            self._db.drop_table("active_slots")
            fresh = self._table(self._db, "active_slots", self._active_slot_schema())
            migrated = [
                {
                    "project_id": str(row["project_id"]),
                    "slot_id": str(row["slot_id"]),
                    "checkout_key": "",
                    "activation_epoch": int(row["activation_epoch"]),
                    "updated_at": int(row["updated_at"]),
                }
                for row in rows
            ]
            if migrated:
                self._merge(fresh, ["project_id", "checkout_key"], migrated)

    def _migrate_project_slot_status_columns(self, registry_directory: Path) -> None:
        """Add the status-fingerprint and status-paths columns to an older registry.

        Existing rows keep every other field verbatim; the two new columns
        come back null, which simply disables the freshness fast path for
        that slot until the next index run stamps it (see D1/D2 of the
        query-path-overhead remediation plan).
        """
        lock_directory = registry_directory.parent.parent / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_directory / ".project-slots-migrate.lock"):
            try:
                existing = self._db.open_table("project_slots")
            except (ValueError, FileNotFoundError):
                return
            if "indexed_status_fingerprint" in {field.name for field in existing.schema}:
                return
            rows = self._rows(cast(LanceTable, existing))
            self._db.drop_table("project_slots")
            fresh = self._table(self._db, "project_slots", self._project_slot_schema())
            migrated = [
                {
                    "slot_id": str(row["slot_id"]),
                    "project_id": str(row["project_id"]),
                    "partition_id": str(row["partition_id"]),
                    "selector_kind": str(row["selector_kind"]),
                    "selector_value": str(row["selector_value"]),
                    "repository_identity": _optional_str(row.get("repository_identity")),
                    "checkout_identity": _optional_str(row.get("checkout_identity")),
                    "project_prefix": str(row.get("project_prefix") or ""),
                    "indexed_head": _optional_str(row.get("indexed_head")),
                    "indexed_clean": (
                        None if row.get("indexed_clean") is None else bool(row["indexed_clean"])
                    ),
                    "indexed_status_fingerprint": None,
                    "indexed_status_paths": None,
                    "scan_config_hash": str(row.get("scan_config_hash") or ""),
                    "model_id": str(row.get("model_id") or ""),
                    "vector_dimension": int(row.get("vector_dimension") or 0),
                    "schema_version": int(row.get("schema_version") or 0),
                    "state": str(row.get("state") or "pending"),
                    "created_at": int(row.get("created_at") or 0),
                    "last_used_at": int(row.get("last_used_at") or 0),
                }
                for row in rows
            ]
            if migrated:
                self._merge(fresh, "slot_id", migrated)
