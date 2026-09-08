"""Storage maintenance: compaction, retention, and the scheduled-pass timestamp.

Split out of ``Application`` in the review's Track 5 (see
``docs/plans/2026-09-02-review-remediation-5-application-split-plan.md``, decision
D2) so storage housekeeping is owned by one object instead of being interleaved
with project registry, backend, and query code. ``Application`` constructs one
instance as ``self.maintenance`` and keeps one-line delegates -- with identical
signatures -- for the three methods every external caller reaches through
``Application`` itself:

- ``server.py`` calls ``app.storage_status(...)``, ``app.maintain_storage(...)``
  through the MCP tool surface, and both are part of the ``ApplicationLike``
  Protocol daemon clients rely on.
- ``daemon.py`` runs ``app.maybe_run_maintenance()`` on startup and exposes
  ``storage_status``/``maintain_storage`` over the broker.
- ``tests/test_application.py`` calls all three as ``app.storage_status(...)``,
  ``app.maintain_storage(...)``, and ``app.maybe_run_maintenance()`` unchanged.

``MaintenanceService`` needs the store, paths, and settings directly, plus
target resolution that only ``Application`` can provide: which projects are
registered, how to resolve one project selector to a ``ProjectInfo``, and how
to resolve a project's live ``ActiveIndexTarget`` (single, bulk, and the
repository-stable retry wrapper used for reads). Those are passed as explicit
callables so the dependency is visible and this service is constructible in
tests with lambdas, rather than taking the whole ``Application``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from filelock import FileLock, Timeout

from .errors import CodeIndexingError, ErrorCode
from .models import (
    MaintenanceProjectResult,
    MaintenanceReport,
    ProjectInfo,
    ProjectStorageStats,
    StorageStatus,
    TableStorageStats,
)
from .settings import IndexSettings
from .staging import pending_recovery
from .storage import ActiveIndexTarget, LanceStore, overlap_warnings, worktree_warnings

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from .application import RuntimePaths

logger = logging.getLogger(__name__)

_Result = TypeVar("_Result")

# Automatic maintenance repeats its overdue check at most this often. The check
# itself is gated by the persisted last-successful-maintenance timestamp.
MAINTENANCE_CHECK_INTERVAL = timedelta(hours=24)

MAINTENANCE_TIMESTAMP_FILE = "maintenance.json"
MAINTENANCE_LOCK_FILE = "maintenance-schedule.lock"


def _read_maintenance_timestamp(path: Path) -> datetime | None:
    """Return the last successful maintenance time, or None when unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(payload["last_maintenance_at"])
        return timestamp if timestamp.utcoffset() is not None else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_maintenance_timestamp(path: Path) -> None:
    """Persist the maintenance timestamp atomically so readers never parse a partial file."""
    payload = json.dumps(
        {
            "schema_version": 1,
            "last_maintenance_at": datetime.now(UTC).isoformat(),
        },
        sort_keys=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _estimate_reclaimable(stats: ProjectStorageStats) -> int:
    """Estimate bytes a table could reclaim: physical minus logical, floor zero."""
    return sum(max(0, table.physical_bytes - table.logical_bytes) for table in stats.tables)


def _versions_removed(before: ProjectStorageStats, after: ProjectStorageStats) -> int:
    """Sum retained versions removed across the tables shared by both snapshots."""
    removed = 0
    after_by_name = {table.name: table for table in after.tables}
    for table in before.tables:
        after_table = after_by_name.get(table.name)
        if after_table is not None:
            removed += max(0, table.retained_version_count - after_table.retained_version_count)
    return removed


def _primary_target(
    targets: Mapping[str, Sequence[ActiveIndexTarget]], project_id: str
) -> ActiveIndexTarget:
    """Return the primary -- first requested -- checkout's target.

    A private mirror of ``Application._primary_target``: that staticmethod is
    shared by every query path (search, symbols, references, ...), which stay
    in ``Application`` per D6, so it cannot move here without either creating
    an import cycle (``application.py`` already imports this module for
    ``MaintenanceService``) or promoting a stateless eight-line lookup to a
    dependency callable. Duplicating it is the smaller cost.
    """
    checkouts = targets.get(project_id) or ()
    if not checkouts:
        raise CodeIndexingError(
            ErrorCode.PROJECT_NOT_FOUND,
            f"No active index slot was resolved for project {project_id}",
        )
    return checkouts[0]


class _StableQueryRunner(Protocol):
    """Type of ``Application._run_repository_stable_query``.

    A plain ``Callable[[...], _Result]`` field on a non-generic dataclass binds
    ``_Result`` to one concrete type at construction time; this method needs a
    fresh type variable on every call (``storage_status`` returns
    ``StorageStatus``, a caller elsewhere returns something else), which only a
    generic ``Protocol.__call__`` expresses correctly.
    """

    def __call__(
        self,
        projects: Sequence[ProjectInfo],
        operation: Callable[[Mapping[str, Sequence[ActiveIndexTarget]]], _Result],
    ) -> _Result: ...


@dataclass(frozen=True)
class _Dependencies:
    """The Application-provided callables MaintenanceService cannot compute itself."""

    list_projects: Callable[[], list[ProjectInfo]]
    resolve_project: Callable[[str | None, list[Path] | None], ProjectInfo]
    resolve_active_target: Callable[[ProjectInfo, bool], ActiveIndexTarget]
    resolve_active_targets: Callable[
        [Sequence[ProjectInfo], bool], Mapping[str, Sequence[ActiveIndexTarget]]
    ]
    run_repository_stable_query: _StableQueryRunner


@dataclass(frozen=True)
class _RegistryMaintenanceResult:
    before: tuple[TableStorageStats, ...]
    after: tuple[TableStorageStats, ...]
    status: str
    skip_reason: str | None
    error: str | None
    versions_removed: int
    bytes_reclaimed: int


class MaintenanceService:
    """Compact tables, evict retained versions, and run the scheduled pass."""

    def __init__(
        self,
        *,
        store: LanceStore,
        paths: RuntimePaths,
        settings: IndexSettings,
        list_projects: Callable[[], list[ProjectInfo]],
        resolve_project: Callable[[str | None, list[Path] | None], ProjectInfo],
        resolve_active_target: Callable[[ProjectInfo, bool], ActiveIndexTarget],
        resolve_active_targets: Callable[
            [Sequence[ProjectInfo], bool], Mapping[str, Sequence[ActiveIndexTarget]]
        ],
        run_repository_stable_query: _StableQueryRunner,
    ) -> None:
        self.store = store
        self.paths = paths
        self.settings = settings
        self._deps = _Dependencies(
            list_projects=list_projects,
            resolve_project=resolve_project,
            resolve_active_target=resolve_active_target,
            resolve_active_targets=resolve_active_targets,
            run_repository_stable_query=run_repository_stable_query,
        )

    def storage_status(
        self, project: str | None = None, *, roots: list[Path] | None = None
    ) -> StorageStatus:
        """Read-only storage statistics for one project or the whole installation.

        Resolving the current checkout may create or activate an empty pending
        slot, but reads never materialize its physical partition. Root-overlap
        and shared-Git worktree warnings are advisory and best-effort.
        """
        registered = self._deps.list_projects()
        if project is not None:
            resolved = self._deps.resolve_project(project, roots)
            scope = [resolved]
        else:
            scope = registered

        def collect(targets: Mapping[str, Sequence[ActiveIndexTarget]]) -> StorageStatus:
            snapshot_at = datetime.now(UTC).isoformat()
            registry_before = tuple(self.store.registry_table_stats())
            project_stats = [
                self.store.storage_stats_for(
                    registered_project,
                    partition_ref=_primary_target(targets, registered_project.id).partition,
                )
                for registered_project in scope
            ]
            registry_after = tuple(self.store.registry_table_stats())
            partition_bytes: dict[str, int] = {}
            for stats in project_stats:
                if stats.slots:
                    partition_bytes.update(
                        {slot.partition_id: slot.physical_bytes for slot in stats.slots}
                    )
                else:
                    partition_bytes[stats.project.id] = stats.partition_physical_bytes
            return StorageStatus(
                snapshot_at=snapshot_at,
                registry=registry_after[0],
                projects=project_stats,
                physical_bytes_total=registry_after[0].physical_bytes
                + sum(partition_bytes.values()),
                consistent=all(
                    before.current_version == after.current_version
                    for before, after in zip(registry_before, registry_after, strict=True)
                )
                and all(stats.consistent for stats in project_stats),
                overlap_warnings=overlap_warnings(registered),
                worktree_warnings=worktree_warnings(registered),
            )

        return self._deps.run_repository_stable_query(scope, collect)

    @staticmethod
    def _acquire_lock(lock: FileLock, *, wait_for_lock: bool) -> bool:
        if wait_for_lock:
            lock.acquire()
            return True
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return False
        return True

    def _dry_run_project(
        self, project: ProjectInfo, target: ActiveIndexTarget
    ) -> MaintenanceProjectResult:
        try:
            before = self.store.storage_stats_for(project, partition_ref=target.partition)
            estimate = _estimate_reclaimable(before)
            if before.partition_open_failed:
                return MaintenanceProjectResult(
                    project=project,
                    before=before,
                    status="error",
                    error="Partition exists but its tables could not be opened",
                    reclaimable_bytes_estimate=estimate,
                )
            return MaintenanceProjectResult(
                project=project,
                before=before,
                skip_reason="not-indexed" if not before.tables else "dry-run",
                reclaimable_bytes_estimate=estimate,
            )
        except Exception as exc:
            return MaintenanceProjectResult(
                project=project,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _maintain_project(
        self,
        project: ProjectInfo,
        *,
        lock_directory: Path,
        retention: timedelta,
        branch_cache_limit: int,
        wait_for_lock: bool,
    ) -> MaintenanceProjectResult:
        """Maintain one project while preserving the global-then-project lock order."""
        global_lock = FileLock(lock_directory / "index-global.lock")
        project_lock = FileLock(lock_directory / f"{project.id}.lock")
        global_acquired = False
        project_acquired = False
        before: ProjectStorageStats | None = None
        estimate = 0
        try:
            global_acquired = self._acquire_lock(global_lock, wait_for_lock=wait_for_lock)
            if not global_acquired:
                return MaintenanceProjectResult(project=project, skip_reason="busy")
            project_acquired = self._acquire_lock(project_lock, wait_for_lock=wait_for_lock)
            if not project_acquired:
                return MaintenanceProjectResult(project=project, skip_reason="busy")
            target = self._deps.resolve_active_target(project, True)
            recovery = pending_recovery(self.paths.data / "staging", project.id)
            if recovery.project_wide:
                return MaintenanceProjectResult(project=project, skip_reason="recovery-pending")
            before = self.store.storage_stats_for(project, partition_ref=target.partition)
            estimate = _estimate_reclaimable(before)
            if before.partition_open_failed:
                return MaintenanceProjectResult(
                    project=project,
                    before=before,
                    status="error",
                    error="Partition exists but its tables could not be opened",
                    reclaimable_bytes_estimate=estimate,
                )
            if not before.tables and not any(slot.physical_bytes > 0 for slot in before.slots):
                return MaintenanceProjectResult(
                    project=project,
                    before=before,
                    skip_reason="not-indexed",
                    reclaimable_bytes_estimate=estimate,
                )
            self.store.maintain_project(
                project.id,
                cleanup_older_than=retention,
                branch_cache_limit=branch_cache_limit,
                protected_slot_ids=recovery.slot_ids,
            )
            after = self.store.storage_stats_for(project, partition_ref=target.partition)
            if after.partition_open_failed:
                raise RuntimeError("Partition became unreadable during maintenance")
            return MaintenanceProjectResult(
                project=project,
                before=before,
                after=after,
                status="ok",
                versions_removed=_versions_removed(before, after),
                bytes_reclaimed=max(
                    0, before.partition_physical_bytes - after.partition_physical_bytes
                ),
                reclaimable_bytes_estimate=estimate,
            )
        except Exception as exc:
            return MaintenanceProjectResult(
                project=project,
                before=before,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                reclaimable_bytes_estimate=estimate,
            )
        finally:
            if project_acquired:
                project_lock.release()
            if global_acquired:
                global_lock.release()

    def _maintain_registry(
        self,
        *,
        lock_directory: Path,
        retention: timedelta,
        wait_for_lock: bool,
        dry_run: bool,
    ) -> _RegistryMaintenanceResult:
        if dry_run:
            return _RegistryMaintenanceResult(
                before=tuple(self.store.registry_table_stats()),
                after=(),
                status="skipped",
                skip_reason="dry-run",
                error=None,
                versions_removed=0,
                bytes_reclaimed=0,
            )
        lock = FileLock(lock_directory / "index-global.lock")
        acquired = False
        try:
            acquired = self._acquire_lock(lock, wait_for_lock=wait_for_lock)
            if not acquired:
                return _RegistryMaintenanceResult(
                    before=(),
                    after=(),
                    status="skipped",
                    skip_reason="busy",
                    error=None,
                    versions_removed=0,
                    bytes_reclaimed=0,
                )
            before = tuple(self.store.registry_table_stats())
            self.store.maintain_registry(cleanup_older_than=retention)
            after = tuple(self.store.registry_table_stats())
            return _RegistryMaintenanceResult(
                before=before,
                after=after,
                status="ok",
                skip_reason=None,
                error=None,
                versions_removed=sum(
                    max(0, previous.retained_version_count - current.retained_version_count)
                    for previous, current in zip(before, after, strict=True)
                ),
                bytes_reclaimed=sum(
                    max(0, previous.physical_bytes - current.physical_bytes)
                    for previous, current in zip(before, after, strict=True)
                ),
            )
        except Exception as exc:
            return _RegistryMaintenanceResult(
                before=(),
                after=(),
                status="error",
                skip_reason=None,
                error=f"{type(exc).__name__}: {exc}",
                versions_removed=0,
                bytes_reclaimed=0,
            )
        finally:
            if acquired:
                lock.release()

    def maintain_storage(
        self,
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        dry_run: bool = False,
        wait_for_lock: bool = False,
        trigger: str = "manual",
    ) -> MaintenanceReport:
        """Compact tables and remove verified versions older than the retention window.

        Runs under the same global and per-project writer locks indexing uses,
        so a pass never races a commit. Automatic passes attempt the locks
        without waiting and record busy projects as skipped; manual passes wait
        them out. A dry run collects the before statistics, a reclaimable-bytes
        estimate, and no after statistics (``after`` stays null) but mutates
        nothing and takes no locks.

        ``trigger`` labels the pass as ``manual`` or ``scheduled`` for audit
        purposes. The registry is maintained once under the global lock; when an
        automatic pass cannot get that lock, the registry is reported without
        maintenance rather than failing the whole run.
        """
        # Retention reads last_used_at for LRU eviction; a buffered touch that
        # never made it to disk must not make a recently used slot look old
        # enough to evict. A dry run reads the in-memory overlay and must not
        # turn inspection into a registry write.
        if not dry_run:
            self.store.flush_slot_touches()
        started = time.monotonic_ns()
        started_at = datetime.now(UTC).isoformat()
        retention = timedelta(hours=self.settings.version_retention_hours)
        registered = self._deps.list_projects()
        if project is not None:
            resolved = self._deps.resolve_project(project, roots)
            scope: list[ProjectInfo] = [resolved]
        else:
            scope = registered
        targets = self._deps.resolve_active_targets(scope, False) if dry_run else {}
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)

        results: list[MaintenanceProjectResult] = []
        for registered_project in scope:
            if dry_run:
                target = _primary_target(targets, registered_project.id)
                results.append(self._dry_run_project(registered_project, target))
            else:
                results.append(
                    self._maintain_project(
                        registered_project,
                        lock_directory=lock_directory,
                        retention=retention,
                        branch_cache_limit=self.settings.branch_cache_limit,
                        wait_for_lock=wait_for_lock,
                    )
                )

        registry_result = self._maintain_registry(
            lock_directory=lock_directory,
            retention=retention,
            wait_for_lock=wait_for_lock,
            dry_run=dry_run,
        )

        finished_at = datetime.now(UTC).isoformat()
        duration_ms = (time.monotonic_ns() - started) // 1_000_000
        skipped = [
            result.project.id
            for result in results
            if result.status == "skipped" and result.skip_reason != "busy"
        ]
        busy = [
            result.project.id
            for result in results
            if result.status == "skipped" and result.skip_reason == "busy"
        ]
        failed = [result.project.id for result in results if result.status == "error"]
        return MaintenanceReport(
            trigger=trigger,
            dry_run=dry_run,
            retention_hours=self.settings.version_retention_hours,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            projects=results,
            registry_before=registry_result.before[0] if registry_result.before else None,
            registry_after=registry_result.after[0] if registry_result.after else None,
            registry_tables_before=list(registry_result.before),
            registry_tables_after=list(registry_result.after),
            registry_status=registry_result.status,
            registry_skip_reason=registry_result.skip_reason,
            registry_error=registry_result.error,
            registry_versions_removed=registry_result.versions_removed,
            registry_bytes_reclaimed=registry_result.bytes_reclaimed,
            versions_removed_total=sum(result.versions_removed for result in results),
            bytes_reclaimed_total=sum(result.bytes_reclaimed for result in results),
            reclaimable_bytes_estimate_total=sum(
                result.reclaimable_bytes_estimate for result in results
            ),
            skipped_projects=skipped,
            busy_projects=busy,
            failed_projects=failed,
        )

    def maybe_run_maintenance(self) -> MaintenanceReport | None:
        """Run scheduled maintenance when it is due, persisting the last-run stamp.

        Gated by ``CODE_INDEXING_AUTO_MAINTENANCE`` and checked at most once per
        24 hours, using the timestamp of the last complete pass. Runs in any
        indexing mode: maintenance never scans source files or creates a new
        logical generation, so lazy, eager, and manual modes are all eligible.
        Busy, damaged, or recovery-dependent projects leave the stamp stale so
        a later startup retries them instead of waiting another 24 hours.
        """
        if not self.settings.auto_maintenance:
            return None
        timestamp_path = self.paths.data / MAINTENANCE_TIMESTAMP_FILE
        lock_directory = self.paths.data / "locks"
        lock_directory.mkdir(parents=True, exist_ok=True)
        schedule_lock = FileLock(lock_directory / MAINTENANCE_LOCK_FILE)
        try:
            schedule_lock.acquire(timeout=0)
        except Timeout:
            return None
        try:
            # Re-check after acquiring the cross-process lock: another startup
            # may have completed the overdue pass while this process waited to run.
            last = _read_maintenance_timestamp(timestamp_path)
            if last is not None and datetime.now(UTC) - last < MAINTENANCE_CHECK_INTERVAL:
                return None
            report = self.maintain_storage(dry_run=False, wait_for_lock=False, trigger="scheduled")
            projects_complete = all(
                result.status == "ok"
                or (result.status == "skipped" and result.skip_reason == "not-indexed")
                for result in report.projects
            )
            if report.registry_status == "ok" and projects_complete:
                _write_maintenance_timestamp(timestamp_path)
            return report
        finally:
            schedule_lock.release()
