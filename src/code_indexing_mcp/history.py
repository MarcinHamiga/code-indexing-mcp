"""Durable audit history for indexing runs, in a small SQLite WAL database.

The progress snapshot is deliberately ephemeral -- it describes the run that is
happening right now and is deleted when that run ends. Completed work is stored
here instead: one row per indexing or backfill run, bounded before it is
written (error details and skipped-path samples are capped) and pruned to a
fixed per-project window, so a machine that indexes for years never accumulates
an unbounded history.

SQLite runs in WAL mode with a busy timeout because the per-user daemon and
direct-mode servers can open the same database from different processes; WAL
keeps a reader from ever blocking a writer, and the busy timeout covers the
rare simultaneous writer collision.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from .models import HistoryPage, IndexIssue, ProjectInfo, RunAudit, RunSummary

# Bounds that keep history small and bounded on any workload. A repository can
# be reindexed hundreds of times a day; 100 runs per project plus a hard age
# ceiling is enough audit depth for the foreseeable future.
MAX_RUNS_PER_PROJECT = 100
MAX_RUN_AGE_DAYS = 30
MAX_ERROR_SAMPLES = 20
MAX_SKIPPED_SAMPLES = 20
BUSY_TIMEOUT_SECONDS = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    server_version TEXT NOT NULL DEFAULT '',
    git_revision TEXT,
    model_id TEXT NOT NULL DEFAULT '',
    schema_version INTEGER NOT NULL DEFAULT 0,
    scan_config_hash TEXT NOT NULL DEFAULT '',
    force INTEGER NOT NULL DEFAULT 0,
    rebuild_reason TEXT,
    pid INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    state TEXT NOT NULL DEFAULT 'running',
    phase_durations TEXT NOT NULL DEFAULT '{}',
    eligible_files INTEGER NOT NULL DEFAULT 0,
    changed_files INTEGER NOT NULL DEFAULT 0,
    unchanged_files INTEGER NOT NULL DEFAULT 0,
    parsed_files INTEGER NOT NULL DEFAULT 0,
    failed_files INTEGER NOT NULL DEFAULT 0,
    removed_files INTEGER NOT NULL DEFAULT 0,
    skipped_total INTEGER NOT NULL DEFAULT 0,
    chunks_extracted INTEGER NOT NULL DEFAULT 0,
    chunks_embedded INTEGER NOT NULL DEFAULT 0,
    chunks_staged INTEGER NOT NULL DEFAULT 0,
    staged_bytes INTEGER NOT NULL DEFAULT 0,
    bytes_read INTEGER NOT NULL DEFAULT 0,
    skip_reasons TEXT NOT NULL DEFAULT '{}',
    errors TEXT NOT NULL DEFAULT '[]',
    skipped_samples TEXT NOT NULL DEFAULT '[]',
    embedding_backend TEXT NOT NULL DEFAULT 'cpu',
    embedding_fallback_reason TEXT,
    worker_used INTEGER NOT NULL DEFAULT 0,
    storage_before TEXT NOT NULL DEFAULT '{}',
    storage_after TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runs_project_started ON runs(project_id, started_at DESC);
"""

# Columns updated by HistoryStore.finish; anything else is rejected so a
# recording bug fails loudly instead of silently widening the schema.
_FINISH_COLUMNS = frozenset(
    {
        "finished_at",
        "state",
        "phase_durations",
        "eligible_files",
        "changed_files",
        "unchanged_files",
        "parsed_files",
        "failed_files",
        "removed_files",
        "skipped_total",
        "chunks_extracted",
        "chunks_embedded",
        "chunks_staged",
        "staged_bytes",
        "bytes_read",
        "skip_reasons",
        "errors",
        "skipped_samples",
        "embedding_backend",
        "embedding_fallback_reason",
        "worker_used",
        "storage_before",
        "storage_after",
    }
)

_JSON_COLUMNS = frozenset(
    {
        "phase_durations",
        "skip_reasons",
        "errors",
        "skipped_samples",
        "storage_before",
        "storage_after",
    }
)

_SCALAR_INT_COLUMNS = frozenset(
    {
        "eligible_files",
        "changed_files",
        "unchanged_files",
        "parsed_files",
        "failed_files",
        "removed_files",
        "skipped_total",
        "chunks_extracted",
        "chunks_embedded",
        "chunks_staged",
        "staged_bytes",
        "bytes_read",
        "worker_used",
    }
)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


class HistoryStore:
    """SQLite-backed, bounded audit history for indexing runs."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "runs.sqlite"
        directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            # Databases created before pid tracking have no owner column;
            # ALTER is cheap and idempotent because of the guard above.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
            if "pid" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN pid INTEGER NOT NULL DEFAULT 0")
            if "rebuild_reason" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN rebuild_reason TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_SECONDS * 1000)}")
        return connection

    def begin(self, audit: RunAudit) -> None:
        """Insert a run in the ``running`` state before its work starts."""

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, project_id, trigger, server_version, git_revision,
                    model_id, schema_version, scan_config_hash, force, rebuild_reason,
                    pid, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.run_id,
                    audit.project_id,
                    audit.trigger,
                    audit.server_version,
                    audit.git_revision,
                    audit.model_id,
                    audit.schema_version,
                    audit.scan_config_hash,
                    int(audit.force),
                    audit.rebuild_reason,
                    int(audit.pid),
                    audit.started_at,
                ),
            )

    def finish(self, run_id: str, **updates: Any) -> None:
        """Complete or fail a run, pruning history in the same transaction.

        Only the allowlisted columns may be updated; dicts and lists are
        serialized to JSON, and scalar counts are coerced to integers.
        """

        if not updates:
            raise ValueError("finish requires at least one field to record")
        unknown = set(updates) - _FINISH_COLUMNS
        if unknown:
            raise ValueError(f"unknown audit finish fields: {sorted(unknown)}")
        values: list[Any] = []
        assignments: list[str] = []
        for column in sorted(updates):
            value = updates[column]
            if column in _JSON_COLUMNS:
                if column == "errors":
                    value = [item.model_dump(mode="json") for item in value]
                value = json.dumps(value, sort_keys=True)
            elif column in _SCALAR_INT_COLUMNS:
                value = int(value)
            assignments.append(f"{column} = ?")
            values.append(value)
        values.append(run_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE run_id = ?", values)
            connection.execute(
                """
                DELETE FROM runs WHERE project_id = (
                    SELECT project_id FROM runs WHERE run_id = ?
                ) AND rowid NOT IN (
                    SELECT rowid FROM runs WHERE project_id = (
                        SELECT project_id FROM runs WHERE run_id = ?
                    )
                    ORDER BY started_at DESC, rowid DESC LIMIT ?
                )
                """,
                (run_id, run_id, MAX_RUNS_PER_PROJECT),
            )
            connection.execute(
                "DELETE FROM runs WHERE started_at < ?",
                ((datetime.now(UTC) - timedelta(days=MAX_RUN_AGE_DAYS)).isoformat(),),
            )

    def mark_interrupted(self) -> None:
        """Mark every ``running`` row of a dead process interrupted.

        Called once at process start. A process that died mid-run leaves its
        row in ``running`` forever otherwise. The rows' owning pids separate
        those crashed runs from runs another live process is executing right
        now, which must not be touched; a pid of 0 predates pid tracking and
        cannot be verified, so it is treated as dead.
        """

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, pid FROM runs WHERE state = 'running'"
            ).fetchall()
            interrupted = [
                row["run_id"]
                for row in rows
                if row["pid"] == 0 or not psutil.pid_exists(row["pid"])
            ]
            if interrupted:
                placeholders = ", ".join("?" for _ in interrupted)
                connection.execute(
                    f"UPDATE runs SET state = 'interrupted' WHERE run_id IN ({placeholders})",
                    interrupted,
                )

    def list_runs(
        self,
        project_id: str,
        *,
        cursor: str | None = None,
        limit: int = 20,
        project: ProjectInfo | None = None,
    ) -> HistoryPage:
        """Return runs for *project_id*, newest first, with an opaque cursor."""

        if limit < 1:
            raise ValueError("limit must be positive")
        limit = min(limit, MAX_RUNS_PER_PROJECT)
        conditions = ["project_id = ?"]
        parameters: list[Any] = [project_id]
        if cursor is not None:
            parts = cursor.split("|", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError("invalid history cursor")
            started_at, run_id = parts
            conditions.append(
                "(started_at, rowid) < (?, (SELECT rowid FROM runs WHERE run_id = ?))"
            )
            parameters.extend([started_at, run_id])
        with closing(self._connect()) as connection:
            rows = list(
                connection.execute(
                    f"""
                    SELECT * FROM runs WHERE {" AND ".join(conditions)}
                    ORDER BY started_at DESC, rowid DESC LIMIT ?
                    """,
                    [*parameters, limit + 1],
                ).fetchall()
            )
        audits = [_row_to_audit(row) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit and audits:
            latest = audits[-1]
            next_cursor = f"{latest.started_at}|{latest.run_id}"
        return HistoryPage(project=project, runs=audits, next_cursor=next_cursor)

    def recent(self, project_id: str) -> RunSummary | None:
        """Return the most recent finished run for *project_id*, or None.

        One row, one index probe -- deliberately cheap enough for project
        status, which runs on every freshness check. Rows without a
        ``finished_at`` are excluded: an in-flight run is covered by the live
        progress snapshot, and a crashed run's ``interrupted`` stub carries
        only zeroed counters -- surfacing either would hide the last completed
        run's summary that ``last_run`` is documented to be.
        """

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM runs WHERE project_id = ? AND finished_at IS NOT NULL
                ORDER BY started_at DESC, rowid DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_summary(row)

    def close(self) -> None:
        """No-op for symmetry; connections are per-operation."""


def _row_to_audit(row: sqlite3.Row) -> RunAudit:
    return RunAudit(
        run_id=row["run_id"],
        project_id=row["project_id"],
        trigger=row["trigger"],
        server_version=row["server_version"],
        git_revision=row["git_revision"],
        model_id=row["model_id"],
        schema_version=row["schema_version"],
        scan_config_hash=row["scan_config_hash"],
        force=bool(row["force"]),
        rebuild_reason=row["rebuild_reason"],
        pid=row["pid"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        state=row["state"],
        phase_durations=json.loads(row["phase_durations"]),
        eligible_files=row["eligible_files"],
        changed_files=row["changed_files"],
        unchanged_files=row["unchanged_files"],
        parsed_files=row["parsed_files"],
        failed_files=row["failed_files"],
        removed_files=row["removed_files"],
        skipped_total=row["skipped_total"],
        chunks_extracted=row["chunks_extracted"],
        chunks_embedded=row["chunks_embedded"],
        chunks_staged=row["chunks_staged"],
        staged_bytes=row["staged_bytes"],
        bytes_read=row["bytes_read"],
        skip_reasons=json.loads(row["skip_reasons"]),
        errors=[IndexIssue.model_validate(item) for item in json.loads(row["errors"])],
        skipped_samples=json.loads(row["skipped_samples"]),
        embedding_backend=row["embedding_backend"],
        embedding_fallback_reason=row["embedding_fallback_reason"],
        worker_used=bool(row["worker_used"]),
        storage_before=json.loads(row["storage_before"]),
        storage_after=json.loads(row["storage_after"]),
    )


def _row_to_summary(row: sqlite3.Row) -> RunSummary:
    started_at = row["started_at"]
    finished_at = row["finished_at"]
    duration_ms = 0
    if finished_at is not None:
        try:
            start = datetime.fromisoformat(started_at)
            finish = datetime.fromisoformat(finished_at)
            duration_ms = int((finish - start).total_seconds() * 1000)
        except ValueError:
            duration_ms = 0
    return RunSummary(
        run_id=row["run_id"],
        trigger=row["trigger"],
        state=row["state"],
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        eligible_files=row["eligible_files"],
        changed_files=row["changed_files"],
        failed_files=row["failed_files"],
        skipped_total=row["skipped_total"],
        chunks_embedded=row["chunks_embedded"],
    )
