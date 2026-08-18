/** Durable, bounded SQLite audit history for indexing and backfill runs. */

import fs from "node:fs";
import path from "node:path";
import {
  HistoryPage,
  type ProjectInfo,
  RunAudit,
  RunSummary,
  type RunSummary as RunSummaryType,
} from "./models.ts";
import { pythonJsonDumps } from "./python-compat.ts";
import {
  openSQLite,
  type SQLiteDatabase,
  type SQLiteRow,
  type SQLiteValue,
} from "./runtime/sqlite.ts";

export const MAX_RUNS_PER_PROJECT = 100;
export const MAX_RUN_AGE_DAYS = 30;
export const MAX_ERROR_SAMPLES = 20;
export const MAX_SKIPPED_SAMPLES = 20;
export const BUSY_TIMEOUT_SECONDS = 30;

const SCHEMA = `
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
`;

const FINISH_COLUMNS = [
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
] as const;

const FINISH_COLUMN_SET = new Set<string>(FINISH_COLUMNS);
const JSON_COLUMNS = new Set<string>([
  "phase_durations",
  "skip_reasons",
  "errors",
  "skipped_samples",
  "storage_before",
  "storage_after",
]);
const INTEGER_COLUMNS = new Set<string>([
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
]);

type FinishColumn = (typeof FINISH_COLUMNS)[number];
export type FinishUpdates = Partial<Pick<RunAudit, FinishColumn>>;

/** SQLite-backed, bounded audit history for indexing runs. */
export class HistoryStore {
  readonly directory: string;
  readonly path: string;
  readonly #database: SQLiteDatabase;

  constructor(directory: string) {
    this.directory = directory;
    this.path = path.join(directory, "runs.sqlite");
    fs.mkdirSync(directory, { recursive: true });
    this.#database = openSQLite(this.path);
    this.#database.exec(
      `PRAGMA journal_mode = WAL; PRAGMA busy_timeout = ${BUSY_TIMEOUT_SECONDS * 1000};`,
    );
    this.#database.exec(SCHEMA);
    this.#migrate();
  }

  close(): void {
    this.#database.close();
  }

  /** Insert a run in the running state before its work starts. */
  begin(audit: RunAudit): void {
    const record = RunAudit.parse(audit);
    this.#database
      .query(
        `INSERT OR REPLACE INTO runs (
          run_id, project_id, trigger, server_version, git_revision,
          model_id, schema_version, scan_config_hash, force, rebuild_reason,
          pid, started_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        record.run_id,
        record.project_id,
        record.trigger,
        record.server_version,
        record.git_revision,
        record.model_id,
        record.schema_version,
        record.scan_config_hash,
        Number(record.force),
        record.rebuild_reason,
        record.pid,
        record.started_at,
      );
  }

  /** Complete or fail a run, pruning its project and expired history atomically. */
  finish(runId: string, updates: FinishUpdates): void {
    const entries = Object.entries(updates);
    if (entries.length === 0) throw new Error("finish requires at least one field to record");
    const unknown = entries
      .map(([column]) => column)
      .filter((column) => !FINISH_COLUMN_SET.has(column));
    if (unknown.length > 0)
      throw new Error(`unknown audit finish fields: ${unknown.sort().join(", ")}`);

    const assignments: string[] = [];
    const values: SQLiteValue[] = [];
    for (const [column, value] of entries.sort(([left], [right]) => left.localeCompare(right))) {
      assignments.push(`${column} = ?`);
      values.push(this.#finishValue(column, value));
    }

    this.#transaction(() => {
      this.#database
        .query(`UPDATE runs SET ${assignments.join(", ")} WHERE run_id = ?`)
        .run(...values, runId);
      this.#database
        .query(
          `DELETE FROM runs WHERE project_id = (
            SELECT project_id FROM runs WHERE run_id = ?
          ) AND rowid NOT IN (
            SELECT rowid FROM runs WHERE project_id = (
              SELECT project_id FROM runs WHERE run_id = ?
            ) ORDER BY started_at DESC, rowid DESC LIMIT ?
          )`,
        )
        .run(runId, runId, MAX_RUNS_PER_PROJECT);
      this.#database
        .query("DELETE FROM runs WHERE started_at < ?")
        .run(isoTimestamp(Date.now() - MAX_RUN_AGE_DAYS * 24 * 60 * 60 * 1000));
    });
  }

  /** Mark running rows whose owning process is gone as interrupted. */
  markInterrupted(): void {
    const interrupted = this.#database
      .query("SELECT run_id, pid FROM runs WHERE state = 'running'")
      .all()
      .filter((row) => !pidExists(numberValue(row.pid)))
      .map((row) => stringValue(row.run_id));
    if (interrupted.length === 0) return;
    this.#database
      .query(
        `UPDATE runs SET state = 'interrupted' WHERE run_id IN (${interrupted.map(() => "?").join(", ")})`,
      )
      .run(...interrupted);
  }

  /** Return runs for a project, newest first, with an opaque cursor. */
  listRuns(
    projectId: string,
    {
      cursor = null,
      limit = 20,
      project = null,
    }: { cursor?: string | null; limit?: number; project?: ProjectInfo | null } = {},
  ): HistoryPage {
    if (limit < 1) throw new Error("limit must be positive");
    const conditions = ["project_id = ?"];
    const parameters: SQLiteValue[] = [projectId];
    if (cursor !== null) {
      const split = cursor.indexOf("|");
      if (split <= 0 || split === cursor.length - 1) throw new Error("invalid history cursor");
      const startedAt = cursor.slice(0, split);
      const runId = cursor.slice(split + 1);
      conditions.push("(started_at, rowid) < (?, (SELECT rowid FROM runs WHERE run_id = ?))");
      parameters.push(startedAt, runId);
    }
    const rows = this.#database
      .query(
        `SELECT * FROM runs WHERE ${conditions.join(" AND ")} ORDER BY started_at DESC, rowid DESC LIMIT ?`,
      )
      .all(...parameters, Math.min(limit, MAX_RUNS_PER_PROJECT) + 1);
    const audits = rows.slice(0, Math.min(limit, MAX_RUNS_PER_PROJECT)).map(rowToAudit);
    return HistoryPage.parse({
      project,
      runs: audits,
      next_cursor:
        rows.length > Math.min(limit, MAX_RUNS_PER_PROJECT) && audits.length > 0
          ? `${audits.at(-1)?.started_at}|${audits.at(-1)?.run_id}`
          : null,
    });
  }

  /** Return the most recent finished run's compact status summary. */
  recent(projectId: string): RunSummaryType | null {
    const row = this.#database
      .query(
        "SELECT * FROM runs WHERE project_id = ? AND finished_at IS NOT NULL ORDER BY started_at DESC, rowid DESC LIMIT 1",
      )
      .get(projectId);
    return row === null ? null : rowToSummary(row);
  }

  #migrate(): void {
    const columns = new Set(
      this.#database
        .query("PRAGMA table_info(runs)")
        .all()
        .map((row) => stringValue(row.name)),
    );
    if (!columns.has("pid"))
      this.#database.exec("ALTER TABLE runs ADD COLUMN pid INTEGER NOT NULL DEFAULT 0");
    if (!columns.has("rebuild_reason"))
      this.#database.exec("ALTER TABLE runs ADD COLUMN rebuild_reason TEXT");
  }

  #finishValue(column: string, value: unknown): SQLiteValue {
    if (JSON_COLUMNS.has(column)) return pythonJsonDumps(value);
    if (INTEGER_COLUMNS.has(column)) return Number(value);
    if (
      value === null ||
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      return value;
    }
    throw new TypeError(`invalid value for audit finish field ${column}`);
  }

  #transaction(operation: () => void): void {
    this.#database.exec("BEGIN");
    try {
      operation();
      this.#database.exec("COMMIT");
    } catch (error) {
      this.#database.exec("ROLLBACK");
      throw error;
    }
  }
}

function rowToAudit(row: SQLiteRow): RunAudit {
  return RunAudit.parse({
    run_id: stringValue(row.run_id),
    project_id: stringValue(row.project_id),
    trigger: stringValue(row.trigger),
    server_version: stringValue(row.server_version),
    git_revision: nullableStringValue(row.git_revision),
    model_id: stringValue(row.model_id),
    schema_version: numberValue(row.schema_version),
    scan_config_hash: stringValue(row.scan_config_hash),
    force: Boolean(row.force),
    rebuild_reason: nullableStringValue(row.rebuild_reason),
    pid: numberValue(row.pid),
    started_at: stringValue(row.started_at),
    finished_at: nullableStringValue(row.finished_at),
    state: stringValue(row.state),
    phase_durations: jsonValue(row.phase_durations),
    eligible_files: numberValue(row.eligible_files),
    changed_files: numberValue(row.changed_files),
    unchanged_files: numberValue(row.unchanged_files),
    parsed_files: numberValue(row.parsed_files),
    failed_files: numberValue(row.failed_files),
    removed_files: numberValue(row.removed_files),
    skipped_total: numberValue(row.skipped_total),
    chunks_extracted: numberValue(row.chunks_extracted),
    chunks_embedded: numberValue(row.chunks_embedded),
    chunks_staged: numberValue(row.chunks_staged),
    staged_bytes: numberValue(row.staged_bytes),
    bytes_read: numberValue(row.bytes_read),
    skip_reasons: jsonValue(row.skip_reasons),
    errors: jsonValue(row.errors),
    skipped_samples: jsonValue(row.skipped_samples),
    embedding_backend: stringValue(row.embedding_backend),
    embedding_fallback_reason: nullableStringValue(row.embedding_fallback_reason),
    worker_used: Boolean(row.worker_used),
    storage_before: jsonValue(row.storage_before),
    storage_after: jsonValue(row.storage_after),
  });
}

function rowToSummary(row: SQLiteRow): RunSummaryType {
  const startedAt = stringValue(row.started_at);
  const finishedAt = nullableStringValue(row.finished_at);
  const started = Date.parse(startedAt);
  const finished = finishedAt === null ? Number.NaN : Date.parse(finishedAt);
  return RunSummary.parse({
    run_id: stringValue(row.run_id),
    trigger: stringValue(row.trigger),
    state: stringValue(row.state),
    started_at: startedAt,
    finished_at: finishedAt,
    duration_ms:
      Number.isNaN(started) || Number.isNaN(finished) ? 0 : Math.trunc(finished - started),
    eligible_files: numberValue(row.eligible_files),
    changed_files: numberValue(row.changed_files),
    failed_files: numberValue(row.failed_files),
    skipped_total: numberValue(row.skipped_total),
    chunks_embedded: numberValue(row.chunks_embedded),
  });
}

function jsonValue(value: SQLiteValue | undefined): unknown {
  return JSON.parse(stringValue(value));
}

function stringValue(value: SQLiteValue | undefined): string {
  if (typeof value !== "string") throw new TypeError("expected SQLite text value");
  return value;
}

function nullableStringValue(value: SQLiteValue | undefined): string | null {
  return value === null || value === undefined ? null : stringValue(value);
}

function numberValue(value: SQLiteValue | undefined): number {
  if (typeof value !== "number") throw new TypeError("expected SQLite numeric value");
  return value;
}

function pidExists(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

function isoTimestamp(milliseconds: number): string {
  return new Date(milliseconds).toISOString().replace("Z", "+00:00");
}
