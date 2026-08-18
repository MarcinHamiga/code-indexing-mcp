/** Durable audit history, ported from `tests/test_history.py`. */

import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import {
  HistoryStore,
  MAX_RUN_AGE_DAYS,
  MAX_RUNS_PER_PROJECT,
  type FinishUpdates,
} from "../src/history.ts";
import { RunAudit, type RunAudit as RunAuditType } from "../src/models.ts";
import { openSQLite } from "../src/runtime/sqlite.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

const directories: string[] = [];

afterEach(() => {
  for (const directory of directories.splice(0)) removeDirectory(directory);
});

function store(): HistoryStore {
  const directory = temporaryDirectory();
  directories.push(directory);
  return new HistoryStore(path.join(directory, "history"));
}

function audit(
  runId: string,
  startedAt = new Date().toISOString(),
  pid = process.pid,
): RunAuditType {
  return RunAudit.parse({
    run_id: runId,
    project_id: "project-1",
    trigger: "manual",
    server_version: "0.4.0",
    git_revision: "abc1234",
    model_id: "test/tiny",
    schema_version: 2,
    scan_config_hash: "feedface",
    force: true,
    pid,
    started_at: startedAt,
  });
}

function complete(history: HistoryStore, runId: string, updates: FinishUpdates = {}): void {
  history.finish(runId, {
    state: "completed",
    finished_at: new Date().toISOString(),
    ...updates,
  });
}

describe("HistoryStore", () => {
  test("round-trips an audit run with JSON fields", () => {
    const history = store();
    history.begin(audit("run-1"));
    complete(history, "run-1", {
      phase_durations: { scan: 10, embed: 40 },
      eligible_files: 3,
      changed_files: 2,
      unchanged_files: 1,
      parsed_files: 2,
      skipped_total: 5,
      chunks_extracted: 4,
      chunks_embedded: 4,
      chunks_staged: 4,
      staged_bytes: 1024,
      bytes_read: 2048,
      skip_reasons: { unsupported: 3, binary: 2 },
      errors: [{ path: "a.py", message: "boom" }],
      skipped_samples: ["b.txt", "c.bin"],
      worker_used: true,
      storage_before: { files: 1, chunks: 2, references: 3 },
      storage_after: { files: 2, chunks: 4, references: 5 },
    });

    const run = history.listRuns("project-1").runs[0];
    expect(run).toMatchObject({
      run_id: "run-1",
      state: "completed",
      force: true,
      phase_durations: { scan: 10, embed: 40 },
      skip_reasons: { unsupported: 3, binary: 2 },
      errors: [{ path: "a.py", message: "boom" }],
      worker_used: true,
      storage_after: { files: 2, chunks: 4, references: 5 },
    });
  });

  test("rejects empty and unknown finish updates", () => {
    const history = store();
    history.begin(audit("run-1"));
    expect(() => history.finish("run-1", {})).toThrow("at least one field");
    expect(() => history.finish("run-1", { made_up_column: 1 } as never)).toThrow(
      "unknown audit finish fields",
    );
  });

  test("prunes expired and excess runs, then pages newest first", () => {
    const history = store();
    const old = new Date(Date.now() - (MAX_RUN_AGE_DAYS + 1) * 24 * 60 * 60 * 1000).toISOString();
    history.begin(audit("old", old));
    complete(history, "old");
    for (let number = 0; number < MAX_RUNS_PER_PROJECT + 5; number += 1) {
      const started = new Date(Date.now() - (MAX_RUNS_PER_PROJECT - number) * 60_000).toISOString();
      const runId = `run-${number.toString().padStart(4, "0")}`;
      history.begin(audit(runId, started));
      complete(history, runId);
    }

    const first = history.listRuns("project-1", { limit: 2 });
    expect(first.runs.map((run) => run.run_id)).toEqual(["run-0104", "run-0103"]);
    expect(first.next_cursor).not.toBeNull();
    const second = history.listRuns("project-1", { limit: 2, cursor: first.next_cursor });
    expect(second.runs.map((run) => run.run_id)).toEqual(["run-0102", "run-0101"]);
    expect(history.listRuns("project-1", { limit: 1000 }).runs).toHaveLength(MAX_RUNS_PER_PROJECT);
    expect(() => history.listRuns("project-1", { cursor: "bad" })).toThrow(
      "invalid history cursor",
    );
  });

  test("marks only dead running owners interrupted and keeps recent completed work", () => {
    const history = store();
    history.begin(audit("completed", new Date(Date.now() - 2_000).toISOString()));
    complete(history, "completed", { eligible_files: 7, chunks_embedded: 9 });
    history.begin(audit("dead", new Date(Date.now() + 1_000).toISOString(), 999_999_999));
    history.begin(audit("live", new Date(Date.now() + 2_000).toISOString(), process.pid));

    history.markInterrupted();

    expect(
      Object.fromEntries(history.listRuns("project-1").runs.map((run) => [run.run_id, run.state])),
    ).toMatchObject({
      dead: "interrupted",
      live: "running",
    });
    expect(history.recent("project-1")).toMatchObject({
      run_id: "completed",
      eligible_files: 7,
      chunks_embedded: 9,
    });
  });

  test("creates the Python-compatible SQLite file in WAL mode", () => {
    const history = store();
    expect(path.basename(history.path)).toBe("runs.sqlite");
    expect(fs.existsSync(history.path)).toBe(true);
    const database = openSQLite(history.path);
    expect(database.query("PRAGMA journal_mode").get()?.journal_mode).toBe("wal");
    database.close();
  });

  test("recent skips running and interrupted stubs", () => {
    const history = store();
    expect(history.recent("project-1")).toBeNull();
    history.begin(audit("run-1", new Date(Date.now() - 2_000).toISOString()));
    complete(history, "run-1", { eligible_files: 7 });
    history.begin(audit("run-2", new Date(Date.now() + 1_000).toISOString()));
    expect(history.recent("project-1")?.run_id).toBe("run-1");
    history.begin(audit("run-3", new Date(Date.now() + 2_000).toISOString(), 999_999_999));
    history.markInterrupted();
    expect(history.recent("project-1")?.run_id).toBe("run-1");
  });

  test("concurrent writers do not corrupt history", async () => {
    const history = store();
    const writers = Array.from({ length: 4 }, async (_, number) => {
      for (let index = 0; index < 40; index += 1) {
        const runId = `w${number}-${index}`;
        history.begin(audit(runId, new Date(Date.now() + number * 1_000 + index).toISOString()));
        complete(history, runId);
      }
    });
    await Promise.all(writers);
    const page = history.listRuns("project-1", { limit: 1000 });
    expect(page.runs).toHaveLength(MAX_RUNS_PER_PROJECT);
    expect(new Set(page.runs.map((run) => run.run_id)).size).toBe(page.runs.length);
  });
});
