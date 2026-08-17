/**
 * Live progress for an indexing run, in process and across processes.
 *
 * An index can run for minutes, and the caller who has to wait it out is rarely
 * the process running it: the MCP server usually delegates to the per-user
 * daemon, and the CLI can be watching a run someone else started. So progress is
 * both handed to an in-process listener and published as a small JSON snapshot
 * under `<data>/progress/<project-id>.json` that any process can read.
 *
 * The snapshot is a status file, not a queue: every update replaces it, readers
 * only ever see the latest state, and a stale file left behind by a killed
 * process is ignored once it stops being refreshed.
 */

import fs from "node:fs";
import path from "node:path";
// IndexProgress lives in models.ts (ProjectStatus embeds it); it is re-exported
// here because this module owns everything else about progress. Nothing in this
// module may be imported by models.ts, or the import cycle this layout removed
// comes back.
import { IndexProgress, type IndexTrigger } from "./models.ts";

export { IndexProgress, type IndexTrigger };

/**
 * How long a snapshot stays trustworthy without a refresh. Comfortably above the
 * publish interval, and above the pauses a single slow file can cause between
 * updates, so a live run is never mistaken for a dead one.
 */
export const STALE_AFTER_SECONDS = 60;
export const PUBLISH_INTERVAL_SECONDS = 0.25;

export function progressPath(directory: string, projectId: string): string {
  return path.join(directory, `${projectId}.json`);
}

/** Seconds since the epoch, the unit the snapshot's timestamps are written in. */
function wallClockSeconds(): number {
  return Date.now() / 1000;
}

/** `time.monotonic()`: seconds from an arbitrary origin that never goes back. */
function monotonicSeconds(): number {
  return performance.now() / 1000;
}

export interface ProgressPublisherOptions {
  readonly runId?: string;
  readonly trigger?: IndexTrigger;
  readonly directory?: string | undefined;
  readonly listener?: ((progress: IndexProgress) => void) | undefined;
  readonly intervalSeconds?: number;
  readonly clock?: () => number;
}

/** Throttled writer for one run's snapshot file and in-process listener. */
export class ProgressPublisher {
  readonly directory: string | undefined;
  readonly listener: ((progress: IndexProgress) => void) | undefined;
  readonly intervalSeconds: number;
  state: IndexProgress;

  readonly #clock: () => number;
  #lastPublished: number | null = null;

  constructor(projectId: string, options: ProgressPublisherOptions = {}) {
    this.directory = options.directory;
    this.listener = options.listener;
    this.intervalSeconds = options.intervalSeconds ?? PUBLISH_INTERVAL_SECONDS;
    this.#clock = options.clock ?? monotonicSeconds;
    this.state = IndexProgress.parse({
      project_id: projectId,
      run_id: options.runId ?? "",
      trigger: options.trigger ?? "manual",
      started_at: wallClockSeconds(),
      phase_started_at: this.#clock(),
    });
  }

  get enabled(): boolean {
    return this.directory !== undefined || this.listener !== undefined;
  }

  /**
   * Merge *fields* into the snapshot, publishing at most once per interval.
   *
   * Publishing is never allowed to break an index: a full disk or a racing
   * reader costs an update, not the run.
   */
  update(fields: Partial<IndexProgress>, { force = false }: { force?: boolean } = {}): void {
    if (!this.enabled) return;
    const merged: Partial<IndexProgress> = { ...fields };
    if (merged.phase !== undefined && merged.phase !== this.state.phase) {
      // Anchor the new phase strictly after the previous one: clocks with coarse
      // granularity (some Windows runners read milliseconds) can return the same
      // tick twice, which would make the anchor not advance and phase durations
      // collapse to zero.
      merged.phase_started_at = Math.max(this.#clock(), this.state.phase_started_at + 1e-6);
    }
    // Deep-copy the merged fields, and replace the state object rather than
    // mutating it: a published snapshot must not share a nested value (the
    // skipped_by_reason record, say) with a caller that keeps mutating it, and a
    // listener that retains what it was handed must keep a true point-in-time
    // picture.
    this.state = { ...this.state, ...structuredClone(merged) };
    const now = this.#clock();
    if (
      !force &&
      this.#lastPublished !== null &&
      now - this.#lastPublished < this.intervalSeconds
    ) {
      return;
    }
    this.#lastPublished = now;
    this.state = { ...this.state, updated_at: wallClockSeconds() };
    this.listener?.(this.state);
    if (this.directory !== undefined) {
      try {
        writeSnapshot(this.directory, this.state);
      } catch {
        // An unwritable snapshot is not the run's problem.
      }
    }
  }

  /** Remove the snapshot once the run is over. */
  clear(): void {
    if (this.directory === undefined) return;
    try {
      fs.rmSync(progressPath(this.directory, this.state.project_id), { force: true });
    } catch {
      // Same contract as update(): losing the file is never worth an exception.
    }
  }
}

/**
 * Write the snapshot whole and rename it into place, so a reader polling the
 * file never parses a half-written one.
 */
function writeSnapshot(directory: string, state: IndexProgress): void {
  fs.mkdirSync(directory, { recursive: true });
  const target = progressPath(directory, state.project_id);
  const temporary = path.join(directory, `.${state.project_id}.${process.pid}.${nextSeq()}.tmp`);
  try {
    fs.writeFileSync(temporary, JSON.stringify(state), { flag: "wx" });
    fs.renameSync(temporary, target);
  } catch (error) {
    fs.rmSync(temporary, { force: true });
    throw error;
  }
}

let sequence = 0;
function nextSeq(): number {
  sequence += 1;
  return sequence;
}

/**
 * Return the latest snapshot for *projectId*, or null when there is none.
 *
 * A snapshot older than *staleAfterSeconds* is treated as absent: it was left
 * behind by a process that died rather than by one still working.
 */
export function readProgress(
  directory: string,
  projectId: string,
  { staleAfterSeconds = STALE_AFTER_SECONDS }: { staleAfterSeconds?: number } = {},
): IndexProgress | null {
  let progress: IndexProgress;
  try {
    progress = IndexProgress.parse(
      JSON.parse(fs.readFileSync(progressPath(directory, projectId), "utf8")),
    );
  } catch {
    return null;
  }
  if (wallClockSeconds() - progress.updated_at > staleAfterSeconds) return null;
  return progress;
}
