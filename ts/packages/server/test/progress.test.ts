/**
 * Live indexing progress: in-process listeners and the cross-process snapshot.
 *
 * The Python suite drives most of this through a real `Indexer`, which arrives
 * in Phase 5. What is asserted here is everything the publisher and the reader
 * own by themselves; the end-to-end counters (`candidates_seen` rising
 * monotonically through a run, skip reasons aggregating, a run id on every
 * update) come back with the indexer that produces them.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { describeProgress, progressFraction } from "../src/models.ts";
import { IndexProgress, ProgressPublisher, progressPath, readProgress } from "../src/progress.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let directory: string;

beforeEach(() => {
  directory = temporaryDirectory();
});

afterEach(() => {
  removeDirectory(directory);
});

describe("the snapshot's own reporting", () => {
  test("119 eligible plus 1,367 skipped never reads as 1486/~119", () => {
    // The review-measured case. The old contract counted every candidate in the
    // seen counter and only the eligible files in the total, so the description
    // could print `1486/~119`.
    const progress = IndexProgress.parse({
      project_id: "p",
      run_id: "r",
      candidates_seen: 1486,
      candidates_total: 1486,
      eligible_files: 119,
      skipped_total: 1367,
    });

    expect(describeProgress(progress)).not.toContain("/~119");
    expect(progressFraction(progress)).toBe(1);

    // Without a candidate total the fraction must stay unknown rather than
    // falling back to an eligible-file denominator.
    const ambiguous = { ...progress, candidates_total: null };
    expect(progressFraction(ambiguous)).toBeNull();
    expect(describeProgress(ambiguous)).toContain("1486 candidates, 119 eligible");
  });

  test("a run that has seen nothing says so", () => {
    const progress = IndexProgress.parse({ project_id: "p" });

    expect(describeProgress(progress)).toBe("Scanning for changed files");
    expect(progressFraction(progress)).toBeNull();
  });

  test("committing and reference extraction have distinct descriptions", () => {
    expect(describeProgress(IndexProgress.parse({ project_id: "p", phase: "committing" }))).toBe(
      "Committing the index",
    );
    expect(
      describeProgress(IndexProgress.parse({ project_id: "p", phase: "extracting_references" })),
    ).toBe("Extracting structural references");
  });

  test("the fraction never exceeds one even when the total was underestimated", () => {
    const progress = IndexProgress.parse({
      project_id: "p",
      candidates_seen: 200,
      candidates_total: 100,
    });

    expect(progressFraction(progress)).toBe(1);
  });
});

describe("publishing", () => {
  test("updates are throttled but the forced ones always land", () => {
    // The first tick is consumed for the initial phase anchor.
    const ticks = [0.0, 0.05, 0.1, 0.15, 0.2];
    let index = 0;
    const seen: number[] = [];
    const publisher = new ProgressPublisher("project", {
      listener: (progress) => seen.push(progress.candidates_seen),
      intervalSeconds: 1,
      clock: () => ticks[index++] ?? 0,
    });

    publisher.update({ candidates_seen: 1 });
    publisher.update({ candidates_seen: 2 });
    publisher.update({ candidates_seen: 3 });
    publisher.update({ candidates_seen: 4 }, { force: true });

    expect(seen).toEqual([1, 4]);
  });

  test("a publisher with nowhere to publish does no work", () => {
    const publisher = new ProgressPublisher("project");

    publisher.update({ candidates_seen: 7 }, { force: true });

    expect(publisher.enabled).toBe(false);
    expect(publisher.state.candidates_seen).toBe(0);
  });

  test("a phase change stamps phase_started_at and publishes immediately", () => {
    const publisher = new ProgressPublisher("project", {
      listener: () => undefined,
      intervalSeconds: 1,
    });
    const before = publisher.state.phase_started_at;

    publisher.update({ phase: "embedding", candidates_seen: 1 }, { force: true });

    expect(publisher.state.phase).toBe("embedding");
    expect(publisher.state.phase_started_at).toBeGreaterThan(before);
  });

  test("a phase anchor advances even when the clock does not", () => {
    // Some Windows runners read milliseconds, so the same tick can come back
    // twice and phase durations would collapse to zero.
    const publisher = new ProgressPublisher("project", {
      listener: () => undefined,
      clock: () => 5,
    });
    const before = publisher.state.phase_started_at;

    publisher.update({ phase: "parsing" }, { force: true });

    expect(publisher.state.phase_started_at).toBeGreaterThan(before);
  });

  test("retained snapshots do not change when the source object is mutated", () => {
    const publisher = new ProgressPublisher("project", {
      listener: () => undefined,
      intervalSeconds: 0,
    });
    const reasons: Record<string, number> = { binary: 1 };
    publisher.update({ skipped_by_reason: reasons }, { force: true });
    const snapshot = publisher.state;

    reasons.binary = 99;
    publisher.update({ skipped_by_reason: reasons }, { force: true });

    expect(snapshot.skipped_by_reason).toEqual({ binary: 1 });
    expect(publisher.state.skipped_by_reason).toEqual({ binary: 99 });
  });

  test("the run id and trigger ride along on every snapshot", () => {
    const seen: IndexProgress[] = [];
    const publisher = new ProgressPublisher("project", {
      runId: "run-1",
      trigger: "watcher",
      listener: (progress) => seen.push(progress),
      intervalSeconds: 0,
    });

    publisher.update({ candidates_seen: 1 }, { force: true });
    publisher.update({ candidates_seen: 2 }, { force: true });

    expect(seen.map((progress) => progress.run_id)).toEqual(["run-1", "run-1"]);
    expect(seen.map((progress) => progress.trigger)).toEqual(["watcher", "watcher"]);
  });
});

describe("the cross-process snapshot", () => {
  test("another process can read it, and it is gone after clear()", () => {
    const publisher = new ProgressPublisher("abc", { directory });

    publisher.update({ candidates_seen: 3, eligible_files: 3 }, { force: true });
    const published = readProgress(directory, "abc");
    publisher.clear();

    expect(published?.candidates_seen).toBe(3);
    expect(readProgress(directory, "abc")).toBeNull();
    expect(fs.existsSync(progressPath(directory, "abc"))).toBe(false);
  });

  test("clearing a snapshot that was never written is not an error", () => {
    expect(() => new ProgressPublisher("abc", { directory }).clear()).not.toThrow();
  });

  test("the snapshot is replaced atomically", () => {
    const publisher = new ProgressPublisher("abc", { directory });

    publisher.update({ candidates_seen: 1 }, { force: true });
    publisher.update({ candidates_seen: 2 }, { force: true });

    const payload = JSON.parse(fs.readFileSync(progressPath(directory, "abc"), "utf8"));
    expect(payload.candidates_seen).toBe(2);
    // No temporary file survives the rename, in either direction.
    expect(fs.readdirSync(directory)).toEqual(["abc.json"]);
  });

  test("the directory is created on demand", () => {
    const nested = path.join(directory, "deep", "progress");
    const publisher = new ProgressPublisher("abc", { directory: nested });

    publisher.update({ candidates_seen: 1 }, { force: true });

    expect(readProgress(nested, "abc")?.candidates_seen).toBe(1);
  });

  test("a snapshot left behind by a dead process is ignored", () => {
    const stale = IndexProgress.parse({ project_id: "abc", updated_at: Date.now() / 1000 - 3600 });
    fs.writeFileSync(progressPath(directory, "abc"), JSON.stringify(stale), "utf8");

    expect(readProgress(directory, "abc")).toBeNull();
    expect(readProgress(directory, "abc", { staleAfterSeconds: 7200 })).not.toBeNull();
  });

  test("a corrupt or absent snapshot is treated as absent", () => {
    expect(readProgress(directory, "missing")).toBeNull();
    fs.writeFileSync(progressPath(directory, "abc"), "{not json", "utf8");
    expect(readProgress(directory, "abc")).toBeNull();
  });

  test("a snapshot that is valid JSON but not a snapshot is treated as absent", () => {
    fs.writeFileSync(progressPath(directory, "abc"), JSON.stringify({ nope: true }), "utf8");

    expect(readProgress(directory, "abc")).toBeNull();
  });

  test("an unwritable directory costs the update, not the run", () => {
    // A file where the directory should be is the cheapest reproduction of a
    // publish that cannot succeed.
    const blocked = path.join(directory, "blocked");
    fs.writeFileSync(blocked, "");
    const publisher = new ProgressPublisher("abc", { directory: blocked });

    expect(() => publisher.update({ candidates_seen: 1 }, { force: true })).not.toThrow();
    expect(publisher.state.candidates_seen).toBe(1);
  });
});
