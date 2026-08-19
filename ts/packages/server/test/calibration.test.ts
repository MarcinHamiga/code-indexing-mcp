/** Batch calibration and the workload crossover. */

import { expect, test } from "bun:test";
import {
  CANDIDATE_BATCH_SIZES,
  calibrate,
  calibrationCandidates,
  calibrationResult,
  crossoverCharacters,
  loadMs,
} from "../src/calibration.ts";
import {
  type EmbeddedSegment,
  embeddedSegment,
  type PassageCandidate,
  type SegmentPlan,
  segmentPlan,
} from "../src/embedding.ts";
import { CodeIndexingError } from "../src/errors.ts";

const PLAN = segmentPlan({ maxItems: 1 });

class FakeSession {
  nanosecondsPerCall: Record<number, number>;
  now = 0;
  batchSizes: number[] = [];

  constructor(nanosecondsPerCall: Record<number, number>) {
    this.nanosecondsPerCall = nanosecondsPerCall;
  }

  clock(): number {
    return this.now;
  }

  planAndEmbed(candidates: readonly PassageCandidate[], plan: SegmentPlan): EmbeddedSegment[][] {
    this.batchSizes.push(plan.maxItems);
    this.now += this.duration(plan.maxItems);
    return candidates.map((candidate) => [
      embeddedSegment(0, candidate.content.length, 0, new Uint8Array()),
    ]);
  }

  duration(maxItems: number): number {
    return this.nanosecondsPerCall[maxItems] ?? 0;
  }
}

class ExhaustedSession extends FakeSession {
  failsAbove: number;

  constructor(nanosecondsPerCall: Record<number, number>, failsAbove: number) {
    super(nanosecondsPerCall);
    this.failsAbove = failsAbove;
  }

  override planAndEmbed(
    candidates: readonly PassageCandidate[],
    plan: SegmentPlan,
  ): EmbeddedSegment[][] {
    if (plan.maxItems > this.failsAbove) {
      this.batchSizes.push(plan.maxItems);
      throw new CodeIndexingError("INDEX_RESOURCE_LIMIT", "Indexing exceeded its memory ceiling");
    }
    return super.planAndEmbed(candidates, plan);
  }
}

function halving(nanoseconds: number): Record<number, number> {
  const durations: Record<number, number> = {};
  for (const size of CANDIDATE_BATCH_SIZES) {
    durations[size] = Math.max(1, Math.trunc(nanoseconds / size));
  }
  return durations;
}

test("the calibration corpus is deterministic and code shaped", () => {
  const first = calibrationCandidates();
  const second = calibrationCandidates();

  expect(first.map((candidate) => candidate.content)).toEqual(
    second.map((candidate) => candidate.content),
  );
  expect(first.some((candidate) => candidate.content.includes("def "))).toBe(true);
  expect(new Set(first.map((candidate) => candidate.content.length)).size).toBeGreaterThanOrEqual(
    2,
  );
});

test("the fastest batch size is the calibrated one", async () => {
  const session = new FakeSession(halving(1_000_000_000));

  const result = await calibrate(session, PLAN, { loadNs: 0, clock: () => session.clock() });

  expect(result).toBeDefined();
  expect(result?.maxItems).toBe(CANDIDATE_BATCH_SIZES[CANDIDATE_BATCH_SIZES.length - 1]);
  expect(result?.limitedBy).toBe("");
});

test("the sweep stops once a larger batch stops paying", async () => {
  const durations: Record<number, number> = {};
  for (const size of CANDIDATE_BATCH_SIZES) durations[size] = 1_000_000_000;
  durations[2] = 400_000_000;
  const session = new FakeSession(durations);

  const result = await calibrate(session, PLAN, { loadNs: 0, clock: () => session.clock() });

  expect(result).toBeDefined();
  expect(result?.maxItems).toBe(2);
  expect(session.batchSizes).toEqual([1, 2, 4]);
});

test("a batch size that overruns the ceiling is not calibrated", async () => {
  const session = new ExhaustedSession(halving(1_000_000_000), 4);

  const result = await calibrate(session, PLAN, { loadNs: 0, clock: () => session.clock() });

  expect(result).toBeDefined();
  expect(result?.maxItems).toBe(4);
  expect(result?.limitedBy).toBe("memory");
});

test("the first batch size overrunning leaves nothing calibrated", async () => {
  const session = new ExhaustedSession(halving(1_000_000_000), 0);

  expect(
    await calibrate(session, PLAN, { loadNs: 0, clock: () => session.clock() }),
  ).toBeUndefined();
});

test("a session that fails outright is not calibrated", async () => {
  class BrokenSession extends FakeSession {
    override planAndEmbed(): EmbeddedSegment[][] {
      throw new CodeIndexingError("EMBEDDING_WORKER_FAILED", "the worker died");
    }
  }

  expect(
    await calibrate(new BrokenSession({}), PLAN, { loadNs: 0, clock: () => 0 }),
  ).toBeUndefined();
});

test("a batch that kills the worker keeps what smaller ones measured", async () => {
  class DyingSession extends FakeSession {
    override planAndEmbed(
      candidates: readonly PassageCandidate[],
      plan: SegmentPlan,
    ): EmbeddedSegment[][] {
      if (plan.maxItems > 4) {
        throw new CodeIndexingError("EMBEDDING_WORKER_FAILED", "the worker died");
      }
      return super.planAndEmbed(candidates, plan);
    }
  }

  const session = new DyingSession(halving(1_000_000_000));
  const result = await calibrate(session, PLAN, { loadNs: 0, clock: () => session.clock() });

  expect(result).toBeDefined();
  expect(result?.maxItems).toBe(4);
  expect(result?.limitedBy).toBe("failure");
});

test("the measured rate counts the same characters the crossover does", async () => {
  const session = new FakeSession(
    Object.fromEntries(CANDIDATE_BATCH_SIZES.map((size) => [size, 2_000_000_000])),
  );
  const candidates = calibrationCandidates();
  const characters = candidates.reduce(
    (sum, candidate) => sum + candidate.prefix.length + candidate.content.length,
    0,
  );
  expect(characters).toBeGreaterThan(
    candidates.reduce((sum, candidate) => sum + candidate.content.length, 0),
  );

  const result = await calibrate(session, PLAN, { loadNs: 0, clock: () => session.clock() });

  expect(result).toBeDefined();
  expect(result?.charactersPerSecond).toBeCloseTo(characters / 2.0);
});

test("the calibrated size never exceeds the configured maximum", async () => {
  const session = new FakeSession(halving(1_000_000_000));

  const result = await calibrate(session, PLAN, {
    loadNs: 0,
    clock: () => session.clock(),
    maxItems: 4,
  });

  expect(result).toBeDefined();
  expect(result?.maxItems).toBe(4);
  expect(Math.max(...session.batchSizes)).toBe(4);
});

test("the crossover is where startup stops costing more than it saves", () => {
  expect(
    crossoverCharacters({
      acceleratorLoadNs: 2_000_000_000,
      cpuLoadNs: 0,
      cpuCharactersPerSecond: 1_000.0,
      acceleratorCharactersPerSecond: 2_000.0,
    }),
  ).toBe(4_000);
});

test("only the startup the accelerator costs beyond cpu has to be earned back", () => {
  expect(
    crossoverCharacters({
      acceleratorLoadNs: 3_000_000_000,
      cpuLoadNs: 1_000_000_000,
      cpuCharactersPerSecond: 1_000.0,
      acceleratorCharactersPerSecond: 2_000.0,
    }),
  ).toBe(4_000);
});

test("an accelerator that loads faster than cpu is worth starting at once", () => {
  expect(
    crossoverCharacters({
      acceleratorLoadNs: 370_000_000,
      cpuLoadNs: 655_000_000,
      cpuCharactersPerSecond: 14_030.0,
      acceleratorCharactersPerSecond: 46_783.0,
    }),
  ).toBe(0);
});

test("an accelerator no faster than cpu has no crossover", () => {
  expect(
    crossoverCharacters({
      acceleratorLoadNs: 2_000_000_000,
      cpuLoadNs: 0,
      cpuCharactersPerSecond: 2_000.0,
      acceleratorCharactersPerSecond: 2_000.0,
    }),
  ).toBeNull();
});

test("a free start crosses over immediately", () => {
  expect(
    crossoverCharacters({
      acceleratorLoadNs: 0,
      cpuLoadNs: 0,
      cpuCharactersPerSecond: 1_000.0,
      acceleratorCharactersPerSecond: 2_000.0,
    }),
  ).toBe(0);
});

test("an unmeasured rate has no crossover", () => {
  expect(
    crossoverCharacters({
      acceleratorLoadNs: 1_000_000_000,
      cpuLoadNs: 0,
      cpuCharactersPerSecond: 0.0,
      acceleratorCharactersPerSecond: 2_000.0,
    }),
  ).toBeNull();
});

test("a calibration result reports what it measured", () => {
  const result = calibrationResult({
    maxItems: 8,
    charactersPerSecond: 1_234.5,
    loadNs: 2_000_000_000,
  });

  expect(loadMs(result)).toBe(2_000);
});
