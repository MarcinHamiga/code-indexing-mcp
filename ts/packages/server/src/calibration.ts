/**
 * One-time measurement of what a backend can actually do.
 *
 * A probe answers "does this work". Nothing until now answered "how fast, at
 * what batch size, and is starting it worth the wait" -- so batch size was
 * configured rather than measured, and an accelerator was used for a one-file
 * re-index it could not possibly repay.
 */

import {
  type EmbeddedSegment,
  type PassageCandidate,
  passageCandidate,
  type SegmentPlan,
  segmentPlan,
} from "./embedding.ts";
import { type CodeIndexingError, isCodeIndexingError } from "./errors.ts";
import { MAX_BATCH_SIZE } from "./settings.ts";

export const CANDIDATE_BATCH_SIZES = [1, 2, 4, 8, 16, 32] as const;
export const CALIBRATION_CANDIDATE_COUNT = 16;
export const REPRESENTATIVE_CHARACTERS = [384, 1536] as const;
export const IMPROVEMENT_RATIO = 1.05;
export const LIMITED_BY_MEMORY = "memory";
export const LIMITED_BY_FAILURE = "failure";

const RESOURCE_CODES = new Set(["INDEX_RESOURCE_LIMIT"]);

export interface MeasurableSession {
  planAndEmbed(
    candidates: readonly PassageCandidate[],
    plan: SegmentPlan,
  ): Promise<EmbeddedSegment[][]> | EmbeddedSegment[][];
}

export interface CalibrationResult {
  readonly maxItems: number;
  readonly charactersPerSecond: number;
  readonly loadNs: number;
  readonly limitedBy: string;
}

export function calibrationResult(fields: {
  maxItems: number;
  charactersPerSecond: number;
  loadNs: number;
  limitedBy?: string;
}): CalibrationResult {
  return {
    maxItems: fields.maxItems,
    charactersPerSecond: fields.charactersPerSecond,
    loadNs: fields.loadNs,
    limitedBy: fields.limitedBy ?? "",
  };
}

export function loadMs(result: CalibrationResult): number {
  return Math.trunc(result.loadNs / 1_000_000);
}

export function calibrationCandidates({
  count = CALIBRATION_CANDIDATE_COUNT,
  lengths = REPRESENTATIVE_CHARACTERS,
}: {
  count?: number;
  lengths?: readonly number[];
} = {}): PassageCandidate[] {
  const candidates: PassageCandidate[] = [];
  for (let index = 0; index < count; index++) {
    const target = lengths[index % lengths.length] ?? 384;
    const body: string[] = [
      `def calibration_${String(index).padStart(3, "0")}(values: list[int]) -> int:`,
    ];
    let line = 0;
    while (body.reduce((sum, entry) => sum + entry.length + 1, 0) < target) {
      body.push(
        `    total_${String(line).padStart(3, "0")} = sum(value * ${line + 2} for value in values[:${line + 1}])`,
      );
      line += 1;
    }
    const content = body.join("\n");
    candidates.push(
      passageCandidate(
        `# module calibration_${String(index).padStart(3, "0")}`,
        content.slice(0, target),
      ),
    );
  }
  return candidates;
}

export async function calibrate(
  session: MeasurableSession,
  plan: SegmentPlan,
  {
    loadNs,
    maxItems = MAX_BATCH_SIZE,
    candidates,
    clock = monotonicNs,
  }: {
    loadNs: number;
    maxItems?: number;
    candidates?: readonly PassageCandidate[];
    clock?: () => number;
  },
): Promise<CalibrationResult | undefined> {
  const corpus = candidates === undefined ? calibrationCandidates() : [...candidates];
  const characters = corpus.reduce(
    (sum, candidate) => sum + candidate.prefix.length + candidate.content.length,
    0,
  );
  let best: CalibrationResult | undefined;
  for (const size of CANDIDATE_BATCH_SIZES) {
    if (size > maxItems) break;
    const started = clock();
    try {
      await session.planAndEmbed(corpus, segmentPlan({ ...plan, maxItems: size }));
    } catch (error) {
      if (isCodeIndexingError(error)) {
        if (best === undefined) return undefined;
        const limitedBy = RESOURCE_CODES.has(error.code) ? LIMITED_BY_MEMORY : LIMITED_BY_FAILURE;
        return { ...best, limitedBy };
      }
      return best === undefined ? undefined : { ...best, limitedBy: LIMITED_BY_FAILURE };
    }
    const elapsed = Math.max(1, clock() - started);
    const rate = (characters * 1_000_000_000) / elapsed;
    if (best === undefined) {
      best = calibrationResult({ maxItems: size, charactersPerSecond: rate, loadNs });
      continue;
    }
    if (rate <= best.charactersPerSecond * IMPROVEMENT_RATIO) break;
    best = calibrationResult({ maxItems: size, charactersPerSecond: rate, loadNs });
  }
  return best;
}

export function crossoverCharacters({
  acceleratorLoadNs,
  cpuLoadNs,
  cpuCharactersPerSecond,
  acceleratorCharactersPerSecond,
}: {
  acceleratorLoadNs: number;
  cpuLoadNs: number;
  cpuCharactersPerSecond: number;
  acceleratorCharactersPerSecond: number;
}): number | null {
  if (cpuCharactersPerSecond <= 0 || acceleratorCharactersPerSecond <= 0) return null;
  if (acceleratorCharactersPerSecond <= cpuCharactersPerSecond) return null;
  const extraLoadNs = acceleratorLoadNs - cpuLoadNs;
  if (extraLoadNs <= 0) return 0;
  const savedPerCharacter = 1 / cpuCharactersPerSecond - 1 / acceleratorCharactersPerSecond;
  return Math.trunc(extraLoadNs / 1_000_000_000 / savedPerCharacter);
}

function monotonicNs(): number {
  return Number(process.hrtime.bigint());
}

export type { CodeIndexingError };
