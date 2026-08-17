/** Validated runtime settings for memory-safe indexing. */

import os from "node:os";
import { type Accelerator, parseAccelerator } from "./backends.ts";
import { CodeIndexingError } from "./errors.ts";
import { DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS } from "./token-batching.ts";

/**
 * The batch size `auto` resolves to when no calibration record applies. One item
 * per microbatch is what CPU indexing has always used and what the memory
 * ceiling was measured against; a larger default belongs to a backend that has
 * earned it through calibration.
 */
export const DEFAULT_AUTO_BATCH_SIZE = 1;
export const MAX_BATCH_SIZE = 256;
/**
 * A gigabyte of source in one run is already far past any measured crossover, so
 * a larger figure is a mistyped setting rather than a policy.
 */
export const MAX_CROSSOVER_CHARACTERS = 1024 ** 3;
/**
 * Automatic maintenance reclaims verified versions older than this by default.
 * The lower bound of one hour keeps zero-age cleanup unreachable from
 * configuration: concurrent readers must never have live versions reaped under
 * them. The manual storage vacuum command performs the same bounded cleanup on
 * demand rather than unlocking a lower floor.
 */
export const DEFAULT_VERSION_RETENTION_HOURS = 24;
export const MAX_VERSION_RETENTION_HOURS = 24 * 30;

export const INDEX_MODES = ["lazy", "eager", "manual"] as const;
export type IndexMode = (typeof INDEX_MODES)[number];

/** An environment as this module reads it: absent and empty are distinct. */
export type Environment = Readonly<Partial<Record<string, string>>>;

export interface IndexSettings {
  readonly mode: IndexMode;
  readonly indexWaitSeconds: number;
  readonly embeddingBatchSize: number;
  readonly embeddingMaxTokens: number;
  readonly embeddingOverlapTokens: number;
  readonly embeddingThreads: number;
  readonly embeddingCpuArena: boolean;
  readonly vectorIndex: "exact" | "hnsw";
  readonly vectorStorage: "float32" | "float16";
  readonly indexMemoryBytes: number;
  readonly indexExecution: "worker" | "in-process";
  readonly brokerMode: "auto" | "on" | "off";
  readonly embeddingAccelerator: Accelerator;
  /**
   * True when the batch size was left to the runtime, which lets calibration
   * raise it for a backend that was measured to handle more.
   */
  readonly embeddingBatchAuto: boolean;
  /**
   * The run size, in candidate characters, above which starting an accelerator
   * repays its model load. 0 with `Auto` set means nothing has measured one yet;
   * 0 with it clear means the operator turned deferral off.
   */
  readonly embeddingCrossoverCharacters: number;
  readonly embeddingCrossoverAuto: boolean;
  /**
   * Measuring a backend costs one sweep per configuration. Declining it leaves
   * the batch size and the crossover unmeasured, which is the behaviour every
   * release before this one had.
   */
  readonly embeddingCalibrate: boolean;
  /**
   * Strict mode refuses the CPU fallback. A run that cannot reach the requested
   * accelerator fails with BACKEND_UNAVAILABLE instead of quietly indexing more
   * slowly than the caller asked for.
   */
  readonly embeddingStrict: boolean;
  /**
   * Automatic maintenance compacts tables and removes verified versions older
   * than `versionRetentionHours`. It never uses zero-age cleanup and never sets
   * delete_unverified, regardless of configuration.
   */
  readonly autoMaintenance: boolean;
  readonly versionRetentionHours: number;
}

function configurationError(name: string, value: string, expected: string): CodeIndexingError {
  return new CodeIndexingError(
    "INVALID_CONFIGURATION",
    `${name}=${JSON.stringify(value)} is invalid; expected ${expected}`,
    { setting: name, value },
  );
}

/**
 * Parse exactly what Python's `int()` parses.
 *
 * `Number.parseInt` would accept `"12abc"` as 12 and `Number()` would accept
 * `"1e3"` and `" "`, either of which turns a typo into a silently different
 * setting. Underscore grouping is allowed because Python allows it and an
 * operator who wrote `1_048_576` meant it.
 */
function parseInteger(raw: string): number | null {
  const trimmed = raw.trim();
  if (!/^[+-]?\d+(?:_\d+)*$/.test(trimmed)) return null;
  const value = Number(trimmed.replaceAll("_", ""));
  return Number.isSafeInteger(value) ? value : null;
}

function integerSetting(
  environment: Environment,
  name: string,
  defaultValue: number,
  minimum: number,
  maximum: number,
): number {
  const raw = environment[name];
  if (raw === undefined) return defaultValue;
  const expected = `an integer from ${minimum} to ${maximum}`;
  const value = parseInteger(raw);
  if (value === null || value < minimum || value > maximum) {
    throw configurationError(name, raw, expected);
  }
  return value;
}

const TRUE_VALUES = new Set(["1", "true", "yes", "on"]);
const FALSE_VALUES = new Set(["0", "false", "no", "off"]);

function booleanSetting(environment: Environment, name: string, defaultValue: boolean): boolean {
  const raw = environment[name];
  if (raw === undefined) return defaultValue;
  const normalized = raw.toLowerCase();
  if (TRUE_VALUES.has(normalized)) return true;
  if (FALSE_VALUES.has(normalized)) return false;
  throw configurationError(name, raw, "a boolean");
}

function choiceSetting<const T extends readonly string[]>(
  environment: Environment,
  name: string,
  choices: T,
  defaultValue: T[number],
  expected: string,
): T[number] {
  const normalized = (environment[name] ?? defaultValue).toLowerCase();
  const chosen = choices.find((candidate) => candidate === normalized);
  if (chosen === undefined) throw configurationError(name, normalized, expected);
  return chosen;
}

/** The configured microbatch size, and whether it was left automatic. */
function batchSize(environment: Environment): { size: number; auto: boolean } {
  const raw = environment.CODE_INDEXING_EMBED_BATCH_SIZE;
  if (raw === undefined || raw.trim().toLowerCase() === "auto") {
    return { size: DEFAULT_AUTO_BATCH_SIZE, auto: true };
  }
  return {
    size: integerSetting(environment, "CODE_INDEXING_EMBED_BATCH_SIZE", 1, 1, MAX_BATCH_SIZE),
    auto: false,
  };
}

/**
 * The configured crossover in characters, and whether it is measured.
 *
 * `off` is a size of zero, which reads correctly everywhere downstream: no run
 * is smaller than the threshold, so the accelerator starts on the first chunk,
 * exactly as it did before anything measured whether that paid.
 */
function crossover(environment: Environment): { characters: number; auto: boolean } {
  const name = "CODE_INDEXING_EMBED_CROSSOVER";
  const raw = environment[name];
  if (raw === undefined || raw.trim().toLowerCase() === "auto") {
    return { characters: 0, auto: true };
  }
  if (raw.trim().toLowerCase() === "off") return { characters: 0, auto: false };
  const value = parseInteger(raw);
  if (value === null) {
    throw configurationError(name, raw, "auto, off, or a character count");
  }
  if (value < 0 || value > MAX_CROSSOVER_CHARACTERS) {
    throw configurationError(name, raw, `a character count up to ${MAX_CROSSOVER_CHARACTERS}`);
  }
  return { characters: value, auto: false };
}

/**
 * Resolve the indexing memory ceiling from either accepted variable.
 *
 * `CODE_INDEXING_EMBED_MEMORY_MB` is the documented name.
 * `CODE_INDEXING_INDEX_MEMORY_MB` predates it and keeps working; the newer name
 * wins when both are set.
 */
function memoryBytes(environment: Environment, defaultMegabytes: number): number {
  // Truthiness, not presence: an exported-but-empty variable is how a shell says
  // "unset", and letting it win would both fail to parse and shadow a perfectly
  // good value under the legacy name.
  const name = environment.CODE_INDEXING_EMBED_MEMORY_MB
    ? "CODE_INDEXING_EMBED_MEMORY_MB"
    : "CODE_INDEXING_INDEX_MEMORY_MB";
  return integerSetting(environment, name, defaultMegabytes, 1024, 1024 * 1024) * 1024 * 1024;
}

function indexMode(environment: Environment): IndexMode {
  const raw = environment.CODE_INDEXING_INDEX_MODE;
  if (raw === undefined) {
    if (environment.CODE_INDEXING_AUTO_INDEX === undefined) return "lazy";
    return booleanSetting(environment, "CODE_INDEXING_AUTO_INDEX", false) ? "eager" : "manual";
  }
  const normalized = raw.toLowerCase();
  const mode = INDEX_MODES.find((candidate) => candidate === normalized);
  if (mode === undefined) {
    throw configurationError("CODE_INDEXING_INDEX_MODE", raw, "lazy, eager, or manual");
  }
  return mode;
}

export function indexSettingsFromEnvironment(
  environment: Environment = process.env,
): IndexSettings {
  const vectorIndex = choiceSetting(
    environment,
    "CODE_INDEXING_VECTOR_INDEX",
    ["exact", "hnsw"] as const,
    "exact",
    "exact or hnsw",
  );
  const vectorStorage = choiceSetting(
    environment,
    "CODE_INDEXING_VECTOR_STORAGE",
    ["float32", "float16"] as const,
    "float16",
    "float32 or float16",
  );
  const indexExecution = choiceSetting(
    environment,
    "CODE_INDEXING_INDEX_EXECUTION",
    ["worker", "in-process"] as const,
    "worker",
    "worker or in-process",
  );
  const brokerMode = choiceSetting(
    environment,
    "CODE_INDEXING_BROKER",
    ["auto", "on", "off"] as const,
    "auto",
    "auto, on, or off",
  );
  const defaultMemoryMegabytes = Math.max(
    1024,
    Math.min(2048, Math.floor(Math.floor(os.totalmem() * 0.25) / (1024 * 1024))),
  );
  const batch = batchSize(environment);
  const deferral = crossover(environment);

  return {
    mode: indexMode(environment),
    // How long a startup index waits out a competing job before failing. The
    // global index lock serializes every job on the machine, so a cold index
    // elsewhere can hold it for minutes; 0 disables waiting.
    indexWaitSeconds: integerSetting(
      environment,
      "CODE_INDEXING_INDEX_WAIT_SECONDS",
      300,
      0,
      24 * 60 * 60,
    ),
    embeddingBatchSize: batch.size,
    embeddingBatchAuto: batch.auto,
    embeddingCrossoverCharacters: deferral.characters,
    embeddingCrossoverAuto: deferral.auto,
    embeddingCalibrate: booleanSetting(environment, "CODE_INDEXING_EMBED_CALIBRATE", true),
    embeddingAccelerator: parseAccelerator(environment.CODE_INDEXING_EMBED_ACCELERATOR ?? "auto"),
    embeddingStrict: booleanSetting(environment, "CODE_INDEXING_EMBED_STRICT", false),
    // Sequence length, not character count, drives embedding memory: attention is
    // quadratic in tokens. 1,024 keeps the widest window well inside the model's
    // 8,192-token limit and inside the default memory ceiling even for
    // token-dense minified source.
    embeddingMaxTokens: integerSetting(
      environment,
      "CODE_INDEXING_EMBED_MAX_TOKENS",
      DEFAULT_MAX_TOKENS,
      64,
      8192,
    ),
    embeddingOverlapTokens: integerSetting(
      environment,
      "CODE_INDEXING_EMBED_OVERLAP_TOKENS",
      DEFAULT_OVERLAP_TOKENS,
      0,
      4096,
    ),
    embeddingThreads: integerSetting(
      environment,
      "CODE_INDEXING_EMBED_THREADS",
      // `availableParallelism` rather than the raw CPU count: it honours the
      // cgroup quota a container was given, which is where an over-threaded
      // embedding session actually hurts.
      Math.max(1, Math.min(2, os.availableParallelism())),
      1,
      64,
    ),
    embeddingCpuArena: booleanSetting(environment, "CODE_INDEXING_EMBED_CPU_ARENA", false),
    vectorIndex,
    vectorStorage,
    indexMemoryBytes: memoryBytes(environment, defaultMemoryMegabytes),
    indexExecution,
    brokerMode,
    autoMaintenance: booleanSetting(environment, "CODE_INDEXING_AUTO_MAINTENANCE", true),
    // One hour floor: configuration must never be able to reap versions that
    // concurrent searches could still be reading.
    versionRetentionHours: integerSetting(
      environment,
      "CODE_INDEXING_VERSION_RETENTION_HOURS",
      DEFAULT_VERSION_RETENTION_HOURS,
      1,
      MAX_VERSION_RETENTION_HOURS,
    ),
  };
}
