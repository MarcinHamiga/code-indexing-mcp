import os from "node:os";
import path from "node:path";
import { expandUser } from "./config-files.ts";

export type SettingType = "bool" | "int" | "choice" | "path" | "auto_int" | "auto_off_int";

const TRUE_VALUES = new Set(["1", "true", "yes", "on"]);
const FALSE_VALUES = new Set(["0", "false", "no", "off"]);

export interface Setting {
  readonly name: string;
  readonly group: string;
  readonly label: string;
  readonly help: string;
  readonly type: SettingType;
  readonly default: string;
  readonly choices: readonly string[];
  readonly minimum: number;
  readonly maximum: number;
  readonly dynamicDefault?: () => string;
}

function defaultMemoryMb(): string {
  const total = os.totalmem();
  return String(Math.max(1024, Math.min(2048, Math.trunc((total * 0.25) / (1024 * 1024)))));
}

function defaultThreads(): string {
  return String(Math.max(1, Math.min(2, os.availableParallelism())));
}

function defaultDataHome(): string {
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support");
  }
  if (process.platform === "win32") {
    return process.env.APPDATA ?? path.join(os.homedir(), "AppData", "Roaming");
  }
  return process.env.XDG_DATA_HOME ?? path.join(os.homedir(), ".local", "share");
}

function defaultCacheHome(): string {
  if (process.platform === "darwin") return path.join(os.homedir(), "Library", "Caches");
  if (process.platform === "win32") {
    return process.env.LOCALAPPDATA ?? path.join(os.homedir(), "AppData", "Local");
  }
  return process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache");
}

function setting(
  name: string,
  group: string,
  label: string,
  help: string,
  type: SettingType,
  defaultValue: string,
  extras: Partial<Omit<Setting, "name" | "group" | "label" | "help" | "type" | "default">> = {},
): Setting {
  return {
    name,
    group,
    label,
    help,
    type,
    default: defaultValue,
    choices: extras.choices ?? [],
    minimum: extras.minimum ?? 0,
    maximum: extras.maximum ?? 0,
    ...(extras.dynamicDefault === undefined ? {} : { dynamicDefault: extras.dynamicDefault }),
  };
}

export const SETTINGS: readonly Setting[] = [
  setting(
    "CODE_INDEXING_INDEX_MODE",
    "Indexing",
    "Index mode",
    "Lazy checks freshness before each code query; eager indexes at startup and watches for changes; manual only indexes explicitly.",
    "choice",
    "lazy",
    { choices: ["lazy", "eager", "manual"] },
  ),
  setting(
    "CODE_INDEXING_INDEX_WAIT_SECONDS",
    "Indexing",
    "Index wait (seconds)",
    "How long a startup index waits out a competing job before failing; 0 disables waiting.",
    "int",
    "300",
    { minimum: 0, maximum: 24 * 60 * 60 },
  ),
  setting(
    "CODE_INDEXING_EMBED_MEMORY_MB",
    "Indexing",
    "Indexing memory (MB)",
    "Ceiling for the indexing worker. The default is 25% of RAM clamped to 1024-2048.",
    "int",
    "",
    { minimum: 1024, maximum: 1024 * 1024, dynamicDefault: defaultMemoryMb },
  ),
  setting(
    "CODE_INDEXING_VECTOR_INDEX",
    "Indexing",
    "Vector index",
    "exact search, or approximate HNSW indexing.",
    "choice",
    "exact",
    { choices: ["exact", "hnsw"] },
  ),
  setting(
    "CODE_INDEXING_VECTOR_STORAGE",
    "Indexing",
    "Vector storage",
    "float16 halves vector bytes with no measured retrieval loss; float32 restores the previous layout (a rebuild follows either change).",
    "choice",
    "float16",
    { choices: ["float16", "float32"] },
  ),
  setting(
    "CODE_INDEXING_INDEX_EXECUTION",
    "Indexing",
    "Index execution",
    "worker enforces the memory ceiling; in-process is a diagnostic rollback.",
    "choice",
    "worker",
    { choices: ["worker", "in-process"] },
  ),
  setting(
    "CODE_INDEXING_BROKER",
    "Indexing",
    "Broker",
    "Share one indexing process between clients through the daemon.",
    "choice",
    "auto",
    { choices: ["auto", "on", "off"] },
  ),
  setting(
    "CODE_INDEXING_DATA_DIR",
    "Indexing",
    "Data directory",
    "Where the indexes live.",
    "path",
    "",
    { dynamicDefault: () => path.join(defaultDataHome(), "code-indexing-mcp") },
  ),
  setting(
    "CODE_INDEXING_CACHE_DIR",
    "Indexing",
    "Cache directory",
    "Where the embedding model is cached.",
    "path",
    "",
    { dynamicDefault: () => path.join(defaultCacheHome(), "code-indexing-mcp") },
  ),
  setting(
    "CODE_INDEXING_OFFLINE",
    "Indexing",
    "Offline mode",
    "Never download the model; fail if it is missing.",
    "bool",
    "0",
  ),
  setting(
    "CODE_INDEXING_EMBED_BATCH_SIZE",
    "Embedding",
    "Batch size",
    "Embedding microbatch size; auto resolves to 1 unless calibration raised it.",
    "auto_int",
    "auto",
    { minimum: 1, maximum: 256 },
  ),
  setting(
    "CODE_INDEXING_EMBED_MAX_TOKENS",
    "Embedding",
    "Max tokens",
    "Sequence window per chunk; attention memory is quadratic in tokens.",
    "int",
    "1024",
    { minimum: 64, maximum: 8192 },
  ),
  setting(
    "CODE_INDEXING_EMBED_OVERLAP_TOKENS",
    "Embedding",
    "Overlap tokens",
    "Overlap between consecutive windows of a long chunk.",
    "int",
    "64",
    { minimum: 0, maximum: 4096 },
  ),
  setting(
    "CODE_INDEXING_EMBED_THREADS",
    "Embedding",
    "Threads",
    "CPU inference threads.",
    "int",
    "",
    { minimum: 1, maximum: 64, dynamicDefault: defaultThreads },
  ),
  setting(
    "CODE_INDEXING_EMBED_CPU_ARENA",
    "Embedding",
    "CPU arena",
    "Preallocate the CPU inference arena.",
    "bool",
    "0",
  ),
  setting(
    "CODE_INDEXING_EMBED_CROSSOVER",
    "Embedding",
    "Accelerator crossover",
    "Run size in characters above which starting the accelerator repays its model load.",
    "auto_off_int",
    "auto",
    { minimum: 0, maximum: 1024 ** 3 },
  ),
  setting(
    "CODE_INDEXING_EMBED_CALIBRATE",
    "Embedding",
    "Calibrate",
    "Measure the backend once to set the batch size and crossover.",
    "bool",
    "1",
  ),
  setting(
    "CODE_INDEXING_EMBED_STRICT",
    "Embedding",
    "Strict accelerator",
    "Refuse the CPU fallback when the requested backend is unavailable.",
    "bool",
    "0",
  ),
  setting(
    "CODE_INDEXING_EMBED_ACCELERATOR",
    "Embedding",
    "Backend override",
    "Expert override; auto uses the backend the installer prepared.",
    "choice",
    "auto",
    { choices: ["auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"] },
  ),
  setting(
    "CODE_INDEXING_AUTO_MAINTENANCE",
    "Maintenance",
    "Automatic maintenance",
    "Compacts tables and removes verified versions older than the retention window on a schedule; never uses zero-age cleanup.",
    "bool",
    "1",
  ),
  setting(
    "CODE_INDEXING_VERSION_RETENTION_HOURS",
    "Maintenance",
    "Version retention (hours)",
    "How long old Lance versions are kept before verified cleanup; the floor of one hour keeps concurrent readers safe.",
    "int",
    "24",
    { minimum: 1, maximum: 24 * 30 },
  ),
];

export const BY_NAME: Readonly<Record<string, Setting>> = Object.fromEntries(
  SETTINGS.map((item) => [item.name, item]),
);

export function defaultValue(item: Setting): string {
  return item.dynamicDefault !== undefined ? item.dynamicDefault() : item.default;
}

export function asBool(raw: string): boolean {
  return TRUE_VALUES.has(raw.trim().toLowerCase());
}

export function validate(item: Setting, raw: string): string | null {
  const value = raw.trim();
  if (item.type === "bool") {
    if (TRUE_VALUES.has(value.toLowerCase()) || FALSE_VALUES.has(value.toLowerCase())) return null;
    return `${item.name} expects a boolean (1/0, true/false, yes/no, on/off)`;
  }
  if (item.type === "path") return value !== "" ? null : `${item.name} expects a path`;
  if (item.type === "choice") {
    if (item.choices.includes(value.toLowerCase())) return null;
    return `${item.name} expects one of: ${item.choices.join(", ")}`;
  }
  if (item.type === "auto_int" && value.toLowerCase() === "auto") return null;
  if (
    item.type === "auto_off_int" &&
    (value.toLowerCase() === "auto" || value.toLowerCase() === "off")
  ) {
    return null;
  }
  let prefix = "";
  if (item.type === "auto_int") prefix = "auto or ";
  else if (item.type === "auto_off_int") prefix = "auto, off, or ";
  const number = Number.parseInt(value, 10);
  if (!/^[+-]?\d+$/.test(value) || !Number.isSafeInteger(number)) {
    return `${item.name} expects ${prefix}an integer from ${item.minimum} to ${item.maximum}`;
  }
  if (item.minimum <= number && number <= item.maximum) return null;
  return `${item.name} expects ${prefix}an integer from ${item.minimum} to ${item.maximum}`;
}

export function normalize(item: Setting, raw: string): string {
  const value = raw.trim();
  if (item.type === "bool") return TRUE_VALUES.has(value.toLowerCase()) ? "1" : "0";
  if (item.type === "path") {
    if (value.startsWith("~")) {
      try {
        return expandUser(value);
      } catch {
        return value;
      }
    }
    return value;
  }
  if (
    (item.type === "choice" || item.type === "auto_int" || item.type === "auto_off_int") &&
    !/^\d+$/.test(value.replace(/^-+/, ""))
  ) {
    return value.toLowerCase();
  }
  return value;
}
