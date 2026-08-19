/**
 * Local cache of successful backend probes and batch calibration.
 *
 * Probing an accelerator costs a process spawn, a model load, and a real
 * inference. That cost is worth paying once per machine configuration, not once
 * per index run -- but a cached "this works" is only meaningful while nothing
 * underneath it moved.
 */

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import lockfile from "proper-lockfile";

export const CACHE_SCHEMA_VERSION = 2;
export const MAX_RECORDS = 32;
export const LOCK_TIMEOUT_SECONDS = 5;

export interface ProbeKey {
  readonly modelId: string;
  readonly modelArtifact: string;
  readonly accelerator: string;
  readonly provider: string;
  readonly runtimeVersion: string;
  readonly platform: string;
  readonly device: string;
  readonly driverVersion: string;
}

export function probeKey(fields: {
  modelId: string;
  modelArtifact: string;
  accelerator: string;
  provider: string;
  runtimeVersion: string;
  platform: string;
  device: string;
  driverVersion?: string;
}): ProbeKey {
  return {
    modelId: fields.modelId,
    modelArtifact: fields.modelArtifact,
    accelerator: fields.accelerator,
    provider: fields.provider,
    runtimeVersion: fields.runtimeVersion,
    platform: fields.platform,
    device: fields.device,
    driverVersion: fields.driverVersion ?? "",
  };
}

export function probeFingerprint(key: ProbeKey): string {
  const parts = [
    key.modelId,
    key.modelArtifact,
    key.accelerator,
    key.provider,
    key.runtimeVersion,
    key.platform,
    key.device,
    key.driverVersion,
  ];
  return createHash("sha256").update(parts.join("\0")).digest("hex");
}

export interface ProbeRecord {
  readonly fingerprint: string;
  readonly batchSize: number;
  readonly dimension: number;
  readonly recordedAtNs: number;
  readonly detail: string;
  readonly charactersPerSecond: number;
  readonly loadNs: number;
  readonly limitedBy: string;
}

export function probeRecordToJson(record: ProbeRecord): Record<string, unknown> {
  return {
    fingerprint: record.fingerprint,
    batch_size: record.batchSize,
    dimension: record.dimension,
    recorded_at_ns: record.recordedAtNs,
    detail: record.detail,
    characters_per_second: record.charactersPerSecond,
    load_ns: record.loadNs,
    limited_by: record.limitedBy,
  };
}

export function probeRecordFromJson(value: unknown): ProbeRecord | undefined {
  if (value === null || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  try {
    return {
      fingerprint: String(required(raw, "fingerprint")),
      batchSize: asInt(required(raw, "batch_size")),
      dimension: asInt(required(raw, "dimension")),
      recordedAtNs: asInt(required(raw, "recorded_at_ns")),
      detail: String(raw.detail ?? ""),
      charactersPerSecond: asFloat(raw.characters_per_second ?? 0),
      loadNs: asInt(raw.load_ns ?? 0),
      limitedBy: String(raw.limited_by ?? ""),
    };
  } catch {
    return undefined;
  }
}

function required(raw: Record<string, unknown>, key: string): unknown {
  if (!(key in raw)) throw new Error(key);
  return raw[key];
}

function asInt(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error("int");
  return Math.trunc(parsed);
}

function asFloat(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error("float");
  return parsed;
}

export function modelDirectory(cacheDirectory: string, modelId: string): string {
  const candidate = path.join(cacheDirectory, `models--${modelId.replaceAll("/", "--")}`);
  return fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()
    ? candidate
    : cacheDirectory;
}

export function modelArtifactFingerprint(cacheDirectory: string, modelId: string): string {
  const root = modelDirectory(cacheDirectory, modelId);
  const entries: string[] = [];
  try {
    for (const file of listFiles(root).sort()) {
      try {
        const relative = path.relative(root, file).split(path.sep).join("/");
        entries.push(`${relative}:${fs.statSync(file).size}`);
      } catch {}
    }
  } catch {
    return createHash("sha256").update(`${modelId}\0unreadable`).digest("hex");
  }
  const digest = createHash("sha256").update(modelId);
  for (const entry of entries) {
    digest.update("\0");
    digest.update(entry);
  }
  return digest.digest("hex");
}

function listFiles(root: string): string[] {
  const files: string[] = [];
  const walk = (directory: string): void => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(directory, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile()) files.push(full);
    }
  };
  walk(root);
  return files;
}

export class ProbeCache {
  readonly path: string;

  constructor(filePath: string) {
    this.path = filePath;
  }

  load(key: ProbeKey): ProbeRecord | undefined {
    const fingerprint = probeFingerprint(key);
    return this.records().find((record) => record.fingerprint === fingerprint);
  }

  store(
    key: ProbeKey,
    {
      batchSize,
      dimension,
      detail = "",
      charactersPerSecond = 0,
      loadNs = 0,
      limitedBy = "",
    }: {
      batchSize: number;
      dimension: number;
      detail?: string;
      charactersPerSecond?: number;
      loadNs?: number;
      limitedBy?: string;
    },
  ): void {
    const record: ProbeRecord = {
      fingerprint: probeFingerprint(key),
      batchSize,
      dimension,
      recordedAtNs: Date.now() * 1_000_000,
      detail,
      charactersPerSecond,
      loadNs,
      limitedBy,
    };
    this.guard(() => {
      const kept = this.records().filter((existing) => existing.fingerprint !== record.fingerprint);
      kept.push(record);
      kept.sort((left, right) => left.recordedAtNs - right.recordedAtNs);
      this.write(kept.slice(-MAX_RECORDS));
    });
  }

  state(key: ProbeKey): "hit" | "miss" {
    return this.load(key) === undefined ? "miss" : "hit";
  }

  private guard(body: () => void): void {
    let release: (() => void) | undefined;
    try {
      fs.mkdirSync(path.dirname(this.path), { recursive: true });
      if (!fs.existsSync(this.path)) fs.writeFileSync(this.path, "");
      release = lockfile.lockSync(this.path, {
        retries: { retries: 0 },
        stale: LOCK_TIMEOUT_SECONDS * 1000,
      });
    } catch {
      try {
        body();
      } catch {
        return;
      }
      return;
    }
    try {
      body();
    } finally {
      void release?.();
    }
  }

  private records(): ProbeRecord[] {
    let raw: unknown;
    try {
      raw = JSON.parse(fs.readFileSync(this.path, "utf8"));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      return [];
    }
    if (raw === null || typeof raw !== "object") return [];
    const payload = raw as { schema_version?: unknown; records?: unknown };
    if (payload.schema_version !== CACHE_SCHEMA_VERSION) return [];
    if (!Array.isArray(payload.records)) return [];
    return payload.records
      .map((entry) => probeRecordFromJson(entry))
      .filter((record): record is ProbeRecord => record !== undefined);
  }

  private write(records: ProbeRecord[]): void {
    const payload = {
      schema_version: CACHE_SCHEMA_VERSION,
      records: records.map(probeRecordToJson),
    };
    const temporary = `${this.path}.${process.pid}.tmp`;
    try {
      fs.mkdirSync(path.dirname(this.path), { recursive: true });
      fs.writeFileSync(temporary, JSON.stringify(payload));
      fs.renameSync(temporary, this.path);
    } catch {
      try {
        fs.unlinkSync(temporary);
      } catch {
        return;
      }
    }
  }
}
