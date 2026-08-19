/**
 * The installer-prepared accelerator record as the runtime sees it.
 *
 * A record nominates a backend but never proves it still works. The embedding
 * worker performs real inference before the backend is promoted for a run.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { type Accelerator, type BackendDescriptor, parseAccelerator } from "./backends.ts";

export const RECORD_SCHEMA_VERSION = 1;
export const RECORD_FILENAME = "accelerator.json";
export const RECORD_PATH_VARIABLE = "CODE_INDEXING_ACCEL_ENV";

export interface AcceleratorEnvironmentRecord {
  readonly accelerator: Accelerator;
  readonly interpreter: string;
  readonly providers: readonly string[];
  readonly runtimeVersion: string;
  readonly driverVersion: string;
  readonly device: string;
  // Retained for byte-compatible records written by the Python installer. In
  // the Bun build it identifies the Bun runtime that wrote the record.
  readonly pythonVersion: string;
  readonly recordedAtNs: string;
  readonly detail: string;
}

export interface AcceleratorEnvironmentStatus {
  readonly environment: AcceleratorEnvironmentRecord | null;
  readonly path: string;
  readonly reason: string | null;
  readonly providers: readonly string[];
}

export function runningRuntimeVersion(): string {
  return process.versions.bun ?? process.version;
}

export function recordPath(
  dataDirectory: string,
  environment: NodeJS.ProcessEnv = process.env,
): string {
  const configured = environment[RECORD_PATH_VARIABLE];
  if (!configured) return path.join(dataDirectory, RECORD_FILENAME);
  const candidate = expandHome(configured);
  try {
    return fs.statSync(candidate).isDirectory() ? path.join(candidate, RECORD_FILENAME) : candidate;
  } catch {
    return candidate;
  }
}

export function loadEnvironment(
  dataDirectory: string,
  environment: NodeJS.ProcessEnv = process.env,
): AcceleratorEnvironmentStatus {
  const filePath = recordPath(dataDirectory, environment);
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return emptyStatus(filePath);
    return emptyStatus(filePath, `the accelerator record is unreadable: ${error}`);
  }
  let record: AcceleratorEnvironmentRecord;
  try {
    record = recordFromJson(raw);
  } catch (error) {
    return emptyStatus(filePath, `the accelerator record is unusable: ${messageOf(error)}`);
  }
  const reason = verifyEnvironment(record);
  return reason === null
    ? { environment: record, path: filePath, reason: null, providers: record.providers }
    : emptyStatus(filePath, reason);
}

export function recordFromJson(value: unknown): AcceleratorEnvironmentRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("the record is not an object");
  }
  const raw = value as Record<string, unknown>;
  if (raw.schema_version !== RECORD_SCHEMA_VERSION) {
    throw new Error(
      `the record uses schema version ${JSON.stringify(raw.schema_version)}, ` +
        `this build reads version ${RECORD_SCHEMA_VERSION}`,
    );
  }
  const accelerator = parseRecordAccelerator(raw.accelerator);
  const interpreter = String(raw.interpreter ?? "");
  if (interpreter === "") throw new Error("the record names no interpreter");
  if (!Array.isArray(raw.providers) || raw.providers.length === 0) {
    throw new Error("the record lists no execution providers");
  }
  return {
    accelerator,
    interpreter,
    providers: raw.providers.map((provider) => String(provider)),
    runtimeVersion: String(raw.runtime_version ?? ""),
    driverVersion: String(raw.driver_version ?? ""),
    device: String(raw.device ?? ""),
    pythonVersion: String(raw.python_version ?? ""),
    recordedAtNs: String(raw.recorded_at_ns ?? "0"),
    detail: String(raw.detail ?? ""),
  };
}

export function recordToJson(record: AcceleratorEnvironmentRecord): Record<string, unknown> {
  return {
    schema_version: RECORD_SCHEMA_VERSION,
    accelerator: record.accelerator,
    interpreter: record.interpreter,
    providers: [...record.providers],
    runtime_version: record.runtimeVersion,
    driver_version: record.driverVersion,
    device: record.device,
    python_version: record.pythonVersion,
    recorded_at_ns: record.recordedAtNs,
    detail: record.detail,
  };
}

export function verifyEnvironment(record: AcceleratorEnvironmentRecord): string | null {
  try {
    if (!fs.statSync(record.interpreter).isFile()) throw new Error();
  } catch {
    return (
      `the accelerator environment's interpreter is gone (${record.interpreter}); ` +
      "reinstall to prepare it again"
    );
  }
  const running = runningRuntimeVersion();
  if (record.pythonVersion !== "" && record.pythonVersion !== running) {
    return (
      `the accelerator environment was built for runtime ${record.pythonVersion} ` +
      `and this server runs ${running}; reinstall to rebuild it`
    );
  }
  return null;
}

export function writeEnvironment(filePath: string, record: AcceleratorEnvironmentRecord): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify(recordToJson(record), null, 2)}\n`, "utf8");
  fs.renameSync(temporary, filePath);
}

export function clearEnvironment(filePath: string): boolean {
  try {
    fs.unlinkSync(filePath);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

export function applyEnvironment(
  descriptor: BackendDescriptor,
  record: AcceleratorEnvironmentRecord | null,
): BackendDescriptor {
  if (record === null || descriptor.accelerator !== record.accelerator) return descriptor;
  return new (descriptor.constructor as typeof BackendDescriptor)({
    accelerator: descriptor.accelerator,
    provider: descriptor.provider,
    device: record.device || descriptor.device,
    stability: descriptor.stability,
    precision: descriptor.precision,
    runtimeVersion: record.runtimeVersion || descriptor.runtimeVersion,
    driverVersion: record.driverVersion || descriptor.driverVersion,
    runtime: descriptor.runtime,
  });
}

function parseRecordAccelerator(value: unknown): Accelerator {
  const accelerator = parseAccelerator(String(value ?? ""));
  if (accelerator === "auto") {
    throw new Error("'auto' names a selection policy, not a prepared environment");
  }
  return accelerator;
}

function emptyStatus(path: string, reason: string | null = null): AcceleratorEnvironmentStatus {
  return { environment: null, path, reason, providers: [] };
}

function expandHome(candidate: string): string {
  return candidate === "~" || candidate.startsWith("~/")
    ? path.join(os.homedir(), candidate.slice(2))
    : candidate;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
