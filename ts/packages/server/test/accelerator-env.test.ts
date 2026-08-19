/** Prepared accelerator record discovery and validation. */

import { expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import {
  applyEnvironment,
  clearEnvironment,
  loadEnvironment,
  RECORD_FILENAME,
  RECORD_PATH_VARIABLE,
  recordPath,
  runningRuntimeVersion,
  writeEnvironment,
} from "../src/accelerator-env.ts";
import { backendFor } from "../src/backends.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

function record() {
  return {
    accelerator: "cuda" as const,
    interpreter: process.execPath,
    providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    runtimeVersion: "1.27.0",
    driverVersion: "550.54.14",
    device: "cuda:0",
    pythonVersion: runningRuntimeVersion(),
    recordedAtNs: "1",
    detail: "probed 2 passages on CUDAExecutionProvider",
  };
}

test("a valid record round trips through the installer path", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, "data", RECORD_FILENAME);
    const expected = record();
    writeEnvironment(filePath, expected);

    const status = loadEnvironment(path.join(directory, "data"));

    expect(status.environment).toEqual(expected);
    expect(status.reason).toBeNull();
    expect(status.providers).toEqual(expected.providers);
  } finally {
    removeDirectory(directory);
  }
});

test("a missing record is a normal CPU-only installation", () => {
  const directory = temporaryDirectory();
  try {
    const status = loadEnvironment(directory);
    expect(status.environment).toBeNull();
    expect(status.reason).toBeNull();
    expect(status.providers).toEqual([]);
  } finally {
    removeDirectory(directory);
  }
});

test("a stale runtime record is rejected rather than reinterpreted", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, RECORD_FILENAME);
    writeEnvironment(filePath, { ...record(), pythonVersion: "old-runtime" });

    expect(loadEnvironment(directory).reason).toContain("built for runtime old-runtime");
  } finally {
    removeDirectory(directory);
  }
});

test("a malformed or unsupported record reports a diagnostic", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, RECORD_FILENAME);
    fs.writeFileSync(filePath, JSON.stringify({ schema_version: 99 }), "utf8");

    expect(loadEnvironment(directory).reason).toContain("schema version");
  } finally {
    removeDirectory(directory);
  }
});

test("a record naming a vanished interpreter is refused with a reason", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, RECORD_FILENAME);
    writeEnvironment(filePath, {
      ...record(),
      interpreter: path.join(directory, "venv-accel", "python"),
    });

    expect(loadEnvironment(directory).reason).toContain("interpreter is gone");
  } finally {
    removeDirectory(directory);
  }
});

test("an unreadable record reports why rather than raising", () => {
  const directory = temporaryDirectory();
  try {
    fs.writeFileSync(path.join(directory, RECORD_FILENAME), "{not json", "utf8");

    expect(loadEnvironment(directory).reason).toContain("unreadable");
  } finally {
    removeDirectory(directory);
  }
});

test("a record without providers or naming auto is unusable", () => {
  const directory = temporaryDirectory();
  try {
    const filePath = path.join(directory, RECORD_FILENAME);
    const payload: Record<string, unknown> = {
      schema_version: 1,
      accelerator: "cuda",
      interpreter: process.execPath,
      providers: ["CUDAExecutionProvider", "CPUExecutionProvider"],
      runtime_version: "1.27.0",
      driver_version: "550.54.14",
      device: "cuda:0",
      python_version: runningRuntimeVersion(),
      recorded_at_ns: 1,
      detail: "probed 2 passages on CUDAExecutionProvider",
    };

    payload.providers = [];
    fs.writeFileSync(filePath, JSON.stringify(payload), "utf8");
    expect(loadEnvironment(directory).reason).toContain("no execution providers");

    payload.providers = ["CUDAExecutionProvider"];
    payload.accelerator = "auto";
    fs.writeFileSync(filePath, JSON.stringify(payload), "utf8");
    expect(loadEnvironment(directory).reason).toContain("selection policy");
  } finally {
    removeDirectory(directory);
  }
});

test("an explicit record directory overrides the data directory and clear is idempotent", () => {
  const directory = temporaryDirectory();
  try {
    const elsewhere = path.join(directory, "elsewhere");
    fs.mkdirSync(elsewhere);
    const filePath = path.join(elsewhere, RECORD_FILENAME);
    writeEnvironment(filePath, record());
    const environment = { [RECORD_PATH_VARIABLE]: elsewhere };

    expect(recordPath(path.join(directory, "data"), environment)).toBe(filePath);
    expect(loadEnvironment(path.join(directory, "data"), environment).environment).toEqual(
      record(),
    );
    expect(clearEnvironment(filePath)).toBe(true);
    expect(clearEnvironment(filePath)).toBe(false);
  } finally {
    removeDirectory(directory);
  }
});

test("a matching record supplies diagnostic runtime details only to its backend", () => {
  const cuda = backendFor("cuda");
  const cpu = backendFor("cpu");
  expect(cuda).toBeDefined();
  expect(cpu).toBeDefined();
  if (cuda === undefined || cpu === undefined) return;

  const described = applyEnvironment(cuda, record());

  expect(described.device).toBe("cuda:0");
  expect(described.runtimeVersion).toBe("1.27.0");
  expect(described.driverVersion).toBe("550.54.14");
  expect(applyEnvironment(cpu, record())).toBe(cpu);
});
