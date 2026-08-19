import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { atomicWrite, expandUser, InstallerError } from "./config-files.ts";
import { BY_NAME, defaultValue } from "./settings-spec.ts";

export const ACCELERATOR_CHOICES = [
  "auto",
  "cpu",
  "cuda",
  "mlx",
  "webgpu",
  "migraphx",
  "coreml",
] as const;

export const ACCELERATOR_PROBE_TARGETS = new Set(["cuda", "mlx", "webgpu", "migraphx"]);

const MINIMUM_NVIDIA_DRIVER: Record<string, readonly [number, number]> = {
  linux: [525, 60],
  win32: [527, 41],
};
const CUDA_PLATFORMS: Record<string, ReadonlySet<string>> = {
  linux: new Set(["x86_64"]),
  win32: new Set(["amd64"]),
};
const WEBGPU_PLATFORMS: Record<string, ReadonlySet<string>> = {
  darwin: new Set(["arm64"]),
  linux: new Set(["x86_64"]),
  win32: new Set(["amd64"]),
};
const MINIMUM_WEBGPU_MACOS = [14, 0] as const;
const MLX_PLATFORMS: Record<string, ReadonlySet<string>> = {
  darwin: new Set(["arm64"]),
};
const MINIMUM_MLX_MACOS = [14, 0] as const;
const MIGRAPHX_PLATFORM = ["linux", "x86_64"] as const;
const MINIMUM_MIGRAPHX_ROCM = [7, 2, 1] as const;
const ROCM_ROOT = "/opt/rocm";
const ROCM_VERSION_FILES = [".info/version", ".info/version-dev"] as const;

export interface AcceleratorPlan {
  readonly accelerator: string;
  readonly reason: string;
  readonly driverVersion: string;
  readonly deviceName: string;
  readonly honored: boolean;
  readonly lockFingerprint: string;
}

export function planPreparesEnvironment(plan: AcceleratorPlan): boolean {
  return ACCELERATOR_PROBE_TARGETS.has(plan.accelerator);
}

function makePlan(
  accelerator: string,
  reason: string,
  extras: Partial<Omit<AcceleratorPlan, "accelerator" | "reason">> = {},
): AcceleratorPlan {
  return {
    accelerator,
    reason,
    driverVersion: extras.driverVersion ?? "",
    deviceName: extras.deviceName ?? "",
    honored: extras.honored ?? true,
    lockFingerprint: extras.lockFingerprint ?? "",
  };
}

export function serverExecutable(
  installDirectory: string,
  platformName: string = process.platform,
): string {
  if (platformName.startsWith("win")) {
    return path.join(installDirectory, "bin", "code-indexing-mcp.cmd");
  }
  return path.join(installDirectory, "bin", "code-indexing-mcp");
}

export function cliEntry(installDirectory: string): string {
  return path.join(installDirectory, "ts", "packages", "server", "src", "cli.ts");
}

export function writeServerLauncher(
  installDirectory: string,
  platformName: string = process.platform,
): string {
  const executable = serverExecutable(installDirectory, platformName);
  const entry = cliEntry(installDirectory);
  const bun = process.execPath;
  fs.mkdirSync(path.dirname(executable), { recursive: true, mode: 0o700 });
  if (platformName.startsWith("win")) {
    const content = `@echo off\r\n"${bun}" "${entry}" %*\r\n`;
    fs.writeFileSync(executable, content);
  } else {
    const content = `#!/bin/sh\nexec "${bun}" "${entry}" "$@"\n`;
    fs.writeFileSync(executable, content, { mode: 0o755 });
    fs.chmodSync(executable, 0o755);
  }
  return executable;
}

function hostMachine(arch: string = os.arch()): string {
  if (arch === "x64") return "x86_64";
  if (arch === "ia32") return "x86";
  return arch;
}

function macosVersion(): string {
  const result = spawnSync("sw_vers", ["-productVersion"], { encoding: "utf8" });
  if (result.error !== undefined || result.status !== 0) return "";
  return result.stdout.trim();
}

function nvidiaSmiReport(): string | null {
  const executable = process.platform.startsWith("win") ? "nvidia-smi.exe" : "nvidia-smi";
  const result = spawnSync(
    executable,
    ["--query-gpu=driver_version,name", "--format=csv,noheader"],
    {
      encoding: "utf8",
      timeout: 30_000,
    },
  );
  if (result.error !== undefined || result.status !== 0) return null;
  return result.stdout;
}

function rocmVersionDirectories(root: string): string[] {
  const directories: string[] = [];
  const override = (process.env.ROCM_PATH ?? "").trim();
  if (override !== "") directories.push(override);
  directories.push(root, path.join(root, "core"));
  try {
    for (const name of fs.readdirSync(root).sort()) {
      if (name.startsWith("core-")) directories.push(path.join(root, name));
    }
  } catch {
    // No ROCm tree.
  }
  return directories;
}

function rocmVersion(root: string): string {
  for (const directory of rocmVersionDirectories(root)) {
    for (const name of ROCM_VERSION_FILES) {
      try {
        const contents = fs.readFileSync(path.join(directory, name), "utf8");
        const match = /\d+\.\d+(?:\.\d+)?/.exec(contents);
        if (match !== null) return match[0];
      } catch {}
    }
  }
  return "";
}

function rocminfoDeviceName(output: string): string {
  for (const block of output.split(/^Agent \d+/m).slice(1)) {
    if (!/^\s*Device Type:\s*GPU\s*$/m.test(block)) continue;
    const match = /^\s*Marketing Name:\s*(.+?)\s*$/m.exec(block);
    if (match !== null && match[1] !== undefined && match[1].trim().toLowerCase() !== "unknown") {
      return match[1].trim();
    }
  }
  return "";
}

function rocmReport(root: string = ROCM_ROOT): string | null {
  const version = rocmVersion(root);
  if (version === "") return null;
  let device = "";
  const result = spawnSync("rocminfo", [], { encoding: "utf8", timeout: 30_000 });
  if (result.error === undefined && result.status === 0) {
    device = rocminfoDeviceName(result.stdout);
  }
  return `${version}, ${device || "AMD GPU"}`;
}

function driverComponents(version: string): number[] {
  const components: number[] = [];
  for (const part of version.trim().split(".")) {
    if (!/^\d+$/.test(part)) break;
    components.push(Number(part));
  }
  return components;
}

function compareComponents(left: readonly number[], right: readonly number[]): number {
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    if (a !== b) return a < b ? -1 : 1;
  }
  return 0;
}

function normalizedPlatform(platformName: string): string {
  return platformName.startsWith("win") ? "win32" : platformName;
}

function webgpuPlan(options: {
  platformName: string;
  machine: string;
  platformVersion: string;
  reasonPrefix?: string;
}): AcceleratorPlan {
  const supported = WEBGPU_PLATFORMS[options.platformName];
  let problem = "";
  if (supported === undefined || !supported.has(options.machine)) {
    problem = `no native WebGPU plugin wheel is published for ${options.platformName}/${options.machine}`;
  } else if (options.platformName === "darwin") {
    const components = driverComponents(options.platformVersion);
    if (components.length === 0 || compareComponents(components, MINIMUM_WEBGPU_MACOS) < 0) {
      problem = `the locked WebGPU plugin requires macOS ${MINIMUM_WEBGPU_MACOS.join(".")} or newer`;
    }
  }
  if (problem !== "") {
    const prefix =
      options.reasonPrefix !== undefined
        ? `${options.reasonPrefix}; `
        : "WebGPU was requested but ";
    return makePlan("cpu", `${prefix}${problem}`, { honored: false });
  }
  const reason =
    options.reasonPrefix === undefined
      ? `the locked WebGPU plugin is available for ${options.platformName}/${options.machine}`
      : `${options.reasonPrefix}; falling back to WebGPU with the locked plugin`;
  return makePlan("webgpu", reason, { honored: options.reasonPrefix === undefined });
}

function mlxProblem(options: {
  platformName: string;
  machine: string;
  platformVersion: string;
}): string {
  const supported = MLX_PLATFORMS[options.platformName];
  if (supported === undefined || !supported.has(options.machine)) {
    return `MLX runs on Metal, and there is no Metal on ${options.platformName}/${options.machine}`;
  }
  const components = driverComponents(options.platformVersion);
  if (components.length === 0 || compareComponents(components, MINIMUM_MLX_MACOS) < 0) {
    return `the locked MLX build requires macOS ${MINIMUM_MLX_MACOS.join(".")} or newer`;
  }
  return "";
}

function mlxPlan(options: {
  platformName: string;
  machine: string;
  platformVersion: string;
}): AcceleratorPlan {
  const problem = mlxProblem(options);
  if (problem !== "") {
    return makePlan("cpu", `MLX was requested but ${problem}`, { honored: false });
  }
  return makePlan(
    "mlx",
    `the locked MLX build is available for macOS ${options.platformVersion} on ${options.machine}`,
    {
      driverVersion: options.platformVersion,
      deviceName: "Apple Silicon GPU",
    },
  );
}

export function planAccelerator(
  requested: string,
  options: {
    platformName?: string;
    machine?: string;
    nvidiaReport?: () => string | null;
    rocmReport?: () => string | null;
    platformVersion?: string;
  } = {},
): AcceleratorPlan {
  const platformName = normalizedPlatform((options.platformName ?? process.platform).toLowerCase());
  const machine = (options.machine ?? hostMachine()).toLowerCase();
  const platformVersion =
    options.platformVersion ?? (platformName === "darwin" ? macosVersion() : "");
  const nvidiaReport = options.nvidiaReport ?? nvidiaSmiReport;
  const rocm = options.rocmReport ?? rocmReport;
  const wanted = requested.trim().toLowerCase();

  if (wanted === "cpu") return makePlan("cpu", "CPU was requested");
  if (wanted === "coreml") {
    return makePlan(
      "cpu",
      "Core ML needs no separate environment and stays manual-only: it lost to " +
        "CPU on this model. Set CODE_INDEXING_EMBED_ACCELERATOR=coreml to measure it",
    );
  }
  if (wanted === "mlx") {
    return mlxPlan({ platformName, machine, platformVersion });
  }
  if (wanted === "webgpu") {
    return webgpuPlan({ platformName, machine, platformVersion });
  }
  if (wanted === "migraphx") {
    let problem = "";
    if (platformName !== MIGRAPHX_PLATFORM[0] || machine !== MIGRAPHX_PLATFORM[1]) {
      problem = `the pinned MIGraphX wheel is published only for ${MIGRAPHX_PLATFORM[0]}/${MIGRAPHX_PLATFORM[1]}`;
    } else {
      const report = rocm();
      if (report === null || report.trim() === "") {
        problem = "ROCm was not detected";
      } else {
        const first = report.trim().split(/\r?\n/)[0] ?? "";
        const [rocmVersionRaw, deviceRaw] = first.split(",");
        const rocmVersion = (rocmVersionRaw ?? "").trim();
        const deviceName = (deviceRaw ?? "").trim();
        const components = driverComponents(rocmVersion);
        const minimum = MINIMUM_MIGRAPHX_ROCM.join(".");
        if (
          components.length === 0 ||
          components[0] !== MINIMUM_MIGRAPHX_ROCM[0] ||
          compareComponents(components, MINIMUM_MIGRAPHX_ROCM) < 0
        ) {
          problem =
            `ROCm ${rocmVersion || "unknown"} is outside the ${minimum}+ ` +
            `support window this release's MIGraphX runtime was built ` +
            `against (ROCm ${MINIMUM_MIGRAPHX_ROCM[0]} only)`;
        } else {
          return makePlan(
            "migraphx",
            `ROCm ${rocmVersion} on ${deviceName || "an AMD device"} is within this release's MIGraphX support window`,
            { driverVersion: rocmVersion, deviceName },
          );
        }
      }
    }
    return webgpuPlan({
      platformName,
      machine,
      platformVersion,
      reasonPrefix: `MIGraphX was requested but ${problem}`,
    });
  }

  if (wanted !== "cuda" && mlxProblem({ platformName, machine, platformVersion }) === "") {
    return mlxPlan({ platformName, machine, platformVersion });
  }

  const supported = CUDA_PLATFORMS[platformName];
  const explicit = wanted === "cuda" ? "CUDA was requested but " : "";
  const honored = explicit === "";
  if (supported === undefined || !supported.has(machine)) {
    return makePlan(
      "cpu",
      `${explicit}no CUDA wheels are published for ${platformName}/${machine}`,
      {
        honored,
      },
    );
  }
  const report = nvidiaReport();
  if (report === null || report.trim() === "") {
    return makePlan(
      "cpu",
      `${explicit}no usable NVIDIA driver was detected (nvidia-smi reported nothing)`,
      { honored },
    );
  }
  const first = report.trim().split(/\r?\n/)[0] ?? "";
  const [driverRaw, deviceRaw] = first.split(",");
  const driverVersion = (driverRaw ?? "").trim();
  const deviceName = (deviceRaw ?? "").trim();
  const floor = MINIMUM_NVIDIA_DRIVER[platformName];
  const components = driverComponents(driverVersion);
  if (
    floor !== undefined &&
    (components.length === 0 || compareComponents(components, floor) < 0)
  ) {
    return makePlan(
      "cpu",
      `${explicit}NVIDIA driver ${driverVersion || "unknown"} is below the ` +
        `${floor.join(".")} this release's CUDA 12 runtime needs; the installer does not change drivers`,
      { driverVersion, deviceName, honored },
    );
  }
  return makePlan(
    "cuda",
    `NVIDIA driver ${driverVersion} on ${deviceName || "an NVIDIA device"} satisfies the pinned CUDA 12 runtime`,
    { driverVersion, deviceName },
  );
}

export function acceleratorLockFingerprint(installDirectory: string, accelerator: string): string {
  const lockfile = path.join(installDirectory, "ts", "bun.lock");
  let locked: Buffer;
  try {
    locked = fs.readFileSync(lockfile);
  } catch (error) {
    throw new InstallerError(
      `The accelerator lockfile cannot be read at ${lockfile}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return createHash("sha256").update(accelerator).update("\0").update(locked).digest("hex");
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

export function dataDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  const configured = environment.CODE_INDEXING_DATA_DIR;
  if (configured !== undefined && configured !== "") return path.resolve(expandUser(configured));
  return path.resolve(path.join(defaultDataHome(), "code-indexing-mcp"));
}

export function cacheDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  const configured = environment.CODE_INDEXING_CACHE_DIR;
  if (configured !== undefined && configured !== "") return path.resolve(expandUser(configured));
  const setting = BY_NAME.CODE_INDEXING_CACHE_DIR;
  if (setting !== undefined) return path.resolve(defaultValue(setting));
  const home =
    process.platform === "darwin"
      ? path.join(os.homedir(), "Library", "Caches")
      : (process.env.XDG_CACHE_HOME ?? path.join(os.homedir(), ".cache"));
  return path.resolve(path.join(home, "code-indexing-mcp"));
}

export function acceleratorRecordPath(environment: NodeJS.ProcessEnv = process.env): string {
  const configured = environment.CODE_INDEXING_ACCEL_ENV;
  if (configured !== undefined && configured !== "") {
    const candidate = expandUser(configured);
    try {
      return fs.statSync(candidate).isDirectory()
        ? path.join(candidate, "accelerator.json")
        : candidate;
    } catch {
      return candidate;
    }
  }
  return path.join(dataDirectory(environment), "accelerator.json");
}

export function preparedAccelerator(_installDirectory: string): string | null {
  try {
    const payload = JSON.parse(fs.readFileSync(acceleratorRecordPath(), "utf8")) as {
      accelerator?: unknown;
    };
    return typeof payload.accelerator === "string" && payload.accelerator !== ""
      ? payload.accelerator
      : null;
  } catch {
    return null;
  }
}

export function clearAcceleratorRecord(filePath: string): boolean {
  try {
    fs.unlinkSync(filePath);
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return false;
    throw new InstallerError(
      `Could not remove the stale accelerator record: ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function writeAcceleratorRecord(
  filePath: string,
  plan: AcceleratorPlan,
  probe: {
    interpreter: string;
    providers: readonly string[];
    runtimeVersion: string;
    device: string;
    pythonVersion: string;
    recordedAtNs: string;
    detail: string;
  },
): void {
  const record = {
    schema_version: 1,
    accelerator: plan.accelerator,
    interpreter: probe.interpreter,
    providers: [...probe.providers],
    runtime_version: probe.runtimeVersion,
    lock_fingerprint: plan.lockFingerprint,
    driver_version: plan.driverVersion,
    device: probe.device,
    python_version: probe.pythonVersion,
    recorded_at_ns: probe.recordedAtNs,
    detail: probe.detail,
  };
  atomicWrite(filePath, `${JSON.stringify(record, null, 2)}\n`);
}

function reusableRecord(filePath: string, plan: AcceleratorPlan, runtimeVersion: string): boolean {
  try {
    const record = JSON.parse(fs.readFileSync(filePath, "utf8")) as Record<string, unknown>;
    return (
      record.schema_version === 1 &&
      record.accelerator === plan.accelerator &&
      String(record.lock_fingerprint ?? "") === plan.lockFingerprint &&
      String(record.driver_version ?? "") === plan.driverVersion &&
      String(record.python_version ?? "") === runtimeVersion &&
      typeof record.interpreter === "string" &&
      fs.existsSync(record.interpreter)
    );
  } catch {
    return false;
  }
}

export async function configureAccelerator(
  installDirectory: string,
  requested: string,
  options: {
    platformName?: string;
    machine?: string;
    nvidiaReport?: () => string | null;
    rocmReport?: () => string | null;
    platformVersion?: string;
    offline?: boolean;
    probe?: (
      accelerator: string,
      offline: boolean,
    ) => Promise<{
      interpreter: string;
      providers: readonly string[];
      runtimeVersion: string;
      device: string;
      pythonVersion: string;
      recordedAtNs: string;
      detail: string;
    }>;
  } = {},
): Promise<AcceleratorPlan> {
  writeServerLauncher(installDirectory, options.platformName);
  let plan = planAccelerator(requested, options);
  const record = acceleratorRecordPath();
  if (!planPreparesEnvironment(plan)) {
    clearAcceleratorRecord(record);
    return plan;
  }
  const runtimeVersion = process.versions.bun ?? process.version;
  try {
    plan = {
      ...plan,
      lockFingerprint: acceleratorLockFingerprint(installDirectory, plan.accelerator),
    };
    if (reusableRecord(record, plan, runtimeVersion)) {
      return { ...plan, reason: `${plan.reason}; reusing the recorded ${plan.accelerator} probe` };
    }
    const probe =
      options.probe ??
      (async (accelerator, offline) => {
        const module = await import("../../server/src/accelerator-probe.ts");
        return module.probeAccelerator(accelerator as never, {
          cacheDirectory: cacheDirectory(),
          offline,
        });
      });
    const report = await probe(plan.accelerator, options.offline === true);
    writeAcceleratorRecord(record, plan, report);
    return { ...plan, reason: `${plan.reason}; ${report.detail || "probe passed"}` };
  } catch (error) {
    clearAcceleratorRecord(record);
    return makePlan(
      "cpu",
      `${plan.accelerator} was detected but could not be prepared: ${error instanceof Error ? error.message : String(error)}`,
      {
        driverVersion: plan.driverVersion,
        deviceName: plan.deviceName,
        honored: false,
      },
    );
  }
}
