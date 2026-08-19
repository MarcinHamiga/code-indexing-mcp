import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import {
  ACCELERATOR_PROBE_TARGETS,
  acceleratorLockFingerprint,
  acceleratorRecordPath,
  configureAccelerator,
  preparedAccelerator,
  serverExecutable,
  writeServerLauncher,
} from "./accelerator.ts";
import { canonicalRepositoryUrl, DEFAULT_REPOSITORY_URL } from "./bootstrap.ts";
import { InstallerError, resolveExisting } from "./config-files.ts";
import { commandFromEntry } from "./env-blocks.ts";
import { configurationPath, installSkills, readServerEntry, skillDirectory } from "./harnesses.ts";
import { defaultInstallDirectory } from "./orchestrator.ts";
import { formatCheck, runUpdateChecks } from "./verify.ts";
import { loadPrefill } from "./wizard.ts";

export const UPDATE_BRANCH = "main";
export const UPDATE_LOCK_NAME = ".update.lock";
export const CHECK_UPDATE_AVAILABLE_EXIT = 10;
const LOG_LINE_LIMIT = 15;
const DIRTY_PATH_LIMIT = 5;
const LS_REMOTE_TIMEOUT_SECONDS = 10;

export type RunCommand = (
  arguments_: string[],
  options?: { cwd?: string },
) => { stdout: string; status: number };
export type Spawn = (argv: string[], cwd: string) => number;

function defaultRunCommand(
  arguments_: string[],
  options: { cwd?: string } = {},
): { stdout: string; status: number } {
  const result = spawnSync(arguments_[0] ?? "", arguments_.slice(1), {
    cwd: options.cwd,
    encoding: "utf8",
  });
  if (result.error !== undefined) {
    throw new InstallerError(`Required command was not found: ${arguments_[0]}`);
  }
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || "").trim();
    throw new InstallerError(
      detail === ""
        ? `Command failed: ${arguments_.join(" ")}`
        : `Command failed: ${arguments_.join(" ")}\n${detail}`,
    );
  }
  return { stdout: result.stdout, status: result.status ?? 0 };
}

function defaultSpawn(argv: string[], cwd: string): number {
  const result = spawnSync(argv[0] ?? "", argv.slice(1), { cwd, stdio: "inherit" });
  if (result.error !== undefined) {
    process.stderr.write(
      `Error: could not start the updated code to finish the update: ${result.error.message}\n`,
    );
    return 1;
  }
  return result.status ?? 1;
}

function error(message: string): number {
  process.stderr.write(`Error: ${message}\n`);
  return 1;
}

function printStatus(step: string, status: string, detail: string): void {
  const stream = status === "warning" ? process.stderr : process.stdout;
  stream.write(`[${step}] ${status}: ${detail}\n`);
}

function git(runCommand: RunCommand, directory: string, ...arguments_: string[]): string {
  return runCommand(["git", ...arguments_], { cwd: directory }).stdout.trim();
}

function fastForwardPossible(directory: string): boolean {
  const result = spawnSync("git", ["merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"], {
    cwd: directory,
    encoding: "utf8",
  });
  return result.status === 0;
}

function acquireLock(directory: string): () => void {
  const lockPath = path.join(directory, UPDATE_LOCK_NAME);
  try {
    const handle = fs.openSync(lockPath, "wx");
    fs.writeFileSync(handle, String(process.pid));
    fs.closeSync(handle);
  } catch {
    throw new InstallerError(
      `another update is already running in ${directory}; wait for it to finish`,
    );
  }
  return () => {
    fs.rmSync(lockPath, { force: true });
  };
}

export async function updateMain(options: {
  installDir?: string | null;
  check?: boolean;
  skipAccelerator?: boolean;
  finalize?: boolean;
  previousSha?: string | null;
  runCommand?: RunCommand;
  spawn?: Spawn;
}): Promise<number> {
  const directory =
    options.installDir != null
      ? resolveExisting(options.installDir)
      : resolveExisting(defaultInstallDirectory());
  const runCommand = options.runCommand ?? defaultRunCommand;
  const spawn = options.spawn ?? defaultSpawn;
  if (options.finalize === true) {
    return finalizeMain(directory, {
      previousSha: options.previousSha ?? null,
      skipAccelerator: options.skipAccelerator === true,
      runCommand,
    });
  }
  if (options.check === true) return checkMain(directory, runCommand);
  return phase1(directory, {
    skipAccelerator: options.skipAccelerator === true,
    runCommand,
    spawn,
  });
}

function preflight(directory: string, runCommand: RunCommand): [string, string] {
  if (!fs.existsSync(path.join(directory, ".git"))) {
    throw new InstallerError(
      `${directory} is not a git checkout; reinstall with install.sh to update it with this command`,
    );
  }
  const origin = git(runCommand, directory, "remote", "get-url", "origin");
  const expected = process.env.CODE_INDEXING_MCP_REPO_URL ?? DEFAULT_REPOSITORY_URL;
  if (canonicalRepositoryUrl(origin) !== canonicalRepositoryUrl(expected)) {
    throw new InstallerError(
      `the checkout at ${directory} tracks ${origin}, not ${expected}; update it manually or point CODE_INDEXING_MCP_REPO_URL at the remote it came from`,
    );
  }
  const branch = git(runCommand, directory, "rev-parse", "--abbrev-ref", "HEAD");
  if (branch !== UPDATE_BRANCH) {
    throw new InstallerError(
      `the checkout at ${directory} is on branch ${branch}, not ${UPDATE_BRANCH}; switch it back with \`git switch ${UPDATE_BRANCH}\` before updating`,
    );
  }
  const status = git(runCommand, directory, "status", "--porcelain", "--untracked-files=no");
  const dirty = status === "" ? [] : status.split(/\r?\n/);
  if (dirty.length > 0) {
    const names = dirty
      .slice(0, DIRTY_PATH_LIMIT)
      .map((line) => line.trim().split(/\s+/).at(-1) ?? line)
      .join(", ");
    const extra =
      dirty.length > DIRTY_PATH_LIMIT ? ` and ${dirty.length - DIRTY_PATH_LIMIT} more` : "";
    throw new InstallerError(
      `the checkout at ${directory} has uncommitted changes (${names}${extra}); commit, stash, or discard them before updating`,
    );
  }
  try {
    runCommand(["git", "fetch", "origin", UPDATE_BRANCH], { cwd: directory });
  } catch (cause) {
    throw new InstallerError(
      `could not fetch origin/${UPDATE_BRANCH}; check the network connection and try again: ${cause instanceof Error ? cause.message : String(cause)}`,
    );
  }
  const head = git(runCommand, directory, "rev-parse", "HEAD");
  const remote = git(runCommand, directory, "rev-parse", "FETCH_HEAD");
  if (head !== remote && !fastForwardPossible(directory)) {
    throw new InstallerError(
      `the checkout at ${directory} has diverged from origin/${UPDATE_BRANCH}; update it manually or reinstall`,
    );
  }
  return [head, remote];
}

function phase1(
  directory: string,
  options: { skipAccelerator: boolean; runCommand: RunCommand; spawn: Spawn },
): number {
  if (!fs.existsSync(serverExecutable(directory))) {
    return error(`no installation found at ${directory}`);
  }
  let release: (() => void) | null = null;
  try {
    release = acquireLock(directory);
    let head: string;
    let remote: string;
    try {
      [head, remote] = preflight(directory, options.runCommand);
    } catch (cause) {
      return error(cause instanceof Error ? cause.message : String(cause));
    }
    if (head === remote) process.stdout.write("Already up to date.\n");
    else {
      try {
        options.runCommand(["git", "merge", "--ff-only", "FETCH_HEAD"], { cwd: directory });
      } catch (cause) {
        return error(
          `could not fast-forward the checkout at ${directory}: ${cause instanceof Error ? cause.message : String(cause)}`,
        );
      }
    }
    try {
      options.runCommand([process.execPath, "install", "--frozen-lockfile"], {
        cwd: path.join(directory, "ts"),
      });
    } catch (cause) {
      return error(
        `${cause instanceof Error ? cause.message : String(cause)}\nthe checkout was updated but its environment was not; re-run \`code-indexing-mcp update\``,
      );
    }
    writeServerLauncher(directory);
    const argv = [
      process.execPath,
      path.join(directory, "ts", "packages", "installer", "src", "command.ts"),
      "update",
      "--finalize",
      "--previous-sha",
      head,
      "--install-dir",
      directory,
    ];
    if (options.skipAccelerator) argv.push("--skip-accelerator");
    return options.spawn(argv, directory);
  } catch (cause) {
    return error(cause instanceof Error ? cause.message : String(cause));
  } finally {
    release?.();
  }
}

async function reconcileAccelerator(
  directory: string,
  skipAccelerator: boolean,
): Promise<{ status: string; detail: string; rebuilt: boolean; prepared: string | null }> {
  const prepared = preparedAccelerator(directory);
  if (prepared === null) {
    return {
      status: "skipped",
      detail: "no accelerator environment is recorded",
      rebuilt: false,
      prepared: null,
    };
  }
  if (!ACCELERATOR_PROBE_TARGETS.has(prepared)) {
    return {
      status: "skipped",
      detail: `the recorded ${prepared} runtime needs no separate environment`,
      rebuilt: false,
      prepared: null,
    };
  }
  let expected: string;
  try {
    expected = acceleratorLockFingerprint(directory, prepared);
  } catch (cause) {
    return {
      status: "warning",
      detail: `the ${prepared} environment could not be checked: ${cause instanceof Error ? cause.message : String(cause)}`,
      rebuilt: false,
      prepared,
    };
  }
  let recorded = "";
  try {
    const payload = JSON.parse(fs.readFileSync(acceleratorRecordPath(), "utf8")) as {
      lock_fingerprint?: unknown;
    };
    recorded = String(payload.lock_fingerprint ?? "");
  } catch {
    recorded = "";
  }
  if (recorded === expected) {
    return {
      status: "skipped",
      detail: `the ${prepared} environment is unchanged since the last build`,
      rebuilt: false,
      prepared,
    };
  }
  if (skipAccelerator) {
    return {
      status: "warning",
      detail: `the ${prepared} environment was resolved from an older lockfile and keeps serving as it is; rebuild it with \`code-indexing-mcp configure --accelerator ${prepared}\``,
      rebuilt: false,
      prepared,
    };
  }
  try {
    const plan = await configureAccelerator(directory, prepared);
    return {
      status: plan.honored ? "ok" : "warning",
      detail: `${plan.accelerator} (${plan.reason})`,
      rebuilt: true,
      prepared: ACCELERATOR_PROBE_TARGETS.has(plan.accelerator) ? plan.accelerator : null,
    };
  } catch (cause) {
    return {
      status: "warning",
      detail: `the ${prepared} environment could not be rebuilt: ${cause instanceof Error ? cause.message : String(cause)}`,
      rebuilt: false,
      prepared,
    };
  }
}

async function finalizeMain(
  directory: string,
  options: { previousSha: string | null; skipAccelerator: boolean; runCommand: RunCommand },
): Promise<number> {
  const head = spawnSync("git", ["rev-parse", "HEAD"], {
    cwd: directory,
    encoding: "utf8",
  }).stdout.trim();
  const accelerator = await reconcileAccelerator(directory, options.skipAccelerator);
  printStatus("accelerator", accelerator.status, accelerator.detail);
  const expectedCommand = serverExecutable(directory);
  const configuredSlugs = [...loadPrefill().configuredSlugs];
  const owned: Record<string, boolean> = {};
  for (const slug of configuredSlugs) {
    const entry = readServerEntry(slug);
    const command = entry === null ? null : commandFromEntry(slug, entry);
    owned[slug] = command !== null && command === expectedCommand;
  }
  const skillGroups = new Map<string, string[]>();
  for (const slug of configuredSlugs) {
    const skills = skillDirectory(slug);
    if (skills === null) continue;
    let groupKey = skills;
    try {
      groupKey = fs.realpathSync(skills);
    } catch {
      groupKey = path.resolve(skills);
    }
    const group = skillGroups.get(groupKey) ?? [];
    group.push(slug);
    skillGroups.set(groupKey, group);
  }
  const installSlugs: string[] = [];
  for (const [skills, group] of skillGroups) {
    const ours = group.filter((slug) => owned[slug] === true);
    const others = group.filter((slug) => owned[slug] !== true);
    if (ours.length > 0 && others.length > 0) {
      for (const slug of ours) {
        printStatus(
          "skills",
          "warn",
          `${slug}: skipped: ${skills} is shared with ${others.join(", ")} configured for another installation`,
        );
      }
    } else {
      installSlugs.push(...ours);
    }
  }
  const ownedSlugs = configuredSlugs.filter((slug) => owned[slug] === true);
  const configured = ownedSlugs.map((slug) => [slug, configurationPath(slug)] as [string, string]);
  for (const [slug, message] of installSkills(installSlugs, directory)) {
    printStatus("skills", message.startsWith("skipped:") ? "warn" : "ok", `${slug}: ${message}`);
  }
  const checks = runUpdateChecks(directory, configured, {
    acceleratorWasPrepared: accelerator.prepared !== null,
  });
  for (const check of checks) {
    (check.ok ? process.stdout : process.stderr).write(`${formatCheck(check)}\n`);
  }
  if (options.previousSha !== null && options.previousSha !== head) {
    process.stdout.write(`Updated ${options.previousSha.slice(0, 7)} -> ${head.slice(0, 7)}\n`);
    try {
      const lines = git(
        options.runCommand,
        directory,
        "log",
        "--oneline",
        "--no-decorate",
        `${options.previousSha}..HEAD`,
      ).split(/\r?\n/);
      for (const line of lines.slice(0, LOG_LINE_LIMIT)) process.stdout.write(`  ${line}\n`);
      if (lines.length > LOG_LINE_LIMIT) {
        process.stdout.write(`  ... and ${lines.length - LOG_LINE_LIMIT} more commits\n`);
      }
    } catch {
      // Summary is best-effort.
    }
  } else {
    process.stdout.write(
      `Already at ${head.slice(0, 7)}; the environment was re-synced and re-checked.\n`,
    );
  }
  process.stdout.write(`Accelerator: ${accelerator.detail}\n`);
  process.stdout.write("Restart your MCP clients to load the updated server.\n");
  return checks.some((check) => check.status === "fail" && check.name === "server executable")
    ? 1
    : 0;
}

function checkMain(directory: string, runCommand: RunCommand): number {
  if (!fs.existsSync(path.join(directory, ".git"))) {
    return error(`${directory} is not a git checkout; there is no update to check for`);
  }
  let origin: string;
  try {
    origin = git(runCommand, directory, "remote", "get-url", "origin");
  } catch (cause) {
    return error(
      `${directory} has no origin remote to check against: ${cause instanceof Error ? cause.message : String(cause)}`,
    );
  }
  const expected = process.env.CODE_INDEXING_MCP_REPO_URL ?? DEFAULT_REPOSITORY_URL;
  if (canonicalRepositoryUrl(origin) !== canonicalRepositoryUrl(expected)) {
    return error(`the checkout at ${directory} tracks ${origin}, not ${expected}`);
  }
  const result = spawnSync("git", ["ls-remote", "origin", "refs/heads/main"], {
    cwd: directory,
    encoding: "utf8",
    timeout: LS_REMOTE_TIMEOUT_SECONDS * 1000,
  });
  if (result.error !== undefined || result.status !== 0) {
    return error(
      `could not reach origin to check for updates: ${result.error?.message ?? result.stderr}`,
    );
  }
  const lines = result.stdout.split(/\r?\n/).filter((line) => line.trim() !== "");
  if (lines.length === 0) return error("origin has no refs/heads/main to compare against");
  const remoteSha = (lines[0] ?? "").split(/\s+/)[0] ?? "";
  const localSha = spawnSync("git", ["rev-parse", "HEAD"], {
    cwd: directory,
    encoding: "utf8",
  }).stdout.trim();
  process.stdout.write(
    `${JSON.stringify(
      {
        install_dir: directory,
        local_sha: localSha,
        remote_sha: remoteSha,
        update_available: localSha !== remoteSha,
      },
      null,
      2,
    )}\n`,
  );
  return localSha !== remoteSha ? CHECK_UPDATE_AVAILABLE_EXIT : 0;
}
