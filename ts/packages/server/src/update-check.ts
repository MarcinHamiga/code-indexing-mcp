/**
 * Throttled "an update is available" check for managed installations.
 *
 * This runs on the serve path, so it is deliberately cheap: the local revision
 * is read straight out of `.git` rather than through a subprocess, the remote is
 * contacted at most once a day, and every failure -- no network, no git, a
 * corrupt cache -- is swallowed. An update check may never fail the command that
 * happened to trigger it.
 *
 * It is also silent for anything that is not a managed install: a development
 * checkout has no `code-indexing-mcp update` to run, so it must never be nagged.
 *
 * The one shape that differs from the Python original is how the check gets off
 * the hot path. Python needed a daemon thread because `subprocess.run` blocks;
 * here the remote call is already asynchronous, so `startBackgroundRefresh`
 * hands back an unawaited promise and the caller carries on. The seams the tests
 * drive -- the environment, the install root, the git runner, the clock -- are
 * explicit parameters rather than module attributes to monkeypatch.
 */

import { execFile, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { isRelativeTo, resolvePath } from "./paths.ts";

export const CACHE_FILENAME = "update-check.json";
/**
 * Once a day: the checkout only ever moves when a human runs an update, and the
 * notice is a convenience rather than a security signal.
 */
export const CHECK_INTERVAL_SECONDS = 86400;
/**
 * A hung network must not hold up an interactive command for longer than the
 * command itself would plausibly take.
 */
export const LS_REMOTE_TIMEOUT_SECONDS = 5;
export const DISABLE_VARIABLE = "CODE_INDEXING_UPDATE_CHECK";
/**
 * Bumped whenever a stored record's meaning changes. Records written by another
 * version are treated as absent rather than reinterpreted.
 */
export const CACHE_SCHEMA_VERSION = 1;

const DISABLED_VALUES = new Set(["off", "0", "false", "no"]);
const INSTALL_DIRECTORY_VARIABLE = "CODE_INDEXING_MCP_INSTALL_DIR";
const REMOTE_BRANCH_REF = "refs/heads/main";

type Environment = Readonly<Partial<Record<string, string>>>;

/** The subprocess seam, injectable so tests never need a network or a git. */
export type GitRunner = (
  command: readonly string[],
  cwd: string,
  timeoutSeconds: number,
) => Promise<{ stdout: string }>;

export interface UpdateStatus {
  readonly checkedAt: number;
  readonly localSha: string;
  readonly remoteSha: string;
}

export function updateAvailable(status: UpdateStatus): boolean {
  return status.localSha !== "" && status.remoteSha !== "" && status.localSha !== status.remoteSha;
}

/** Where this module looks to decide whether it is running from a managed install. */
export interface InstallContextOptions {
  readonly environment?: Environment;
  /**
   * The directory the running code was loaded from.
   *
   * Python asked `sys.prefix` -- the virtualenv root -- because the interpreter
   * is what an install owns there. A TypeScript install owns the *code*, so the
   * module's own location answers the same question more directly and without a
   * second indirection to get wrong.
   */
  readonly runtimeRoot?: string;
}

/**
 * Return the managed install directory, or `null` when this is not one.
 *
 * `null` turns every other entry point here into a no-op.
 */
export function installContext({
  environment = process.env,
  runtimeRoot = import.meta.dirname,
}: InstallContextOptions = {}): string | null {
  const configured = environment[INSTALL_DIRECTORY_VARIABLE] ?? "";
  try {
    // Kept in step with the installer's default install directory, which cannot
    // be imported here: the installer stays off the serve path.
    const directory = configured
      ? configured
      : path.join(os.homedir(), ".local", "share", "code-indexing-mcp");
    const resolved = resolvePath(directory);
    // A worktree's `.git` is a file, so existence is the test, not is-directory.
    if (!fs.existsSync(path.join(resolved, ".git"))) return null;
    if (!isRelativeTo(resolvePath(runtimeRoot), resolved)) return null;
    return resolved;
  } catch {
    return null;
  }
}

/**
 * Return the checked-out revision of *directory*, reading files first.
 *
 * `git rev-parse` is the last resort only: on the serve path this has to cost
 * microseconds, and the plain-file layout covers every case a managed install
 * can be in.
 */
export function checkoutHead(directory: string): string | null {
  try {
    const gitDirectory = findGitDirectory(directory);
    if (gitDirectory === null) return null;
    const head = fs.readFileSync(path.join(gitDirectory, "HEAD"), "utf8").trim();
    if (!head.startsWith("ref:")) return head === "" ? null : head;
    const reference = head.slice("ref:".length).trim();
    return referenceSha(gitDirectory, reference) ?? revParse(directory);
  } catch {
    return null;
  }
}

/** Return the cached status, treating anything unreadable as absent. */
export function readCache(cacheDirectory: string): UpdateStatus | null {
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(path.join(cacheDirectory, CACHE_FILENAME), "utf8"));
  } catch {
    return null;
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const record = raw as Record<string, unknown>;
  if (record.schema_version !== CACHE_SCHEMA_VERSION) return null;
  const checkedAt = Number(record.checked_at);
  if (record.checked_at === undefined || !Number.isFinite(checkedAt)) return null;
  if (record.local_sha === undefined || record.remote_sha === undefined) return null;
  return {
    checkedAt,
    localSha: String(record.local_sha),
    remoteSha: String(record.remote_sha),
  };
}

/** Persist *status* atomically, so a reader never sees a partial file. */
export function writeCache(cacheDirectory: string, status: UpdateStatus): void {
  // Snake-case keys and sorted order, because the Python build reads this file
  // too for as long as both are installed.
  const payload = JSON.stringify({
    checked_at: status.checkedAt,
    local_sha: status.localSha,
    remote_sha: status.remoteSha,
    schema_version: CACHE_SCHEMA_VERSION,
  });
  const target = path.join(cacheDirectory, CACHE_FILENAME);
  const temporary = `${target}.${process.pid}.tmp`;
  fs.mkdirSync(cacheDirectory, { recursive: true });
  fs.writeFileSync(temporary, payload, "utf8");
  fs.renameSync(temporary, target);
}

/** Ask the remote for the tip of main. Rejects when the check fails. */
export async function checkRemote(
  installDirectory: string,
  {
    timeoutSeconds = LS_REMOTE_TIMEOUT_SECONDS,
    runCommand = runGit,
    now = wallClockSeconds,
  }: { timeoutSeconds?: number; runCommand?: GitRunner; now?: () => number } = {},
): Promise<UpdateStatus> {
  // "origin" rather than a URL, so an install created with
  // CODE_INDEXING_MCP_REPO_URL keeps checking the remote it came from.
  const completed = await runCommand(
    ["git", "ls-remote", "origin", REMOTE_BRANCH_REF],
    installDirectory,
    timeoutSeconds,
  );
  const lines = completed.stdout.split(/\r?\n/).filter((line) => line.trim() !== "");
  const first = lines[0];
  if (first === undefined) {
    throw new Error(`no remote branch ${REMOTE_BRANCH_REF} at origin`);
  }
  return {
    checkedAt: now(),
    localSha: checkoutHead(installDirectory) ?? "",
    remoteSha: first.split(/\s+/)[0] ?? "",
  };
}

/** Re-check the remote when the cached answer has aged out. */
export async function refreshIfDue(
  installDirectory: string,
  cacheDirectory: string,
  {
    now,
    runCommand,
    environment = process.env,
  }: { now?: number; runCommand?: GitRunner; environment?: Environment } = {},
): Promise<void> {
  if (isDisabled(environment)) return;
  const moment = now ?? wallClockSeconds();
  const cached = readCache(cacheDirectory);
  if (cached !== null && moment - cached.checkedAt < CHECK_INTERVAL_SECONDS) return;
  try {
    const options = runCommand === undefined ? {} : { runCommand };
    writeCache(cacheDirectory, await checkRemote(installDirectory, options));
  } catch {
    // Missing git, no network, a timeout, an unwritable cache: none of them are
    // this caller's problem, and none may surface as its failure.
  }
}

/**
 * Start the refresh off the hot path, or return `null` when it is moot.
 *
 * The throttle is re-checked here on purpose: the point is to not start any work
 * at all on the overwhelmingly common already-checked-today path. The returned
 * promise is for tests and for a shutdown that wants to wait; production callers
 * drop it.
 */
export function startBackgroundRefresh(
  cacheDirectory: string,
  options: InstallContextOptions = {},
): Promise<void> | null {
  const environment = options.environment ?? process.env;
  if (isDisabled(environment)) return null;
  const installDirectory = installContext(options);
  if (installDirectory === null) return null;
  const cached = readCache(cacheDirectory);
  if (cached !== null && wallClockSeconds() - cached.checkedAt < CHECK_INTERVAL_SECONDS) {
    return null;
  }
  return refreshIfDue(installDirectory, cacheDirectory, { environment });
}

/**
 * Return the update message, or `null` when there is nothing to say.
 *
 * The cached remote is compared against the *live* head rather than the cached
 * one, so an update applied by any means -- the update command, a re-run of
 * install.sh, a manual pull -- silences this immediately.
 */
export function notice(cacheDirectory: string, options: InstallContextOptions = {}): string | null {
  const cached = readCache(cacheDirectory);
  if (cached === null) return null;
  const installDirectory = installContext(options);
  if (installDirectory === null) return null;
  const local = checkoutHead(installDirectory) ?? "";
  if (local === "" || cached.remoteSha === "" || local === cached.remoteSha) return null;
  return (
    `A code-indexing-mcp update is available ` +
    `(${local.slice(0, 7)} -> ${cached.remoteSha.slice(0, 7)}). ` +
    `Run: code-indexing-mcp update`
  );
}

function wallClockSeconds(): number {
  return Date.now() / 1000;
}

function isDisabled(environment: Environment): boolean {
  return DISABLED_VALUES.has((environment[DISABLE_VARIABLE] ?? "").trim().toLowerCase());
}

function findGitDirectory(directory: string): string | null {
  const candidate = path.join(directory, ".git");
  let info: fs.Stats;
  try {
    info = fs.statSync(candidate);
  } catch {
    return null;
  }
  if (info.isDirectory()) return candidate;
  if (!info.isFile()) return null;
  const content = fs.readFileSync(candidate, "utf8").trim();
  if (!content.startsWith("gitdir:")) return null;
  const target = content.slice("gitdir:".length).trim();
  return path.isAbsolute(target) ? target : path.join(directory, target);
}

function referenceSha(gitDirectory: string, reference: string): string | null {
  try {
    const sha = fs.readFileSync(path.join(gitDirectory, reference), "utf8").trim();
    if (sha !== "") return sha;
  } catch {
    // Packed instead of loose, or absent entirely; both fall through.
  }
  let packed: string;
  try {
    packed = fs.readFileSync(path.join(gitDirectory, "packed-refs"), "utf8");
  } catch {
    return null;
  }
  for (const line of packed.split(/\r?\n/)) {
    if (line.startsWith("#") || line.startsWith("^")) continue;
    const parts = line.split(/\s+/).filter((part) => part !== "");
    if (parts.length === 2 && parts[1] === reference) return parts[0] ?? null;
  }
  return null;
}

function revParse(directory: string): string | null {
  // Synchronous on purpose: this is the fallback inside `notice`, which callers
  // treat as a cheap string lookup and must not have to await.
  const completed = spawnSync("git", ["rev-parse", "HEAD"], {
    cwd: directory,
    encoding: "utf8",
    timeout: LS_REMOTE_TIMEOUT_SECONDS * 1000,
  });
  if (completed.error !== undefined || completed.status !== 0) return null;
  const head = (completed.stdout ?? "").trim();
  return head === "" ? null : head;
}

const runGit: GitRunner = (command, cwd, timeoutSeconds) =>
  new Promise((resolve, reject) => {
    const [executable, ...rest] = command;
    if (executable === undefined) {
      reject(new Error("empty git command"));
      return;
    }
    execFile(
      executable,
      rest,
      { cwd, timeout: timeoutSeconds * 1000, encoding: "utf8" },
      (error, stdout) => {
        if (error) reject(error);
        else resolve({ stdout });
      },
    );
  });
