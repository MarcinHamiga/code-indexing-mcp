import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { preparedAccelerator, serverExecutable } from "./accelerator.ts";
import { commandFromEntry } from "./env-blocks.ts";
import { isBundledSkillLink, readServerEntry, skillDirectory } from "./harnesses.ts";
import {
  activationHint,
  LAUNCHER_NAME,
  type LauncherResult,
  launcherOk,
  launcherPath,
} from "./shell-path.ts";

export const HELP_TIMEOUT_SECONDS = 30;

export interface Check {
  readonly name: string;
  readonly status: "ok" | "warn" | "fail";
  readonly detail: string;
  readonly ok: boolean;
}

function check(name: string, status: Check["status"], detail = ""): Check {
  return { name, status, detail, ok: status === "ok" };
}

function whichOnPath(name: string, pathValue: string): string | null {
  for (const directory of pathValue.split(path.delimiter)) {
    if (directory === "") continue;
    for (const candidate of [name, `${name}.cmd`, `${name}.exe`]) {
      const filePath = path.join(directory, candidate);
      try {
        if (fs.statSync(filePath).isFile()) return filePath;
      } catch {}
    }
  }
  return null;
}

function serverRuns(installDirectory: string): Check {
  const executable = serverExecutable(installDirectory);
  try {
    if (!fs.statSync(executable).isFile()) {
      return check("server executable", "fail", `missing: ${executable}`);
    }
  } catch {
    return check("server executable", "fail", `missing: ${executable}`);
  }
  const completed = spawnSync(executable, ["--help"], {
    encoding: "utf8",
    timeout: HELP_TIMEOUT_SECONDS * 1000,
  });
  if (completed.error !== undefined) {
    if (completed.error.message.includes("ETIMEDOUT") || completed.signal === "SIGTERM") {
      return check(
        "server executable",
        "fail",
        `${executable} did not answer --help within ${HELP_TIMEOUT_SECONDS.toFixed(0)}s`,
      );
    }
    return check(
      "server executable",
      "fail",
      `${executable} could not be launched: ${completed.error.message}`,
    );
  }
  if (completed.status !== 0) {
    const message = (completed.stderr ?? "").trim().split(/\r?\n/);
    return check(
      "server executable",
      "fail",
      `${executable} --help exited ${completed.status}: ${message[message.length - 1] ?? ""}`,
    );
  }
  return check("server executable", "ok", executable);
}

function launcherResolves(
  launcher: LauncherResult | null,
  profilesUpdated: readonly string[],
  environment: NodeJS.ProcessEnv,
): Check {
  if (launcher === null) return check("command on PATH", "warn", "no launcher was requested");
  if (!launcherOk(launcher)) {
    return check("command on PATH", "warn", `launcher not created: ${launcher.detail}`);
  }
  const found = whichOnPath(LAUNCHER_NAME, environment.PATH ?? environment.Path ?? "");
  if (found !== null) {
    try {
      if (fs.realpathSync(found) === fs.realpathSync(launcher.path)) {
        return check("command on PATH", "ok", found);
      }
    } catch {
      // Fall through.
    }
  }
  if (profilesUpdated.length > 0) {
    const hint = activationHint(profilesUpdated, { environment });
    return check("command on PATH", "warn", `resolves once you start a new shell (${hint})`);
  }
  if (found !== null) {
    return check("command on PATH", "warn", `${found} is found first, not ${launcher.path}`);
  }
  return check(
    "command on PATH",
    "warn",
    `${path.dirname(launcher.path)} is not on PATH; add it to use the command by name`,
  );
}

function harnessEntries(
  configured: readonly [string, string][],
  installDirectory: string,
  options: { home?: string; environment?: NodeJS.ProcessEnv },
): Check[] {
  const expected = serverExecutable(installDirectory);
  const checks: Check[] = [];
  for (const [slug, filePath] of configured) {
    const name = `${slug} configuration`;
    let entry: Record<string, unknown> | null;
    try {
      entry = readServerEntry(slug, options);
    } catch (error) {
      checks.push(
        check(
          name,
          "warn",
          `could not be re-read: ${error instanceof Error ? error.message : String(error)}`,
        ),
      );
      continue;
    }
    if (entry === null) {
      checks.push(check(name, "warn", `no server entry found in ${filePath} after writing it`));
      continue;
    }
    const command = commandFromEntry(slug, entry);
    if (command === null)
      checks.push(check(name, "warn", `the entry in ${filePath} names no command`));
    else if (command !== expected) {
      checks.push(check(name, "warn", `the entry names ${command}, expected ${expected}`));
    } else {
      checks.push(check(name, "ok", filePath));
    }
  }
  return checks;
}

function acceleratorRecorded(): Check {
  const prepared = preparedAccelerator("");
  if (prepared === null) {
    return check("accelerator record", "warn", "no record; the server will resolve at startup");
  }
  return check("accelerator record", "ok", prepared);
}

function skillLinks(
  slugs: readonly string[],
  options: { home?: string; environment?: NodeJS.ProcessEnv },
): Check[] {
  const checks: Check[] = [];
  for (const slug of slugs) {
    const directory = skillDirectory(slug, options);
    if (directory === null || !fs.existsSync(directory) || !fs.statSync(directory).isDirectory()) {
      continue;
    }
    const broken = fs
      .readdirSync(directory)
      .map((name) => path.join(directory, name))
      .filter((entry) => {
        if (!isBundledSkillLink(entry)) return false;
        try {
          return !fs.statSync(fs.realpathSync(entry)).isDirectory();
        } catch {
          return true;
        }
      })
      .map((entry) => path.basename(entry));
    const name = `${slug} skills`;
    if (broken.length > 0) {
      checks.push(check(name, "warn", `links point nowhere: ${broken.sort().join(", ")}`));
    } else {
      checks.push(check(name, "ok", directory));
    }
  }
  return checks;
}

export function runChecks(
  installDirectory: string,
  configured: readonly [string, string][] = [],
  options: {
    launcher?: LauncherResult | null;
    profilesUpdated?: readonly string[];
    acceleratorWasPrepared?: boolean;
    home?: string;
    environment?: NodeJS.ProcessEnv;
  } = {},
): Check[] {
  const environment = options.environment ?? process.env;
  const checks = [serverRuns(installDirectory)];
  checks.push(
    launcherResolves(options.launcher ?? null, options.profilesUpdated ?? [], environment),
  );
  checks.push(...harnessEntries(configured, installDirectory, options));
  if (options.acceleratorWasPrepared !== false) checks.push(acceleratorRecorded());
  checks.push(
    ...skillLinks(
      configured.map(([slug]) => slug),
      options,
    ),
  );
  return checks;
}

export function runUpdateChecks(
  installDirectory: string,
  configured: readonly [string, string][] = [],
  options: {
    acceleratorWasPrepared?: boolean;
    home?: string;
    environment?: NodeJS.ProcessEnv;
  } = {},
): Check[] {
  const checks = [serverRuns(installDirectory)];
  checks.push(...harnessEntries(configured, installDirectory, options));
  if (options.acceleratorWasPrepared !== false) checks.push(acceleratorRecorded());
  checks.push(
    ...skillLinks(
      configured.map(([slug]) => slug),
      options,
    ),
  );
  return checks;
}

export function formatCheck(item: Check): string {
  const marker = { ok: "ok  ", warn: "warn", fail: "FAIL" }[item.status];
  const line = `${marker} - ${item.name}`;
  return item.detail !== "" ? `${line}: ${item.detail}` : line;
}

export { launcherPath };
