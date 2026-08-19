import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expandUser, resolveExisting } from "./config-files.ts";
import {
  configurationPath,
  deconfigureSelectedHarnesses,
  parseHarnessSelection,
  removeSkills,
} from "./harnesses.ts";
import { isUnder } from "./links.ts";
import { defaultInstallDirectory, type StepEvent } from "./orchestrator.ts";
import { BY_NAME, defaultValue } from "./settings-spec.ts";
import {
  defaultBinDirectory,
  launcherPath,
  removeLauncher,
  removePathBlock,
  shellProfiles,
} from "./shell-path.ts";
import { loadPrefill } from "./wizard.ts";

export interface UninstallPlan {
  readonly installDirectory: string;
  readonly harnessSlugs: readonly string[];
  readonly binDirectory: string | null;
  readonly removeLauncher: boolean;
  readonly removePathBlock: boolean;
  readonly removeData: boolean;
  readonly removeCheckout: boolean;
}

export interface UninstallResult {
  harnessesCleared: readonly [string, string, boolean][];
  skills: readonly [string, string][];
  launcherRemoved: string | null;
  profilesCleared: readonly string[];
  directoriesRemoved: readonly string[];
  failures: [string, string][];
}

export function dataDirectories(environment: NodeJS.ProcessEnv = process.env): string[] {
  const directories: string[] = [];
  for (const name of ["CODE_INDEXING_DATA_DIR", "CODE_INDEXING_CACHE_DIR"] as const) {
    const setting = BY_NAME[name];
    if (setting === undefined) continue;
    const configured = environment[name] || defaultValue(setting);
    if (configured !== "") directories.push(expandUser(configured));
  }
  return directories;
}

const DATA_MARKERS = [
  "lancedb",
  "locks",
  "staging",
  "accelerator.json",
  "daemon.token",
  "daemon.log",
  "models",
];

function refuseReason(
  directory: string,
  options: { checkout: boolean; home?: string },
): string | null {
  const home = path.resolve(options.home ?? os.homedir());
  let resolved: string;
  try {
    resolved = fs.realpathSync(directory);
  } catch (error) {
    return `cannot be resolved (${error instanceof Error ? error.message : String(error)})`;
  }
  if (path.dirname(resolved) === resolved) return "is a filesystem root";
  if (resolved === home) return "is your home directory";
  if (isUnder(home, resolved)) return "contains your home directory";
  if (options.checkout) {
    if (!fs.existsSync(path.join(resolved, "pyproject.toml"))) {
      return "does not look like a code-indexing-mcp checkout (no pyproject.toml)";
    }
    if (
      !fs.existsSync(path.join(resolved, "src", "code_indexing_mcp")) &&
      !fs.existsSync(path.join(resolved, "ts", "packages", "server"))
    ) {
      return "does not look like a code-indexing-mcp checkout (no src/code_indexing_mcp)";
    }
    return null;
  }
  if (path.basename(resolved) === "code-indexing-mcp") return null;
  if (DATA_MARKERS.some((marker) => fs.existsSync(path.join(resolved, marker)))) return null;
  return "holds no code-indexing-mcp index or cache, so it is not ours to delete";
}

export function runUninstall(
  plan: UninstallPlan,
  onEvent: (event: StepEvent) => void = () => undefined,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): UninstallResult {
  const result: UninstallResult = {
    harnessesCleared: [],
    skills: [],
    launcherRemoved: null,
    profilesCleared: [],
    directoriesRemoved: [],
    failures: [],
  };

  onEvent({
    step: "harnesses",
    status: "started",
    detail: plan.harnessSlugs.join(", ") || "none selected",
  });
  const [cleared, failures] = deconfigureSelectedHarnesses(plan.harnessSlugs, options);
  result.harnessesCleared = cleared;
  result.failures.push(...failures);
  for (const [slug, filePath, changed] of cleared) {
    onEvent({
      step: "harnesses",
      status: changed ? "finished" : "skipped",
      detail: `${slug}: ${changed ? "removed from" : "no entry in"} ${filePath}`,
    });
  }
  for (const [slug, message] of failures) {
    onEvent({ step: "harnesses", status: "failed", detail: `${slug}: ${message}` });
  }

  onEvent({ step: "skills", status: "started", detail: "" });
  result.skills = removeSkills(plan.harnessSlugs, plan.installDirectory, options);
  for (const [slug, message] of result.skills) {
    onEvent({ step: "skills", status: "finished", detail: `${slug}: ${message}` });
  }

  removeLauncherStep(plan, result, onEvent, options);
  removeDirectories(plan, result, onEvent, options);
  return result;
}

function removeLauncherStep(
  plan: UninstallPlan,
  result: UninstallResult,
  onEvent: (event: StepEvent) => void,
  options: { home?: string; environment?: NodeJS.ProcessEnv },
): void {
  const binDirectory = plan.binDirectory ?? defaultBinDirectory(options);
  onEvent({ step: "path", status: "started", detail: binDirectory });
  if (plan.removeLauncher) {
    try {
      const removed = removeLauncher(binDirectory, plan.installDirectory);
      result.launcherRemoved = removed;
      onEvent({
        step: "path",
        status: removed !== null ? "finished" : "skipped",
        detail: removed !== null ? `removed ${removed}` : `no launcher of ours in ${binDirectory}`,
      });
    } catch (error) {
      result.failures.push(["launcher", error instanceof Error ? error.message : String(error)]);
      onEvent({
        step: "path",
        status: "failed",
        detail: error instanceof Error ? error.message : String(error),
      });
    }
  }
  if (!plan.removePathBlock) return;
  const cleared: string[] = [];
  for (const profile of shellProfiles(options)) {
    try {
      if (removePathBlock(profile)) {
        cleared.push(profile);
        onEvent({
          step: "path",
          status: "finished",
          detail: `removed the PATH block from ${profile}`,
        });
      }
    } catch (error) {
      result.failures.push([profile, error instanceof Error ? error.message : String(error)]);
      onEvent({
        step: "path",
        status: "failed",
        detail: `${profile}: ${error instanceof Error ? error.message : String(error)}`,
      });
    }
  }
  result.profilesCleared = cleared;
}

function removeDirectories(
  plan: UninstallPlan,
  result: UninstallResult,
  onEvent: (event: StepEvent) => void,
  options: { home?: string; environment?: NodeJS.ProcessEnv },
): void {
  const targets: [string, boolean][] = [];
  if (plan.removeData) {
    for (const directory of dataDirectories(options.environment)) targets.push([directory, false]);
  }
  if (plan.removeCheckout) targets.push([plan.installDirectory, true]);
  if (targets.length === 0) return;
  onEvent({ step: "directories", status: "started", detail: "" });
  const removed: string[] = [];
  for (const [directory, isCheckout] of targets) {
    if (!fs.existsSync(directory) || !fs.statSync(directory).isDirectory()) {
      onEvent({ step: "directories", status: "skipped", detail: `${directory} does not exist` });
      continue;
    }
    const refusal = refuseReason(directory, {
      checkout: isCheckout,
      ...(options.home === undefined ? {} : { home: options.home }),
    });
    if (refusal !== null) {
      result.failures.push([directory, `not removed: it ${refusal}`]);
      onEvent({
        step: "directories",
        status: "failed",
        detail: `${directory} ${refusal}; left alone`,
      });
      continue;
    }
    try {
      fs.rmSync(directory, { recursive: true, force: false });
      removed.push(directory);
      onEvent({ step: "directories", status: "finished", detail: `removed ${directory}` });
    } catch (error) {
      result.failures.push([directory, error instanceof Error ? error.message : String(error)]);
      onEvent({
        step: "directories",
        status: "failed",
        detail: `${directory}: ${error instanceof Error ? error.message : String(error)}`,
      });
    }
  }
  result.directoriesRemoved = removed;
}

export function resolveSlugs(
  selection: string | null,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): string[] {
  if (selection !== null) return parseHarnessSelection(selection);
  return [...loadPrefill(options).configuredSlugs];
}

export function describePlan(
  plan: UninstallPlan,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): string[] {
  const lines: string[] = [];
  if (plan.harnessSlugs.length > 0) {
    lines.push("Remove the code-indexing-mcp entry from:");
    for (const slug of plan.harnessSlugs) {
      lines.push(`  ${slug}: ${configurationPath(slug, options)}`);
    }
    lines.push("Unlink the bundled skills from those harnesses.");
  } else {
    lines.push("No configured harnesses to clear.");
  }
  const binDirectory = plan.binDirectory ?? defaultBinDirectory(options);
  if (plan.removeLauncher) {
    lines.push(`Remove the launcher at ${launcherPath(binDirectory)}.`);
  }
  if (plan.removePathBlock) lines.push("Remove the PATH block from your shell profiles.");
  if (plan.removeData) {
    for (const directory of dataDirectories(options.environment)) {
      lines.push(`DELETE the index/cache directory ${directory}.`);
    }
  }
  if (plan.removeCheckout) {
    lines.push(`DELETE the installation checkout ${plan.installDirectory}.`);
  }
  return lines;
}

export async function uninstallMain(options: {
  installDir?: string | null;
  harnessesSelection?: string | null;
  binDir?: string | null;
  keepLauncher?: boolean;
  keepPath?: boolean;
  purge?: boolean;
  removeCheckout?: boolean;
  assumeYes?: boolean;
  inputFn?: (prompt: string) => Promise<string> | string;
  output?: (line: string) => void;
  errorOutput?: (line: string) => void;
}): Promise<number> {
  const output = options.output ?? console.log;
  const errorOutput = options.errorOutput ?? console.error;
  const installDirectory =
    options.installDir != null ? resolveExisting(options.installDir) : defaultInstallDirectory();
  let slugs: string[];
  try {
    slugs = resolveSlugs(options.harnessesSelection ?? null);
  } catch (error) {
    errorOutput(`Error: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
  const plan: UninstallPlan = {
    installDirectory,
    harnessSlugs: slugs,
    binDirectory: options.binDir == null ? null : expandUser(options.binDir),
    removeLauncher: options.keepLauncher !== true,
    removePathBlock: options.keepPath !== true,
    removeData: options.purge === true,
    removeCheckout: options.removeCheckout === true,
  };
  for (const line of describePlan(plan)) output(line);
  if (options.assumeYes !== true) {
    const ask = options.inputFn ?? ((prompt: string) => prompt);
    const answer = String(await ask("Proceed? [y/N]: "))
      .trim()
      .toLowerCase();
    if (answer !== "y" && answer !== "yes") {
      output("Uninstall cancelled.");
      return 130;
    }
  }
  const result = runUninstall(plan, (event) => {
    const line = `[${event.step}] ${event.status}: ${event.detail}`;
    (event.status === "failed" ? errorOutput : output)(line);
  });
  if (result.failures.length > 0) {
    errorOutput(`Uninstall finished with ${result.failures.length} problem(s); see above.`);
    return 1;
  }
  output("Uninstall complete.");
  if (!plan.removeData) {
    output("Indexes and caches were kept; re-run with --purge to delete them.");
  }
  return 0;
}
