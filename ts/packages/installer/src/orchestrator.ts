import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { type AcceleratorPlan, configureAccelerator, serverExecutable } from "./accelerator.ts";
import { expandUser, InstallerError } from "./config-files.ts";
import { configureSelectedHarnesses, installSkills } from "./harnesses.ts";
import {
  defaultBinDirectory,
  installLauncher,
  isOnPath,
  type LauncherResult,
  launcherOk,
  shellProfiles,
  updateProfiles,
} from "./shell-path.ts";
import { type Check, runChecks } from "./verify.ts";

export function defaultInstallDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  const configured = environment.CODE_INDEXING_MCP_INSTALL_DIR;
  if (configured !== undefined && configured !== "") return expandUser(configured);
  return path.join(os.homedir(), ".local", "share", "code-indexing-mcp");
}

export interface InstallPlan {
  readonly installDirectory: string;
  readonly accelerator: string | null;
  readonly harnessSlugs: readonly string[];
  readonly envUpdates: Readonly<Record<string, string | null>>;
  readonly offline: boolean;
  readonly binDirectory: string | null;
  readonly installLauncher: boolean;
  readonly modifyShellProfiles: boolean;
}

export interface StepEvent {
  readonly step: string;
  readonly status: string;
  readonly detail: string;
}

export interface InstallResult {
  readonly acceleratorPlan: AcceleratorPlan | null;
  readonly configured: readonly [string, string][];
  readonly failures: readonly [string, string][];
  readonly skills: readonly [string, string][];
  readonly launcher: LauncherResult | null;
  readonly profilesUpdated: readonly string[];
  readonly checks: readonly Check[];
}

export function installWarnings(result: InstallResult): readonly Check[] {
  return result.checks.filter((check) => !check.ok);
}

export async function runInstall(
  plan: InstallPlan,
  onEvent: (event: StepEvent) => void = () => undefined,
  shouldContinue: () => boolean = () => true,
): Promise<InstallResult> {
  let acceleratorPlan: AcceleratorPlan | null = null;
  if (plan.accelerator === null) {
    onEvent({ step: "accelerator", status: "skipped", detail: "keeping the prepared backend" });
  } else if (shouldContinue()) {
    onEvent({ step: "accelerator", status: "started", detail: plan.accelerator });
    acceleratorPlan = await configureAccelerator(plan.installDirectory, plan.accelerator, {
      offline: plan.offline,
    });
    onEvent({
      step: "accelerator",
      status: acceleratorPlan.honored ? "finished" : "warning",
      detail: `${acceleratorPlan.accelerator} (${acceleratorPlan.reason})`,
    });
  }

  let launcher: LauncherResult | null = null;
  let profilesUpdated: readonly string[] = [];
  if (!plan.installLauncher) {
    onEvent({ step: "path", status: "skipped", detail: "launcher not requested" });
  } else if (shouldContinue()) {
    const installed = installLauncherStep(plan, onEvent);
    launcher = installed.launcher;
    profilesUpdated = installed.profiles;
  }

  let configured: [string, string][] = [];
  let failures: [string, string][] = [];
  let skills: [string, string][] = [];
  if (shouldContinue()) {
    const command = serverExecutable(plan.installDirectory);
    if (plan.harnessSlugs.length > 0 && !fs.existsSync(command)) {
      throw new InstallerError(
        `No prepared installation at ${plan.installDirectory}: expected the server executable at ${command}`,
      );
    }
    onEvent({
      step: "harnesses",
      status: "started",
      detail: plan.harnessSlugs.join(", ") || "none selected",
    });
    [configured, failures] = configureSelectedHarnesses(plan.harnessSlugs, command, {
      env: plan.envUpdates,
    });
    for (const [slug, filePath] of configured) {
      onEvent({ step: "harnesses", status: "finished", detail: `${slug}: ${filePath}` });
    }
    for (const [slug, message] of failures) {
      onEvent({ step: "harnesses", status: "failed", detail: `${slug}: ${message}` });
    }
  }
  if (shouldContinue()) {
    onEvent({ step: "skills", status: "started", detail: "" });
    skills = installSkills(plan.harnessSlugs, plan.installDirectory);
    for (const [slug, message] of skills) {
      onEvent({ step: "skills", status: "finished", detail: `${slug}: ${message}` });
    }
  }

  let checks: readonly Check[] = [];
  if (shouldContinue()) {
    onEvent({ step: "verify", status: "started", detail: "" });
    checks = runChecks(plan.installDirectory, configured, {
      launcher,
      profilesUpdated,
      acceleratorWasPrepared: plan.accelerator !== null,
    });
    for (const check of checks) {
      onEvent({
        step: "verify",
        status: check.ok ? "finished" : "warning",
        detail: `${check.name}: ${check.detail}`,
      });
    }
  }
  return {
    acceleratorPlan,
    configured,
    failures,
    skills,
    launcher,
    profilesUpdated,
    checks,
  };
}

function installLauncherStep(
  plan: InstallPlan,
  onEvent: (event: StepEvent) => void,
): { launcher: LauncherResult; profiles: readonly string[] } {
  const binDirectory = plan.binDirectory ?? defaultBinDirectory();
  onEvent({ step: "path", status: "started", detail: binDirectory });
  const launcher = installLauncher(plan.installDirectory, binDirectory);
  onEvent({
    step: "path",
    status: launcherOk(launcher) ? "finished" : "warning",
    detail: `${launcher.path}: ${launcher.status} (${launcher.detail})`,
  });
  if (!plan.modifyShellProfiles) return { launcher, profiles: [] };
  if (isOnPath(binDirectory)) {
    onEvent({ step: "path", status: "skipped", detail: `${binDirectory} is already on PATH` });
    return { launcher, profiles: [] };
  }
  const profiles = shellProfiles();
  if (profiles.length === 0) {
    onEvent({
      step: "path",
      status: "warning",
      detail: `add ${binDirectory} to PATH yourself on this platform`,
    });
    return { launcher, profiles: [] };
  }
  const [written, profileFailures] = updateProfiles(binDirectory, profiles);
  for (const profile of written) {
    onEvent({
      step: "path",
      status: "finished",
      detail: `added ${binDirectory} to PATH in ${profile}`,
    });
  }
  for (const [profile, message] of profileFailures) {
    onEvent({ step: "path", status: "warning", detail: `${profile}: ${message}` });
  }
  if (written.length === 0 && profileFailures.length === 0) {
    onEvent({ step: "path", status: "skipped", detail: "the shell profiles already set PATH" });
  }
  return { launcher, profiles: written };
}
