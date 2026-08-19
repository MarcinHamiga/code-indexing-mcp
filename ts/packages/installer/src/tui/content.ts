import fs from "node:fs";
import { ACCELERATOR_PROBE_TARGETS, serverExecutable } from "../accelerator.ts";
import { configurationPath, HARNESS_CHOICES } from "../harnesses.ts";
import { installWarnings, type InstallResult } from "../orchestrator.ts";
import { envUpdates, type WizardState } from "../wizard.ts";
import { activationHint, inspect, launcherOk, launcherPath } from "../shell-path.ts";
import { formatCheck } from "../verify.ts";

export const PANEL_ORDER = [
  "welcome",
  "location",
  "accelerator",
  "harnesses",
  "path",
  "indexing",
  "embedding",
  "summary",
  "progress",
  "done",
] as const;

export type PanelName = (typeof PANEL_ORDER)[number];

export const PANEL_TITLES: Record<PanelName, string> = {
  welcome: "Welcome",
  location: "Install location",
  accelerator: "Accelerator",
  harnesses: "MCP clients",
  path: "Command-line access",
  indexing: "Indexing settings",
  embedding: "Embedding settings",
  summary: "Summary",
  progress: "Installing",
  done: "Done",
};

export function welcomeText(state: WizardState): string {
  const lines: string[] = [];
  if (state.mode === "reconfigure") {
    lines.push("Reconfigure Code Indexing MCP");
    lines.push(`Installation: ${state.installDirectory}`);
    lines.push(
      "Your current settings were read from the configured harnesses. Walk through the sections and confirm on the summary screen.",
    );
  } else {
    lines.push("Install Code Indexing MCP");
    lines.push(`Installation: ${state.installDirectory}`);
    lines.push(
      "This wizard prepares the accelerator, configures your MCP clients, and lets you customize the server's settings.",
    );
  }
  if (state.disagreements.length > 0) {
    lines.push(
      `Your harnesses disagree on: ${state.disagreements.join(", ")}. The value from the earliest configured harness in the list is prefilled; confirming unifies them.`,
    );
  }
  if (state.mode === "reconfigure") {
    const executable = serverExecutable(state.installDirectory);
    if (!fs.existsSync(executable)) {
      lines.push(`This installation has no server executable at ${executable}.`);
    }
    const launcher = launcherPath(state.binDirectory);
    if (!fs.existsSync(launcher)) {
      lines.push(
        `The ${pathBasename(launcher)} launcher is missing from ${pathDirname(launcher)}; the Command-line access step will put it back.`,
      );
    }
    if (!fs.existsSync(executable) || !fs.existsSync(launcher)) {
      lines.push(
        "To restore everything without walking the wizard, quit and run: code-indexing-mcp configure --repair",
      );
    }
  }
  return lines.join("\n");
}

export function summaryText(state: WizardState): string {
  const lines = [`Install directory: ${state.installDirectory}`];
  if (state.accelerator === null) {
    lines.push(`Accelerator: keep the prepared backend (${state.preparedAccelerator ?? "none"})`);
  } else {
    lines.push(`Accelerator: ${state.accelerator}`);
    if (ACCELERATOR_PROBE_TARGETS.has(state.accelerator)) {
      lines.push("  A real inference probe runs before this backend is recorded.");
    }
  }
  lines.push(`Harnesses: ${state.harnessSlugs.join(", ") || "none"}`);
  if (state.installLauncher) {
    lines.push(`Launcher: ${launcherPath(state.binDirectory)}`);
    const statePath = inspect(state.binDirectory);
    if (state.modifyShellProfiles && !statePath.onPath) {
      lines.push("PATH: the launcher directory will be added to your shell profile");
    }
  } else {
    lines.push("Launcher: not requested");
  }
  const updates = envUpdates(state);
  if (Object.keys(updates).length > 0) {
    lines.push("Settings:");
    for (const name of Object.keys(updates).sort()) {
      const value = updates[name];
      lines.push(`  ${name} = ${value === null ? "(removed)" : value}`);
    }
  } else {
    lines.push("Settings: all defaults (nothing written to env blocks)");
  }
  const written = state.harnessSlugs.map((slug) => configurationPath(slug));
  if (written.length > 0) {
    lines.push("Files that will be written:");
    for (const filePath of written) lines.push(`  ${filePath}`);
    if (state.accelerator !== null) {
      lines.push("  the accelerator record in the server's data directory");
    }
  }
  if (state.disagreements.length > 0) {
    lines.push(
      `Harnesses that disagreed (${state.disagreements.join(", ")}) will be unified on the values above.`,
    );
  }
  return lines.join("\n");
}

export function doneText(
  result: InstallResult | null,
  options: { error?: Error; cancelled?: boolean } = {},
): string {
  const lines: string[] = [];
  let title = "Installation complete";
  if (options.cancelled === true) {
    title = "Installation cancelled";
    lines.push("Stopped between steps; anything already written above still applies.");
  } else if (options.error !== undefined) {
    title = "Installation failed";
    lines.push(options.error.message);
  } else if (result === null) {
    title = "Installation failed";
    lines.push("No result was produced.");
  } else {
    if (result.acceleratorPlan !== null) {
      const plan = result.acceleratorPlan;
      const marker = plan.honored ? "" : " (fell back to CPU)";
      lines.push(`Accelerator: ${plan.accelerator}${marker}\n  ${plan.reason}`);
    }
    if (result.launcher !== null) {
      const launcher = result.launcher;
      const verb = launcherOk(launcher) ? "Launcher" : "Launcher NOT created";
      lines.push(`${verb}: ${launcher.path}\n  ${launcher.detail}`);
    }
    for (const profile of result.profilesUpdated) lines.push(`Added to PATH in ${profile}`);
    for (const [slug, filePath] of result.configured) lines.push(`Configured ${slug}: ${filePath}`);
    for (const [slug, message] of result.failures) lines.push(`FAILED ${slug}: ${message}`);
    for (const [slug, message] of result.skills) lines.push(`Skills for ${slug}: ${message}`);
    if (result.checks.length > 0) {
      lines.push("");
      lines.push("Checks:");
      lines.push(...result.checks.map((item) => `  ${formatCheck(item)}`));
    }
    if (result.failures.length > 0) title = "Installation complete with failures";
    else if (installWarnings(result).length > 0) title = "Installation complete with warnings";
  }
  lines.push("");
  lines.push("Restart configured clients to load the MCP server.");
  if (result !== null && result.profilesUpdated.length > 0) {
    lines.push(`PATH is not live in this shell; run: ${activationHint(result.profilesUpdated)}`);
  }
  return `${title}\n${lines.join("\n")}`;
}

export function harnessChoices(): readonly { slug: string; label: string }[] {
  return HARNESS_CHOICES;
}

function pathBasename(filePath: string): string {
  const parts = filePath.split(/[\\/]/);
  return parts[parts.length - 1] ?? filePath;
}

function pathDirname(filePath: string): string {
  const index = Math.max(filePath.lastIndexOf("/"), filePath.lastIndexOf("\\"));
  return index === -1 ? filePath : filePath.slice(0, index);
}
