import fs from "node:fs";
import { createInterface } from "node:readline/promises";
import { Command, Option } from "commander";
import { ACCELERATOR_CHOICES, serverExecutable } from "./accelerator.ts";
import { expandUser, InstallerError, resolveExisting } from "./config-files.ts";
import { HARNESS_CHOICES, parseHarnessSelection } from "./harnesses.ts";
import {
  defaultInstallDirectory,
  installWarnings,
  runInstall,
  type StepEvent,
} from "./orchestrator.ts";
import { asBool, BY_NAME, normalize, validate } from "./settings-spec.ts";
import { activationHint } from "./shell-path.ts";
import { loadPrefill } from "./wizard.ts";

export function parseSettings(
  pairs: readonly string[],
  unsets: readonly string[],
): Record<string, string | null> {
  const updates: Record<string, string | null> = {};
  for (const pair of pairs) {
    const separator = pair.indexOf("=");
    if (separator === -1) {
      throw new InstallerError(`--set expects NAME=VALUE, got ${JSON.stringify(pair)}`);
    }
    const name = pair.slice(0, separator).trim();
    const value = pair.slice(separator + 1);
    const setting = BY_NAME[name];
    if (setting === undefined) {
      throw new InstallerError(
        `unknown setting ${JSON.stringify(name)}; managed settings: ${Object.keys(BY_NAME).sort().join(", ")}`,
      );
    }
    const error = validate(setting, value);
    if (error !== null) throw new InstallerError(error);
    updates[name] = normalize(setting, value);
  }
  for (const raw of unsets) {
    const name = raw.trim();
    if (!(name in BY_NAME)) {
      throw new InstallerError(
        `unknown setting ${JSON.stringify(name)}; managed settings: ${Object.keys(BY_NAME).sort().join(", ")}`,
      );
    }
    updates[name] = null;
  }
  return updates;
}

function printEvent(event: StepEvent): void {
  const stream =
    event.status === "warning" || event.status === "failed" ? process.stderr : process.stdout;
  stream.write(`[${event.step}] ${event.status}: ${event.detail}\n`);
}

export async function promptHarnesses(
  inputFn?: (prompt: string) => Promise<string>,
  outputFn: (line: string) => void = console.log,
): Promise<string[]> {
  outputFn("Select the harnesses to configure:");
  HARNESS_CHOICES.forEach((choice, index) => {
    outputFn(`  ${index + 1}. ${choice.label}`);
  });
  const ask =
    inputFn ??
    (async (prompt: string) => {
      const rl = createInterface({ input: process.stdin, output: process.stdout });
      try {
        return await rl.question(prompt);
      } finally {
        rl.close();
      }
    });
  return parseHarnessSelection(
    await ask("Enter comma-separated choices, 'all', or leave blank to skip: "),
  );
}

interface ParsedArgs {
  installDir: string;
  accelerator: string | null;
  harnesses: string | null;
  settings: string[];
  unsets: string[];
  binDir: string | null;
  noLauncher: boolean;
  noModifyPath: boolean;
  offline: boolean;
  tui: boolean;
  noPrompt: boolean;
  reconfigure: boolean;
  repair: boolean;
}

function collect(value: string, previous: string[]): string[] {
  return [...previous, value];
}

export function parseArgv(argv: readonly string[]): ParsedArgs {
  const parser = new Command();
  parser
    .name("code_indexing_mcp.installer")
    .exitOverride()
    .configureOutput({
      writeErr: (text) => {
        throw new InstallerError(text.trim());
      },
    })
    .option("--install-dir <path>", "checkout location", defaultInstallDirectory())
    .addOption(new Option("--accelerator <name>").choices([...ACCELERATOR_CHOICES]))
    .option("--harnesses <selection>")
    .option("--set <NAME=VALUE>", "set a managed setting", collect, [])
    .option("--unset <NAME>", "remove a managed setting", collect, [])
    .option("--bin-dir <path>")
    .option("--no-launcher")
    .option("--no-modify-path")
    .option("--offline", undefined, asBool(process.env.CODE_INDEXING_OFFLINE ?? ""))
    .option("--tui")
    .option("--no-prompt")
    .option("--reconfigure")
    .option("--repair");
  parser.parse([...argv], { from: "user" });
  const opts = parser.opts<{
    installDir: string;
    accelerator?: string;
    harnesses?: string;
    set: string[];
    unset: string[];
    binDir?: string;
    launcher?: boolean;
    modifyPath?: boolean;
    offline?: boolean;
    tui?: boolean;
    prompt?: boolean;
    reconfigure?: boolean;
    repair?: boolean;
  }>();
  return {
    installDir: opts.installDir,
    accelerator: opts.accelerator ?? null,
    harnesses: opts.harnesses ?? null,
    settings: opts.set,
    unsets: opts.unset,
    binDir: opts.binDir ?? null,
    noLauncher: opts.launcher === false,
    noModifyPath: opts.modifyPath === false,
    offline: opts.offline === true,
    tui: opts.tui === true,
    noPrompt: opts.prompt === false,
    reconfigure: opts.reconfigure === true,
    repair: opts.repair === true,
  };
}

async function runTui(
  args: ParsedArgs,
  installDirectory: string,
  envUpdates: Record<string, string | null>,
): Promise<number> {
  let startWizard: typeof import("./tui/app.ts").startInstallerApp;
  try {
    ({ startInstallerApp: startWizard } = await import("./tui/app.ts"));
  } catch {
    process.stderr.write(
      "Error: the interactive wizard needs @opentui/core; re-run with --no-tui.\n",
    );
    return 1;
  }
  const { wizardStateForInstall, wizardStateForReconfigure } = await import("./wizard.ts");
  const preset = Object.fromEntries(
    Object.entries(envUpdates).filter((entry): entry is [string, string] => entry[1] !== null),
  );
  const state = args.reconfigure
    ? wizardStateForReconfigure(installDirectory)
    : wizardStateForInstall(installDirectory, {
        presetValues: preset,
        presetAccelerator: args.accelerator,
      });
  if (args.reconfigure) {
    Object.assign(state.values, preset);
    if (args.accelerator !== null) state.accelerator = args.accelerator;
  }
  for (const [name, value] of Object.entries(envUpdates)) {
    if (value === null) delete state.values[name];
  }
  if (args.harnesses !== null) state.harnessSlugs = parseHarnessSelection(args.harnesses);
  state.offline = args.offline;
  if (args.binDir !== null) state.binDirectory = expandUser(args.binDir);
  state.installLauncher = !args.noLauncher;
  state.modifyShellProfiles = !args.noModifyPath;
  return startWizard(state);
}

async function repair(installDirectory: string, args: ParsedArgs): Promise<number> {
  const prefill = loadPrefill();
  const selected =
    args.harnesses !== null ? parseHarnessSelection(args.harnesses) : [...prefill.configuredSlugs];
  const result = await runInstall(
    {
      installDirectory,
      accelerator: null,
      harnessSlugs: selected,
      envUpdates: { ...prefill.values },
      offline: args.offline,
      binDirectory: args.binDir === null ? null : expandUser(args.binDir),
      installLauncher: !args.noLauncher,
      modifyShellProfiles: !args.noModifyPath,
    },
    printEvent,
  );
  if (result.failures.length > 0) {
    process.stderr.write(`Repair finished with ${result.failures.length} failure(s); see above.\n`);
    return 1;
  }
  process.stdout.write("Repair complete.\n");
  return 0;
}

export async function main(argv?: readonly string[]): Promise<number> {
  let args: ParsedArgs;
  try {
    args = parseArgv(argv ?? process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`Error: ${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
  const installDirectory = resolveExisting(args.installDir);
  try {
    const envUpdates = parseSettings(args.settings, args.unsets);
    if (args.tui) return await runTui(args, installDirectory, envUpdates);
    if (args.repair) return await repair(installDirectory, args);
    let selected: string[];
    if (args.harnesses !== null) selected = parseHarnessSelection(args.harnesses);
    else if (args.reconfigure) selected = [...loadPrefill().configuredSlugs];
    else if (args.noPrompt || !process.stdin.isTTY) selected = [];
    else selected = await promptHarnesses();
    const accelerator = args.accelerator === null && !args.reconfigure ? "auto" : args.accelerator;
    const result = await runInstall(
      {
        installDirectory,
        accelerator,
        harnessSlugs: selected,
        envUpdates,
        offline: args.offline,
        binDirectory: args.binDir === null ? null : expandUser(args.binDir),
        installLauncher: !args.noLauncher,
        modifyShellProfiles: !args.noModifyPath,
      },
      printEvent,
    );
    if (
      result.configured.length === 0 &&
      result.failures.length === 0 &&
      result.skills.length === 0
    ) {
      process.stdout.write("No harness configuration selected.\n");
    }
    if (result.failures.length > 0) {
      process.stderr.write(
        `Installation finished with ${result.failures.length} failed harness configuration(s); see the errors above.\n`,
      );
      return 1;
    }
    if (installWarnings(result).length > 0) {
      process.stdout.write(
        `Installation complete with ${installWarnings(result).length} check warning(s); see the [verify] lines above.\n`,
      );
    }
    process.stdout.write(
      "Installation complete. Restart configured clients to load the MCP server.\n",
    );
    if (result.profilesUpdated.length > 0) {
      process.stdout.write(
        `PATH was updated in ${result.profilesUpdated.join(", ")}; start a new shell or run: \n`,
      );
      process.stdout.write(`  ${activationHint(result.profilesUpdated)}\n`);
    }
    return 0;
  } catch (error) {
    if (error instanceof InstallerError) {
      process.stderr.write(`Error: ${error.message}\n`);
      return 1;
    }
    throw error;
  }
}

export async function configureMain(options: {
  installDir?: string | null;
  accelerator?: string | null;
  harnesses?: string | null;
  settings?: readonly string[];
  unsets?: readonly string[];
  noTui?: boolean;
  binDir?: string | null;
  noLauncher?: boolean;
  noModifyPath?: boolean;
  repair?: boolean;
}): Promise<number> {
  const installDirectory =
    options.installDir !== undefined && options.installDir !== null
      ? resolveExisting(options.installDir)
      : resolveExisting(defaultInstallDirectory());
  if (!fs.existsSync(serverExecutable(installDirectory))) {
    process.stderr.write(`Error: no installation found at ${installDirectory}\n`);
    return 1;
  }
  const argv = ["--install-dir", installDirectory, "--reconfigure", "--no-prompt"];
  if (options.accelerator != null) argv.push("--accelerator", options.accelerator);
  if (options.harnesses != null) argv.push("--harnesses", options.harnesses);
  for (const pair of options.settings ?? []) argv.push("--set", pair);
  for (const name of options.unsets ?? []) argv.push("--unset", name);
  if (options.binDir != null) argv.push("--bin-dir", options.binDir);
  if (options.noLauncher === true) argv.push("--no-launcher");
  if (options.noModifyPath === true) argv.push("--no-modify-path");
  if (options.repair === true) argv.push("--repair");
  const scripted = Boolean(
    (options.settings?.length ?? 0) > 0 ||
      (options.unsets?.length ?? 0) > 0 ||
      options.harnesses != null ||
      options.accelerator != null ||
      options.repair === true,
  );
  if (options.noTui !== true && !scripted && process.stdin.isTTY) {
    const index = argv.indexOf("--no-prompt");
    if (index !== -1) argv.splice(index, 1);
    argv.push("--tui");
  }
  return main(argv);
}

if (import.meta.main) {
  main()
    .then((code) => {
      process.exit(code);
    })
    .catch((error: unknown) => {
      process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
      process.exit(1);
    });
}
