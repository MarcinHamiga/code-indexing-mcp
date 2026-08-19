import { preparedAccelerator } from "./accelerator.ts";
import { envFromEntry } from "./env-blocks.ts";
import { HARNESS_CHOICES, readServerEntry } from "./harnesses.ts";
import type { InstallPlan } from "./orchestrator.ts";
import { defaultInstallDirectory } from "./orchestrator.ts";
import { defaultBinDirectory } from "./shell-path.ts";
import { BY_NAME, defaultValue, normalize, SETTINGS, validate } from "./settings-spec.ts";

export interface Prefill {
  readonly values: Readonly<Record<string, string>>;
  readonly configuredSlugs: readonly string[];
  readonly disagreements: readonly string[];
}

export function loadPrefill(
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): Prefill {
  const values: Record<string, string> = {};
  const configured: string[] = [];
  const disagreements: string[] = [];
  for (const choice of HARNESS_CHOICES) {
    const entry = readServerEntry(choice.slug, options);
    if (entry === null) continue;
    configured.push(choice.slug);
    for (const [name, raw] of Object.entries(envFromEntry(choice.slug, entry))) {
      const setting = BY_NAME[name];
      if (setting === undefined) continue;
      if (validate(setting, raw) !== null) continue;
      const value = normalize(setting, raw);
      if (name in values && values[name] !== value) {
        if (!disagreements.includes(name)) disagreements.push(name);
      } else {
        values[name] = value;
      }
    }
  }
  return { values, configuredSlugs: configured, disagreements };
}

export interface WizardState {
  mode: "install" | "reconfigure";
  installDirectory: string;
  accelerator: string | null;
  preparedAccelerator: string | null;
  harnessSlugs: string[];
  configuredSlugs: readonly string[];
  values: Record<string, string>;
  prefilledNames: Set<string>;
  disagreements: string[];
  offline: boolean;
  binDirectory: string;
  installLauncher: boolean;
  modifyShellProfiles: boolean;
}

export function wizardStateForInstall(
  installDirectory: string,
  options: {
    presetValues?: Readonly<Record<string, string>>;
    presetAccelerator?: string | null;
    home?: string;
    environment?: NodeJS.ProcessEnv;
  } = {},
): WizardState {
  const prefill = loadPrefill(options);
  return {
    mode: "install",
    installDirectory,
    accelerator: options.presetAccelerator ?? "auto",
    preparedAccelerator: null,
    harnessSlugs: [...prefill.configuredSlugs],
    configuredSlugs: prefill.configuredSlugs,
    values: { ...prefill.values, ...(options.presetValues ?? {}) },
    prefilledNames: new Set(Object.keys(prefill.values)),
    disagreements: [...prefill.disagreements],
    offline: false,
    binDirectory: defaultBinDirectory(options),
    installLauncher: true,
    modifyShellProfiles: true,
  };
}

export function wizardStateForReconfigure(
  installDirectory: string,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): WizardState {
  const prefill = loadPrefill(options);
  return {
    mode: "reconfigure",
    installDirectory,
    accelerator: null,
    preparedAccelerator: preparedAccelerator(installDirectory),
    harnessSlugs: [...prefill.configuredSlugs],
    configuredSlugs: prefill.configuredSlugs,
    values: { ...prefill.values },
    prefilledNames: new Set(Object.keys(prefill.values)),
    disagreements: [...prefill.disagreements],
    offline: false,
    binDirectory: defaultBinDirectory(options),
    installLauncher: true,
    modifyShellProfiles: true,
  };
}

export function fieldValue(state: WizardState, name: string): string {
  return state.values[name] ?? "";
}

export function setField(state: WizardState, name: string, raw: string): void {
  state.values[name] = raw;
}

export function envUpdates(state: WizardState): Record<string, string | null> {
  const updates: Record<string, string | null> = {};
  for (const setting of SETTINGS) {
    const raw = (state.values[setting.name] ?? "").trim();
    if (raw === "" || raw === defaultValue(setting)) {
      if (state.prefilledNames.has(setting.name)) updates[setting.name] = null;
      continue;
    }
    updates[setting.name] = normalize(setting, raw);
  }
  return updates;
}

export function toPlan(state: WizardState): InstallPlan {
  return {
    installDirectory: state.installDirectory,
    accelerator: state.accelerator,
    harnessSlugs: state.harnessSlugs,
    envUpdates: envUpdates(state),
    offline: state.offline,
    binDirectory: state.binDirectory,
    installLauncher: state.installLauncher,
    modifyShellProfiles: state.modifyShellProfiles,
  };
}

export { defaultInstallDirectory };
