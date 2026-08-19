import fs from "node:fs";
import { ACCELERATOR_CHOICES, serverExecutable } from "../accelerator.ts";
import { InstallerError } from "../config-files.ts";
import { parseHarnessSelection } from "../harnesses.ts";
import { defaultValue, normalize, SETTINGS, type Setting, validate } from "../settings-spec.ts";
import { setField, type WizardState } from "../wizard.ts";
import { PANEL_ORDER, PANEL_TITLES, type PanelName, summaryText, welcomeText } from "./content.ts";

export interface SelectChoice {
  readonly name: string;
  readonly description: string;
  readonly value: string;
}

export type WizardControl =
  | { readonly kind: "none" }
  | { readonly kind: "input"; readonly value: string; readonly placeholder: string }
  | {
      readonly kind: "select";
      readonly choices: readonly SelectChoice[];
      readonly selected: string;
    };

export interface WizardView {
  readonly panel: PanelName;
  readonly title: string;
  readonly body: string;
  readonly control: WizardControl;
  readonly error: string | null;
}

export interface AdvanceResult {
  readonly startInstall: boolean;
  readonly error: string | null;
}

const KEEP_ACCELERATOR = "__keep__";

function yesNo(selected: boolean): WizardControl {
  return {
    kind: "select",
    choices: [
      { name: "yes", description: "enabled", value: "yes" },
      { name: "no", description: "disabled", value: "no" },
    ],
    selected: selected ? "yes" : "no",
  };
}

function settingsFor(panel: PanelName): readonly Setting[] {
  if (panel === "indexing") return SETTINGS.filter((setting) => setting.group === "Indexing");
  if (panel === "embedding") return SETTINGS.filter((setting) => setting.group === "Embedding");
  return [];
}

function settingControl(state: WizardState, setting: Setting): WizardControl {
  const raw = state.values[setting.name] ?? "";
  if (setting.type === "bool") {
    const value = raw === "" ? setting.default : normalize(setting, raw);
    return yesNo(value === "1");
  }
  if (setting.type === "choice") {
    const selected = normalize(setting, raw || setting.default);
    return {
      kind: "select",
      choices: setting.choices.map((choice) => ({
        name: choice,
        description: setting.help,
        value: choice,
      })),
      selected,
    };
  }
  return { kind: "input", value: raw, placeholder: defaultValue(setting) };
}

export class WizardController {
  readonly state: WizardState;
  private panelIndex = 0;
  private fieldIndex = 0;
  private error: string | null = null;

  constructor(state: WizardState) {
    this.state = state;
  }

  get panel(): PanelName {
    return PANEL_ORDER[this.panelIndex] ?? "done";
  }

  view(): WizardView {
    const panel = this.panel;
    const title = this.title(panel);
    if (panel === "welcome") {
      return {
        panel,
        title,
        body: welcomeText(this.state),
        control: { kind: "none" },
        error: this.error,
      };
    }
    if (panel === "location") {
      return {
        panel,
        title,
        body: "The checkout this wizard configures. It must already contain the prepared server launcher.",
        control: { kind: "input", value: this.state.installDirectory, placeholder: "" },
        error: this.error,
      };
    }
    if (panel === "accelerator") {
      const choices: SelectChoice[] = ACCELERATOR_CHOICES.map((choice) => ({
        name: choice === "auto" ? "auto (recommended)" : choice,
        description: choice === "auto" ? "detect a supported backend and fall back to CPU" : choice,
        value: choice,
      }));
      if (this.state.mode === "reconfigure") {
        choices.unshift({
          name: `keep prepared (${this.state.preparedAccelerator ?? "none"})`,
          description: "do not probe or replace the prepared backend",
          value: KEEP_ACCELERATOR,
        });
      }
      return {
        panel,
        title,
        body: "Choose the passage embedding accelerator to prepare.",
        control: {
          kind: "select",
          choices,
          selected: this.state.accelerator ?? KEEP_ACCELERATOR,
        },
        error: this.error,
      };
    }
    if (panel === "harnesses") {
      return {
        panel,
        title,
        body: "Enter comma-separated harness numbers/slugs, all, or an empty value for none.",
        control: {
          kind: "input",
          value: this.state.harnessSlugs.join(","),
          placeholder: "all",
        },
        error: this.error,
      };
    }
    if (panel === "path") return this.pathView(title);
    if (panel === "indexing" || panel === "embedding") return this.settingView(panel, title);
    if (panel === "summary") {
      return {
        panel,
        title,
        body: summaryText(this.state),
        control: { kind: "none" },
        error: this.error,
      };
    }
    return { panel, title, body: "", control: { kind: "none" }, error: this.error };
  }

  advance(value: string | null): AdvanceResult {
    const error = this.commit(value);
    if (error !== null) {
      this.error = error;
      return { startInstall: false, error };
    }
    this.error = null;
    const fields = this.fieldCount(this.panel);
    if (this.fieldIndex + 1 < fields) {
      this.fieldIndex += 1;
      return { startInstall: false, error: null };
    }
    if (this.panel === "summary") {
      this.panelIndex = PANEL_ORDER.indexOf("progress");
      this.fieldIndex = 0;
      return { startInstall: true, error: null };
    }
    const next = PANEL_ORDER[this.panelIndex + 1];
    if (next !== undefined && next !== "progress" && next !== "done") {
      this.panelIndex += 1;
      this.fieldIndex = 0;
    }
    return { startInstall: false, error: null };
  }

  back(): void {
    this.error = null;
    if (this.fieldIndex > 0) {
      this.fieldIndex -= 1;
      return;
    }
    if (this.panelIndex === 0 || this.panel === "progress" || this.panel === "done") return;
    this.panelIndex -= 1;
    this.fieldIndex = Math.max(0, this.fieldCount(this.panel) - 1);
  }

  showDone(): void {
    this.panelIndex = PANEL_ORDER.indexOf("done");
    this.fieldIndex = 0;
    this.error = null;
  }

  private title(panel: PanelName): string {
    const walked: readonly PanelName[] = PANEL_ORDER.filter(
      (name) => name !== "progress" && name !== "done",
    );
    const panelPosition = walked.indexOf(panel);
    const fieldCount = this.fieldCount(panel);
    const field = fieldCount > 1 ? `, field ${this.fieldIndex + 1} of ${fieldCount}` : "";
    return panelPosition === -1
      ? PANEL_TITLES[panel]
      : `Step ${panelPosition + 1} of ${walked.length}${field} - ${PANEL_TITLES[panel]}`;
  }

  private pathView(title: string): WizardView {
    if (this.fieldIndex === 0) {
      return {
        panel: "path",
        title,
        body: "Create or refresh the code-indexing-mcp command-line launcher?",
        control: yesNo(this.state.installLauncher),
        error: this.error,
      };
    }
    if (this.fieldIndex === 1) {
      return {
        panel: "path",
        title,
        body: "Add the launcher directory to a supported shell profile when it is not on PATH?",
        control: yesNo(this.state.modifyShellProfiles),
        error: this.error,
      };
    }
    return {
      panel: "path",
      title,
      body: "Launcher directory.",
      control: { kind: "input", value: this.state.binDirectory, placeholder: "~/.local/bin" },
      error: this.error,
    };
  }

  private settingView(panel: "indexing" | "embedding", title: string): WizardView {
    const settings = settingsFor(panel);
    const setting = settings[this.fieldIndex];
    if (setting === undefined) {
      return { panel, title, body: "No settings.", control: { kind: "none" }, error: this.error };
    }
    return {
      panel,
      title,
      body: `${setting.label}\n${setting.name}\n${setting.help}\nLeave an input empty to use ${defaultValue(setting)}.`,
      control: settingControl(this.state, setting),
      error: this.error,
    };
  }

  private fieldCount(panel: PanelName): number {
    if (panel === "path") return 3;
    if (panel === "indexing" || panel === "embedding") return settingsFor(panel).length;
    return 1;
  }

  private commit(value: string | null): string | null {
    const panel = this.panel;
    if (panel === "location") {
      const directory = (value ?? "").trim();
      if (directory === "") return "Install directory cannot be empty.";
      if (!fs.existsSync(serverExecutable(directory))) {
        return `No prepared installation there; expected ${serverExecutable(directory)}.`;
      }
      this.state.installDirectory = directory;
    } else if (panel === "accelerator") {
      this.state.accelerator = value === KEEP_ACCELERATOR ? null : value;
    } else if (panel === "harnesses") {
      try {
        this.state.harnessSlugs = parseHarnessSelection(value ?? "");
      } catch (error) {
        return error instanceof InstallerError ? error.message : String(error);
      }
    } else if (panel === "path") {
      if (this.fieldIndex === 0) this.state.installLauncher = value === "yes";
      else if (this.fieldIndex === 1) this.state.modifyShellProfiles = value === "yes";
      else {
        const directory = (value ?? "").trim();
        if (this.state.installLauncher && directory === "")
          return "Launcher directory cannot be empty.";
        if (directory !== "") this.state.binDirectory = directory;
      }
    } else if (panel === "indexing" || panel === "embedding") {
      const setting = settingsFor(panel)[this.fieldIndex];
      if (setting === undefined) return null;
      const raw = value ?? "";
      if (raw !== "") {
        const problem = validate(setting, raw);
        if (problem !== null) return problem;
      }
      setField(this.state, setting.name, raw === "" ? "" : normalize(setting, raw));
    }
    return null;
  }
}
