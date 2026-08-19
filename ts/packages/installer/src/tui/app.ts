import {
  BoxRenderable,
  createCliRenderer,
  InputRenderable,
  SelectRenderable,
  TextRenderable,
} from "@opentui/core";
import { ACCELERATOR_CHOICES } from "../accelerator.ts";
import { parseHarnessSelection } from "../harnesses.ts";
import { runInstall, type StepEvent } from "../orchestrator.ts";
import { SETTINGS } from "../settings-spec.ts";
import { toPlan, type WizardState } from "../wizard.ts";
import {
  doneText,
  PANEL_ORDER,
  PANEL_TITLES,
  type PanelName,
  summaryText,
  welcomeText,
} from "./content.ts";

export async function startInstallerApp(state: WizardState): Promise<number> {
  const renderer = await createCliRenderer({ exitOnCtrlC: false });
  const wizard = new InstallerWizard(renderer, state);
  renderer.root.add(wizard.root);
  wizard.show(PANEL_ORDER[0]);
  return await wizard.finished;
}

class InstallerWizard {
  readonly root: InstanceType<typeof BoxRenderable>;
  readonly finished: Promise<number>;
  doneCode: number | null = null;
  private readonly resolveFinished: (code: number) => void;
  private readonly body: InstanceType<typeof TextRenderable>;
  private readonly title: InstanceType<typeof TextRenderable>;
  private readonly hint: InstanceType<typeof TextRenderable>;
  private readonly input: InstanceType<typeof InputRenderable>;
  private readonly select: InstanceType<typeof SelectRenderable>;
  private panel: PanelName = "welcome";
  private cancelled = false;

  private readonly renderer: Awaited<ReturnType<typeof createCliRenderer>>;
  private readonly state: WizardState;

  constructor(renderer: Awaited<ReturnType<typeof createCliRenderer>>, state: WizardState) {
    this.renderer = renderer;
    this.state = state;
    this.root = new BoxRenderable(renderer, {
      id: "installer",
      flexGrow: 1,
      flexDirection: "column",
      padding: 1,
    });
    this.title = new TextRenderable(renderer, {
      id: "title",
      content: "Code Indexing MCP Installer",
    });
    this.body = new TextRenderable(renderer, { id: "body", content: "", flexGrow: 1 });
    this.input = new InputRenderable(renderer, { id: "field", width: 80, visible: false });
    this.select = new SelectRenderable(renderer, {
      id: "choices",
      options: [],
      visible: false,
      height: 8,
    });
    this.hint = new TextRenderable(renderer, {
      id: "hint",
      content: "Ctrl+N next  Ctrl+B back  Esc cancel",
    });
    this.root.add(this.title);
    this.root.add(this.body);
    this.root.add(this.input);
    this.root.add(this.select);
    this.root.add(this.hint);
    let resolveFinished: (code: number) => void = () => undefined;
    this.finished = new Promise((resolve) => {
      resolveFinished = resolve;
    });
    this.resolveFinished = resolveFinished;
    renderer.keyInput.on("keypress", (event: { name?: string; ctrl?: boolean }) => {
      if (this.panel === "progress" || this.panel === "done") return;
      if (event.ctrl === true && event.name === "n") void this.advance();
      if (event.ctrl === true && event.name === "b") this.back();
      if (event.name === "escape") this.cancel();
    });
  }

  show(name: PanelName): void {
    this.commit();
    this.panel = name;
    const walked = PANEL_ORDER.filter(
      (panel): panel is Exclude<PanelName, "progress" | "done"> =>
        panel !== "progress" && panel !== "done",
    );
    const index = walked.indexOf(name as Exclude<PanelName, "progress" | "done">);
    const title =
      index === -1
        ? PANEL_TITLES[name]
        : `Step ${index + 1} of ${walked.length} - ${PANEL_TITLES[name]}`;
    this.title.content = `Code Indexing MCP Installer  ${title}`;
    this.input.visible = false;
    this.select.visible = false;
    if (name === "welcome") this.body.content = welcomeText(this.state);
    else if (name === "location") {
      this.body.content = "The checkout this wizard configures.";
      this.showInput(this.state.installDirectory);
    } else if (name === "accelerator") {
      this.body.content = "Which accelerator to prepare. auto detects one and falls back to CPU.";
      this.showSelect(
        ACCELERATOR_CHOICES.map((choice) => ({ name: choice, description: choice, value: choice })),
        this.state.accelerator ?? "auto",
      );
    } else if (name === "harnesses") {
      this.body.content = "Comma-separated harness numbers/slugs, or all.";
      this.showInput(this.state.harnessSlugs.join(",") || "all");
    } else if (name === "path") {
      this.body.content = `Launcher directory: ${this.state.binDirectory}`;
      this.showInput(this.state.binDirectory);
    } else if (name === "indexing" || name === "embedding") {
      const group = name === "indexing" ? "Indexing" : "Embedding";
      const settings = SETTINGS.filter((setting) => setting.group === group);
      this.body.content = settings
        .map(
          (setting) =>
            `${setting.name}=${this.state.values[setting.name] ?? (setting.default || "(default)")}`,
        )
        .join("\n");
    } else if (name === "summary") this.body.content = summaryText(this.state);
    else if (name === "progress") {
      this.body.content = "Running the installation…";
      void this.run();
    } else if (name === "done") {
      this.hint.content = "Esc to exit";
    }
  }

  private showInput(value: string): void {
    this.input.visible = true;
    this.input.value = value;
    this.input.focus();
  }

  private showSelect(
    options: { name: string; description: string; value: string }[],
    selected: string,
  ): void {
    this.select.visible = true;
    this.select.options = options;
    const index = options.findIndex((option) => option.value === selected);
    if (index >= 0) this.select.selectedIndex = index;
    this.select.focus();
  }

  private commit(): void {
    if (this.panel === "location" && this.input.visible) {
      this.state.installDirectory = this.input.value.trim() || this.state.installDirectory;
    }
    if (this.panel === "accelerator" && this.select.visible) {
      const selected = this.select.options[this.select.selectedIndex]?.value;
      this.state.accelerator = typeof selected === "string" ? selected : this.state.accelerator;
    }
    if (this.panel === "harnesses" && this.input.visible) {
      try {
        this.state.harnessSlugs = parseHarnessSelection(this.input.value);
      } catch {
        // Keep the previous selection; advance validates by staying put only on parse in CLI.
      }
    }
    if (this.panel === "path" && this.input.visible) {
      this.state.binDirectory = this.input.value.trim() || this.state.binDirectory;
    }
  }

  private async advance(): Promise<void> {
    this.commit();
    const order = [...PANEL_ORDER];
    const index = order.indexOf(this.panel);
    const next = order[index + 1];
    if (next === undefined) return;
    this.show(next);
  }

  private back(): void {
    const order = PANEL_ORDER.filter(
      (panel): panel is Exclude<PanelName, "progress" | "done"> =>
        panel !== "progress" && panel !== "done",
    );
    const index = order.indexOf(this.panel as Exclude<PanelName, "progress" | "done">);
    const previous = order[index - 1];
    if (previous === undefined) return;
    this.show(previous);
  }

  private cancel(): void {
    this.cancelled = true;
    this.finish(130);
  }

  private async run(): Promise<void> {
    const events: string[] = [];
    const onEvent = (event: StepEvent) => {
      events.push(`[${event.step}] ${event.status}: ${event.detail}`);
      this.body.content = events.join("\n");
    };
    try {
      const result = await runInstall(toPlan(this.state), onEvent, () => !this.cancelled);
      this.body.content = doneText(this.cancelled ? result : result, {
        cancelled: this.cancelled,
      });
      this.panel = "done";
      this.finish(this.cancelled ? 130 : result.failures.length > 0 ? 1 : 0);
    } catch (error) {
      this.body.content = doneText(null, {
        error: error instanceof Error ? error : new Error(String(error)),
      });
      this.panel = "done";
      this.finish(1);
    }
  }

  private finish(code: number): void {
    if (this.doneCode !== null) return;
    this.doneCode = code;
    this.renderer.destroy();
    this.resolveFinished(code);
  }
}
