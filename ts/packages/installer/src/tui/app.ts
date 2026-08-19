import {
  BoxRenderable,
  createCliRenderer,
  InputRenderable,
  SelectRenderable,
  TextRenderable,
} from "@opentui/core";
import { runInstall, type StepEvent } from "../orchestrator.ts";
import { toPlan, type WizardState } from "../wizard.ts";
import { doneText } from "./content.ts";
import { WizardController, type WizardControl } from "./controller.ts";

type Renderer = Awaited<ReturnType<typeof createCliRenderer>>;
type InstallRunner = typeof runInstall;

export async function startInstallerApp(state: WizardState): Promise<number> {
  const renderer = await createCliRenderer({ exitOnCtrlC: false });
  const wizard = new InstallerWizard(renderer, state);
  renderer.root.add(wizard.root);
  wizard.render();
  return await wizard.finished;
}

export class InstallerWizard {
  readonly root: InstanceType<typeof BoxRenderable>;
  readonly finished: Promise<number>;
  private readonly resolveFinished: (code: number) => void;
  private readonly body: InstanceType<typeof TextRenderable>;
  private readonly title: InstanceType<typeof TextRenderable>;
  private readonly hint: InstanceType<typeof TextRenderable>;
  private readonly input: InstanceType<typeof InputRenderable>;
  private readonly select: InstanceType<typeof SelectRenderable>;
  private readonly renderer: Renderer;
  private readonly controller: WizardController;
  private readonly installRunner: InstallRunner;
  private cancelled = false;
  private doneCode: number | null = null;
  private closed = false;

  constructor(renderer: Renderer, state: WizardState, installRunner: InstallRunner = runInstall) {
    this.renderer = renderer;
    this.controller = new WizardController(state);
    this.installRunner = installRunner;
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
      if (event.name === "escape") {
        this.cancelOrClose();
        return;
      }
      if (this.controller.panel === "progress" || this.controller.panel === "done") return;
      if (event.ctrl === true && event.name === "n") void this.advance();
      if (event.ctrl === true && event.name === "b") {
        this.controller.back();
        this.render();
      }
    });
  }

  render(): void {
    const view = this.controller.view();
    this.title.content = `Code Indexing MCP Installer  ${view.title}`;
    this.body.content = view.error === null ? view.body : `${view.body}\n\nError: ${view.error}`;
    this.input.visible = false;
    this.select.visible = false;
    this.showControl(view.control);
  }

  private showControl(control: WizardControl): void {
    if (control.kind === "input") {
      this.input.visible = true;
      this.input.value = control.value;
      this.input.placeholder = control.placeholder;
      this.input.focus();
      return;
    }
    if (control.kind === "select") {
      this.select.visible = true;
      this.select.options = [...control.choices];
      const index = control.choices.findIndex((choice) => choice.value === control.selected);
      this.select.selectedIndex = Math.max(0, index);
      this.select.focus();
    }
  }

  private controlValue(): string | null {
    if (this.input.visible) return this.input.value.trim();
    if (this.select.visible) {
      const value = this.select.getSelectedOption()?.value;
      return typeof value === "string" ? value : null;
    }
    return null;
  }

  private async advance(): Promise<void> {
    const result = this.controller.advance(this.controlValue());
    this.render();
    if (result.startInstall) await this.run();
  }

  private cancelOrClose(): void {
    if (this.controller.panel === "done") {
      this.close(this.doneCode ?? 0);
      return;
    }
    if (this.controller.panel === "progress") {
      this.cancelled = true;
      this.hint.content = "Cancelling after the current step...";
      return;
    }
    this.close(130);
  }

  private async run(): Promise<void> {
    this.input.visible = false;
    this.select.visible = false;
    this.title.content = "Code Indexing MCP Installer  Installing";
    this.hint.content = "Esc cancels after the current step";
    const events: string[] = [];
    const onEvent = (event: StepEvent) => {
      events.push(`[${event.step}] ${event.status}: ${event.detail}`);
      this.body.content = events.join("\n");
    };
    try {
      const result = await this.installRunner(
        toPlan(this.controller.state),
        onEvent,
        () => !this.cancelled,
      );
      this.doneCode = this.cancelled ? 130 : result.failures.length > 0 ? 1 : 0;
      this.showDone(doneText(result, { cancelled: this.cancelled }));
    } catch (error) {
      this.doneCode = 1;
      this.showDone(
        doneText(null, { error: error instanceof Error ? error : new Error(String(error)) }),
      );
    }
  }

  private showDone(content: string): void {
    this.controller.showDone();
    this.title.content = "Code Indexing MCP Installer  Done";
    this.body.content = content;
    this.hint.content = "Esc exit";
    this.input.visible = false;
    this.select.visible = false;
  }

  private close(code: number): void {
    if (this.closed) return;
    this.closed = true;
    this.renderer.destroy();
    this.resolveFinished(code);
  }
}
