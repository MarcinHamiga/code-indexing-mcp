import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { serverExecutable } from "../src/accelerator.ts";
import { WizardController } from "../src/tui/controller.ts";
import type { WizardState } from "../src/wizard.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

function state(mode: "install" | "reconfigure" = "reconfigure"): WizardState {
  workspace = temporaryDirectory();
  const installDirectory = path.join(workspace, "install");
  const executable = serverExecutable(installDirectory);
  fs.mkdirSync(path.dirname(executable), { recursive: true });
  fs.writeFileSync(executable, "launcher");
  return {
    mode,
    installDirectory,
    accelerator: mode === "reconfigure" ? null : "auto",
    preparedAccelerator: "cuda",
    harnessSlugs: [],
    configuredSlugs: [],
    values: {},
    prefilledNames: new Set(),
    disagreements: [],
    offline: false,
    binDirectory: path.join(workspace, "bin"),
    installLauncher: true,
    modifyShellProfiles: true,
  };
}

function reachPath(controller: WizardController): void {
  controller.advance(null);
  controller.advance(controller.state.installDirectory);
  controller.advance("__keep__");
  controller.advance("");
  expect(controller.panel).toBe("path");
}

describe("OpenTUI wizard controller", () => {
  test("reconfiguration keeps the prepared accelerator when selected", () => {
    const wizardState = state();
    const controller = new WizardController(wizardState);
    controller.advance(null);
    controller.advance(wizardState.installDirectory);
    const accelerator = controller.view();
    expect(accelerator.control.kind).toBe("select");
    if (accelerator.control.kind === "select") {
      expect(accelerator.control.choices.map((choice) => choice.value)).toContain("__keep__");
    }
    expect(controller.advance("__keep__").error).toBeNull();
    expect(wizardState.accelerator).toBeNull();
  });

  test("path choices and settings are editable", () => {
    const wizardState = state();
    const controller = new WizardController(wizardState);
    reachPath(controller);
    controller.advance("no");
    controller.advance("no");
    controller.advance("");
    expect(wizardState.installLauncher).toBe(false);
    expect(wizardState.modifyShellProfiles).toBe(false);
    expect(controller.panel).toBe("indexing");
    expect(controller.advance("eager").error).toBeNull();
    expect(wizardState.values.CODE_INDEXING_INDEX_MODE).toBe("eager");
    const invalid = controller.advance("-1");
    expect(invalid.error).toContain("CODE_INDEXING_INDEX_WAIT_SECONDS");
    expect(controller.panel).toBe("indexing");
  });

  test("invalid harness input blocks navigation and remains visible", () => {
    const wizardState = state();
    const controller = new WizardController(wizardState);
    controller.advance(null);
    controller.advance(wizardState.installDirectory);
    controller.advance("__keep__");
    expect(controller.panel).toBe("harnesses");
    expect(controller.advance("not-a-harness").error).toContain("Unknown harness");
    expect(controller.panel).toBe("harnesses");
    expect(controller.view().error).toContain("Unknown harness");
  });
});
