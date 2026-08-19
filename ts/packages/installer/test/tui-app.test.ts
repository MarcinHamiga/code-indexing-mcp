import { describe, expect, test } from "bun:test";
import { createTestRenderer } from "@opentui/core/testing";
import fs from "node:fs";
import path from "node:path";
import { serverExecutable } from "../src/accelerator.ts";
import type { InstallResult } from "../src/orchestrator.ts";
import { InstallerWizard } from "../src/tui/app.ts";
import type { WizardState } from "../src/wizard.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

function result(): InstallResult {
  return {
    acceleratorPlan: null,
    configured: [],
    failures: [],
    skills: [],
    launcher: null,
    profilesUpdated: [],
    checks: [],
  };
}

describe("OpenTUI application", () => {
  test("keeps the completion screen visible until Escape", async () => {
    const workspace = temporaryDirectory();
    try {
      const installDirectory = path.join(workspace, "install");
      const executable = serverExecutable(installDirectory);
      fs.mkdirSync(path.dirname(executable), { recursive: true });
      fs.writeFileSync(executable, "launcher");
      const state: WizardState = {
        mode: "reconfigure",
        installDirectory,
        accelerator: null,
        preparedAccelerator: "cpu",
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
      const { renderer, mockInput, renderOnce, captureCharFrame } = await createTestRenderer({
        width: 100,
        height: 30,
      });
      const wizard = new InstallerWizard(renderer, state, async () => result());
      renderer.root.add(wizard.root);
      wizard.render();
      let settled = false;
      void wizard.finished.then(() => {
        settled = true;
      });

      for (let index = 0; index < 27; index += 1) {
        mockInput.pressKey("n", { ctrl: true });
        await Bun.sleep(1);
      }
      await renderOnce();
      expect(captureCharFrame()).toContain("Installation complete");
      expect(settled).toBe(false);
      mockInput.pressEscape();
      expect(await wizard.finished).toBe(0);
    } finally {
      removeDirectory(workspace);
    }
  });
});
