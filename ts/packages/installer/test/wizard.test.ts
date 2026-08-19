import { afterEach, describe, expect, test } from "bun:test";
import path from "node:path";
import { configureHarness } from "../src/harnesses.ts";
import {
  envUpdates,
  loadPrefill,
  toPlan,
  wizardStateForInstall,
  wizardStateForReconfigure,
} from "../src/wizard.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

describe("wizard state", () => {
  test("prefills configured harnesses and unsets defaults", () => {
    workspace = temporaryDirectory();
    configureHarness("kimi-code", "/opt/ci-mcp", {
      env: { CODE_INDEXING_BROKER: "off" },
      home: workspace,
      environment: { KIMI_CODE_HOME: workspace },
    });
    const prefill = loadPrefill({
      home: workspace,
      environment: { KIMI_CODE_HOME: workspace },
    });
    expect(prefill.configuredSlugs).toEqual(["kimi-code"]);
    expect(prefill.values.CODE_INDEXING_BROKER).toBe("off");
    const state = wizardStateForReconfigure(path.join(workspace, "install"), {
      home: workspace,
      environment: { KIMI_CODE_HOME: workspace },
    });
    expect(state.accelerator).toBeNull();
    expect(state.harnessSlugs).toEqual(["kimi-code"]);
    delete state.values.CODE_INDEXING_BROKER;
    expect(envUpdates(state)).toEqual({ CODE_INDEXING_BROKER: null });
  });

  test("install mode defaults accelerator to auto", () => {
    workspace = temporaryDirectory();
    const state = wizardStateForInstall(path.join(workspace, "install"), {
      home: workspace,
      environment: {},
    });
    expect(state.accelerator).toBe("auto");
    expect(toPlan(state).installLauncher).toBe(true);
  });
});
