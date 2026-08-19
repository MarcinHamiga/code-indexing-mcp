import { describe, expect, test } from "bun:test";
import path from "node:path";
import { doneText, summaryText, welcomeText } from "../src/tui/content.ts";
import { wizardStateForInstall } from "../src/wizard.ts";

describe("wizard panel text", () => {
  test("welcome describes an install", () => {
    const state = wizardStateForInstall("/tmp/install", { environment: {} });
    expect(welcomeText(state)).toContain("Install Code Indexing MCP");
    expect(welcomeText(state)).toContain("/tmp/install");
  });

  test("summary lists harnesses and settings", () => {
    const state = wizardStateForInstall("/tmp/install", { environment: {} });
    state.harnessSlugs = ["kimi-code"];
    state.values.CODE_INDEXING_BROKER = "off";
    const text = summaryText(state);
    expect(text).toContain("Accelerator: auto");
    expect(text).toContain("Harnesses: kimi-code");
    expect(text).toContain("CODE_INDEXING_BROKER = off");
  });

  test("done panel mentions the launcher and exec hint", () => {
    const text = doneText({
      acceleratorPlan: null,
      configured: [],
      failures: [],
      skills: [],
      launcher: {
        path: path.join("/tmp", "bin", "code-indexing-mcp"),
        status: "created",
        detail: "points at it",
      },
      profilesUpdated: [path.join("/tmp", ".zshrc")],
      checks: [],
    });
    expect(text).toContain("code-indexing-mcp");
    expect(text).toContain(".zshrc");
    expect(text).toContain("exec");
  });
});
