import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { writeServerLauncher } from "../src/accelerator.ts";
import { configureHarness } from "../src/harnesses.ts";
import { installLauncher, updateProfile } from "../src/shell-path.ts";
import { runUninstall } from "../src/uninstall.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

function checkout(root: string): string {
  const directory = path.join(root, "checkout");
  fs.mkdirSync(path.join(directory, "ts", "packages", "server", "src"), { recursive: true });
  fs.writeFileSync(
    path.join(directory, "ts", "packages", "server", "src", "cli.ts"),
    "export {}\n",
  );
  writeServerLauncher(directory);
  return directory;
}

describe("uninstall", () => {
  test("takes back the entry, launcher, and PATH block", () => {
    if (process.platform.startsWith("win")) return;
    workspace = temporaryDirectory();
    const installDirectory = checkout(workspace);
    configureHarness("kimi-code", path.join(installDirectory, "bin", "code-indexing-mcp"), {
      env: {},
      home: workspace,
      environment: { KIMI_CODE_HOME: workspace },
    });
    const binDirectory = path.join(workspace, "bin");
    installLauncher(installDirectory, binDirectory);
    const profile = path.join(workspace, ".zshrc");
    fs.writeFileSync(profile, "alias ll='ls -l'\n");
    updateProfile(profile, binDirectory, { home: workspace });
    const result = runUninstall(
      {
        installDirectory,
        harnessSlugs: ["kimi-code"],
        binDirectory,
        removeLauncher: true,
        removePathBlock: true,
        removeData: false,
        removeCheckout: false,
      },
      () => undefined,
      { home: workspace, environment: { SHELL: "/bin/zsh", KIMI_CODE_HOME: workspace } },
    );
    expect(result.failures).toEqual([]);
    expect(result.launcherRemoved).toBe(path.join(binDirectory, "code-indexing-mcp"));
    expect(fs.existsSync(path.join(binDirectory, "code-indexing-mcp"))).toBe(false);
    expect(fs.readFileSync(profile, "utf8")).toBe("alias ll='ls -l'\n");
    expect(result.profilesCleared).toEqual([profile]);
  });

  test("purges marked data directories when asked", () => {
    workspace = temporaryDirectory();
    const data = path.join(workspace, "data");
    const cache = path.join(workspace, "cache");
    fs.mkdirSync(path.join(data, "lancedb"), { recursive: true });
    fs.mkdirSync(path.join(cache, "lancedb"), { recursive: true });
    const result = runUninstall(
      {
        installDirectory: checkout(workspace),
        harnessSlugs: [],
        binDirectory: null,
        removeLauncher: false,
        removePathBlock: false,
        removeData: true,
        removeCheckout: false,
      },
      () => undefined,
      {
        home: workspace,
        environment: {
          CODE_INDEXING_DATA_DIR: data,
          CODE_INDEXING_CACHE_DIR: cache,
          SHELL: "/bin/zsh",
        },
      },
    );
    expect(new Set(result.directoriesRemoved)).toEqual(new Set([data, cache]));
    expect(fs.existsSync(data)).toBe(false);
    expect(fs.existsSync(cache)).toBe(false);
  });
});
