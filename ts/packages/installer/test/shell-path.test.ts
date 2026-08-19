import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { writeServerLauncher } from "../src/accelerator.ts";
import {
  BLOCK_END,
  BLOCK_START,
  defaultBinDirectory,
  installLauncher,
  launcherOk,
  profileMentionsDirectory,
  removeLauncher,
  removePathBlock,
  updateProfile,
} from "../src/shell-path.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

describe("shell path", () => {
  test("creates a launcher pointing at the server executable", () => {
    if (process.platform.startsWith("win")) return;
    workspace = temporaryDirectory();
    const checkout = path.join(workspace, "checkout");
    fs.mkdirSync(path.join(checkout, "ts", "packages", "server", "src"), { recursive: true });
    fs.writeFileSync(
      path.join(checkout, "ts", "packages", "server", "src", "cli.ts"),
      "export {}\n",
    );
    writeServerLauncher(checkout);
    const bin = path.join(workspace, "bin");
    const result = installLauncher(checkout, bin);
    expect(launcherOk(result)).toBe(true);
    expect(fs.lstatSync(result.path).isSymbolicLink()).toBe(true);
    expect(removeLauncher(bin, checkout)).toBe(result.path);
    expect(fs.existsSync(result.path)).toBe(false);
  });

  test("writes and removes a marked PATH block", () => {
    workspace = temporaryDirectory();
    const profile = path.join(workspace, ".zshrc");
    fs.writeFileSync(profile, "alias ll='ls -l'\n");
    const bin = path.join(workspace, "bin");
    expect(updateProfile(profile, bin, { home: workspace })).toBe(true);
    const updated = fs.readFileSync(profile, "utf8");
    expect(updated).toContain(BLOCK_START);
    expect(updated).toContain(BLOCK_END);
    expect(profileMentionsDirectory(updated, bin, workspace)).toBe(true);
    expect(updateProfile(profile, bin, { home: workspace })).toBe(false);
    expect(removePathBlock(profile)).toBe(true);
    expect(fs.readFileSync(profile, "utf8")).toBe("alias ll='ls -l'\n");
  });

  test("honours XDG_BIN_HOME", () => {
    expect(
      defaultBinDirectory({
        home: "/tmp/home",
        environment: { XDG_BIN_HOME: "/custom/bin" },
      }),
    ).toBe("/custom/bin");
  });
});
