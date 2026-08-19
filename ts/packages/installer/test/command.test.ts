import { afterEach, describe, expect, test } from "bun:test";
import { commandHandlers, main } from "../src/command.ts";

const restored: Array<() => void> = [];

afterEach(() => {
  while (restored.length > 0) restored.pop()?.();
});

describe("installed command dispatch", () => {
  test("configure forwards the complete scripted option set", async () => {
    const original = commandHandlers.configure;
    let received: unknown = null;
    commandHandlers.configure = async (options) => {
      received = options;
      return 7;
    };
    restored.push(() => {
      commandHandlers.configure = original;
    });

    expect(
      await main([
        "configure",
        "--install-dir",
        "/opt/ci-mcp",
        "--accelerator",
        "cpu",
        "--harnesses",
        "codex",
        "--set",
        "CODE_INDEXING_OFFLINE=1",
        "--unset",
        "CODE_INDEXING_BROKER",
        "--bin-dir",
        "/opt/bin",
        "--no-launcher",
        "--no-modify-path",
        "--no-tui",
        "--repair",
      ]),
    ).toBe(7);
    expect(received).toEqual({
      installDir: "/opt/ci-mcp",
      accelerator: "cpu",
      harnesses: "codex",
      settings: ["CODE_INDEXING_OFFLINE=1"],
      unsets: ["CODE_INDEXING_BROKER"],
      noTui: true,
      binDir: "/opt/bin",
      noLauncher: true,
      noModifyPath: true,
      repair: true,
    });
  });

  test("update forwards its internal handoff options", async () => {
    const original = commandHandlers.update;
    let received: unknown = null;
    commandHandlers.update = async (options) => {
      received = options;
      return 10;
    };
    restored.push(() => {
      commandHandlers.update = original;
    });

    expect(
      await main([
        "update",
        "--install-dir",
        "/opt/ci-mcp",
        "--check",
        "--skip-accelerator",
        "--finalize",
        "--previous-sha",
        "abc",
      ]),
    ).toBe(10);
    expect(received).toEqual({
      installDir: "/opt/ci-mcp",
      check: true,
      skipAccelerator: true,
      finalize: true,
      previousSha: "abc",
    });
  });

  test("uninstall forwards destructive choices explicitly", async () => {
    const original = commandHandlers.uninstall;
    let received: unknown = null;
    commandHandlers.uninstall = async (options) => {
      received = options;
      return 0;
    };
    restored.push(() => {
      commandHandlers.uninstall = original;
    });

    expect(
      await main([
        "uninstall",
        "--install-dir",
        "/opt/ci-mcp",
        "--harnesses",
        "all",
        "--bin-dir",
        "/opt/bin",
        "--keep-launcher",
        "--keep-path",
        "--purge",
        "--remove-checkout",
        "--yes",
      ]),
    ).toBe(0);
    expect(received).toEqual({
      installDir: "/opt/ci-mcp",
      harnessesSelection: "all",
      binDir: "/opt/bin",
      keepLauncher: true,
      keepPath: true,
      purge: true,
      removeCheckout: true,
      assumeYes: true,
    });
  });
});
