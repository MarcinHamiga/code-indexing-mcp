import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { parse as parseToml } from "smol-toml";
import { envFromEntry } from "../src/env-blocks.ts";
import {
  configureHarness,
  HARNESS_CHOICES,
  parseHarnessSelection,
  readServerEntry,
} from "../src/harnesses.ts";
import { InstallerError } from "../src/config-files.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

const SERVER_COMMAND = "/opt/ci-mcp";
let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

describe("harness selection", () => {
  test("accepts numbers, slugs, and all", () => {
    expect(parseHarnessSelection("1,codex")).toEqual(["codex"]);
    expect(parseHarnessSelection("all")).toEqual(HARNESS_CHOICES.map((choice) => choice.slug));
    expect(parseHarnessSelection("")).toEqual([]);
    expect(() => parseHarnessSelection("nope")).toThrow(InstallerError);
  });
});

describe("configureHarness", () => {
  test("writes env and preserves unmanaged keys", () => {
    workspace = temporaryDirectory();
    const config = path.join(workspace, "mcp.json");
    fs.writeFileSync(
      config,
      JSON.stringify({
        mcpServers: {
          "code-indexing-mcp": {
            command: "/old",
            args: ["serve"],
            env: { KEEP: "x", CODE_INDEXING_BROKER: "off" },
          },
        },
      }),
    );
    configureHarness("kimi-code", SERVER_COMMAND, {
      env: { CODE_INDEXING_BROKER: null, CODE_INDEXING_INDEX_MODE: "eager" },
      environment: { KIMI_CODE_HOME: workspace },
    });
    const entry = JSON.parse(fs.readFileSync(config, "utf8")).mcpServers["code-indexing-mcp"];
    expect(entry).toEqual({
      command: SERVER_COMMAND,
      args: ["serve"],
      env: { KEEP: "x", CODE_INDEXING_INDEX_MODE: "eager" },
    });
  });

  test("writes a Codex TOML env table", () => {
    workspace = temporaryDirectory();
    configureHarness("codex", SERVER_COMMAND, {
      env: { CODE_INDEXING_OFFLINE: "1" },
      environment: { CODEX_HOME: workspace },
    });
    const parsed = parseToml(fs.readFileSync(path.join(workspace, "config.toml"), "utf8")) as {
      mcp_servers: { "code-indexing-mcp": Record<string, unknown> };
    };
    expect(parsed.mcp_servers["code-indexing-mcp"]).toEqual({
      command: SERVER_COMMAND,
      args: ["serve"],
      env: { CODE_INDEXING_OFFLINE: "1" },
    });
  });

  test("reads back the server entry", () => {
    workspace = temporaryDirectory();
    configureHarness("kimi-code", SERVER_COMMAND, {
      env: {},
      environment: { KIMI_CODE_HOME: workspace },
    });
    const entry = readServerEntry("kimi-code", { environment: { KIMI_CODE_HOME: workspace } });
    expect(entry).not.toBeNull();
    expect(envFromEntry("kimi-code", entry ?? {})).toEqual({});
  });
});
