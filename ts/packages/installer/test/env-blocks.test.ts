import { describe, expect, test } from "bun:test";
import { commandFromEntry, entryFromText, envFromEntry, mergeEnv } from "../src/env-blocks.ts";

describe("env blocks", () => {
  test("merges managed updates and deletes nulls", () => {
    expect(
      mergeEnv(
        { KEEP: "x", CODE_INDEXING_BROKER: "off" },
        {
          CODE_INDEXING_BROKER: null,
          CODE_INDEXING_INDEX_MODE: "eager",
        },
      ),
    ).toEqual({ KEEP: "x", CODE_INDEXING_INDEX_MODE: "eager" });
  });

  test("parses a JSON server entry", () => {
    const entry = entryFromText(
      "kimi-code",
      JSON.stringify({
        mcpServers: { "code-indexing-mcp": { command: "/opt/ci-mcp", env: { A: "1" } } },
      }),
    );
    expect(entry).not.toBeNull();
    expect(commandFromEntry("kimi-code", entry ?? {})).toBe("/opt/ci-mcp");
    expect(envFromEntry("kimi-code", entry ?? {})).toEqual({ A: "1" });
  });

  test("reads the first list item as the OpenCode command", () => {
    expect(commandFromEntry("opencode", { command: ["/opt/ci-mcp", "serve"] })).toBe("/opt/ci-mcp");
  });
});
