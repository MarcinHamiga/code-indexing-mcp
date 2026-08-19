import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { parse as parseToml } from "smol-toml";
import { InstallerError, mergeCodexServer, mergeJsonObjectEntry } from "../src/config-files.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

const SERVER_COMMAND = String(path.sep === "\\" ? "C:\\opt\\ci-mcp" : "/opt/ci-mcp");

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

describe("json/jsonc merge", () => {
  test("creates parent and top-level object", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "nested", "config.json");
    const changed = mergeJsonObjectEntry(filePath, "mcpServers", "code-indexing-mcp", {
      command: "/opt/ci-mcp",
      args: ["serve"],
    });
    expect(changed).toBe(true);
    expect(JSON.parse(fs.readFileSync(filePath, "utf8"))).toEqual({
      mcpServers: {
        "code-indexing-mcp": { command: "/opt/ci-mcp", args: ["serve"] },
      },
    });
    expect(fs.existsSync(`${filePath}.bak`)).toBe(false);
  });

  test("preserves comments, trailing commas, and unrelated entries", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.jsonc");
    const original = `{
  // This setting belongs to the user.
  "theme": "dark",
  "mcp": {
    "existing": {
      "enabled": false,
    },
    "code-indexing-mcp": {"old": true}, // keep this note
  },
}
`;
    fs.writeFileSync(filePath, original);
    const changed = mergeJsonObjectEntry(filePath, "mcp", "code-indexing-mcp", {
      type: "local",
      command: ["/opt/ci-mcp", "serve"],
      enabled: true,
    });
    const updated = fs.readFileSync(filePath, "utf8");
    expect(changed).toBe(true);
    expect(updated).toContain("// This setting belongs to the user.");
    expect(updated).toContain('"theme": "dark"');
    expect(updated).toContain('"existing": {');
    expect(updated).toContain("// keep this note");
    expect(updated).not.toContain('"old"');
    expect(updated.split('"code-indexing-mcp"').length - 1).toBe(1);
    expect(fs.readFileSync(`${filePath}.bak`, "utf8")).toBe(original);
  });

  test("is idempotent", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.jsonc");
    const entry = { command: "/opt/ci-mcp", args: ["serve"] };
    expect(mergeJsonObjectEntry(filePath, "mcpServers", "server", entry)).toBe(true);
    const first = fs.readFileSync(filePath, "utf8");
    expect(mergeJsonObjectEntry(filePath, "mcpServers", "server", entry)).toBe(false);
    expect(fs.readFileSync(filePath, "utf8")).toBe(first);
  });

  test("rejects invalid input without modifying it", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.jsonc");
    const original = '{"mcp": [}';
    fs.writeFileSync(filePath, original);
    expect(() => mergeJsonObjectEntry(filePath, "mcp", "server", { enabled: true })).toThrow(
      InstallerError,
    );
    expect(fs.readFileSync(filePath, "utf8")).toBe(original);
    expect(fs.existsSync(`${filePath}.bak`)).toBe(false);
  });

  test("validates unrelated nested values", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.jsonc");
    const original = '{"unrelated": {"broken": nope}}\n';
    fs.writeFileSync(filePath, original);
    expect(() => mergeJsonObjectEntry(filePath, "mcp", "server", { enabled: true })).toThrow(
      InstallerError,
    );
    expect(fs.readFileSync(filePath, "utf8")).toBe(original);
    expect(fs.existsSync(`${filePath}.bak`)).toBe(false);
  });
});

describe("codex toml merge", () => {
  test("creates server table", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.toml");
    expect(mergeCodexServer(filePath, SERVER_COMMAND)).toBe(true);
    expect(fs.readFileSync(filePath, "utf8")).toBe(
      `[mcp_servers.code-indexing-mcp]\ncommand = ${JSON.stringify(SERVER_COMMAND)}\nargs = ["serve"]\n`,
    );
  });

  test("replaces only the target table and subtables", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.toml");
    const original = `# Keep this comment.
model = "gpt-5"

[mcp_servers.other]
command = "other"

[mcp_servers.code-indexing-mcp]
command = "old"
args = ["old"]

[mcp_servers.code-indexing-mcp.env]
OLD = "value"

# Keep the feature explanation too.
[features]
example = true
`;
    fs.writeFileSync(filePath, original);
    expect(mergeCodexServer(filePath, "/new/ci-mcp")).toBe(true);
    const updated = fs.readFileSync(filePath, "utf8");
    expect(updated).toContain("# Keep this comment.");
    expect(updated).toContain('[mcp_servers.other]\ncommand = "other"');
    expect(updated).toContain("# Keep the feature explanation too.");
    expect(updated).toContain("[features]\nexample = true");
    expect(updated).not.toContain('command = "old"');
    expect(updated).not.toContain("OLD");
    expect(updated.split("[mcp_servers.code-indexing-mcp]").length - 1).toBe(1);
    expect(updated).toContain(`command = ${JSON.stringify("/new/ci-mcp")}`);
    expect(fs.readFileSync(`${filePath}.bak`, "utf8")).toBe(original);
  });

  test("preserves following array tables", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.toml");
    fs.writeFileSync(
      filePath,
      `[mcp_servers.code-indexing-mcp]
command = "old"

[[skills.config]]
path = "/tmp/skill"
enabled = false
`,
    );
    mergeCodexServer(filePath, "/new/ci-mcp");
    const parsed = parseToml(fs.readFileSync(filePath, "utf8")) as {
      skills: { config: unknown };
    };
    expect(parsed.skills.config).toEqual([{ path: "/tmp/skill", enabled: false }]);
    expect(fs.readFileSync(`${filePath}.bak`, "utf8")).toContain('command = "old"');
  });

  test("is idempotent", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.toml");
    expect(mergeCodexServer(filePath, "/opt/ci-mcp")).toBe(true);
    const first = fs.readFileSync(filePath, "utf8");
    expect(mergeCodexServer(filePath, "/opt/ci-mcp")).toBe(false);
    expect(fs.readFileSync(filePath, "utf8")).toBe(first);
  });

  test("rejects inline target without corrupting config", () => {
    workspace = temporaryDirectory();
    const filePath = path.join(workspace, "config.toml");
    const original = '[mcp_servers]\ncode-indexing-mcp = { command = "old", args = ["old"] }\n';
    fs.writeFileSync(filePath, original);
    expect(() => mergeCodexServer(filePath, "/opt/ci-mcp")).toThrow(/inline or dotted/);
    expect(fs.readFileSync(filePath, "utf8")).toBe(original);
    expect(fs.existsSync(`${filePath}.bak`)).toBe(false);
  });
});
