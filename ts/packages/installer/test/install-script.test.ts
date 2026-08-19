import { afterEach, describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

let workspace = "";

afterEach(() => {
  if (workspace !== "") removeDirectory(workspace);
  workspace = "";
});

describe("shell bootstrap", () => {
  test.skipIf(process.platform === "win32")("provisions Bun when it is absent", () => {
    workspace = temporaryDirectory();
    const home = path.join(workspace, "home");
    const bunHome = path.join(workspace, "bun");
    const fakeBunSource = path.join(workspace, "fake-bun");
    const invocation = path.join(workspace, "invocation.txt");
    const bunInstaller = path.join(workspace, "bun-installer.sh");
    fs.mkdirSync(home);
    fs.writeFileSync(fakeBunSource, `#!/bin/sh\nprintf '%s\\n' "$*" > "${invocation}"\n`, {
      mode: 0o755,
    });
    fs.writeFileSync(
      bunInstaller,
      '#!/bin/sh\nmkdir -p "$BUN_INSTALL/bin"\ncp "$FAKE_BUN_SOURCE" "$BUN_INSTALL/bin/bun"\nchmod +x "$BUN_INSTALL/bin/bun"\n',
      { mode: 0o755 },
    );
    const script = path.resolve(import.meta.dir, "../../../install.sh");
    const result = spawnSync("/bin/sh", [script, "--help"], {
      cwd: workspace,
      encoding: "utf8",
      env: {
        HOME: home,
        PATH: "/usr/bin:/bin",
        BUN_INSTALL: bunHome,
        FAKE_BUN_SOURCE: fakeBunSource,
        CODE_INDEXING_MCP_BUN_INSTALLER_URL: `file://${bunInstaller}`,
      },
    });
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("Bun was not found; installing it");
    expect(fs.readFileSync(invocation, "utf8")).toContain("bootstrap.ts --help");
  });
});
