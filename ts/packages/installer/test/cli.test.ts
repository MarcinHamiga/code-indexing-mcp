import { describe, expect, test } from "bun:test";
import { parseArgv, parseSettings } from "../src/cli.ts";
import { InstallerError } from "../src/config-files.ts";

describe("installer CLI", () => {
  test("parses managed settings", () => {
    expect(parseSettings(["CODE_INDEXING_BROKER=off"], ["CODE_INDEXING_OFFLINE"])).toEqual({
      CODE_INDEXING_BROKER: "off",
      CODE_INDEXING_OFFLINE: null,
    });
  });

  test("rejects unknown settings", () => {
    expect(() => parseSettings(["NOPE=1"], [])).toThrow(InstallerError);
  });

  test("forwards launcher flags without implying a scripted run", () => {
    const args = parseArgv(["--bin-dir", "/tmp/bin", "--no-modify-path"]);
    expect(args.binDir).toBe("/tmp/bin");
    expect(args.noModifyPath).toBe(true);
    expect(args.noLauncher).toBe(false);
    expect(args.tui).toBe(false);
    expect(args.harnesses).toBeNull();
  });
});
