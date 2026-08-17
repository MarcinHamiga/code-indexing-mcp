import { describe, expect, test } from "bun:test";
import { runtimeName, runtimeVersion } from "../src/runtime/index.ts";

describe("runtime detection", () => {
  test("identifies the executing runtime", () => {
    // The suite is run by `bun test`, so this is also a check that the harness
    // itself is what the CI gate claims to be running.
    expect(runtimeName()).toBe("bun");
  });

  test("reports a version string", () => {
    expect(runtimeVersion()).toMatch(/^\d+\.\d+\.\d+/);
  });
});
