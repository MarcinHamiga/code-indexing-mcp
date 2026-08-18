import { describe, expect, test } from "bun:test";
import { comparePythonPaths, comparePythonStrings, pythonJsonDumps } from "../src/python-compat.ts";

describe("Python-compatible ordering", () => {
  test("strings compare by code point rather than UTF-16 code unit", () => {
    expect(["😀.py", "\ue000.py"].sort(comparePythonStrings)).toEqual(["\ue000.py", "😀.py"]);
  });

  test.skipIf(process.platform === "win32")("paths compare one component at a time", () => {
    expect(["a-.py", "a/file.py"].sort(comparePythonPaths)).toEqual(["a/file.py", "a-.py"]);
  });
});

describe("Python-compatible JSON", () => {
  test("sorts keys and ASCII-escapes Unicode", () => {
    expect(pythonJsonDumps({ new_name: "café", kind: "rename" })).toBe(
      '{"kind":"rename","new_name":"caf\\u00e9"}',
    );
  });

  test("escapes non-BMP characters as a surrogate pair", () => {
    expect(pythonJsonDumps(["😀"])).toBe('["\\ud83d\\ude00"]');
  });
});
