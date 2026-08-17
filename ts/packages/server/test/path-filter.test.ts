/**
 * Glob-to-regex translation for the search path pushdown.
 *
 * The Python suite proves the translation against `PurePosixPath.match`
 * directly, which is only possible where that function exists. Here the oracle
 * is a committed fixture holding what the Python build actually produces --
 * every emitted regex, and the ground-truth match of every pattern against every
 * corpus path -- so a divergence between the two builds fails here rather than
 * silently returning different search results. Regenerate it with
 * `scripts/write_path_filter_parity.py` whenever either side changes.
 */

import { describe, expect, test } from "bun:test";
import fs from "node:fs";
import { globToRegex, pathCondition } from "../src/path-filter.ts";

interface Fixture {
  paths: string[];
  patterns: { pattern: string; regex: string | null; matches: string }[];
  untranslatable: string[];
  conditions: { patterns: string[]; condition: string | null }[];
}

const fixture: Fixture = JSON.parse(
  fs.readFileSync(new URL("./fixtures/path-filter.json", import.meta.url), "utf8"),
);

describe("parity with the Python translation", () => {
  test.each(fixture.patterns.map((entry) => [entry.pattern, entry] as const))(
    "%s emits the same expression Python does",
    (_pattern, entry) => {
      expect(globToRegex(entry.pattern)).toBe(entry.regex);
    },
  );

  test.each(fixture.patterns.map((entry) => [entry.pattern, entry] as const))(
    "%s agrees with PurePosixPath.match on every corpus path",
    (_pattern, entry) => {
      // A narrower pushdown loses results, which is the bug this module exists
      // to fix; a broader one only wastes rows. Assert exact equivalence so
      // neither drifts. The `u` flag is deliberately absent: `re.escape` emits
      // identity escapes such as `\ ` and `\~` that Unicode mode rejects and
      // that LanceDB's engine already accepts from the shipping build.
      expect(entry.regex).not.toBeNull();
      const compiled = new RegExp(entry.regex as string);
      const actual = fixture.paths.map((path) => (compiled.test(path) ? "1" : "0")).join("");
      expect(actual).toBe(entry.matches);
    },
  );
});

test("patterns Python refuses are refused here too", () => {
  for (const pattern of fixture.untranslatable) {
    expect(globToRegex(pattern)).toBeNull();
  }
});

test("path conditions match Python's, quoting included", () => {
  for (const entry of fixture.conditions) {
    expect(pathCondition(entry.patterns)).toBe(entry.condition);
  }
});

test("redundant and trailing separators are normalized", () => {
  expect(globToRegex("*/")).toBe(globToRegex("*"));
  expect(globToRegex("foo//bar")).toBe(globToRegex("foo/bar"));
});

test("path_condition ORs every pattern", () => {
  expect(pathCondition(["rare/*", "tests/*"])).toBe(
    "(regexp_like(path, '(^|/)rare/[^/]*$') OR regexp_like(path, '(^|/)tests/[^/]*$'))",
  );
});

test("one untranslatable pattern disables the whole pushdown", () => {
  // Patterns are OR-ed, so one pattern we cannot express would make the whole
  // predicate too narrow. Skip the pushdown rather than lose rows.
  expect(pathCondition(["src/*", "/absolute"])).toBeNull();
  expect(pathCondition([])).toBeNull();
});

test("single quotes are doubled for the SQL literal", () => {
  const condition = pathCondition(["it's.py"]);
  expect(condition).not.toBeNull();
  expect(condition).toContain("it''s");
});

test("the column is configurable", () => {
  expect(pathCondition(["*.py"], { column: "file_path" })).toBe(
    "(regexp_like(file_path, '(^|/)[^/]*\\.py$'))",
  );
});
