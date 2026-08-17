/**
 * Gitignore semantics, held to what the Python build actually decides.
 *
 * The scanner reads gitignore syntax in three places -- include patterns,
 * config excludes, and the nested `.gitignore` stack -- and the port swaps
 * `pathspec.GitIgnoreSpec` for the `ignore` npm package. That is a *second*
 * independent implementation of a specification whose corners (anchoring,
 * directory-only patterns, negation ordering, `**` spans, character classes)
 * are precisely where implementations drift, and a drift here changes which
 * files get indexed rather than raising anything.
 *
 * So the oracle is the shipping build's own verdicts, recorded by
 * `scripts/write_ignore_parity.py`: `match_file` for every pattern set against
 * every corpus path, plus the tri-state `check_file(...).include` the nested
 * stack folds. Regenerate the fixture whenever either side changes; never edit
 * it to make this pass.
 */

import { describe, expect, test } from "bun:test";
import fs from "node:fs";
import { compileSpec } from "../src/scanner.ts";

interface Fixture {
  paths: string[];
  specs: Array<{
    patterns: string[];
    matches: boolean[];
    include: Array<boolean | null>;
  }>;
}

const fixture: Fixture = JSON.parse(
  fs.readFileSync(new URL("./fixtures/ignore.json", import.meta.url), "utf8"),
);

describe("parity with pathspec.GitIgnoreSpec", () => {
  test.each(fixture.specs.map((spec) => [spec.patterns.join(" | "), spec] as const))(
    "%s matches the same paths",
    (_label, spec) => {
      const compiled = compileSpec(spec.patterns);

      expect(fixture.paths.map((item) => compiled.ignores(item))).toEqual(spec.matches);
    },
  );

  test.each(fixture.specs.map((spec) => [spec.patterns.join(" | "), spec] as const))(
    "%s reports the same tri-state verdict",
    (_label, spec) => {
      const compiled = compileSpec(spec.patterns);

      // pathspec answers True (ignored), False (re-included by a negation), or
      // None (no pattern matched, so an outer directory's verdict stands). The
      // third case is the one that matters: collapsing it to "not ignored" would
      // let a nested `.gitignore` that says nothing about a file silently undo
      // its parent's rule.
      const verdicts = fixture.paths.map((item) => {
        const result = compiled.test(item);
        if (result.ignored) return true;
        if (result.unignored) return false;
        return null;
      });

      expect(verdicts).toEqual(spec.include);
    },
  );

  test("the fixture covers a meaningful number of decisions", () => {
    // A fixture that quietly shrank would turn this file green without testing
    // anything, which is the failure mode a generated oracle invites.
    const decisions = fixture.specs.length * fixture.paths.length;

    expect(decisions).toBeGreaterThan(500);
  });
});
