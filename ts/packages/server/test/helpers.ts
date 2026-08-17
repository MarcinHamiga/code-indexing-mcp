/** Helpers shared by the test suite -- the `conftest.py` of this tree. */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * The repository root, so tests can reach the shared fixture corpus.
 *
 * The migration plan (§7) has both suites read the same
 * `tests/fixtures/` files rather than each keeping a copy: a golden snapshot
 * that exists twice is a snapshot that can disagree with itself. This resolves
 * from the test file's own location, so it survives the tree being promoted
 * from `ts/` to the package root at cutover.
 */
export function repositoryRoot(): string {
  let directory = path.dirname(new URL(import.meta.url).pathname);
  for (;;) {
    if (fs.existsSync(path.join(directory, "tests", "fixtures", "extractor_corpus"))) {
      return directory;
    }
    const parent = path.dirname(directory);
    if (parent === directory) throw new Error("could not locate the repository root");
    directory = parent;
  }
}

/**
 * A fresh temporary directory that removes itself when the suite ends.
 *
 * Returns a function rather than a value so each `beforeEach` gets its own, the
 * way pytest's `tmp_path` hands a new one to every test.
 */
export function temporaryDirectory(): string {
  const directory = fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "ci-mcp-test-"));
  return directory;
}

export function removeDirectory(directory: string): void {
  fs.rmSync(directory, { recursive: true, force: true });
}

/**
 * A differently-cased spelling of *value* that names the same directory, or null
 * when the filesystem distinguishes them.
 *
 * Case-insensitive path handling is a real behaviour on macOS and Windows and a
 * meaningless one on ext4, so the tests that depend on it skip rather than
 * assert something the platform cannot exhibit.
 */
export function caseInsensitiveAlias(value: string): string | null {
  const swapped = path.join(path.dirname(value), swapCase(path.basename(value)));
  if (swapped === value) return null;
  try {
    const original = fs.statSync(value, { bigint: true });
    const alias = fs.statSync(swapped, { bigint: true });
    return original.dev === alias.dev && original.ino === alias.ino ? swapped : null;
  } catch {
    return null;
  }
}

function swapCase(value: string): string {
  return Array.from(value, (character) => {
    const upper = character.toUpperCase();
    return character === upper ? character.toLowerCase() : upper;
  }).join("");
}
