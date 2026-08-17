/** Helpers shared by the test suite -- the `conftest.py` of this tree. */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

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
