"""Record what `pathspec.GitIgnoreSpec` decides, so the TypeScript port can be held to it.

The scanner leans on gitignore semantics in three places -- the include
patterns, the config excludes, and the nested `.gitignore` stack -- and all
three go through `GitIgnoreSpec`. The port swaps that for the `ignore` npm
package, which is a *second* independent implementation of a specification
whose corners (anchoring, directory-only patterns, negation ordering, `**`
spans, character classes) are exactly where implementations differ. Asserting
the port against a hand-written expectation would only test the expectation.

So this writes down the shipping build's own verdicts: for every pattern set,
the `match_file` answer and the tri-state `check_file(...).include` answer
(ignored / re-included / no rule matched) for every corpus path. The TypeScript
suite asserts against those, and a divergence fails there rather than surfacing
as a file the two builds disagree about indexing.

Run it from the repository root after touching either side:

    uv run python ts/packages/server/scripts/write_ignore_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from pathspec import GitIgnoreSpec

# Each entry is one spec, exercising a distinct part of the syntax. The include
# lists are the scanner's own defaults; the rest are the shapes real
# `.gitignore` files use.
PATTERN_SETS: list[list[str]] = [
    ["**/*.py"],
    ["**/*.py", "**/*.ts"],
    ["*.py"],
    ["ignored.py"],
    ["package/*.py"],
    ["package/*.py", "!keep.py"],
    ["!keep.py"],
    ["build/"],
    ["/root_only.py"],
    ["docs/**"],
    ["**/generated/**"],
    ["*.log", "!important.log"],
    ["src/**/*.py"],
    ["a?c.py"],
    ["[abc]*.py"],
    ["node_modules"],
    ["dist/"],
    ["# a comment", "", "value.py"],
    ["\\#literal.py"],
    ["with space.py"],
    ["nested/deep/file.py"],
    ["nested/"],
    ["**"],
    ["*"],
    ["!*.py"],
    ["sub/*", "!sub/keep/", "!sub/keep/**"],
]

PATHS: list[str] = [
    "main.py",
    "MAIN.PY",
    "ignored.py",
    "keep.py",
    "value.py",
    "#literal.py",
    "with space.py",
    "root_only.py",
    "important.log",
    "server.log",
    "abc.py",
    "axc.py",
    "bzz.py",
    "component.tsx",
    "app.ts",
    "notes.md",
    "package/keep.py",
    "Package/Keep.py",
    "package/drop.py",
    "package/nested/keep.py",
    "src/app/main.py",
    "src/main.py",
    "src/main.ts",
    "build/output.py",
    "build/nested/output.py",
    "dist/bundle.js",
    "docs/index.md",
    "docs/deep/index.md",
    "generated/api.py",
    "src/generated/api.py",
    "node_modules/vendor.js",
    "nested/deep/file.py",
    "nested/other.py",
    "sub/keep/file.py",
    "sub/drop/file.py",
    "a/b/c/d/e.py",
]


def main() -> None:
    entries = []
    for patterns in PATTERN_SETS:
        spec = GitIgnoreSpec.from_lines(patterns)
        entries.append(
            {
                "patterns": patterns,
                # `match_file` is what the include and config-exclude checks use.
                "matches": [spec.match_file(path) for path in PATHS],
                # `check_file(...).include` is the tri-state the nested
                # `.gitignore` stack folds: True (ignored), False (re-included by
                # a negation), or null (no pattern matched, so an outer
                # directory's verdict stands).
                "include": [spec.check_file(path).include for path in PATHS],
            }
        )
    payload = {"paths": PATHS, "specs": entries}
    destination = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "ignore.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
