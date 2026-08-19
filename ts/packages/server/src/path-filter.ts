/**
 * Translate search path globs into a LanceDB pushdown predicate.
 *
 * `search_code` filters paths with Python's `PurePosixPath.match`, which is the
 * authority on what matches -- and stays the authority after the port, because
 * the predicates this module emits are stored nowhere and compared against
 * nothing but the rows a scan returns. Applying the filter only in process
 * means it runs on rows the database has already truncated, so a match that
 * ranks below the fetch window disappears and is indistinguishable from "no
 * such code exists". These translations let the same semantics be pushed into
 * the scan, where they narrow the rows instead of discarding them.
 *
 * Everything here is pure so the equivalence with `PurePosixPath.match` is
 * testable without a database. The one rule that matters: the predicate may be
 * equivalent or broader, never narrower. Broader only costs rows the
 * post-filter then drops; narrower loses results, which is the bug this exists
 * to fix.
 *
 * The output is compared against the Python build's, character for character,
 * by the parity fixture in `test/fixtures/path-filter-parity.json`. That is why
 * `escapeRegex` below reproduces `re.escape`'s exact character set rather than
 * escaping whatever JavaScript happens to consider special: the emitted string
 * is consumed by LanceDB's Rust regex engine, which already accepts what the
 * shipping Python build sends it.
 */

/**
 * The characters `re.escape` prefixes with a backslash on Python 3.7 and later.
 *
 * Notably wider than JavaScript needs -- `&`, `~`, `#`, and whitespace are not
 * regex metacharacters anywhere -- and notably narrower in one place: `/` is
 * left alone, which matters because these expressions are never wrapped in
 * JavaScript regex literals.
 */
const RE_ESCAPE_CHARACTERS = new Set("()[]{}?*+-|^$\\.&~# \t\n\r\v\f");

function escapeRegex(character: string): string {
  return RE_ESCAPE_CHARACTERS.has(character) ? `\\${character}` : character;
}

/**
 * Apply the normalization `PurePosixPath` performs before matching.
 *
 * Redundant separators collapse, `.` components and trailing separators fall
 * away, and `..` is deliberately left alone -- a pure path never resolves it.
 * Translating the canonical spelling is what keeps the database predicate from
 * being narrower than the authoritative post-filter.
 */
function normalizePosixPattern(pattern: string): string {
  const parts = pattern.split("/").filter((part) => part !== "" && part !== ".");
  return parts.length === 0 ? "." : parts.join("/");
}

/**
 * Translate one conservative fnmatch character class.
 *
 * Returning `null` is always safe: the caller then skips pushdown and lets the
 * post-filter remain the authority. Ranges and regex set-operation characters
 * are deliberately left to that fallback because fnmatch's rules and LanceDB's
 * regex engine do not share identical class syntax.
 */
function characterClass(
  pattern: string,
  cursor: number,
): { expression: string; next: number } | null {
  let search = cursor + 1;
  if (pattern[search] === "!") search += 1;
  // A leading ] is a class member in fnmatch, not the closing delimiter.
  if (pattern[search] === "]") search += 1;
  const closing = pattern.indexOf("]", search);
  if (closing === -1) return null;

  const body = pattern.slice(cursor + 1, closing);
  const negated = body.startsWith("!");
  const members = negated ? body.slice(1) : body;
  if (members === "" || /[-\\/&~|]/.test(members)) return null;

  const escaped = Array.from(members, (character) =>
    "[]^".includes(character) ? `\\${character}` : character,
  ).join("");
  return { expression: `[${negated ? "^" : ""}${escaped}]`, next: closing + 1 };
}

/**
 * Return a regex equivalent to `PurePosixPath(path).match(pattern)`.
 *
 * Returns `null` when the pattern cannot be translated with confidence, which
 * tells the caller to skip the pushdown rather than risk a narrower predicate.
 *
 * Two properties of `PurePosixPath.match` drive the output:
 *
 * - Relative patterns match **from the right**, so `*.py` matches `a/b/c.py`.
 *   Hence the `(^|/)` prefix and `$` suffix rather than a leading `^`.
 * - `**` spans exactly one segment, the same as `*`
 *   (`PurePosixPath("src/deep/b.py").match("src/**")` is false), so runs of
 *   asterisks collapse to a single `[^/]*`. Recursive matching lives in a
 *   separate `full_match` and is not what this mirrors.
 */
export function globToRegex(pattern: string): string | null {
  if (pattern === "" || pattern.startsWith("/")) return null;
  const normalized = normalizePosixPattern(pattern);
  const parts: string[] = [];
  let cursor = 0;
  while (cursor < normalized.length) {
    const character = normalized[cursor];
    if (character === "*") {
      while (normalized[cursor] === "*") cursor += 1;
      parts.push("[^/]*");
    } else if (character === "?") {
      parts.push("[^/]");
      cursor += 1;
    } else if (character === "[") {
      const translated = characterClass(normalized, cursor);
      if (translated === null) return null;
      parts.push(translated.expression);
      cursor = translated.next;
    } else {
      // `character` is in range: the loop condition guarantees it, but
      // noUncheckedIndexedAccess cannot see that.
      parts.push(escapeRegex(character ?? ""));
      cursor += 1;
    }
  }
  return `(^|/)${parts.join("")}$`;
}

/**
 * Return a SQL predicate matching *column* against any of *patterns*.
 *
 * `null` means no pushdown: either there is nothing to filter, or at least one
 * pattern is untranslatable. Because the patterns are OR-ed, dropping one would
 * make the predicate narrower than the post-filter and lose rows, so a single
 * untranslatable pattern disables the whole pushdown.
 */
export function pathCondition(
  patterns: readonly string[],
  { column = "path" }: { column?: string } = {},
): string | null {
  if (patterns.length === 0) return null;
  const expressions: string[] = [];
  for (const pattern of patterns) {
    const translated = globToRegex(pattern);
    if (translated === null) return null;
    const quoted = `'${translated.replaceAll("'", "''")}'`;
    expressions.push(`regexp_like(${column}, ${quoted})`);
  }
  return `(${expressions.join(" OR ")})`;
}

/**
 * `PurePosixPath(path).match(pattern)` for the post-filter.
 *
 * Pushdown uses `globToRegex`; this is the authority that still runs on the
 * fetched rows. Absolute patterns never match a relative indexed path. An
 * empty `[]` class is fnmatch's "leading `]` is a member", so `*[]*.py`
 * matches `a[].py` rather than failing the search.
 */
export function pathMatches(filePath: string, pattern: string): boolean {
  if (pattern.startsWith("/")) return false;
  const translated = globToRegex(pattern) ?? globToRegex(pattern.replaceAll("[]", "[]]"));
  if (translated === null) return false;
  return new RegExp(translated).test(filePath);
}
