# Phase 2 notes — scan & extract

Date: 2026-08-18
Status: complete
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 2 ports the three modules that turn a directory into rows: `scanner`,
`extractor` (with both query packs), and `reference_service` — roughly 3,900
source lines, against roughly 4,900 lines of Python tests.

The gate is green on all four checks (`biome format`, `biome lint`,
`tsc --noEmit`, `bun test`): 600 tests across 20 files, up from 282 across 12.

## What is in the tree

| Python | TypeScript | Tests |
|---|---|---|
| `scanner.py` | `src/scanner.ts` | `test/scanner.test.ts`, `test/ignore.test.ts` |
| `extractor.py` | `src/extractor.ts` | `test/extractor.test.ts`, `test/extractor-equivalence.test.ts`, `test/reference-extraction.test.ts`, `test/unicode-extraction.test.ts` |
| `queries/*.scm`, `reference_queries/*.scm` | `src/queries/`, `src/reference-queries/` | byte-identity assertion in `test/extractor.test.ts` |
| `extractor.py::_languages` | `src/grammars.ts` | `test/extractor.test.ts` |
| `reference_service.py` | `src/reference-service.ts` | `test/reference-service.test.ts`, `test/refactors.test.ts`, `test/resolver-corpus.test.ts` |
| `storage.py::ReferenceRecord` (the read slice) | `src/reference-store.ts` | driven by `test/reference-fixtures.ts` |
| `indexing.py::_reference_rows` | `src/reference-records.ts` | driven by `test/reference-fixtures.ts` |
| — (tree-sitter offset semantics) | `src/source-text.ts` | `test/extractor.test.ts` |

New dependencies: `ignore` 7.0.5 plus `tree-sitter` 0.25.1 and the eighteen
grammar packages S2 settled on, all pinned exactly and all already resolved in
`bun.lock` from Phase 0.

## The one thing that would have broken everything

**Tree-sitter's Node binding reports UTF-16 code-unit indices where the Python
binding reports UTF-8 byte offsets.** For `x = "éé𝄞"`, the `def` that follows
starts at byte 15 and at code unit 11.

Those offsets are not incidental. They are the chunk id's digest input, the
`start_byte`/`end_byte` stored in every chunk and reference row, and — through
`analyze_refactor` — the literal instruction a caller uses to splice a rename
into a file. Feeding a tree-sitter index straight into a byte field would have
corrupted every offset in any file containing one non-ASCII character, silently,
and only for those files. A pure-ASCII test suite would never have noticed.

`src/source-text.ts` holds the conversion: one pass over the decoded text builds
the breakpoints where the byte/code-unit delta changes, and every node offset in
the extractor passes through `SourceText.byte()`. The common case — an ASCII
source — detects in one length comparison and maps by identity, so the cost is a
comparison per file rather than per node.

The same file carries a third coordinate system. Python's `str` indexes by *code
point*, so `len(content)` and `content[a:b]` in the chunk splitter count code
points where JavaScript counts UTF-16 units. They agree everywhere except astral
characters, and the splitter's `max_chars` decisions determine byte ranges, so
`codePointLength`/`sliceCodePoints` carry the same fast-path shape.

Two smaller semantics were reproduced rather than approximated for the same
reason: `str.rstrip()` strips `\x1c`–`\x1f` and `\x85` that JavaScript's `\s`
does not, and `str.splitlines()` breaks on a form feed — a real page separator in
C and Python sources. Both decide where a chunk's content ends, and therefore its
byte range.

**The shared corpus cannot test any of this**, because it is entirely ASCII —
the one input where the two coordinate systems coincide. So
`scripts/write_unicode_extraction_parity.py` records what the Python build
extracts from sources covering all three regimes that differ (two-byte Latin-1,
three-byte CJK, four-byte astral), plus a BOM and an oversized line built from
multi-byte characters. `test/unicode-extraction.test.ts` asserts the recorded
offsets *and* — the stronger statement — that each stored range, applied to the
file's bytes, slices the chunk's own content back out. A uniformly shifted offset
can satisfy an equality against a fixture; it cannot satisfy that.

## The extractor is held to Python's own output, not to a re-derivation

`tests/fixtures/extractor_snapshot.json` already records what the shipping
Python build emits for all 18 languages in the extractor corpus: every chunk's
kind, symbol, byte range, line range, part index and content digests, plus every
structural reference and declaration shape. The Python suite gates its own
refactors on that file. `test/extractor-equivalence.test.ts` is held to the
*same* file — not a copy, the same path — so a divergence between the two builds
fails there rather than surfacing later as a search result the Python build would
not have returned.

This is §8's "golden fixtures" at its strongest: the oracle is the shipping
build's own output, and it covers exactly the fields chunk identity is digested
from. It passed on the first run, for all 18 languages, which is the strongest
evidence available that the transliteration is faithful.

The snapshot is regenerated only from Python
(`python -m tests.test_extractor_equivalence`), never from the TypeScript side.

## Gitignore semantics get the same treatment

`pathspec.GitIgnoreSpec` becomes the `ignore` npm package — a *second*
independent implementation of a specification whose corners (anchoring,
directory-only patterns, negation ordering, `**` spans, character classes) are
exactly where implementations drift. A drift there changes which files get
indexed and raises nothing.

`scripts/write_ignore_parity.py` records the shipping build's verdicts for 26
pattern sets against 34 corpus paths: both `match_file` and the tri-state
`check_file(...).include` the nested `.gitignore` stack folds. All 884 decisions
agree. The tri-state matters on its own: collapsing "no pattern matched" to "not
ignored" would let a nested `.gitignore` that says nothing about a file silently
undo its parent's rule.

## Decisions this phase forced

### The scanner is async; the extractor is not

`scanner.py` is synchronous and the MCP server offloads it to a thread. Here it
is an `AsyncGenerator`. A stdio MCP server and a JSON-RPC daemon share one event
loop with the scan, and a synchronous walk of a large repository would stall
every in-flight request until it finished — the problem the Python build solved
with threads, which this runtime does not have to reach for.

It also made the `git ls-files` deadline simpler rather than harder.
`_iter_git_batches` needs a daemon reader thread and a bounded staging queue
because `select` cannot wait on a pipe on Windows; streaming stdout with an abort
timer is the same guarantee in a tenth of the code. Batching, ordering, and the
yielded items are unchanged.

The extractor stays synchronous because tree-sitter is. The one place inside it
that waits on a network round trip — the language pack's first-use grammar
download — blocks on `Atomics.wait`, which is the only portable synchronous sleep
on either runtime.

### `reference_service` takes a port, not a store

`reference_service.py` takes a concrete `LanceStore`. The migration order lands
the resolver in Phase 2 and LanceDB in Phase 3, so the TypeScript version takes a
`ReferenceStore` interface (`src/reference-store.ts`) that Phase 3 implements.

That was forced, but it turned out to be worth doing anyway. Every method on the
interface is a *narrowed, pushed-down* query rather than a table scan, and
several of them exist because the obvious version was once a full-table
materialization per page. Writing the interface down makes that a contract the
reader can check instead of a property of one implementation that a future change
could quietly lose. The suite asserts the pushdowns actually happen — that the
declaration fetch is a proper subset of the known files, that the reference table
is fetched exactly once per analysis, that the old signature shape is fetched
once rather than once per call site.

`src/reference-records.ts` (the port of `indexing.py::_reference_rows`) landed
here rather than in Phase 5 for a related reason: it produces the exact rows the
resolver reads, and defining the writer beside the schema is what makes the
reader's assumptions checkable. Phase 5's indexer calls it unchanged.

### The query packs are copied, and the copy is asserted

`src/queries/` and `src/reference-queries/` are byte-identical copies of the
Python tree's packs. During the dual-maintenance window a copy that can drift is
a copy that will, so `test/extractor.test.ts` asserts byte-identity against the
Python originals; once that tree retires, the assertion simply stops having a
source to compare against and skips.

The alternative — reading the Python tree's files at runtime — would have made
the package non-self-contained and broken at cutover, when the TypeScript tree is
promoted to the package root.

### `gdshader` on Windows is a supported state, not an error

§5.5's accepted capability gap needed somewhere to live.
`grammars.ts::unavailableLanguages()` is the single table, and
`scanner.ts::languageForExtension()` reads it: on Windows, `.gdshader` and
`.gdshaderinc` classify as unsupported extensions exactly as `.md` does. A Godot
repository indexes its scripts and scenes and quietly omits its shaders, rather
than failing the run — which is the whole point of the decision.

The extractor's equivalence and behaviour suites check the same table rather than
attempting a load, so what decides whether those tests run is the recorded
decision, not a network round trip.

### Faithfulness beat tidiness in three places

- `_parameter_shapes` sets `positional_only = False` on the separator branch and
  never to `True`, which makes its `elif positional_only` arm unreachable; the
  `positional_only` kind is produced solely by the retroactive rewrite of
  already-collected rows. The port keeps the same shape, with a note, so the two
  stay diffable.
- `_module_candidates` interpolates a path into a string (`f"{stem}.py"`), and
  `str(PurePosixPath())` is `"."` — so a bare `from . import x` really does
  produce a candidate named `.py`. `displayPath` reproduces it. A candidate set
  that differs from Python's is a resolution that differs from Python's.
- `_edit_span` searches bytes, not text. The port decodes the span as `latin1`
  so a string index and a byte index stay the same number, and compiles the
  pattern without the `u` flag so `\w` means the ASCII set Python's bytes
  patterns mean.

## What is deliberately not here

- **Two `test_references.py` cases that need real storage.** One renames a
  `.lance` directory on disk to simulate a never-built reference table; the other
  heals a stale file by re-running the indexer. The *behaviours* are covered
  (`referenceTableExists` and the stale-file suppression both have tests) — what
  is missing is the on-disk mechanism, which arrives with Phase 3 and Phase 5.
- **A real indexed store behind the resolver suites.** `test/reference-fixtures.ts`
  writes the fixtures to disk, runs the *real* extractor over them, and assembles
  rows with the *real* `referenceRows`; only persistence is replaced. When Phase 5
  lands, these suites should be re-pointed at a real indexed store — the fixtures
  and assertions carry over unchanged, and anything that only holds against the
  in-memory store will show up then.
- **`ScanConfig.max_file_bytes` enforcement above 2 GiB.** `fs.stat` reports size
  as a `bigint` and the port narrows it to a `number` for the model field, as
  Python does with `int`. Files that large are rejected long before the cast
  matters.

## Notes for Phase 3

`src/reference-store.ts` is the interface `LanceStore` must satisfy, and
`REFERENCE_SCHEMA_VERSION` lives there because it is a property of the stored
rows that both the writer and every reader need. `ReferenceRecord`'s field names
are the storage column names, snake_case, for the same reason `models.ts` keeps
snake_case: it is a wire contract, and a hand-written mapping is where a
transliteration hides a bug no type checker catches.
