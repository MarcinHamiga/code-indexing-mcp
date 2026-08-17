# Phase 1 notes — foundations

Date: 2026-08-17
Status: complete
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 1 ports the nine modules the rest of the tree stands on: `errors`,
`models`, `settings`, `path_filter`, `token_batching`, `acceptance`, `progress`,
`projects`, and `update_check` — roughly 2,500 source lines — plus the slice of
`backends` that `settings` validates against, and the `pathlib` semantics the
port leans on.

The gate is green on all four checks (`biome format`, `biome lint`,
`tsc --noEmit`, `bun test`): 282 tests across 12 files.

## What is in the tree

| Python | TypeScript | Tests |
|---|---|---|
| `errors.py` | `src/errors.ts` | `test/errors.test.ts` |
| `models.py` | `src/models.ts` | `test/models.test.ts` |
| `settings.py` | `src/settings.ts` | `test/settings.test.ts` |
| `path_filter.py` | `src/path-filter.ts` | `test/path-filter.test.ts` |
| `token_batching.py` | `src/token-batching.ts` | `test/token-batching.test.ts` |
| `acceptance.py` | `src/acceptance.ts` | `test/acceptance.test.ts` |
| `progress.py` | `src/progress.ts` | `test/progress.test.ts` |
| `projects.py` | `src/projects.ts` | `test/projects.test.ts` |
| `update_check.py` | `src/update-check.ts` | `test/update-check.test.ts` |
| `backends.py` (accelerator name only) | `src/backends.ts` | `test/backends.test.ts` |
| — (`pathlib` semantics) | `src/paths.ts` | `test/paths.test.ts` |

New dependencies, both pinned exactly: `zod` 4.4.3 and `smol-toml` 1.8.0.

## Decisions this phase forced

### Model field names stay snake_case

The models are the wire contract — the MCP tool schemas derive from them, the
project marker and the progress snapshot are written from them, and the daemon
frames them. Renaming fifty models to camelCase would have meant a hand-written
serde mapping per model, and a mapping is exactly where a transliteration hides
a bug that no type checker catches. `src/models.ts` is therefore the one file in
the package where fields are snake_case; everything else is camelCase, including
`IndexSettings`, which is internal and never serialized.

### `mtime_ns` is a bigint, and nothing else is

Nanosecond mtimes run around 1.7e18, two hundred times past
`Number.MAX_SAFE_INTEGER`. A `number` would round the low digits away, and since
that value is what change detection compares against the stored one, rounding it
would make **every file in a migrated index look modified exactly once** — a
silent full reindex on first use, which is precisely what S3's verdict of "keep
indexes" was meant to avoid. `fs.stat({ bigint: true })` hands it over as a
bigint already.

Two consequences carry forward. `JSON.stringify` refuses bigints outright, so
the MCP surface owes `ScanInspectionItem` an explicit serializer in Phase 6.
And `reference_extraction_ns` deliberately stayed a `number`: it is a duration,
not a timestamp, so it never leaves the safe-integer range.

### The path pushdown is held to a generated fixture, not a reimplementation

`glob_to_regex` exists to agree with `PurePosixPath.match` exactly, and the
translation is subtle in three independent ways: matching is right-anchored,
`**` spans a single segment, and the escaping follows `re.escape`'s idiosyncratic
character set (which escapes `&`, `~`, `#`, and spaces, and leaves `/` alone).
Reimplementing that oracle in TypeScript would only have tested the
reimplementation.

So `scripts/write_path_filter_parity.py` records what the shipping Python build
actually does — every emitted regex, character for character, and the
ground-truth match for 37 patterns against 932 corpus paths — and the TypeScript
suite asserts against it. All 37 regexes are byte-identical and all 34,484
match decisions agree. This is §8's "golden fixtures" idea applied to the first
module that needed it; Phase 2 inherits the pattern for the extractor corpus.

One detail worth knowing: the emitted expressions are compiled **without** the
`u` flag, because `re.escape` produces identity escapes such as `\ ` and `\~`
that Unicode mode rejects and that LanceDB's Rust engine already accepts from
the shipping build.

### Two dependency-table refinements

- **`zod-to-json-schema` is not needed.** §4 hedged that the SDK might need raw
  schemas; zod v4 ships `z.toJSONSchema()` in the box, and a test asserts that a
  path field emits a plain `{"type": "string"}` — the warning that pydantic's
  `_PathAsPlainString` wrapper existed to suppress.
- **`psutil` splits by caller.** §4 pairs it with `pidusage`, which is right for
  the RSS polling in Phase 4. `settings.py`'s only use is
  `virtual_memory().total`, which is `node:os`'s `totalmem()` and needs no
  dependency at all. Thread defaults use `availableParallelism()` rather than the
  raw CPU count, so a container's cgroup quota is honoured.

### Background work replaces the daemon thread

`update_check.start_background_refresh` needed a Python thread because
`subprocess.run` blocks. The remote call is asynchronous here, so the function
returns an unawaited promise instead. The seams the tests drive — environment,
install root, git runner, clock — became explicit parameters rather than module
attributes to monkeypatch, which is why `notice()` and `startBackgroundRefresh()`
take an options argument the production callers omit.

`install_context` also changed what it asks. Python tested `sys.prefix`, the
virtualenv root, because the interpreter is what a Python install owns. A
TypeScript install owns the *code*, so the check is whether the running module's
own directory sits inside the install directory.

### Immutability is a `tsc` property, not a runtime one

Pydantic's `frozen=True` has no zod analogue that survives composition —
`.extend()` on a frozen schema is not expressible, and `RefactorFinding` extends
`ReferenceHit` while `StoredChunk` extends `IndexedChunk`. Nothing mutates a
parsed model instead, and `ProgressPublisher` — the one place that used to need a
mutable model — replaces its state object rather than mutating it, which makes
every snapshot handed to a listener a true point-in-time picture for free.

## What is deliberately not here

- **The indexer-driven progress tests.** Most of `test_progress.py` drives a
  real `Indexer`: counters rising monotonically through a run, skip reasons
  aggregating by cause, a run id on every update. Those return with the indexer
  in Phase 5. Everything the publisher and reader own by themselves is asserted
  now.
- **The rest of `backends.py`.** Descriptors, provider probing, and nomination
  depend on what ONNX Runtime reports, and nothing in Phase 1 loads it. Phase 4
  grows the module around the accelerator name.
- **A `zod` schema for the MCP tool inputs.** Phase 6 generates those from these
  models; Phase 1 only proves the generation produces the right primitive types.

## Note for whoever runs the Phase 0 spikes again

`packages/spikes/src/acceptance.ts` still holds its own copy of the metrics. The
canonical port is `packages/server/src/acceptance.ts` now, and it has
`topKRankCorrelation` and a suite; the spike copy stays because a Phase 0 spike
is a record of an experiment, and one reaching into code written after it ran
would stop reproducing the verdict in
[the spike results](2026-08-17-phase-0-spike-results.md). Fix bugs in the server
module.
