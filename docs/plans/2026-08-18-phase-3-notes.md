# Phase 3 notes — storage

Date: 2026-08-18
Status: complete
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 3 ports the three modules that persist an index: `storage`, `staging`,
and `history` — roughly 2,800 source lines. A first cut landed earlier on this
branch; this close fills the parity gaps the Python suite treats as the spec
and expands the tests to match.

The gate is green on all four checks (`biome format`, `biome lint`,
`tsc --noEmit`, `bun test`): 681 tests across 25 files, up from 633 across 22.

## What is in the tree

| Python | TypeScript | Tests |
|---|---|---|
| `storage.py` | `src/storage.ts` | `test/storage.test.ts` |
| `staging.py` | `src/staging.ts` | `test/staging.test.ts` |
| `history.py` | `src/history.ts` | `test/history.test.ts` |
| SQLite (`sqlite3`) | `src/runtime/sqlite.ts` (`bun:sqlite` behind the adapter) | driven by history tests |

New dependencies, both pinned exactly: `@lancedb/lancedb` 0.37.1 and
`apache-arrow` 18.1.0 (the version LanceDB JS peer-requires).

## Decisions this phase forced

### The commit API takes JS objects; the on-disk staging format stays Arrow IPC

Python's `replace_files_from_arrow` consumes `pa.Table`. The JS bindings accept
plain records natively, and a hand-written table mapping is where a
transliteration hides a type the checker cannot see. The TS method therefore
takes `ReplacementBatch` records. The journalled payloads under
`<data>/staging/<project-id>/<job-id>/` remain Arrow IPC with the same schemas,
so a half-migrated machine still degrades via `PROTOCOL_VERSION` rather than
corrupting state.

### The merge-semantics probe is still a write-path gate

A regression of `whenNotMatchedBySourceDelete` to all-or-nothing gate semantics
would silently delete untouched files on the second commit of a multi-file
project. The probe runs on the first batched commit, not on store construction
— read-only processes never pay for it — and a failed probe raises
`UNSUPPORTED_RUNTIME` rather than committing. S1 already proved the installed
JS SDK filters per row; the probe keeps that a runtime fact.

### Partition generation and an in-process reader lock replace `filelock`

Python's `partition_access` is a `FileLock` so a rebuild cannot invalidate an
active query, including across processes. Phase 3 is in-process only (the
daemon arrives in Phase 6), so the port uses a per-project promise chain for
the same exclusion and a `partition-generations/<id>` file so a deleted
partition cannot be served from a stale cache handle. Cross-process locking
lands with the daemon.

### Coverage queries live on `LanceStore`, not only on `ReferenceStore`

`coverageForFile` and `referenceCoverage` are storage methods the indexer
backfill will call in Phase 5. They are not on the Phase 2 `ReferenceStore`
interface because the resolver never needs them; leaving them off `LanceStore`
would have made the Phase 5 call site invent a second read path.

## What is deliberately not here

- **Indexer-coupled staging tests.** Crash-during-embed, failed `_commit_staged`,
  and Application startup-lock recovery need `indexing.ts` and `application.ts`.
  The journal/recovery behaviours those tests care about are covered against a
  `StagingStore` fake; the live commit path arrives in Phase 5.
- **Monkeypatched `optimize` assertions.** The TS bindings do not expose a
  clean hook for "was `deleteUnverified` passed". `maintainProject` always
  passes `deleteUnverified: false`, matching the Python contract.
- **Cross-process partition locks.** See above; Phase 6.

## Notes for Phase 4

`LanceStore.chunkArrowSchema` is the schema the embedder must write: float16
by default, float32 only when `vectorStorage` is set. A dtype flip is an
incompatibility reason, not a mixed-generation write. The probe cache and
passage backend should treat the store's `vectorDimension` / `vectorStorage` as
authoritative rather than re-deriving them.
