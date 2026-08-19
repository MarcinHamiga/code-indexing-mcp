# Phase 5 notes — orchestration & search

Date: 2026-08-19
Status: complete
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 5 ports incremental indexing, hybrid search, and the application
service layer that MCP and CLI will sit on in Phase 6.

## What is in the tree

| Python | TypeScript | Tests |
|---|---|---|
| `indexing.py` | `src/indexing.ts` | `test/indexing.test.ts`, `test/indexing-backend.test.ts` |
| `search.py` | `src/search.ts` | `test/search.test.ts` |
| `application.py` | `src/application.ts` | `test/application.test.ts` |
| — | `src/accelerator-env.ts` (CPU-only stub) | — |

`Indexer.index` / `backfillReferences`, `SearchService.searchCode`, and
Application methods that touch the store are async. Locks use
`proper-lockfile`. The embedding worker is a distinct authenticated child;
parent growth uses `process.memoryUsage()` and worker RSS uses `pidusage` on
the child's PID.

## Decisions this phase forced

### Session methods stay Promise-based

Phase 4 made `PassageBackendSession.planAndEmbed` async. The indexer
awaits it, so the whole index/backfill/search path is async. Application
methods `await` a constructor-started recovery promise so staged-commit
rollback finishes before the first query.

### Packed float32 on the wire, unpacked floats in Arrow

`Indexer` unpacks little-endian float32 worker bytes into `number[]`
before staging. Arrow/Lance own the float16 column encoding, matching
the Phase 3 staging contract.

### `accelerator_env` is a CPU-only stub

Phase 7 owns the installer-written environment record. `loadEnvironment`
returns no record, so Application selects CPU and still wires
`_passageSessionFactory` for `OnnxEmbedder` when
`index_execution === "worker"`. CUDA/external-interpreter Application
tests stay in Phase 7.

### Vector-storage compatibility compares names, not constructors

Lance's JS schema does not always report `Float16` as
`constructor.name`. `incompatibilityReason` now classifies the stored
dtype from the type name so a just-written float16 partition is not
marked `rebuild_required` on the commit upsert.

### Real-model memory is a dedicated gate

`scripts/benchmark-index-memory.ts` indexes a deterministic near-cap corpus
through the real ONNX child with 256-token windows. The Ubuntu CI job caches
the model, enforces the worker budget plus the 256 MiB allowance, and uploads
the JSON report without slowing the ordinary three-OS test matrix.

## What is deliberately not here

- MCP server, daemon, CLI (Phase 6)
- Live CUDA/MLX/WebGPU/CoreML execution and `accelerator_env` file
  format (Phase 7)

## Notes for Phase 6

`Application` is the adapter target: init, index, status, search,
references, storage, and maintenance are all on it. Surfaces should call
these methods rather than constructing `Indexer` / `SearchService`
directly.
