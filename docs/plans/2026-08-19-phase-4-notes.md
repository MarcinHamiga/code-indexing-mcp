# Phase 4 notes — embedding (CPU)

Date: 2026-08-19
Status: complete
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 4 ports the CPU embedding stack: backend nomination, the direct-ONNX
embedder (the only path, per §5.1), disposable workers with a memory ceiling,
authenticated dial-back launch, calibration, the probe cache, and the
passage-backend session that degrades from an accelerator to CPU in place.

The gate is green on all four checks (`biome format`, `biome lint`,
`tsc --noEmit`, `bun test`): 784 tests across 32 files, up from 681 across 25.

## What is in the tree

| Python | TypeScript | Tests |
|---|---|---|
| `backends.py` | `src/backends.ts` | `test/backends.test.ts` |
| `embedding.py` | `src/embedding.ts` | `test/embedding.test.ts` |
| `direct_onnx.py` | `src/direct-onnx.ts` | `test/direct-onnx.test.ts` |
| `calibration.py` | `src/calibration.ts` | `test/calibration.test.ts` |
| `probe_cache.py` | `src/probe-cache.ts` | `test/probe-cache.test.ts` |
| `worker_launcher.py` | `src/worker-launcher.ts`, `src/worker-channel.ts` | `test/worker-launcher.test.ts` |
| `embedding_worker.py` | `src/embedding-worker.ts` | `test/embedding-worker.test.ts` |
| `passage_backend.py` | `src/passage-backend.ts` | `test/passage-backend.test.ts` |

New dependencies: `onnxruntime-node` 1.27.0, `@huggingface/hub` 2.15.0,
`@huggingface/tokenizers` 0.1.3, `proper-lockfile` 4.1.2.

## Decisions this phase forced

### FastEmbed is gone; `OnnxEmbedder` is the only in-process model

§5.1 said the direct ONNX path becomes the only path. Query and passage
embedding share it so they stay in one vector space. S3 already proved
`onnxruntime-node` + `@huggingface/tokenizers` match Python FastEmbed vectors
to cosine 1.0. The worker load path uses `DirectOnnxEmbedding` for CPU too —
there is no second stack to avoid.

### All workers dial back over an authenticated socket

Python kept `multiprocessing` spawn for CPU and a separate external launcher
for a second interpreter. §5.3 collapses the two environments, so both
production launchers are `ChildProcessLauncher`: `node:child_process.spawn`,
Unix socket (loopback TCP on Windows), HMAC-SHA256 challenge-response, config
on the channel after the handshake. Tests inject a `FunctionLauncher` that
runs the worker body against a pair of in-process queues — TypeScript cannot
pickle a function into a child the way `multiprocessing` did.

The HMAC wire is not Python `multiprocessing.connection`'s MD5 exchange.
Workers only talk to a TS parent, so the protocol is ours; the behaviour the
Python suite specified (right key accepted, wrong key dropped, quiet peer
timed out, stranger cannot hold the launch) is what the tests hold.

### Session methods are async

Python's `poll`/`recv` block a thread. The TS session is Promise-based
(`initialize`, `probe`, `planAndEmbed`, `close`) so the parent can sample RSS
between 100 ms polls without a second thread. Passage-backend follows.

### The probe cache JSON schema is unchanged

`CACHE_SCHEMA_VERSION = 2`, snake_case record fields, 32-record bound, corrupt
or foreign-version files read as empty. A half-migrated machine can still
reuse a Python-written `backend-probes.json`. The lock is `proper-lockfile`
instead of `filelock`; a lock failure still proceeds rather than failing the
run.

### `LanceStore.chunkArrowSchema` is still the write contract

Vectors on the worker wire are little-endian float32 bytes, as in Python.
Float16 storage conversion stays on the Phase 5 indexer, which already reads
`vectorStorage` from the store.

## What is deliberately not here

- **MLX / CoreML / live WebGPU / CUDA execution.** The registry names them and
  `auto` will pick CUDA or MLX when their provider is listed, but this phase
  does not load those runtimes. D2 remains open. Phase 7.
- **`accelerator_env` / `accelerator_probe`.** Phase 7.
- **Application wiring.** `Application._passage_session_factory` arrives with
  the indexer in Phase 5.
- **Real-model integration and the memory gate.** `test_model_integration.py`
  and `test_memory_acceptance.py` stay until the CI model cache is wired.
- **Bun.spawn.** `runtime/index.ts` offered it; `node:child_process` is the
  Node-compat path and S0 already confirmed it. No Bun-only spawn adapter.

## Notes for Phase 5

`OnnxEmbedder` and `PassageBackendSession` are the two embedder roles the
indexer needs. `PassageBackendSession.planAndEmbed` is async. The store's
`vectorDimension` / `vectorStorage` remain authoritative for the Arrow schema
the embedder's packed float32 rows are written into.
