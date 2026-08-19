# Phase 7 notes — accelerators

Date: 2026-08-19
Status: implementation complete; real-hardware promotion pending
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 7 replaces the CPU-only accelerator-record stub and supplies the
real-inference proof path that the installer will use in Phase 8.

## What is in the tree

| TypeScript module | Responsibility | Tests |
|---|---|---|
| `src/accelerator-env.ts` | Schema-versioned, atomically written accelerator records, validation, record-path override, and diagnostic metadata | `test/accelerator-env.test.ts` |
| `src/accelerator-probe.ts` | Loads the configured provider, embeds the common probe corpus, validates vectors, and rejects sessions that resolved without the requested provider | `test/accelerator-probe.test.ts` |
| `src/embedding-worker.ts` | Exposes the common direct-ONNX model loader for the probe and worker paths | probe tests |
| `src/passage-backend.ts` | Treats an empty or CPU-only provider resolution as an accelerator failure and falls back through the existing strict-mode and cache paths | `test/passage-backend.test.ts` |

`Application.effectiveBackendSelection()` and `Application.modelStatus()` continue
to be the sole surface for accelerator selection and diagnostics. A validated
record contributes its provider and runtime/driver metadata; workers still run
the real probe before use, so a record cannot promote a silently CPU-resolved
session.

## Record compatibility

The record remains `accelerator.json` with schema version 1 and the Python
field names. `python_version` is retained for cross-runtime invalidation; new
Bun records store the Bun runtime version there. A Python-written record is
therefore rejected by the TypeScript runtime rather than being reinterpreted as
a valid same-runtime worker configuration.

## Promotion status

CUDA, CoreML, MIGraphX, and WebGPU provider wiring remains available through
the existing backend descriptors and the direct ONNX session factory. This
machine has no qualifying accelerator runner, so the promotion gates cannot be
claimed here. The remaining release evidence is a real-model probe and
calibration run on each candidate's hardware, including vector acceptance and
the performance threshold required before changing a backend's stability.

MLX remains unresolved under D2: no supported Node MLX binding is installed,
so a prepared MLX record will fail its real probe and safely fall back to CPU.
