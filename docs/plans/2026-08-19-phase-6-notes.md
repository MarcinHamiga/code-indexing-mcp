# Phase 6 notes — surfaces

Date: 2026-08-19
Status: complete
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7

Phase 6 ports the MCP server, per-user daemon, CLI, and benchmark
commands onto the Phase 5 `Application` adapter.

## What is in the tree

| Python | TypeScript | Tests |
|---|---|---|
| `server.py` | `src/server.ts` | `test/server.test.ts` |
| `daemon.py` | `src/daemon.ts` | `test/daemon.test.ts` |
| `cli.py` | `src/cli.ts` | `test/cli.test.ts` |
| `benchmark.py` | `src/benchmark.ts` | `test/benchmark.test.ts` |
| — | `src/jsonable.ts` | covered by daemon/CLI tests |

`Application` remains the only adapter target. Surfaces call its
methods; they do not construct `Indexer` / `SearchService`.

## Decisions this phase forced

### Official MCP SDK, not FastMCP

Tools are registered on `McpServer` with zod input schemas derived from
`models.ts`. Descriptions, titles, and annotations are copied from the
Python server. `z.toJSONSchema()` plus stripping defaulted keys from
`required` keeps the 16-tool contract (field names, optionality, bounds)
aligned with the Python schemas.

In-process tests drive `CreatedServer.callTool` rather than FastMCP's
session helper. Stdio serving uses `StdioServerTransport`.

### Application methods are already async

Python wrapped every Application call in `asyncio.to_thread`. The
TypeScript Application is Promise-based, so the MCP and daemon adapters
await it directly.

### Daemon framing buffers immediately

Unix-socket JSON frames (`!I` length prefix, `PROTOCOL_VERSION = 2`,
token compare) match the Python wire. A reader is attached as soon as a
connection is accepted so a request that arrives before `receive()` is
called is not lost. Windows still has no daemon: `serve` falls back to
`--direct`, matching today's `daemon_supported()`.

### Watcher is `@parcel/watcher`

`StartupCoordinator` keeps the dirty-root generation, one-slot limiter,
and backoff from `server.py`. The watch backend is `@parcel/watcher`
behind `watchRoot.current` so tests can replace it.

### Installer commands stay stubs

`configure`, `update`, and `uninstall` parse and exit 2 with
`UNSUPPORTED_RUNTIME` until Phase 8.

## What is deliberately not here

- CUDA/MLX/WebGPU/CoreML execution and the installer-written
  `accelerator_env` record (Phase 7)
- Installer TUI, harness merge, self-update (Phase 8)
- The two `benchmark_index_memory` environment-pinning tests wait for the
  Phase 9 memory-gate harness to grow its `--memory-mb` plumbing

## The parity-completion pass

A completeness review against `tests/test_server.py` (53 tests),
`test_daemon.py` (24), `test_cli.py` (24), and `test_benchmark.py` (17) found
the surfaces ported but roughly two thirds of their spec tests missing. The
pass ported them and, in doing so, surfaced and fixed real defects:

- **Tool-schema parity is now held to the Python build's own output**, via
  `scripts/write_tool_schema_parity.py` (the §8 MCP contract fixture),
  consumed by `test/tool-schema-parity.test.ts`. Writing it caught three
  divergences: nested `required` lists kept defaulted keys (pydantic omits
  them at every level), zod advertised ±MAX_SAFE_INTEGER bounds on unbounded
  integers, and `inspect_scan`'s `outcome`/`reason` filters had been
  tightened into enums the Python schema deliberately leaves open.
  `jsonSchema()` now normalizes all three at the source.
- **A race that failed ~1 run in 3**: `LanceStore`'s constructor starts the
  registry open as a floating promise; when a store is dropped before the
  native commit lands (fast test teardown, a CLI one-shot), it rejects
  unhandled, and Bun attributes the rejection to whichever test is in
  flight — abandoning that test's pending awaits mid-operation. Python opens
  storage synchronously, so nothing floats; `#openRegistry`, the
  Application's `#ready`, and the partition lock chain now mark their
  floating rejections handled (awaiters still observe them).
- **Close semantics**: `StartupCoordinator.close()` now cancels lock-waiters
  between attempts (Python's task-group teardown does), `schedule()` refuses
  once closed, and a close mid-index no longer drops the in-flight write.
- **Progress notifications**: the Python surface reports MCP progress through
  `ctx.report_progress`; the port had dropped the wiring. Tool callbacks now
  send `notifications/progress` when the client supplies a progressToken,
  and the in-process harness accepts an `onProgress` hook.
- **The daemon wire broke on `inspect_scan`**: frames stringify `mtime_ns`
  (≈1.7e18, unsafe as a JSON number) to keep precision, and the broker's
  `z.bigint()` parse rejected them; `ScanInspectionItem.mtime_ns` now accepts
  the wire form and coerces.
- **`serve --broker=on` on a socket-less platform** exited uncleanly;
  commander needed `parseAsync` for the action's error to reach the exit-2
  path.
- **The CLI update notice ignored `CODE_INDEXING_UPDATE_CHECK`** (a dead
  branch fell through to the cache check).

Surfaces gained the seams their Python tests get from monkeypatching:
`applicationFactory`/`serverFactory`/`benchmarkCommands` in `cli.ts`,
`runtimeRootHolder` in `update-check.ts` (standing in for `sys.prefix`),
patchable timing holders beside `watchRoot`, exported `ProgressPrinter`, and
exported `retireStaleDaemon`.

Test counts after the pass: server 51, daemon 26, CLI 21, benchmark 15 —
the remaining deltas are the two memory-harness tests above and tests whose
subject does not exist yet (installer delegation, Phase 8).

## Notes for Phase 7

`Application.modelStatus` and `effectiveBackendSelection` already expose
the CPU-only stub. Accelerator provider selection should keep answering
those methods rather than adding a parallel CLI path.
