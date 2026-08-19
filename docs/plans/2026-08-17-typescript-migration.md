# Python → TypeScript migration plan

Date: 2026-08-17
Status: in progress. Phase 0 executed — see
[2026-08-17-phase-0-spike-results.md](2026-08-17-phase-0-spike-results.md).
Phase 1 complete — see
[2026-08-17-phase-1-notes.md](2026-08-17-phase-1-notes.md).
Phase 2 complete — see
[2026-08-18-phase-2-notes.md](2026-08-18-phase-2-notes.md).
Phase 3 complete — see
[2026-08-18-phase-3-notes.md](2026-08-18-phase-3-notes.md).
Phase 4 complete — see
[2026-08-19-phase-4-notes.md](2026-08-19-phase-4-notes.md).
Phase 5 complete — see
[2026-08-19-phase-5-notes.md](2026-08-19-phase-5-notes.md).
Branch: `ts-migration`

## 1. Purpose and scope

This document plans a full migration of `code-indexing-mcp` from Python
(~26,000 source lines across 56 modules, ~29,200 test lines across 74 files)
to TypeScript running on Bun. The migration must preserve, verbatim, the
externally observable contract:

- the 16-tool MCP surface and its JSON schemas,
- the CLI command set (`init`, `index`, `status`, `projects`, `model`,
  `storage`, `daemon`, `benchmark`, serve),
- the on-disk data layout (LanceDB tables, project markers, progress
  snapshots, history database, probe cache, accelerator records),
- the search semantics (hybrid vector + FTS ranking, path-filter pushdown,
  structural reference lookup), and
- the memory-safety guarantees the embedding pipeline was built around.

Non-goals: feature work during the port, changes to the tool schemas, or a
redesign of the accelerator architecture. The port is a *transliteration
with idiomatic surface*, not a rewrite; every deliberate behavior documented
in module docstrings and `docs/plans/` carries over.

## 2. Current-state inventory

### 2.1 Process and runtime model

| Component | Today (Python) | Notes |
|---|---|---|
| MCP server | FastMCP over stdio; `AutoIndexingMCP` subclass with lifespan-driven startup coordinator (`server.py`, 1,479 ln) | asyncio; background auto-indexing with `watchfiles` monitors, slot-limited concurrency, backoff |
| Per-user daemon | JSON-RPC over Unix socket (loopback TCP on Windows), length-prefixed frames, HMAC challenge-response auth, `filelock` singleton, `PROTOCOL_VERSION` guard (`daemon.py`, 756 ln) | MCP server and CLI delegate application calls to it |
| CPU embedding worker | `multiprocessing` spawn child with memory ceiling and batch retries (`embedding_worker.py`, 589 ln) | disposable per run |
| Accelerator worker | External subprocess started from a *separate* Python environment; dials back over an authenticated socket (`worker_launcher.py`, 460 ln) | exists because `fastembed`/`fastembed-gpu` cannot share one env — see §5.3 for what this becomes in TS |
| CLI | argparse; delegates to daemon or runs in-process (`cli.py`, 523 ln) | update-notice throttling on human-facing commands |
| Installer | Separate package (~4,600 ln incl. Textual TUI); curl-pipeable stdlib bootstrap `install.py` → `uv sync` → installer | harness config merging, accelerator env preparation, self-update, uninstall |

### 2.2 Core pipeline

| Module | Lines | Responsibility |
|---|---|---|
| `storage.py` | 1,745 | Partitioned LanceDB persistence: `projects` table plus per-project `files`/`chunks`/`references` tables, PyArrow schemas, FTS + BTree index management |
| `reference_service.py` | 1,685 | Conservative syntax-only reference classification over structural rows |
| `indexing.py` | 1,610 | Incremental indexing orchestration (content hashing, delta computation) |
| `extractor.py` | 1,583 | Tree-sitter symbol/module chunk extraction; 18 `.scm` chunk queries + 4 reference queries, 18 languages |
| `application.py` | 1,455 | Application services shared by MCP and CLI adapters |
| `models.py` | 1,084 | ~50 frozen pydantic v2 domain models (the MCP tool schemas derive from these) |
| `staging.py` | 655 | Journalled Arrow IPC staging for crash-recoverable index commits |
| `scanner.py` | 631 | `git ls-files`-backed enumeration inside worktrees, ignore-rule walk elsewhere |
| `passage_backend.py` | 604 | Embedding session that fails over accelerator → CPU mid-run, invisibly to the indexer |
| `mlx_backend.py` | 540 | JinaBERT v2 re-implemented in MLX for Apple Metal (reads float32 initializers out of the ONNX artifact) |
| `history.py` | 397 | SQLite WAL audit database of indexing runs, bounded and pruned |
| `embedding.py` | 389 | FastEmbed adapter; model `jinaai/jina-embeddings-v2-base-code` |
| `direct_onnx.py` | 261 | Direct ONNX Runtime passage embedding for providers FastEmbed cannot configure (WebGPU, MIGraphX) |
| `token_batching.py` | 204 | Tokenizer-bounded window planning (memory is quadratic in sequence length; character windows are not enough) |
| `search.py` + `path_filter.py` | 320 | Hybrid retrieval; glob → LanceDB pushdown predicate translation |
| calibration / acceptance / probe_cache / accelerator_* | ~980 | Backend nomination, real-inference probing, batch calibration, promotion gates, keyed probe cache |

### 2.3 Tooling and CI

- `ruff` (format + lint), `mypy --strict`, `pytest -n auto`, `uv` lockfile
  with five mutually-exclusive accelerator extras (`cpu`, `cuda`, `webgpu`,
  `mlx`, `migraphx`).
- CI: Ubuntu + Windows × Python 3.12/3.13 test matrix, a macOS job, and a
  real-model memory/benchmark gate with model and grammar caches.

## 3. Target stack

- **Runtime**: Bun ≥ 1.2, pinned in `package.json#engines` and CI. Bun
  executes TypeScript directly (no build step for dev or serve), starts
  fast enough to matter for a stdio MCP server and a CLI, and ships
  `bun:sqlite`, `Bun.spawn`, and a package manager in the box. Single
  workspace with `server` and `installer` packages mirroring today's
  package split.
- **Node-compat discipline**: core modules import only `node:`-namespaced
  APIs and Web standards; Bun-only APIs (`bun:sqlite`, `Bun.spawn`,
  `$` shell) live behind thin adapters. This costs little and keeps
  per-process fallback to Node possible if a native module hits a Bun
  N-API gap (risk register, and spike S0).
- **Language/config**: TypeScript, `"strict": true` plus
  `noUncheckedIndexedAccess` — the parity target for `mypy --strict`.
  Bun does not type-check, so `tsc --noEmit` stays a CI gate.
- **Package manager**: `bun install` with the committed `bun.lock` (the
  `uv.lock` equivalent). Accelerator variants become optional
  dependencies / install-time choices rather than conflicting extras —
  see §5.3.
- **Test runner**: `bun test` (jest-compatible, parallel ≈
  `pytest-xdist`, built-in coverage).
- **Lint/format**: Biome (single tool ≈ ruff's role).
- **Distribution**: npm package with a `code-indexing-mcp` bin, plus the
  same curl-pipeable bootstrap installer (now provisioning Bun instead of
  uv). `bun build --compile` additionally allows shipping per-platform
  single-file executables, which would remove runtime provisioning from
  the install entirely — evaluate in Phase 8 once S0 settles how the
  native `.node` addons embed (decision D5).

## 4. Dependency map

| Python | TypeScript replacement | Confidence |
|---|---|---|
| `mcp` (FastMCP) | `@modelcontextprotocol/sdk` (`McpServer` + stdio transport, zod tool schemas) | High — official SDK, same protocol |
| `pydantic` v2 | `zod` v4 + inferred types | High — Phase 1 ported all ~50 models. `zod-to-json-schema` proved unnecessary: v4 ships `z.toJSONSchema()` |
| `lancedb` | `@lancedb/lancedb` (native Node bindings over the same Rust core) | High — S1 confirmed storage and index APIs; Phase 5 retains reranker scores explicitly before TypeScript-side projection so cross-project ranking does not depend on deprecated score autoprojection |
| `pyarrow` | `apache-arrow` JS **pinned to 18.1.0** for IPC staging files (LanceDB peer-requires `>=15 <=18.1.0`, so the current 21.x is not available); LanceDB JS accepts Arrow tables natively | High — confirmed by S1 |
| `tree-sitter-*` PyPI packages | `tree-sitter` native Node bindings + per-grammar npm packages; `.scm` queries port unchanged | High — S2 confirmed all 18 languages and every committed query. **The Node binding reports UTF-16 code-unit indices where the Python one reports UTF-8 byte offsets**; Phase 2 converts every node offset, since those offsets are the stored contract (see the Phase 2 notes) |
| `tree-sitter-language-pack` (PyPI) | `@kreuzberg/tree-sitter-language-pack` (306 grammars, native addons fetched on first use) | High on macOS and Linux — S2 verified `gdshader` against the Python pack's own output. The reviewed artifact identity is pinned to `@kreuzberg/tree-sitter-language-pack@1.10.9:gdshader` and checked at runtime. Used only for `gdshader`; unlike the Python side, `gdscript` and `godot_resource` have dedicated npm packages. **Its Windows binding does not load, so `gdshader` is an accepted non-goal there** (§5.5) |
| `fastembed` / `onnxruntime` | `onnxruntime-node` + `@huggingface/hub` (model download) + `@huggingface/tokenizers` — i.e. the TS port owns what `direct_onnx.py` does today, for CPU too | High — S3 measured exact vector parity |
| `mlx` | No first-party Node binding. Options: onnxruntime-node CoreML EP, `@frost-beta/mlx` community bindings, or keep the Python MLX worker as a sidecar (the worker protocol is already cross-process) | Low — decision point D2 |
| `watchfiles` | `@parcel/watcher` (native, recursive, battle-tested); confirm under Bun in S0, with `fs.watch` as fallback | Medium–High |
| `filelock` | `proper-lockfile` | High |
| `platformdirs` | `env-paths` | High |
| `psutil` | `pidusage` + `node:os`/`process.memoryUsage`; RSS ceilings via the same polling loop | High — the production sampler reads the requested worker PID through `pidusage` on every OS, while parent growth is measured separately. `settings.py`'s only use is `virtual_memory().total`, which is `node:os`'s `totalmem()` and needed no dependency (Phase 1) |
| `pathspec` | `ignore` npm package (gitignore semantics) | High — Phase 2 held it to a generated fixture of `GitIgnoreSpec`'s own verdicts; all 884 decisions agree |
| SQLite (`history.py`) | `bun:sqlite` (built-in), behind the storage adapter per §3's compat discipline | High |
| `multiprocessing` spawn + `Connection` auth | `Bun.spawn`/`node:child_process` + the same socket dial-back and HMAC challenge-response, now used for **all** workers (§5.3) | High |
| `struct` framing | `Buffer`/`DataView` | High |
| `tomllib`/`tomli-w` | `smol-toml` | High |
| `argparse` | `commander` | High |
| Textual TUI | OpenTUI Core (`@opentui/core`) | High — required by resolved decision D3; Phase 8 must build directly on the core renderer rather than Ink or a prompt-only toolkit |
| `pytest` fixtures/parametrize | `bun test` with `test.each`, fixtures as helpers | High |

## 5. Architecture decisions forced by the migration

### 5.1 The embedding stack loses FastEmbed — and doesn't need it

FastEmbed is glue over ONNX Runtime + tokenizers + model download. The
codebase already contains a from-scratch ONNX path (`direct_onnx.py`) with
model-specific pooling/normalization, because index compatibility depends on
tokenizer, pooling, normalization, dimension, and artifact revision — not
just the ONNX file. In TS that direct path becomes the *only* path: pin the
same `jinaai/jina-embeddings-v2-base-code` ONNX artifact revision, download
via `@huggingface/hub`, tokenize with the HF tokenizers bindings, run
through `onnxruntime-node`, and reuse the existing acceptance metrics
(`acceptance.py`) as the parity gate against vectors produced by the Python
build.

### 5.2 Index compatibility: verify, don't assume

The LanceDB storage format is language-neutral (same Rust core), so
existing indexes *should* open. But vectors are only reusable if the TS
embedder is numerically acceptable against the Python one under the
existing promotion-gate metrics. The plan treats both as testable claims:

- If vector parity passes the acceptance gate: migrated installs keep their
  indexes untouched.
- If it fails: bump the index schema/model revision and let the existing
  staleness machinery trigger a rebuild on first use. This is the fallback,
  not a blocker — the system already knows how to rebuild.

### 5.3 The two-environment accelerator design simplifies

The external-worker architecture exists because `fastembed` and
`fastembed-gpu` fight over the `onnxruntime` import — a Python packaging
problem. Node's equivalent conflict (CPU vs GPU onnxruntime builds) is
real but milder: `onnxruntime-node` ships CPU + CUDA/DML in one package,
and execution providers are selected at session-creation time. Plan of
record: **one environment, provider selected at runtime**, keeping the
worker-process boundary purely for memory isolation (a disposable child
that can be killed at the ceiling), via `child_process` with the same
authenticated socket dial-back. The probe → calibrate → promote → cache
pipeline ports unchanged; only "which interpreter runs the worker"
collapses.

### 5.4 What stays byte-compatible on disk

`.ci-mcp/project.toml` markers, progress JSON snapshots, probe-cache
records, accelerator-env records, the history SQLite schema, and the
staging journal format all carry over unchanged, so a half-migrated
machine (Python daemon still running, TS CLI installed) degrades loudly
via the existing daemon `PROTOCOL_VERSION` guard rather than corrupting
state.

### 5.5 One accepted capability difference: `gdshader` on Windows

`@kreuzberg/tree-sitter-language-pack` cannot load its `win32-x64` binding, and
it is the only source for the GDShader grammar on npm. Rather than ship a VC++
redistributable, run a second WASM parser stack for one language, or vendor and
build the grammar's C sources, the capability is dropped on that platform: one
shader format on one OS is worth less than any of those costs.

The consequences are small but real, and Phase 2 owns them (done: the gap is a
single table in `grammars.ts`, which `scanner.ts` reads). A missing grammar
must be a *supported state* rather than an error — `.gdshader`/`.gdshaderinc`
files are skipped on Windows the way an unsupported extension is, because an
index that fails on a Godot repository would be much worse than one that
quietly omits its shaders. And because the Python build does index them there
today, §12's cutover material has to say so.

Everything else in the Godot family is unaffected: `.gd` and `.tscn`/`.tres`
come from dedicated npm packages and work on all three platforms.

## 6. Phase 0 spikes — do these before committing to anything

Each spike is a small throwaway program with a pass/fail written into this
document's follow-up. Order matters: S0–S3 are the bets the whole plan
rests on.

**Executed 2026-08-17.** The programs live in `ts/packages/spikes/` and the
verdicts are in
[2026-08-17-phase-0-spike-results.md](2026-08-17-phase-0-spike-results.md):
all six pass. D1 resolves to *keep indexes*. D2 remains open — S3 measured the
CPU provider only.

- **S0 — Bun native-module matrix** (~2 days): under Bun on Linux, macOS,
  and Windows, load `@lancedb/lancedb`, `tree-sitter` plus one grammar,
  `onnxruntime-node`, and `@parcel/watcher`, and exercise one real call
  through each (open a table, parse a file, run an inference, watch a
  directory). These are N-API addons and Bun's N-API layer is the newest
  part of this stack; any gap found here picks that module's fallback
  (WASM tree-sitter, `fs.watch`) or — worst case, per §3's compat
  discipline — runs that one process under Node while everything else
  stays on Bun.
- **S1 — LanceDB Node parity** (~3 days): open an index written by the
  Python build; create tables from Arrow schemas; build FTS + BTree indexes
  with the config the write path needs; run hybrid query + pushdown
  predicate. Fail → escalate: the migration is blocked until the JS SDK
  gains the API (or we contribute it upstream).
- **S2 — Grammar coverage** (~2 days): load all 19 languages' grammars in
  Node, run the committed `.scm` queries against the extractor test
  fixtures, diff chunk output against the Python snapshot. Identify
  sourcing for GDScript/gdshader/godot-resource (today from
  `tree-sitter-language-pack`), HCL, SQL — may require building grammars
  from their C sources into the package.
- **S3 — Embedding parity** (~3 days): embed the probe corpus through
  onnxruntime-node + HF tokenizers; score with the ported acceptance
  metrics against Python-produced vectors. This decides §5.2.
- **S4 — Memory ceiling** (~2 days): reproduce the token-batching memory
  model in Node (RSS polling of a child, kill-at-ceiling, batch retry);
  compare peak RSS at the fixture scales `test_memory_acceptance.py` uses.
- **S5 — Watcher + scanner semantics** (~1 day): `@parcel/watcher` event
  coalescing vs `watchfiles`; `git ls-files` streaming through
  `Bun.spawn`.

## 7. Migration order

The port proceeds bottom-up through the dependency graph so every phase
lands with its tests green and nothing above it stubbed. Each phase ports
the corresponding test files in the same PR — the Python tests are the
spec, and the ~29k test lines are most of the migration's real cost.

| Phase | Content | Source ln (approx) | Estimate |
|---|---|---|---|
| 0 | ✅ Spikes S0–S5; repo scaffolding (Bun workspace, tsconfig, Biome, `bun test`, CI skeleton) | — | 2–3 wk |
| 1 | ✅ Foundations: `errors`, `models` (→ zod), `settings`, `path_filter`, `token_batching`, `acceptance`, `progress`, `projects`, `update_check` | ~2,600 | 2 wk |
| 2 | ✅ Scan & extract: `scanner`, `extractor`, query packs, `reference_service` | ~3,900 | 3–4 wk |
| 3 | ✅ Storage: `storage`, `staging`, `history` | ~2,800 | 3 wk |
| 4 | ✅ Embedding (CPU): `embedding` (direct-ONNX based, per §5.1), `embedding_worker`, `worker_launcher`, `passage_backend`, `backends`, `calibration`, `probe_cache` | ~2,900 | 3 wk |
| 5 | ✅ Orchestration & search: `indexing`, `search`, `application` | ~3,200 | 3 wk |
| 6 | Surfaces: `server` (MCP), `daemon`, `cli`, `benchmark` | ~3,600 | 3–4 wk |
| 7 | Accelerators: provider selection (CUDA/DML/WebGPU/CoreML per D2), `accelerator_env`/`accelerator_probe`, promotion gates on real hardware | ~1,200 | 2–3 wk |
| 8 | Installer: bootstrap, harness config merging (JSON/JSONC/TOML comment-preserving), shell PATH launcher, self-update, uninstall, OpenTUI Core TUI (per D3) | ~4,600 | 3–4 wk |
| 9 | Cutover: side-by-side parity soak, benchmark comparison, docs, release | — | 2 wk |

Total: roughly 23–28 engineer-weeks for a single engineer; phases 2–4 can
overlap across two engineers once Phase 1 lands.

During phases 1–8 the Python tree remains the shipping product on `main`;
the TS tree grows in-repo under `ts/` (sharing the extractor fixtures and
golden snapshots directly) and is promoted to the package root at cutover.

## 8. Parity harness (cross-cutting)

- **Golden fixtures**: the extractor fixture corpus and committed output
  snapshots under `tests/fixtures/` become language-neutral golden files
  consumed by both test suites. Any diff between Python and TS extraction
  is a migration bug by definition. Phase 1 established the shape with
  `ts/packages/server/scripts/write_path_filter_parity.py`, which records what
  the Python build emits for every search glob so the TS suite is held to it
  instead of to a reimplemented oracle. Phase 2 applied it twice more: the TS
  extractor is held to `tests/fixtures/extractor_snapshot.json` — the very file
  the Python suite gates on, not a copy — across all 18 languages, and the
  gitignore port to a generated record of `pathspec.GitIgnoreSpec`'s verdicts
  (`scripts/write_ignore_parity.py`).
- **MCP contract tests**: capture each tool's JSON schema and
  representative request/response pairs from the Python server into
  fixtures; the TS server must reproduce schemas (field names, optionality,
  serialization quirks like path-as-plain-string) exactly.
- **Dual-run soak** (Phase 9): index the same real repositories with both
  builds; diff chunk tables row-by-row, compare search rankings for a
  query set, and run the existing benchmark to bound the performance
  regression (target: within 15% on index time, no regression on search
  latency).
- **Memory gate**: the server package runs a deterministic near-cap corpus with
  256-token windows through the real ONNX worker and enforces the memory budget
  plus the existing 256 MiB allowance. A dedicated Ubuntu CI job caches the
  model and uploads the JSON report; ordinary three-OS tests stay model-free.
- **Persisted integration gates**: Phase 5 tests now drive real `Indexer` and
  `LanceStore` instances for missing reference tables, stale-file healing, and
  live progress counters. Storage tests retain `_relevance_score` and verify
  global ordering across project partitions.

## 9. Decision points (need an owner's call, flagged as they arrive)

- **D1 — Index rebuild on migrate?** Resolved by S3 (§5.2). Prefer keeping
  indexes; accept rebuild if parity fails.
- **D2 — Apple Metal path.** Try onnxruntime-node CoreML EP first (S3 can
  measure it); fall back to community MLX bindings; keeping a Python MLX
  sidecar is the option of last resort since it drags a Python runtime
  back into the install.
- **D3 — Installer TUI. Resolved: OpenTUI Core.** Phase 8 must use
  `@opentui/core` directly for the installer UI. Keep the wizard state machine
  and orchestrator events UI-agnostic, but do not substitute Ink, React-based
  terminal components, or a prompt-only toolkit for the renderer.
- **D4 — Windows daemon transport.** Keep loopback TCP + HMAC as today, or
  move both platforms to named pipes (`net.createServer` supports
  `\\.\pipe\`). Default: keep today's design; revisit only if the port
  surfaces a problem.
- **D5 — Compiled binaries.** If `bun build --compile` embeds the native
  addons cleanly (S0 tells us), the installer can ship per-platform
  single-file executables and stop provisioning a runtime at all — the
  biggest install-story simplification available in this migration.
  Decide at Phase 8; npm-package distribution works regardless.

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A native addon (lancedb, tree-sitter, onnxruntime-node, parcel-watcher) hits a Bun N-API gap | Medium | High | S0 first; core code sticks to `node:` APIs behind adapters (§3), so a single process can run under Node without forking the codebase |
| LanceDB JS lacks the FTS/BTree index-config API the write path needs | Medium | Blocking | S1 first; upstream contribution window is the schedule buffer |
| Embedding vectors drift → all user indexes rebuild | Medium | Medium | S3 + acceptance gates; rebuild path already exists and is exercised |
| ~~Niche grammars (GDScript family, SQL, HCL) unavailable as npm packages~~ | — | — | **Retired by S2**: all resolve to npm packages, `gdshader` through `@kreuzberg/tree-sitter-language-pack` |
| Native-module install pain (tree-sitter, onnxruntime, parcel-watcher, lancedb across 3 OSes) | Medium | Medium | Prebuilt binaries exist for all; CI matrix mirrors today's OS coverage from Phase 0 |
| Bun regression lands in a release (younger runtime, faster release cadence than Node LTS) | Medium | Low–Medium | Pin the Bun version in `engines` and CI; upgrade deliberately with the full suite as the gate |
| ~~`worker_threads`/child RSS accounting differs from `psutil` semantics~~ | — | — | **Retired by Phase 5 closure**: `pidusage` samples the actual worker PID cross-platform, with a real-model CI memory gate |
| Performance regression in extraction (Node bindings overhead per node visit) | Low–Medium | Medium | S2 measures on the perf fixtures from `2026-07-27-extractor-performance.md` |
| Long dual-maintenance window | High | Medium | Feature freeze on Python except critical fixes once Phase 5 lands; parity harness makes backports mechanical |

## 11. CI for the migration

From Phase 0, a second workflow mirrors today's gates for the TS tree:
Biome check, `tsc --noEmit`, and `bun test` on Ubuntu + Windows + macOS
with the pinned Bun version (`oven-sh/setup-bun`, `bun install --frozen-lockfile`).
The existing model-download and grammar caches carry over. The real-model
memory gate is present as a dedicated cached Ubuntu job; the broader benchmark
comparison joins in Phase 9. Both suites run on every PR until cutover; the
Python workflow is retired one release after the TS build ships as default.

## 12. Cutover and rollback

1. Ship the TS build as `code-indexing-mcp@next` (npm) while the Python
   build remains the installer default for one release.
2. Installer gains a `--runtime ts` flag for opt-in; harness configs are
   already absolute-path launchers, so switching is a config rewrite the
   installer knows how to do.
3. Flip the default once the dual-run soak and one release of `@next`
   feedback are clean. The Python daemon's `PROTOCOL_VERSION` guard
   retires stale daemons cleanly at switch time.
4. Rollback is the same installer path in reverse; on-disk state is shared
   (§5.4) or rebuildable, so no migration-back tooling is needed.

The release notes must call out the one deliberate capability difference:
**Windows loses GDShader indexing** (§5.5). Scripts and scenes are unaffected,
and every other platform is unaffected, but a Windows user with a Godot project
would otherwise discover it by searching for a shader and finding nothing.
