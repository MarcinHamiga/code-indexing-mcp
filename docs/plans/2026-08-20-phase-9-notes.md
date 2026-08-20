# Phase 9 notes — cutover

Date: 2026-08-20
Status: retired; tooling, CI, packaging, and the installer opt-in were
completed, but the release flip was cancelled after the real-repository
benchmark failed the migration's performance objective. See the
[retrospective](2026-08-20-typescript-migration-retrospective.md).
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §7–§12

Phase 9 delivered the cutover evidence machinery §8 calls for (the dual-run
soak and the benchmark comparison), the §11 CI job, and the §12 npm `@next`
packaging. The Python tree remains the shipping implementation; there will be
no TypeScript release flip from this branch.

## What is in the tree

| Piece | Responsibility | Tests |
|---|---|---|
| `ts/packages/server/src/soak.ts` | Snapshot schemas + chunk-diff/ranking comparator and gates | `test/soak.test.ts` |
| `ts/packages/server/scripts/write_soak_snapshot.py` | Index the manifest's repositories with the Python build; dump chunk rows + rankings | operator-run |
| `ts/packages/server/scripts/write_soak_snapshot.ts` | The same with the TypeScript build | operator-run |
| `ts/packages/server/scripts/soak_compare.ts` | Compare two snapshots, write the report, exit nonzero on gate failure | via `src/soak.ts` |
| `ts/packages/server/scripts/soak.manifest.example.json` | Manifest shape: repositories + queries + limit | — |
| `ts/packages/server/src/benchmark-compare.ts` | Index/search report comparison + the 15%/no-regression gates | `test/benchmark-compare.test.ts` |
| `ts/packages/server/scripts/benchmark_compare.ts` | Drive both CLIs (`uv run` + Bun), parse, compare, report | CI job exercises it |
| `.github/workflows/ci-ts-parity.yml` | Ubuntu job: shared model cache, warmed for both builds, comparison artifact | — |
| `ts/packages/server/package.json` | Renamed to `code-indexing-mcp` 0.4.0, publish metadata, `publishConfig.tag: next` | — |

## The soak

Run the Python writer first — `init_project` writes the
`.ci-mcp/project.toml` markers whose project ids both snapshots must share
for chunk ids to be comparable (the writers must never pass
`--force-new-id`):

```sh
uv run python ts/packages/server/scripts/write_soak_snapshot.py \
  --manifest soak.manifest.json --output soak-python.json
bun ts/packages/server/scripts/write_soak_snapshot.ts \
  --manifest soak.manifest.json --output soak-ts.json
bun ts/packages/server/scripts/soak_compare.ts \
  --python soak-python.json --typescript soak-ts.json --output soak-report.json
```

Each build indexes into its own scratch data directory (removed afterwards);
set `CODE_INDEXING_CACHE_DIR` to a shared cache so only one model download
serves the run. Both writers force `index_execution="in-process"` and
`broker_mode="off"`, exactly like the benchmark commands.

Gates:

- **chunk_rows_identical** — every `list_chunks` column (vector excluded, by
  the same design as `LanceStore.list_chunks`) must match field for field.
  The diff reports counts plus bounded examples.
- **recall_at_k ≥ 0.85** (default) — Python is the reference; at the default
  limit of 8 one swapped near-tie costs 0.125, so the floor sits below that
  or it fires on float16 rounding, not divergence.
- **rank_correlation ≥ 0.90** — the same `topKRankCorrelation` (Kendall
  tau-b over top-k windows) the precision benchmark uses.
- **model_ids_match** — a silent gate that only appears on mismatch.

### Validation run (2026-08-20, this machine, real model, CPU forced)

A 12-file corpus of real modules from both trees, four queries:

- chunk rows: 151/151 identical, zero field differences, project ids shared
  via one marker;
- rankings: mean recall 0.969, min 0.875 (one tail swap at ranks 8/9,
  scores 0.0289 vs 0.0291 — the same file, a genuine float16 tie), rank
  correlation 0.986.

The full-scale soak on the real target repositories (the plan's "index the
same real repositories") is the remaining operator step before the flip.

## The benchmark comparison

`benchmark_compare.ts` runs `benchmark index` and `benchmark search` through
both CLIs and gates the parsed JSON:

- index: per-scenario `reported_duration_ms` (the indexer's pipeline clock,
  not `wall_ms`, which would charge startup and warmup to indexing), gated
  when the Python duration is ≥ 50 ms; `post_maintenance` and
  `repeated_edits` report no pipeline duration and stay informational;
- search: per-scope `median_ms` and `p95_ms`.

Defaults are the plan's targets: `--index-tolerance 0.15`,
`--search-tolerance 0.10` (a small allowance over "no regression" because
the CI runner shares no clock with the Python process; tighten to 0 for a
strict read).

### Validation run (2026-08-20, this machine, default parameters, CPU forced)

- index: cold_start 1.043×, forced_reindex 1.047×, single_file_edit 1.051× —
  PASS within 15%;
- search: TypeScript is faster at every scope — 28.2 vs 30.0 ms (1
  project), 77.1 vs 98.3 ms (8), 412.0 vs 577.3 ms (50) on the median —
  PASS with no regression.

A tiny-scale run (`--projects 2`) showed +16% search latency from a fixed
~3 ms per-query overhead; at the default scale the JS hybrid-query path more
than pays it back. The CI job runs the default scale.

## Packaging for `@next`

`ts/packages/server` is renamed `code-indexing-mcp` (from
`@code-indexing-mcp/server`), versioned 0.4.0 to track the Python build's
`__version__`, with MIT/license/repository/engines/files metadata and
`publishConfig: { access: public, tag: next }` so a bare `npm publish`
lands on the `next` dist-tag rather than `latest`. `VERSION` in `cli.ts`
now reads the package.json (the importlib.metadata analogue), and
`ts/packages/server/README.md` is the npm-facing document, including the
Windows GDShader callout.

Publishing itself (§12 step 1) remains an operator action: `npm publish`
from `ts/packages/server`. The bins execute `.ts` sources, so the package
requires Bun ≥ 1.2 — the README and `engines.bun` say so, and D5
(`bun build --compile`) stays deferred per the Phase 8 notes.

## Release notes obligations (§12)

Draft text for the `@next` release:

> The TypeScript build is available as `code-indexing-mcp@next` (requires
> Bun ≥ 1.2; the Python build remains the default). On-disk state is
> shared: existing indexes, project markers, history, and probe caches open
> unchanged, and the daemon `PROTOCOL_VERSION` guard retires stale daemons
> of either build safely.
>
> **Windows loses GDShader indexing.** `.gdshader`/`.gdshaderinc` files are
> skipped there — the only npm source for that grammar cannot load its
> Windows binding — while GDScript and Godot resources are unaffected, as
> is every other platform.
>
> Embedding runs on CPU through `onnxruntime-node`. Accelerator providers
> (CUDA, DirectML, WebGPU, CoreML) are wired but unpromoted on real
> hardware and fall back to CPU; see the Phase 7 notes. MLX resolves to the
> CPU fallback under decision D2.

## Remaining before the default flips

1. Full-scale soak on the real target repositories (tooling above).
2. `npm publish` of `code-indexing-mcp@next` and one release of feedback.
3. Real-hardware accelerator promotion, still pending from Phase 7 — this
   machine's CUDA device was deliberately bypassed (CPU forced) so the
   comparison described like-for-like backends.
