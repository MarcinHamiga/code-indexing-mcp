# TypeScript tree

The in-progress port described in
[docs/plans/2026-08-17-typescript-migration.md](../docs/plans/2026-08-17-typescript-migration.md).
Until cutover, the Python tree at the repository root remains the shipping
product; this tree grows beside it and is promoted to the package root at
Phase 9.

Phase 0 verdicts are in
[docs/plans/2026-08-17-phase-0-spike-results.md](../docs/plans/2026-08-17-phase-0-spike-results.md).

## Layout

```
ts/
  packages/server/      the MCP server, CLI, daemon, and pipeline (Phases 1-7)
  packages/installer/   the installer and its wizard (Phase 8)
  packages/spikes/      the Phase 0 experiments -- see its README
```

`packages/server/src` currently holds Phases 1–4, each named after the Python
module it ports:

```
errors.ts          models.ts        settings.ts      backends.ts
path-filter.ts     token-batching.ts  acceptance.ts
progress.ts        projects.ts      update-check.ts
paths.ts           the pathlib semantics the rest of it leans on
scanner.ts         extractor.ts     reference-service.ts
storage.ts         staging.ts       history.ts
embedding.ts       direct-onnx.ts   calibration.ts   probe-cache.ts
embedding-worker.ts  worker-launcher.ts  worker-channel.ts
passage-backend.ts
runtime/           the one directory allowed to import bun:* APIs
```

## Checks

Run these before pushing. They are the same four gates CI applies, in the same
order, and they mirror what `ruff format` / `ruff check` / `mypy` / `pytest` do
for the Python tree:

```sh
bun install
bun run check          # format:check, lint, typecheck, test
```

Individually:

```sh
bun run format         # Biome, writes
bun run lint           # Biome lint
bun run typecheck      # tsc --noEmit, per package
bun test
```

`tsc --noEmit` is not optional and not redundant with running the code: Bun
executes TypeScript by stripping types without checking them, so this is the
only gate standing where `mypy --strict` stands today.

## Four conventions worth knowing before writing code here

**Bun-only APIs live in `src/runtime/`.** Core modules import `node:`-namespaced
APIs and Web standards only, so any single process can be run under Node if a
native addon regresses under Bun. Biome enforces the boundary — `bun:*` imports
are a lint error elsewhere.

**Source must survive Node's type stripping.** Parameter properties, `enum`, and
`namespace` are lint errors: Node can execute `.ts` directly but rejects
TypeScript syntax that does not erase, and that ability is what makes the
paragraph above true rather than aspirational.

**Model fields stay snake_case; everything else is camelCase.** The models in
`src/models.ts` *are* the wire contract — the MCP tool schemas are generated from
them, the `.ci-mcp/project.toml` marker and the progress snapshot are written
from them, and the daemon frames them. Renaming them would mean a hand-written
mapping for each of the fifty models, which is exactly where a migration hides
its bugs. Nothing outside that file follows the convention.

**A schema and its type share a name.** `ProjectInfo.parse(raw)` is the
constructor and `ProjectInfo` is the type, so the two halves read the way
`ProjectInfo.model_validate(raw)` and `ProjectInfo` did in Python.

## Parity fixtures

Where a port has to agree with the Python build *exactly* rather than merely
behave sensibly, the oracle is a committed fixture generated from the shipping
implementation rather than a reimplementation of it in the test. Today that is
the search path pushdown:

```sh
uv run python ts/packages/server/scripts/write_path_filter_parity.py
```

It records every regex `glob_to_regex` emits, character for character, and the
ground-truth `PurePosixPath.match` result for 37 patterns against 932 corpus
paths. Regenerate it whenever either implementation changes, and expect the diff
to be empty.
