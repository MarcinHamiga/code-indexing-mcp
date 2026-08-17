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

## Two conventions worth knowing before writing code here

**Bun-only APIs live in `src/runtime/`.** Core modules import `node:`-namespaced
APIs and Web standards only, so any single process can be run under Node if a
native addon regresses under Bun. Biome enforces the boundary — `bun:*` imports
are a lint error elsewhere.

**Source must survive Node's type stripping.** Parameter properties, `enum`, and
`namespace` are lint errors: Node can execute `.ts` directly but rejects
TypeScript syntax that does not erase, and that ability is what makes the
paragraph above true rather than aspirational.
