# AGENTS.md

## Before pushing

ALWAYS run the formatter before pushing. CI rejects unformatted code at the
`Format` step, and no test or lint run happens until it passes:

```sh
uv run ruff format .
uv run ruff check .
uv run mypy src
```

Prefer formatting after every batch of edits, not only at the end. The full
gate that CI enforces is `ruff format --check .`, `ruff check .`, `mypy src`,
and `uv run pytest -n auto`.

## The `ts/` tree

`ts/` is the in-progress TypeScript port
([plan](docs/plans/2026-08-17-typescript-migration.md)) and has its own CI
workflow and its own gates. If you touched anything under `ts/`, run them from
that directory — the Python commands above do not cover it, and vice versa:

```sh
cd ts && bun run check
```

That is `biome format --check`, `biome lint`, `tsc --noEmit`, and `bun test`.
See [ts/README.md](ts/README.md) for the two conventions the linter enforces
there.
