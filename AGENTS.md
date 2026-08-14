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
