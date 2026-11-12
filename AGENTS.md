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
and `uv run pytest -n auto --dist worksteal`.

## MLX backend changes

`tests/test_mlx_backend.py` is skipped by that gate: it needs the `mlx` extra,
which the default environment and CI's Linux runners do not have, and the macOS
job only runs on manual dispatch. A change to `src/code_indexing_mcp/mlx_backend.py`
has to be run against it on Apple Silicon before pushing, without touching the
project environment:

```sh
uv run --with "mlx==0.32.0" --with "onnx>=1.22,<2" pytest tests/test_mlx_backend.py
```

Say in the PR that it was run; "N skipped" in the gate's summary includes it.
