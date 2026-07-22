# Incode MCP

Incode is a local-only codebase indexer for MCP clients. It uses Tree-sitter to extract
syntax-aware chunks, FastEmbed to create embeddings on the local machine, and LanceDB for
persistent vector and full-text search.

It does not require a hosted database, embedding API, daemon, or network transport. The only
network access is the initial download of the default
`jinaai/jina-embeddings-v2-base-code` model (approximately 640 MB). Once cached, indexing and
search work offline.

## Requirements and setup

- macOS, Linux, or Windows
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync --all-groups
uv run incode model pull
```

The model preparation step is optional; the first index operation downloads the model when it
is not already cached.

## MCP configuration

Run the server over stdio:

```bash
uv run incode serve
```

A generic MCP client configuration looks like this:

```json
{
  "mcpServers": {
    "incode": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/incode",
        "run",
        "incode",
        "serve"
      ]
    }
  }
}
```

The server exposes `init_project`, `index_project`, `project_status`, `list_projects`,
`remove_project`, `search_code`, `find_symbol`, `file_outline`, and `get_chunk`.

## Project workflow

```bash
cd /path/to/project
uv run --project /path/to/incode incode init
uv run --project /path/to/incode incode index
uv run --project /path/to/incode incode status
```

Initialization creates `.incode/project.toml` and a self-ignoring `.incode/.gitignore`. The
marker contains a checkout-local UUID and scanning configuration. It is not intended to be
committed.

Index refreshes are explicit and incremental:

- Matching size and nanosecond mtime skips reading the file.
- Changed metadata triggers SHA-256 verification.
- Unchanged content is neither parsed nor embedded.
- Changed files are replaced transactionally in LanceDB.
- Removed files are deleted from the active index.
- A parse or embedding failure preserves the previous indexed version.

Python, Python stubs, JavaScript, JSX, TypeScript, and TSX are supported. The scanner respects
root and nested `.gitignore` files and excludes symlinks, binary files, files over 1 MiB, build
outputs, virtual environments, and dependency directories.

## Multi-project search

Tools use the current MCP root or nearest `.incode/project.toml` by default. `search_code` can
instead receive a list of project IDs/names/paths or set `all_projects=true`. Searching all
projects is always explicit, preventing accidental context mixing.

`remove_project` deletes only central index data. It never removes source files or the local
`.incode` marker.

## Storage and offline operation

Platform-specific user data and cache locations are selected with `platformdirs`. Override them
when needed:

```bash
export INCODE_DATA_DIR=/path/to/index-data
export INCODE_CACHE_DIR=/path/to/model-cache
export INCODE_OFFLINE=1
```

With `INCODE_OFFLINE=1`, Incode will not download a missing model and returns
`MODEL_UNAVAILABLE` instead. Source code, embeddings, and search queries remain local; there is
no telemetry.

## Development

```bash
uv run pytest
uv run pytest --cov=incode_mcp
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

To exercise the real model integration, provide a persistent cache directory and opt in:

```bash
INCODE_MODEL_TEST_CACHE=/path/to/cache uv run pytest -m model
```

V1 intentionally excludes filesystem watching, HTTP transports, dependency/call graphs,
cross-reference resolution, custom embedding profiles, and automatic storage migrations.
