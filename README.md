# Code Indexing MCP

Code Indexing MCP is a local-only codebase indexer for MCP clients. It uses Tree-sitter to extract
syntax-aware chunks, FastEmbed to create embeddings on the local machine, and LanceDB for
persistent vector and full-text search.

It does not require a hosted database, embedding API, or network service. A private per-user
daemon is started on demand so all connected MCP clients share one scheduler and model. The only
network access is the initial download of the default
`jinaai/jina-embeddings-v2-base-code` model (approximately 640 MB). Once cached, indexing and
search work offline.

## Install

- [Git](https://git-scm.com/)
- [uv](https://docs.astral.sh/uv/)
- Python 3.12 or 3.13

On macOS or Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/install.sh | sh
```

On Windows PowerShell:

```powershell
$installer = Join-Path $env:TEMP "code-indexing-mcp-install.py"
Invoke-WebRequest https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/install.py -OutFile $installer
py -3 $installer
```

The installer clones the repository to `~/.local/share/code-indexing-mcp`, creates its locked
virtual environment, and displays this multi-select menu:

1. Codex (CLI + Desktop)
2. Claude Code
3. Kimi Code
4. Claude Desktop
5. OpenCode
6. KiloCode

Codex CLI and Codex Desktop share one configuration and therefore use one menu choice.
Configuration changes are limited to the `code-indexing-mcp` entry. An existing configuration is
backed up alongside the original with a `.bak` suffix before it changes.

Run the same command later to update an existing clean checkout with a fast-forward-only pull and
refresh its environment. The installer refuses to overwrite a different repository or a checkout
with local changes.

For a noninteractive installation, pass comma-separated harness slugs or `all`:

```bash
curl -fsSL https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/install.sh |
  sh -s -- --harnesses codex,claude-code,opencode
```

Use `--install-dir /custom/path` or `CODE_INDEXING_MCP_INSTALL_DIR` to change the checkout
location. Run `python3 install.py --help` for all installer options.

## Manual setup

```bash
git clone https://github.com/MarcinHamiga/code-indexing-mcp.git
cd code-indexing-mcp
uv sync --locked
uv run code-indexing-mcp model pull
```

The model preparation step is optional; the first index operation downloads the model when it
is not already cached.

## MCP configuration

Run the server over stdio:

```bash
uv run code-indexing-mcp serve
```

A generic MCP client configuration looks like this:

```json
{
  "mcpServers": {
    "code-indexing-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/code-indexing-mcp",
        "run",
        "code-indexing-mcp",
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
uv run --project /path/to/code-indexing-mcp code-indexing-mcp init
uv run --project /path/to/code-indexing-mcp code-indexing-mcp index
uv run --project /path/to/code-indexing-mcp code-indexing-mcp status
```

Initialization creates `.ci-mcp/project.toml` and a self-ignoring `.ci-mcp/.gitignore`. The
marker contains a checkout-local UUID and scanning configuration. It is not intended to be
committed. Markers created by earlier releases under `.incode` remain readable, but all new
markers use `.ci-mcp`.

CLI index refreshes are explicit and incremental. MCP indexing is lazy by default: listing tools
does not discover projects, load the model, or start indexing. The first project-scoped code query
discovers and refreshes each qualifying root, then waits for that bounded refresh. A new root
qualifies when it has at least one supported,
non-ignored source file and contains `.git`, `pyproject.toml`, `setup.py`, `setup.cfg`,
`package.json`, `tsconfig.json`, or `jsconfig.json`. The server creates the usual local
`.ci-mcp/project.toml` marker only after that check passes.

Set `INCODE_INDEX_MODE=eager` to refresh when tools are listed or
`INCODE_INDEX_MODE=manual` for explicit-only indexing. The legacy `INCODE_AUTO_INDEX` flag remains
supported. Clients that do not provide filesystem roots keep the explicit workflow.

Incremental refreshes:

- Matching size and nanosecond mtime skips reading the file.
- Changed metadata triggers SHA-256 verification.
- Unchanged content is neither parsed nor embedded.
- Changed files are replaced transactionally in LanceDB.
- Removed files are deleted from the active index.
- A parse or embedding failure preserves the previous indexed version.

Python, Python stubs, Java, JavaScript, JSX, TypeScript, and TSX are supported. Java indexing
extracts classes, interfaces, records, enums, annotation types, methods, constructors, and nested
declarations without requiring a JDK, Maven, or Gradle. The scanner respects root and nested
`.gitignore` files and excludes symlinks, binary files, files over 1 MiB, build outputs, virtual
environments, and dependency directories.

Existing project markers that use the exact pre-Java default include list automatically include
`**/*.java` at runtime. If you use a customized `scan.include` list, add `**/*.java` explicitly.

## Multi-project search

Tools use the current MCP root or nearest `.ci-mcp/project.toml` by default. `search_code` can
instead receive a list of project IDs/names/paths or set `all_projects=true`. Searching all
projects is always explicit, preventing accidental context mixing.

`remove_project` deletes only central index data. It never removes source files or the local
`.ci-mcp` marker.

## Storage and offline operation

Platform-specific user data and cache locations are selected with `platformdirs`. Override them
when needed:

```bash
export INCODE_DATA_DIR=/path/to/index-data
export INCODE_CACHE_DIR=/path/to/model-cache
export INCODE_OFFLINE=1
```

Indexing uses a spawned embedding worker with an adaptive ceiling of 25% of physical RAM, clamped
to 1–2 GiB and reduced further to retain 512 MiB of currently available RAM for the system.
Configure it with:

```bash
export INCODE_INDEX_MEMORY_MB=1536
export INCODE_EMBED_BATCH_SIZE=1
export INCODE_EMBED_THREADS=2
export INCODE_EMBED_CPU_ARENA=0
export INCODE_VECTOR_INDEX=exact
```

`INCODE_INDEX_EXECUTION=in-process` is a temporary diagnostic rollback. It does not enforce the
worker ceiling. `INCODE_VECTOR_INDEX=hnsw` opts into approximate vector indexing; exact search is
the safer default.

All stdio adapters use the per-user daemon by default. Administrative commands are:

```bash
uv run code-indexing-mcp daemon status
uv run code-indexing-mcp daemon restart
uv run code-indexing-mcp daemon stop
```

Set `INCODE_BROKER=off` or run `serve --direct` to bypass it. The daemon authenticates over a
current-user-only local socket, starts under leader election, and exits after five idle minutes.

Storage schema v2 keeps a registry plus one LanceDB partition per project. On first upgrade from
v1, the old global store is moved to a timestamped `lancedb-v1-backup-*` directory and projects are
rebuilt lazily from source. Old chunk rows are never copied, which repairs duplicate chunk IDs.

With `INCODE_OFFLINE=1`, Code Indexing MCP will not download a missing model and returns
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

The project intentionally excludes filesystem watching, HTTP transports, dependency/call graphs,
cross-reference resolution, and custom embedding profiles.

## License

This project is licensed under the [MIT License](LICENSE).
