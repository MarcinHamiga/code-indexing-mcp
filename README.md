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

### Bundled skills

The installer also symlinks four agent skills into skill-capable harnesses
(Claude Code, Kimi Code, Codex, OpenCode), pointing into the cloned repo so
they update on every re-install: `codebase-exploration` (index-first
navigation), `feature-dev` (index-grounded feature workflow), `indexed-review`
(angle-based code review), and `impact-analysis` (blast-radius mapping before a
change). Harnesses without skill support are skipped.

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

The server exposes nine tools. Read tools are annotated `readOnlyHint` so hosts may auto-approve
them; `remove_project` is annotated `destructiveHint`.

| Tool | Kind | Purpose |
| --- | --- | --- |
| `init_project` | write | Register a directory and write its `.ci-mcp/project.toml` marker. |
| `index_project` | write | Incrementally scan, parse, embed, and commit changed files. |
| `remove_project` | destructive | Delete a registration and its whole index partition. |
| `project_status` | read | Index state plus file and chunk counts. |
| `list_projects` | read | Every registered project, sorted by name. |
| `search_code` | read | Hybrid semantic and keyword search returning ranked snippets. |
| `find_symbol` | read | Exact, prefix, or substring lookup of declaration names. |
| `file_outline` | read | One file's declared symbols, metadata only. |
| `get_chunk` | read | Full stored text for one `chunk_id`. |

`limit` is capped at 50 and `match` accepts only `exact`, `prefix`, or `contains`; both are
enforced by the tool schema, so an out-of-range value is rejected rather than silently clamped.

`get_chunk` returns one chunk's full stored text with its path, symbol, line range, byte range, and
content hash. It deliberately excludes the embedding vector and the derived `embedding_text` and
`search_text` columns, which exist for ranking and are not useful to a caller.

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

Because that first query waits for the refresh, it reports progress while the initial index builds
so clients can tell a slow index from a stalled tool call. On a large repository the first query
can still take a while; `INCODE_INDEX_MODE=eager` moves the refresh to tool listing instead, and
`INCODE_INDEX_MODE=manual` restricts indexing to explicit `index_project` calls. The legacy
`INCODE_AUTO_INDEX` flag remains supported. Clients that do not provide filesystem roots keep the
explicit workflow.

Two things can make an automatic refresh wait: another root queued ahead of it in the same session,
and another process holding the global index lock. One budget covers both. The refresh retries with
exponential backoff for up to five minutes, then fails the waiting query with `INDEX_BUSY` rather
than blocking indefinitely:

```bash
export INCODE_INDEX_WAIT_SECONDS=300
```

Set it to `0` to fail immediately whenever anything else is already indexing, or raise it when a
single cold index legitimately takes longer than the default.

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

`search_code`'s `paths` argument takes glob patterns relative to the project root. Patterns match
from the right, so `*.py` matches a Python file at any depth while `src/*` matches only direct
children of `src`. A single `*` and `**` both span one path segment. Patterns are translated into
the index scan itself, so a filtered search finds matches that rank below the unfiltered result
window instead of returning an empty result.

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
export INCODE_EMBED_MAX_TOKENS=1024
export INCODE_EMBED_OVERLAP_TOKENS=64
export INCODE_EMBED_THREADS=2
export INCODE_EMBED_CPU_ARENA=0
export INCODE_VECTOR_INDEX=exact
```

The ceiling covers indexing memory: the embedding worker plus any growth in the host process while
indexing runs. Memory the host already held when the worker started — the daemon's query model and
open Lance datasets — is not charged to the budget, so a warm daemon can still index. `IndexReport`
reports both the budget and the true combined peak, plus a scan/parse/embed/commit duration split.

### Token-bounded chunks

Sequence length, not character count, drives embedding memory: attention is quadratic in tokens.
The same 4,096 characters cost wildly different amounts depending on how densely they tokenize —
ordinary source is ~984 tokens, a minified line ~2,157 — and embedding the latter as one sequence
adds ~1,172 MiB of resident memory against ~266 MiB for the same characters split into windows.

Every chunk is therefore windowed to at most `INCODE_EMBED_MAX_TOKENS` tokens with
`INCODE_EMBED_OVERLAP_TOKENS` of overlap before it reaches the model, and each window is stored as
its own chunk with its own byte and line offsets. Ordinary code is unaffected: a 1,024-token budget
is roughly 4,096 characters of source, so chunks that already fit stay whole and unchanged.
`IndexReport` carries `embedded_segments`, `embedded_tokens`, `embedding_retries`, and
`token_windowing` so a run's shape is visible without re-running it.

When a batch does trip the ceiling, the worker is replaced and the batch retried at half the
microbatch size (4 → 2 → 1) before the error surfaces. Window boundaries come only from the
tokenization, so a retry re-derives identical chunks.

Extraction is linear in file size and in definition count. Each Tree-sitter query is compiled once
per language per process, and the definition and newline indexes are built once per file. A
definition-dense generated file near the 1 MiB scan cap — 699 KB, 16,384 definitions — extracts in
well under a second; earlier releases took roughly 31 seconds on the same shape because those
indexes were rebuilt per definition.

### Measured throughput

`INCODE_EMBED_BATCH_SIZE` stays at 1. Measured with
`scripts/benchmark_index_memory.py` on Apple Silicon macOS against a 1.0 MiB, 6,330-chunk
dense-Python corpus at `INCODE_INDEX_MEMORY_MB=2048`:

| `INCODE_EMBED_BATCH_SIZE` | Wall clock | Chunks/s | Peak worker RSS |
| ------------------------- | ---------- | -------- | --------------- |
| 1 (default)               | 147.0 s    | 44.8     | 1,415 MiB       |
| 2                         | 136.2 s    | 48.7     | 1,419 MiB       |
| 4                         | 130.4 s    | 50.9     | 1,427 MiB       |
| 8                         | 126.7 s    | 52.5     | 1,451 MiB       |

Batch size 8 buys 17% throughput for 36 MiB more resident memory — not enough to justify spending
headroom that the worst-case file shape already needs. Embedding dominates: 141 s of the 147 s at
batch size 1. Plan for roughly **45 chunks per second**, and remember that in the default lazy mode
the first `search_code` call waits for that work. On a large repository prefer
`INCODE_INDEX_MODE=eager` (index during tool listing) or `INCODE_INDEX_MODE=manual` with an
explicit `code-indexing-mcp index`, so no query blocks on a cold index.

### Single-line and generated files

A single-line source file near the 1 MiB scan cap — a bundled or minified artifact — used to drive
the embedding worker past every ceiling measured (2048, 3072, and 4096 MiB alike) and abort the run
with `INDEX_RESOURCE_LIMIT`. Token-bounded windows fix that: the same file now indexes cleanly at
321/1,879/2,073 MiB parent/worker/combined against a 2,048 MiB ceiling.

Such files still cost time and index space for little retrieval value, so excluding them in
`.ci-mcp/project.toml` remains worthwhile:

```toml
[scan]
exclude = ["**/*.min.js", "**/*.bundle.js", "**/generated/**"]
```

A file that cannot be parsed, planned, or embedded is recorded as a per-file issue and skipped until
it changes. Environment failures — `MODEL_UNAVAILABLE`, `INDEX_RESOURCE_LIMIT`, and
`EMBEDDING_WORKER_FAILED` — are not attributable to a file, so they abort the run and surface to the
caller instead of silently leaving that file permanently unindexed.

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
The socket lives under `XDG_RUNTIME_DIR` when set and the platform temporary directory otherwise;
the containing directory must be a real directory owned by the current user, or startup fails
rather than binding somewhere another user controls. Startup output goes to `daemon.log` in the
data directory.

The daemon needs Unix domain sockets. Where they are unavailable — currently Windows — the default
`INCODE_BROKER=auto` serves directly and logs a warning; an explicit `INCODE_BROKER=on` fails with
`INVALID_CONFIGURATION` instead of being silently downgraded.

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
