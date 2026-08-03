# Code Indexing MCP

Code Indexing MCP is a local-only codebase indexer for MCP clients. It uses Tree-sitter to extract
syntax-aware chunks, FastEmbed for CPU embeddings, a direct ONNX path for selected passage
accelerators, and LanceDB for persistent vector and full-text search.

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
virtual environment, and opens an interactive wizard. The wizard walks through accelerator
selection (with live hardware detection), the MCP clients to configure, and the server's
settings — indexing, embedding, memory, storage, and offline behavior — before a summary
screen runs everything. Settings you change are written into each client's MCP configuration
(an `env` block, or `environment` for OpenCode and KiloCode); anything you leave at its
default is not written anywhere.

The supported clients are Codex (CLI + Desktop, one shared configuration), Claude Code,
Kimi Code, Claude Desktop, OpenCode, and KiloCode. Configuration changes are limited to the
`code-indexing-mcp` entry; an existing configuration is backed up alongside the original
with a `.bak` suffix before it changes, and unrelated keys in the entry's env block are
preserved. That `.bak` is the file as *you* wrote it and is never overwritten — later
writes roll into a `.bak.prev` beside it, so re-running `configure` cannot cost you the
original. For Codex the whole `[mcp_servers.code-indexing-mcp]` table is rewritten, so any
other key you added inside that one table is replaced rather than merged.

Later, `code-indexing-mcp update` updates an existing clean checkout with a fast-forward-only
pull and refreshes its environment — see [Update](#update). Re-running the install command above
does the same thing. To change settings or harnesses without updating, run
`code-indexing-mcp configure` — it opens the same wizard offline, prefilled from your
current configuration. Naming what to change applies it directly instead:
`code-indexing-mcp configure --set CODE_INDEXING_BROKER=off --unset CODE_INDEXING_INDEX_MODE`.

### The `code-indexing-mcp` command

Your MCP clients launch the server by absolute path and never need it on PATH, so the
installer adds a launcher for you: a symlink at `~/.local/bin/code-indexing-mcp` (a `.cmd`
shim on Windows) pointing at the executable inside the installation's virtual environment.
That is what makes `configure`, `index`, `status`, `projects`, `model`, and `daemon` work
from any shell.

If that directory is not already on your `PATH`, the wizard offers — checked by default — to
add it to your shell profile (`~/.zshrc`, `~/.bashrc` and `~/.bash_profile` on macOS,
`~/.config/fish/config.fish`, or `~/.profile`) as a marked block:

```sh
# >>> code-indexing-mcp >>>
export PATH="$HOME/.local/bin:$PATH"
# <<< code-indexing-mcp <<<
```

The block is written once — a second install finds it and leaves the file alone, as does a
profile where you already put that directory on PATH yourself — and the original is backed up
with a `.bak` suffix first. The entry only reaches shells started afterwards; the installer
prints the `exec` line that makes it live in the session you are sitting in. Windows profiles
are not edited; add the directory yourself.

An existing file at the launcher's name that the installer did not create is moved aside to
`code-indexing-mcp.bak` rather than overwritten. If some *other* `code-indexing-mcp` sits
earlier on your PATH, the wizard says so instead of quietly losing the name.

Three flags control all of this, on both `install.py` and `configure`:

```bash
python3 install.py --bin-dir ~/bin      # put the launcher somewhere else
python3 install.py --no-modify-path     # create the launcher, never touch a shell profile
python3 install.py --no-launcher        # do not create a launcher at all
```

`CODE_INDEXING_MCP_BIN_DIR` and `XDG_BIN_HOME` set the directory too, in that order of
preference. None of these count as "scripted" flags, so passing them still opens the wizard.

On a terminal that cannot host the wizard (or with `--no-tui`), the installer falls back to
a plain text interface with the numbered harness menu. Flags that already say what to
install — `--harnesses`, `--set`, `--unset`, `--no-prompt` — skip the wizard too, so a
scripted run never waits for a keypress; pass `--tui` to open it over them anyway. For a
fully noninteractive installation, pass harness slugs and any settings:

```bash
curl -fsSL https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/install.sh |
  sh -s -- --harnesses codex,claude-code,opencode --set CODE_INDEXING_INDEX_MODE=eager
```

By default the installer detects whether this machine can be given an automatic GPU
accelerator for indexing and prepares one when it can. Experimental backends must be named
explicitly:

```bash
python3 install.py --accelerator auto      # CPU, a supported CUDA installation, or Metal via MLX
python3 install.py --accelerator mlx       # Metal on Apple Silicon through MLX
python3 install.py --accelerator webgpu   # experimental Metal, Vulkan, or D3D12 path
python3 install.py --accelerator migraphx # experimental pinned AMD/ROCm path
```

Detection that finds nothing, an environment that cannot be built, and a probe that does not
pass all leave the installation on CPU and report why. Nothing here changes system drivers,
and no package is ever installed while the server is running. See
[Embedding backends](#embedding-backends).

Preparing an accelerator downloads the embedding model, because the probe that confirms it
embeds a real passage on the device. A later run reuses an environment whose accelerator,
driver, Python, and runtime-lock fingerprint all still match its record, so only the first
install — and one that finds something moved — pays for the build and the probe.
Reinstalling as `--accelerator cpu` removes the environment again.

Use `--install-dir /custom/path` or `CODE_INDEXING_MCP_INSTALL_DIR` to change the checkout
location. Run `python3 install.py --help` for all installer options.

### Update

`code-indexing-mcp update` updates an existing installation in place. It checks first — git on
PATH, a clean checkout on `main` whose origin is this repository, and `uv` resolvable — and
refuses with one actionable sentence before touching anything if any of that does not hold. Then
it fast-forwards to the tip of `main`, refreshes the locked environment, rebuilds the prepared
accelerator environment only when the pull moved the runtime lock it was built against, stops the
daemon so the next client starts it on the new code, and re-runs the installation checks for the
harnesses you already have configured. Two updates cannot run at once; the second refuses
immediately.

```bash
code-indexing-mcp update
code-indexing-mcp update --check              # report only, change nothing
code-indexing-mcp update --skip-accelerator   # defer the accelerator rebuild
```

`--check` prints a JSON report of the local and remote commits and exits `0` when you are up to
date, `10` when an update is available, and `1` when the check itself could not be made — offline,
no checkout, or an origin that is not this repository.

`--skip-accelerator` leaves a stale accelerator environment in place. Nothing retires it at
runtime, so the previously prepared accelerator keeps serving the *old* locked runtime until you
rebuild it; the command prints the exact repair command.

Once a day, in the background, the CLI compares your checkout against the tip of `main` and prints
one line on stderr when an update is available. It never updates anything by itself, it is silent
on a development checkout, and `CODE_INDEXING_UPDATE_CHECK=off` turns it off entirely.

Restart your MCP clients after an update: a running server keeps the code it started with.

On Windows, an update that raises the project's required Python cannot rebuild the environment it
is running from — the sync fails and says so; re-run the installer.

### Repair and uninstall

`code-indexing-mcp configure --repair` re-applies the cheap steps — the launcher, the client
entries, and the skill links — for the harnesses already configured, keeping the prepared
accelerator and every setting exactly as they are. It changes no choice; it puts back what
went missing.

`code-indexing-mcp uninstall` removes what the installer added: the `code-indexing-mcp` entry
from each configured client, the bundled skill links, the launcher, and the PATH block. It
prints what it will do and asks before doing any of it (`--yes` skips the prompt).

Removal is evidence-based, not name-based. A launcher that does not point into *this*
checkout stays, even if it points into some other virtual environment; a skill directory
entry that does not point into this checkout stays, so a second installation keeps its own
links; a PATH block whose end marker was edited away stays, because removing to end-of-file
would take your edits with it. Client configs are restored to what they were before the
install — comments, formatting, and neighbouring servers included.

Indexes and caches are **kept** by default; they cost minutes of CPU to rebuild and an
uninstall that discards them silently is not one you can undo. The checkout is kept too:

```bash
code-indexing-mcp uninstall                    # entries, skills, launcher, PATH block
code-indexing-mcp uninstall --purge            # also delete the index and cache directories
code-indexing-mcp uninstall --remove-checkout  # also delete ~/.local/share/code-indexing-mcp
code-indexing-mcp uninstall --keep-launcher --keep-path   # leave the command in place
```

Both deleting flags check the directory before touching it. `--purge` removes a data or
cache directory only if it is named `code-indexing-mcp` or holds a recognizable index or
model cache, and `--remove-checkout` only a directory that actually looks like a checkout.
A `CODE_INDEXING_DATA_DIR` pointing somewhere else is reported and left alone: a
confirmation prompt is not a safety net for a recursive delete you cannot undo.

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
uv sync --locked --extra cpu
uv run code-indexing-mcp model pull
```

`--extra cpu` is required: the embedding runtime is an extra rather than a plain dependency,
because the CPU, CUDA, WebGPU, and MIGraphX runtimes conflict and cannot share one environment. See
[Embedding backends](#embedding-backends). Add `--extra tui` as well if you want the
`code-indexing-mcp configure` wizard; without it, scripted `configure --set` still works.

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

The server exposes nine tools. Only `list_projects` and `get_chunk` are annotated `readOnlyHint`,
so hosts may auto-approve them. The other query tools are not: on a root the server has not seen
before they register it first, which writes a `.ci-mcp/project.toml` marker, and the three code
queries also build its initial index. `remove_project` is annotated `destructiveHint`;
`init_project` carries the same hint and is non-idempotent because `force_new_id=true` can
overwrite a marker and orphan the previous index.

| Tool | Kind | Purpose |
| --- | --- | --- |
| `init_project` | destructive write | Register a directory and write its `.ci-mcp/project.toml` marker. |
| `index_project` | write | Incrementally scan, parse, embed, and commit changed files. |
| `remove_project` | destructive | Delete a registration and its whole index partition. |
| `project_status` | read, registers | Index state plus file and chunk counts. |
| `list_projects` | read only | Every registered project, sorted by name. |
| `search_code` | read, registers and indexes | Hybrid semantic and keyword search returning ranked snippets. |
| `find_symbol` | read, registers and indexes | Exact, prefix, or substring lookup of declaration names. |
| `file_outline` | read, registers and indexes | One file's declared symbols, metadata only. |
| `get_chunk` | read only | Full stored text for one `chunk_id`. |

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

`index` reports live progress — the phase, how many files it has walked against the previous run's
count, and how many chunks it has embedded — on stderr, as a status line on a terminal and as
periodic lines when redirected. The JSON report stays alone on stdout, so piping is unaffected. The
same numbers reach an MCP client as tool progress notifications, whether the work runs in this
process or in the shared daemon.

Benchmark the CPU indexing pipeline with a generated, deterministic corpus:

```bash
uv run --project /path/to/code-indexing-mcp code-indexing-mcp benchmark index \
  --files 128 --functions-per-file 2 --batch-size 8
```

The command writes one JSON document to stdout with `cold_start`, `warm_index`,
`incremental_index`, and `forced_reindex` results, including phase timings, peak memory, and
chunks per second. It reuses the configured model cache but isolates index data in a temporary
workspace. Pass `--work-dir /fresh/path` to retain the corpus and index for inspection.

Initialization creates `.ci-mcp/project.toml` and a self-ignoring `.ci-mcp/.gitignore`. The
marker contains a checkout-local UUID and scanning configuration. It is not intended to be
committed. Markers created by earlier releases under `.code-indexing-mcp` remain readable, but all new
markers use `.ci-mcp`.

CLI index refreshes are explicit and incremental. MCP indexing is lazy by default: listing tools
does not discover projects, load the model, or start indexing. Every project-scoped code query
compares eligible file paths, sizes, and nanosecond mtimes with the stored index. It refreshes and
waits only when that metadata has changed; the first query also discovers and indexes qualifying
roots. A new root qualifies when it has at least one supported,
non-ignored source file and contains `.git`, `pyproject.toml`, `setup.py`, `setup.cfg`,
`package.json`, `tsconfig.json`, or `jsconfig.json`. The server creates the usual local
`.ci-mcp/project.toml` marker only after that check passes.

Because a query waits for any required refresh, it reports progress while the initial index builds
so clients can tell a slow index from a stalled tool call. On a large repository the first query
can still take a while. `CODE_INDEXING_INDEX_MODE=eager` indexes during tool listing, then keeps one
debounced filesystem monitor per discovered root and refreshes it after later changes. Changes that
arrive during a refresh are coalesced into one follow-up pass. A stat-only reconciliation every 30
seconds catches missed notifications and Git exclusion changes outside the watched root, and a
failed filesystem watcher restarts with bounded backoff. `CODE_INDEXING_INDEX_MODE=manual` restricts
indexing to explicit `index_project` calls. The legacy `CODE_INDEXING_AUTO_INDEX` flag remains
supported. Clients that do not provide filesystem roots can still auto-refresh explicitly selected,
already registered projects; discovering a new project requires a root or `init_project`.

`project_status` performs the same metadata comparison without rebuilding and reports `stale` when
a stored `ready` or `partial` index has drifted from the source tree.

Two things can make an automatic refresh wait: another root queued ahead of it in the same session,
and another process holding the global index lock. One budget covers both. The refresh retries with
exponential backoff for up to five minutes, then fails the waiting query with `INDEX_BUSY` rather
than blocking indefinitely:

```bash
export CODE_INDEXING_INDEX_WAIT_SECONDS=300
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
- Binary and undecodable files are detected while the file is read for indexing, not during the
  scan, so each changed file is read once. They count toward `skipped_files` and are not recorded as
  per-file errors.

These languages are supported, all through bundled grammars that need no toolchain of their own —
no JDK, Maven, Gradle, .NET SDK, Godot, or database connection:

The `languages` column is the exact value `search_code`'s `languages` filter accepts, which is not
always the lowercased language name — `.tsx` is classified as its own language rather than as
TypeScript, and the two Godot data formats are one language.

| Language   | Extensions                     | `languages` value | Extracted symbol kinds                                       |
| ---------- | ------------------------------ | ----------------- | ------------------------------------------------------------ |
| Python     | `.py`, `.pyi`                  | `python`          | classes, functions, methods                                  |
| Java       | `.java`                        | `java`            | classes, interfaces, records, enums, annotation types, methods, constructors, enum constants |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs`  | `javascript`      | classes, functions, methods                                  |
| TypeScript | `.ts`, `.mts`, `.cts`          | `typescript`      | classes, interfaces, type aliases, enums, functions, methods  |
| TSX        | `.tsx`                         | `tsx`             | classes, interfaces, type aliases, enums, functions, methods  |
| C#         | `.cs`, `.csx`                  | `csharp`          | classes, interfaces, structs, records, enums, enum members, delegates, methods, local functions, constructors, destructors, properties |
| GDScript   | `.gd`                          | `gdscript`        | classes (including `class_name` and inner classes), functions, methods, signals, enums, constants |
| Godot shaders | `.gdshader`, `.gdshaderinc` | `gdshader`        | functions, structs, uniforms (as properties), constants       |
| Godot scenes and resources | `.tscn`, `.tres`, `.godot` | `godot_resource` | scene nodes by `name`, resource references by `id` |
| SQL        | `.sql`                         | `sql`             | tables, views, materialized views, indexes, functions, triggers, types |
| YAML       | `.yaml`, `.yml`                | `yaml`            | collection-valued keys, qualified by their path               |
| JSON       | `.json`                        | `json`            | collection-valued keys, qualified by their path               |

Nested declarations are qualified by their enclosing scope in every language, so a C# method
indexes as `Outer.Inner.Work` and a Compose service port list as `services.web.ports`.

The kinds above name what is extracted, not always the `kind` recorded against it. A C# destructor
is recorded as a `method`, under the type's own name — `~Catalog` indexes as `Catalog.Catalog`,
the same symbol as the constructor — because the grammar names it with the type identifier alone.
Filter on `method` to reach either.

YAML and JSON deliberately extract only keys whose value is a mapping/object or a
sequence/array. Making a symbol of every scalar leaf would turn one large configuration file into
thousands of one-line chunks; scalars still reach the index inside the enclosing key's chunk.
SQL `CREATE PROCEDURE` is not extracted, because the bundled SQL grammar does not parse it.

Godot scene and resource files name only some of their sections: `[gd_scene]`, `[gd_resource]`,
`[resource]`, `[connection]`, and the sections of a `project.godot` carry neither a `name` nor an
`id`, so they reach the index as searchable text rather than as symbols.

The scanner respects Git's complete standard ignore stack: root and nested `.gitignore` files,
`.git/info/exclude`, and the configured global excludes file. It also excludes symlinks, binary
files, files over 1 MiB, build outputs, virtual environments, dependency directories, and Godot's
own `.godot` asset cache — a `project.godot` file is still indexed. That 1 MiB cap is what
usually keeps a large generated `package-lock.json` or similar out of the index; exclude it through
`scan.exclude` if a smaller one is not worth indexing.

Existing project markers whose `scan.include` still holds an older default list are upgraded to the
current one at runtime, so a project created before a language was supported picks it up without
being re-initialized. A customized `scan.include` list is never rewritten — add the patterns you
want (`**/*.cs`, `**/*.gd`, `**/*.gdshader`, `**/*.tscn`, `**/*.tres`, `**/*.sql`, `**/*.yaml`,
`**/*.yml`, `**/*.json`) explicitly.

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
export CODE_INDEXING_DATA_DIR=/path/to/index-data
export CODE_INDEXING_CACHE_DIR=/path/to/model-cache
export CODE_INDEXING_OFFLINE=1
```

Indexing uses a spawned embedding worker with an adaptive ceiling of 25% of physical RAM, clamped
to 1–2 GiB and reduced further to retain 512 MiB of currently available RAM for the system.
Configure it with:

```bash
export CODE_INDEXING_EMBED_MEMORY_MB=1536   # CODE_INDEXING_INDEX_MEMORY_MB is the older name and still works
export CODE_INDEXING_EMBED_BATCH_SIZE=auto  # auto, or 1–256
export CODE_INDEXING_EMBED_MAX_TOKENS=1024
export CODE_INDEXING_EMBED_OVERLAP_TOKENS=64
export CODE_INDEXING_EMBED_THREADS=2
export CODE_INDEXING_EMBED_CPU_ARENA=0
export CODE_INDEXING_VECTOR_INDEX=exact
```

The ceiling covers indexing memory: the embedding worker plus any growth in the host process while
indexing runs. Memory the host already held when the worker started — the daemon's query model and
open Lance datasets — is not charged to the budget, so a warm daemon can still index. `IndexReport`
reports both the budget and the true combined peak, plus a scan/parse/embed/commit duration split.

Microbatches are bounded by three things at once: the item count above, the token budget per window,
and the padded matrix `item_count × longest_padded_tokens`, which is what a batch actually
materializes. That last budget scales with `CODE_INDEXING_EMBED_MEMORY_MB` — it is memory charged to the
same ceiling — floored so a single longest window always forms a batch and capped at eight times the
2 GiB reference, because padding cost is quadratic in the widest member and nothing has been
measured past that.

A batch that overruns the ceiling is halved and retried, and the size that survived is adopted for
the rest of that run — otherwise every group after it asks for the size that just overran and pays
the same retries again — and written to the probe cache so the next run starts there rather than
rediscovering the same limit. `model status` reports that size as `"reduced"` rather than
`"measured"`, so a machine pinned low by one bad run says so. An explicit `CODE_INDEXING_EMBED_BATCH_SIZE`
overrides it outright.

The same applies to a batch that takes the worker down with it rather than tripping the ceiling,
which is what a device allocation failure usually looks like: the sizes measured below it still
stand, so the backend is calibrated rather than left unmeasured. Only the memory ceiling has a
setting behind it, so only that case recommends one.

### Embedding backends

Acceleration targets passage indexing only. The query model stays in the serving process on CPU, so
a search never waits on a worker spawning or a model loading onto a device.

```bash
export CODE_INDEXING_EMBED_ACCELERATOR=auto  # auto, cpu, cuda, mlx, webgpu, migraphx, coreml
export CODE_INDEXING_EMBED_STRICT=0          # 1 disables the CPU fallback and the crossover
export CODE_INDEXING_EMBED_CROSSOVER=auto    # auto, off, or a character count
export CODE_INDEXING_EMBED_CALIBRATE=1       # 0 declines the one-time measurement
```

`auto` selects the best backend that has passed its promotion gates *and* that this installation
actually offers. CUDA is the first backend promoted to automatic selection; WebGPU and MIGraphX are
still experimental and Core ML stays manual-only, because on this model it offloaded only part of
the graph and lost to CPU. Naming a backend explicitly overrides its stability, and is how one earns
the benchmark evidence its own promotion needs.

Promotion makes CUDA *eligible*, not present. `auto` still resolves to CPU on a machine where the
installer never prepared it, which is every machine without a supported NVIDIA driver.

The locked installation matrix is:

| Backend | Installer support | Pinned runtime | Low-level route | Selection |
| --- | --- | --- | --- | --- |
| CPU | Python 3.12/3.13 on every supported OS | `fastembed` | CPU | automatic fallback |
| CUDA | Linux x86-64 or Windows x86-64; NVIDIA driver 525.60+ or 527.41+ | ONNX Runtime GPU 1.22–1.23, CUDA 12, cuDNN 9 | CUDA | automatic |
| MLX | macOS 14+ Apple Silicon | MLX 0.32.0 | Metal | automatic |
| WebGPU | macOS 14+ Apple Silicon, Linux x86-64 with glibc 2.27+, or Windows x86-64 | ONNX Runtime 1.24.4 + WebGPU plugin 0.1.0 | Metal, Vulkan, or D3D12/Vulkan | experimental, explicit only |
| MIGraphX | Linux x86-64, Python 3.12, ROCm 7.2.1 | AMD ONNX Runtime/MIGraphX 1.23.2 | ROCm/MIGraphX | experimental, explicit only |
| Core ML | macOS, when the serving runtime exposes it | serving CPU environment | Core ML | manual only |

An unsupported explicit MIGraphX request tries the locked WebGPU path when that platform has the
complete plugin/core wheel pair, then falls back to CPU. An unsupported explicit MLX request does
not: a request for Metal on a machine with no Metal is not a request for Vulkan, so it reports CPU
and says why. WebGPU and MIGraphX are not considered by `auto` while experimental.

MLX is the Metal path because WebGPU lost on Apple Silicon. On an M4 Pro, WebGPU indexed 1,000
chunks at 1.11× of CPU against a 1.25× promotion threshold; MLX reached 1.52–1.56× on the same
corpus, with vectors matching CPU to cosine 1.0 and identical top-5 rankings, so `auto` prepares it
there.

#### The accelerator environment

`fastembed` and `fastembed-gpu` install the same module over two different ONNX Runtime
distributions that both own the `onnxruntime` import. The direct WebGPU and MIGraphX runtimes own
that import too, so none can share an environment. The serving environment is therefore pinned to
the `cpu` extra — it embeds queries in-process and is what every accelerator falls back to — and an
accelerator gets a second locked environment of its own under the install directory, sharing the
same model cache. All five runtime extras are resolved from one lockfile that declares them
mutually exclusive.

The installer prepares that environment and records it at `accelerator.json` in the runtime data
directory. It is written only after a real inference passes in the environment it describes:
detection nominates a backend, and only the probe confirms one. The record is re-checked on every
start — a missing interpreter or a server upgraded past the Python the environment was built for
retires it with a reason rather than being repaired, since repairing it would mean installing
something while serving.

CUDA continues through FastEmbed. WebGPU and MIGraphX instead use the project-owned direct ONNX
passage model because FastEmbed cannot configure those providers. It resolves the same
`jinaai/jina-embeddings-v2-base-code` snapshot and ONNX artifact as CPU, applies the same tokenizer
configuration, attention-mask mean pooling, float32 normalization, and 768-vector shape, and
reports the providers the session actually resolved.

MLX cannot execute ONNX at all, so it reproduces that model instead of running it: the float32
initializers are lifted out of the same ONNX artifact and the JinaBERT v2 graph they belong to —
ALiBi, post-norm query and key projections, a GEGLU feed-forward — is written out again in MLX. The
tokenizer, pooling, normalization, and vector shape are shared with the direct ONNX model, and the
configuration and every tensor name and shape are checked on load, so an artifact whose graph moved
fails rather than returning vectors that no longer retrieve. The one-time conversion is written to
`<cache>/models/mlx/<revision>-jina-v1-f32.safetensors` and memory-mapped afterwards; it is keyed by
model revision, so a model that moves is converted again rather than read from a stale file.

A passage worker for an accelerator runs from that environment's own interpreter, dialling back to
the serving process over an authenticated local socket. `multiprocessing` cannot cross that
boundary: `spawn` hands the child the parent's `sys.path`, so the accelerator interpreter would
start up and then import the serving environment's CPU runtime. Everything above the handshake —
the command protocol, the memory ceiling, the batch retries, the CPU fallback — is the same for
both kinds of worker.

That socket is a Unix socket in a private directory, or a loopback port on Windows, which has no
filesystem permissions to lean on. One deadline covers connecting *and* authenticating, and a peer
that fails either is dropped so the wait can go on — so no other local process can take the slot
the worker needs, or hold a start-up open by connecting and going quiet.

Passage embedding runs in a disposable worker that is torn down after indexing, releasing VRAM or
unified memory. Before any real content reaches an accelerator, the worker loads the model and runs
a minimum-batch inference, and the parent checks that the vectors are the right width, finite, and
normalised. A backend that fails to load, silently falls back to a different execution provider,
returns unusable vectors, overruns the memory ceiling, or dies mid-run is terminated, and the
chunks it had not committed are re-embedded on CPU. Chunks are committed per file only after they
are fully embedded, so a worker crash can neither fail the run nor corrupt an existing index.

The ceiling is measured as host resident memory, which on unified memory covers the accelerator too.
A discrete GPU's VRAM is not visible to it, so exhausting a graphics card surfaces as a worker that
died rather than as a budget that was exceeded — both fall back to CPU, but only one names a number.

`CODE_INDEXING_EMBED_STRICT=1` refuses that fallback and raises `BACKEND_UNAVAILABLE` instead, for callers
who would rather fail than index at CPU speed without noticing. An `auto` selection that settles on
CPU is not a fallback and is unaffected.

Successful probes are cached under the cache directory, keyed by model artifact, execution
provider, ONNX Runtime version, OS/architecture, and device — so any of those moving invalidates
the record rather than vouching for a backend that no longer works. The driver version comes from
the installer's record, so a driver upgrade retires the verdict recorded under the old one. A cached
probe skips the inference but never the model load, which is what proves the provider still
initialises on this boot — so a driver change that breaks a provider outright still surfaces as a
failed load.

A backend that fails is not retried for the life of the process. Only successes are cached, so
without that a long-lived daemon would reload a dead accelerator onto the device before every
index run.

#### When the accelerator is worth starting

A prepared accelerator is not worth using for every run. Starting one means spawning an interpreter,
loading the model onto a device, and verifying it — which a re-index of a single edited file cannot
repay. So the first run that verifies a backend also measures it, through the ordinary embedding
path, at a ladder of batch sizes against a synthetic code-shaped corpus. It measures CPU the same
way at the same time, because the decision is a comparison and a comparison needs both sides. Both
results go in the probe cache, under the same key as the probe: model artifact, accelerator,
provider, runtime version, platform, device, and driver. Anything moving re-measures.

Two policies then have a cost each — `L_cpu + n/R_cpu` for staying on CPU and `L_accel + n/R_accel`
for using the accelerator — and they meet at

```text
crossover = (L_accel - L_cpu) / (1/R_cpu - 1/R_accel)
```

characters of candidate text. What the accelerator has to earn back is the difference between the
two loads, not its whole load: staying on CPU spawns a worker and loads a model too. On unified
memory the accelerator often loads *faster* — an M4 Pro maps the converted MLX weights in 370 ms
against 655 ms for the CPU ONNX graph — so the crossover is zero and it starts immediately.

Where there is a crossover, the run begins on CPU and switches to the accelerator on the request
that carries it past the threshold, counting the request in hand so the one group large enough to
justify the device is not itself sent to CPU. Deciding this late costs at most one model load on a
run that turns out to be large, and saves the same load on one that turns out to be small; the
pipeline streams, so the size of a run is not known until it is over. The run that does the
measuring also embeds at the size it just measured, rather than at the one its plan was built with
before the sweep existed.

An accelerator measured no faster than CPU has no crossover at all and is never started. That is
reported as no threshold rather than as a very large one — `crossover_characters` is `null` in both
`model status` and `IndexReport`, and the run says the backend measured no faster than CPU instead
of naming a size it stayed below.

A deferral is not a fallback. `fallback_count` stays at zero, no `embedding_fallback_reason` is set,
and the process is not pinned to CPU the way a real degradation pins it — the next run decides again
from its own size. `IndexReport` carries `embedded_characters`, `embedding_crossover_characters`, and
`embedding_selection_reason`, so a run that stayed on CPU because it was small reads differently from
one that fell back because something broke.

`CODE_INDEXING_EMBED_CROSSOVER=off` starts the accelerator on the first chunk, which is what every release
before this one did; a character count pins the threshold. `CODE_INDEXING_EMBED_CALIBRATE=0` declines the
measurement, leaving both the batch size and the crossover unmeasured.

`CODE_INDEXING_EMBED_STRICT=1` turns the crossover off too. Strict mode is for a caller who would rather
fail than quietly index at CPU speed, and a deferral is exactly that — quiet CPU indexing that no
degradation reports and that strict mode could not refuse, because nothing failed.

Measurement never installs, downloads, or changes anything: it embeds through a worker that is
already running and writes one JSON file under the cache directory. The sweep runs on the worker
the run will go on to use, so what it embedded, the retries it provoked, and the ceiling it walked
up to are put back afterwards: they are measurement, not work the run did, and an `IndexReport` that
counted them would describe a failure that never happened. The CPU reference is measured last of
all, once the accelerator's worker has been retired, so the two models are never resident against
the same ceiling at the same time.

`code-indexing-mcp model status` reports the whole resolution without loading or probing anything:

```console
$ code-indexing-mcp model status
{
  "accelerator_characters_per_second": null,
  "accelerator_environment": null,
  "accelerator_load_ms": null,
  "accelerator_prepared": null,
  "available_providers": ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"],
  "batch_calibration": "default",
  "batch_size": 1,
  "cpu_characters_per_second": null,
  "crossover_characters": null,
  "device": "cpu",
  "dimension": 768,
  "driver_version": "",
  "embedding_model": "jinaai/jina-embeddings-v2-base-code",
  "execution_provider": "CPUExecutionProvider",
  "fallback_reason": "no accelerator is prepared and eligible on this machine; reinstall with --accelerator to prepare one, or set CODE_INDEXING_EMBED_ACCELERATOR to force a backend this installation already offers",
  "precision": "float32",
  "probe_cache_state": "not-applicable",
  "recommended_override": null,
  "requested_accelerator": "auto",
  "resolved_accelerator": "cpu",
  "runtime_version": "1.27.0",
  "stability": "automatic",
  "strict": false
}
```

`accelerator_environment` names the interpreter passage embedding will run in when that is not
this process's own, and `accelerator_prepared` names what the installer prepared, whether or not
selection chose it.

The measured fields are null until a run has verified and measured an accelerator on this machine.
`crossover_characters` is null both before measurement and when the accelerator never overtakes CPU
at any size — the two are distinguished by `accelerator_characters_per_second` being present.
`recommended_override` names the one setting change the measurements argue for, when they argue for
one: `CODE_INDEXING_EMBED_ACCELERATOR=cpu` for an accelerator that lost, or `CODE_INDEXING_EMBED_MEMORY_MB` for a
batch size a ceiling overrun pinned down. It stays null the rest of the time rather than offering
advice nothing measured.

If installation reports a fallback, rerun the explicit installer command to see the build/probe
reason, then use `code-indexing-mcp model status` to distinguish the requested backend, prepared
environment, resolved provider, probe state, and fallback reason. Native provider failures never
trigger a package install or driver change in the server; the installer removes a failed
environment before CPU is reported.

`IndexReport` carries `embedding_backend`, `embedding_fallback_reason`, and `fallback_count`, so a
run that started on an accelerator and finished on CPU says so rather than merely being slow.
`embedding_backend` is where the work actually happened, which is not always what selection named: a
run below the crossover reports `cpu` with an `embedding_selection_reason` and no fallback.

The embedding model, tokenizer, pooling, normalisation, and vector dimension are the same on every
backend, and execution provider and precision are diagnostic metadata rather than part of index
compatibility — switching backends never requires a reindex.

### Token-bounded chunks

Sequence length, not character count, drives embedding memory: attention is quadratic in tokens.
The same 4,096 characters cost wildly different amounts depending on how densely they tokenize —
ordinary source is ~984 tokens, a minified line ~2,157 — and embedding the latter as one sequence
adds ~1,172 MiB of resident memory against ~266 MiB for the same characters split into windows.

Every chunk is therefore windowed to at most `CODE_INDEXING_EMBED_MAX_TOKENS` tokens with
`CODE_INDEXING_EMBED_OVERLAP_TOKENS` of overlap before it reaches the model, and each window is stored as
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

`CODE_INDEXING_EMBED_BATCH_SIZE` resolves to 1 on CPU. Measured with
`scripts/benchmark_index_memory.py` on Apple Silicon macOS against a 1.0 MiB, 6,330-chunk
dense-Python corpus at `CODE_INDEXING_EMBED_MEMORY_MB=2048`:

| `CODE_INDEXING_EMBED_BATCH_SIZE` | Wall clock | Chunks/s | Peak worker RSS |
| ------------------------- | ---------- | -------- | --------------- |
| 1 (default)               | 147.0 s    | 44.8     | 1,415 MiB       |
| 2                         | 136.2 s    | 48.7     | 1,419 MiB       |
| 4                         | 130.4 s    | 50.9     | 1,427 MiB       |
| 8                         | 126.7 s    | 52.5     | 1,451 MiB       |

Batch size 8 buys 17% throughput for 36 MiB more resident memory — not enough to justify spending
headroom that the worst-case file shape already needs. Embedding dominates: 141 s of the 147 s at
batch size 1. Plan for roughly **45 chunks per second**, and remember that in the default lazy mode
the first `search_code` call waits for that work. On a large repository prefer
`CODE_INDEXING_INDEX_MODE=eager` (index during tool listing and monitor later changes) or
`CODE_INDEXING_INDEX_MODE=manual` with an explicit `code-indexing-mcp index`, so no query blocks on
a cold index.

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

`CODE_INDEXING_INDEX_EXECUTION=in-process` is a temporary diagnostic rollback. It does not enforce the
worker ceiling. `CODE_INDEXING_VECTOR_INDEX=hnsw` opts into approximate vector indexing; exact search is
the safer default.

All stdio adapters use the per-user daemon by default. Administrative commands are:

```bash
uv run code-indexing-mcp daemon status
uv run code-indexing-mcp daemon restart
uv run code-indexing-mcp daemon stop
```

Set `CODE_INDEXING_BROKER=off` or run `serve --direct` to bypass it. The daemon authenticates over a
current-user-only local socket, starts under leader election, and exits after five idle minutes.
The socket lives under `XDG_RUNTIME_DIR` when set and the platform temporary directory otherwise;
the containing directory must be a real directory owned by the current user, or startup fails
rather than binding somewhere another user controls. Startup output goes to `daemon.log` in the
data directory.

The daemon needs Unix domain sockets. Where they are unavailable — currently Windows — the default
`CODE_INDEXING_BROKER=auto` serves directly and logs a warning; an explicit `CODE_INDEXING_BROKER=on` fails with
`INVALID_CONFIGURATION` instead of being silently downgraded.

Storage schema v2 keeps a registry plus one LanceDB partition per project. On first upgrade from
v1, the old global store is moved to a timestamped `lancedb-v1-backup-*` directory and projects are
rebuilt lazily from source. Old chunk rows are never copied, which repairs duplicate chunk IDs.

With `CODE_INDEXING_OFFLINE=1`, Code Indexing MCP will not download a missing model and returns
`MODEL_UNAVAILABLE` instead. Source code, embeddings, and search queries remain local; there is
no telemetry.

## Development

```bash
uv sync --all-groups --extra cpu --locked
uv run pytest
uv run pytest --cov=code_indexing_mcp
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

A version bump in `src/code_indexing_mcp/__init__.py` must ship with a regenerated `uv.lock`
(`uv lock`), or every user's `update` fails at the locked sync; CI's `--locked` sync gates it.

To exercise the real model integration, provide a persistent cache directory and opt in:

```bash
CODE_INDEXING_MODEL_TEST_CACHE=/path/to/cache uv run pytest -m model
```

Dedicated MLX/WebGPU/MIGraphX runners can exercise the correctness and ≥1,000-chunk performance
gates against an installer-created accelerator record:

```bash
CODE_INDEXING_MODEL_TEST_CACHE=/path/to/cache \
CODE_INDEXING_ACCEL_ENV=/path/to/accelerator.json \
CODE_INDEXING_TEST_ACCELERATOR=mlx \
  uv run pytest -m accelerator
```

The gate requires row cosine similarity of at least 0.999, search top-k overlap of at least 99%,
and an end-to-end forced-index speedup of at least 1.25× before a backend can be considered for
automatic promotion.

MLX's own unit tests need MLX, which the serving environment does not have. They run in the `mlx`
extra's environment, and check the forward pass against an independent NumPy implementation of the
same graph:

```bash
uv run --extra mlx pytest tests/test_mlx_backend.py
```

The project intentionally excludes HTTP transports, dependency/call graphs, cross-reference
resolution, and custom embedding profiles.

## License

This project is licensed under the [MIT License](LICENSE).
