# Textual TUI Installer — Design

Date: 2026-07-31
Status: Approved design, pre-implementation
Branch: `feat/textual-installer-tui` (from `main`)

## Context

The current installer is a 1,678-line stdlib-only `install.py`, driven by `install.sh`
(curl-pipe one-liner). It runs a fixed linear sequence — clone/update, `uv sync`,
accelerator plan/build/probe, a numbered harness menu via `input()`, config merge,
skills symlinking — and offers exactly four customization flags: `--install-dir`,
`--repo-url`, `--accelerator`, `--harnesses`.

The server, meanwhile, has grown a large runtime configuration surface (18 `INCODE_*`
environment variables in `src/incode_mcp/settings.py` and `application.py`), none of
which the installer can set. Because MCP servers are spawned by the harness with a fixed
command, the only way those variables reach the server is through the harness config's
per-server environment mechanism. Today the installer writes `"env": {}` for Claude Code
and nothing at all for the other five harnesses, so initial customization means hand-editing
every harness config after installation.

This design replaces the install experience with a Textual TUI wizard that covers the full
configuration surface, while keeping the curl-pipe one-liner, the scripted CI path, and every
behavioral guarantee of the existing installer.

## Decisions (from brainstorming)

1. **Delivery**: the `install.sh` one-liner stays the entry point. It bootstraps (clone/update,
   `uv sync` including the new `tui` extra), then launches the Textual TUI from the synced
   environment.
2. **Settings home**: customized settings are written into each harness's MCP server entry
   environment block. No server-side changes; `settings.py` is untouched.
3. **Scope**: install/update plus a reconfigure mode that prefills the wizard from existing
   harness config environment blocks.
4. **CI path**: existing flags keep working unchanged; a repeatable `--set INCODE_FOO=bar`
   flag carries the new settings non-interactively. The TUI launches only on an interactive
   TTY (auto-disabled for unusable `TERM`); `--tui` / `--no-tui` force the mode.
5. **Reconfigure entry**: the TUI ships inside the installed package as
   `code-indexing-mcp configure` (Textual lazily imported). Works offline; no git/uv needed.
6. **Architecture**: all install logic moves from `install.py` into a new
   `src/incode_mcp/installer/` subpackage (package-embedded). `install.py` becomes a thin
   bootstrap: clone/update → sync → delegate. One orchestrator, one source of truth;
   reconfigure can do everything install can, including accelerator changes.

## Goals

- A guided, user-friendly TUI covering installation, accelerator selection with live
  hardware detection, harness selection, and the full settings surface.
- Settings persisted into harness config environment blocks with correct per-harness key
  names, preserving unrelated user keys.
- Non-interactive parity: everything the TUI can do, `--set` + flags can do.
- `code-indexing-mcp configure` re-opens the wizard offline, prefilled from current config.
- The existing installer test suite moves with the code and stays green (import-only changes),
  proving the move is lossless.

## Non-goals

- No server-side configuration file or changes to how the server reads settings.
- No uninstall flow, no doctor/status console.
- Legacy/test-only variables stay out of the UI: `INCODE_AUTO_INDEX`,
  `INCODE_INDEX_MEMORY_MB` (legacy alias), `INCODE_ACCEL_ENV`, `INCODE_MODEL_TEST_CACHE`,
  `INCODE_TEST_ACCELERATOR`.
- `install.sh` itself is unchanged.
- The plain-text `--no-tui` interactive path keeps today's simple prompts (numbered harness
  menu, accelerator flag). It does not gain per-setting prompts; customization there is via
  `--set` only.

## Architecture

### New subpackage `src/incode_mcp/installer/`

- `config_files.py` — JSONC/JSON/TOML comment-preserving merge machinery moved **verbatim**
  from `install.py`: JSONC scanner/parser helpers, `_merge_jsonc_text`,
  `merge_json_object_entry`, TOML table handling, `merge_codex_server`, `_atomic_write`,
  `_write_changed_configuration`, `_read_configuration`, `InstallerError`.
- `accelerator.py` — moved verbatim: `AcceleratorPlan`, `plan_accelerator` and its platform
  helpers, environment build (`sync_accelerator_environment`), `probe_accelerator`,
  record write/clear/reuse, `configure_accelerator`, lock fingerprinting,
  `runtime_record_path`.
- `harnesses.py` — moved verbatim: `HarnessChoice`/`HARNESS_CHOICES`,
  `configuration_path`, `configure_harness`, `parse_harness_selection`,
  `configure_selected_harnesses`, `skill_directory`, `install_skills`.
- `settings_spec.py` — **new**: declarative catalog of every exposed setting (see
  "Settings catalog" below). One source drives TUI form generation, `--set` validation,
  and the summary screen.
- `env_blocks.py` — **new**: read/merge/write the environment mapping inside a harness's
  MCP server entry. Per-harness key names (see "Persistence" below). Merges keys the wizard
  manages; preserves unrelated keys the user placed there. Codex: reads the existing `env`
  table via `tomllib` before the block is regenerated, re-emits managed + preserved keys.
- `orchestrator.py` — **new**: the post-sync pipeline as discrete steps with progress
  callbacks: accelerator prepare → harness configuration (with env blocks) → skills. Emits
  events (step started / log line / step finished / warning) consumed by both the TUI
  progress screen and the plain CLI printer. All accelerator failure paths keep the
  degrade-to-CPU-with-reason semantics.
- `cli.py` + `__main__.py` — **new**: non-interactive module CLI,
  `python -m incode_mcp.installer`, accepting `--accelerator`, `--harnesses`,
  `--set KEY=VALUE` (repeatable), `--offline`, `--tui`. Used by the bootstrap for CI and
  by `--no-tui` fallback; `--tui` launches the Textual app instead.
- `tui/` — **new**: the Textual app (`app.py`, `screens.py`, `settings_form.py`). Forms are
  generated from `settings_spec.py`, not hand-built per setting.

### Bootstrap `install.py` (shrinks to ~250 lines, stdlib-only)

Keeps: argument parsing (existing flags + `--set` repeatable + `--tui`/`--no-tui`),
`clone_or_update_repository` and its helpers (must run before the package exists on fresh
machines), `sync_environment` (extended to sync `--extra tui` alongside `--extra cpu`),
TTY/`TERM` mode detection, and delegation:

- Interactive TTY, TUI allowed → re-exec `.venv/bin/python -m incode_mcp.installer --tui`
  (forwarding `--install-dir`, `--accelerator` if given, `--set` values as prefill).
- Otherwise → `.venv/bin/python -m incode_mcp.installer` with the parsed flags; behaves as
  today's installer (plus `--set`), including the plain-text harness menu on a TTY.
- TUI exits non-zero → bootstrap prints the error and advises re-running with `--no-tui`.

`install.sh` is untouched.

### Package entry point

`src/incode_mcp/cli.py` gains a `configure` subcommand. It lazily imports
`incode_mcp.installer.tui`, so `serve` and every other command never import Textual. It
opens the wizard in reconfigure mode against the existing installation: no clone, no sync.
`configure --set KEY=VALUE` applies scripted changes without opening the UI.
If Textual is missing (a dev environment synced without the `tui` extra), the error says
exactly how to get it (`uv sync --extra tui`).

### Dependency changes

- New optional extra in `pyproject.toml`: `tui = ["textual>=8.2,<9"]`. The exact version is
  pinned by the regenerated `uv.lock`, which is committed. The extra is **not** added to
  the `[tool.uv]` conflicts list — it must combine with `cpu`.
- Bootstrap sync becomes `uv sync --locked --extra cpu --extra tui`. The serving environment
  permanently carries Textual; it is lazily imported, so `serve` never pays the import cost.

## Entry-point flow summary

| Invocation | Path |
| --- | --- |
| `curl … install.sh \| sh` on a TTY | bootstrap: clone/update → sync → TUI wizard (fresh or update mode) |
| Same, non-TTY | bootstrap: clone/update → sync → module CLI (flags/`--set` only) |
| `install.sh --no-tui` | bootstrap → module CLI with plain-text prompts on a TTY |
| `code-indexing-mcp configure` | TUI wizard, reconfigure mode, offline |
| `code-indexing-mcp configure --set …` | module CLI applies settings + rewrites harness env blocks |

## TUI design

One Textual app; a linear wizard with Back/Next, and a summary screen that can jump back to
any section. Reconfigure mode is the same wizard, prefilled, with the Location screen
replaced by a fixed display of the existing install.

1. **Welcome** — what was detected (fresh machine / existing checkout to update / configured
   install to reconfigure) and what the wizard will do.
2. **Location** — install dir + repo URL under a collapsed "Advanced" section, defaults
   filled. Absent in reconfigure mode.
3. **Accelerator** — radio list (`auto`, `cpu`, `cuda`, `mlx`, `webgpu`, `migraphx`,
   `coreml`) annotated with live detection: NVIDIA driver version and device, ROCm version,
   macOS version, platform wheel support — the same facts `plan_accelerator` evaluates,
   computed with the same helpers before the user chooses. `auto` marked recommended; a note
   explains the build-and-probe step and the CPU fallback.
4. **Harnesses** — checkbox list of the six harnesses, each showing its resolved config
   path, whether an entry already exists, and whether it supports skills.
5. **Indexing** — index mode, wait seconds, memory MB, vector index, worker execution,
   broker, data dir, cache dir, offline.
6. **Embedding** — batch size, max tokens, overlap tokens, threads, CPU arena, crossover,
   calibrate, strict, and the expert `INCODE_EMBED_ACCELERATOR` override.
7. **Summary** — every non-default choice; the exact files that will be written (config
   paths, accelerator record); a disk-cost note when an accelerator environment will be
   built (multi-GB). Confirm runs the pipeline; any section can be jumped back to.
8. **Progress** — step list with live log tail. `uv sync`, environment builds, and the probe
   run in Textual thread workers streaming output; the 15-minute probe timeout and
   `INCODE_OFFLINE` behavior carry over. Cancel stops cleanly between steps.
9. **Done** — what changed, "restart your clients", and the `code-indexing-mcp configure`
  pointer. Accelerator fallbacks appear here as warnings with their reasons.

## Settings catalog (`settings_spec.py`)

Each entry: env name, group, label, help text, type, default, validation. Types:
`bool`, `int` (min/max), `choice`, `path`, plus the two tri-state values
(`auto|int`, `auto|off|int`). Validation mirrors `settings.py` exactly.

| Group | Variable | Type | Default | Validation |
| --- | --- | --- | --- | --- |
| Indexing | `INCODE_INDEX_MODE` | choice | `lazy` | lazy, eager, manual |
| Indexing | `INCODE_INDEX_WAIT_SECONDS` | int | 300 | 0–86400 |
| Indexing | `INCODE_EMBED_MEMORY_MB` | int | dynamic (25% RAM, clamped 1024–2048) | 1024–1048576 |
| Indexing | `INCODE_VECTOR_INDEX` | choice | `exact` | exact, hnsw |
| Indexing | `INCODE_INDEX_EXECUTION` | choice | `worker` | worker, in-process |
| Indexing | `INCODE_BROKER` | choice | `auto` | auto, on, off |
| Indexing | `INCODE_DATA_DIR` | path | platformdirs user data | — |
| Indexing | `INCODE_CACHE_DIR` | path | platformdirs user cache | — |
| Indexing | `INCODE_OFFLINE` | bool | off | — |
| Embedding | `INCODE_EMBED_BATCH_SIZE` | auto\|int | auto | 1–256 |
| Embedding | `INCODE_EMBED_MAX_TOKENS` | int | 1024 | 64–8192 |
| Embedding | `INCODE_EMBED_OVERLAP_TOKENS` | int | 64 | 0–4096 |
| Embedding | `INCODE_EMBED_THREADS` | int | min(2, cpu_count) | 1–64 |
| Embedding | `INCODE_EMBED_CPU_ARENA` | bool | off | — |
| Embedding | `INCODE_EMBED_CROSSOVER` | auto\|off\|int | auto | 0–1073741824 |
| Embedding | `INCODE_EMBED_CALIBRATE` | bool | on | — |
| Embedding | `INCODE_EMBED_STRICT` | bool | off | — |
| Embedding | `INCODE_EMBED_ACCELERATOR` | choice | auto | auto, cpu, cuda, mlx, webgpu, migraphx, coreml |

Rules:

- **Defaults are omitted.** Only values the user explicitly changes are written to env
  blocks. Future default changes then flow through, and configs stay minimal.
- **The wizard's accelerator selection is not written to env blocks.** It controls which
  environment gets built; the runtime stays on `auto` and picks the prepared backend.
  `INCODE_EMBED_ACCELERATOR` remains an expert override on the Embedding screen.
- Dynamic defaults (memory, threads) display their resolved value at render time.

## Persistence: harness environment blocks

Per-harness environment key names (verified against each harness's docs/schema):

| Harness | File | Entry shape | Env key |
| --- | --- | --- | --- |
| codex | `~/.codex/config.toml` | `[mcp_servers.code-indexing-mcp]` | `env = { K = "V" }` |
| claude-code | `~/.claude.json` | `mcpServers` entry | `env` |
| kimi-code | `~/.kimi-code/mcp.json` | `mcpServers` entry | `env` |
| claude-desktop | platform-specific JSON | `mcpServers` entry | `env` |
| opencode | `opencode.json` | `mcp` entry (`type: local`) | `environment` |
| kilocode | `kilo.json`/`kilo.jsonc` | `mcp` entry (`type: local`) | `environment` (opencode-format schema; re-confirm against the kilo schema during implementation) |

Merge semantics:

- Writing updates only keys the wizard manages (the catalog's `INCODE_*` names); unrelated
  keys already in the entry's env mapping survive. Managed keys removed by the user (reset to
  default) are deleted from the block; an emptied block is written as `{}` (omitted for
  Codex, where the block regenerates as `command`/`args` plus `env` only when non-empty).
- Codex: the server table is regenerated as today, extended to emit `env` when non-empty;
  pre-existing env values are read via `tomllib` first so unmanaged keys survive.
- Reconfigure prefill: reads the env blocks of all currently configured harnesses; on
  disagreement the value from the harness earliest in the fixed `HARNESS_CHOICES` order
  wins, and the disagreement is listed on the summary screen.
- `--set` validates against the same catalog: unknown key, bad choice, or out-of-range
  integer fails fast with the same message the TUI shows.

## Error handling

- Failure vocabulary unchanged: `InstallerError` with actionable messages; accelerator
  failures keep degrade-to-CPU-with-reason semantics (warning panel in the TUI, same stderr
  report and exit code in the CLI).
- Nothing writes until the Summary screen's Confirm. Config file writes keep the existing
  backup + atomic-replace behavior, so a cancelled wizard (Esc/Ctrl+C, with confirm) leaves
  no half-written state. A cancel after the accelerator environment was built still leaves a
  valid CPU-fallback state (record cleared on failure, as today).
- Preflight checks in the wizard: git present, uv present, writable install dir, minimum
  terminal size — each with a fix-it message, not a traceback.
- Textual is never on the `serve` path; a TUI bug cannot break the installed server.

## Testing

- **Move-proof**: the existing installer tests (`tests/test_installer.py` and any others
  importing `install.py`) update imports to `incode_mcp.installer.*` with behavior
  assertions unchanged; green suite ⇒ lossless move.
- **New unit tests**: catalog validation for every type/range/choice; env-block
  read/merge/write for all six harnesses (unknown-key preservation, Codex TOML round-trip,
  emptied-block behavior); `--set` parsing and errors; orchestrator step sequencing with
  subprocess mocked; reconfigure prefill and disagreement reporting.
- **TUI tests** with Textual's headless pilot (`App.run_test()`): screen navigation,
  validation error display, summary content, reconfigure prefill, cancel between steps.
  The orchestrator is faked; no real subprocesses.
- **Bootstrap tests**: arg parsing, TTY/`TERM` mode selection, delegation command
  construction.
- No network, no real `uv sync`, no real probe in any test — the current suite's mocking
  discipline carries over.

## Documentation

- README's Install section updated: TUI flow, `configure` subcommand, `--set`, `--no-tui`,
  and a note that settings now land in harness config env blocks.
- `AGENTS.md` files: none exist in this repository today; if any are added covering the
  installer, they must describe the new layout.
