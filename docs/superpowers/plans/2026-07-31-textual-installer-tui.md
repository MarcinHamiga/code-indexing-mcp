# Textual TUI Installer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the linear `install.py` experience with a Textual TUI wizard covering installation, accelerator selection, harness setup, and the full 18-setting `INCODE_*` surface, while keeping the curl-pipe one-liner and a scripted CI path.

**Architecture:** All install logic moves from `install.py` into a new `src/incode_mcp/installer/` subpackage. `install.py` becomes a stdlib-only bootstrap (clone/update → `uv sync --extra cpu --extra tui` → delegate to `python -m incode_mcp.installer`). The TUI ships in the package and is reachable as `code-indexing-mcp configure` for offline reconfiguration. Settings persist into per-harness MCP config env blocks (`env`, or `environment` for OpenCode/KiloCode).

**Tech Stack:** Python 3.12/3.13, Textual `>=8.2,<9` (lazily imported), uv-locked environment, pytest + pytest-asyncio (strict marker mode), ruff, mypy strict.

**Spec:** `docs/superpowers/specs/2026-07-31-textual-installer-design.md` (approved). One deviation from the spec's draft text: the Textual pin is `>=8.2,<9` (8.2.8 is current on PyPI), not `>=3,<4`. A second deliberate refinement: the Progress screen streams *step events* (with captured subprocess output on failure), not raw subprocess bytes — this matches today's installer output behavior exactly.

## Global Constraints

- Python `>=3.12,<3.14`; ruff line-length 100; ruff lint `E,F,I,UP,B,SIM,RUF`; mypy `strict = true` with `packages = ["incode_mcp"]` — the new package is fully type-checked.
- `install.py` must stay stdlib-only and self-contained: `install.sh` downloads it into a temp directory and runs it before any virtual environment exists. It must never import `incode_mcp` or Textual.
- Textual is imported lazily (inside functions), never at module import time on the `serve` path. `python -c "import incode_mcp.cli"` must not import Textual.
- The `tui` extra is NOT added to the `[tool.uv]` conflicts list; it must combine with `cpu`.
- Only non-default settings are written to harness env blocks. The wizard's accelerator selection is never written as `INCODE_EMBED_ACCELERATOR`.
- Per-harness env keys: `env` for codex (TOML table), claude-code, kimi-code, claude-desktop; `environment` for opencode and kilocode.
- Accelerator failures always degrade to CPU with the reason attached — never fail the install.
- No network, no real `uv sync`, no real probe in tests. Async TUI tests use `@pytest.mark.asyncio` (pytest-asyncio runs in strict mode).
- Commit style follows the repo: `feat:`, `fix:`, `docs:`, `refactor:` imperative subject lines.
- Branch: `feat/textual-installer-tui` (already created from `main`; the spec is committed there).

---

### Task 1: Move install logic into the `incode_mcp.installer` package

Mechanical, behavior-preserving move. `install.py` is left completely untouched in this task (its copies die in Task 13); the test suite is re-pointed at the package so a green run proves the move is lossless.

**Files:**
- Create: `src/incode_mcp/installer/__init__.py`
- Create: `src/incode_mcp/installer/config_files.py`
- Create: `src/incode_mcp/installer/accelerator.py`
- Create: `src/incode_mcp/installer/harnesses.py`
- Modify: `tests/test_installer.py` (imports + monkeypatch targets only)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `incode_mcp.installer.config_files` (`InstallerError`, `SERVER_NAME`, `merge_json_object_entry`, `merge_codex_server`, `_jsonc_as_json`), `incode_mcp.installer.accelerator` (`AcceleratorPlan`, `ACCELERATOR_EXTRAS`, `ACCELERATOR_ENVIRONMENT_DIRECTORY`, `ACCELERATOR_CHOICES`, `plan_accelerator`, `configure_accelerator`, `sync_accelerator_environment`, `probe_accelerator`, `write_accelerator_record`, `clear_accelerator_record`, `reusable_accelerator_environment`, `accelerator_lock_fingerprint`, `accelerator_record_path`, `runtime_record_path`, `interpreter_version`, `server_executable`, `environment_python`, `_run_command`, `_nvidia_smi_report`, `_rocm_report`), `incode_mcp.installer.harnesses` (`HarnessChoice`, `HARNESS_CHOICES`, `parse_harness_selection`, `configuration_path`, `configure_harness`, `configure_selected_harnesses`, `skill_directory`, `install_skills`, `harness_label`).

- [ ] **Step 1: Create the package skeleton**

`src/incode_mcp/installer/__init__.py`:

```python
"""Install-time logic for Code Indexing MCP: configuration, accelerators, harnesses.

Everything in this package runs inside the synced installation environment.
The curl-pipeable bootstrap at the repository root (`install.py`) stays
stdlib-only and delegates here after `uv sync`.
"""
```

- [ ] **Step 2: Create `config_files.py` from `install.py`**

Copy these regions of `install.py` **verbatim** (same order):
- Module docstring replaced with `"""Comment-preserving JSON/JSONC/TOML configuration merging for harness setup."""`
- Imports: `json`, `os`, `re`, `shutil`, `tempfile`, `tomllib`, `from collections.abc import Mapping` (drop `Callable`), `from pathlib import Path`, `from typing import Any`
- `SERVER_NAME = "code-indexing-mcp"` (line 22)
- `InstallerError` (lines 94–96)
- Everything from `_skip_jsonc_trivia` through `merge_codex_server` (lines 98–514)

- [ ] **Step 3: Create `accelerator.py` from `install.py`**

Copy verbatim:
- Module docstring: `"""Accelerator planning, environment building, probing, and recording."""`
- Imports: `hashlib`, `json`, `os`, `platform`, `re`, `shutil`, `subprocess`, `sys`, `time`, `from collections.abc import Callable, Mapping`, `from pathlib import Path`, `from typing import Any, NamedTuple`, plus `from .config_files import InstallerError`
- Constants `SERVING_EXTRA` is NOT copied (bootstrap-only). Copy `ACCELERATOR_EXTRAS`, `ACCELERATOR_CHOICES`, `ACCELERATOR_ENVIRONMENT_DIRECTORY`, `ACCELERATOR_RECORD_SCHEMA_VERSION`, `PROBE_TIMEOUT_SECONDS`, `MINIMUM_NVIDIA_DRIVER`, `CUDA_PLATFORMS`, `WEBGPU_PLATFORMS`, `MINIMUM_WEBGPU_MACOS`, `MLX_PLATFORMS`, `MINIMUM_MLX_MACOS`, `MIGRAPHX_PLATFORM`, `MIGRAPHX_PYTHON_VERSION`, `MIGRAPHX_ROCM_VERSION` (lines 31–76)
- `_run_command` (lines 667–690)
- `server_executable`, `environment_python`, `_uv_executable` (lines 748–775)
- Everything from `AcceleratorPlan` through `configure_accelerator` (lines 796–1406)

- [ ] **Step 4: Create `harnesses.py` from `install.py`**

Copy verbatim:
- Module docstring: `"""Harness detection, configuration merging, and bundled-skill installation."""`
- Imports: `from collections.abc import Mapping`, `from pathlib import Path`, `from typing import Any, NamedTuple`, `import os`, `import sys`, plus `from .config_files import InstallerError, SERVER_NAME, merge_codex_server, merge_json_object_entry`
- `HarnessChoice`, `HARNESS_CHOICES` (lines 79–91)
- `parse_harness_selection` (lines 517–538)
- `_configured_directory`, `_preferred_json_config`, `configuration_path`, `configure_harness` (lines 541–664)
- `configure_selected_harnesses` (lines 1409–1434)
- `skill_directory` through `install_skills` (lines 1437–1558)
- `_harness_label` (lines 1604–1605), renamed to public `harness_label`

- [ ] **Step 5: Re-point `tests/test_installer.py` at the package**

Replace the header (lines 1–29, the `load_installer` helper and `installer = load_installer()` module-level call) with:

```python
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from incode_mcp.installer import accelerator, config_files, harnesses
from incode_mcp.installer.accelerator import (
    ACCELERATOR_ENVIRONMENT_DIRECTORY,
    ACCELERATOR_EXTRAS,
    AcceleratorPlan,
    accelerator_lock_fingerprint,
    configure_accelerator,
    plan_accelerator,
    probe_accelerator,
    sync_accelerator_environment,
    write_accelerator_record,
)
from incode_mcp.installer.config_files import (
    InstallerError,
    merge_codex_server,
    merge_json_object_entry,
)
from incode_mcp.installer.harnesses import (
    HARNESS_CHOICES,
    configuration_path,
    configure_harness,
    configure_selected_harnesses,
    install_skills,
    parse_harness_selection,
    skill_directory,
)

INSTALLER_PATH = Path(__file__).parents[1] / "install.py"
SHELL_INSTALLER_PATH = Path(__file__).parents[1] / "install.sh"

# The installer stringifies Path objects with the native separator, so expected
# values must go through the same conversion to stay correct on Windows.
SERVER_BINARY = Path("/opt/ci-mcp")
SERVER_COMMAND = str(SERVER_BINARY)


def load_installer() -> ModuleType:
    """Load the stdlib-only bootstrap by path; only its own surface is tested here."""
    assert INSTALLER_PATH.exists(), "install.py does not exist"
    spec = importlib.util.spec_from_file_location("code_indexing_mcp_installer", INSTALLER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Then transform the 60 tests that use moved names, and leave these six bootstrap-surface tests completely unchanged (they keep using `installer = load_installer()` and `installer.<name>`):
- `test_repository_is_cloned_then_fast_forwarded_on_update` (line 479)
- `test_repository_update_rejects_non_repo_dirty_and_mismatched_targets` (line 496)
- `test_sync_environment_runs_locked_sync_and_finds_server` (line 519)
- `test_main_runs_noninteractive_install_and_reports_harness_failures` (line 601)
- `test_main_prompts_for_harnesses_when_option_is_omitted` (line 661)
- `test_main_reports_actionable_installer_error` (line 700)

Transformation rules for all other tests:
1. Delete the line `installer = load_installer()` inside the test body.
2. Replace `installer.<name>` with the bare imported `<name>` (the imports above cover every used name except `main`, `clone_or_update_repository`, `sync_environment`, `server_executable`, `shutil`, which appear only in the six kept tests — verify with grep after editing).
3. Replace `monkeypatch.setattr(installer, "<name>", value)` with `monkeypatch.setattr(<module>, "<name>", value)` where `<module>` is the imported module object holding that name:
   - `configure_harness`, `configure_selected_harnesses` → `harnesses`
   - `runtime_record_path`, `interpreter_version`, `probe_accelerator`, `sync_accelerator_environment`, `configure_accelerator` internals → `accelerator`

- [ ] **Step 6: Run the suite, ruff, and mypy**

```bash
uv run pytest tests/test_installer.py -q
uv run ruff check src/incode_mcp/installer tests/test_installer.py
uv run mypy
```

Expected: all tests pass as before. Ruff may flag import order — run `uv run ruff check --fix` for `I` rules. **mypy may report new errors: the moved code was never type-checked as part of `incode_mcp` before.** Fix any fallout minimally (typical: `Callable[[], str | None]` defaults, `subprocess` env typing). Do not weaken `strict`.

- [ ] **Step 7: Commit**

```bash
git add src/incode_mcp/installer tests/test_installer.py
git commit -m "refactor: move install logic into the incode_mcp.installer package"
```

---

### Task 2: Settings catalog (`settings_spec.py`)

The single source of truth for the 18 manageable settings: drives TUI form generation, `--set` validation, and summary display.

**Files:**
- Create: `src/incode_mcp/installer/settings_spec.py`
- Test: `tests/test_installer_settings_spec.py`

**Interfaces:**
- Consumes: nothing from Task 1 (stdlib + `psutil` + `platformdirs`, both project deps).
- Produces: `Setting` (frozen dataclass), `SETTINGS: tuple[Setting, ...]`, `BY_NAME: dict[str, Setting]`, `default_value(setting) -> str`, `validate(setting, raw) -> str | None`, `normalize(setting, raw) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_installer_settings_spec.py`:

```python
"""Tests for the installer's declarative settings catalog."""

import pytest

from incode_mcp.installer.settings_spec import (
    BY_NAME,
    SETTINGS,
    default_value,
    normalize,
    validate,
)


def test_catalog_covers_exactly_the_documented_settings() -> None:
    assert {setting.name for setting in SETTINGS} == {
        "INCODE_INDEX_MODE",
        "INCODE_INDEX_WAIT_SECONDS",
        "INCODE_EMBED_MEMORY_MB",
        "INCODE_VECTOR_INDEX",
        "INCODE_INDEX_EXECUTION",
        "INCODE_BROKER",
        "INCODE_DATA_DIR",
        "INCODE_CACHE_DIR",
        "INCODE_OFFLINE",
        "INCODE_EMBED_BATCH_SIZE",
        "INCODE_EMBED_MAX_TOKENS",
        "INCODE_EMBED_OVERLAP_TOKENS",
        "INCODE_EMBED_THREADS",
        "INCODE_EMBED_CPU_ARENA",
        "INCODE_EMBED_CROSSOVER",
        "INCODE_EMBED_CALIBRATE",
        "INCODE_EMBED_STRICT",
        "INCODE_EMBED_ACCELERATOR",
    }


def test_every_setting_has_display_metadata_and_a_group() -> None:
    for setting in SETTINGS:
        assert setting.group in {"Indexing", "Embedding"}
        assert setting.label and setting.help


@pytest.mark.parametrize(
    ("name", "raw", "ok"),
    [
        ("INCODE_INDEX_MODE", "eager", True),
        ("INCODE_INDEX_MODE", "sometimes", False),
        ("INCODE_INDEX_WAIT_SECONDS", "300", True),
        ("INCODE_INDEX_WAIT_SECONDS", "86401", False),
        ("INCODE_INDEX_WAIT_SECONDS", "-1", False),
        ("INCODE_EMBED_MEMORY_MB", "2048", True),
        ("INCODE_EMBED_MEMORY_MB", "512", False),
        ("INCODE_EMBED_BATCH_SIZE", "auto", True),
        ("INCODE_EMBED_BATCH_SIZE", "256", True),
        ("INCODE_EMBED_BATCH_SIZE", "0", False),
        ("INCODE_EMBED_CROSSOVER", "off", True),
        ("INCODE_EMBED_CROSSOVER", "auto", True),
        ("INCODE_EMBED_CROSSOVER", "100000", True),
        ("INCODE_EMBED_CROSSOVER", "banana", False),
        ("INCODE_EMBED_MAX_TOKENS", "8192", True),
        ("INCODE_EMBED_MAX_TOKENS", "63", False),
        ("INCODE_EMBED_OVERLAP_TOKENS", "0", True),
        ("INCODE_EMBED_THREADS", "64", True),
        ("INCODE_EMBED_THREADS", "65", False),
        ("INCODE_OFFLINE", "yes", True),
        ("INCODE_OFFLINE", "maybe", False),
        ("INCODE_DATA_DIR", "/tmp/data", True),
        ("INCODE_DATA_DIR", "", False),
        ("INCODE_EMBED_ACCELERATOR", "coreml", True),
        ("INCODE_EMBED_ACCELERATOR", "tpu", False),
    ],
)
def test_validate(name: str, raw: str, ok: bool) -> None:
    assert (validate(BY_NAME[name], raw) is None) is ok


def test_validate_unknown_names_are_rejected_by_lookup() -> None:
    assert "INCODE_FROBNICATE" not in BY_NAME


@pytest.mark.parametrize(
    ("name", "raw", "stored"),
    [
        ("INCODE_OFFLINE", "YES", "1"),
        ("INCODE_OFFLINE", "off", "0"),
        ("INCODE_INDEX_MODE", "EAGER", "eager"),
        ("INCODE_EMBED_BATCH_SIZE", "AUTO", "auto"),
        ("INCODE_EMBED_BATCH_SIZE", "8", "8"),
        ("INCODE_DATA_DIR", "/data", "/data"),
    ],
)
def test_normalize(name: str, raw: str, stored: str) -> None:
    assert normalize(BY_NAME[name], raw) == stored


def test_dynamic_defaults_resolve_to_valid_values() -> None:
    assert validate(BY_NAME["INCODE_EMBED_MEMORY_MB"], default_value(BY_NAME["INCODE_EMBED_MEMORY_MB"])) is None
    assert validate(BY_NAME["INCODE_EMBED_THREADS"], default_value(BY_NAME["INCODE_EMBED_THREADS"])) is None
    assert default_value(BY_NAME["INCODE_DATA_DIR"]).endswith("incode")
```

Run: `uv run pytest tests/test_installer_settings_spec.py -q` — expect FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement `settings_spec.py`**

```python
"""Declarative catalog of the runtime settings the installer can manage.

One source drives the Textual wizard's forms, ``--set`` validation, and the
summary screen. Validation mirrors ``incode_mcp.settings`` exactly; the server
keeps reading only real environment variables.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import psutil
from platformdirs import user_cache_path, user_data_path

SettingType = Literal["bool", "int", "choice", "path", "auto_int", "auto_off_int"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Setting:
    name: str  # environment variable name
    group: str  # "Indexing" | "Embedding"
    label: str
    help: str
    type: SettingType
    default: str  # static default as a string, "" when dynamic_default is set
    choices: tuple[str, ...] = ()
    minimum: int = 0
    maximum: int = 0
    dynamic_default: Callable[[], str] | None = None


def _default_memory_mb() -> str:
    total = psutil.virtual_memory().total
    return str(max(1024, min(2048, int(total * 0.25) // (1024 * 1024))))


def _default_threads() -> str:
    return str(max(1, min(2, os.cpu_count() or 1)))


SETTINGS: tuple[Setting, ...] = (
    Setting(
        "INCODE_INDEX_MODE", "Indexing", "Index mode",
        "When projects get indexed: lazy on first use, eager at startup, manual only.",
        "choice", "lazy", choices=("lazy", "eager", "manual"),
    ),
    Setting(
        "INCODE_INDEX_WAIT_SECONDS", "Indexing", "Index wait (seconds)",
        "How long a startup index waits out a competing job before failing; 0 disables waiting.",
        "int", "300", minimum=0, maximum=24 * 60 * 60,
    ),
    Setting(
        "INCODE_EMBED_MEMORY_MB", "Indexing", "Indexing memory (MB)",
        "Ceiling for the indexing worker. The default is 25% of RAM clamped to 1024-2048.",
        "int", "", minimum=1024, maximum=1024 * 1024, dynamic_default=_default_memory_mb,
    ),
    Setting(
        "INCODE_VECTOR_INDEX", "Indexing", "Vector index",
        "exact search, or approximate HNSW indexing.",
        "choice", "exact", choices=("exact", "hnsw"),
    ),
    Setting(
        "INCODE_INDEX_EXECUTION", "Indexing", "Index execution",
        "worker enforces the memory ceiling; in-process is a diagnostic rollback.",
        "choice", "worker", choices=("worker", "in-process"),
    ),
    Setting(
        "INCODE_BROKER", "Indexing", "Broker",
        "Share one indexing process between clients through the daemon.",
        "choice", "auto", choices=("auto", "on", "off"),
    ),
    Setting(
        "INCODE_DATA_DIR", "Indexing", "Data directory",
        "Where the indexes live.",
        "path", "", dynamic_default=lambda: str(user_data_path("incode")),
    ),
    Setting(
        "INCODE_CACHE_DIR", "Indexing", "Cache directory",
        "Where the embedding model is cached.",
        "path", "", dynamic_default=lambda: str(user_cache_path("incode")),
    ),
    Setting(
        "INCODE_OFFLINE", "Indexing", "Offline mode",
        "Never download the model; fail if it is missing.",
        "bool", "0",
    ),
    Setting(
        "INCODE_EMBED_BATCH_SIZE", "Embedding", "Batch size",
        "Embedding microbatch size; auto resolves to 1 unless calibration raised it.",
        "auto_int", "auto", minimum=1, maximum=256,
    ),
    Setting(
        "INCODE_EMBED_MAX_TOKENS", "Embedding", "Max tokens",
        "Sequence window per chunk; attention memory is quadratic in tokens.",
        "int", "1024", minimum=64, maximum=8192,
    ),
    Setting(
        "INCODE_EMBED_OVERLAP_TOKENS", "Embedding", "Overlap tokens",
        "Overlap between consecutive windows of a long chunk.",
        "int", "64", minimum=0, maximum=4096,
    ),
    Setting(
        "INCODE_EMBED_THREADS", "Embedding", "Threads",
        "CPU inference threads.",
        "int", "", minimum=1, maximum=64, dynamic_default=_default_threads,
    ),
    Setting(
        "INCODE_EMBED_CPU_ARENA", "Embedding", "CPU arena",
        "Preallocate the CPU inference arena.",
        "bool", "0",
    ),
    Setting(
        "INCODE_EMBED_CROSSOVER", "Embedding", "Accelerator crossover",
        "Run size in characters above which starting the accelerator repays its model load.",
        "auto_off_int", "auto", minimum=0, maximum=1024**3,
    ),
    Setting(
        "INCODE_EMBED_CALIBRATE", "Embedding", "Calibrate",
        "Measure the backend once to set the batch size and crossover.",
        "bool", "1",
    ),
    Setting(
        "INCODE_EMBED_STRICT", "Embedding", "Strict accelerator",
        "Refuse the CPU fallback when the requested backend is unavailable.",
        "bool", "0",
    ),
    Setting(
        "INCODE_EMBED_ACCELERATOR", "Embedding", "Backend override",
        "Expert override; auto uses the backend the installer prepared.",
        "choice", "auto",
        choices=("auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"),
    ),
)

BY_NAME: dict[str, Setting] = {setting.name: setting for setting in SETTINGS}


def default_value(setting: Setting) -> str:
    if setting.dynamic_default is not None:
        return setting.dynamic_default()
    return setting.default


def validate(setting: Setting, raw: str) -> str | None:
    """Return an error message, or None when ``raw`` is acceptable."""
    value = raw.strip()
    if setting.type == "bool":
        if value.lower() in _TRUE | _FALSE:
            return None
        return f"{setting.name} expects a boolean (1/0, true/false, yes/no, on/off)"
    if setting.type == "path":
        return None if value else f"{setting.name} expects a path"
    if setting.type == "choice":
        if value.lower() in setting.choices:
            return None
        return f"{setting.name} expects one of: {', '.join(setting.choices)}"
    if setting.type == "auto_int" and value.lower() == "auto":
        return None
    if setting.type == "auto_off_int" and value.lower() in {"auto", "off"}:
        return None
    prefix = ""
    if setting.type == "auto_int":
        prefix = "auto or "
    elif setting.type == "auto_off_int":
        prefix = "auto, off, or "
    try:
        number = int(value)
    except ValueError:
        return f"{setting.name} expects {prefix}an integer from {setting.minimum} to {setting.maximum}"
    if setting.minimum <= number <= setting.maximum:
        return None
    return f"{setting.name} expects {prefix}an integer from {setting.minimum} to {setting.maximum}"


def normalize(setting: Setting, raw: str) -> str:
    """Canonical string form for storage in a harness env block."""
    value = raw.strip()
    if setting.type == "bool":
        return "1" if value.lower() in _TRUE else "0"
    if setting.type in {"choice", "auto_int", "auto_off_int"} and not value.lstrip("-").isdigit():
        return value.lower()
    return value
```

- [ ] **Step 3: Run tests, ruff, mypy; commit**

```bash
uv run pytest tests/test_installer_settings_spec.py -q
uv run ruff check src/incode_mcp/installer/settings_spec.py tests/test_installer_settings_spec.py
uv run mypy
git add src/incode_mcp/installer/settings_spec.py tests/test_installer_settings_spec.py
git commit -m "feat: add the installer's declarative settings catalog"
```

---

### Task 3: Harness environment blocks

Read the current server entry out of any harness config, merge managed env updates (preserving unrelated keys), and extend `configure_harness` / `merge_codex_server` to write env blocks. `env=None` must reproduce today's exact behavior so Task 1's suite stays green.

**Files:**
- Create: `src/incode_mcp/installer/env_blocks.py`
- Modify: `src/incode_mcp/installer/harnesses.py` (`configure_harness`, `configure_selected_harnesses`, new `read_server_entry`)
- Modify: `src/incode_mcp/installer/config_files.py` (`_codex_server_block`, `merge_codex_server` gain `env`)
- Test: `tests/test_installer_env_blocks.py`

**Interfaces:**
- Consumes: `config_files.SERVER_NAME`, `config_files._jsonc_as_json`, `config_files.merge_json_object_entry`, `config_files.merge_codex_server`, `harnesses.configuration_path` (Task 1).
- Produces: `ENV_KEYS: dict[str, str]`, `entry_from_text(slug, text) -> dict | None`, `env_from_entry(slug, entry) -> dict[str, str]`, `merge_env(existing, updates) -> dict[str, str]`, `harnesses.read_server_entry(slug, *, home, environment, platform_name) -> dict | None`, `configure_harness(..., env: Mapping[str, str | None] | None = None)`, `configure_selected_harnesses(..., env=...)`, `merge_codex_server(path, command, *, env: Mapping[str, str] | None = None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_installer_env_blocks.py`:

```python
"""Tests for harness environment-block reading, merging, and writing."""

import json
import tomllib
from pathlib import Path

from incode_mcp.installer.env_blocks import entry_from_text, env_from_entry, merge_env
from incode_mcp.installer.harnesses import configure_harness, read_server_entry

SERVER_COMMAND = str(Path("/opt/ci-mcp"))


def test_entry_from_text_reads_jsonc_with_comments() -> None:
    text = '{\n  // mine\n  "mcpServers": {"code-indexing-mcp": {"command": "/old", "args": ["serve"],}}\n}\n'
    entry = entry_from_text("kimi-code", text)
    assert entry == {"command": "/old", "args": ["serve"]}


def test_entry_from_text_reads_codex_toml() -> None:
    text = '[mcp_servers.code-indexing-mcp]\ncommand = "/old"\nargs = ["serve"]\nenv = { INCODE_OFFLINE = "1" }\n'
    assert entry_from_text("codex", text) == {
        "command": "/old",
        "args": ["serve"],
        "env": {"INCODE_OFFLINE": "1"},
    }


def test_entry_from_text_returns_none_for_missing_or_invalid() -> None:
    assert entry_from_text("kimi-code", "{}\n") is None
    assert entry_from_text("kimi-code", "not json") is None
    assert entry_from_text("codex", "not = = toml") is None


def test_env_from_entry_uses_the_per_harness_key() -> None:
    assert env_from_entry("opencode", {"environment": {"A": "1"}, "env": {"B": "2"}}) == {"A": "1"}
    assert env_from_entry("kimi-code", {"env": {"B": "2"}}) == {"B": "2"}
    assert env_from_entry("kimi-code", {}) == {}


def test_merge_env_applies_updates_deletions_and_preserves_unknown_keys() -> None:
    merged = merge_env({"KEEP": "x", "INCODE_OFFLINE": "1", "INCODE_BROKER": "off"}, {
        "INCODE_OFFLINE": "0",
        "INCODE_BROKER": None,
    })
    assert merged == {"KEEP": "x", "INCODE_OFFLINE": "0"}


def test_configure_harness_writes_env_and_preserves_unmanaged_keys(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({
        "mcpServers": {"code-indexing-mcp": {"command": "/old", "args": ["serve"],
                                             "env": {"KEEP": "x", "INCODE_BROKER": "off"}}}
    }))
    configure_harness(
        "kimi-code", Path(SERVER_COMMAND),
        env={"INCODE_BROKER": None, "INCODE_INDEX_MODE": "eager"},
        environment={"KIMI_CODE_HOME": str(tmp_path)},
    )
    entry = json.loads(config.read_text())["mcpServers"]["code-indexing-mcp"]
    assert entry == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"KEEP": "x", "INCODE_INDEX_MODE": "eager"},
    }


def test_configure_harness_opencode_uses_environment_key(tmp_path: Path) -> None:
    configure_harness(
        "opencode", Path(SERVER_COMMAND), env={"INCODE_OFFLINE": "1"},
        environment={"OPENCODE_CONFIG_DIR": str(tmp_path)},
    )
    entry = json.loads((tmp_path / "opencode.json").read_text())["mcp"]["code-indexing-mcp"]
    assert entry == {
        "type": "local",
        "command": [SERVER_COMMAND, "serve"],
        "enabled": True,
        "environment": {"INCODE_OFFLINE": "1"},
    }
    assert "env" not in entry


def test_configure_harness_codex_writes_toml_env_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    configure_harness("codex", Path(SERVER_COMMAND), env={"INCODE_OFFLINE": "1"},
                      environment={"CODEX_HOME": str(tmp_path)})
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"] == {
        "command": SERVER_COMMAND,
        "args": ["serve"],
        "env": {"INCODE_OFFLINE": "1"},
    }


def test_configure_harness_codex_update_preserves_unmanaged_env(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[mcp_servers.code-indexing-mcp]\ncommand = "/old"\nargs = ["serve"]\n'
        'env = { KEEP = "x" }\n'
    )
    configure_harness("codex", Path(SERVER_COMMAND), env={"INCODE_OFFLINE": "1"},
                      environment={"CODEX_HOME": str(tmp_path)})
    parsed = tomllib.loads(path.read_text())
    assert parsed["mcp_servers"]["code-indexing-mcp"]["env"] == {"KEEP": "x", "INCODE_OFFLINE": "1"}


def test_configure_harness_without_env_reproduces_the_legacy_entries(tmp_path: Path) -> None:
    configure_harness("kimi-code", Path(SERVER_COMMAND), environment={"KIMI_CODE_HOME": str(tmp_path)})
    entry = json.loads((tmp_path / "mcp.json").read_text())["mcpServers"]["code-indexing-mcp"]
    assert entry == {"command": SERVER_COMMAND, "args": ["serve"]}


def test_read_server_entry_returns_none_when_unconfigured(tmp_path: Path) -> None:
    assert read_server_entry("kimi-code", environment={"KIMI_CODE_HOME": str(tmp_path)}) is None
```

Run: `uv run pytest tests/test_installer_env_blocks.py -q` — expect FAIL.

- [ ] **Step 2: Implement `env_blocks.py`**

```python
"""Environment-block handling for harness MCP server entries.

Each harness passes environment variables to a stdio MCP server under its own
key: ``env`` almost everywhere, ``environment`` for the OpenCode-schema
harnesses (OpenCode and KiloCode). Managed updates merge; unrelated keys the
user placed in the block are preserved.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from typing import Any

from .config_files import SERVER_NAME, _jsonc_as_json

ENV_KEYS: dict[str, str] = {
    "codex": "env",
    "claude-code": "env",
    "kimi-code": "env",
    "claude-desktop": "env",
    "opencode": "environment",
    "kilocode": "environment",
}

_OBJECT_KEYS: dict[str, str] = {
    "claude-code": "mcpServers",
    "kimi-code": "mcpServers",
    "claude-desktop": "mcpServers",
    "opencode": "mcp",
    "kilocode": "mcp",
}


def entry_from_text(slug: str, text: str) -> dict[str, Any] | None:
    """Parse the Code Indexing MCP server entry out of a harness config's text."""
    servers: Any
    try:
        if slug == "codex":
            servers = tomllib.loads(text).get("mcp_servers")
        else:
            object_key = _OBJECT_KEYS.get(slug)
            if object_key is None:
                return None
            servers = json.loads(_jsonc_as_json(text)).get(object_key)
    except ValueError:
        return None
    if not isinstance(servers, dict):
        return None
    entry = servers.get(SERVER_NAME)
    return dict(entry) if isinstance(entry, dict) else None


def env_from_entry(slug: str, entry: Mapping[str, Any]) -> dict[str, str]:
    """Return the entry's environment block under this harness's key."""
    raw = entry.get(ENV_KEYS[slug])
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def merge_env(existing: Mapping[str, str], updates: Mapping[str, str | None]) -> dict[str, str]:
    """Apply managed updates to an existing block; a None value deletes the key."""
    merged = dict(existing)
    for key, value in updates.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged
```

- [ ] **Step 3: Extend `merge_codex_server` in `config_files.py`**

Replace `_codex_server_block` and the `merge_codex_server` signature:

```python
def _codex_server_block(command: Path, env: Mapping[str, str] | None = None) -> str:
    encoded_command = json.dumps(str(command), ensure_ascii=False)
    lines = [
        f"[mcp_servers.{SERVER_NAME}]",
        f"command = {encoded_command}",
        'args = ["serve"]',
    ]
    if env:
        pairs = ", ".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in sorted(env.items())
        )
        lines.append(f"env = {{ {pairs} }}")
    return "\n".join(lines) + "\n"


def merge_codex_server(path: Path, command: Path, *, env: Mapping[str, str] | None = None) -> bool:
    """Create or replace only the Code Indexing MCP table in a Codex config."""
```

(INCODE_* names are valid TOML bare keys: uppercase letters, digits, underscores only.) Change the one call site inside `merge_codex_server` from `block = _codex_server_block(command)` to `block = _codex_server_block(command, env)`. Keep everything else in the function byte-identical, including the comment-preserving replacement logic.

- [ ] **Step 4: Extend `harnesses.py`**

Add imports: `from .env_blocks import ENV_KEYS, entry_from_text, env_from_entry, merge_env`.

Add the reader:

```python
def read_server_entry(
    slug: str,
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> dict[str, Any] | None:
    """Return the current server entry in a harness config, or None."""

    path = configuration_path(
        slug, home=home, environment=environment, platform_name=platform_name
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return entry_from_text(slug, text)
```

Replace `configure_harness` with the env-aware version (docstring preserved and extended):

```python
def configure_harness(
    slug: str,
    command: Path,
    *,
    env: Mapping[str, str | None] | None = None,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    """Merge the Code Indexing MCP entry into one user-wide harness config.

    ``env`` maps managed setting names to values, or to None to delete a key;
    unrelated keys already in the entry's env block are preserved. When ``env``
    is None the legacy entries are written exactly as before.
    """

    path = configuration_path(
        slug, home=home, environment=environment, platform_name=platform_name
    )
    merged_env: dict[str, str] = {}
    if env is not None:
        existing = read_server_entry(
            slug, home=home, environment=environment, platform_name=platform_name
        )
        merged_env = merge_env(env_from_entry(slug, existing) if existing else {}, env)
    if slug == "codex":
        merge_codex_server(path, command, env=merged_env if env is not None else None)
        return path

    if slug == "claude-code":
        object_key = "mcpServers"
        entry: dict[str, Any] = {
            "type": "stdio",
            "command": str(command),
            "args": ["serve"],
            "env": merged_env if env is not None else {},
        }
    elif slug in {"kimi-code", "claude-desktop"}:
        object_key = "mcpServers"
        entry = {"command": str(command), "args": ["serve"]}
        if merged_env:
            entry["env"] = merged_env
    elif slug in {"opencode", "kilocode"}:
        object_key = "mcp"
        entry = {
            "type": "local",
            "command": [str(command), "serve"],
            "enabled": True,
        }
        if merged_env:
            entry["environment"] = merged_env
    else:
        raise InstallerError(f"Unknown harness {slug!r}")

    merge_json_object_entry(path, object_key, SERVER_NAME, entry)
    return path
```

Note: with `env=None`, `merged_env` stays `{}`, so kimi-code/claude-desktop/opencode/kilocode entries gain no env key and claude-code keeps its `"env": {}` — byte-for-byte the legacy shapes.

Extend `configure_selected_harnesses` with `env: Mapping[str, str | None] | None = None` after `command`, passed straight through to `configure_harness`.

- [ ] **Step 5: Run everything; commit**

```bash
uv run pytest tests/test_installer_env_blocks.py tests/test_installer.py -q
uv run ruff check src/incode_mcp/installer tests/test_installer_env_blocks.py
uv run mypy
git add src/incode_mcp/installer/env_blocks.py src/incode_mcp/installer/harnesses.py src/incode_mcp/installer/config_files.py tests/test_installer_env_blocks.py
git commit -m "feat: write managed settings into harness environment blocks"
```

---

### Task 4: Orchestrator (`orchestrator.py`)

The post-sync pipeline as event-emitting steps, shared by the module CLI and the TUI progress screen.

**Files:**
- Create: `src/incode_mcp/installer/orchestrator.py`
- Test: `tests/test_installer_orchestrator.py`

**Interfaces:**
- Consumes: `accelerator.configure_accelerator`, `accelerator.server_executable`, `harnesses.configure_selected_harnesses`, `harnesses.install_skills` (Tasks 1, 3).
- Produces: `DEFAULT_REPOSITORY_URL`, `default_install_directory() -> Path`, `InstallPlan` (frozen dataclass: `install_directory: Path`, `accelerator: str | None = "auto"`, `harness_slugs: tuple[str, ...] = ()`, `env_updates: Mapping[str, str | None]`, `offline: bool = False`), `StepEvent(step: str, status: str, detail: str = "")` with statuses `started|finished|warning|failed|skipped`, `InstallResult(accelerator_plan, configured, failures, skills)`, `run_install(plan, on_event, should_continue) -> InstallResult`.

- [ ] **Step 1: Write the failing tests**

`tests/test_installer_orchestrator.py`:

```python
"""Tests for the shared install pipeline."""

from pathlib import Path

import pytest

from incode_mcp.installer import accelerator, harnesses
from incode_mcp.installer.orchestrator import (
    InstallPlan,
    StepEvent,
    default_install_directory,
    run_install,
)


def _plan(**overrides) -> InstallPlan:
    values = {
        "install_directory": Path("/opt/ci-mcp"),
        "accelerator": "cpu",
        "harness_slugs": ("kimi-code",),
        "env_updates": {"INCODE_OFFLINE": "1"},
    }
    values.update(overrides)
    return InstallPlan(**values)


def test_run_install_emits_step_events_in_order(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        accelerator, "configure_accelerator",
        lambda directory, requested, *, offline=False: calls.append(("accel", requested))
        or accelerator.AcceleratorPlan("cpu", "CPU was requested"),
    )
    monkeypatch.setattr(accelerator, "server_executable", lambda directory: tmp_path / "server")
    monkeypatch.setattr(
        harnesses, "configure_selected_harnesses",
        lambda slugs, command, *, env=None, **kwargs: (
            calls.append(("harnesses", tuple(slugs), dict(env or {}))),
            ([("kimi-code", tmp_path / "mcp.json")], []),
        )[1],
    )
    monkeypatch.setattr(
        harnesses, "install_skills",
        lambda slugs, directory: calls.append(("skills", tuple(slugs)))
        or [("kimi-code", "1 linked, 3 already installed")],
    )
    events = []

    result = run_install(_plan(), on_event=events.append)

    assert calls == [
        ("accel", "cpu"),
        ("harnesses", ("kimi-code",), {"INCODE_OFFLINE": "1"}),
        ("skills", ("kimi-code",)),
    ]
    assert [event.step for event in events] == [
        "accelerator", "accelerator", "harnesses", "harnesses", "skills", "skills",
    ]
    assert events[0] == StepEvent("accelerator", "started", "cpu")
    assert result.failures == ()
    assert result.accelerator_plan is not None
    assert result.accelerator_plan.accelerator == "cpu"


def test_run_install_reports_unhonored_accelerator_as_warning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        accelerator, "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan(
            "cpu", "CUDA was requested but no driver", honored=False
        ),
    )
    monkeypatch.setattr(accelerator, "server_executable", lambda directory: tmp_path / "server")
    monkeypatch.setattr(
        harnesses, "configure_selected_harnesses", lambda *args, **kwargs: ([], [])
    )
    monkeypatch.setattr(harnesses, "install_skills", lambda *args: [])
    events = []

    run_install(_plan(accelerator="cuda"), on_event=events.append)

    accelerator_events = [event for event in events if event.step == "accelerator"]
    assert accelerator_events[-1].status == "warning"
    assert "no driver" in accelerator_events[-1].detail


def test_run_install_skips_accelerator_when_plan_keeps_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        accelerator, "configure_accelerator",
        lambda *args, **kwargs: pytest.fail("accelerator step must not run"),
    )
    monkeypatch.setattr(accelerator, "server_executable", lambda directory: tmp_path / "server")
    monkeypatch.setattr(
        harnesses, "configure_selected_harnesses", lambda *args, **kwargs: ([], [])
    )
    monkeypatch.setattr(harnesses, "install_skills", lambda *args: [])
    events = []

    result = run_install(_plan(accelerator=None), on_event=events.append)

    assert result.accelerator_plan is None
    assert events[0] == StepEvent("accelerator", "skipped", "keeping the prepared backend")


def test_run_install_stops_between_steps_when_cancelled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        accelerator, "configure_accelerator",
        lambda directory, requested, *, offline=False: accelerator.AcceleratorPlan("cpu", "ok"),
    )
    monkeypatch.setattr(
        harnesses, "configure_selected_harnesses",
        lambda *args, **kwargs: pytest.fail("must not run after cancellation"),
    )
    monkeypatch.setattr(
        harnesses, "install_skills",
        lambda *args: pytest.fail("must not run after cancellation"),
    )

    result = run_install(_plan(), should_continue=lambda: False)

    assert result.accelerator_plan is None
    assert result.configured == () and result.skills == ()


def test_default_install_directory_honours_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CODE_INDEXING_MCP_INSTALL_DIR", str(tmp_path / "custom"))
    assert default_install_directory() == tmp_path / "custom"
    monkeypatch.delenv("CODE_INDEXING_MCP_INSTALL_DIR")
    assert default_install_directory() == Path.home() / ".local" / "share" / "code-indexing-mcp"
```

Run: `uv run pytest tests/test_installer_orchestrator.py -q` — expect FAIL.

- [ ] **Step 2: Implement `orchestrator.py`**

```python
"""The install pipeline as event-emitting steps, shared by the CLI and the TUI."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import accelerator, harnesses

DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git"


def default_install_directory() -> Path:
    configured = os.environ.get("CODE_INDEXING_MCP_INSTALL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "code-indexing-mcp"


@dataclass(frozen=True)
class InstallPlan:
    install_directory: Path
    # None skips the accelerator step: the reconfigure default, which keeps the
    # backend the last install prepared.
    accelerator: str | None = "auto"
    harness_slugs: tuple[str, ...] = ()
    env_updates: Mapping[str, str | None] = field(default_factory=dict)
    offline: bool = False


@dataclass(frozen=True)
class StepEvent:
    step: str  # "accelerator" | "harnesses" | "skills"
    status: str  # "started" | "finished" | "warning" | "failed" | "skipped"
    detail: str = ""


@dataclass(frozen=True)
class InstallResult:
    accelerator_plan: accelerator.AcceleratorPlan | None
    configured: tuple[tuple[str, Path], ...]
    failures: tuple[tuple[str, str], ...]
    skills: tuple[tuple[str, str], ...]


def run_install(
    plan: InstallPlan,
    on_event: Callable[[StepEvent], None] = lambda event: None,
    should_continue: Callable[[], bool] = lambda: True,
) -> InstallResult:
    """Run the pipeline, emitting an event at every step boundary.

    ``should_continue`` is checked between steps so a UI can cancel cleanly;
    a step already running always finishes.
    """

    accelerator_plan: accelerator.AcceleratorPlan | None = None
    if plan.accelerator is None:
        on_event(StepEvent("accelerator", "skipped", "keeping the prepared backend"))
    elif should_continue():
        on_event(StepEvent("accelerator", "started", plan.accelerator))
        accelerator_plan = accelerator.configure_accelerator(
            plan.install_directory, plan.accelerator, offline=plan.offline
        )
        status = "finished" if accelerator_plan.honored else "warning"
        detail = f"{accelerator_plan.accelerator} ({accelerator_plan.reason})"
        on_event(StepEvent("accelerator", status, detail))

    configured: list[tuple[str, Path]] = []
    failures: list[tuple[str, str]] = []
    skills: list[tuple[str, str]] = []
    if should_continue():
        command = accelerator.server_executable(plan.install_directory)
        on_event(
            StepEvent("harnesses", "started", ", ".join(plan.harness_slugs) or "none selected")
        )
        configured, failures = harnesses.configure_selected_harnesses(
            list(plan.harness_slugs), command, env=plan.env_updates
        )
        for slug, path in configured:
            on_event(StepEvent("harnesses", "finished", f"{slug}: {path}"))
        for slug, message in failures:
            on_event(StepEvent("harnesses", "failed", f"{slug}: {message}"))
    if should_continue():
        on_event(StepEvent("skills", "started"))
        skills = harnesses.install_skills(list(plan.harness_slugs), plan.install_directory)
        for slug, message in skills:
            on_event(StepEvent("skills", "finished", f"{slug}: {message}"))
    return InstallResult(accelerator_plan, tuple(configured), tuple(failures), tuple(skills))
```

- [ ] **Step 3: Run tests, ruff, mypy; commit**

```bash
uv run pytest tests/test_installer_orchestrator.py -q
uv run ruff check src/incode_mcp/installer/orchestrator.py tests/test_installer_orchestrator.py
uv run mypy
git add src/incode_mcp/installer/orchestrator.py tests/test_installer_orchestrator.py
git commit -m "feat: add the event-emitting install pipeline"
```

---

### Task 5: Wizard state (`wizard.py`)

Textual-free state for the wizard: mode, prefill from existing harness configs, and conversion to an `InstallPlan`. Fully unit-testable without a terminal.

**Files:**
- Create: `src/incode_mcp/installer/wizard.py`
- Test: `tests/test_installer_wizard.py`

**Interfaces:**
- Consumes: `harnesses.HARNESS_CHOICES`, `harnesses.read_server_entry` (Tasks 1, 3), `env_blocks.env_from_entry` (Task 3), `settings_spec` (Task 2), `orchestrator.InstallPlan`, `orchestrator.default_install_directory`, `orchestrator.DEFAULT_REPOSITORY_URL` (Task 4), `accelerator.prepared_accelerator` (new in this task).
- Produces: `Prefill(values, configured_slugs, disagreements)`, `load_prefill(*, home, environment) -> Prefill`, `WizardState` dataclass with `for_install(...)`, `for_reconfigure(...)`, `field_value/set_field`, `env_updates() -> dict[str, str | None]`, `to_plan() -> InstallPlan`; `accelerator.prepared_accelerator(install_directory) -> str | None`, `accelerator.detection_report() -> list[str]`.

- [ ] **Step 1: Add the two new `accelerator.py` helpers with tests**

Append to `tests/test_installer_wizard.py` (created in Step 2 below) or extend `tests/test_installer.py`-style coverage in a new section of the new test file:

```python
def test_prepared_accelerator_reads_the_record(monkeypatch, tmp_path: Path) -> None:
    record = tmp_path / "accelerator.json"
    record.write_text(json.dumps({"accelerator": "mlx"}))
    monkeypatch.setattr(accelerator, "accelerator_record_path", lambda directory: record)
    assert accelerator.prepared_accelerator(tmp_path) == "mlx"
    record.unlink()
    assert accelerator.prepared_accelerator(tmp_path) is None


def test_detection_report_mentions_platform_and_devices(monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "_nvidia_smi_report", lambda: "555.42, GeForce RTX")
    monkeypatch.setattr(accelerator, "_rocm_report", lambda: None)
    facts = accelerator.detection_report()
    assert any(line.startswith("Platform:") for line in facts)
    assert any("555.42" in line for line in facts)
    assert any(line == "ROCm: not detected" for line in facts)
```

Implement in `src/incode_mcp/installer/accelerator.py`:

```python
def prepared_accelerator(install_directory: Path) -> str | None:
    """Return the accelerator the recorded environment provides, if any."""

    try:
        record = accelerator_record_path(install_directory)
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError, InstallerError):
        return None
    value = payload.get("accelerator")
    return value if isinstance(value, str) and value else None


def detection_report() -> list[str]:
    """Human-readable facts about this machine's accelerator options."""

    platform_name = _normalized_platform(sys.platform.lower())
    machine = platform.machine().lower()
    facts = [f"Platform: {platform_name}/{machine}"]
    nvidia = _nvidia_smi_report()
    facts.append(
        "NVIDIA: " + (nvidia.strip().splitlines()[0] if nvidia and nvidia.strip() else "not detected")
    )
    rocm = _rocm_report()
    facts.append(f"ROCm: {rocm or 'not detected'}")
    if platform_name == "darwin":
        version = platform.mac_ver()[0]
        facts.append(f"macOS: {version}")
        problem = _mlx_problem(
            platform_name=platform_name, machine=machine, platform_version=version
        )
        facts.append(f"MLX: {'available' if not problem else f'unavailable — {problem}'}")
    supported = WEBGPU_PLATFORMS.get(platform_name)
    facts.append(
        "WebGPU plugin wheels: "
        + ("available" if supported and machine in supported else f"not published for {platform_name}/{machine}")
    )
    cuda_supported = CUDA_PLATFORMS.get(platform_name)
    facts.append(
        "CUDA wheels: "
        + ("published" if cuda_supported and machine in cuda_supported else f"not published for {platform_name}/{machine}")
    )
    return facts
```

- [ ] **Step 2: Write the failing wizard tests**

`tests/test_installer_wizard.py`:

```python
"""Tests for the Textual-free wizard state."""

import json
from pathlib import Path

from incode_mcp.installer import accelerator
from incode_mcp.installer.wizard import WizardState, load_prefill


def _write_kimi_config(home: Path, env: dict[str, str]) -> None:
    directory = home / ".kimi-code"
    directory.mkdir(parents=True)
    (directory / "mcp.json").write_text(json.dumps({
        "mcpServers": {"code-indexing-mcp": {"command": "/opt/ci-mcp", "args": ["serve"],
                                             "env": env}}
    }))


def test_load_prefill_collects_values_and_configured_harnesses(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"INCODE_INDEX_MODE": "eager", "UNRELATED": "keep"})
    prefill = load_prefill(home=tmp_path)
    assert prefill.values == {"INCODE_INDEX_MODE": "eager"}
    assert prefill.configured_slugs == ("kimi-code",)
    assert prefill.disagreements == ()


def test_load_prefill_reports_disagreements_in_choice_order(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"INCODE_INDEX_MODE": "manual"})
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "config.toml").write_text(
        '[mcp_servers.code-indexing-mcp]\ncommand = "/opt/ci-mcp"\nargs = ["serve"]\n'
        'env = { INCODE_INDEX_MODE = "eager" }\n'
    )
    prefill = load_prefill(home=tmp_path)
    # codex precedes kimi-code in HARNESS_CHOICES, so its value wins.
    assert prefill.values == {"INCODE_INDEX_MODE": "eager"}
    assert prefill.disagreements == ("INCODE_INDEX_MODE",)
    assert prefill.configured_slugs == ("codex", "kimi-code")


def test_env_updates_omit_defaults_and_delete_reset_prefills(tmp_path: Path) -> None:
    _write_kimi_config(tmp_path, {"INCODE_INDEX_MODE": "eager", "INCODE_BROKER": "off"})
    state = WizardState.for_reconfigure(Path("/opt/ci-mcp"), home=tmp_path)
    state.set_field("INCODE_INDEX_MODE", "lazy")   # back to default -> delete
    state.set_field("INCODE_BROKER", "on")         # non-default -> write
    state.set_field("INCODE_EMBED_THREADS", "")    # untouched -> no entry
    assert state.env_updates() == {"INCODE_INDEX_MODE": None, "INCODE_BROKER": "on"}


def test_install_mode_never_deletes(tmp_path: Path) -> None:
    state = WizardState.for_install(Path("/opt/ci-mcp"), "https://example.invalid/repo.git",
                                    home=tmp_path)
    state.set_field("INCODE_INDEX_MODE", "lazy")
    assert state.env_updates() == {}


def test_to_plan_carries_everything(tmp_path: Path) -> None:
    state = WizardState.for_install(Path("/opt/ci-mcp"), "https://example.invalid/repo.git",
                                    home=tmp_path)
    state.accelerator = "mlx"
    state.harness_slugs = ["kimi-code"]
    state.set_field("INCODE_OFFLINE", "1")
    plan = state.to_plan()
    assert plan.install_directory == Path("/opt/ci-mcp")
    assert plan.accelerator == "mlx"
    assert plan.harness_slugs == ("kimi-code",)
    assert plan.env_updates == {"INCODE_OFFLINE": "1"}


def test_for_reconfigure_keeps_prepared_backend_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(accelerator, "prepared_accelerator", lambda directory: "mlx")
    state = WizardState.for_reconfigure(Path("/opt/ci-mcp"), home=tmp_path)
    assert state.accelerator is None
    assert state.prepared_accelerator == "mlx"
```

Run: `uv run pytest tests/test_installer_wizard.py -q` — expect FAIL.

- [ ] **Step 3: Implement `wizard.py`**

```python
"""Wizard state shared by the Textual UI; no Textual imports in this module."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import accelerator, harnesses
from .env_blocks import env_from_entry
from .orchestrator import DEFAULT_REPOSITORY_URL, InstallPlan, default_install_directory
from .settings_spec import BY_NAME, SETTINGS, default_value, normalize


@dataclass(frozen=True)
class Prefill:
    values: Mapping[str, str]  # managed env values found in harness configs
    configured_slugs: tuple[str, ...]  # harnesses that already have a server entry
    disagreements: tuple[str, ...]  # env names whose values differ between harnesses


def load_prefill(
    *,
    home: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Prefill:
    """Read every harness's current server entry for wizard prefill."""

    values: dict[str, str] = {}
    configured: list[str] = []
    disagreements: list[str] = []
    for choice in harnesses.HARNESS_CHOICES:
        entry = harnesses.read_server_entry(choice.slug, home=home, environment=environment)
        if entry is None:
            continue
        configured.append(choice.slug)
        for name, value in env_from_entry(choice.slug, entry).items():
            if name not in BY_NAME:
                continue
            if name in values and values[name] != value:
                if name not in disagreements:
                    disagreements.append(name)
            else:
                values[name] = value
    return Prefill(values, tuple(configured), tuple(disagreements))


@dataclass
class WizardState:
    mode: str  # "install" | "reconfigure"
    install_directory: Path = field(default_factory=default_install_directory)
    repo_url: str = DEFAULT_REPOSITORY_URL
    # None keeps the prepared backend (reconfigure default); install mode uses "auto".
    accelerator: str | None = "auto"
    prepared_accelerator: str | None = None
    harness_slugs: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)  # env name -> raw field value
    prefilled_names: set[str] = field(default_factory=set)  # names found in existing configs
    disagreements: list[str] = field(default_factory=list)
    offline: bool = False

    @classmethod
    def for_install(
        cls,
        install_directory: Path,
        repo_url: str,
        *,
        preset_values: Mapping[str, str] | None = None,
        preset_accelerator: str | None = None,
        home: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> WizardState:
        prefill = load_prefill(home=home, environment=environment)
        values = dict(prefill.values)
        values.update(preset_values or {})
        return cls(
            mode="install",
            install_directory=install_directory,
            repo_url=repo_url,
            accelerator=preset_accelerator or "auto",
            harness_slugs=list(prefill.configured_slugs),
            values=values,
            prefilled_names=set(prefill.values),
            disagreements=list(prefill.disagreements),
        )

    @classmethod
    def for_reconfigure(
        cls,
        install_directory: Path,
        *,
        home: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> WizardState:
        prefill = load_prefill(home=home, environment=environment)
        return cls(
            mode="reconfigure",
            install_directory=install_directory,
            accelerator=None,
            prepared_accelerator=accelerator.prepared_accelerator(install_directory),
            harness_slugs=list(prefill.configured_slugs),
            values=dict(prefill.values),
            prefilled_names=set(prefill.values),
            disagreements=list(prefill.disagreements),
        )

    def field_value(self, name: str) -> str:
        return self.values.get(name, "")

    def set_field(self, name: str, raw: str) -> None:
        self.values[name] = raw

    def env_updates(self) -> dict[str, str | None]:
        """Non-default values to write; prefilled values reset to default delete the key."""

        updates: dict[str, str | None] = {}
        for setting in SETTINGS:
            raw = self.values.get(setting.name, "").strip()
            if not raw or raw == default_value(setting):
                if self.mode == "reconfigure" and setting.name in self.prefilled_names:
                    updates[setting.name] = None
                continue
            updates[setting.name] = normalize(setting, raw)
        return updates

    def to_plan(self) -> InstallPlan:
        return InstallPlan(
            install_directory=self.install_directory,
            accelerator=self.accelerator,
            harness_slugs=tuple(self.harness_slugs),
            env_updates=self.env_updates(),
            offline=self.offline,
        )
```

- [ ] **Step 4: Run tests, ruff, mypy; commit**

```bash
uv run pytest tests/test_installer_wizard.py -q
uv run ruff check src/incode_mcp/installer/wizard.py src/incode_mcp/installer/accelerator.py tests/test_installer_wizard.py
uv run mypy
git add src/incode_mcp/installer/wizard.py src/incode_mcp/installer/accelerator.py tests/test_installer_wizard.py
git commit -m "feat: add wizard state with harness-config prefill"
```

---

### Task 6: Installer module CLI (`installer/cli.py` + `__main__.py`)

The non-interactive entry point the bootstrap delegates to. Also serves `configure --set` scripted reconfiguration via `--reconfigure`.

**Files:**
- Create: `src/incode_mcp/installer/cli.py`
- Create: `src/incode_mcp/installer/__main__.py`
- Test: `tests/test_installer_cli.py`

**Interfaces:**
- Consumes: `orchestrator` (Task 4), `harnesses.parse_harness_selection`, `harnesses.HARNESS_CHOICES`, `settings_spec` (Task 2), `wizard.load_prefill` (Task 5), `config_files.InstallerError` (Task 1), `accelerator.ACCELERATOR_CHOICES` (Task 1).
- Produces: `main(argv: Sequence[str] | None = None) -> int`, `configure_main(*, install_dir, accelerator, harnesses, settings, unsets, no_tui) -> int`, `parse_settings(pairs, unsets) -> dict[str, str | None]`, `build_parser() -> argparse.ArgumentParser`.

- [ ] **Step 1: Write the failing tests**

`tests/test_installer_cli.py`:

```python
"""Tests for the installer's module CLI."""

from pathlib import Path

import pytest

from incode_mcp.installer import orchestrator
from incode_mcp.installer.cli import main, parse_settings
from incode_mcp.installer.config_files import InstallerError


def test_parse_settings_validates_and_normalizes() -> None:
    updates = parse_settings(
        ["INCODE_INDEX_MODE=EAGER", "INCODE_OFFLINE=yes"], ["INCODE_BROKER"]
    )
    assert updates == {"INCODE_INDEX_MODE": "eager", "INCODE_OFFLINE": "1", "INCODE_BROKER": None}


@pytest.mark.parametrize(
    "pair",
    ["INCODE_FROBNICATE=1", "INCODE_INDEX_MODE=sometimes", "INCODE_OFFLINE"],
)
def test_parse_settings_rejects_bad_input(pair: str) -> None:
    with pytest.raises(InstallerError):
        parse_settings([pair], [])


def _stub_pipeline(monkeypatch, recorded: list) -> None:
    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):
        recorded.append(plan)
        return orchestrator.InstallResult(None, (), (), ())

    monkeypatch.setattr(orchestrator, "run_install", fake_run_install)
    # cli.py imported run_install by name; patch the name it looks up.
    import incode_mcp.installer.cli as cli

    monkeypatch.setattr(cli, "run_install", fake_run_install)


def test_main_runs_plan_without_prompting(monkeypatch, tmp_path: Path, capsys) -> None:
    recorded = []
    _stub_pipeline(monkeypatch, recorded)
    code = main([
        "--install-dir", str(tmp_path),
        "--accelerator", "cpu",
        "--harnesses", "kimi-code",
        "--set", "INCODE_OFFLINE=1",
        "--no-prompt",
    ])
    assert code == 0
    (plan,) = recorded
    assert plan.accelerator == "cpu"
    assert plan.harness_slugs == ("kimi-code",)
    assert plan.env_updates == {"INCODE_OFFLINE": "1"}


def test_main_defaults_to_auto_accelerator_on_install(monkeypatch, tmp_path: Path) -> None:
    recorded = []
    _stub_pipeline(monkeypatch, recorded)
    assert main(["--install-dir", str(tmp_path), "--no-prompt"]) == 0
    assert recorded[0].accelerator == "auto"
    assert recorded[0].harness_slugs == ()


def test_reconfigure_keeps_backend_and_prefills_harnesses(monkeypatch, tmp_path: Path) -> None:
    recorded = []
    _stub_pipeline(monkeypatch, recorded)
    monkeypatch.setattr(
        "incode_mcp.installer.cli.load_prefill",
        lambda: __import__("incode_mcp.installer.wizard", fromlist=["Prefill"]).Prefill(
            {}, ("kimi-code",), ()
        ),
    )
    assert main(["--install-dir", str(tmp_path), "--reconfigure", "--no-prompt"]) == 0
    assert recorded[0].accelerator is None
    assert recorded[0].harness_slugs == ("kimi-code",)


def test_main_reports_installer_errors(monkeypatch, tmp_path: Path, capsys) -> None:
    _stub_pipeline(monkeypatch, [])
    code = main(["--install-dir", str(tmp_path), "--set", "INCODE_NOPE=1", "--no-prompt"])
    assert code == 1
    assert "INCODE_NOPE" in capsys.readouterr().err


def test_main_tui_flag_delegates(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class FakeApp:
        def __init__(self, state):
            calls.append(state)

        def run(self):
            return None

    import sys as _sys
    import types

    fake_module = types.ModuleType("incode_mcp.installer.tui.app")
    fake_module.InstallerApp = FakeApp  # type: ignore[attr-defined]
    monkeypatch.setitem(_sys.modules, "incode_mcp.installer.tui.app", fake_module)
    monkeypatch.setitem(_sys.modules, "incode_mcp.installer.tui", types.ModuleType("incode_mcp.installer.tui"))
    code = main(["--install-dir", str(tmp_path), "--tui", "--set", "INCODE_OFFLINE=1"])
    assert code == 0
    assert calls and calls[0].values.get("INCODE_OFFLINE") == "1"
```

Run: `uv run pytest tests/test_installer_cli.py -q` — expect FAIL.

- [ ] **Step 2: Implement `cli.py` and `__main__.py`**

`src/incode_mcp/installer/cli.py`:

```python
"""Non-interactive installer entry shared by the bootstrap and ``configure``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .accelerator import ACCELERATOR_CHOICES
from .config_files import InstallerError
from .harnesses import HARNESS_CHOICES, parse_harness_selection
from .orchestrator import (
    DEFAULT_REPOSITORY_URL,
    InstallPlan,
    StepEvent,
    default_install_directory,
    run_install,
)
from .settings_spec import BY_NAME, normalize, validate
from .wizard import load_prefill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="incode_mcp.installer",
        description="Install, update, or reconfigure Code Indexing MCP.",
    )
    parser.add_argument("--install-dir", default=str(default_install_directory()))
    parser.add_argument(
        "--accelerator",
        choices=ACCELERATOR_CHOICES,
        default=None,
        help="accelerator to prepare; omit to keep the prepared backend",
    )
    parser.add_argument("--harnesses", help="comma-separated harness numbers/slugs or 'all'")
    parser.add_argument(
        "--set", dest="settings", action="append", default=[], metavar="NAME=VALUE",
        help="set a managed INCODE_* value; repeatable",
    )
    parser.add_argument(
        "--unset", dest="unsets", action="append", default=[], metavar="NAME",
        help="remove a managed INCODE_* value from harness configs; repeatable",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("INCODE_OFFLINE", "").lower() in {"1", "true", "yes"},
    )
    parser.add_argument("--tui", action="store_true", help="open the interactive wizard")
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="never prompt; a missing harness selection configures none",
    )
    parser.add_argument("--reconfigure", action="store_true", help=argparse.SUPPRESS)
    return parser


def parse_settings(pairs: Sequence[str], unsets: Sequence[str]) -> dict[str, str | None]:
    updates: dict[str, str | None] = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        name = name.strip()
        if not separator:
            raise InstallerError(f"--set expects NAME=VALUE, got {pair!r}")
        setting = BY_NAME.get(name)
        if setting is None:
            options = ", ".join(sorted(BY_NAME))
            raise InstallerError(f"unknown setting {name!r}; managed settings: {options}")
        error = validate(setting, value)
        if error is not None:
            raise InstallerError(error)
        updates[name] = normalize(setting, value)
    for name in unsets:
        name = name.strip()
        if name not in BY_NAME:
            options = ", ".join(sorted(BY_NAME))
            raise InstallerError(f"unknown setting {name!r}; managed settings: {options}")
        updates[name] = None
    return updates


def _print_event(event: StepEvent) -> None:
    stream = sys.stderr if event.status in {"warning", "failed"} else sys.stdout
    print(f"[{event.step}] {event.status}: {event.detail}", file=stream)


def _prompt_harnesses(
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> list[str]:
    output_fn("Select the harnesses to configure:")
    for index, choice in enumerate(HARNESS_CHOICES, start=1):
        output_fn(f"  {index}. {choice.label}")
    return parse_harness_selection(
        input_fn("Enter comma-separated choices, 'all', or leave blank to skip: ")
    )


def _run_tui(args: argparse.Namespace, install_directory: Path, env_updates: dict[str, str | None]) -> int:
    from .tui.app import InstallerApp  # lazy: Textual is an optional dependency
    from .wizard import WizardState

    preset = {name: value for name, value in env_updates.items() if value is not None}
    if args.reconfigure:
        state = WizardState.for_reconfigure(install_directory)
        state.values.update(preset)
    else:
        state = WizardState.for_install(
            install_directory,
            os.environ.get("CODE_INDEXING_MCP_REPO_URL", DEFAULT_REPOSITORY_URL),
            preset_values=preset,
            preset_accelerator=args.accelerator,
        )
    state.offline = args.offline
    app = InstallerApp(state)
    app.run()
    return app.return_code


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    install_directory = Path(args.install_dir).expanduser().resolve()
    try:
        env_updates = parse_settings(args.settings, args.unsets)
        if args.tui:
            return _run_tui(args, install_directory, env_updates)
        if args.harnesses is not None:
            selected = parse_harness_selection(args.harnesses)
        elif args.reconfigure:
            selected = list(load_prefill().configured_slugs)
        elif args.no_prompt or not sys.stdin.isatty():
            selected = []
        else:
            selected = _prompt_harnesses()
        accelerator = args.accelerator
        if accelerator is None and not args.reconfigure:
            accelerator = "auto"
        plan = InstallPlan(
            install_directory=install_directory,
            accelerator=accelerator,
            harness_slugs=tuple(selected),
            env_updates=env_updates,
            offline=args.offline,
        )
        result = run_install(plan, on_event=_print_event)
    except InstallerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Installation cancelled.", file=sys.stderr)
        return 130
    if not result.configured and not result.failures and not result.skills:
        print("No harness configuration selected.")
    print("Installation complete. Restart configured clients to load the MCP server.")
    return 1 if result.failures else 0


def configure_main(
    *,
    install_dir: str | None,
    accelerator: str | None,
    harnesses: str | None,
    settings: Sequence[str],
    unsets: Sequence[str],
    no_tui: bool,
) -> int:
    """Entry for ``code-indexing-mcp configure``: reconfigure an existing install."""

    install_directory = (
        Path(install_dir).expanduser().resolve() if install_dir else default_install_directory().resolve()
    )
    from .accelerator import server_executable

    if not server_executable(install_directory).is_file():
        print(f"Error: no installation found at {install_directory}", file=sys.stderr)
        return 1
    argv = ["--install-dir", str(install_directory), "--reconfigure", "--no-prompt"]
    if accelerator is not None:
        argv += ["--accelerator", accelerator]
    if harnesses is not None:
        argv += ["--harnesses", harnesses]
    for pair in settings:
        argv += ["--set", pair]
    for name in unsets:
        argv += ["--unset", name]
    interactive = not no_tui and not settings and not unsets and sys.stdin.isatty()
    if interactive:
        argv.remove("--no-prompt")
        argv.append("--tui")
    return main(argv)
```

`src/incode_mcp/installer/__main__.py`:

```python
"""``python -m incode_mcp.installer`` — the bootstrap's delegation target."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run tests, ruff, mypy; commit**

```bash
uv run pytest tests/test_installer_cli.py -q
uv run ruff check src/incode_mcp/installer tests/test_installer_cli.py
uv run mypy
git add src/incode_mcp/installer/cli.py src/incode_mcp/installer/__main__.py tests/test_installer_cli.py
git commit -m "feat: add the installer module CLI with --set/--unset"
```

---

### Task 7: `configure` subcommand in `incode_mcp.cli`

**Files:**
- Modify: `src/incode_mcp/cli.py` (parser + dispatch)
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `installer.cli.configure_main` (Task 6).
- Produces: `code-indexing-mcp configure [--install-dir D] [--accelerator A] [--harnesses H] [--set K=V ...] [--unset K ...] [--no-tui]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_configure_delegates_to_the_installer(monkeypatch, capsys) -> None:
    calls = []

    def fake_configure_main(**kwargs):
        calls.append(kwargs)
        return 0

    import incode_mcp.installer.cli as installer_cli

    monkeypatch.setattr(installer_cli, "configure_main", fake_configure_main)
    code = main(["configure", "--install-dir", "/opt/ci-mcp", "--set", "INCODE_OFFLINE=1"])
    assert code == 0
    assert calls == [
        {
            "install_dir": "/opt/ci-mcp",
            "accelerator": None,
            "harnesses": None,
            "settings": ["INCODE_OFFLINE=1"],
            "unsets": [],
            "no_tui": False,
        }
    ]


def test_serve_path_does_not_import_textual() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import incode_mcp.cli, sys; print('textual' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
```

(Check `tests/test_cli.py`'s existing `main` import and match it — it already imports `from incode_mcp.cli import main` or similar; reuse its convention.)

Run: `uv run pytest tests/test_cli.py -q -k configure` — expect FAIL.

- [ ] **Step 2: Add the subcommand**

In `_parser()`, after the `daemon` subparser block, add:

```python
    configure = commands.add_parser(
        "configure", help="Reconfigure this installation (wizard, or scripted with --set)"
    )
    configure.add_argument("--install-dir", help="checkout location of the installation")
    configure.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"],
        default=None,
        help="prepare a different accelerator; omit to keep the prepared backend",
    )
    configure.add_argument("--harnesses", help="comma-separated harness slugs or 'all'")
    configure.add_argument(
        "--set", dest="settings", action="append", default=[], metavar="NAME=VALUE",
        help="set a managed INCODE_* value; repeatable",
    )
    configure.add_argument(
        "--unset", dest="unsets", action="append", default=[], metavar="NAME",
        help="remove a managed INCODE_* value from harness configs; repeatable",
    )
    configure.add_argument(
        "--no-tui", action="store_true", help="apply without opening the wizard"
    )
```

In `main()`, immediately after the `daemon` branch (before `settings = IndexSettings.from_environment()`), add:

```python
        if args.command == "configure":
            # Lazy import: the installer package is never on the serve path, and
            # Textual is only imported when the wizard actually opens.
            from .installer.cli import configure_main

            return configure_main(
                install_dir=args.install_dir,
                accelerator=args.accelerator,
                harnesses=args.harnesses,
                settings=args.settings,
                unsets=args.unsets,
                no_tui=args.no_tui,
            )
```

- [ ] **Step 3: Run tests, ruff, mypy; commit**

```bash
uv run pytest tests/test_cli.py -q
uv run ruff check src/incode_mcp/cli.py tests/test_cli.py
uv run mypy
git add src/incode_mcp/cli.py tests/test_cli.py
git commit -m "feat: add the code-indexing-mcp configure subcommand"
```

---

### Task 8: The `tui` extra and lockfile

**Files:**
- Modify: `pyproject.toml` (optional-dependencies)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Consumes: nothing.
- Produces: `textual` importable from the synced environment.

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, after the `migraphx` extra block, add:

```toml
# The interactive installer's only dependency. It is not part of the conflicts
# list: it combines with the serving environment's cpu extra and is lazily
# imported, so the serve path never pays for it.
tui = ["textual>=8.2,<9"]
```

- [ ] **Step 2: Regenerate the lockfile and sync**

```bash
uv lock
uv sync --locked --extra cpu --extra tui
uv run python -c "import textual; print(textual.__version__)"
```

Expected: resolves and prints `8.2.x`. If `uv sync` reports the lock is stale or the extra missing, re-run `uv lock` and inspect `uv tree --extra tui | grep textual`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add the tui extra with textual"
```

---

### Task 9: TUI app shell and navigation

The `InstallerApp` with a `ContentSwitcher`, central Back/Next/Cancel navigation, and per-panel `commit()` validation hooks. Panels live in `tui/panels.py`; this task creates them as minimal placeholders that later tasks fill in — with their final class signatures, so later tasks only add behavior.

**Files:**
- Create: `src/incode_mcp/installer/tui/__init__.py`
- Create: `src/incode_mcp/installer/tui/app.py`
- Create: `src/incode_mcp/installer/tui/panels.py`
- Create: `src/incode_mcp/installer/tui/settings_form.py`
- Test: `tests/test_installer_tui.py`

**Interfaces:**
- Consumes: `wizard.WizardState` (Task 5), `orchestrator.run_install`, `orchestrator.StepEvent` (Task 4).
- Produces: `InstallerApp(state)` with `return_code: int` (130 until the Done screen sets 0/1); panel classes `WelcomePanel`, `LocationPanel`, `AcceleratorPanel`, `HarnessesPanel`, `SettingsPanel`, `SummaryPanel`, `ProgressPanel`, `DonePanel`, each a `Vertical` subclass taking `state` first; the `commit() -> bool` protocol (True = may advance).

- [ ] **Step 1: Write the failing navigation tests**

`tests/test_installer_tui.py`:

```python
"""Headless tests for the Textual installer wizard."""

import pytest
from pathlib import Path

from incode_mcp.installer.tui.app import InstallerApp
from incode_mcp.installer.wizard import WizardState


def _install_state(tmp_path: Path) -> WizardState:
    return WizardState.for_install(tmp_path, "https://example.invalid/repo.git", home=tmp_path)


def _reconfigure_state(tmp_path: Path, monkeypatch) -> WizardState:
    import incode_mcp.installer.wizard as wizard

    monkeypatch.setattr(wizard.accelerator, "prepared_accelerator", lambda directory: None)
    return WizardState.for_reconfigure(tmp_path, home=tmp_path)


@pytest.mark.asyncio
async def test_install_wizard_walks_forward_and_back(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        assert app.current == "welcome"
        await pilot.click("#next")
        assert app.current == "location"
        await pilot.click("#next")
        assert app.current == "accelerator"
        await pilot.click("#back")
        assert app.current == "location"
        assert app.return_code == 130


@pytest.mark.asyncio
async def test_reconfigure_skips_the_location_panel(tmp_path: Path, monkeypatch) -> None:
    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test() as pilot:
        await pilot.click("#next")
        assert app.current == "accelerator"


@pytest.mark.asyncio
async def test_cancel_exits_with_130(tmp_path: Path) -> None:
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        await pilot.click("#cancel")
    assert app.return_code == 130
```

Run: `uv run pytest tests/test_installer_tui.py -q` — expect FAIL (module missing).

- [ ] **Step 2: Create `tui/__init__.py` and `tui/app.py`**

`src/incode_mcp/installer/tui/__init__.py`:

```python
"""The Textual installer wizard. Imported lazily; Textual is optional."""
```

`src/incode_mcp/installer/tui/app.py`:

```python
"""The Textual installer wizard application."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, ContentSwitcher, Footer, Header

from ..orchestrator import InstallResult
from ..wizard import WizardState
from .panels import (
    AcceleratorPanel,
    DonePanel,
    HarnessesPanel,
    LocationPanel,
    ProgressPanel,
    SettingsPanel,
    SummaryPanel,
    WelcomePanel,
)

PANEL_ORDER = (
    "welcome",
    "location",
    "accelerator",
    "harnesses",
    "indexing",
    "embedding",
    "summary",
    "progress",
    "done",
)


class InstallerApp(App[None]):
    TITLE = "Code Indexing MCP Installer"
    CSS = """
    #screens { height: 1fr; }
    .panel { padding: 1 2; height: 1fr; overflow-y: auto; }
    .help { color: $text-muted; }
    .field { height: auto; margin-bottom: 1; }
    .error { color: $error; }
    #nav { height: auto; dock: bottom; padding: 0 1; }
    #nav Button { margin: 0 1; }
    """

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self.state = state
        self.return_code = 130

    def compose(self) -> ComposeResult:
        yield Header()
        with ContentSwitcher(id="screens", initial="welcome"):
            yield WelcomePanel(self.state, id="welcome")
            yield LocationPanel(self.state, id="location")
            yield AcceleratorPanel(self.state, id="accelerator")
            yield HarnessesPanel(self.state, id="harnesses")
            yield SettingsPanel(self.state, "Indexing", id="indexing")
            yield SettingsPanel(self.state, "Embedding", id="embedding")
            yield SummaryPanel(self.state, id="summary")
            yield ProgressPanel(self.state, id="progress")
            yield DonePanel(id="done")
        with Horizontal(id="nav"):
            yield Button("Back", id="back", disabled=True)
            yield Button("Next", id="next", variant="primary")
            yield Button("Cancel", id="cancel", variant="error")
        yield Footer()

    def _order(self) -> tuple[str, ...]:
        if self.state.mode == "reconfigure":
            return tuple(panel for panel in PANEL_ORDER if panel != "location")
        return PANEL_ORDER

    @property
    def current(self) -> str:
        return str(self.query_one("#screens", ContentSwitcher).current)

    def show_panel(self, name: str) -> None:
        self.query_one("#screens", ContentSwitcher).current = name
        order = self._order()
        index = order.index(name)
        locked = name in {"progress", "done"}
        self.query_one("#back", Button).disabled = locked or index == 0
        next_button = self.query_one("#next", Button)
        next_button.disabled = locked
        next_button.label = "Confirm" if name == "summary" else "Next"
        self.query_one("#cancel", Button).disabled = locked
        panel = self.query_one(f"#{name}")
        became_visible = getattr(panel, "on_became_visible", None)
        if became_visible is not None:
            became_visible()

    def advance(self) -> None:
        panel = self.query_one(f"#{self.current}")
        commit = getattr(panel, "commit", None)
        if commit is not None and not commit():
            return  # validation failed; the panel displayed the reason
        if self.current == "summary":
            self.show_panel("progress")
            self.query_one("#progress", ProgressPanel).start()
            return
        self.show_panel(self._order()[self._order().index(self.current) + 1])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "next":
            self.advance()
        elif button == "back":
            order = self._order()
            self.show_panel(order[order.index(self.current) - 1])
        elif button == "cancel":
            self.exit()
        elif button == "exit":
            self.exit()

    def finish(
        self,
        result: InstallResult | None,
        *,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        if cancelled:
            self.return_code = 130
        elif error is not None or result is None or result.failures:
            self.return_code = 1
        else:
            self.return_code = 0
        self.query_one("#done", DonePanel).show_result(result, error=error, cancelled=cancelled)
        self.show_panel("done")
```

- [ ] **Step 3: Create placeholder panels with final signatures**

`src/incode_mcp/installer/tui/panels.py` (later tasks replace the placeholder bodies one panel at a time):

```python
"""Wizard panels. Each panel owns its widgets and a commit() into WizardState."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Label, RichLog

from ..orchestrator import InstallResult
from ..wizard import WizardState


class WelcomePanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Welcome")


class LocationPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Location")

    def commit(self) -> bool:
        return True


class AcceleratorPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Accelerator")

    def commit(self) -> bool:
        return True


class HarnessesPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Harnesses")

    def commit(self) -> bool:
        return True


class SummaryPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Summary")


class ProgressPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.cancelled = False

    def compose(self) -> ComposeResult:
        yield Label("Progress")
        yield RichLog(id="progress-log")
        yield Button("Cancel", id="progress-cancel", variant="error")

    def start(self) -> None:
        raise NotImplementedError


class DonePanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Label(id="done-title")
        yield Label("", id="done-body")
        yield Button("Exit", id="exit", variant="primary")

    def show_result(
        self,
        result: InstallResult | None,
        *,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        raise NotImplementedError
```

`src/incode_mcp/installer/tui/settings_form.py` (filled in Task 11):

```python
"""Spec-driven settings forms for the wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical

from ...settings_spec import Setting
from ...wizard import WizardState


class SettingField(Vertical):
    """One labelled input for a catalog setting."""

    def __init__(self, setting: Setting, value: str = "") -> None:
        super().__init__(classes="field")
        self.setting = setting
        self.initial = value

    def compose(self) -> ComposeResult:
        yield from ()


class SettingsPanel(Vertical):
    """A group of SettingFields built from the catalog."""

    def __init__(self, state: WizardState, group: str, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.group = group

    def compose(self) -> ComposeResult:
        yield from ()

    def commit(self) -> bool:
        return True
```

Wait — `app.py` imports `SettingsPanel` from `.panels`, but it's defined in `settings_form.py`. Re-export it in `panels.py`: add `from .settings_form import SettingsPanel` at the top of `panels.py` and remove the duplicate definition concern (panels.py does not define SettingsPanel itself). Keep the import list in `app.py` unchanged.

- [ ] **Step 4: Run tests; commit**

```bash
uv run pytest tests/test_installer_tui.py -q
uv run ruff check src/incode_mcp/installer/tui tests/test_installer_tui.py
uv run mypy
git add src/incode_mcp/installer/tui tests/test_installer_tui.py
git commit -m "feat: add the wizard app shell and navigation"
```

---

### Task 10: Wizard panels — Welcome, Location, Accelerator, Harnesses

**Files:**
- Modify: `src/incode_mcp/installer/tui/panels.py`
- Test: `tests/test_installer_tui.py` (extend)

**Interfaces:**
- Consumes: `accelerator.ACCELERATOR_CHOICES`, `accelerator.detection_report` (Task 5), `harnesses.HARNESS_CHOICES`, `harnesses.configuration_path`, `harnesses.read_server_entry`, `harnesses.skill_directory` (Tasks 1, 3).
- Produces: four finished panels whose `commit()` writes into `WizardState`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_installer_tui.py`:

```python
@pytest.mark.asyncio
async def test_location_commit_updates_state(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await pilot.click("#next")
        target = tmp_path / "custom"
        app.query_one("#install-dir", Input).value = str(target)
        await pilot.click("#next")
        assert state.install_directory == target


@pytest.mark.asyncio
async def test_location_rejects_an_empty_directory(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await pilot.click("#next")
        app.query_one("#install-dir", Input).value = "   "
        await pilot.click("#next")
        assert app.current == "location"  # blocked


@pytest.mark.asyncio
async def test_accelerator_panel_shows_detection_and_commits_choice(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        await pilot.click("#next")
        await pilot.click("#next")
        assert app.current == "accelerator"
        text = str(app.query_one("#detection", Static).render())
        assert "Platform:" in text
        app.query_one("#accel-cpu", RadioButton).toggle()
        await pilot.click("#next")
        assert state.accelerator == "cpu"


@pytest.mark.asyncio
async def test_reconfigure_offers_keep_prepared_backend(tmp_path: Path, monkeypatch) -> None:
    app = InstallerApp(_reconfigure_state(tmp_path, monkeypatch))
    async with app.run_test() as pilot:
        await pilot.click("#next")
        assert app.current == "accelerator"
        assert app.query_one("#accel-keep", RadioButton).value is True
        await pilot.click("#next")
        assert app.state.accelerator is None


@pytest.mark.asyncio
async def test_harnesses_panel_commits_checked_slugs(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    state.harness_slugs = []
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(3):
            await pilot.click("#next")
        assert app.current == "harnesses"
        app.query_one("#harness-kimi-code", Checkbox).toggle()
        app.query_one("#harness-codex", Checkbox).toggle()
        await pilot.click("#next")
        assert state.harness_slugs == ["codex", "kimi-code"]
```

Add the new widget imports to the test file's import block: `from textual.widgets import Checkbox, Input, RadioButton, Static`.

Run: `uv run pytest tests/test_installer_tui.py -q` — expect FAIL on the new tests.

- [ ] **Step 2: Implement the four panels**

Replace the four placeholder panel classes in `panels.py` (keep the other placeholders; new imports at top: `from textual.widgets import Checkbox, Collapsible, Input, RadioButton, RadioSet, Static`, plus `from .. import accelerator as accelerator_module`, `from .. import harnesses`, `from pathlib import Path`):

```python
class WelcomePanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        if self.state.mode == "reconfigure":
            headline = "Reconfigure Code Indexing MCP"
            detail = (
                f"Installation: {self.state.install_directory}\n"
                "Your current settings were read from the configured harnesses. "
                "Walk through the sections and confirm on the summary screen."
            )
        else:
            headline = "Install Code Indexing MCP"
            detail = (
                f"Installation: {self.state.install_directory}\n"
                "This wizard prepares the accelerator, configures your MCP clients, "
                "and lets you customize the server's settings."
            )
        yield Label(headline, id="welcome-headline")
        yield Static(detail)
        if self.state.disagreements:
            names = ", ".join(self.state.disagreements)
            yield Static(
                f"Your harnesses disagree on: {names}. The value from the earliest "
                "configured harness in the list is prefilled; confirming unifies them.",
                classes="help",
            )


class LocationPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Install location")
        yield Static(
            "Where the repository is cloned. Changing this moves only the checkout; "
            "indexes and caches live in the data directory (Indexing section).",
            classes="help",
        )
        with Collapsible(title="Advanced", collapsed=True):
            yield Label("Install directory")
            yield Input(value=str(self.state.install_directory), id="install-dir")
            yield Label("Repository URL")
            yield Input(value=self.state.repo_url, id="repo-url")
        yield Label("", id="location-error", classes="error")

    def commit(self) -> bool:
        directory = self.query_one("#install-dir", Input).value.strip()
        if not directory:
            self.query_one("#location-error", Label).update("Install directory cannot be empty.")
            return False
        self.state.install_directory = Path(directory).expanduser()
        self.state.repo_url = self.query_one("#repo-url", Input).value.strip()
        return True


class AcceleratorPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Passage embedding accelerator")
        yield Static(
            "auto detects a supported GPU and prepares it; anything that cannot be "
            "detected, built, or probed leaves the installation on CPU and says why.",
            classes="help",
        )
        yield Static("\n".join(accelerator_module.detection_report()), id="detection")
        with RadioSet(id="accel-choices"):
            if self.state.mode == "reconfigure":
                prepared = self.state.prepared_accelerator or "none prepared"
                yield RadioButton(f"Keep the prepared backend ({prepared})", id="accel-keep", value=True)
            for choice in accelerator_module.ACCELERATOR_CHOICES:
                yield RadioButton(
                    "auto (recommended)" if choice == "auto" else choice,
                    id=f"accel-{choice}",
                    value=self.state.accelerator == choice,
                )
        yield Static(
            "Preparing an accelerator downloads the embedding model and can take several "
            "minutes and a few gigabytes; a matching record is reused next time.",
            classes="help",
        )

    def commit(self) -> bool:
        if self.state.mode == "reconfigure" and self.query_one("#accel-keep", RadioButton).value:
            self.state.accelerator = None
            return True
        for choice in accelerator_module.ACCELERATOR_CHOICES:
            if self.query_one(f"#accel-{choice}", RadioButton).value:
                self.state.accelerator = choice
                return True
        self.state.accelerator = "auto"
        return True


class HarnessesPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("MCP clients to configure")
        yield Static(
            "The server entry is merged into each selected client's user-wide "
            "configuration; existing files are backed up with a .bak suffix first.",
            classes="help",
        )
        for choice in harnesses.HARNESS_CHOICES:
            path = harnesses.configuration_path(choice.slug)
            existing = harnesses.read_server_entry(choice.slug) is not None
            skills = harnesses.skill_directory(choice.slug) is not None
            notes = [str(path)]
            if existing:
                notes.append("already configured")
            if skills:
                notes.append("skills supported")
            yield Checkbox(
                f"{choice.label} — {', '.join(notes)}",
                value=choice.slug in self.state.harness_slugs,
                id=f"harness-{choice.slug}",
            )

    def commit(self) -> bool:
        self.state.harness_slugs = [
            choice.slug
            for choice in harnesses.HARNESS_CHOICES
            if self.query_one(f"#harness-{choice.slug}", Checkbox).value
        ]
        return True
```

- [ ] **Step 3: Run tests; commit**

```bash
uv run pytest tests/test_installer_tui.py -q
uv run ruff check src/incode_mcp/installer/tui tests/test_installer_tui.py
uv run mypy
git add src/incode_mcp/installer/tui/panels.py tests/test_installer_tui.py
git commit -m "feat: add the wizard's welcome, location, accelerator, and harness panels"
```

---

### Task 11: Settings forms and the summary panel

**Files:**
- Modify: `src/incode_mcp/installer/tui/settings_form.py`
- Modify: `src/incode_mcp/installer/tui/panels.py` (`SummaryPanel`)
- Test: `tests/test_installer_tui.py` (extend)

**Interfaces:**
- Consumes: `settings_spec.SETTINGS`, `default_value`, `validate`, `normalize` (Task 2), `harnesses.configuration_path` (Task 1), `accelerator.ACCELERATOR_EXTRAS` (Task 1).
- Produces: `SettingField.value() -> str`, `SettingsPanel.commit() -> bool` (validates all fields, writes `state.values`), `SummaryPanel.on_became_visible()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_installer_tui.py`:

```python
@pytest.mark.asyncio
async def test_settings_panels_validate_and_commit(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(4):
            await pilot.click("#next")
        assert app.current == "indexing"
        field = app.query_one("#f-INCODE_INDEX_WAIT_SECONDS", Input)
        field.value = "99999999"
        await pilot.click("#next")
        assert app.current == "indexing"  # blocked by validation
        field.value = "60"
        await pilot.click("#next")
        assert app.current == "embedding"
        assert state.values["INCODE_INDEX_WAIT_SECONDS"] == "60"


@pytest.mark.asyncio
async def test_summary_lists_updates_and_target_files(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    state.values["INCODE_OFFLINE"] = "1"
    state.harness_slugs = ["kimi-code"]
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(6):
            await pilot.click("#next")
        assert app.current == "summary"
        text = str(app.query_one("#summary-body", Static).render())
        assert "INCODE_OFFLINE" in text
        assert "mcp.json" in text
        assert "auto" in text  # accelerator choice


@pytest.mark.asyncio
async def test_summary_warns_about_accelerator_disk_cost(tmp_path: Path) -> None:
    state = _install_state(tmp_path)
    state.accelerator = "mlx"
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(6):
            await pilot.click("#next")
        text = str(app.query_one("#summary-body", Static).render())
        assert "gigabytes" in text
```

Run the new tests — expect FAIL.

- [ ] **Step 2: Implement `settings_form.py`**

```python
"""Spec-driven settings forms for the wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Checkbox, Input, Label, Select, Static

from ...settings_spec import SETTINGS, Setting, default_value, validate
from ...wizard import WizardState


class SettingField(Vertical):
    """One labelled input for a catalog setting, generated from its spec."""

    def __init__(self, setting: Setting, value: str = "") -> None:
        super().__init__(classes="field")
        self.setting = setting
        self.initial = value

    def compose(self) -> ComposeResult:
        widget_id = f"f-{self.setting.name}"
        if self.setting.type == "bool":
            yield Checkbox(
                self.setting.label,
                value=(self.initial or self.setting.default) == "1",
                id=widget_id,
            )
        elif self.setting.type == "choice":
            options = [(choice, choice) for choice in self.setting.choices]
            yield Select(
                options,
                value=self.initial or self.setting.default,
                id=widget_id,
                allow_blank=False,
            )
        else:
            yield Label(self.setting.label)
            yield Input(
                value=self.initial,
                placeholder=default_value(self.setting),
                id=widget_id,
            )
        yield Static(self.setting.help, classes="help")

    def value(self) -> str:
        widget = self.query_one(f"#f-{self.setting.name}")
        if isinstance(widget, Checkbox):
            return "1" if widget.value else "0"
        if isinstance(widget, Select):
            return str(widget.value)
        text = widget.value.strip()
        return text or default_value(self.setting)

    def raw_input(self) -> str:
        """The typed text for Input fields ("" means 'use the default')."""

        widget = self.query_one(f"#f-{self.setting.name}")
        if isinstance(widget, Input):
            return widget.value.strip()
        return self.value()


class SettingsPanel(Vertical):
    """A group of SettingFields built from the catalog."""

    def __init__(self, state: WizardState, group: str, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.group = group

    def compose(self) -> ComposeResult:
        yield Label(f"{self.group} settings")
        yield Static(
            "Fields left empty keep their default and are not written to any config.",
            classes="help",
        )
        for setting in SETTINGS:
            if setting.group == self.group:
                yield SettingField(setting, self.state.field_value(setting.name))
        yield Label("", id=f"{self.group.lower()}-error", classes="error")

    def commit(self) -> bool:
        error_label = self.query_one(f"#{self.group.lower()}-error", Label)
        for field in self.query(SettingField):
            raw = field.raw_input()
            if raw:  # empty means default; defaults are valid by construction
                error = validate(field.setting, raw)
                if error is not None:
                    error_label.update(error)
                    return False
            self.state.set_field(field.setting.name, raw)
        error_label.update("")
        return True
```

Careful: `set_field` stores the *raw* value, and `WizardState.env_updates()` treats empty as default — so committing an empty input stores `""`, which is correct.

- [ ] **Step 3: Implement `SummaryPanel` in `panels.py`**

Replace the `SummaryPanel` placeholder:

```python
class SummaryPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("Summary")
        yield Static("", id="summary-body")
        yield Static("Confirm to run the installation.", classes="help")

    def on_became_visible(self) -> None:
        lines = [f"Install directory: {self.state.install_directory}"]
        if self.state.accelerator is None:
            prepared = self.state.prepared_accelerator or "none"
            lines.append(f"Accelerator: keep the prepared backend ({prepared})")
        else:
            lines.append(f"Accelerator: {self.state.accelerator}")
            if self.state.accelerator in accelerator_module.ACCELERATOR_EXTRAS:
                lines.append(
                    "  Building this environment downloads several gigabytes and runs "
                    "a real inference probe."
                )
        lines.append("Harnesses: " + (", ".join(self.state.harness_slugs) or "none"))
        updates = self.state.env_updates()
        if updates:
            lines.append("Settings:")
            for name, value in sorted(updates.items()):
                lines.append(f"  {name} = {value if value is not None else '(removed)'}")
        else:
            lines.append("Settings: all defaults (nothing written to env blocks)")
        if self.state.harness_slugs:
            lines.append("Files that will be written:")
            for slug in self.state.harness_slugs:
                lines.append(f"  {harnesses.configuration_path(slug)}")
            if self.state.accelerator is not None:
                lines.append("  the accelerator record in the server's data directory")
        if self.state.disagreements:
            lines.append(
                "Harnesses that disagreed (" + ", ".join(self.state.disagreements)
                + ") will be unified on the values above."
            )
        self.query_one("#summary-body", Static).update("\n".join(lines))
```

`configuration_path` can raise `InstallerError` for claude-desktop on an unsupported platform — the Harnesses panel would have raised at render already; if you want the summary robust on such platforms, wrap that loop body in try/except and render `str(exc)`. (Do the same in `HarnessesPanel.compose` if the panel is ever constructed on such a platform; out of scope for the tested paths.)

- [ ] **Step 4: Run tests; commit**

```bash
uv run pytest tests/test_installer_tui.py -q
uv run ruff check src/incode_mcp/installer/tui tests/test_installer_tui.py
uv run mypy
git add src/incode_mcp/installer/tui tests/test_installer_tui.py
git commit -m "feat: add spec-driven settings forms and the summary panel"
```

---

### Task 12: Progress and Done panels

**Files:**
- Modify: `src/incode_mcp/installer/tui/panels.py` (`ProgressPanel`, `DonePanel`)
- Test: `tests/test_installer_tui.py` (extend)

**Interfaces:**
- Consumes: `orchestrator.run_install`, `StepEvent`, `InstallResult` (Task 4); `InstallerApp.finish` (Task 9).
- Produces: `ProgressPanel.start()`, `DonePanel.show_result(...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_installer_tui.py`:

```python
def _fake_result(failures: tuple = ()) -> "InstallResult":
    from incode_mcp.installer.orchestrator import InstallResult
    from incode_mcp.installer.accelerator import AcceleratorPlan

    return InstallResult(
        AcceleratorPlan("cpu", "CPU was requested"),
        (("kimi-code", Path("/home/u/.kimi-code/mcp.json")),),
        failures,
        (("kimi-code", "2 linked, 2 already installed"),),
    )


@pytest.mark.asyncio
async def test_progress_runs_pipeline_and_finishes_on_done(tmp_path: Path, monkeypatch) -> None:
    import incode_mcp.installer.tui.panels as panels

    def fake_run_install(plan, on_event=lambda event: None, should_continue=lambda: True):
        on_event(StepEvent("accelerator", "started", "auto"))
        on_event(StepEvent("accelerator", "finished", "cpu (ok)"))
        return _fake_result()

    monkeypatch.setattr(panels, "run_install", fake_run_install)
    state = _install_state(tmp_path)
    app = InstallerApp(state)
    async with app.run_test() as pilot:
        for _ in range(7):
            await pilot.click("#next")
        await pilot.pause()
        assert app.current == "done"
        assert app.return_code == 0
        body = str(app.query_one("#done-body", Static).render())
        assert "mcp.json" in body


@pytest.mark.asyncio
async def test_done_reports_failures_with_exit_1(tmp_path: Path, monkeypatch) -> None:
    import incode_mcp.installer.tui.panels as panels

    monkeypatch.setattr(
        panels, "run_install",
        lambda plan, on_event=None, should_continue=None: _fake_result((("codex", "broken"),)),
    )
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        for _ in range(7):
            await pilot.click("#next")
        await pilot.pause()
        assert app.return_code == 1
        assert "codex" in str(app.query_one("#done-body", Static).render())


@pytest.mark.asyncio
async def test_pipeline_error_finishes_with_exit_1(tmp_path: Path, monkeypatch) -> None:
    import incode_mcp.installer.tui.panels as panels
    from incode_mcp.installer.config_files import InstallerError

    def explode(plan, on_event=None, should_continue=None):
        raise InstallerError("boom")

    monkeypatch.setattr(panels, "run_install", explode)
    app = InstallerApp(_install_state(tmp_path))
    async with app.run_test() as pilot:
        for _ in range(7):
            await pilot.click("#next")
        await pilot.pause()
        assert app.return_code == 1
        assert "boom" in str(app.query_one("#done-body", Static).render())
```

Add `from incode_mcp.installer.orchestrator import StepEvent` to the test imports.

Run — expect FAIL (`ProgressPanel.start` is `NotImplementedError`).

- [ ] **Step 2: Implement `ProgressPanel` and `DonePanel`**

Replace both placeholders in `panels.py`. New imports at top: `from textual import work`, `from textual.widgets import Static` (already there), `from ..orchestrator import StepEvent, run_install`.

```python
class ProgressPanel(Vertical):
    def __init__(self, state: WizardState, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="panel")
        self.state = state
        self.cancelled = False

    def compose(self) -> ComposeResult:
        yield Label("Running the installation")
        yield Static(
            "The accelerator environment build and its probe can take several minutes.",
            classes="help",
        )
        yield RichLog(id="progress-log")
        yield Button("Cancel", id="progress-cancel", variant="error")

    def start(self) -> None:
        self.cancelled = False
        self.query_one("#progress-cancel", Button).disabled = False
        self._run_pipeline()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "progress-cancel":
            self.cancelled = True
            event.button.disabled = True
            self._log_line("cancel requested; stopping after the current step")
            event.stop()

    def _log_line(self, line: str) -> None:
        self.query_one("#progress-log", RichLog).write(line)

    @work(thread=True)
    def _run_pipeline(self) -> None:
        result: InstallResult | None = None
        error: Exception | None = None
        try:
            result = run_install(
                self.state.to_plan(),
                on_event=lambda event: self.app.call_from_thread(
                    self._log_line,
                    f"[{event.step}] {event.status}: {event.detail}",
                ),
                should_continue=lambda: not self.cancelled,
            )
        except Exception as exc:  # surfaced on the Done screen
            error = exc
        self.app.call_from_thread(
            self.app.finish, result, error=error, cancelled=self.cancelled
        )


class DonePanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Label(id="done-title")
        yield Static("", id="done-body")
        yield Button("Exit", id="exit", variant="primary")

    def show_result(
        self,
        result: InstallResult | None,
        *,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        lines: list[str] = []
        if cancelled:
            title = "Installation cancelled"
            lines.append("Stopped between steps; anything already written above still applies.")
        elif error is not None:
            title = "Installation failed"
            lines.append(str(error))
        elif result is None:
            title = "Installation failed"
            lines.append("No result was produced.")
        else:
            title = "Installation complete"
            if result.accelerator_plan is not None:
                plan = result.accelerator_plan
                marker = "" if plan.honored else " (fell back to CPU)"
                lines.append(f"Accelerator: {plan.accelerator}{marker}\n  {plan.reason}")
            for slug, path in result.configured:
                lines.append(f"Configured {slug}: {path}")
            for slug, message in result.failures:
                lines.append(f"FAILED {slug}: {message}")
            for slug, message in result.skills:
                lines.append(f"Skills for {slug}: {message}")
            if result.failures:
                title = "Installation complete with failures"
        lines.append("")
        lines.append("Restart configured clients to load the MCP server.")
        lines.append("Reconfigure later with: code-indexing-mcp configure")
        self.query_one("#done-title", Label).update(title)
        self.query_one("#done-body", Static).update("\n".join(lines))
```

- [ ] **Step 3: Run the full TUI suite; commit**

```bash
uv run pytest tests/test_installer_tui.py -q
uv run ruff check src/incode_mcp/installer/tui tests/test_installer_tui.py
uv run mypy
git add src/incode_mcp/installer/tui/panels.py tests/test_installer_tui.py
git commit -m "feat: add the progress and done panels"
```

---

### Task 13: Rewrite `install.py` as the bootstrap

The old installer internals (everything moved in Task 1) are deleted. What remains is stdlib-only: argument parsing, clone/update, sync with both extras, TTY/TERM detection, and delegation to the module CLI. The six bootstrap-surface tests kept in Task 1 are rewritten here.

**Files:**
- Rewrite: `install.py`
- Modify: `tests/test_installer.py` (rewrite the six bootstrap-surface tests; add delegation tests)

**Interfaces:**
- Consumes: `incode_mcp.installer` module CLI (Task 6) via subprocess.
- Produces: `install.py` surface: `main(argv) -> int`, `clone_or_update_repository`, `sync_environment` (now syncs `--extra cpu --extra tui`), `server_executable`, `environment_python`, `tui_available() -> bool`, `build_argument_parser()`.

- [ ] **Step 1: Rewrite `install.py`**

Full new content:

```python
#!/usr/bin/env python3
"""Bootstrap installer for Code Indexing MCP.

This file is stdlib-only and self-contained: install.sh downloads it into a
temporary directory and runs it before any virtual environment exists. It
clones or updates the repository, builds the locked environment, and delegates
everything else to ``python -m incode_mcp.installer`` inside that environment.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

DEFAULT_REPOSITORY_URL = "https://github.com/MarcinHamiga/code-indexing-mcp.git"
# The serving environment always gets the CPU extra (it is the fallback every
# accelerator degrades to) plus the TUI extra for the interactive wizard.
SERVING_EXTRAS = ("cpu", "tui")
ACCELERATOR_CHOICES = ("auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml")


class InstallerError(RuntimeError):
    """An actionable installer failure."""


def _run_command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=None if environment is None else {**os.environ, **environment},
        )
    except FileNotFoundError as exc:
        raise InstallerError(f"Required command was not found: {arguments[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        command = " ".join(arguments)
        message = f"Command failed: {command}"
        if detail:
            message = f"{message}\n{detail}"
        raise InstallerError(message) from exc


def _canonical_repository_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        return f"github.com/{value.removeprefix('git@github.com:').lower()}"
    for prefix in ("https://github.com/", "http://github.com/", "ssh://git@github.com/"):
        if value.startswith(prefix):
            return f"github.com/{value.removeprefix(prefix).lower()}"
    if "://" not in value:
        return str(Path(value).expanduser().resolve())
    return value


def clone_or_update_repository(repository_url: str, install_directory: Path) -> str:
    """Clone a fresh checkout or fast-forward an existing clean checkout."""

    git = shutil.which("git")
    if git is None:
        raise InstallerError("Git is required but was not found in PATH")
    install_directory = install_directory.expanduser().resolve()

    if not install_directory.exists():
        install_directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _run_command([git, "clone", "--", repository_url, str(install_directory)])
        return "installed"

    if not (install_directory / ".git").exists():
        raise InstallerError(
            f"Install target exists but is not a Git repository: {install_directory}"
        )

    origin = _run_command(
        [git, "remote", "get-url", "origin"],
        cwd=install_directory,
    ).stdout.strip()
    if _canonical_repository_url(origin) != _canonical_repository_url(repository_url):
        raise InstallerError(
            "Existing checkout origin does not match the requested repository: "
            f"{origin} != {repository_url}"
        )

    status = _run_command(
        [git, "status", "--porcelain"],
        cwd=install_directory,
    ).stdout
    if status.strip():
        raise InstallerError(
            f"Existing checkout has uncommitted changes; update it manually: {install_directory}"
        )

    _run_command([git, "pull", "--ff-only"], cwd=install_directory)
    return "updated"


def server_executable(
    install_directory: Path,
    *,
    platform_name: str | None = None,
) -> Path:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return install_directory / ".venv" / "Scripts" / "code-indexing-mcp.exe"
    return install_directory / ".venv" / "bin" / "code-indexing-mcp"


def environment_python(directory: Path, *, platform_name: str | None = None) -> Path:
    """Return the interpreter inside a virtual environment directory."""

    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return directory / "Scripts" / "python.exe"
    return directory / "bin" / "python"


def _uv_executable(uv_executable: str | None) -> str:
    uv = uv_executable or shutil.which("uv")
    if uv is None:
        raise InstallerError(
            "uv is required but was not found in PATH. Install it from https://docs.astral.sh/uv/"
        )
    return uv


def sync_environment(
    install_directory: Path,
    *,
    uv_executable: str | None = None,
    platform_name: str | None = None,
) -> Path:
    """Create or refresh the locked virtual environment and return its server command."""

    uv = _uv_executable(uv_executable)
    command = [uv, "sync", "--locked"]
    for extra in SERVING_EXTRAS:
        command += ["--extra", extra]
    _run_command(command, cwd=install_directory)
    command_path = server_executable(install_directory, platform_name=platform_name)
    if not command_path.is_file():
        raise InstallerError(f"uv sync completed but the MCP executable is missing: {command_path}")
    return command_path


def _default_install_directory() -> Path:
    configured = os.environ.get("CODE_INDEXING_MCP_INSTALL_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "code-indexing-mcp"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install or update Code Indexing MCP and configure it for selected MCP harnesses."
        )
    )
    parser.add_argument(
        "--install-dir",
        default=str(_default_install_directory()),
        help="checkout location (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-url",
        default=os.environ.get("CODE_INDEXING_MCP_REPO_URL", DEFAULT_REPOSITORY_URL),
        help="Git repository to clone or update (default: %(default)s)",
    )
    parser.add_argument(
        "--accelerator",
        choices=ACCELERATOR_CHOICES,
        default=os.environ.get("CODE_INDEXING_MCP_ACCELERATOR", "auto"),
        help=(
            "which accelerator to prepare for passage indexing (default: %(default)s). "
            "auto detects one; anything that cannot be detected, built, or probed "
            "falls back to CPU with the reason reported"
        ),
    )
    parser.add_argument(
        "--harnesses",
        help=(
            "comma-separated harness numbers/slugs or 'all'; omit for the interactive menu "
            "(codex, claude-code, kimi-code, claude-desktop, opencode, kilocode)"
        ),
    )
    parser.add_argument(
        "--set", dest="settings", action="append", default=[], metavar="NAME=VALUE",
        help="set a managed INCODE_* value in harness configs; repeatable",
    )
    parser.add_argument(
        "--unset", dest="unsets", action="append", default=[], metavar="NAME",
        help="remove a managed INCODE_* value from harness configs; repeatable",
    )
    parser.add_argument(
        "--tui", action="store_true", help="force the interactive wizard"
    )
    parser.add_argument(
        "--no-tui", action="store_true", help="force the plain text interface"
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="never prompt; a missing harness selection configures none",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=os.environ.get("INCODE_OFFLINE", "").lower() in {"1", "true", "yes"},
        help="never download the embedding model",
    )
    return parser


def tui_available() -> bool:
    """True when the terminal can host the Textual wizard."""

    term = os.environ.get("TERM", "")
    return sys.stdin.isatty() and sys.stdout.isatty() and bool(term) and term != "dumb"


def _delegate(install_directory: Path, tail: list[str]) -> int:
    python = environment_python(install_directory / ".venv")
    try:
        completed = subprocess.run(
            [str(python), "-m", "incode_mcp.installer", *tail],
            cwd=install_directory,
        )
    except OSError as exc:
        print(f"Error: could not launch the installer module: {exc}", file=sys.stderr)
        return 1
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    install_directory = Path(arguments.install_dir).expanduser().resolve()

    try:
        action = clone_or_update_repository(arguments.repo_url, install_directory)
        print(f"{action.title()} repository: {install_directory}")
        command = sync_environment(install_directory)
        print(f"Prepared MCP executable: {command}")
    except InstallerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Installation cancelled.", file=sys.stderr)
        return 130

    tail = ["--install-dir", str(install_directory), "--accelerator", arguments.accelerator]
    if arguments.harnesses is not None:
        tail += ["--harnesses", arguments.harnesses]
    for pair in arguments.settings:
        tail += ["--set", pair]
    for name in arguments.unsets:
        tail += ["--unset", name]
    if arguments.offline:
        tail.append("--offline")
    if arguments.no_prompt:
        tail.append("--no-prompt")
    use_tui = arguments.tui or (not arguments.no_tui and tui_available())
    if use_tui:
        tail.append("--tui")

    returncode = _delegate(install_directory, tail)
    if use_tui and returncode not in (0, 1, 130):
        print(
            "The interactive installer failed; re-run with --no-tui for the plain interface.",
            file=sys.stderr,
        )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Rewrite the six bootstrap-surface tests**

In `tests/test_installer.py`, the six tests kept from Task 1 exercise behavior that changed. Rewrite them:

`test_repository_is_cloned_then_fast_forwarded_on_update` and `test_repository_update_rejects_non_repo_dirty_and_mismatched_targets` — keep their bodies; they pass unchanged against the new `install.py` (`clone_or_update_repository` is verbatim). Only `test_sync_environment_runs_locked_sync_and_finds_server` needs an updated expectation, and the three `main` tests need rewrites:

```python
def test_sync_environment_runs_locked_sync_and_finds_server(tmp_path: Path, monkeypatch) -> None:
    installer = load_installer()
    commands = []

    def fake_run(arguments, **kwargs):
        commands.append(arguments)
        server = installer.server_executable(tmp_path)
        server.parent.mkdir(parents=True, exist_ok=True)
        server.write_text("")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(installer, "_run_command", fake_run)
    result = installer.sync_environment(tmp_path, uv_executable="/usr/bin/uv")
    assert commands == [["/usr/bin/uv", "sync", "--locked", "--extra", "cpu", "--extra", "tui"]]
    assert result == installer.server_executable(tmp_path)
```

(Adjust to the file's existing fixture conventions — the original test already monkeypatches `_run_command`; only the expected command changes to include `"--extra", "tui"`.)

Replace the three `main` tests with delegation tests:

```python
def test_main_delegates_to_the_module_cli_with_forwarded_flags(tmp_path: Path, monkeypatch) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "installed")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: False)
    delegated = []
    monkeypatch.setattr(
        installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0
    )

    code = installer.main([
        "--install-dir", str(tmp_path),
        "--accelerator", "mlx",
        "--harnesses", "kimi-code",
        "--set", "INCODE_OFFLINE=1",
        "--unset", "INCODE_BROKER",
        "--offline",
    ])

    assert code == 0
    (tail,) = delegated
    assert tail[:4] == ["--install-dir", str(tmp_path), "--accelerator", "mlx"]
    for fragment in (
        ["--harnesses", "kimi-code"],
        ["--set", "INCODE_OFFLINE=1"],
        ["--unset", "INCODE_BROKER"],
        ["--offline"],
    ):
        assert any(tail[index : index + len(fragment)] == fragment for index in range(len(tail)))
    assert "--tui" not in tail


def test_main_adds_tui_flag_on_a_capable_terminal(tmp_path: Path, monkeypatch) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "updated")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: True)
    delegated = []
    monkeypatch.setattr(
        installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0
    )

    assert installer.main(["--install-dir", str(tmp_path)]) == 0
    assert "--tui" in delegated[0]


def test_main_no_tui_flag_suppresses_the_wizard(tmp_path: Path, monkeypatch) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer, "clone_or_update_repository", lambda url, directory: "updated")
    monkeypatch.setattr(installer, "sync_environment", lambda directory: tmp_path / "server")
    monkeypatch.setattr(installer, "tui_available", lambda: True)
    delegated = []
    monkeypatch.setattr(
        installer, "_delegate", lambda directory, tail: delegated.append(tail) or 0
    )

    assert installer.main(["--install-dir", str(tmp_path), "--no-tui"]) == 0
    assert "--tui" not in delegated[0]


def test_main_reports_actionable_installer_error(tmp_path: Path, monkeypatch, capsys) -> None:
    installer = load_installer()

    def fail(url, directory):
        raise installer.InstallerError("Git is required but was not found in PATH")

    monkeypatch.setattr(installer, "clone_or_update_repository", fail)
    code = installer.main(["--install-dir", str(tmp_path)])
    assert code == 1
    assert "Git is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("tty", "term", "expected"),
    [(True, "xterm-256color", True), (True, "dumb", False), (True, "", False), (False, "xterm", False)],
)
def test_tui_available_detects_capable_terminals(monkeypatch, tty: bool, term: str, expected: bool) -> None:
    installer = load_installer()
    monkeypatch.setattr(installer.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr(installer.sys.stdout, "isatty", lambda: tty)
    monkeypatch.setenv("TERM", term)
    assert installer.tui_available() is expected
```

Note the existing tests in this file that called `installer.main` with mocked `input_fn`/`output_fn` kwargs are removed by this rewrite — the new `main` no longer takes those kwargs.

- [ ] **Step 3: Run the entire test suite, ruff, mypy**

```bash
uv run pytest tests/test_installer.py tests/test_installer_cli.py -q
uv run pytest -q
uv run ruff check install.py tests/test_installer.py
uv run mypy
```

Expected: everything green. `install.py` is not in mypy's `packages`, so it stays unchecked, as before.

- [ ] **Step 4: Commit**

```bash
git add install.py tests/test_installer.py
git commit -m "feat: rewrite install.py as a delegating bootstrap"
```

---

### Task 14: README and final verification

**Files:**
- Modify: `README.md` (Install section, lines 13–80)

**Interfaces:**
- Consumes: everything.
- Produces: user-facing docs matching the new behavior.

- [ ] **Step 1: Update the Install section**

Replace the paragraphs after the PowerShell block (the lines from "The installer clones the repository…" through "…Run `python3 install.py --help` for all installer options.", currently lines 33–80) with:

~~~~markdown
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
preserved.

Re-run the same command later to update an existing clean checkout with a fast-forward-only
pull and refresh its environment. To change settings or harnesses without updating, run
`code-indexing-mcp configure` — it opens the same wizard offline, prefilled from your
current configuration. Scripted changes work too:
`code-indexing-mcp configure --set INCODE_BROKER=off --unset INCODE_INDEX_MODE`.

On a terminal that cannot host the wizard (or with `--no-tui`), the installer falls back to
a plain text interface with the numbered harness menu. For a fully noninteractive
installation, pass harness slugs and any settings:

```bash
curl -fsSL https://raw.githubusercontent.com/MarcinHamiga/code-indexing-mcp/main/install.sh |
  sh -s -- --harnesses codex,claude-code,opencode --set INCODE_INDEX_MODE=eager
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
~~~~

- [ ] **Step 2: Final verification — full suite, lint, types, and a real smoke test**

```bash
uv run pytest -q
uv run ruff check .
uv run mypy
uv run python -c "import incode_mcp.cli, sys; assert 'textual' not in sys.modules"
uv run python -m incode_mcp.installer --help
uv run code-indexing-mcp configure --help
```

Then one real end-to-end smoke test in a scratch directory (real uv sync, no probe):

```bash
SCRATCH=$(mktemp -d)
python3 install.py --install-dir "$SCRATCH/install" --harnesses "" --accelerator cpu --no-tui --no-prompt
ls "$SCRATCH/install/.venv/bin/code-indexing-mcp"
"$SCRATCH/install/.venv/bin/code-indexing-mcp" configure --no-tui --set INCODE_BROKER=off --harnesses "" 
rm -rf "$SCRATCH"
```

Expected: clone, sync (with the `tui` extra), module CLI runs to exit 0; `configure` runs offline against the scratch install and exits 0. (This step needs network for the clone; run it once locally, not in CI.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe the wizard installer and configure subcommand"
```

---

## Self-Review Notes (already applied)

- **Spec coverage:** every spec section maps to a task — module layout (1), settings catalog (2), env blocks incl. Codex TOML (3), orchestrator (4), reconfigure prefill/disagreement (5), module CLI + `--set`/`--unset` (6), `configure` subcommand (7), `tui` extra (8), TUI screens 1–9 (9–12), bootstrap + delegation + `--no-tui`/TTY detection (13), README (14). The spec's "Progress streams subprocess output" is deliberately refined to step-event streaming (see header); failure output still surfaces through `InstallerError` details, matching today's installer.
- **Type consistency:** `InstallPlan.accelerator: str | None` (None = keep backend) is used identically in orchestrator, wizard, CLI, and panels. `env_updates` values are `str | None` (None = delete) in `parse_settings`, `WizardState.env_updates`, `configure_harness(env=...)`, and `merge_env` — checked across tasks. `InstallerApp.return_code` is written by `finish()` and read by `cli._run_tui`.
- **Known limitation accepted in design:** between Task 1 and Task 13, `install.py` and the package hold duplicate copies of the moved functions. Tests pin the package copies from Task 1 on; the duplicates in `install.py` are deleted in Task 13. No other task edits `install.py` in between.
