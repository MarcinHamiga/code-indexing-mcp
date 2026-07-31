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
        "INCODE_INDEX_MODE",
        "Indexing",
        "Index mode",
        "When projects get indexed: lazy on first use, eager at startup, manual only.",
        "choice",
        "lazy",
        choices=("lazy", "eager", "manual"),
    ),
    Setting(
        "INCODE_INDEX_WAIT_SECONDS",
        "Indexing",
        "Index wait (seconds)",
        "How long a startup index waits out a competing job before failing; 0 disables waiting.",
        "int",
        "300",
        minimum=0,
        maximum=24 * 60 * 60,
    ),
    Setting(
        "INCODE_EMBED_MEMORY_MB",
        "Indexing",
        "Indexing memory (MB)",
        "Ceiling for the indexing worker. The default is 25% of RAM clamped to 1024-2048.",
        "int",
        "",
        minimum=1024,
        maximum=1024 * 1024,
        dynamic_default=_default_memory_mb,
    ),
    Setting(
        "INCODE_VECTOR_INDEX",
        "Indexing",
        "Vector index",
        "exact search, or approximate HNSW indexing.",
        "choice",
        "exact",
        choices=("exact", "hnsw"),
    ),
    Setting(
        "INCODE_INDEX_EXECUTION",
        "Indexing",
        "Index execution",
        "worker enforces the memory ceiling; in-process is a diagnostic rollback.",
        "choice",
        "worker",
        choices=("worker", "in-process"),
    ),
    Setting(
        "INCODE_BROKER",
        "Indexing",
        "Broker",
        "Share one indexing process between clients through the daemon.",
        "choice",
        "auto",
        choices=("auto", "on", "off"),
    ),
    Setting(
        "INCODE_DATA_DIR",
        "Indexing",
        "Data directory",
        "Where the indexes live.",
        "path",
        "",
        dynamic_default=lambda: str(user_data_path("incode")),
    ),
    Setting(
        "INCODE_CACHE_DIR",
        "Indexing",
        "Cache directory",
        "Where the embedding model is cached.",
        "path",
        "",
        dynamic_default=lambda: str(user_cache_path("incode")),
    ),
    Setting(
        "INCODE_OFFLINE",
        "Indexing",
        "Offline mode",
        "Never download the model; fail if it is missing.",
        "bool",
        "0",
    ),
    Setting(
        "INCODE_EMBED_BATCH_SIZE",
        "Embedding",
        "Batch size",
        "Embedding microbatch size; auto resolves to 1 unless calibration raised it.",
        "auto_int",
        "auto",
        minimum=1,
        maximum=256,
    ),
    Setting(
        "INCODE_EMBED_MAX_TOKENS",
        "Embedding",
        "Max tokens",
        "Sequence window per chunk; attention memory is quadratic in tokens.",
        "int",
        "1024",
        minimum=64,
        maximum=8192,
    ),
    Setting(
        "INCODE_EMBED_OVERLAP_TOKENS",
        "Embedding",
        "Overlap tokens",
        "Overlap between consecutive windows of a long chunk.",
        "int",
        "64",
        minimum=0,
        maximum=4096,
    ),
    Setting(
        "INCODE_EMBED_THREADS",
        "Embedding",
        "Threads",
        "CPU inference threads.",
        "int",
        "",
        minimum=1,
        maximum=64,
        dynamic_default=_default_threads,
    ),
    Setting(
        "INCODE_EMBED_CPU_ARENA",
        "Embedding",
        "CPU arena",
        "Preallocate the CPU inference arena.",
        "bool",
        "0",
    ),
    Setting(
        "INCODE_EMBED_CROSSOVER",
        "Embedding",
        "Accelerator crossover",
        "Run size in characters above which starting the accelerator repays its model load.",
        "auto_off_int",
        "auto",
        minimum=0,
        maximum=1024**3,
    ),
    Setting(
        "INCODE_EMBED_CALIBRATE",
        "Embedding",
        "Calibrate",
        "Measure the backend once to set the batch size and crossover.",
        "bool",
        "1",
    ),
    Setting(
        "INCODE_EMBED_STRICT",
        "Embedding",
        "Strict accelerator",
        "Refuse the CPU fallback when the requested backend is unavailable.",
        "bool",
        "0",
    ),
    Setting(
        "INCODE_EMBED_ACCELERATOR",
        "Embedding",
        "Backend override",
        "Expert override; auto uses the backend the installer prepared.",
        "choice",
        "auto",
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
        return (
            f"{setting.name} expects {prefix}an integer "
            f"from {setting.minimum} to {setting.maximum}"
        )
    if setting.minimum <= number <= setting.maximum:
        return None
    return (
        f"{setting.name} expects {prefix}an integer "
        f"from {setting.minimum} to {setting.maximum}"
    )


def normalize(setting: Setting, raw: str) -> str:
    """Canonical string form for storage in a harness env block."""
    value = raw.strip()
    if setting.type == "bool":
        return "1" if value.lower() in _TRUE else "0"
    if setting.type in {"choice", "auto_int", "auto_off_int"} and not value.lstrip("-").isdigit():
        return value.lower()
    return value
