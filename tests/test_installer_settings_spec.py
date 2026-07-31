"""Tests for the installer's declarative settings catalog."""

from pathlib import Path

import pytest

from code_indexing_mcp.installer.settings_spec import (
    BY_NAME,
    SETTINGS,
    as_bool,
    default_value,
    normalize,
    validate,
)


def test_catalog_covers_exactly_the_documented_settings() -> None:
    assert {setting.name for setting in SETTINGS} == {
        "CODE_INDEXING_INDEX_MODE",
        "CODE_INDEXING_INDEX_WAIT_SECONDS",
        "CODE_INDEXING_EMBED_MEMORY_MB",
        "CODE_INDEXING_VECTOR_INDEX",
        "CODE_INDEXING_INDEX_EXECUTION",
        "CODE_INDEXING_BROKER",
        "CODE_INDEXING_DATA_DIR",
        "CODE_INDEXING_CACHE_DIR",
        "CODE_INDEXING_OFFLINE",
        "CODE_INDEXING_EMBED_BATCH_SIZE",
        "CODE_INDEXING_EMBED_MAX_TOKENS",
        "CODE_INDEXING_EMBED_OVERLAP_TOKENS",
        "CODE_INDEXING_EMBED_THREADS",
        "CODE_INDEXING_EMBED_CPU_ARENA",
        "CODE_INDEXING_EMBED_CROSSOVER",
        "CODE_INDEXING_EMBED_CALIBRATE",
        "CODE_INDEXING_EMBED_STRICT",
        "CODE_INDEXING_EMBED_ACCELERATOR",
    }


def test_every_setting_has_display_metadata_and_a_group() -> None:
    for setting in SETTINGS:
        assert setting.group in {"Indexing", "Embedding"}
        assert setting.label and setting.help


@pytest.mark.parametrize(
    ("name", "raw", "ok"),
    [
        ("CODE_INDEXING_INDEX_MODE", "eager", True),
        ("CODE_INDEXING_INDEX_MODE", "sometimes", False),
        ("CODE_INDEXING_INDEX_WAIT_SECONDS", "300", True),
        ("CODE_INDEXING_INDEX_WAIT_SECONDS", "86401", False),
        ("CODE_INDEXING_INDEX_WAIT_SECONDS", "-1", False),
        ("CODE_INDEXING_EMBED_MEMORY_MB", "2048", True),
        ("CODE_INDEXING_EMBED_MEMORY_MB", "512", False),
        ("CODE_INDEXING_EMBED_BATCH_SIZE", "auto", True),
        ("CODE_INDEXING_EMBED_BATCH_SIZE", "256", True),
        ("CODE_INDEXING_EMBED_BATCH_SIZE", "0", False),
        ("CODE_INDEXING_EMBED_CROSSOVER", "off", True),
        ("CODE_INDEXING_EMBED_CROSSOVER", "auto", True),
        ("CODE_INDEXING_EMBED_CROSSOVER", "100000", True),
        ("CODE_INDEXING_EMBED_CROSSOVER", "banana", False),
        ("CODE_INDEXING_EMBED_MAX_TOKENS", "8192", True),
        ("CODE_INDEXING_EMBED_MAX_TOKENS", "63", False),
        ("CODE_INDEXING_EMBED_OVERLAP_TOKENS", "0", True),
        ("CODE_INDEXING_EMBED_THREADS", "64", True),
        ("CODE_INDEXING_EMBED_THREADS", "65", False),
        ("CODE_INDEXING_OFFLINE", "yes", True),
        ("CODE_INDEXING_OFFLINE", "maybe", False),
        ("CODE_INDEXING_DATA_DIR", "/tmp/data", True),
        ("CODE_INDEXING_DATA_DIR", "", False),
        ("CODE_INDEXING_EMBED_ACCELERATOR", "coreml", True),
        ("CODE_INDEXING_EMBED_ACCELERATOR", "tpu", False),
    ],
)
def test_validate(name: str, raw: str, ok: bool) -> None:
    assert (validate(BY_NAME[name], raw) is None) is ok


def test_validate_unknown_names_are_rejected_by_lookup() -> None:
    assert "CODE_INDEXING_FROBNICATE" not in BY_NAME


@pytest.mark.parametrize(
    ("name", "raw", "stored"),
    [
        ("CODE_INDEXING_OFFLINE", "YES", "1"),
        ("CODE_INDEXING_OFFLINE", "off", "0"),
        ("CODE_INDEXING_INDEX_MODE", "EAGER", "eager"),
        ("CODE_INDEXING_EMBED_BATCH_SIZE", "AUTO", "auto"),
        ("CODE_INDEXING_EMBED_BATCH_SIZE", "8", "8"),
        # Paths are stored as typed on every platform: rewriting separators
        # would hand the server a path its own OS never asked for.
        ("CODE_INDEXING_DATA_DIR", "/data", "/data"),
        ("CODE_INDEXING_DATA_DIR", r"C:\data", r"C:\data"),
    ],
)
def test_normalize(name: str, raw: str, stored: str) -> None:
    assert normalize(BY_NAME[name], raw) == stored


def test_normalize_expands_a_tilde_no_shell_is_left_to_expand() -> None:
    assert normalize(BY_NAME["CODE_INDEXING_DATA_DIR"], "~/indexes") == str(Path.home() / "indexes")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False)],
)
def test_as_bool_reads_every_spelling_the_server_accepts(raw: str, expected: bool) -> None:
    assert as_bool(raw) is expected


def test_dynamic_defaults_resolve_to_valid_values() -> None:
    memory = BY_NAME["CODE_INDEXING_EMBED_MEMORY_MB"]
    threads = BY_NAME["CODE_INDEXING_EMBED_THREADS"]
    assert validate(memory, default_value(memory)) is None
    assert validate(threads, default_value(threads)) is None
    assert default_value(BY_NAME["CODE_INDEXING_DATA_DIR"]).endswith("code-indexing-mcp")
