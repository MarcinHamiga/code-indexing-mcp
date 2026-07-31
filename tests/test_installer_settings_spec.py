"""Tests for the installer's declarative settings catalog."""

from pathlib import Path

import pytest

from incode_mcp.installer.settings_spec import (
    BY_NAME,
    SETTINGS,
    as_bool,
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
        # Paths are stored as typed on every platform: rewriting separators
        # would hand the server a path its own OS never asked for.
        ("INCODE_DATA_DIR", "/data", "/data"),
        ("INCODE_DATA_DIR", r"C:\data", r"C:\data"),
    ],
)
def test_normalize(name: str, raw: str, stored: str) -> None:
    assert normalize(BY_NAME[name], raw) == stored


def test_normalize_expands_a_tilde_no_shell_is_left_to_expand() -> None:
    assert normalize(BY_NAME["INCODE_DATA_DIR"], "~/indexes") == str(Path.home() / "indexes")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False)],
)
def test_as_bool_reads_every_spelling_the_server_accepts(raw: str, expected: bool) -> None:
    assert as_bool(raw) is expected


def test_dynamic_defaults_resolve_to_valid_values() -> None:
    memory = BY_NAME["INCODE_EMBED_MEMORY_MB"]
    threads = BY_NAME["INCODE_EMBED_THREADS"]
    assert validate(memory, default_value(memory)) is None
    assert validate(threads, default_value(threads)) is None
    assert default_value(BY_NAME["INCODE_DATA_DIR"]).endswith("incode")
