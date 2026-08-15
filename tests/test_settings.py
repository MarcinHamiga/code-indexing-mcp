from __future__ import annotations

import pytest

from code_indexing_mcp.backends import Accelerator
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.settings import IndexMode, IndexSettings


def test_indexing_defaults_to_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODE_INDEXING_INDEX_MODE", raising=False)
    monkeypatch.delenv("CODE_INDEXING_AUTO_INDEX", raising=False)

    settings = IndexSettings.from_environment()

    assert settings.mode is IndexMode.LAZY
    assert settings.embedding_batch_size == 1
    assert settings.embedding_threads >= 1
    assert settings.embedding_cpu_arena is False
    assert settings.vector_index == "exact"
    assert settings.vector_storage == "float16"
    assert settings.index_wait_seconds == 300


def test_vector_storage_is_parsed_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_VECTOR_STORAGE", "FLOAT32")
    assert IndexSettings.from_environment().vector_storage == "float32"

    monkeypatch.setenv("CODE_INDEXING_VECTOR_STORAGE", "int8")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_index_wait_seconds_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_INDEX_WAIT_SECONDS", "0")
    assert IndexSettings.from_environment().index_wait_seconds == 0

    monkeypatch.setenv("CODE_INDEXING_INDEX_WAIT_SECONDS", "-1")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [("1", IndexMode.EAGER), ("true", IndexMode.EAGER), ("0", IndexMode.MANUAL)],
)
def test_legacy_auto_index_maps_to_index_mode(
    monkeypatch: pytest.MonkeyPatch, legacy: str, expected: IndexMode
) -> None:
    monkeypatch.delenv("CODE_INDEXING_INDEX_MODE", raising=False)
    monkeypatch.setenv("CODE_INDEXING_AUTO_INDEX", legacy)

    assert IndexSettings.from_environment().mode is expected


def test_index_mode_takes_precedence_over_legacy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_INDEX_MODE", "manual")
    monkeypatch.setenv("CODE_INDEXING_AUTO_INDEX", "1")

    assert IndexSettings.from_environment().mode is IndexMode.MANUAL


def test_invalid_index_settings_raise_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_EMBED_BATCH_SIZE", "0")

    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_memory_budget_override_and_worker_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_INDEX_MEMORY_MB", "1536")
    monkeypatch.delenv("CODE_INDEXING_INDEX_EXECUTION", raising=False)

    settings = IndexSettings.from_environment()

    assert settings.index_memory_bytes == 1536 * 1024 * 1024
    assert settings.index_execution == "worker"


def test_token_window_settings_default_to_the_measured_budget() -> None:
    settings = IndexSettings.from_environment({})

    assert settings.embedding_max_tokens == 1024
    assert settings.embedding_overlap_tokens == 64


def test_token_window_settings_are_configurable() -> None:
    settings = IndexSettings.from_environment(
        {"CODE_INDEXING_EMBED_MAX_TOKENS": "512", "CODE_INDEXING_EMBED_OVERLAP_TOKENS": "32"}
    )

    assert settings.embedding_max_tokens == 512
    assert settings.embedding_overlap_tokens == 32


def test_a_token_budget_above_the_model_limit_is_rejected() -> None:
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_MAX_TOKENS": "16384"})

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_the_accelerator_defaults_to_automatic_selection() -> None:
    settings = IndexSettings.from_environment({})

    assert settings.embedding_accelerator is Accelerator.AUTO
    assert settings.embedding_strict is False
    assert settings.embedding_batch_auto is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("cpu", Accelerator.CPU),
        ("CUDA", Accelerator.CUDA),
        ("webgpu", Accelerator.WEBGPU),
        ("migraphx", Accelerator.MIGRAPHX),
        ("coreml", Accelerator.COREML),
    ],
)
def test_the_accelerator_is_configurable(value: str, expected: Accelerator) -> None:
    settings = IndexSettings.from_environment({"CODE_INDEXING_EMBED_ACCELERATOR": value})

    assert settings.embedding_accelerator is expected


def test_an_unknown_accelerator_is_a_configuration_error() -> None:
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_ACCELERATOR": "tpu"})

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_strict_mode_is_configurable() -> None:
    assert (
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_STRICT": "1"}).embedding_strict is True
    )
    assert (
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_STRICT": "off"}).embedding_strict
        is False
    )


def test_an_automatic_batch_size_keeps_the_cpu_default() -> None:
    settings = IndexSettings.from_environment({"CODE_INDEXING_EMBED_BATCH_SIZE": "auto"})

    assert settings.embedding_batch_size == 1
    assert settings.embedding_batch_auto is True


def test_an_explicit_batch_size_is_marked_as_not_calibratable() -> None:
    settings = IndexSettings.from_environment({"CODE_INDEXING_EMBED_BATCH_SIZE": "64"})

    assert settings.embedding_batch_size == 64
    assert settings.embedding_batch_auto is False


def test_the_crossover_is_measured_by_default() -> None:
    settings = IndexSettings.from_environment({})

    assert settings.embedding_crossover_auto is True
    assert settings.embedding_crossover_characters == 0
    assert settings.embedding_calibrate is True


def test_the_crossover_can_be_turned_off_entirely() -> None:
    """ "off" means the accelerator starts on the first chunk, which is what
    every run did before anything measured whether that paid."""
    settings = IndexSettings.from_environment({"CODE_INDEXING_EMBED_CROSSOVER": "off"})

    assert settings.embedding_crossover_auto is False
    assert settings.embedding_crossover_characters == 0


def test_an_explicit_crossover_overrides_the_measured_one() -> None:
    settings = IndexSettings.from_environment({"CODE_INDEXING_EMBED_CROSSOVER": "250000"})

    assert settings.embedding_crossover_auto is False
    assert settings.embedding_crossover_characters == 250_000


def test_a_crossover_that_is_neither_a_mode_nor_a_size_is_rejected() -> None:
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_CROSSOVER": "sometimes"})

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_calibration_can_be_declined() -> None:
    assert (
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_CALIBRATE": "0"}).embedding_calibrate
        is False
    )


def test_the_batch_size_range_reaches_the_documented_maximum() -> None:
    assert IndexSettings.from_environment(
        {"CODE_INDEXING_EMBED_BATCH_SIZE": "256"}
    ).embedding_batch_size

    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment({"CODE_INDEXING_EMBED_BATCH_SIZE": "257"})

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_the_documented_memory_variable_is_accepted() -> None:
    settings = IndexSettings.from_environment({"CODE_INDEXING_EMBED_MEMORY_MB": "2048"})

    assert settings.index_memory_bytes == 2048 * 1024 * 1024


def test_the_newer_memory_variable_wins_over_the_legacy_one() -> None:
    settings = IndexSettings.from_environment(
        {"CODE_INDEXING_EMBED_MEMORY_MB": "2048", "CODE_INDEXING_INDEX_MEMORY_MB": "1024"}
    )

    assert settings.index_memory_bytes == 2048 * 1024 * 1024


def test_an_exported_but_empty_memory_variable_does_not_shadow_the_legacy_name() -> None:
    """An empty export is a shell saying "unset", not a value of zero length."""
    settings = IndexSettings.from_environment(
        {"CODE_INDEXING_EMBED_MEMORY_MB": "", "CODE_INDEXING_INDEX_MEMORY_MB": "1536"}
    )

    assert settings.index_memory_bytes == 1536 * 1024 * 1024


def test_maintenance_defaults_to_enabled_with_24h_retention() -> None:
    settings = IndexSettings.from_environment({})

    assert settings.auto_maintenance is True
    assert settings.version_retention_hours == 24


def test_maintenance_is_configurable() -> None:
    settings = IndexSettings.from_environment(
        {"CODE_INDEXING_AUTO_MAINTENANCE": "off", "CODE_INDEXING_VERSION_RETENTION_HOURS": "48"}
    )

    assert settings.auto_maintenance is False
    assert settings.version_retention_hours == 48


def test_version_retention_never_reaches_zero_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-hour automatic retention would reap versions concurrent readers use."""
    monkeypatch.setenv("CODE_INDEXING_VERSION_RETENTION_HOURS", "0")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION

    monkeypatch.setenv("CODE_INDEXING_VERSION_RETENTION_HOURS", "-1")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_version_retention_has_a_bounded_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_VERSION_RETENTION_HOURS", "100000")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_branch_cache_limit_defaults_to_four_slots() -> None:
    settings = IndexSettings.from_environment({})

    assert settings.branch_cache_limit == 4


def test_branch_cache_limit_is_configurable() -> None:
    settings = IndexSettings.from_environment({"CODE_INDEXING_BRANCH_CACHE_LIMIT": "12"})

    assert settings.branch_cache_limit == 12


def test_branch_cache_limit_keeps_at_least_the_active_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A limit below one would evict the active slot itself."""
    monkeypatch.setenv("CODE_INDEXING_BRANCH_CACHE_LIMIT", "0")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_branch_cache_limit_has_a_bounded_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODE_INDEXING_BRANCH_CACHE_LIMIT", "33")
    with pytest.raises(CodeIndexingError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION
