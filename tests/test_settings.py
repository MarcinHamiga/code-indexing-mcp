from __future__ import annotations

import pytest

from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.settings import IndexMode, IndexSettings


def test_indexing_defaults_to_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INCODE_INDEX_MODE", raising=False)
    monkeypatch.delenv("INCODE_AUTO_INDEX", raising=False)

    settings = IndexSettings.from_environment()

    assert settings.mode is IndexMode.LAZY
    assert settings.embedding_batch_size == 1
    assert settings.embedding_threads >= 1
    assert settings.embedding_cpu_arena is False
    assert settings.vector_index == "exact"


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [("1", IndexMode.EAGER), ("true", IndexMode.EAGER), ("0", IndexMode.MANUAL)],
)
def test_legacy_auto_index_maps_to_index_mode(
    monkeypatch: pytest.MonkeyPatch, legacy: str, expected: IndexMode
) -> None:
    monkeypatch.delenv("INCODE_INDEX_MODE", raising=False)
    monkeypatch.setenv("INCODE_AUTO_INDEX", legacy)

    assert IndexSettings.from_environment().mode is expected


def test_index_mode_takes_precedence_over_legacy_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INCODE_INDEX_MODE", "manual")
    monkeypatch.setenv("INCODE_AUTO_INDEX", "1")

    assert IndexSettings.from_environment().mode is IndexMode.MANUAL


def test_invalid_index_settings_raise_stable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INCODE_EMBED_BATCH_SIZE", "0")

    with pytest.raises(IncodeError) as caught:
        IndexSettings.from_environment()

    assert caught.value.code is ErrorCode.INVALID_CONFIGURATION


def test_memory_budget_override_and_worker_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INCODE_INDEX_MEMORY_MB", "1536")
    monkeypatch.delenv("INCODE_INDEX_EXECUTION", raising=False)

    settings = IndexSettings.from_environment()

    assert settings.index_memory_bytes == 1536 * 1024 * 1024
    assert settings.index_execution == "worker"
