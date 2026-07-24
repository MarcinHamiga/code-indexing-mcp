"""Validated runtime settings for memory-safe indexing."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import psutil

from .errors import ErrorCode, IncodeError


class IndexMode(StrEnum):
    LAZY = "lazy"
    EAGER = "eager"
    MANUAL = "manual"


def _configuration_error(name: str, value: str, expected: str) -> IncodeError:
    return IncodeError(
        ErrorCode.INVALID_CONFIGURATION,
        f"{name}={value!r} is invalid; expected {expected}",
        setting=name,
        value=value,
    )


def _integer(
    environment: Mapping[str, str], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise _configuration_error(name, raw, f"an integer from {minimum} to {maximum}") from exc
    if not minimum <= value <= maximum:
        raise _configuration_error(name, raw, f"an integer from {minimum} to {maximum}")
    return value


def _boolean(environment: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environment.get(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise _configuration_error(name, raw, "a boolean")


@dataclass(frozen=True)
class IndexSettings:
    mode: IndexMode
    embedding_batch_size: int
    embedding_threads: int
    embedding_cpu_arena: bool
    vector_index: str
    index_memory_bytes: int
    index_execution: str
    broker_mode: str

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> IndexSettings:
        environment = os.environ if environment is None else environment
        mode_raw = environment.get("INCODE_INDEX_MODE")
        if mode_raw is None:
            legacy = environment.get("INCODE_AUTO_INDEX")
            if legacy is None:
                mode = IndexMode.LAZY
            else:
                mode = (
                    IndexMode.EAGER
                    if _boolean(environment, "INCODE_AUTO_INDEX", False)
                    else IndexMode.MANUAL
                )
        else:
            try:
                mode = IndexMode(mode_raw.lower())
            except ValueError as exc:
                raise _configuration_error(
                    "INCODE_INDEX_MODE", mode_raw, "lazy, eager, or manual"
                ) from exc

        vector_index = environment.get("INCODE_VECTOR_INDEX", "exact").lower()
        if vector_index not in {"exact", "hnsw"}:
            raise _configuration_error("INCODE_VECTOR_INDEX", vector_index, "exact or hnsw")
        execution = environment.get("INCODE_INDEX_EXECUTION", "worker").lower()
        if execution not in {"worker", "in-process"}:
            raise _configuration_error("INCODE_INDEX_EXECUTION", execution, "worker or in-process")
        broker_mode = environment.get("INCODE_BROKER", "auto").lower()
        if broker_mode not in {"auto", "on", "off"}:
            raise _configuration_error("INCODE_BROKER", broker_mode, "auto, on, or off")
        default_memory_mb = max(
            1024,
            min(2048, int(psutil.virtual_memory().total * 0.25) // (1024 * 1024)),
        )

        return cls(
            mode=mode,
            embedding_batch_size=_integer(environment, "INCODE_EMBED_BATCH_SIZE", 1, 1, 32),
            embedding_threads=_integer(
                environment,
                "INCODE_EMBED_THREADS",
                max(1, min(2, os.cpu_count() or 1)),
                1,
                64,
            ),
            embedding_cpu_arena=_boolean(environment, "INCODE_EMBED_CPU_ARENA", False),
            vector_index=vector_index,
            index_memory_bytes=_integer(
                environment,
                "INCODE_INDEX_MEMORY_MB",
                default_memory_mb,
                1024,
                1024 * 1024,
            )
            * 1024
            * 1024,
            index_execution=execution,
            broker_mode=broker_mode,
        )
