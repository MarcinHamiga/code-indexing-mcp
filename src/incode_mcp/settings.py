"""Validated runtime settings for memory-safe indexing."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import psutil

from .backends import Accelerator, parse_accelerator
from .errors import ErrorCode, IncodeError
from .token_batching import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS

# The batch size ``auto`` resolves to when no calibration record applies. One
# item per microbatch is what CPU indexing has always used and what the memory
# ceiling was measured against; a larger default belongs to a backend that has
# earned it through calibration.
DEFAULT_AUTO_BATCH_SIZE = 1
MAX_BATCH_SIZE = 256
# A gigabyte of source in one run is already far past any measured crossover, so
# a larger figure is a mistyped setting rather than a policy.
MAX_CROSSOVER_CHARACTERS = 1024**3


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


def _batch_size(environment: Mapping[str, str]) -> tuple[int, bool]:
    """Return the configured microbatch size and whether it was left automatic."""
    raw = environment.get("INCODE_EMBED_BATCH_SIZE")
    if raw is None or raw.strip().lower() == "auto":
        return DEFAULT_AUTO_BATCH_SIZE, True
    return _integer(environment, "INCODE_EMBED_BATCH_SIZE", 1, 1, MAX_BATCH_SIZE), False


def _crossover(environment: Mapping[str, str]) -> tuple[int, bool]:
    """Return the configured crossover in characters and whether it is measured.

    ``off`` is a size of zero, which reads correctly everywhere downstream: no
    run is smaller than the threshold, so the accelerator starts on the first
    chunk, exactly as it did before anything measured whether that paid.
    """
    raw = environment.get("INCODE_EMBED_CROSSOVER")
    if raw is None or raw.strip().lower() == "auto":
        return 0, True
    if raw.strip().lower() == "off":
        return 0, False
    try:
        value = int(raw)
    except ValueError as exc:
        raise _configuration_error(
            "INCODE_EMBED_CROSSOVER", raw, "auto, off, or a character count"
        ) from exc
    if not 0 <= value <= MAX_CROSSOVER_CHARACTERS:
        raise _configuration_error(
            "INCODE_EMBED_CROSSOVER", raw, f"a character count up to {MAX_CROSSOVER_CHARACTERS}"
        )
    return value, False


def _memory_bytes(environment: Mapping[str, str], default_megabytes: int) -> int:
    """Resolve the indexing memory ceiling from either accepted variable.

    ``INCODE_EMBED_MEMORY_MB`` is the documented name. ``INCODE_INDEX_MEMORY_MB``
    predates it and keeps working; the newer name wins when both are set.
    """
    # Truthiness, not membership: an exported-but-empty variable is how a shell
    # says "unset", and letting it win would both fail on int("") and shadow a
    # perfectly good value under the legacy name.
    name = (
        "INCODE_EMBED_MEMORY_MB"
        if environment.get("INCODE_EMBED_MEMORY_MB")
        else "INCODE_INDEX_MEMORY_MB"
    )
    return _integer(environment, name, default_megabytes, 1024, 1024 * 1024) * 1024 * 1024


@dataclass(frozen=True)
class IndexSettings:
    mode: IndexMode
    index_wait_seconds: int
    embedding_batch_size: int
    embedding_max_tokens: int
    embedding_overlap_tokens: int
    embedding_threads: int
    embedding_cpu_arena: bool
    vector_index: str
    index_memory_bytes: int
    index_execution: str
    broker_mode: str
    embedding_accelerator: Accelerator = Accelerator.AUTO
    # True when the batch size was left to the runtime, which lets calibration
    # raise it for a backend that was measured to handle more.
    embedding_batch_auto: bool = True
    # The run size, in candidate characters, above which starting an accelerator
    # repays its model load. 0 with ``_auto`` set means nothing has measured one
    # yet; 0 with it clear means the operator turned deferral off.
    embedding_crossover_characters: int = 0
    embedding_crossover_auto: bool = True
    # Measuring a backend costs one sweep per configuration. Declining it leaves
    # the batch size and the crossover unmeasured, which is the behaviour every
    # release before this one had.
    embedding_calibrate: bool = True
    # Strict mode refuses the CPU fallback. A run that cannot reach the
    # requested accelerator fails with BACKEND_UNAVAILABLE instead of quietly
    # indexing more slowly than the caller asked for.
    embedding_strict: bool = False

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
        batch_size, batch_auto = _batch_size(environment)
        crossover, crossover_auto = _crossover(environment)
        accelerator = parse_accelerator(environment.get("INCODE_EMBED_ACCELERATOR", "auto"))

        return cls(
            mode=mode,
            # How long a startup index waits out a competing job before failing.
            # The global index lock serializes every job on the machine, so a
            # cold index elsewhere can hold it for minutes; 0 disables waiting.
            index_wait_seconds=_integer(
                environment, "INCODE_INDEX_WAIT_SECONDS", 300, 0, 24 * 60 * 60
            ),
            embedding_batch_size=batch_size,
            embedding_batch_auto=batch_auto,
            embedding_crossover_characters=crossover,
            embedding_crossover_auto=crossover_auto,
            embedding_calibrate=_boolean(environment, "INCODE_EMBED_CALIBRATE", True),
            embedding_accelerator=accelerator,
            embedding_strict=_boolean(environment, "INCODE_EMBED_STRICT", False),
            # Sequence length, not character count, drives embedding memory:
            # attention is quadratic in tokens. 1,024 keeps the widest window
            # well inside the model's 8,192-token limit and inside the default
            # memory ceiling even for token-dense minified source.
            embedding_max_tokens=_integer(
                environment, "INCODE_EMBED_MAX_TOKENS", DEFAULT_MAX_TOKENS, 64, 8192
            ),
            embedding_overlap_tokens=_integer(
                environment, "INCODE_EMBED_OVERLAP_TOKENS", DEFAULT_OVERLAP_TOKENS, 0, 4096
            ),
            embedding_threads=_integer(
                environment,
                "INCODE_EMBED_THREADS",
                max(1, min(2, os.cpu_count() or 1)),
                1,
                64,
            ),
            embedding_cpu_arena=_boolean(environment, "INCODE_EMBED_CPU_ARENA", False),
            vector_index=vector_index,
            index_memory_bytes=_memory_bytes(environment, default_memory_mb),
            index_execution=execution,
            broker_mode=broker_mode,
        )
