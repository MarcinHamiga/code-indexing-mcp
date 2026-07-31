"""Local cache of successful backend probes and batch calibration.

Probing an accelerator costs a process spawn, a model load, and a real
inference. That cost is worth paying once per machine configuration, not once
per index run -- but a cached "this works" is only meaningful while nothing
underneath it moved. Every input that could change the answer is folded into
the key, so a driver bump, a runtime upgrade, or a re-downloaded model
invalidates the record instead of silently vouching for a backend that no
longer works.

Nothing here leaves the machine: the cache is a plain JSON file under the
user's own cache directory.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

# Bumped whenever a stored record's meaning changes. Records written by another
# version are treated as absent rather than reinterpreted.
CACHE_SCHEMA_VERSION = 2
# A cache that grows without bound would keep every configuration a machine has
# ever had. Records are small, but the file is read on every start.
MAX_RECORDS = 32
# Long enough to outlast a competing write, short enough that a stale lock
# never delays indexing noticeably.
LOCK_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class ProbeKey:
    """Everything a probe result depends on.

    Any field changing means the previous verdict no longer applies: a
    different model artifact may partition differently, a different runtime or
    driver may support different operators, and a different device may not
    exist at all.

    ``driver_version`` is honoured by the key but is not yet populated by
    anything: driver detection arrives with the locked accelerator
    installations in Phase 3, and until then no backend reaches automatic
    selection. A cached probe never skips the model load, so a driver change
    that breaks a provider outright still surfaces as a failed load rather
    than as a trusted stale record.
    """

    model_id: str
    model_artifact: str
    accelerator: str
    provider: str
    runtime_version: str
    platform: str
    device: str
    driver_version: str = ""

    def fingerprint(self) -> str:
        parts = (
            self.model_id,
            self.model_artifact,
            self.accelerator,
            self.provider,
            self.runtime_version,
            self.platform,
            self.device,
            self.driver_version,
        )
        return sha256("\0".join(parts).encode()).hexdigest()


@dataclass(frozen=True)
class ProbeRecord:
    """A backend that ran a real inference, and what measuring it found.

    ``characters_per_second`` and ``load_ns`` are what the workload crossover is
    computed from, and a zero rate means "never measured" rather than "measured
    as nothing" -- which is why a record from before calibration existed is
    rejected by the schema version rather than read as a slow backend.
    """

    fingerprint: str
    batch_size: int
    dimension: int
    recorded_at_ns: int
    detail: str = ""
    characters_per_second: float = 0.0
    load_ns: int = 0
    # "memory" when the batch size above is the last one that fit rather than
    # the fastest one measured.
    limited_by: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: Any) -> ProbeRecord | None:
        if not isinstance(value, dict):
            return None
        try:
            return cls(
                fingerprint=str(value["fingerprint"]),
                batch_size=int(value["batch_size"]),
                dimension=int(value["dimension"]),
                recorded_at_ns=int(value["recorded_at_ns"]),
                detail=str(value.get("detail", "")),
                characters_per_second=float(value.get("characters_per_second", 0.0)),
                load_ns=int(value.get("load_ns", 0)),
                limited_by=str(value.get("limited_by", "")),
            )
        except (KeyError, TypeError, ValueError):
            # A record we cannot read is a record we cannot trust. Dropping it
            # costs one re-probe; honouring a partial one could pick a backend
            # that was never verified.
            return None


def model_directory(cache_directory: Path, model_id: str) -> Path:
    """Return the subtree under *cache_directory* holding *model_id*.

    FastEmbed stores models in the HuggingFace layout, so ``org/name`` lives in
    ``models--org--name``. Scoping to it keeps the rest of a shared cache --
    a second model, the sibling ``.locks`` directory, a partial download -- from
    invalidating a probe none of them had anything to do with. An unrecognised
    layout falls back to the whole directory, which over-invalidates rather
    than vouching for a model that moved.
    """
    candidate = cache_directory / f"models--{model_id.replace('/', '--')}"
    return candidate if candidate.is_dir() else cache_directory


def model_artifact_fingerprint(cache_directory: Path, model_id: str) -> str:
    """Fingerprint the on-disk model artifact behind *model_id*.

    Names and sizes are enough to notice a re-download, a partially written
    cache, or a switch between model revisions, and unlike hashing contents it
    stays cheap on a multi-hundred-megabyte ONNX graph.
    """
    root = model_directory(cache_directory, model_id)
    entries: list[str] = []
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(root).as_posix()
                entries.append(f"{relative}:{path.stat().st_size}")
            except OSError:
                continue
    except OSError:
        # An unreadable cache directory still yields a stable key -- one that
        # simply never matches a record written when it was readable.
        logger.debug("Could not enumerate the model cache at %s", cache_directory)
        return sha256(f"{model_id}\0unreadable".encode()).hexdigest()
    digest = sha256(model_id.encode())
    for entry in entries:
        digest.update(b"\0")
        digest.update(entry.encode())
    return digest.hexdigest()


class ProbeCache:
    """A JSON file of verified probes, keyed by full machine configuration."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, key: ProbeKey) -> ProbeRecord | None:
        """Return the record for *key*, or ``None`` when nothing matches it."""
        fingerprint = key.fingerprint()
        for record in self._records():
            if record.fingerprint == fingerprint:
                return record
        return None

    def store(
        self,
        key: ProbeKey,
        *,
        batch_size: int,
        dimension: int,
        detail: str = "",
        characters_per_second: float = 0.0,
        load_ns: int = 0,
        limited_by: str = "",
    ) -> None:
        """Record a successful probe, replacing any earlier one for *key*.

        Read-modify-write, so a daemon and a CLI probing different backends at
        once are serialised: ``os.replace`` alone would make each write atomic
        while still letting the later one drop the earlier one's record.
        """
        record = ProbeRecord(
            fingerprint=key.fingerprint(),
            batch_size=batch_size,
            dimension=dimension,
            recorded_at_ns=time.time_ns(),
            detail=detail,
            characters_per_second=characters_per_second,
            load_ns=load_ns,
            limited_by=limited_by,
        )
        with self._guard():
            kept = [
                existing
                for existing in self._records()
                if existing.fingerprint != record.fingerprint
            ]
            kept.append(record)
            # Newest wins when the cache is trimmed: an old configuration's
            # verdict is the one least likely to be needed again.
            kept.sort(key=lambda item: item.recorded_at_ns)
            self._write(kept[-MAX_RECORDS:])

    @contextlib.contextmanager
    def _guard(self) -> Iterator[None]:
        """Hold the cache lock, or proceed without it rather than fail a run."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(self.path.with_name(f"{self.path.name}.lock"))
        except OSError:
            yield
            return
        try:
            with lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
                yield
        except Timeout:
            # Another process is mid-write. Losing this record costs one
            # re-probe; blocking an index run on a diagnostics cache would
            # cost far more.
            logger.debug("Could not lock the probe cache at %s", self.path)

    def state(self, key: ProbeKey) -> str:
        """Return ``"hit"`` or ``"miss"`` for diagnostics such as model status."""
        return "hit" if self.load(key) is not None else "miss"

    def _records(self) -> list[ProbeRecord]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError):
            logger.debug("Discarding an unreadable probe cache at %s", self.path)
            return []
        if not isinstance(raw, dict) or raw.get("schema_version") != CACHE_SCHEMA_VERSION:
            return []
        entries = raw.get("records")
        if not isinstance(entries, list):
            return []
        return [record for record in map(ProbeRecord.from_json, entries) if record is not None]

    def _write(self, records: list[ProbeRecord]) -> None:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "records": [record.to_json() for record in records],
        }
        # A half-written cache would be discarded on the next read, but the
        # replace also keeps a concurrent reader from ever seeing one.
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            # Losing the cache costs a re-probe on the next run; it must never
            # cost the run that is happening now.
            logger.debug("Could not persist the probe cache at %s", self.path)
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
