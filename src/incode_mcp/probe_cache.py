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
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bumped whenever a stored record's meaning changes. Records written by another
# version are treated as absent rather than reinterpreted.
CACHE_SCHEMA_VERSION = 1
# A cache that grows without bound would keep every configuration a machine has
# ever had. Records are small, but the file is read on every start.
MAX_RECORDS = 32


@dataclass(frozen=True)
class ProbeKey:
    """Everything a probe result depends on.

    Any field changing means the previous verdict no longer applies: a
    different model artifact may partition differently, a different runtime or
    driver may support different operators, and a different device may not
    exist at all.
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
    """A backend that ran a real inference, and the batch size it settled on."""

    fingerprint: str
    batch_size: int
    dimension: int
    recorded_at_ns: int
    detail: str = ""

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
            )
        except (KeyError, TypeError, ValueError):
            # A record we cannot read is a record we cannot trust. Dropping it
            # costs one re-probe; honouring a partial one could pick a backend
            # that was never verified.
            return None


def model_artifact_fingerprint(cache_directory: Path, model_id: str) -> str:
    """Fingerprint the on-disk model artifact behind *model_id*.

    Names and sizes are enough to notice a re-download, a partially written
    cache, or a switch between model revisions, and unlike hashing contents it
    stays cheap on a multi-hundred-megabyte ONNX graph.
    """
    entries: list[str] = []
    try:
        for path in sorted(cache_directory.rglob("*")):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(cache_directory).as_posix()
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

    def store(self, key: ProbeKey, *, batch_size: int, dimension: int, detail: str = "") -> None:
        """Record a successful probe, replacing any earlier one for *key*."""
        record = ProbeRecord(
            fingerprint=key.fingerprint(),
            batch_size=batch_size,
            dimension=dimension,
            recorded_at_ns=time.time_ns(),
            detail=detail,
        )
        kept = [
            existing for existing in self._records() if existing.fingerprint != record.fingerprint
        ]
        kept.append(record)
        # Newest wins when the cache is trimmed: an old configuration's verdict
        # is the one least likely to be needed again.
        kept.sort(key=lambda item: item.recorded_at_ns)
        self._write(kept[-MAX_RECORDS:])

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
