"""The accelerator environment the installer prepared, as the runtime sees it.

Passage acceleration needs a second, locked environment: ``fastembed`` and
``fastembed-gpu`` cannot share one. Preparing that environment is installation
work -- it resolves wheels, downloads them, and probes the result -- and none of
it may happen while the server is answering requests. So the installer does it
once and leaves behind this record: which accelerator it prepared, which
interpreter runs it, and what that interpreter's ONNX Runtime reported at the
moment it was proven to work.

The record is a claim about a machine, not a promise. Everything it asserts is
re-checked here before it is believed, and the record only ever nominates a
backend -- the real inference probe in the worker still decides. A record that
does not check out is discarded with a reason rather than repaired: repairing it
would mean installing something, which is exactly what this file exists to keep
out of the request path.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .backends import Accelerator, BackendDescriptor

logger = logging.getLogger(__name__)

# Bumped when a record's meaning changes. A record written by another version is
# ignored rather than reinterpreted, which costs a reinstall and never risks
# running an accelerator against a description that no longer means what it did.
RECORD_SCHEMA_VERSION = 1
RECORD_FILENAME = "accelerator.json"
# Points at a record elsewhere, for a machine whose data directory is not where
# the installer wrote one -- and for tests, which need a record without one.
RECORD_PATH_VARIABLE = "CODE_INDEXING_ACCEL_ENV"


def interpreter_path(environment_directory: Path, *, platform_name: str | None = None) -> Path:
    """Return the Python interpreter inside a virtual environment directory."""
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return environment_directory / "Scripts" / "python.exe"
    return environment_directory / "bin" / "python"


def running_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


@dataclass(frozen=True)
class AcceleratorEnvironment:
    """A prepared, verified accelerator installation on this machine."""

    accelerator: Accelerator
    interpreter: Path
    providers: tuple[str, ...]
    runtime_version: str = ""
    driver_version: str = ""
    device: str = ""
    python_version: str = ""
    recorded_at_ns: int = 0
    # Free-text note from the installer's probe, shown in diagnostics.
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "accelerator": self.accelerator.value,
            "interpreter": str(self.interpreter),
            "providers": list(self.providers),
            "runtime_version": self.runtime_version,
            "driver_version": self.driver_version,
            "device": self.device,
            "python_version": self.python_version,
            "recorded_at_ns": self.recorded_at_ns,
            "detail": self.detail,
        }

    @classmethod
    def from_json(cls, value: Any) -> AcceleratorEnvironment:
        """Build a record from parsed JSON, raising ``ValueError`` on anything odd."""
        if not isinstance(value, dict):
            raise ValueError("the record is not an object")
        if value.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"the record uses schema version {value.get('schema_version')!r}, "
                f"this build reads version {RECORD_SCHEMA_VERSION}"
            )
        try:
            accelerator = Accelerator(str(value["accelerator"]).lower())
        except (KeyError, ValueError) as exc:
            raise ValueError(f"unknown accelerator {value.get('accelerator')!r}") from exc
        if accelerator is Accelerator.AUTO:
            raise ValueError("'auto' names a selection policy, not a prepared environment")
        interpreter = str(value.get("interpreter") or "")
        if not interpreter:
            raise ValueError("the record names no interpreter")
        providers = value.get("providers")
        if not isinstance(providers, list) or not providers:
            raise ValueError("the record lists no execution providers")
        return cls(
            accelerator=accelerator,
            interpreter=Path(interpreter),
            providers=tuple(str(name) for name in providers),
            runtime_version=str(value.get("runtime_version", "")),
            driver_version=str(value.get("driver_version", "")),
            device=str(value.get("device", "")),
            python_version=str(value.get("python_version", "")),
            recorded_at_ns=int(value.get("recorded_at_ns", 0) or 0),
            detail=str(value.get("detail", "")),
        )

    def verify(self) -> str | None:
        """Return why this record cannot be used now, or ``None`` when it can."""
        if not self.interpreter.is_file():
            return (
                f"the accelerator environment's interpreter is gone ({self.interpreter}); "
                "reinstall to prepare it again"
            )
        running = running_python_version()
        if self.python_version and self.python_version != running:
            # The worker channel is multiprocessing's own connection protocol,
            # and both ends have to be the same Python to speak it. A server
            # upgraded past the environment the installer built is a reinstall,
            # not a fallback to be papered over silently.
            return (
                f"the accelerator environment was built for Python {self.python_version} "
                f"and this server runs Python {running}; reinstall to rebuild it"
            )
        return None


@dataclass(frozen=True)
class EnvironmentStatus:
    """What discovery found, and why it found nothing when it did not."""

    environment: AcceleratorEnvironment | None = None
    path: Path | None = None
    # Set only when a record existed and was rejected. No record at all is the
    # ordinary state of a CPU-only installation, not a problem to report.
    reason: str | None = None

    @property
    def providers(self) -> tuple[str, ...]:
        return self.environment.providers if self.environment is not None else ()


def record_path(data_directory: Path, environment: Mapping[str, str] | None = None) -> Path:
    environment = os.environ if environment is None else environment
    configured = environment.get(RECORD_PATH_VARIABLE)
    if configured:
        candidate = Path(configured).expanduser()
        return candidate / RECORD_FILENAME if candidate.is_dir() else candidate
    return data_directory / RECORD_FILENAME


def load_environment(
    data_directory: Path, environment: Mapping[str, str] | None = None
) -> EnvironmentStatus:
    """Read and verify the installer's record, if there is one to read."""
    path = record_path(data_directory, environment)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return EnvironmentStatus(path=path)
    except (OSError, ValueError) as exc:
        return EnvironmentStatus(path=path, reason=f"the accelerator record is unreadable: {exc}")
    try:
        record = AcceleratorEnvironment.from_json(raw)
    except ValueError as exc:
        return EnvironmentStatus(path=path, reason=f"the accelerator record is unusable: {exc}")
    problem = record.verify()
    if problem is not None:
        logger.warning("Ignoring the recorded accelerator environment: %s", problem)
        return EnvironmentStatus(path=path, reason=problem)
    return EnvironmentStatus(environment=record, path=path)


def write_environment(path: Path, record: AcceleratorEnvironment) -> None:
    """Persist *record*, replacing any earlier one atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def clear_environment(path: Path) -> bool:
    """Remove a record, so an install that fell back to CPU stops claiming more."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def apply_environment(
    descriptor: BackendDescriptor, record: AcceleratorEnvironment
) -> BackendDescriptor:
    """Describe *descriptor* with what the accelerator environment reported.

    The runtime and driver versions matter beyond diagnostics: they are part of
    the probe cache key, so a driver or runtime that moved under a prepared
    environment invalidates the verdict recorded against the old one.
    """
    if descriptor.accelerator is not record.accelerator:
        return descriptor
    return replace(
        descriptor,
        device=record.device or descriptor.device,
        runtime_version=record.runtime_version or descriptor.runtime_version,
        driver_version=record.driver_version or descriptor.driver_version,
    )
