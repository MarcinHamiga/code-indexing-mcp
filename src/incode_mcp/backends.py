"""Embedding backend contract: descriptors, capability probes, and selection.

Selection is deliberately split from execution. This module answers "which
backend should this machine use, and why", using only the execution providers
the installed ONNX Runtime reports. Whether that backend actually works is
decided later by a real inference probe in a disposable worker -- hardware
detection nominates a backend, only the probe confirms it.
"""

from __future__ import annotations

import logging
import platform
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from .errors import ErrorCode, IncodeError

logger = logging.getLogger(__name__)


class Accelerator(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    WEBGPU = "webgpu"
    MIGRAPHX = "migraphx"
    COREML = "coreml"


class Stability(StrEnum):
    """How far a backend has progressed through its promotion gates.

    Only ``AUTOMATIC`` backends are eligible for ``auto`` selection. The other
    two levels are reachable by explicit override, which is how a backend earns
    the benchmark evidence needed to be promoted.
    """

    EXPERIMENTAL = "experimental"
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class Precision(StrEnum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"


@dataclass(frozen=True)
class BackendDescriptor:
    """One candidate execution target for passage embedding."""

    accelerator: Accelerator
    provider: str
    device: str
    stability: Stability
    precision: Precision
    runtime_version: str = ""
    # Populated in Phase 3, alongside the NVIDIA driver/runtime detection the
    # locked CUDA installation needs. It reaches the probe cache key already,
    # so filling it in there is all that promotion requires.
    driver_version: str = ""

    @property
    def is_cpu(self) -> bool:
        return self.accelerator is Accelerator.CPU

    @property
    def providers(self) -> tuple[str, ...]:
        """Providers to hand the runtime, most specific first.

        Every accelerator keeps CPU behind it so a graph the provider cannot
        partition still executes rather than failing the session outright.
        """
        if self.is_cpu:
            return (CPU_PROVIDER,)
        return (self.provider, CPU_PROVIDER)


CPU_PROVIDER = "CPUExecutionProvider"

CPU_BACKEND = BackendDescriptor(
    accelerator=Accelerator.CPU,
    provider=CPU_PROVIDER,
    device="cpu",
    stability=Stability.AUTOMATIC,
    precision=Precision.FLOAT32,
)

# Every non-CPU backend this project knows how to name, in the order ``auto``
# considers them. Presence here is not an endorsement: an entry is only ever
# selected automatically once its stability reaches AUTOMATIC, which happens in
# the phase that ships and gates it on real hardware.
ACCELERATOR_BACKENDS: tuple[BackendDescriptor, ...] = (
    BackendDescriptor(
        accelerator=Accelerator.CUDA,
        provider="CUDAExecutionProvider",
        device="cuda:0",
        # Promoted to AUTOMATIC in Phase 3, once the locked CUDA installation
        # and the cosine-similarity and throughput gates land with it.
        stability=Stability.MANUAL,
        precision=Precision.FLOAT32,
    ),
    BackendDescriptor(
        accelerator=Accelerator.WEBGPU,
        provider="WebGpuExecutionProvider",
        device="gpu",
        stability=Stability.EXPERIMENTAL,
        precision=Precision.FLOAT32,
    ),
    BackendDescriptor(
        accelerator=Accelerator.MIGRAPHX,
        provider="MIGraphXExecutionProvider",
        device="gpu",
        stability=Stability.EXPERIMENTAL,
        precision=Precision.FLOAT32,
    ),
    BackendDescriptor(
        accelerator=Accelerator.COREML,
        provider="CoreMLExecutionProvider",
        device="ane",
        # Manual-only by decision, not by maturity: on the default Jina model
        # Core ML offloaded only part of the graph and lost to CPU. MANUAL
        # rather than EXPERIMENTAL says it works and was measured, and was
        # kept off automatic selection because of what the measurement said.
        stability=Stability.MANUAL,
        precision=Precision.FLOAT32,
    ),
)

KNOWN_BACKENDS: tuple[BackendDescriptor, ...] = (CPU_BACKEND, *ACCELERATOR_BACKENDS)


@dataclass(frozen=True)
class BackendSelection:
    """The backend a run will attempt, and the diagnosis behind that choice."""

    requested: Accelerator
    descriptor: BackendDescriptor
    available_providers: tuple[str, ...]
    # False only when an explicit, non-CPU request could not be honoured.
    # ``auto`` resolving to CPU is a correct outcome, not a denied request.
    honored: bool = True
    fallback_reason: str | None = None

    @property
    def accelerator(self) -> Accelerator:
        return self.descriptor.accelerator

    @property
    def uses_accelerator(self) -> bool:
        return not self.descriptor.is_cpu

    def fell_back_to(self, descriptor: BackendDescriptor, reason: str) -> BackendSelection:
        """Return this selection re-pointed at *descriptor* after a failure."""
        return replace(self, descriptor=descriptor, honored=False, fallback_reason=reason)

    def require_honored(self) -> None:
        """Raise when strict mode forbids the fallback this selection records."""
        if self.honored:
            return
        raise IncodeError(
            ErrorCode.BACKEND_UNAVAILABLE,
            f"Requested embedding accelerator is unavailable: {self.requested.value}",
            requested=self.requested.value,
            resolved=self.accelerator.value,
            reason=self.fallback_reason or "unavailable",
        )


def parse_accelerator(value: str) -> Accelerator:
    """Parse a configured accelerator name into its enum member."""
    try:
        return Accelerator(value.strip().lower())
    except ValueError as exc:
        expected = ", ".join(member.value for member in Accelerator)
        raise IncodeError(
            ErrorCode.INVALID_CONFIGURATION,
            f"Unknown embedding accelerator: {value!r}; expected one of {expected}",
            value=value,
        ) from exc


def available_execution_providers() -> tuple[str, ...]:
    """Return the execution providers the installed ONNX Runtime exposes.

    Import and query failures are not fatal: the CPU provider is always part of
    every ONNX Runtime distribution this project installs, so a machine whose
    runtime cannot be interrogated still indexes on CPU.
    """
    try:
        # Untyped third-party module; fastembed already depends on it, so this
        # import costs nothing that has not been paid for by the time it runs.
        import onnxruntime
    except Exception:  # pragma: no cover - exercised only where onnxruntime is absent
        logger.debug("ONNX Runtime is not importable; assuming CPU-only providers")
        return (CPU_PROVIDER,)
    try:
        providers = tuple(str(name) for name in onnxruntime.get_available_providers())
    except Exception:  # pragma: no cover - defensive against runtime build quirks
        logger.debug("ONNX Runtime did not report its providers; assuming CPU-only")
        return (CPU_PROVIDER,)
    return providers if CPU_PROVIDER in providers else (*providers, CPU_PROVIDER)


def runtime_version() -> str:
    """Return the ONNX Runtime version, or an empty string when unknown."""
    try:
        import onnxruntime

        return str(onnxruntime.__version__)
    except Exception:  # pragma: no cover - see available_execution_providers
        return ""


def platform_fingerprint() -> str:
    """Return the OS/architecture identity a probe result is only valid for."""
    return f"{platform.system()}-{platform.machine()}-{platform.release()}".lower()


def _by_accelerator(
    registry: Iterable[BackendDescriptor],
) -> dict[Accelerator, BackendDescriptor]:
    return {descriptor.accelerator: descriptor for descriptor in registry}


def select_backend(
    requested: Accelerator,
    *,
    available_providers: Sequence[str],
    registry: Sequence[BackendDescriptor] = KNOWN_BACKENDS,
) -> BackendSelection:
    """Choose the backend to attempt for passage embedding.

    ``auto`` takes the first registry entry that has reached automatic
    stability and whose provider the runtime actually exposes, and settles on
    CPU otherwise. An explicit accelerator is honoured whenever its provider is
    present, regardless of stability -- that is what an override is for -- and
    otherwise degrades to CPU with the reason recorded. Nothing here raises for
    an unavailable backend; strict mode turns the recorded diagnosis into an
    error at the point a run would have used it.
    """
    providers = tuple(str(name) for name in available_providers)
    catalogue = _by_accelerator(registry)
    cpu = catalogue.get(Accelerator.CPU, CPU_BACKEND)

    if requested is Accelerator.CPU:
        return BackendSelection(requested=requested, descriptor=cpu, available_providers=providers)

    if requested is Accelerator.AUTO:
        for descriptor in registry:
            if descriptor.is_cpu or descriptor.stability is not Stability.AUTOMATIC:
                continue
            if descriptor.provider in providers:
                return BackendSelection(
                    requested=requested,
                    descriptor=descriptor,
                    available_providers=providers,
                )
        return BackendSelection(
            requested=requested,
            descriptor=cpu,
            available_providers=providers,
            # honored: automatic selection asked for the best qualifying
            # backend and CPU is the correct answer when none qualifies.
            fallback_reason=(
                "no accelerator has reached automatic selection on this machine; "
                "set INCODE_EMBED_ACCELERATOR to override"
            ),
        )

    explicit = catalogue.get(requested)
    if explicit is None:
        return BackendSelection(
            requested=requested,
            descriptor=cpu,
            available_providers=providers,
            honored=False,
            fallback_reason=f"no backend is registered for {requested.value}",
        )
    if explicit.provider not in providers:
        return BackendSelection(
            requested=requested,
            descriptor=cpu,
            available_providers=providers,
            honored=False,
            fallback_reason=(
                f"{explicit.provider} is not among the execution providers this "
                f"installation offers ({', '.join(providers)})"
            ),
        )
    if explicit.stability is not Stability.AUTOMATIC:
        logger.warning(
            "Embedding accelerator %s is %s; it is selected here only because it "
            "was requested explicitly",
            explicit.accelerator.value,
            explicit.stability.value,
        )
    return BackendSelection(requested=requested, descriptor=explicit, available_providers=providers)


def describe_environment(descriptor: BackendDescriptor) -> BackendDescriptor:
    """Stamp *descriptor* with the runtime version this process is running."""
    return replace(descriptor, runtime_version=descriptor.runtime_version or runtime_version())
