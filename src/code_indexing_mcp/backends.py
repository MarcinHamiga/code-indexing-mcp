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

from .errors import CodeIndexingError, ErrorCode

logger = logging.getLogger(__name__)


class Accelerator(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MLX = "mlx"
    WEBGPU = "webgpu"
    MIGRAPHX = "migraphx"
    COREML = "coreml"


class Runtime(StrEnum):
    """Which inference runtime executes a backend's passage model.

    Most backends are ONNX Runtime execution providers, and the machinery around
    them was shaped by that: providers are published before anything loads, an
    accelerator keeps CPU behind it, and a session may silently drop the
    provider it was given. ``MLX`` is none of those things, so the differences
    are named here once rather than special-cased by accelerator at each site.
    """

    # A provider built into the installed ONNX Runtime distribution.
    ONNX = "onnxruntime"
    # A provider that only exists once its plugin library has been registered,
    # so it is absent from the provider list until the model is being loaded.
    ONNX_PLUGIN = "onnxruntime-plugin"
    # Apple's array framework. No ONNX session, no execution providers, and no
    # graph partitioning: the model is this project's own MLX implementation.
    MLX = "mlx"


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
    # What the loaded model reports as the target it settled on. Usually an ONNX
    # execution provider; for a non-ONNX runtime it is that runtime's own name,
    # which is what selection, the installer's record, and diagnostics carry.
    provider: str
    device: str
    stability: Stability
    precision: Precision
    runtime_version: str = ""
    # Read from the installer's accelerator record, where a probe wrote down the
    # driver it verified against. It is part of the probe cache key, so a driver
    # upgrade retires the verdict recorded under the old one.
    driver_version: str = ""
    runtime: Runtime = Runtime.ONNX

    @property
    def is_cpu(self) -> bool:
        return self.accelerator is Accelerator.CPU

    @property
    def providers(self) -> tuple[str, ...]:
        """Providers to hand the runtime, most specific first.

        Every ONNX accelerator keeps CPU behind it so a graph the provider
        cannot partition still executes rather than failing the session
        outright. A non-ONNX runtime has no such fallback to name: it either
        runs the model it was given or fails to load it.
        """
        if self.is_cpu:
            return (CPU_PROVIDER,)
        if not self.runs_on_onnx:
            return (self.provider,)
        return (self.provider, CPU_PROVIDER)

    @property
    def provider_is_preregistered(self) -> bool:
        """Whether this target appears in the runtime's provider list up front.

        Only built-in ONNX providers do. A plugin provider exists once its
        library is registered, and a non-ONNX runtime never appears there at
        all, so requiring either one to be listed would refuse a working
        backend before its model was loaded.
        """
        return self.runtime is Runtime.ONNX

    @property
    def runs_on_onnx(self) -> bool:
        """Whether ONNX Runtime executes this backend at all.

        Named positively so a runtime added later has to say that it is an ONNX
        one, rather than inheriting the ONNX answer from every predicate that
        happened to be written as "not MLX".
        """
        return self.runtime in {Runtime.ONNX, Runtime.ONNX_PLUGIN}

    @property
    def publishes_execution_providers(self) -> bool:
        """Whether this backend's runtime has execution providers at all.

        Only ONNX Runtime does. ``available_execution_providers`` answers with
        the CPU provider when it cannot import a runtime, which is right for an
        ONNX environment and wrong for one that has no ONNX Runtime in it: it
        would put a provider nothing there can execute into the record the
        installer writes.
        """
        return self.runs_on_onnx

    @property
    def uses_direct_model(self) -> bool:
        """Whether this project loads the passage model itself for this backend.

        A direct model reports the target its own session resolved, so an empty
        report from one means the session is broken. FastEmbed models are read
        through a private layout where empty means "unknown" instead.
        """
        return self.accelerator in DIRECT_MODEL_ACCELERATORS


CPU_PROVIDER = "CPUExecutionProvider"
# MLX has no execution providers; this names the runtime itself so the record
# the installer writes, selection, and ``model status`` all keep describing one
# resolved target per backend.
MLX_PROVIDER = "MlxMetalBackend"

# The accelerators whose passage model this project loads and executes itself,
# rather than through FastEmbed. FastEmbed cannot configure these runtimes, and
# for two of them its ONNX Runtime distribution would conflict with theirs.
DIRECT_MODEL_ACCELERATORS = frozenset({Accelerator.WEBGPU, Accelerator.MIGRAPHX, Accelerator.MLX})

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
        # The first accelerator to reach automatic selection: it is installed
        # from a pinned CUDA/cuDNN/ONNX Runtime window into an environment of
        # its own, and probed there before the record that offers it is written.
        # Reaching AUTOMATIC only makes it *eligible* -- ``auto`` still passes
        # over it on a machine whose installation never prepared it.
        stability=Stability.AUTOMATIC,
        precision=Precision.FLOAT32,
    ),
    BackendDescriptor(
        accelerator=Accelerator.MLX,
        provider=MLX_PROVIDER,
        device="metal",
        # The designated Apple Silicon path, ahead of the cross-platform one:
        # WebGPU reached 1.11x of CPU there against a 1.25x gate, and MLX
        # reached 1.52-1.56x on the same corpus with vectors matching CPU to
        # cosine 1.0 and identical top-5 rankings. Promoted on that evidence,
        # which only makes it eligible -- ``auto`` still passes over it on a
        # machine whose installation never prepared it.
        stability=Stability.AUTOMATIC,
        precision=Precision.FLOAT32,
        runtime=Runtime.MLX,
    ),
    BackendDescriptor(
        accelerator=Accelerator.WEBGPU,
        provider="WebGpuExecutionProvider",
        device="gpu",
        stability=Stability.EXPERIMENTAL,
        precision=Precision.FLOAT32,
        runtime=Runtime.ONNX_PLUGIN,
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


def backend_for(
    accelerator: Accelerator, registry: Sequence[BackendDescriptor] = KNOWN_BACKENDS
) -> BackendDescriptor | None:
    """Return the descriptor registered for *accelerator*, if there is one."""
    for backend in registry:
        if backend.accelerator is accelerator:
            return backend
    return None


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

    def described_as(self, descriptor: BackendDescriptor) -> BackendSelection:
        """Return this selection with the same choice described more fully."""
        return replace(self, descriptor=descriptor)

    def diagnosed(self, reason: str) -> BackendSelection:
        """Return this selection carrying an additional reason for its outcome.

        Whether the request was honoured is not re-decided here: this only adds
        what the caller knows about *why* selection landed where it did.
        """
        combined = f"{self.fallback_reason}; {reason}" if self.fallback_reason else reason
        return replace(self, fallback_reason=combined)

    def require_honored(self) -> None:
        """Raise when strict mode forbids the fallback this selection records."""
        if self.honored:
            return
        raise CodeIndexingError(
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
        raise CodeIndexingError(
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


def runtime_version(runtime: Runtime = Runtime.ONNX) -> str:
    """Return the version of *runtime*, or an empty string when unknown.

    The version identifies what will actually execute the model, so it has to
    follow the backend rather than the import this process happens to have: an
    MLX environment has no ONNX Runtime to report a version for.
    """
    try:
        if runtime is Runtime.MLX:
            import mlx.core

            return str(mlx.core.__version__)
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
                "no accelerator is prepared and eligible on this machine; reinstall "
                "with --accelerator to prepare one, or set CODE_INDEXING_EMBED_ACCELERATOR "
                "to force a backend this installation already offers"
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
                f"installation offers ({', '.join(providers)}); reinstall with "
                f"--accelerator {explicit.accelerator.value} to prepare it"
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
    """Stamp *descriptor* with the runtime version this process is running.

    A version the installer's record already supplied is kept: it came from the
    environment that will run the backend, which this process may not be.
    """
    return replace(
        descriptor,
        runtime_version=descriptor.runtime_version or runtime_version(descriptor.runtime),
    )
