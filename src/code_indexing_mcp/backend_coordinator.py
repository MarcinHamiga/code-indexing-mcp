"""Backend and accelerator selection, with batch-size and crossover calibration.

Split out of ``Application`` in the review's Track 5 (see
``docs/plans/2026-09-02-review-remediation-5-application-split-plan.md``, decision
D1) so backend/accelerator concerns are owned by one object instead of being
interleaved with project registry, query, and maintenance code. ``Application``
constructs one instance as ``self.backends`` and keeps thin delegates for the
members other modules and tests reach through it:

- ``cli.py`` and ``daemon.py`` call ``app.model_status()``.
- ``benchmark.py`` reads ``app.effective_backend_selection``.
- ``tests/test_application.py`` reads ``app.backend_selection``,
  ``app.probe_cache``, ``app.serving_providers``, ``app.crossover_characters()``,
  and ``app.model_status()`` directly, and (per D1) reaches the four privates
  ``_remember_fallback``, ``_build_probe_key``, ``_accelerator_launcher``, and
  ``_cpu_probe_key`` through ``app.backends.<name>`` instead of ``app.<name>``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .accelerator_env import EnvironmentStatus, apply_environment, load_environment
from .backends import (
    CPU_BACKEND,
    BackendDescriptor,
    BackendSelection,
    available_execution_providers,
    backend_for,
    describe_environment,
    platform_fingerprint,
    runtime_version,
    select_backend,
)
from .calibration import LIMITED_BY_MEMORY, crossover_characters
from .embedding import Embedder, FastEmbedder, SegmentPlan
from .embedding_worker import EmbeddingWorkerSession, WorkerConfig, default_launcher
from .models import ModelStatus
from .passage_backend import PassageBackendSession
from .probe_cache import ProbeCache, ProbeKey, ProbeRecord, model_artifact_fingerprint
from .settings import IndexSettings
from .worker_launcher import ExternalInterpreterLauncher, WorkerLauncher

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    from .application import RuntimePaths

logger = logging.getLogger(__name__)


def _rate(record: ProbeRecord | None) -> float | None:
    """Return a measured rate, or None for one that was never measured.

    A stored zero means the record predates its measurement, and reporting it
    as zero characters per second would describe a backend that never finishes.
    """
    if record is None or record.characters_per_second <= 0:
        return None
    return record.characters_per_second


class BackendCoordinator:
    """Owns backend selection, calibration, and probe caching for one process."""

    def __init__(self, paths: RuntimePaths, settings: IndexSettings, embedder: Embedder) -> None:
        self.paths = paths
        self.settings = settings
        self.embedder = embedder
        # Passage embedding is the only role acceleration targets. The query
        # model stays in this process on CPU so a search never waits on a
        # worker spawning or a model loading onto a device.
        #
        # An accelerator usually lives in a second environment the installer
        # prepared, so what this process can execute is not the whole story:
        # the providers that environment reported are candidates too, and a
        # backend chosen from them runs in its interpreter rather than ours.
        self.serving_providers: tuple[str, ...] = available_execution_providers()
        self.accelerator_environment: EnvironmentStatus = load_environment(paths.data)
        self.backend_selection: BackendSelection = self._select_backend()
        self.probe_cache = ProbeCache(paths.cache / "backend-probes.json")
        self._probe_key: ProbeKey | None = None
        # Set when a run actually tried the selected accelerator and it failed.
        # Only successful probes are cached, so without this memo every index
        # run in a long-lived daemon would re-spawn a known-dead backend and
        # reload its model onto the device before giving up again.
        self._runtime_fallback: BackendSelection | None = None
        self.embedding_batch_size: int = settings.embedding_batch_size
        self.batch_calibration: str = "explicit"
        if settings.embedding_batch_auto:
            self.batch_calibration = "default"
            if self.backend_selection.uses_accelerator:
                self._probe_key = self._build_probe_key(embedder)
                cached = self.probe_cache.load(self._probe_key)
                if cached is not None and cached.batch_size > 0:
                    self.embedding_batch_size = cached.batch_size
                    # A size something forced down is not the size calibration
                    # chose, and a machine pinned low by one bad run has to be
                    # able to say so. Which kind of ceiling stopped it is a
                    # question for the recommendation, not for this label.
                    self.batch_calibration = "reduced" if cached.limited_by else "measured"

    def _select_backend(self) -> BackendSelection:
        """Choose a backend from everything this machine can actually execute."""
        record = self.accelerator_environment.environment
        providers = list(self.serving_providers)
        if record is not None:
            # The prepared environment vouches for the one accelerator it was
            # probed for, not for every provider its runtime happens to ship.
            # Widening any further would offer a backend on the strength of a
            # record that never exercised it -- and would let selection land on
            # an accelerator whose device and driver this record cannot describe.
            prepared = backend_for(record.accelerator)
            if prepared is not None and prepared.provider in record.providers:
                providers.append(prepared.provider)
        selection = select_backend(
            self.settings.embedding_accelerator, available_providers=providers
        )
        if record is not None and selection.uses_accelerator:
            selection = selection.described_as(apply_environment(selection.descriptor, record))
        rejection = self.accelerator_environment.reason
        if rejection is not None and not selection.uses_accelerator:
            # A record that was found and refused explains the CPU outcome far
            # better than "no accelerator is prepared" does.
            selection = selection.diagnosed(rejection)
        return selection

    def _runs_externally(self, descriptor: BackendDescriptor) -> bool:
        """Whether a worker for *descriptor* needs the prepared environment.

        A provider this interpreter already exposes needs no second environment
        -- an explicitly requested Core ML on macOS runs in the serving
        environment's own runtime. Anything offered only by the prepared
        accelerator environment runs in that environment's interpreter.
        """
        record = self.accelerator_environment.environment
        return record is not None and descriptor.provider not in self.serving_providers

    def _accelerator_launcher(self, descriptor: BackendDescriptor) -> WorkerLauncher:
        """Return where a worker for *descriptor* has to be started."""
        record = self.accelerator_environment.environment
        if record is None or not self._runs_externally(descriptor):
            return default_launcher()
        return ExternalInterpreterLauncher(
            record.interpreter,
            environment_name=f"{record.accelerator.value} environment",
        )

    @property
    def effective_backend_selection(self) -> BackendSelection:
        """The backend the next run will attempt, after any runtime fallback.

        ``backend_selection`` records what selection resolved to from static
        capability alone. Once a run has tried it and been degraded, that
        verdict stands for the life of this process.
        """
        return self._runtime_fallback or self.backend_selection

    def _remember_fallback(self, degraded: BackendSelection) -> None:
        logger.warning(
            "Pinning passage embedding to CPU for the rest of this process: %s",
            degraded.fallback_reason,
        )
        self._runtime_fallback = degraded

    def _build_probe_key(self, embedder: Embedder) -> ProbeKey:
        descriptor = self.backend_selection.descriptor
        cache_directory = getattr(embedder, "cache_directory", self.paths.cache / "models")
        return ProbeKey(
            model_id=embedder.model_id,
            model_artifact=model_artifact_fingerprint(Path(cache_directory), embedder.model_id),
            accelerator=descriptor.accelerator.value,
            provider=descriptor.provider,
            # The record's version describes the environment that will run the
            # backend; this process's own runtime is only the fallback answer.
            runtime_version=descriptor.runtime_version or runtime_version(descriptor.runtime),
            platform=platform_fingerprint(),
            device=descriptor.device,
            driver_version=descriptor.driver_version,
        )

    def _cpu_probe_key(self) -> ProbeKey:
        """The key CPU's own calibration is stored under.

        CPU needs no probe to be trusted, but the crossover is a comparison and
        a comparison needs both sides measured -- under a key that moves when
        the model, the platform, or this process's runtime does, for the same
        reasons the accelerator's does.
        """
        cache_directory = getattr(self.embedder, "cache_directory", self.paths.cache / "models")
        return ProbeKey(
            model_id=self.embedder.model_id,
            model_artifact=model_artifact_fingerprint(
                Path(cache_directory), self.embedder.model_id
            ),
            accelerator=CPU_BACKEND.accelerator.value,
            provider=CPU_BACKEND.provider,
            runtime_version=runtime_version(CPU_BACKEND.runtime),
            platform=platform_fingerprint(),
            device=CPU_BACKEND.device,
        )

    def _cpu_max_items(self) -> int:
        """Return the microbatch size measured for CPU, if one was.

        0 means CPU keeps whatever the indexer planned, which is correct both
        when nothing has been measured and when the operator set a size
        explicitly -- an explicit size is a size for the whole installation.
        """
        if not self.settings.embedding_batch_auto:
            return 0
        record = self.probe_cache.load(self._cpu_probe_key())
        return 0 if record is None else record.batch_size

    def _measurements(self) -> tuple[ProbeRecord | None, ProbeRecord | None]:
        """Return what calibration recorded for CPU and for the accelerator."""
        selection = self.effective_backend_selection
        if not selection.uses_accelerator:
            return self.probe_cache.load(self._cpu_probe_key()), None
        key = self._probe_key or self._build_probe_key(self.embedder)
        return self.probe_cache.load(self._cpu_probe_key()), self.probe_cache.load(key)

    def crossover_characters(self) -> int | None:
        """Return the run size below which this machine should stay on CPU.

        0 means "start the accelerator immediately", which is the answer when
        the operator turned deferral off and also when nothing has been measured
        yet -- the first run on a machine is what does the measuring, and it
        cannot defer on numbers it is in the middle of producing.

        ``None`` means the accelerator never overtakes CPU, so no run is large
        enough to be worth starting it for. That is not the same statement as a
        very large threshold, and reporting it as one would name a size some run
        could conceivably pass -- and would collide with an operator who pinned
        that size deliberately.

        Strict mode is a third such answer. It exists for a caller who would
        rather fail than quietly index at CPU speed, and a deferral is quiet
        CPU indexing that no degradation reports -- so under strict mode the
        accelerator that was asked for is the one that runs, whatever the run
        turns out to cost.
        """
        if self.settings.embedding_strict:
            return 0
        if not self.settings.embedding_crossover_auto:
            return self.settings.embedding_crossover_characters
        cpu, accelerator = self._measurements()
        if cpu is None or accelerator is None:
            return 0
        return self._measured_crossover()

    def _measured_crossover(self) -> int | None:
        """Return the crossover the recorded measurements imply, if both exist.

        What the machine measured, with no policy applied. ``model status``
        reports this, so an explicit threshold or strict mode changes which runs
        defer without changing what this machine was found to be.
        """
        cpu, accelerator = self._measurements()
        if cpu is None or accelerator is None:
            return None
        return crossover_characters(
            accelerator_load_ns=accelerator.load_ns,
            cpu_load_ns=cpu.load_ns,
            cpu_characters_per_second=cpu.characters_per_second,
            accelerator_characters_per_second=accelerator.characters_per_second,
        )

    def _recommended_override(
        self, cpu: ProbeRecord | None, accelerator: ProbeRecord | None
    ) -> str | None:
        """Return the one setting change the measurements argue for, if any.

        Only a memory ceiling has a setting behind it. A batch that took the
        worker down with it was reduced just the same, but raising the ceiling
        is not what answers a device that could not make the allocation, so
        that case is reported without advice attached.
        """
        if accelerator is not None and accelerator.limited_by == LIMITED_BY_MEMORY:
            return (
                "CODE_INDEXING_EMBED_MEMORY_MB (a batch overran the ceiling and was reduced to "
                f"{accelerator.batch_size})"
            )
        if (
            cpu is not None
            and accelerator is not None
            and cpu.characters_per_second > 0
            and accelerator.characters_per_second > 0
            and accelerator.characters_per_second <= cpu.characters_per_second
        ):
            return (
                "CODE_INDEXING_EMBED_ACCELERATOR=cpu (the accelerator measured no faster than CPU "
                "on this machine)"
            )
        return None

    def _passage_session_factory(
        self, embedder: FastEmbedder, *, segment_plan: Callable[[], SegmentPlan]
    ) -> Callable[[], PassageBackendSession]:
        """Build the factory that opens one passage session per index run.

        Both backends are described up front, but neither process is started
        until indexing asks for one, and the accelerator's is only started if
        the selection actually chose it.

        ``segment_plan`` is read lazily, at call time, rather than captured as
        an attribute: it is ``Application.indexer.segment_plan``, and the
        ``Indexer`` is not built until after this factory is created (moving
        this method out of ``Application`` cannot change that ordering without
        changing behaviour), so the caller passes an accessor instead of a
        value.
        """
        ceiling_bytes = self.settings.index_memory_bytes
        strict = self.settings.embedding_strict
        probe_key = self._probe_key

        def worker_config(providers: tuple[str, ...], accelerator: str) -> WorkerConfig:
            return WorkerConfig(
                cache_directory=str(embedder.cache_directory),
                offline=embedder.offline,
                threads=self.settings.embedding_threads,
                enable_cpu_mem_arena=self.settings.embedding_cpu_arena,
                dimension=embedder.dimension,
                model_id=embedder.model_id,
                providers=providers,
                accelerator=accelerator,
            )

        descriptor = self.backend_selection.descriptor
        accelerator_config = worker_config(descriptor.providers, descriptor.accelerator.value)
        cpu_config = worker_config(CPU_BACKEND.providers, CPU_BACKEND.accelerator.value)
        accelerator_launcher = self._accelerator_launcher(descriptor)

        def session(config: WorkerConfig, launcher: WorkerLauncher) -> EmbeddingWorkerSession:
            return EmbeddingWorkerSession(
                config, configured_ceiling_bytes=ceiling_bytes, launcher=launcher
            )

        def new_passage_session() -> PassageBackendSession:
            return PassageBackendSession(
                # Read per run, not captured: a fallback recorded by an earlier
                # run keeps this one from paying for the same dead backend.
                self.effective_backend_selection,
                accelerator_factory=lambda: session(accelerator_config, accelerator_launcher),
                # The fallback never depends on a prepared environment: it is
                # what a failed accelerator falls back *to*.
                cpu_factory=lambda: session(cpu_config, default_launcher()),
                strict=strict,
                probe_cache=self.probe_cache,
                probe_key=probe_key,
                cpu_probe_key=self._cpu_probe_key(),
                # Only calibration establishes one. A configured default
                # recorded here would make ``model status`` report it as a
                # measurement that never ran.
                calibrated_batch_size=0,
                dimension=embedder.dimension,
                on_degrade=self._remember_fallback,
                # Read per run: the first run on a machine writes the numbers
                # every later run defers on, and a daemon must not have to be
                # restarted to start using them.
                crossover_characters=self.crossover_characters(),
                calibration_plan=(segment_plan() if self.settings.embedding_calibrate else None),
                # The plan the indexer builds is packed for the accelerator,
                # because that is whose batch size calibration adopted. A run
                # that defers or degrades to CPU is packed for CPU instead.
                cpu_max_items=self._cpu_max_items(),
            )

        return new_passage_session

    def model_status(self) -> ModelStatus:
        """Report the resolved embedding stack without loading or probing it."""
        selection: BackendSelection = self.effective_backend_selection
        descriptor = describe_environment(selection.descriptor)
        if not selection.uses_accelerator:
            # CPU is the reference backend; it needs no probe to be trusted.
            probe_state = "not-applicable"
        else:
            key = self._probe_key or self._build_probe_key(self.embedder)
            probe_state = self.probe_cache.state(key)
        record = self.accelerator_environment.environment
        external = selection.uses_accelerator and self._runs_externally(descriptor)
        cpu, accelerator = self._measurements()
        # What was measured, not what policy does with it: an explicit setting
        # or strict mode changes which runs defer, and neither changes what this
        # machine turned out to be.
        measured_crossover = self._measured_crossover()
        return ModelStatus(
            embedding_model=self.embedder.model_id,
            dimension=self.embedder.dimension,
            requested_accelerator=selection.requested.value,
            resolved_accelerator=descriptor.accelerator.value,
            device=descriptor.device,
            execution_provider=descriptor.provider,
            available_providers=list(selection.available_providers),
            stability=descriptor.stability.value,
            precision=descriptor.precision.value,
            runtime_version=descriptor.runtime_version,
            driver_version=descriptor.driver_version,
            # Where passage embedding will run. None means this process's own
            # environment, which is always the answer for CPU.
            accelerator_environment=str(record.interpreter) if external and record else None,
            accelerator_prepared=None if record is None else record.accelerator.value,
            batch_size=self.embedding_batch_size,
            batch_calibration=self.batch_calibration,
            probe_cache_state=probe_state,
            strict=self.settings.embedding_strict,
            fallback_reason=selection.fallback_reason,
            cpu_characters_per_second=_rate(cpu),
            accelerator_characters_per_second=_rate(accelerator),
            accelerator_load_ms=None if accelerator is None else accelerator.load_ns // 1_000_000,
            crossover_characters=measured_crossover,
            recommended_override=self._recommended_override(cpu, accelerator),
        )
