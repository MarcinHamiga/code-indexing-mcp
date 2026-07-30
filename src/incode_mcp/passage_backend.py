"""The passage-embedding session that survives its own accelerator failing.

Indexing hands chunks to one object for the length of a run. Behind that object
the work may start on an accelerator and finish on CPU: an accelerator that
cannot load, cannot compile its graph, cannot pass a minimum-batch inference,
or dies partway through is torn down and the remaining chunks are re-embedded
on CPU. The indexer sees a single embedder throughout and never learns that the
device changed -- which is what keeps a worker crash from becoming a failed run
or a half-written index.

Strict mode turns every one of those degradations into ``BACKEND_UNAVAILABLE``
instead, for callers who would rather fail than quietly index at CPU speed.

The same object also decides *when* the accelerator is worth starting. Below the
measured crossover a run embeds on CPU and never spawns the device worker at
all; the request that carries the run past it retires the CPU worker and brings
the accelerator up for the remainder. That is not a fallback and is not counted
as one -- it is the run being too small to repay a model load, which the next
run re-decides from its own size.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from types import TracebackType

from .backends import CPU_BACKEND, BackendSelection
from .calibration import CalibrationResult, calibrate
from .embedding import EmbeddedSegment, PassageCandidate, SegmentPlan
from .embedding_worker import (
    EmbeddingWorkerSession,
    SessionTelemetry,
)
from .errors import ErrorCode, IncodeError
from .probe_cache import ProbeCache, ProbeKey, ProbeRecord

logger = logging.getLogger(__name__)

# Failures that say something about the backend rather than about the content.
# A worker that exited, a provider that could not initialise, and a run that hit
# its memory ceiling are all reasons to try a different device; a malformed file
# is not, and is left to the indexer to charge against that file.
BACKEND_FAILURE_CODES = frozenset(
    {
        ErrorCode.EMBEDDING_WORKER_FAILED,
        ErrorCode.INDEX_RESOURCE_LIMIT,
        ErrorCode.MODEL_UNAVAILABLE,
    }
)

SessionFactory = Callable[[], EmbeddingWorkerSession]


def _recorded_calibration(record: ProbeRecord) -> CalibrationResult | None:
    """Read back a measurement a previous run stored, if it stored one.

    A record written by a probe that was never measured carries a zero rate,
    which means "unmeasured" rather than "measured as nothing" -- reporting it
    as a calibration would put the crossover beyond every possible run.
    """
    if record.characters_per_second <= 0:
        return None
    return CalibrationResult(
        max_items=record.batch_size,
        characters_per_second=record.characters_per_second,
        load_ns=record.load_ns,
        limited_by=record.limited_by,
    )


def _reason(exc: BaseException) -> str:
    """Render a failure for the fallback reason an ``IndexReport`` carries.

    ``IncodeError.__str__`` deliberately omits details because it is used where
    details travel as their own field. ``embedding_fallback_reason`` has no such
    channel, so rendering with ``str`` here would throw away exactly the numbers
    that explain the fallback -- the ceiling a backend overran and by how much.
    """
    return exc.for_client() if isinstance(exc, IncodeError) else str(exc)


class PassageBackendSession:
    """A passage embedder that degrades from its accelerator to CPU in place."""

    def __init__(
        self,
        selection: BackendSelection,
        *,
        accelerator_factory: SessionFactory,
        cpu_factory: SessionFactory,
        strict: bool = False,
        probe_cache: ProbeCache | None = None,
        probe_key: ProbeKey | None = None,
        cpu_probe_key: ProbeKey | None = None,
        calibrated_batch_size: int = 0,
        dimension: int = 0,
        on_degrade: Callable[[BackendSelection], None] | None = None,
        crossover_characters: int = 0,
        calibration_plan: SegmentPlan | None = None,
        cpu_max_items: int = 0,
    ) -> None:
        self.selection = selection
        self.strict = strict
        self._accelerator_factory = accelerator_factory
        self._cpu_factory = cpu_factory
        self._probe_cache = probe_cache
        self._probe_key = probe_key
        self._cpu_probe_key = cpu_probe_key
        # 0 means "nothing measured a batch size for this backend", which is the
        # answer whenever no calibration plan was supplied. Recording the
        # configured default as though it were a measurement would let
        # ``model status`` claim a calibration that never ran.
        self._calibrated_batch_size = calibrated_batch_size
        # None disables measurement outright. The plan is needed because
        # calibration embeds through the same request path indexing uses, so it
        # has to know the token budgets that path would apply.
        self._calibration_plan = calibration_plan
        self.calibration: CalibrationResult | None = None
        # The microbatch size measured for CPU, which is not the one measured
        # for the accelerator. A run that defers to CPU, or degrades to it, must
        # not be handed the accelerator's: on a device that packs four items to
        # CPU's one, that is a run retrying its way through the whole corpus.
        # 0 leaves whatever the caller planned alone.
        self._cpu_max_items = cpu_max_items
        # Characters this run has embedded, and the size above which starting
        # the accelerator repays its model load. 0 means no measured crossover,
        # which is the pre-calibration behaviour: use the accelerator at once.
        self.characters_embedded = 0
        self.crossover_characters = crossover_characters
        self._on_provisional_cpu = False
        self._dimension = dimension
        self._on_degrade = on_degrade
        self._session: EmbeddingWorkerSession | None = None
        self._on_cpu = not selection.uses_accelerator
        # What actually embedded, which is not what was selected: a run may stay
        # below the crossover and never touch the accelerator the selection
        # names. Reported rather than the selection so a report says where the
        # work happened.
        self.backend_used = (
            CPU_BACKEND.accelerator.value if self._on_cpu else selection.accelerator.value
        )
        # The worker generation verification was last established for. The
        # in-worker batch retry replaces the process without asking, so this is
        # what notices that the survivor has proven nothing yet.
        self._verified_spawn = 0
        self.fallback_count = 0
        # Only a denied request carries a reason into the run. ``auto`` settling
        # on CPU is the correct outcome, not a degradation, and reporting it as
        # one on every ordinary CPU report would make the field meaningless.
        self.fallback_reason: str | None = None if selection.honored else selection.fallback_reason
        self.probe_state = "unprobed"
        # Telemetry from sessions already torn down. A run that started on an
        # accelerator and finished on CPU must still report everything it did,
        # not just what the surviving worker remembers.
        self._retired: list[SessionTelemetry] = []

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> PassageBackendSession:
        # A selection that already failed is fatal here rather than at the
        # first chunk, so strict mode reports the real cause before any work.
        if self.strict:
            self.selection.require_honored()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Terminate the active worker, releasing VRAM or unified memory."""
        session = self._session
        if session is None:
            return
        self._retired.append(session.telemetry())
        self._session = None
        session.close()

    # -- embedding ---------------------------------------------------------

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        pending = sum(len(candidate.prefix) + len(candidate.content) for candidate in candidates)
        return self._attempt(
            lambda session: session.plan_and_embed(candidates, self._plan_for(session, plan)),
            pending,
        )

    def _plan_for(self, session: EmbeddingWorkerSession, plan: SegmentPlan) -> SegmentPlan:
        """Return *plan* packed for the backend that is about to run it."""
        if not self._cpu_max_items or session.config.accelerator != CPU_BACKEND.accelerator.value:
            return plan
        return replace(plan, max_items=self._cpu_max_items)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._attempt(
            lambda session: session.embed_passages(texts), sum(len(text) for text in texts)
        )

    def _attempt[Result](
        self, call: Callable[[EmbeddingWorkerSession], Result], pending: int = 0
    ) -> Result:
        """Run *call* on the active backend, retrying once on CPU if it fails.

        The retry re-runs the whole request rather than resuming it. Every
        request here is one bounded group the indexer has not committed yet, so
        re-embedding it on CPU produces the same rows it would have produced
        had the run started on CPU -- at the cost of repeating that one group.

        *pending* is the size of this request, and it counts towards the
        crossover before the request runs rather than after. Charging only what
        is already embedded would send the one group large enough to justify the
        accelerator to CPU, and then start the accelerator for whatever happened
        to follow it.
        """
        try:
            result = call(self._active(pending))
        except IncodeError as exc:
            if self._on_cpu or self._on_provisional_cpu or exc.code not in BACKEND_FAILURE_CODES:
                # Provisional CPU is what an accelerator falls back *to*, so
                # its failure is the machine failing rather than a reason to
                # reach for the device this run had decided against.
                raise
            # _reason already leads with the code, so it is not prefixed again.
            self._degrade(_reason(exc))
            result = call(self._active(pending))
        self.characters_embedded += pending
        self._record_reduced_batch_size()
        return result

    def _active(self, pending: int = 0) -> EmbeddingWorkerSession:
        if self._on_cpu:
            if self._session is None:
                self._session = self._start_cpu()
            return self._session
        if self.crossover_characters > self.characters_embedded + pending:
            # Too little work to repay a model load. Start on CPU without
            # committing the run to it: the next request may well cross.
            if self._session is None:
                self._on_provisional_cpu = True
                self._session = self._start_cpu()
            return self._session
        if self._on_provisional_cpu:
            # Crossed. close() retires the CPU worker's telemetry, so the run
            # still reports everything both workers did.
            self._on_provisional_cpu = False
            self.close()
        session = self._session
        if session is not None and session.spawn_count == self._verified_spawn:
            return session
        # Either nothing is running yet, or a batch retry closed the worker and
        # the successor is an unproven fresh load on the same device. Both need
        # verification before more content reaches them. (The group whose retry
        # forced the respawn is embedded by that unverified process; only the
        # groups after it are covered here.)
        if session is None:
            session = self._accelerator_factory()
        try:
            self._verify(session)
        except (IncodeError, ValueError) as exc:
            # Nothing this backend produced can be trusted, and it may still be
            # holding device memory. Adopting it first means _degrade's close()
            # both kills it and keeps what it measured -- the ceiling it overran
            # and the reason it was terminated are the evidence for the fallback
            # being reported. Under strict mode _degrade raises before that, and
            # __exit__ reaps it on the way out instead.
            self._session = session
            self.probe_state = "failed"
            self._degrade(_reason(exc))
            self._session = self._start_cpu()
            return self._session
        self._session = session
        self.backend_used = self.selection.descriptor.accelerator.value
        return session

    def _start_cpu(self) -> EmbeddingWorkerSession:
        """Open the CPU worker and record that this run has now used it."""
        self.backend_used = CPU_BACKEND.accelerator.value
        return self._cpu_factory()

    def _verify(self, session: EmbeddingWorkerSession) -> None:
        """Load the accelerator's model and confirm it can really embed.

        A cached probe for this exact configuration skips the inference but
        never the load: the load is what proves the provider still initialises
        on this boot, with these drivers.
        """
        info = session.initialize()
        descriptor = self.selection.descriptor
        if info.resolved_providers and descriptor.provider not in info.resolved_providers:
            # ONNX Runtime drops a provider it cannot initialise and keeps
            # running on the next one, so a "CUDA" session that quietly became
            # a CPU session would otherwise look like a successful selection.
            raise IncodeError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"{descriptor.provider} was requested but the session runs on "
                f"{', '.join(info.resolved_providers)}",
                requested=descriptor.provider,
                resolved=list(info.resolved_providers),
            )
        if info.dimension != session.config.dimension:
            raise IncodeError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"{descriptor.accelerator.value} reported dimension {info.dimension}, "
                f"expected {session.config.dimension}",
            )
        cached = self._cached_probe()
        if cached is not None:
            self.probe_state = "cached"
            self._verified_spawn = session.spawn_count
            self.calibration = _recorded_calibration(cached)
            return
        session.probe()
        self.probe_state = "verified"
        self._verified_spawn = session.spawn_count
        self._measure(session)
        self._record_probe()
        self._measure_reference()

    def _measure(self, session: EmbeddingWorkerSession) -> None:
        """Measure the backend that has just proven it works.

        Here rather than anywhere else because this is the one moment a loaded,
        verified, otherwise idle worker exists: measuring later would compete
        with real content for the same ceiling, and measuring earlier would time
        a backend that had not yet been shown to embed at all.
        """
        if self._calibration_plan is None:
            return
        self.calibration = calibrate(
            session, self._calibration_plan, load_ns=session.load_duration_ns
        )
        if self.calibration is not None:
            self._calibrated_batch_size = self.calibration.max_items

    def _measure_reference(self) -> None:
        """Measure CPU too, so the crossover has both of the rates it needs.

        CPU otherwise only ever runs here as a fallback, and a machine whose
        accelerator never fails would never measure the backend every decision
        is made against. This costs one CPU worker and one model load, once per
        configuration -- paid on the run that first verified an accelerator, and
        never on a machine that has none.
        """
        if self._calibration_plan is None or self._probe_cache is None:
            return
        key = self._cpu_probe_key
        if key is None or self._probe_cache.load(key) is not None:
            return
        session = self._cpu_factory()
        try:
            session.initialize()
            measured = calibrate(session, self._calibration_plan, load_ns=session.load_duration_ns)
        except (IncodeError, ValueError) as exc:
            # The reference backend failing to measure is a diagnostic gap, not
            # a run-ending condition: without a crossover the accelerator is
            # simply used as it was before any of this existed.
            logger.debug("Could not measure the CPU reference backend: %s", exc)
            return
        finally:
            session.close()
        if measured is None:
            return
        self._probe_cache.store(
            key,
            batch_size=measured.max_items,
            dimension=self._dimension,
            detail=CPU_BACKEND.provider,
            characters_per_second=measured.characters_per_second,
            load_ns=measured.load_ns,
            limited_by=measured.limited_by,
        )

    def _record_reduced_batch_size(self) -> None:
        """Persist a microbatch size a ceiling overrun forced down.

        Without this the next run asks for the size that overran, overruns
        again, and pays the same retries to arrive back here.
        """
        session = self._session
        if (
            session is None
            or self._on_cpu
            or self._on_provisional_cpu
            or self._probe_cache is None
            or self._probe_key is None
            or not session.safe_max_items
            or session.safe_max_items == self._calibrated_batch_size
        ):
            return
        self._calibrated_batch_size = session.safe_max_items
        if self.calibration is not None:
            self.calibration = replace(
                self.calibration, max_items=session.safe_max_items, limited_by="memory"
            )
        self._record_probe(limited_by="memory")

    def _cached_probe(self) -> ProbeRecord | None:
        if self._probe_cache is None or self._probe_key is None:
            return None
        return self._probe_cache.load(self._probe_key)

    def _record_probe(self, *, limited_by: str = "") -> None:
        if self._probe_cache is None or self._probe_key is None:
            return
        measured = self.calibration
        self._probe_cache.store(
            self._probe_key,
            batch_size=self._calibrated_batch_size,
            dimension=self._dimension,
            detail=self.selection.descriptor.provider,
            characters_per_second=0.0 if measured is None else measured.characters_per_second,
            load_ns=0 if measured is None else measured.load_ns,
            limited_by=limited_by or (measured.limited_by if measured else ""),
        )

    def _degrade(self, reason: str) -> None:
        """Record a backend failure and move the run onto CPU."""
        if self.strict:
            raise IncodeError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"Embedding accelerator {self.selection.accelerator.value} failed and "
                f"INCODE_EMBED_STRICT forbids the CPU fallback: {reason}",
                requested=self.selection.requested.value,
                accelerator=self.selection.accelerator.value,
                reason=reason,
            )
        logger.warning(
            "Embedding accelerator %s failed (%s); continuing on CPU",
            self.selection.accelerator.value,
            reason,
        )
        self.close()
        self.selection = self.selection.fell_back_to(CPU_BACKEND, reason)
        self.fallback_reason = reason
        self.fallback_count += 1
        self._on_cpu = True
        self._on_provisional_cpu = False
        self._verified_spawn = 0
        if self._on_degrade is not None:
            # Lets the owner stop paying for this backend again. A device model
            # load is seconds, and without this a long-lived daemon would spawn,
            # load, and terminate the same dead accelerator on every index run.
            self._on_degrade(self.selection)

    # -- telemetry ---------------------------------------------------------

    def telemetry(self) -> SessionTelemetry:
        entries = self._all_telemetry()
        termination = next(
            (entry.termination_reason for entry in reversed(entries) if entry.termination_reason),
            None,
        )
        tokenizer = next(
            (
                entry.tokenizer_available
                for entry in entries
                if entry.tokenizer_available is not None
            ),
            None,
        )
        return SessionTelemetry(
            backend=self.backend_used,
            memory_budget_bytes=max([entry.memory_budget_bytes for entry in entries] or [0]),
            peak_memory_bytes=max([entry.peak_memory_bytes for entry in entries] or [0]),
            segment_count=sum(entry.segment_count for entry in entries),
            token_count=sum(entry.token_count for entry in entries),
            retry_count=sum(entry.retry_count for entry in entries),
            fallback_count=sum(entry.retry_count for entry in entries) + self.fallback_count,
            termination_reason=termination,
            tokenizer_available=tokenizer,
            fallback_reason=self.fallback_reason,
            character_count=self.characters_embedded,
            crossover_characters=self.crossover_characters,
            selection_reason=self._selection_reason(),
        )

    def _selection_reason(self) -> str | None:
        """Say why the run embedded where it did, when a crossover decided it.

        Only the crossover is explained here. A degradation already reports
        itself through ``fallback_reason``, and a run that simply used what was
        selected needs no explanation at all.
        """
        if not self.crossover_characters or self._on_cpu:
            return None
        if self._on_provisional_cpu or self.backend_used == CPU_BACKEND.accelerator.value:
            return (
                f"embedded {self.characters_embedded} characters, below the "
                f"{self.crossover_characters}-character crossover for "
                f"{self.selection.accelerator.value}"
            )
        return (
            f"passed the {self.crossover_characters}-character crossover for "
            f"{self.selection.accelerator.value}"
        )

    def _all_telemetry(self) -> list[SessionTelemetry]:
        live = [self._session.telemetry()] if self._session is not None else []
        return [*self._retired, *live]
