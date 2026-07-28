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
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from types import TracebackType

from .backends import CPU_BACKEND, BackendSelection
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
        calibrated_batch_size: int = 0,
        dimension: int = 0,
        on_degrade: Callable[[BackendSelection], None] | None = None,
    ) -> None:
        self.selection = selection
        self.strict = strict
        self._accelerator_factory = accelerator_factory
        self._cpu_factory = cpu_factory
        self._probe_cache = probe_cache
        self._probe_key = probe_key
        # 0 means "nothing measured a batch size for this backend". Calibration
        # is Phase 5's job; recording the configured default as though it were
        # a measurement would let ``model status`` claim a calibration that
        # never ran.
        self._calibrated_batch_size = calibrated_batch_size
        self._dimension = dimension
        self._on_degrade = on_degrade
        self._session: EmbeddingWorkerSession | None = None
        self._on_cpu = not selection.uses_accelerator
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
        return self._attempt(lambda session: session.plan_and_embed(candidates, plan))

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._attempt(lambda session: session.embed_passages(texts))

    def _attempt[Result](self, call: Callable[[EmbeddingWorkerSession], Result]) -> Result:
        """Run *call* on the active backend, retrying once on CPU if it fails.

        The retry re-runs the whole request rather than resuming it. Every
        request here is one bounded group the indexer has not committed yet, so
        re-embedding it on CPU produces the same rows it would have produced
        had the run started on CPU -- at the cost of repeating that one group.
        """
        try:
            return call(self._active())
        except IncodeError as exc:
            if self._on_cpu or exc.code not in BACKEND_FAILURE_CODES:
                raise
            # _reason already leads with the code, so it is not prefixed again.
            self._degrade(_reason(exc))
        return call(self._active())

    def _active(self) -> EmbeddingWorkerSession:
        if self._on_cpu:
            if self._session is None:
                self._session = self._cpu_factory()
            return self._session
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
            self._session = self._cpu_factory()
            return self._session
        self._session = session
        return session

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
            return
        session.probe()
        self.probe_state = "verified"
        self._verified_spawn = session.spawn_count
        self._record_probe()

    def _cached_probe(self) -> ProbeRecord | None:
        if self._probe_cache is None or self._probe_key is None:
            return None
        return self._probe_cache.load(self._probe_key)

    def _record_probe(self) -> None:
        if self._probe_cache is None or self._probe_key is None:
            return
        self._probe_cache.store(
            self._probe_key,
            batch_size=self._calibrated_batch_size,
            dimension=self._dimension,
            detail=self.selection.descriptor.provider,
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
            backend=self.selection.accelerator.value,
            memory_budget_bytes=max([entry.memory_budget_bytes for entry in entries] or [0]),
            peak_memory_bytes=max([entry.peak_memory_bytes for entry in entries] or [0]),
            segment_count=sum(entry.segment_count for entry in entries),
            token_count=sum(entry.token_count for entry in entries),
            retry_count=sum(entry.retry_count for entry in entries),
            fallback_count=sum(entry.retry_count for entry in entries) + self.fallback_count,
            termination_reason=termination,
            tokenizer_available=tokenizer,
            fallback_reason=self.fallback_reason,
        )

    def _all_telemetry(self) -> list[SessionTelemetry]:
        live = [self._session.telemetry()] if self._session is not None else []
        return [*self._retired, *live]
