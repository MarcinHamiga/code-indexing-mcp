"""One-time measurement of what a backend can actually do.

A probe answers "does this work". Nothing until now answered "how fast, at what
batch size, and is starting it worth the wait" -- so batch size was configured
rather than measured, and an accelerator was used for a one-file re-index it
could not possibly repay.

Measurement runs from the parent through the ordinary ``plan_and_embed``
request, against a synthetic corpus. Nothing is added to the worker protocol,
which means the numbers include the IPC round trip the real work pays and the
same code measures the FastEmbed, direct-ONNX, and MLX workers alike.

Throughput is in characters per second because that is the only unit the
crossover decision can be made in: segments are known only after windowing,
tokens only after embedding, and characters are exactly known for every
candidate the indexer is about to hand over. Character density varies between
corpora -- the caveat :mod:`code_indexing_mcp.token_batching` documents -- so the
corpus here is code-shaped, and the decision it feeds is whether a run is
seconds or minutes rather than a precise prediction.

Everything is diagnostics. A backend that cannot be measured is simply
uncalibrated: failing here would fail a run over a measurement it only wanted
in order to go faster.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from .embedding import EmbeddedSegment, PassageCandidate, SegmentPlan
from .errors import CodeIndexingError, ErrorCode
from .settings import MAX_BATCH_SIZE

logger = logging.getLogger(__name__)

# The ladder the sweep walks. Doubling means a backend that scales reaches its
# plateau in a handful of measurements rather than one per size.
CANDIDATE_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
# Candidates per measurement. Enough that the largest size is a real batch and
# the model is past its first-call warm-up, few enough that the whole sweep is
# seconds on CPU -- and it is paid once per configuration, not once per run.
CALIBRATION_CANDIDATE_COUNT = 16
# Two sequence shapes: an ordinary function and something closer to a whole
# class body. Measuring only one would calibrate for a corpus shape the real
# one may not have.
REPRESENTATIVE_CHARACTERS = (384, 1536)
# A larger batch has to beat the best rate by this much to justify the memory it
# holds. Anything smaller is noise being paid for in resident bytes.
IMPROVEMENT_RATIO = 1.05
# How many sizes an accelerator's sweep looks past one that did not pay. A GPU
# that is bound by compute rather than by dispatch gains from batching a little
# at every doubling instead of a lot at one: float16 MLX on an M4 Pro measured
# 58.8k, 62.3k, and 65.0k characters per second at 2, 4, and 8, so each step sat
# on the threshold while the three together were worth 10%, and a sweep that
# stopped at the first miss settled on 2 or 8 depending on noise. One more size
# is enough to see the trend; CPU keeps the strict rule, because its sweep is a
# reference measurement and every extra size is one more chance to overrun.
ACCELERATOR_SWEEP_PATIENCE = 1
# Overrunning the ceiling is the ceiling being reported; a device allocation
# that kills the worker outright is not, and arrives as a worker failure. Both
# say the same thing about the size that provoked them, so they are told apart
# only in what they are recorded as -- a memory ceiling has a setting behind it
# to recommend, and a crash does not.
RESOURCE_CODES = frozenset({ErrorCode.INDEX_RESOURCE_LIMIT})
LIMITED_BY_MEMORY = "memory"
LIMITED_BY_FAILURE = "failure"


class MeasurableSession(Protocol):
    """Anything that can embed a planned group -- a worker or passage session."""

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]: ...


@dataclass(frozen=True)
class CalibrationResult:
    """What one backend measured, on one machine, for one configuration."""

    max_items: int
    characters_per_second: float
    load_ns: int
    # "memory" when a larger batch overran the ceiling and "failure" when one
    # took the worker down with it, so a reader can tell a size that won on
    # speed from one that won by being the last that survived -- and so a
    # recommendation is only made where there is a setting behind it.
    limited_by: str = ""
    # The largest size that ran, when the size after it failed; 0 when the
    # sweep ended without a failure. It is kept apart from ``limited_by``
    # because a sweep looking past a size that did not pay can fail above a
    # size that won on speed: that is not a limit on the winner, and it is
    # still the only evidence of where this backend stops for anything that
    # would ask it for more -- an explicit batch size, say.
    survived: int = 0

    @property
    def load_ms(self) -> int:
        return self.load_ns // 1_000_000


def calibration_candidates(
    *,
    count: int = CALIBRATION_CANDIDATE_COUNT,
    lengths: Sequence[int] = REPRESENTATIVE_CHARACTERS,
) -> list[PassageCandidate]:
    """Build the deterministic, code-shaped corpus the sweep is timed against.

    Deterministic because a calibration two runs disagree on is not a
    calibration, and code-shaped because tokenization density is exactly what
    separates a character count from the work it implies.
    """
    candidates: list[PassageCandidate] = []
    for index in range(count):
        target = lengths[index % len(lengths)]
        body: list[str] = [f"def calibration_{index:03d}(values: list[int]) -> int:"]
        line = 0
        while sum(len(entry) + 1 for entry in body) < target:
            body.append(
                f"    total_{line:03d} = sum(value * {line + 2} for value in values[:{line + 1}])"
            )
            line += 1
        content = "\n".join(body)
        candidates.append(PassageCandidate(f"# module calibration_{index:03d}", content[:target]))
    return candidates


def calibrate(
    session: MeasurableSession,
    plan: SegmentPlan,
    *,
    load_ns: int,
    max_items: int = MAX_BATCH_SIZE,
    candidates: Sequence[PassageCandidate] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
    patience: int = 0,
) -> CalibrationResult | None:
    """Measure *session*'s throughput and settle on a batch size.

    *patience* is how many consecutive sizes may fail to beat the best rate
    before the sweep stops. A size tried on patience is adopted only if it beats
    the best by the usual margin, so patience can find a later gain but never
    lowers the bar for accepting one.

    Returns ``None`` when nothing could be measured, which is the honest answer
    for a backend whose first measurement failed: there is no smaller size to
    fall back to, and reporting the size that overran would recommend exactly
    the batch that did.
    """
    corpus = list(calibration_candidates()) if candidates is None else list(candidates)
    # Prefix included, because the crossover charges a request its prefixes too:
    # a rate denominated in content alone would be compared against a size
    # counted in content plus prefix, and cross fractionally early.
    characters = sum(len(candidate.prefix) + len(candidate.content) for candidate in corpus)
    best: CalibrationResult | None = None
    misses = 0
    survived = 0
    for size in CANDIDATE_BATCH_SIZES:
        if size > max_items:
            break
        started = clock()
        try:
            session.plan_and_embed(corpus, replace(plan, max_items=size))
        except CodeIndexingError as exc:
            if best is None:
                logger.debug("Calibration abandoned at max_items=%d: %s", size, exc)
                return None
            # Verification already loaded this backend and made it embed, so a
            # failure that appears only once the batch grows is the size failing
            # rather than the backend -- and every smaller size measured here
            # returned vectors before it. Keeping them is the difference between
            # a calibrated backend and none at all on a machine whose device
            # allocations die rather than trip the ceiling.
            logger.debug("Calibration stopped at max_items=%d: %s", size, exc.code.value)
            if misses:
                # A size that already ran had stopped paying before this one
                # failed, so the best size won on speed. Recording it as
                # limited would recommend more memory for a batch that more
                # memory was already shown not to help. Where it failed is kept
                # all the same, for a caller about to ask for more than that.
                return replace(best, survived=survived)
            limited_by = LIMITED_BY_MEMORY if exc.code in RESOURCE_CODES else LIMITED_BY_FAILURE
            return replace(best, limited_by=limited_by, survived=survived)
        except Exception as exc:  # pragma: no cover - defensive, see module docstring
            logger.debug("Calibration abandoned at max_items=%d: %s", size, exc)
            if best is None:
                return None
            if misses:
                return replace(best, survived=survived)
            return replace(best, limited_by=LIMITED_BY_FAILURE, survived=survived)
        # A clock with any granularity at all can report a zero interval for a
        # fast enough backend, and dividing by it would report infinite speed.
        elapsed = max(1, clock() - started)
        survived = size
        rate = characters * 1_000_000_000 / elapsed
        if best is None:
            best = CalibrationResult(max_items=size, characters_per_second=rate, load_ns=load_ns)
            continue
        if rate <= best.characters_per_second * IMPROVEMENT_RATIO:
            # This size is not paying for the memory it holds. Without patience
            # neither will the ones above it be asked to: doubling again only
            # widens the padded matrix.
            misses += 1
            if misses > patience:
                break
            continue
        misses = 0
        best = CalibrationResult(max_items=size, characters_per_second=rate, load_ns=load_ns)
    return best


def crossover_characters(
    *,
    accelerator_load_ns: int,
    cpu_load_ns: int,
    cpu_characters_per_second: float,
    accelerator_characters_per_second: float,
) -> int | None:
    """Return the run size above which starting the accelerator pays for itself.

    Both policies pay for a worker: staying on CPU costs ``L_cpu + n / R_cpu``
    and using the accelerator costs ``L_accel + n / R_accel``, so what the
    accelerator has to earn back is the *difference* between the two loads, not
    its whole load. Charging it the whole load would defer runs on a machine
    where the accelerator loads faster than CPU -- which is the ordinary case on
    unified memory, where the device model is memory-mapped and the CPU one is
    an ONNX graph being read and prepared.

    ``0`` means the accelerator is worth starting immediately. ``None`` means the
    two never meet: an accelerator no faster than CPU has no size at which
    starting it wins, and must not be reported as a threshold a large enough run
    would eventually pass.
    """
    if cpu_characters_per_second <= 0 or accelerator_characters_per_second <= 0:
        return None
    if accelerator_characters_per_second <= cpu_characters_per_second:
        return None
    extra_load_ns = accelerator_load_ns - cpu_load_ns
    if extra_load_ns <= 0:
        return 0
    saved_per_character = 1 / cpu_characters_per_second - 1 / accelerator_characters_per_second
    return int(extra_load_ns / 1_000_000_000 / saved_per_character)
