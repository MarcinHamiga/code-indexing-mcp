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
corpora -- the caveat :mod:`incode_mcp.token_batching` documents -- so the
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
from .errors import ErrorCode, IncodeError
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
# Overrunning the ceiling says this size is unsafe, not that the backend is
# broken. Anything else is the backend failing, which belongs to verification.
RESOURCE_CODES = frozenset({ErrorCode.INDEX_RESOURCE_LIMIT})


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
    # "memory" when a larger batch overran the ceiling, so a reader can tell a
    # size that won on speed from one that won by being the last that fit.
    limited_by: str = ""

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
) -> CalibrationResult | None:
    """Measure *session*'s throughput and settle on a batch size.

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
    for size in CANDIDATE_BATCH_SIZES:
        if size > max_items:
            break
        started = clock()
        try:
            session.plan_and_embed(corpus, replace(plan, max_items=size))
        except IncodeError as exc:
            if exc.code in RESOURCE_CODES and best is not None:
                logger.debug("Calibration stopped at max_items=%d: %s", size, exc.code.value)
                return replace(best, limited_by="memory")
            logger.debug("Calibration abandoned at max_items=%d: %s", size, exc)
            return None
        except Exception as exc:  # pragma: no cover - defensive, see module docstring
            logger.debug("Calibration abandoned at max_items=%d: %s", size, exc)
            return None
        # A clock with any granularity at all can report a zero interval for a
        # fast enough backend, and dividing by it would report infinite speed.
        elapsed = max(1, clock() - started)
        rate = characters * 1_000_000_000 / elapsed
        if best is None:
            best = CalibrationResult(max_items=size, characters_per_second=rate, load_ns=load_ns)
            continue
        if rate <= best.characters_per_second * IMPROVEMENT_RATIO:
            # This size is not paying for the memory it holds, and neither will
            # the ones above it: doubling again only widens the padded matrix.
            break
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
