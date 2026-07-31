"""Batch calibration and the workload crossover, against a programmed session."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from code_indexing_mcp.calibration import (
    CANDIDATE_BATCH_SIZES,
    CalibrationResult,
    calibrate,
    calibration_candidates,
    crossover_characters,
)
from code_indexing_mcp.embedding import EmbeddedSegment, PassageCandidate, SegmentPlan
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode

PLAN = SegmentPlan(max_items=1)


class FakeSession:
    """A session whose every request takes a programmed time for its batch size."""

    def __init__(self, nanoseconds_per_call: dict[int, int]) -> None:
        self.nanoseconds_per_call = nanoseconds_per_call
        self.now = 0
        self.batch_sizes: list[int] = []

    def clock(self) -> int:
        return self.now

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        self.batch_sizes.append(plan.max_items)
        self.now += self._duration(plan.max_items)
        return [[EmbeddedSegment(0, len(candidate.content), 0, b"")] for candidate in candidates]

    def _duration(self, max_items: int) -> int:
        return self.nanoseconds_per_call[max_items]


class ExhaustedSession(FakeSession):
    """A session that overruns its memory ceiling above a given batch size."""

    def __init__(self, nanoseconds_per_call: dict[int, int], *, fails_above: int) -> None:
        super().__init__(nanoseconds_per_call)
        self.fails_above = fails_above

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        if plan.max_items > self.fails_above:
            self.batch_sizes.append(plan.max_items)
            raise CodeIndexingError(
                ErrorCode.INDEX_RESOURCE_LIMIT, "Indexing exceeded its memory ceiling"
            )
        return super().plan_and_embed(candidates, plan)


def _halving(nanoseconds: int) -> dict[int, int]:
    """Each doubling of the batch size halves the time, to the last size."""
    return {size: max(1, nanoseconds // size) for size in CANDIDATE_BATCH_SIZES}


# -- the calibration corpus ------------------------------------------------


def test_the_calibration_corpus_is_deterministic_and_code_shaped() -> None:
    first = calibration_candidates()
    second = calibration_candidates()

    assert [candidate.content for candidate in first] == [candidate.content for candidate in second]
    assert any("def " in candidate.content for candidate in first)
    # Two representative lengths, so the measurement is not taken entirely on
    # one sequence shape the real corpus may not have.
    assert len({len(candidate.content) for candidate in first}) >= 2


# -- the batch sweep -------------------------------------------------------


def test_the_fastest_batch_size_is_the_calibrated_one() -> None:
    session = FakeSession(_halving(1_000_000_000))

    result = calibrate(session, PLAN, load_ns=0, clock=session.clock)

    assert result is not None
    assert result.max_items == CANDIDATE_BATCH_SIZES[-1]
    assert result.limited_by == ""


def test_the_sweep_stops_once_a_larger_batch_stops_paying() -> None:
    """Every size past the second is no faster, and each one costs memory, so
    the sweep must not walk the whole ladder to prove it."""
    durations = dict.fromkeys(CANDIDATE_BATCH_SIZES, 1_000_000_000)
    durations[2] = 400_000_000
    session = FakeSession(durations)

    result = calibrate(session, PLAN, load_ns=0, clock=session.clock)

    assert result is not None
    assert result.max_items == 2
    assert session.batch_sizes == [1, 2, 4]


def test_a_batch_size_that_overruns_the_ceiling_is_not_calibrated() -> None:
    session = ExhaustedSession(_halving(1_000_000_000), fails_above=4)

    result = calibrate(session, PLAN, load_ns=0, clock=session.clock)

    assert result is not None
    assert result.max_items == 4
    assert result.limited_by == "memory"


def test_the_first_batch_size_overrunning_leaves_nothing_calibrated() -> None:
    """There is no measured size below the one that failed, and reporting the
    failed size would recommend exactly the batch that overran."""
    session = ExhaustedSession(_halving(1_000_000_000), fails_above=0)

    assert calibrate(session, PLAN, load_ns=0, clock=session.clock) is None


def test_a_session_that_fails_outright_is_not_calibrated() -> None:
    """Calibration is diagnostics. A backend that cannot embed has to fail where
    it is verified, not here, and never by escaping this call."""

    class BrokenSession(FakeSession):
        def plan_and_embed(
            self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
        ) -> list[list[EmbeddedSegment]]:
            raise CodeIndexingError(ErrorCode.EMBEDDING_WORKER_FAILED, "the worker died")

    assert calibrate(BrokenSession({}), PLAN, load_ns=0, clock=lambda: 0) is None


def test_a_batch_that_kills_the_worker_keeps_what_smaller_ones_measured() -> None:
    """A device allocation that dies rather than tripping the ceiling arrives as
    a worker failure, but says the same thing about the size that provoked it.
    Verification already made this backend embed, and every size below returned
    vectors -- discarding them leaves the machine uncalibrated forever."""

    class DyingSession(FakeSession):
        def plan_and_embed(
            self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
        ) -> list[list[EmbeddedSegment]]:
            if plan.max_items > 4:
                raise CodeIndexingError(ErrorCode.EMBEDDING_WORKER_FAILED, "the worker died")
            return super().plan_and_embed(candidates, plan)

    session = DyingSession(_halving(1_000_000_000))

    result = calibrate(session, PLAN, load_ns=0, clock=session.clock)

    assert result is not None
    assert result.max_items == 4
    # Not "memory": raising the ceiling is not what answers an allocation the
    # device could not make, so no override is recommended for it.
    assert result.limited_by == "failure"


def test_the_measured_rate_counts_the_same_characters_the_crossover_does() -> None:
    """Prefix included: a request is charged its prefixes against the threshold,
    so a rate denominated in content alone would cross fractionally early."""
    session = FakeSession(dict.fromkeys(CANDIDATE_BATCH_SIZES, 2_000_000_000))
    candidates = calibration_candidates()
    characters = sum(len(candidate.prefix) + len(candidate.content) for candidate in candidates)
    assert characters > sum(len(candidate.content) for candidate in candidates)

    result = calibrate(session, PLAN, load_ns=0, clock=session.clock)

    assert result is not None
    # One size is measured: the second is no faster, so the sweep stops there.
    assert result.characters_per_second == pytest.approx(characters / 2.0)


def test_the_calibrated_size_never_exceeds_the_configured_maximum() -> None:
    session = FakeSession(_halving(1_000_000_000))

    result = calibrate(session, PLAN, load_ns=0, clock=session.clock, max_items=4)

    assert result is not None
    assert result.max_items == 4
    assert max(session.batch_sizes) == 4


# -- the crossover ---------------------------------------------------------


def test_the_crossover_is_where_startup_stops_costing_more_than_it_saves() -> None:
    # 2 s more startup than CPU costs, and the accelerator embeds twice as fast:
    # 1,000 chars/s against 2,000. Below 4,000 characters CPU finishes first.
    assert (
        crossover_characters(
            accelerator_load_ns=2_000_000_000,
            cpu_load_ns=0,
            cpu_characters_per_second=1_000.0,
            accelerator_characters_per_second=2_000.0,
        )
        == 4_000
    )


def test_only_the_startup_the_accelerator_costs_beyond_cpu_has_to_be_earned_back() -> None:
    """Staying on CPU also spawns a worker and loads a model. Charging the
    accelerator for that shared cost would defer runs that were never slower."""
    assert (
        crossover_characters(
            accelerator_load_ns=3_000_000_000,
            cpu_load_ns=1_000_000_000,
            cpu_characters_per_second=1_000.0,
            accelerator_characters_per_second=2_000.0,
        )
        == 4_000
    )


def test_an_accelerator_that_loads_faster_than_cpu_is_worth_starting_at_once() -> None:
    """A memory-mapped device model against an ONNX graph being prepared: there
    is nothing to earn back, so there is nothing to defer."""
    assert (
        crossover_characters(
            accelerator_load_ns=370_000_000,
            cpu_load_ns=655_000_000,
            cpu_characters_per_second=14_030.0,
            accelerator_characters_per_second=46_783.0,
        )
        == 0
    )


def test_an_accelerator_no_faster_than_cpu_has_no_crossover() -> None:
    """There is no size at which starting it wins, so it must not be offered as
    a threshold that a large enough run would eventually pass."""
    assert (
        crossover_characters(
            accelerator_load_ns=2_000_000_000,
            cpu_load_ns=0,
            cpu_characters_per_second=2_000.0,
            accelerator_characters_per_second=2_000.0,
        )
        is None
    )


def test_a_free_start_crosses_over_immediately() -> None:
    assert (
        crossover_characters(
            accelerator_load_ns=0,
            cpu_load_ns=0,
            cpu_characters_per_second=1_000.0,
            accelerator_characters_per_second=2_000.0,
        )
        == 0
    )


def test_an_unmeasured_rate_has_no_crossover() -> None:
    assert (
        crossover_characters(
            accelerator_load_ns=1_000_000_000,
            cpu_load_ns=0,
            cpu_characters_per_second=0.0,
            accelerator_characters_per_second=2_000.0,
        )
        is None
    )


def test_a_calibration_result_reports_what_it_measured() -> None:
    result = CalibrationResult(
        max_items=8, characters_per_second=1_234.5, load_ns=2_000_000_000, limited_by=""
    )

    assert result.load_ms == 2_000
