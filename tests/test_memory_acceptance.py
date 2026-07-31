"""Acceptance criteria for indexing memory, validated without a model.

Only ``evaluate_result`` is exercised here; it is deliberately pure so ordinary
CI can check the criteria from synthetic fixtures. The real-model run lives
behind the ``memory`` marker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from benchmark_index_memory import (
    MIB,
    BenchmarkRun,
    Sample,
    evaluate_result,
    run_benchmark,
    write_corpus,
)

CEILING = 2048 * MIB


def make_run(
    *,
    combined_over: float = 0.0,
    breach_bytes: int = 0,
    worker_exit_seconds: float = 0.0,
    parent_bytes: int = 200 * MIB,
    samples: tuple[Sample, ...] | None = None,
    corpus_scale: int = 1,
    exit_code: int = 0,
) -> BenchmarkRun:
    """Build a run whose combined RSS breaches the cap for *combined_over* seconds."""
    if samples is None:
        built: list[Sample] = []
        for index in range(30):
            at = index * 0.1
            over = breach_bytes if at < combined_over else 0
            built.append(
                Sample(
                    at=at,
                    parent_bytes=parent_bytes,
                    worker_bytes=CEILING - parent_bytes + over,
                    available_bytes=4096 * MIB,
                )
            )
        samples = tuple(built)
    return BenchmarkRun(
        corpus_scale=corpus_scale,
        corpus_bytes=1_000_000,
        batch_size=1,
        configured_ceiling_bytes=CEILING,
        effective_ceiling_bytes=CEILING,
        samples=samples,
        exit_code=exit_code,
        duration_seconds=3.0,
        worker_exit_seconds=worker_exit_seconds,
    )


def test_combined_rss_at_the_allowance_passes() -> None:
    """The allowance is inclusive: exactly ceiling + 256 MiB is acceptable."""
    samples = tuple(
        Sample(
            at=index * 0.1,
            parent_bytes=200 * MIB,
            worker_bytes=CEILING + 256 * MIB - 200 * MIB,
            available_bytes=4096 * MIB,
        )
        for index in range(30)
    )

    verdict = evaluate_result(make_run(samples=samples))

    assert verdict.valid
    assert verdict.passed
    assert verdict.reasons == ()


def test_a_breach_longer_than_one_second_fails() -> None:
    brief = evaluate_result(make_run(combined_over=0.9, breach_bytes=512 * MIB))
    sustained = evaluate_result(make_run(combined_over=2.0, breach_bytes=512 * MIB))

    assert brief.passed
    assert not sustained.passed
    assert any("combined RSS" in reason for reason in sustained.reasons)


def test_a_worker_alive_two_seconds_after_completion_fails() -> None:
    prompt = evaluate_result(make_run(worker_exit_seconds=1.5))
    lingering = evaluate_result(make_run(worker_exit_seconds=2.5))

    assert prompt.passed
    assert not lingering.passed
    assert any("still alive" in reason for reason in lingering.reasons)


def test_parent_growth_between_corpus_scales_is_bounded() -> None:
    baseline = make_run(parent_bytes=200 * MIB, corpus_scale=1)
    steady = make_run(parent_bytes=300 * MIB, corpus_scale=10)
    leaking = make_run(parent_bytes=400 * MIB, corpus_scale=10)

    assert evaluate_result(steady, baseline=baseline).passed
    verdict = evaluate_result(leaking, baseline=baseline)
    assert not verdict.passed
    assert any("parent RSS grew" in reason for reason in verdict.reasons)


def test_fewer_than_two_samples_is_an_invalid_benchmark() -> None:
    single = make_run(samples=(Sample(at=0.0, parent_bytes=1, worker_bytes=1, available_bytes=1),))

    verdict = evaluate_result(single)

    assert not verdict.valid
    assert not verdict.passed
    assert any("invalid benchmark" in reason for reason in verdict.reasons)


def test_a_failed_index_never_passes() -> None:
    verdict = evaluate_result(make_run(exit_code=2))

    assert verdict.valid
    assert not verdict.passed
    assert any("exited with code 2" in reason for reason in verdict.reasons)


def test_run_survives_a_json_round_trip() -> None:
    run = make_run(combined_over=0.5, breach_bytes=64 * MIB, worker_exit_seconds=0.3)

    restored = BenchmarkRun.from_json(run.to_json())

    assert len(restored.samples) == len(run.samples)
    # Sample timestamps are rounded to 4 places on the way out, so they compare
    # approximately; every byte count and the verdict must survive exactly.
    assert all(
        left.at == pytest.approx(right.at, abs=1e-4)
        and (left.parent_bytes, left.worker_bytes, left.available_bytes)
        == (right.parent_bytes, right.worker_bytes, right.available_bytes)
        for left, right in zip(restored.samples, run.samples, strict=True)
    )
    assert restored.peak_combined_bytes == run.peak_combined_bytes
    assert evaluate_result(restored) == evaluate_result(run)


def test_corpus_includes_the_shapes_that_drive_the_peak(tmp_path: Path) -> None:
    """Ordinary files alone never reach the peak; these three shapes do."""
    total = write_corpus(tmp_path / "corpus", scale=1)

    package = tmp_path / "corpus" / "src" / "benchmark"
    near_cap = (package / "near_cap.py").read_text()
    minified = (package / "minified.py").read_text()
    blank_run = (package / "blank_run.py").read_text()

    assert 900_000 < len(near_cap.encode()) < 1_048_576
    assert 900_000 < len(minified.encode()) < 1_048_576
    assert minified.count("\n") == 1
    assert max(len(line) for line in blank_run.splitlines()) > 4_096
    assert blank_run.count("\n\n\n") > 0
    assert total > len(near_cap.encode()) + len(minified.encode())


def _real_model_run(tmp_path: Path, shapes: list[str]) -> BenchmarkRun:
    cache = os.environ.get("CODE_INDEXING_MODEL_TEST_CACHE")
    if not cache:
        pytest.skip("set CODE_INDEXING_MODEL_TEST_CACHE to run the real-model memory gate")
    root = tmp_path / "corpus"
    corpus_bytes = write_corpus(root, scale=1, shapes=shapes)
    return run_benchmark(
        root=root,
        data_directory=tmp_path / "data",
        cache_directory=Path(cache),
        corpus_scale=1,
        corpus_bytes=corpus_bytes,
        batch_size=1,
        memory_mb=2048,
        offline=True,
        sample_interval=0.1,
        command_prefix=[sys.executable, "-m", "code_indexing_mcp.cli"],
    )


@pytest.mark.memory
def test_real_model_index_stays_within_its_ceiling(tmp_path: Path) -> None:
    """The real gate. Opt in with -m memory and a populated model cache."""
    run = _real_model_run(tmp_path, ["near_cap", "blank_run"])

    verdict = evaluate_result(run)

    assert verdict.passed, f"{verdict.reasons}\n{run.stderr_tail}"


@pytest.mark.memory
def test_a_minified_file_stays_within_its_ceiling(tmp_path: Path) -> None:
    """The shape that character-bounded chunking could not survive.

    A single-line file near the 1 MiB cap used to breach every ceiling measured
    (2048/3072/4096 MiB at batch sizes 1 and 4) and abort the whole run with
    INDEX_RESOURCE_LIMIT, because 4,096 characters of minified source is ~2,157
    tokens and attention is quadratic in sequence length. Token-bounded windows
    hold it at 321/1,879/2,073 MiB parent/worker/combined, indexing cleanly.
    """
    run = _real_model_run(tmp_path, ["minified"])

    verdict = evaluate_result(run)

    assert verdict.passed, f"{verdict.reasons}\n{run.stderr_tail}"
    assert run.report is not None
    assert run.report["errors"] == []
    # Windowing is what makes this shape survivable; a silent fallback to
    # whole-candidate embedding would pass on memory but for the wrong reason.
    assert run.report["token_windowing"] is True
    assert run.report["embedded_tokens"] > 0
    assert run.report["embedded_segments"] > 0
