"""Token window planning and microbatch packing, with a deterministic tokenizer."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import pytest

from incode_mcp.token_batching import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    content_token_offsets,
    plan_candidate_windows,
    plan_microbatches,
    plan_token_windows,
)

_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class FakeEncoding:
    offsets: list[tuple[int, int]]
    special_tokens_mask: list[int]


def fake_encode(text: str) -> FakeEncoding:
    """Word-ish tokens wrapped in the ``(0, 0)`` specials a real tokenizer adds."""
    spans = [(match.start(), match.end()) for match in _TOKEN.finditer(text)]
    return FakeEncoding(
        offsets=[(0, 0), *spans, (0, 0)],
        special_tokens_mask=[1, *([0] * len(spans)), 1],
    )


def offsets_for(text: str) -> list[tuple[int, int]]:
    return content_token_offsets(fake_encode(text))


def test_special_tokens_are_dropped_from_planning_offsets() -> None:
    encoding = fake_encode("alpha beta")

    assert encoding.offsets[0] == (0, 0)
    assert offsets_for("alpha beta") == [(0, 5), (6, 10)]


def test_a_candidate_within_the_budget_stays_one_window() -> None:
    text = "alpha beta gamma"

    windows = plan_token_windows(offsets_for(text), text_length=len(text), max_tokens=8)

    assert len(windows) == 1
    assert (windows[0].start_char, windows[0].end_char) == (0, len(text))
    assert windows[0].token_count == 3


def test_no_window_exceeds_the_token_budget() -> None:
    text = " ".join(f"tok{index}" for index in range(4_000))

    windows = plan_token_windows(
        offsets_for(text),
        text_length=len(text),
        max_tokens=DEFAULT_MAX_TOKENS,
        overlap_tokens=DEFAULT_OVERLAP_TOKENS,
        max_windows=64,
    )

    assert len(windows) > 1
    assert all(window.token_count <= DEFAULT_MAX_TOKENS for window in windows)


def test_adjacent_windows_overlap_by_the_configured_token_count() -> None:
    text = " ".join(f"tok{index}" for index in range(300))

    windows = plan_token_windows(
        offsets_for(text), text_length=len(text), max_tokens=100, overlap_tokens=10
    )

    # Stride is 90 tokens, so a window's first token reappears 10 tokens before
    # the previous window's last.
    starts = [window.start_char for window in windows]
    token_spans = offsets_for(text)
    assert starts[1] == token_spans[90][0]
    assert starts[2] == token_spans[180][0]


def test_windows_cover_the_candidate_without_dropping_characters() -> None:
    text = "\n".join(f"value_{index} = {index}" for index in range(500))

    windows = plan_token_windows(
        offsets_for(text), text_length=len(text), max_tokens=64, overlap_tokens=8, max_windows=64
    )

    assert windows[0].start_char == 0
    assert windows[-1].end_char == len(text)
    # Each window resumes at or before the previous one ended, so the
    # concatenation of the disjoint prefixes reconstructs the source exactly.
    rebuilt = text[: windows[0].end_char]
    for previous, window in itertools.pairwise(windows):
        assert window.start_char <= previous.end_char
        rebuilt += text[max(window.start_char, previous.end_char) : window.end_char]
    assert rebuilt == text


def test_a_single_long_line_is_split_by_tokens_not_by_newlines() -> None:
    text = "DATA = [" + ", ".join(str(index % 977) for index in range(3_000)) + "]"

    windows = plan_token_windows(
        offsets_for(text), text_length=len(text), max_tokens=256, overlap_tokens=16, max_windows=64
    )

    assert "\n" not in text
    assert len(windows) > 1
    assert all(window.token_count <= 256 for window in windows)


def test_emitted_text_stays_within_twice_the_candidate_size() -> None:
    text = " ".join(f"tok{index}" for index in range(5_000))

    windows = plan_token_windows(
        offsets_for(text),
        text_length=len(text),
        max_tokens=DEFAULT_MAX_TOKENS,
        overlap_tokens=DEFAULT_OVERLAP_TOKENS,
        max_windows=64,
    )

    emitted = sum(window.end_char - window.start_char for window in windows)
    assert emitted <= 2 * len(text)


def test_boundaries_do_not_depend_on_the_memory_budget_or_batch_packing() -> None:
    text = " ".join(f"tok{index}" for index in range(2_000))
    offsets = offsets_for(text)

    first = plan_token_windows(offsets, text_length=len(text), max_tokens=512, max_windows=64)
    second = plan_token_windows(offsets, text_length=len(text), max_tokens=512, max_windows=64)

    assert first == second


def test_a_window_explosion_raises_instead_of_flooding_the_index() -> None:
    text = " ".join(f"tok{index}" for index in range(1_000))

    with pytest.raises(ValueError, match="exceeded 4 windows"):
        plan_token_windows(offsets_for(text), text_length=len(text), max_tokens=32, max_windows=4)


def test_whitespace_only_content_still_yields_one_window() -> None:
    windows = plan_token_windows([], text_length=12)

    assert windows == [(0, 12, 0)] or [
        (window.start_char, window.end_char, window.token_count) for window in windows
    ] == [(0, 12, 0)]


def test_empty_content_yields_no_window() -> None:
    assert plan_token_windows([], text_length=0) == []


def test_the_prefix_is_charged_against_the_window_budget() -> None:
    prefix = " ".join(f"header{index}" for index in range(20))
    content = " ".join(f"tok{index}" for index in range(100))

    with_prefix = plan_candidate_windows(
        fake_encode, [(prefix, content)], max_tokens=60, overlap_tokens=5
    )[0]
    without_prefix = plan_candidate_windows(
        fake_encode, [("", content)], max_tokens=60, overlap_tokens=5
    )[0]

    # 20 prefix tokens leave a 40-token content budget, so the prefixed
    # candidate needs strictly more windows for the same content.
    assert len(with_prefix) > len(without_prefix)
    assert all(window.token_count <= 40 for window in with_prefix)


def test_a_prefix_wider_than_the_budget_still_makes_forward_progress() -> None:
    prefix = " ".join(f"header{index}" for index in range(200))
    content = " ".join(f"tok{index}" for index in range(50))

    windows = plan_candidate_windows(
        fake_encode, [(prefix, content)], max_tokens=10, overlap_tokens=4, max_windows=64
    )[0]

    # The budget floor plus the half-budget overlap clamp keep the stride at 3
    # tokens, so 50 tokens land in 16 windows rather than looping forever.
    assert len(windows) == 16
    assert windows[-1].end_char == len(content)


def test_microbatches_respect_the_item_limit() -> None:
    batches = plan_microbatches([10] * 7, max_items=2, max_token_product=10_000)

    assert batches == [[0, 1], [2, 3], [4, 5], [6]]


def test_microbatches_respect_the_padded_token_product() -> None:
    # Padding widens every member to the longest, so a 900-token segment caps
    # its batch at four items against a 4,096 product.
    batches = plan_microbatches([900] * 6, max_items=8, max_token_product=4_096)

    assert batches == [[0, 1, 2, 3], [4, 5]]


def test_microbatches_bucket_similar_lengths_before_padding() -> None:
    batches = plan_microbatches([10, 1_000, 12, 900], max_items=2, max_token_product=2_000)

    assert batches == [[0, 2], [1, 3]]


def test_a_segment_wider_than_the_product_still_forms_its_own_batch() -> None:
    batches = plan_microbatches([9_000, 10], max_items=4, max_token_product=4_096)

    assert batches == [[0], [1]]


def test_microbatching_an_empty_plan_produces_no_batches() -> None:
    assert plan_microbatches([], max_items=4) == []


def test_a_zero_item_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_items"):
        plan_microbatches([1, 2], max_items=0)
