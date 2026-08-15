"""Unit tests for the shared top-k retrieval-quality metrics."""

from __future__ import annotations

import pytest

from code_indexing_mcp.acceptance import top_k_rank_correlation


def test_identical_rankings_correlate_perfectly() -> None:
    assert top_k_rank_correlation([["a", "b", "c"]], [["a", "b", "c"]]) == 1.0


def test_reversed_rankings_correlate_negatively() -> None:
    assert top_k_rank_correlation([["a", "b", "c"]], [["c", "b", "a"]]) == -1.0


def test_one_adjacent_swap_costs_the_discordant_share() -> None:
    # Pairs over (a, b, c): (a, b) discordant; (a, c) and (b, c) concordant.
    # tau = (2 - 1) / 3.
    assert top_k_rank_correlation([["a", "b", "c"]], [["b", "a", "c"]]) == pytest.approx(1 / 3)


def test_ids_missing_from_one_window_tie_past_its_end() -> None:
    # Union (a, b, c) with c absent from the reference window and a absent
    # from the candidate window: (a, b) and (a, c) discordant, (b, c)
    # concordant. tau = (1 - 2) / 3.
    assert top_k_rank_correlation([["a", "b"]], [["b", "c"]]) == pytest.approx(-1 / 3)


def test_correlation_is_averaged_over_queries() -> None:
    reference = [["a", "b", "c"], ["x", "y", "z"]]
    candidate = [["a", "b", "c"], ["z", "y", "x"]]

    assert top_k_rank_correlation(reference, candidate) == pytest.approx(0.0)


def test_single_shared_id_windows_are_perfectly_correlated() -> None:
    # One id cannot form a pair, so there is nothing to disagree about.
    assert top_k_rank_correlation([["only"]], [["only"]]) == 1.0


def test_invalid_ranking_inputs_raise() -> None:
    with pytest.raises(ValueError):
        top_k_rank_correlation([], [])
    with pytest.raises(ValueError):
        top_k_rank_correlation([["a"]], [["a"], ["b"]])
    with pytest.raises(ValueError):
        top_k_rank_correlation([[]], [["a"]])
    with pytest.raises(ValueError):
        top_k_rank_correlation([["a"]], [[]])
