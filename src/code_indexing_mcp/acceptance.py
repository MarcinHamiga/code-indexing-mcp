"""Pure correctness metrics shared by accelerator promotion gates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float32]


def _matrix(values: NDArray[Any], *, name: str) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _normalized(values: NDArray[Any], *, name: str) -> FloatMatrix:
    matrix = _matrix(values, name=name)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.float32(1e-12)):
        raise ValueError(f"{name} contains a zero-length row")
    return np.asarray(matrix / norms, dtype=np.float32)


def cosine_rows(reference: NDArray[Any], candidate: NDArray[Any]) -> FloatMatrix:
    """Return cosine similarity for each corresponding pair of matrix rows."""

    left = _normalized(reference, name="reference")
    right = _normalized(candidate, name="candidate")
    if left.shape != right.shape:
        raise ValueError(f"reference and candidate shapes differ: {left.shape} != {right.shape}")
    return np.asarray(np.sum(left * right, axis=1), dtype=np.float32)


def top_k_overlap(
    queries: NDArray[Any],
    reference: NDArray[Any],
    candidate: NDArray[Any],
    *,
    k: int,
) -> float:
    """Return mean top-k result overlap for two candidate vector matrices."""

    query_rows = _normalized(queries, name="queries")
    reference_rows = _normalized(reference, name="reference")
    candidate_rows = _normalized(candidate, name="candidate")
    if reference_rows.shape != candidate_rows.shape:
        raise ValueError(
            f"reference and candidate shapes differ: "
            f"{reference_rows.shape} != {candidate_rows.shape}"
        )
    if query_rows.shape[1] != reference_rows.shape[1]:
        raise ValueError(
            f"query dimension {query_rows.shape[1]} does not match "
            f"candidate dimension {reference_rows.shape[1]}"
        )
    if not 1 <= k <= reference_rows.shape[0]:
        raise ValueError(f"k must be from 1 to {reference_rows.shape[0]}")

    reference_order = np.argsort(
        -(query_rows @ reference_rows.T),
        axis=1,
        kind="stable",
    )[:, :k]
    candidate_order = np.argsort(
        -(query_rows @ candidate_rows.T),
        axis=1,
        kind="stable",
    )[:, :k]
    overlap = sum(
        len(set(left.tolist()).intersection(right.tolist()))
        for left, right in zip(reference_order, candidate_order, strict=True)
    )
    return float(overlap / (query_rows.shape[0] * k))


def top_k_rank_correlation(
    reference_orders: Sequence[Sequence[Any]],
    candidate_orders: Sequence[Sequence[Any]],
) -> float:
    """Return the mean Kendall tau-b between paired top-k id rankings.

    Both rankings are top-k windows over one result set of unique ids, so an
    id present in only one window counts as tied just past that window's end
    (position ``k``) in the other: losing a result outright is a tie among the
    lost rather than an invented rank. 1.0 is an identical ranking and -1.0 a
    fully reversed one over the same ids.
    """

    if len(reference_orders) != len(candidate_orders):
        raise ValueError(
            f"ranking counts differ: {len(reference_orders)} != {len(candidate_orders)}"
        )
    if not reference_orders:
        raise ValueError("at least one pair of rankings is required")
    scores: list[float] = []
    for reference_order, candidate_order in zip(reference_orders, candidate_orders, strict=True):
        if not reference_order or not candidate_order:
            raise ValueError("a top-k ranking must not be empty")
        window = max(len(reference_order), len(candidate_order))
        union = list(dict.fromkeys([*reference_order, *candidate_order]))
        reference_position = {item: index for index, item in enumerate(reference_order)}
        candidate_position = {item: index for index, item in enumerate(candidate_order)}
        reference_ranks = np.asarray(
            [reference_position.get(item, window) for item in union], dtype=np.int64
        )
        candidate_ranks = np.asarray(
            [candidate_position.get(item, window) for item in union], dtype=np.int64
        )
        left, right = np.triu_indices(len(union), k=1)
        reference_sign = np.sign(reference_ranks[left] - reference_ranks[right])
        candidate_sign = np.sign(candidate_ranks[left] - candidate_ranks[right])
        concordant = int(np.sum((reference_sign != 0) & (reference_sign == candidate_sign)))
        discordant = int(np.sum((reference_sign != 0) & (reference_sign == -candidate_sign)))
        tied_reference = int(np.sum((reference_sign == 0) & (candidate_sign != 0)))
        tied_candidate = int(np.sum((candidate_sign == 0) & (reference_sign != 0)))
        denominator = math.sqrt(
            float(
                (concordant + discordant + tied_reference)
                * (concordant + discordant + tied_candidate)
            )
        )
        # Both windows hold exactly the same single id: identical rankings.
        scores.append(1.0 if denominator == 0.0 else (concordant - discordant) / denominator)
    return float(sum(scores) / len(scores))
