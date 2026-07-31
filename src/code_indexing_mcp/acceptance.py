"""Pure correctness metrics shared by accelerator promotion gates."""

from __future__ import annotations

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
