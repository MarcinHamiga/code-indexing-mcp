"""Parent-memory accounting shared by extraction and disposable workers."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import psutil

from .errors import CodeIndexingError, ErrorCode


@dataclass(frozen=True)
class ParentMemoryBudget:
    baseline_bytes: int
    ceiling_bytes: int


_current_budget: ContextVar[ParentMemoryBudget | None] = ContextVar(
    "index_memory_budget", default=None
)


def parent_memory_baseline() -> int:
    """Reuse the run's baseline even when a worker starts after extraction."""
    budget = _current_budget.get()
    return budget.baseline_bytes if budget is not None else psutil.Process().memory_info().rss


def check_parent_memory_budget() -> None:
    budget = _current_budget.get()
    if budget is None:
        return
    growth = max(0, psutil.Process().memory_info().rss - budget.baseline_bytes)
    if growth > budget.ceiling_bytes:
        raise CodeIndexingError(
            ErrorCode.INDEX_RESOURCE_LIMIT,
            "Indexing exceeded its parent memory ceiling during extraction",
            effective_memory_bytes=budget.ceiling_bytes,
            indexing_memory_bytes=growth,
            parent_baseline_bytes=budget.baseline_bytes,
        )


@contextmanager
def indexing_memory_budget(ceiling_bytes: int) -> Iterator[None]:
    budget = ParentMemoryBudget(psutil.Process().memory_info().rss, ceiling_bytes)
    token = _current_budget.set(budget)
    try:
        yield
    finally:
        _current_budget.reset(token)
