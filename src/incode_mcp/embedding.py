"""Local FastEmbed adapter and embedding protocol."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from fastembed import TextEmbedding

from .errors import ErrorCode, IncodeError
from .token_batching import (
    DEFAULT_MAX_TOKEN_PRODUCT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    MAX_WINDOWS_PER_CANDIDATE,
    TokenWindow,
    plan_candidate_windows,
    plan_microbatches,
)

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_DIMENSION = 768

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassageCandidate:
    """A chunk offered for embedding, with its repeated context header split out.

    The prefix is re-attached to every window of the candidate, so it is charged
    against the token budget once rather than being windowed away.
    """

    prefix: str
    content: str


@dataclass(frozen=True)
class EmbeddedSegment:
    """One token-bounded window of a candidate and its vector.

    ``vector`` is contiguous little-endian float32 bytes -- the same wire
    format the embedding worker returns -- so the indexing write path never
    materializes a list of Python floats per chunk.
    """

    start_char: int
    end_char: int
    token_count: int
    vector: bytes


def pack_vector(vector: Sequence[float] | np.ndarray[Any, Any]) -> bytes:
    """Pack a float vector into the little-endian float32 wire format.

    A model's row is whatever its backend returns. FastEmbed hands back a
    numpy array, which packs without copying through Python floats, but the
    embedder contract this module has always accepted -- see ``_vectors`` --
    is only that a row exposes ``tolist()``. Honour both, so a backend or a
    test double that is not numpy-backed still indexes.
    """
    if not isinstance(vector, np.ndarray) and hasattr(vector, "tolist"):
        vector = vector.tolist()
    return np.asarray(vector, dtype="<f4").tobytes()


@dataclass(frozen=True)
class SegmentPlan:
    """Token budgets applied when a candidate is windowed and batched."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    max_items: int = 1
    max_token_product: int = DEFAULT_MAX_TOKEN_PRODUCT
    max_windows: int = MAX_WINDOWS_PER_CANDIDATE


def compose_passage(prefix: str, content: str) -> str:
    """Join a context header to a window exactly as the extractor does."""
    return f"{prefix}\n{content}" if prefix else content


def resolve_tokenizer(model: object) -> Any | None:
    """Return the FastEmbed model's tokenizer, or ``None`` if the layout moved.

    FastEmbed does not expose the tokenizer as public API. Losing it degrades
    windowing rather than indexing, so callers fall back to whole-candidate
    embedding and log instead of failing the run.
    """
    for path in (("model", "tokenizer"), ("tokenizer",)):
        probe: Any = model
        for attribute in path:
            probe = getattr(probe, attribute, None)
            if probe is None:
                break
        if probe is not None and hasattr(probe, "encode"):
            return probe
    return None


def resolve_session_providers(model: object) -> tuple[str, ...]:
    """Return the execution providers the loaded ONNX session actually uses.

    Requesting a provider is not the same as getting one: ONNX Runtime silently
    drops a provider it cannot initialise and runs on the next one. The probe
    reports what the session settled on so a "CUDA" run that quietly became a
    CPU run is visible rather than merely slow.

    Like ``resolve_tokenizer`` this walks a private layout, so an empty tuple
    means "unknown", never "none".
    """
    for path in (("model", "model"), ("model",)):
        probe: Any = model
        for attribute in path:
            probe = getattr(probe, attribute, None)
            if probe is None:
                break
        getter = getattr(probe, "get_providers", None)
        if callable(getter):
            try:
                return tuple(str(name) for name in getter())
            except Exception:  # pragma: no cover - defensive against runtime quirks
                return ()
    return ()


# Two texts, because a single-row batch can hide a provider that only fails
# once a graph is given a real batch dimension.
PROBE_TEXTS: tuple[str, ...] = ("def probe() -> int:\n    return 0\n", "class Probe:\n    pass\n")


def validate_probe_vectors(vectors: Sequence[bytes], *, dimension: int, count: int) -> None:
    """Reject a probe whose vectors could not be used to search an index.

    A backend that returns the wrong width, a NaN, or an unnormalised row would
    corrupt retrieval rather than fail loudly, so this runs before any real
    passage is sent to it.

    Raises ``ValueError``; the caller decides whether that means fall back to
    CPU or, under strict mode, fail the run.
    """
    if len(vectors) != count:
        raise ValueError(f"probe returned {len(vectors)} vectors for {count} inputs")
    expected_bytes = dimension * 4
    for index, packed in enumerate(vectors):
        if len(packed) != expected_bytes:
            raise ValueError(
                f"probe vector {index} is {len(packed) // 4} wide, expected {dimension}"
            )
        row = np.frombuffer(packed, dtype="<f4", count=dimension)
        if not np.all(np.isfinite(row)):
            raise ValueError(f"probe vector {index} contains non-finite values")
        norm = float(np.linalg.norm(row))
        # The model normalises its output, and every stored vector is compared
        # by cosine similarity, so a row far off the unit sphere means pooling
        # or normalisation did not run on this provider.
        if not 0.9 <= norm <= 1.1:
            raise ValueError(f"probe vector {index} has norm {norm:.4f}, expected ~1.0")


def plan_passages(
    encode: Callable[[str], Any] | None,
    candidates: Sequence[PassageCandidate],
    plan: SegmentPlan,
) -> list[list[TokenWindow]]:
    """Window each candidate by token count.

    With *encode* set to ``None`` every candidate stays whole, which is the
    pre-windowing behaviour and the fallback when no tokenizer is reachable.

    Raises ``ValueError`` when a candidate cannot be planned within its window
    cap. That is a property of the file, not of the environment, so callers
    surface it against the file rather than aborting the run.
    """
    if encode is None:
        return [[TokenWindow(0, len(candidate.content), 0)] for candidate in candidates]
    return plan_candidate_windows(
        encode,
        [(candidate.prefix, candidate.content) for candidate in candidates],
        max_tokens=plan.max_tokens,
        overlap_tokens=plan.overlap_tokens,
        max_windows=plan.max_windows,
    )


def embed_windows[Vector](
    embed: Callable[[list[str]], list[Vector]],
    candidates: Sequence[PassageCandidate],
    windows_per_candidate: Sequence[Sequence[TokenWindow]],
    plan: SegmentPlan,
) -> list[list[tuple[TokenWindow, Vector]]]:
    """Embed planned windows in microbatches packed to the token budget."""
    if not candidates:
        return []
    owners: list[int] = []
    windows: list[TokenWindow] = []
    texts: list[str] = []
    for index, (candidate, planned) in enumerate(
        zip(candidates, windows_per_candidate, strict=True)
    ):
        for window in planned:
            owners.append(index)
            windows.append(window)
            texts.append(
                compose_passage(
                    candidate.prefix, candidate.content[window.start_char : window.end_char]
                )
            )

    results: list[list[tuple[TokenWindow, Vector]]] = [[] for _ in candidates]
    for batch in plan_microbatches(
        [window.token_count for window in windows],
        max_items=plan.max_items,
        max_token_product=plan.max_token_product,
    ):
        vectors = embed([texts[position] for position in batch])
        for position, vector in zip(batch, vectors, strict=True):
            results[owners[position]].append((windows[position], vector))
    for candidate_results in results:
        candidate_results.sort(key=lambda item: (item[0].start_char, item[0].end_char))
    return results


def embed_planned_segments[Vector](
    encode: Callable[[str], Any] | None,
    embed: Callable[[list[str]], list[Vector]],
    candidates: Sequence[PassageCandidate],
    plan: SegmentPlan,
) -> list[list[tuple[TokenWindow, Vector]]]:
    """Plan token windows for *candidates* and embed them."""
    if not candidates:
        return []
    return embed_windows(embed, candidates, plan_passages(encode, candidates, plan), plan)


class ModelIdentity(Protocol):
    """The identity every role of one embedding model shares.

    Query and passage embedding may run in different processes on different
    devices, but they must remain the same model producing the same vector
    space -- otherwise a query could never retrieve the passages it indexed.
    """

    model_id: str
    dimension: int


class PassageEmbedder(Protocol):
    """The indexing-side role: many passages, throughput-bound, offloadable."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class QueryEmbedder(ModelIdentity, Protocol):
    """The search-side role: one short query, latency-bound, stays on CPU.

    Deliberately separate from ``PassageEmbedder``: a query must not wait on an
    accelerator worker spawning or a model loading onto a device.
    """

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class SegmentingEmbedder(Protocol):
    """A passage embedder that bounds each sequence by tokens before embedding."""

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]: ...


class Embedder(QueryEmbedder, PassageEmbedder, Protocol):
    """Both roles served by one in-process model, which is the CPU default."""


class FastEmbedder:
    model_id = DEFAULT_MODEL
    dimension = DEFAULT_DIMENSION

    def __init__(
        self,
        cache_directory: Path,
        *,
        offline: bool = False,
        threads: int | None = None,
        enable_cpu_mem_arena: bool = False,
    ) -> None:
        self.cache_directory = cache_directory
        self.offline = offline
        self.threads = threads
        self.enable_cpu_mem_arena = enable_cpu_mem_arena
        self._model: TextEmbedding | None = None
        # The daemon serves each client connection on its own thread, so the
        # lazy load must not be able to build two ONNX sessions concurrently.
        self._model_lock = threading.Lock()

    def prepare(self) -> None:
        self._get_model()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._vectors(self._get_model().passage_embed(texts))

    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        model = self._get_model()
        tokenizer = resolve_tokenizer(model)
        if tokenizer is None:
            logger.warning(
                "No tokenizer reachable on %s; embedding candidates whole, which "
                "leaves sequence length unbounded on token-dense files",
                self.model_id,
            )
        planned = embed_planned_segments(
            None if tokenizer is None else tokenizer.encode,
            lambda texts: [pack_vector(vector) for vector in model.passage_embed(texts)],
            candidates,
            plan,
        )
        return [
            [
                EmbeddedSegment(
                    start_char=window.start_char,
                    end_char=window.end_char,
                    token_count=window.token_count,
                    vector=vector,
                )
                for window, vector in segments
            ]
            for segments in planned
        ]

    def embed_query(self, text: str) -> list[float]:
        vectors = self._vectors(self._get_model().query_embed(text))
        return vectors[0]

    def _get_model(self) -> TextEmbedding:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            try:
                self._model = TextEmbedding(
                    model_name=self.model_id,
                    cache_dir=str(self.cache_directory),
                    local_files_only=self.offline,
                    threads=self.threads,
                    enable_cpu_mem_arena=self.enable_cpu_mem_arena,
                )
            except Exception as exc:
                raise IncodeError(
                    ErrorCode.MODEL_UNAVAILABLE,
                    f"Embedding model is unavailable: {self.model_id}",
                    model=self.model_id,
                    offline=self.offline,
                ) from exc
            return self._model

    @staticmethod
    def _vectors(values: Sequence[object] | object) -> list[list[float]]:
        return [value.tolist() for value in values]  # type: ignore[union-attr]
