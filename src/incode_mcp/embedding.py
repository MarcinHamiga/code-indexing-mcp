"""Local FastEmbed adapter and embedding protocol."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from fastembed import TextEmbedding

from .errors import ErrorCode, IncodeError

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_DIMENSION = 768


class PassageEmbedder(Protocol):
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class Embedder(PassageEmbedder, Protocol):
    model_id: str
    dimension: int

    def embed_query(self, text: str) -> list[float]: ...


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
