"""Deterministic end-to-end indexing benchmarks with JSON-ready results."""

from __future__ import annotations

import hashlib
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS, HnswSq
from lancedb.query import (
    FullTextOperator,
    LanceHybridQueryBuilder,
    LanceVectorQueryBuilder,
    MultiMatchQuery,
)
from numpy.typing import NDArray

from . import update_check
from .acceptance import top_k_rank_correlation
from .application import Application, RuntimePaths
from .embedding import PassageEmbedder, QueryEmbedder
from .errors import CodeIndexingError, ErrorCode
from .models import (
    IndexReport,
    MaintenanceReport,
    ProjectInfo,
    SearchResponse,
    StorageStatus,
)
from .settings import IndexSettings

# The repeated_edits scenario applies this many consecutive edits to one file,
# indexing after each one, so per-edit write amplification shows up as version
# growth over a meaningful sample.
REPEATED_EDITS = 100

# Multi-project search scope sizes the search benchmark measures: one, eight,
# and fifty partitions. With fewer projects available the benchmark measures
# every achievable scope instead.
SEARCH_SCOPES = (1, 8, 50)

# How many times each search scope is timed. Search latency is noisy per run,
# so every scope is a small sample, never a single point estimate.
SEARCH_ITERATIONS = 3

# The vector-precision experiment judges every variant against one
# float32-exact numpy reference, so K, the sample size, and the adoption-gate
# floors live here and are recorded in the report rather than left to whoever
# reads it to reconstruct.
PRECISION_TOP_K = 8
PRECISION_ITERATIONS = 5
DEFAULT_RECALL_FLOOR = 0.99
DEFAULT_RANK_FLOOR = 0.95

# Fixed topic vocabulary for the retrieval corpus. Relevance judgments are by
# construction -- a query is relevant to exactly its topic's passages -- so
# the corpus needs no hand-maintained labels, and the corpus digest in the
# report pins the exact text every number was measured on.
RETRIEVAL_TOPICS: tuple[tuple[str, ...], ...] = (
    ("authorize", "permission", "credential", "session"),
    ("invoice", "billing", "tax", "subtotal"),
    ("hybrid", "ranking", "lexical", "semantic"),
    ("partition", "journal", "rollback", "fragment"),
    ("grammar", "syntax", "node", "token"),
    ("socket", "retry", "timeout", "listener"),
    ("mutex", "atomic", "deadlock", "thread"),
    ("metric", "trace", "span", "histogram"),
)


class IndexBenchmarkApplication(Protocol):
    def init_project(self, path: Path) -> ProjectInfo: ...

    def index_project(self, project: str, *, force: bool = False) -> IndexReport: ...

    def storage_status(self, project: str | None = None) -> StorageStatus: ...

    def maintain_storage(
        self, project: str | None = None, *, wait_for_lock: bool = False
    ) -> MaintenanceReport: ...


class SearchBenchmarkApplication(Protocol):
    def init_project(self, path: Path) -> ProjectInfo: ...

    def index_project(self, project: str, *, force: bool = False) -> IndexReport: ...

    def search_code(self, query: str, *, projects: list[str], limit: int = 8) -> SearchResponse: ...


def _benchmark_application(
    paths: RuntimePaths,
    workspace: Path,
    *,
    batch_size: int | None = None,
    cwd: Path | None = None,
) -> Application:
    """Build every benchmark application from one isolated settings snapshot."""
    base_settings = IndexSettings.from_environment()
    settings = replace(
        base_settings,
        embedding_batch_size=batch_size
        if batch_size is not None
        else base_settings.embedding_batch_size,
        index_execution="in-process",
        broker_mode="off",
    )
    return Application(
        RuntimePaths(data=workspace / "data", cache=paths.cache),
        cwd=cwd or workspace,
        settings=settings,
    )


def _benchmark_metadata(app: Application) -> dict[str, str | None]:
    return {
        "model_id": app.embedder.model_id,
        "embedding_backend": app.effective_backend_selection.descriptor.accelerator.value,
        "revision": update_check.checkout_head(Path(__file__).resolve().parents[2]),
    }


def write_benchmark_corpus(root: Path, *, files: int = 128, functions_per_file: int = 2) -> int:
    """Write a fixed Python corpus and return its source byte count."""
    if files < 1 or functions_per_file < 1:
        raise ValueError("benchmark corpus dimensions must be positive")
    root.mkdir(parents=True, exist_ok=False)
    total = 0
    for file_index in range(files):
        source = "".join(
            (
                f"def function_{file_index:04d}_{function_index:04d}(value: int) -> int:\n"
                f"    return value + {file_index + function_index}\n\n"
            )
            for function_index in range(functions_per_file)
        )
        encoded = source.encode()
        (root / f"module_{file_index:04d}.py").write_bytes(encoded)
        total += len(encoded)
    return total


@dataclass(frozen=True)
class RetrievalPassage:
    """One corpus chunk: the persisted text columns the experiment searches."""

    chunk_id: str
    content: str
    identifier_terms: str


@dataclass(frozen=True)
class RetrievalQuery:
    """One corpus query with its by-construction relevance judgment."""

    text: str
    relevant: tuple[str, ...]


class PrecisionEmbedder(PassageEmbedder, QueryEmbedder, Protocol):
    """Both embedding roles the precision experiment needs."""


def build_retrieval_corpus(
    *, passages: int = 240
) -> tuple[list[RetrievalPassage], list[RetrievalQuery]]:
    """Build the deterministic judged corpus for the precision experiment.

    Passages cycle through the topic vocabulary, so each topic owns
    ``passages / len(RETRIEVAL_TOPICS)`` chunks whose text and identifier
    terms repeat only that topic's words. One query per topic is relevant to
    exactly that topic's passages.
    """

    if passages < len(RETRIEVAL_TOPICS):
        raise ValueError(f"the retrieval corpus needs at least {len(RETRIEVAL_TOPICS)} passages")
    corpus: list[RetrievalPassage] = []
    relevant_by_topic: list[list[str]] = [[] for _ in RETRIEVAL_TOPICS]
    for index in range(passages):
        topic = index % len(RETRIEVAL_TOPICS)
        terms = RETRIEVAL_TOPICS[topic]
        within = index // len(RETRIEVAL_TOPICS)
        name = f"{terms[0]}_{index:04d}"
        chunk_id = f"precision-{index:06d}"
        corpus.append(
            RetrievalPassage(
                chunk_id=chunk_id,
                content=(
                    f"def {name}(request, context):\n"
                    f"    # {terms[1]} policy for {terms[2]} handling\n"
                    f"    validate_{terms[3]}(request, context)\n"
                    f"    return audit_{terms[2]}(context)\n"
                ),
                identifier_terms=f"{name} {' '.join(terms)} {within:04d}",
            )
        )
        relevant_by_topic[topic].append(chunk_id)
    queries = [
        RetrievalQuery(
            text=f"where is {terms[0]} {terms[3]} validated",
            relevant=tuple(relevant_by_topic[topic]),
        )
        for topic, terms in enumerate(RETRIEVAL_TOPICS)
    ]
    return corpus, queries


def _storage_snapshot(app: IndexBenchmarkApplication, project_id: str) -> dict[str, Any]:
    """Return *project_id*'s storage statistics as a JSON-ready dict."""
    status = app.storage_status(project_id)
    entry = next((stats for stats in status.projects if stats.project.id == project_id), None)
    return entry.model_dump(mode="json") if entry is not None else {}


def _measure(
    action: Callable[[], IndexReport],
    *,
    snapshot_after: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic_ns()
    report = action()
    wall_ms = (time.monotonic_ns() - started) / 1_000_000
    # Throughput is computed from the indexer's own duration and nothing else.
    # Falling back to wall time would publish one field name against two
    # different clocks, so runs that report no duration report no throughput --
    # a null a consumer can skip, rather than a number it would wrongly compare.
    reported_ms = int(report.duration_ms)
    result: dict[str, Any] = {
        "wall_ms": round(wall_ms, 3),
        "reported_duration_ms": reported_ms,
        # Pipeline throughput, not embedding throughput: `duration_ms` covers
        # scanning, parsing, embedding and committing, so an incremental
        # scenario that scans the whole corpus to embed one file's chunks is
        # dominated by scan overhead.
        "chunks_per_second": (
            round(report.embedded_chunks * 1_000 / reported_ms, 3) if reported_ms > 0 else None
        ),
        # This run's own staged structural rows (T1) -- not a whole-project
        # table read, which would report the same total for every scenario
        # regardless of how much work that scenario actually did.
        "structural_records": report.staged_reference_rows,
        # Reference extraction's own timing (T1), separate from
        # `parse_duration_ms`, which also covers parsing and chunking.
        "reference_extraction_duration_ms": report.reference_extraction_duration_ms or 0,
        "embedding_backend": report.embedding_backend,
        "embedding_batch_size": report.embedding_batch_size,
        "embedding_cache_status": report.embedding_cache_status,
        "embedding_cache_lookup_ms": report.embedding_cache_lookup_ms,
        "embedding_cache_write_ms": report.embedding_cache_write_ms,
        "embedding_artifact_digest": report.embedding_artifact_digest,
        "embedding_tokenizer_digest": report.embedding_tokenizer_digest,
        "report": report.model_dump(mode="json"),
    }
    if snapshot_after is not None:
        result["storage_after"] = snapshot_after()
    return result


def _duration_summary(samples: Sequence[float]) -> dict[str, Any]:
    """Summarize per-iteration durations, in milliseconds.

    A single total cannot distinguish a constant per-iteration cost from one
    that grows as versions accumulate, so the distribution is reported and the
    head and tail means are reported next to it: a last decile well above the
    first is write amplification, not noise.
    """
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    decile = max(1, len(samples) // 10)
    # Nearest-rank p95, which for small samples is the honest choice:
    # interpolation would invent a value between two observations.
    p95_index = min(len(ordered) - 1, -(-95 * len(ordered) // 100) - 1)
    return {
        "count": len(ordered),
        "total_ms": round(sum(ordered), 3),
        "min_ms": round(ordered[0], 3),
        "median_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_index], 3),
        "max_ms": round(ordered[-1], 3),
        "first_decile_mean_ms": round(statistics.fmean(samples[:decile]), 3),
        "last_decile_mean_ms": round(statistics.fmean(samples[-decile:]), 3),
    }


def run_index_benchmark(app: IndexBenchmarkApplication, root: Path) -> dict[str, Any]:
    """Run the storage-growth scenarios against one isolated application.

    Every scenario records the project's storage statistics after it finishes,
    so table-version deltas and physical growth are observable per scenario
    (contract version 2). ``post_maintenance`` times real cleanup and captures
    the resulting storage snapshot without changing the JSON contract.

    The storage figures are deterministic and single-run comparison is sound.
    The timings are not: apart from ``repeated_edits``, every scenario runs
    once, so ``wall_ms`` is a point estimate that carries no distribution and
    should be read as indicative rather than compared run-to-run on small
    differences. ``repeated_edits`` is the exception and reports ``per_edit_ms``
    order statistics over its 100 iterations.
    """
    project = app.init_project(root)
    scenarios: dict[str, dict[str, Any]] = {}

    def snapshot() -> dict[str, Any]:
        return _storage_snapshot(app, project.id)

    storage_baseline = snapshot()

    scenarios["cold_start"] = _measure(
        lambda: app.index_project(project.id, force=True), snapshot_after=snapshot
    )
    # The first index in a fresh application loads the embedding model, so this
    # scenario's timings are indexing plus one-time warmup. Flagged rather than
    # subtracted: a warmup regression is worth seeing, just not worth silently
    # charging to indexing.
    scenarios["cold_start"]["includes_embedder_warmup"] = True
    scenarios["no_op"] = _measure(
        lambda: app.index_project(project.id, force=False), snapshot_after=snapshot
    )

    edited = root / "module_0000.py"
    with edited.open("a", encoding="utf-8") as stream:
        stream.write("\ndef phase_2_single_edit_marker(value: int) -> int:\n    return value + 1\n")
    scenarios["single_file_edit"] = _measure(
        lambda: app.index_project(project.id, force=False), snapshot_after=snapshot
    )

    repeated_started = time.monotonic_ns()
    per_edit_ms: list[float] = []
    for edit_index in range(REPEATED_EDITS):
        with edited.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\ndef repeated_edit_marker_{edit_index:04d}(value: int) -> int:\n"
                f"    return value + {edit_index}\n"
            )
        edit_started = time.monotonic_ns()
        app.index_project(project.id, force=False)
        per_edit_ms.append((time.monotonic_ns() - edit_started) / 1_000_000)
    scenarios["repeated_edits"] = {
        "wall_ms": round((time.monotonic_ns() - repeated_started) / 1_000_000, 3),
        "edits": REPEATED_EDITS,
        # The one scenario with a real sample size, so it reports a
        # distribution rather than only its total.
        "per_edit_ms": _duration_summary(per_edit_ms),
        "storage_after": snapshot(),
    }

    scenarios["forced_reindex"] = _measure(
        lambda: app.index_project(project.id, force=True), snapshot_after=snapshot
    )

    removed_single = _unlink_if_present(root / "module_0001.py")
    scenarios["single_file_deletion"] = _measure(
        lambda: app.index_project(project.id, force=False), snapshot_after=snapshot
    )
    scenarios["single_file_deletion"]["removed_files"] = removed_single

    removed_group = 0
    for deleted_index in range(2, 10):
        removed_group += _unlink_if_present(root / f"module_{deleted_index:04d}.py")
    scenarios["many_file_deletions"] = _measure(
        lambda: app.index_project(project.id, force=False), snapshot_after=snapshot
    )
    scenarios["many_file_deletions"]["removed_files"] = removed_group

    maintenance_started = time.monotonic_ns()
    maintenance = app.maintain_storage(project.id, wait_for_lock=True)
    scenarios["post_maintenance"] = {
        "wall_ms": round((time.monotonic_ns() - maintenance_started) / 1_000_000, 3),
        "report": maintenance.model_dump(mode="json"),
        "storage_after": snapshot(),
    }
    return {
        "schema_version": 2,
        "storage_baseline": storage_baseline,
        "scenarios": scenarios,
    }


def _unlink_if_present(path: Path) -> int:
    """Delete *path*, returning 1 when it existed. Bounds a deletion group to the corpus."""
    try:
        path.unlink()
    except FileNotFoundError:
        return 0
    return 1


def run_search_benchmark(
    app: SearchBenchmarkApplication,
    roots: list[Path],
    *,
    iterations: int = SEARCH_ITERATIONS,
    query: str = "function returns value",
) -> dict[str, Any]:
    """Measure hybrid-search latency for one, eight, and fifty project scopes.

    Every achievable scope runs the same query repeatedly and then twice more
    to pin the global hit ordering, so ranking determinism is observable next
    to the latency: multi-project latency should approach the slowest
    partition plus merge overhead, not the sum of every partition, and the
    ordered hit list must be identical across runs.
    """
    if len(roots) < 1:
        raise ValueError("the search benchmark needs at least one project")
    if iterations < 1:
        raise ValueError("the search benchmark needs at least one iteration")
    project_ids: list[str] = []
    for root in roots:
        write_benchmark_corpus(root, files=2, functions_per_file=2)
        project = app.init_project(root)
        app.index_project(project.id, force=True)
        project_ids.append(project.id)
    scopes = sorted(
        {scope for scope in SEARCH_SCOPES if scope <= len(project_ids)} | {len(project_ids)}
    )
    scenarios: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        selected = project_ids[:scope]
        samples: list[float] = []
        for _ in range(iterations):
            started = time.monotonic_ns()
            app.search_code(query, projects=selected, limit=8)
            samples.append((time.monotonic_ns() - started) / 1_000_000)
        first = app.search_code(query, projects=selected, limit=8)
        second = app.search_code(query, projects=selected, limit=8)
        scenarios[str(scope)] = {
            "projects": len(selected),
            "latency_ms": _duration_summary(samples),
            "deterministic": [hit.chunk_id for hit in first.hits]
            == [hit.chunk_id for hit in second.hits],
            "top_hits": [
                {
                    "project_id": hit.project_id,
                    "path": hit.path,
                    "start_line": hit.start_line,
                }
                for hit in first.hits
            ],
        }
    return {"schema_version": 1, "projects": len(project_ids), "query": query, "scopes": scenarios}


def _directory_physical_bytes(path: Path) -> int:
    """Sum file sizes under *path* without following symlinks.

    Mirrors the storage-stats convention: a symlink costs its own small
    metadata, never the size of whatever it points at.
    """

    total = 0
    for directory, subdirectories, filenames in os.walk(path, followlinks=False):
        subdirectories[:] = [
            name for name in subdirectories if not (Path(directory) / name).is_symlink()
        ]
        for name in filenames:
            candidate = Path(directory) / name
            if candidate.is_symlink():
                continue
            total += candidate.stat().st_size
    return total


def _precision_schema(dimension: int, dtype: pa.DataType) -> pa.Schema:
    """The production chunk columns the precision experiment needs."""

    return pa.schema(
        [
            ("chunk_id", pa.string()),
            ("content", pa.string()),
            ("identifier_terms", pa.string()),
            ("vector", pa.list_(dtype, dimension)),
        ]
    )


def _run_precision_variant(
    directory: Path,
    *,
    corpus: list[RetrievalPassage],
    query_texts: Sequence[str],
    passage_vectors: NDArray[np.float32],
    query_vectors: NDArray[np.float32],
    storage: str,
    exact: bool,
    top_k: int,
    iterations: int,
    ground_truth: list[list[str]],
) -> dict[str, Any]:
    """Measure one storage x index combination on a standalone table.

    The table mirrors the production chunk columns, FTS configuration, and
    HNSW config, so the numbers describe what shipping that combination
    would actually cost rather than a synthetic stand-in.
    """

    dtype = pa.float32() if storage == "float32" else pa.float16()
    stored = passage_vectors if storage == "float32" else passage_vectors.astype(np.float16)
    schema = _precision_schema(passage_vectors.shape[1], dtype)
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(stored.reshape(-1)), passage_vectors.shape[1]
    )
    data = pa.table(
        {
            "chunk_id": pa.array([passage.chunk_id for passage in corpus], pa.string()),
            "content": pa.array([passage.content for passage in corpus], pa.string()),
            "identifier_terms": pa.array(
                [passage.identifier_terms for passage in corpus], pa.string()
            ),
            "vector": vectors,
        },
        schema=schema,
    )

    started = time.monotonic_ns()
    database = lancedb.connect(directory)
    table = database.create_table("chunks", data=data)
    table_build_ms = (time.monotonic_ns() - started) / 1_000_000

    started = time.monotonic_ns()
    for column in ("content", "identifier_terms"):
        table.create_index(
            column,
            config=FTS(lower_case=True, stem=False, remove_stop_words=False),
            replace=False,
        )
    if not exact:
        table.create_index("vector", config=HnswSq(distance_type="cosine"), replace=False)
    index_build_ms = (time.monotonic_ns() - started) / 1_000_000

    result_orders: list[list[str]] = []
    for query_vector in query_vectors:
        search = cast(
            "LanceVectorQueryBuilder",
            table.search(query_vector.tolist(), vector_column_name="vector"),
        ).select(["chunk_id"])
        search = search.limit(top_k)
        if exact:
            search = search.bypass_vector_index()
        result_orders.append([str(row["chunk_id"]) for row in search.to_list()])
    recall_at_k = sum(
        len(set(reference).intersection(candidate))
        for reference, candidate in zip(ground_truth, result_orders, strict=True)
    ) / (len(ground_truth) * top_k)
    rank_correlation = top_k_rank_correlation(ground_truth, result_orders)

    samples: list[float] = []
    for _ in range(iterations):
        for text, query_vector in zip(query_texts, query_vectors, strict=True):
            started = time.monotonic_ns()
            hybrid = cast(
                "LanceHybridQueryBuilder",
                table.search(query_type="hybrid", vector_column_name="vector"),
            ).vector(query_vector.tolist())
            hybrid = (
                hybrid.text(
                    MultiMatchQuery(
                        text,
                        ["content", "identifier_terms"],
                        boosts=None,
                        operator=FullTextOperator.OR,
                    )
                )
                .limit(top_k)
                .select(["chunk_id"])
                .rerank()
            )
            if exact:
                hybrid = hybrid.bypass_vector_index()
            hybrid.to_list()
            samples.append((time.monotonic_ns() - started) / 1_000_000)

    physical_bytes = _directory_physical_bytes(directory)
    table.optimize()
    post_optimize_bytes = _directory_physical_bytes(directory)
    return {
        "storage": storage,
        "index": "exact" if exact else "hnsw_sq8",
        "table_build_ms": round(table_build_ms, 3),
        "index_build_ms": round(index_build_ms, 3),
        "recall_at_k": round(float(recall_at_k), 6),
        "rank_correlation": round(float(rank_correlation), 6),
        "hybrid_latency_ms": _duration_summary(samples),
        "physical_bytes": physical_bytes,
        "post_optimize_bytes": post_optimize_bytes,
    }


def run_precision_benchmark(
    embedder: PrecisionEmbedder,
    workspace: Path,
    *,
    passages: int = 240,
    top_k: int = PRECISION_TOP_K,
    iterations: int = PRECISION_ITERATIONS,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    rank_floor: float = DEFAULT_RANK_FLOOR,
) -> dict[str, Any]:
    """Compare vector-storage precisions against a float32-exact reference.

    Every variant is judged against top-k rankings computed in numpy float32
    cosine -- one reference for every variant -- and the float32 flat search
    is itself checked against that reference as a baseline sanity anchor.
    Retrieval quality is reported as overlap with the reference top-k and as
    Kendall tau-b over the paired rankings; cost is reported as build time,
    hybrid-search latency samples, and physical bytes before and after
    ``optimize()``. A combination some LanceDB version cannot serve reports
    an ``error`` instead of aborting the whole experiment.
    """

    corpus, queries = build_retrieval_corpus(passages=passages)
    passage_vectors = np.asarray(
        embedder.embed_passages([passage.content for passage in corpus]), dtype=np.float32
    )
    query_vectors = np.asarray(
        [embedder.embed_query(query.text) for query in queries], dtype=np.float32
    )

    # The single reference every variant is judged against: float32 cosine.
    normalized_passages = passage_vectors / np.linalg.norm(passage_vectors, axis=1, keepdims=True)
    normalized_queries = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    ground_truth: list[list[str]] = []
    for query_row in normalized_queries:
        order = np.argsort(-(query_row @ normalized_passages.T), kind="stable")[:top_k]
        ground_truth.append([corpus[index].chunk_id for index in order])

    digest = hashlib.sha256()
    for passage in corpus:
        digest.update(passage.content.encode())
        digest.update(passage.identifier_terms.encode())
    for query in queries:
        digest.update(query.text.encode())
    query_texts = [query.text for query in queries]

    variants: dict[str, dict[str, Any]] = {}
    for storage in ("float32", "float16"):
        for exact in (True, False):
            key = f"{storage}_{'exact' if exact else 'hnsw_sq8'}"
            try:
                variants[key] = _run_precision_variant(
                    workspace / key,
                    corpus=corpus,
                    query_texts=query_texts,
                    passage_vectors=passage_vectors,
                    query_vectors=query_vectors,
                    storage=storage,
                    exact=exact,
                    top_k=top_k,
                    iterations=iterations,
                    ground_truth=ground_truth,
                )
            except Exception as exc:
                variants[key] = {"error": f"{type(exc).__name__}: {exc}"}

    gates = {
        name: {
            "recall_ok": result.get("recall_at_k") is not None
            and result["recall_at_k"] >= recall_floor,
            "rank_ok": result.get("rank_correlation") is not None
            and result["rank_correlation"] >= rank_floor,
        }
        for name, result in variants.items()
    }
    return {
        "schema_version": 1,
        "corpus": {
            "passages": len(corpus),
            "queries": len(queries),
            "topics": len(RETRIEVAL_TOPICS),
            "digest": digest.hexdigest(),
        },
        "top_k": top_k,
        "iterations": iterations,
        "lancedb_version": lancedb.__version__,
        "thresholds": {"recall_at_k": recall_floor, "rank_correlation": rank_floor},
        "baseline_self_recall": variants["float32_exact"].get("recall_at_k"),
        "variants": variants,
        "gates": gates,
    }


def _run_in_workspace(
    paths: RuntimePaths,
    workspace: Path,
    *,
    files: int,
    functions_per_file: int,
    batch_size: int,
) -> dict[str, Any]:
    root = workspace / "corpus"
    source_bytes = write_benchmark_corpus(root, files=files, functions_per_file=functions_per_file)
    app = _benchmark_application(paths, workspace, batch_size=batch_size, cwd=root)
    result = run_index_benchmark(app, root)
    result.update(
        {
            **_benchmark_metadata(app),
            "embedding_batch_size": batch_size,
            "corpus": {
                "files": files,
                "functions_per_file": functions_per_file,
                "source_bytes": source_bytes,
            },
            "revision": update_check.checkout_head(Path(__file__).resolve().parents[2]),
        }
    )
    return result


def run_index_benchmark_command(
    paths: RuntimePaths,
    *,
    files: int,
    functions_per_file: int,
    batch_size: int,
    work_dir: Path | None,
) -> dict[str, Any]:
    """Create an isolated workspace and run the CLI benchmark."""
    if files < 1 or functions_per_file < 1:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark corpus dimensions must be positive",
        )
    if not 1 <= batch_size <= 256:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark batch size must be from 1 to 256",
        )
    if work_dir is not None:
        workspace = work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if (workspace / "corpus").exists() or (workspace / "data").exists():
            raise CodeIndexingError(
                ErrorCode.INVALID_CONFIGURATION,
                f"Benchmark work directory is not fresh: {workspace}",
            )
        return _run_in_workspace(
            paths,
            workspace,
            files=files,
            functions_per_file=functions_per_file,
            batch_size=batch_size,
        )
    with tempfile.TemporaryDirectory(prefix="code-indexing-mcp-index-benchmark-") as temporary:
        return _run_in_workspace(
            paths,
            Path(temporary),
            files=files,
            functions_per_file=functions_per_file,
            batch_size=batch_size,
        )


def run_search_benchmark_command(
    paths: RuntimePaths,
    *,
    projects: int,
    iterations: int,
    work_dir: Path | None,
) -> dict[str, Any]:
    """Create an isolated workspace of *projects* corpora and run the search benchmark."""
    if not 1 <= projects <= 200:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark project count must be from 1 to 200",
        )
    if not 1 <= iterations <= 20:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark iterations must be from 1 to 20",
        )

    def _run(workspace: Path) -> dict[str, Any]:
        app = _benchmark_application(paths, workspace)
        roots = [workspace / f"project_{index:03d}" for index in range(projects)]
        result = run_search_benchmark(app, roots, iterations=iterations)
        result.update(
            {
                **_benchmark_metadata(app),
            }
        )
        return result

    if work_dir is not None:
        workspace = work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if any((workspace / name).exists() for name in ("corpus", "data", "project_000")):
            raise CodeIndexingError(
                ErrorCode.INVALID_CONFIGURATION,
                f"Benchmark work directory is not fresh: {workspace}",
            )
        return _run(workspace)
    with tempfile.TemporaryDirectory(prefix="code-indexing-mcp-search-benchmark-") as temporary:
        return _run(Path(temporary))


def run_precision_benchmark_command(
    paths: RuntimePaths,
    *,
    passages: int,
    iterations: int,
    recall_floor: float,
    rank_floor: float,
    work_dir: Path | None,
) -> dict[str, Any]:
    """Create an isolated workspace and run the vector-precision benchmark."""
    if not len(RETRIEVAL_TOPICS) <= passages <= 100_000:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            f"Benchmark passage count must be from {len(RETRIEVAL_TOPICS)} to 100000",
        )
    if not 1 <= iterations <= 20:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark iterations must be from 1 to 20",
        )
    if not 0 < recall_floor <= 1 or not 0 < rank_floor <= 1:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark gate thresholds must be within (0, 1]",
        )

    def _run(workspace: Path) -> dict[str, Any]:
        app = _benchmark_application(paths, workspace)
        result = run_precision_benchmark(
            app.embedder,
            workspace / "precision",
            passages=passages,
            iterations=iterations,
            recall_floor=recall_floor,
            rank_floor=rank_floor,
        )
        result.update(
            {
                **_benchmark_metadata(app),
            }
        )
        return result

    if work_dir is not None:
        workspace = work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if any((workspace / name).exists() for name in ("precision", "data")):
            raise CodeIndexingError(
                ErrorCode.INVALID_CONFIGURATION,
                f"Benchmark work directory is not fresh: {workspace}",
            )
        return _run(workspace)
    with tempfile.TemporaryDirectory(prefix="code-indexing-mcp-precision-benchmark-") as temporary:
        return _run(Path(temporary))


GRAPHRAG_ITERATIONS = 3
GRAPHRAG_NOISE_FILES = 60


@dataclass(frozen=True)
class GraphRagQueryCase:
    """One judged retrieval query for the Hybrid GraphRAG A/B benchmark."""

    category: str
    query: str
    relevant_targets: tuple[tuple[str, str], ...]


def write_graphrag_benchmark_corpus(
    root: Path,
    *,
    noise_files: int = GRAPHRAG_NOISE_FILES,
) -> tuple[int, list[GraphRagQueryCase]]:
    """Write a multi-language judged repository testing GraphRAG retrieval cases.

    Categories:
      - ``multi_hop``: user-facing entrypoint vocabulary calls low-level systems
        callees whose implementations share zero lexical terms with the query.
      - ``barrel_reexport``: TypeScript barrel ``index.ts`` re-exports concrete
        implementations imported through the barrel by callers; ground truth is
        the concrete definition rather than the empty re-export stub.
      - ``hub_resistance``: a shared telemetry hub is called by every module,
        testing that specificity-adjusted PPR promotes domain callees rather
        than the global utility hub.
      - ``exact_symbol``: exact identifier lookups ensuring zero rank regression.
    """
    if noise_files < 0:
        raise ValueError("noise_files must be non-negative")
    src = root / "src"
    ts_pkg = root / "web" / "ledger"
    src.mkdir(parents=True, exist_ok=False)
    ts_pkg.mkdir(parents=True, exist_ok=False)

    files_to_write: dict[Path, str] = {
        src / "telemetry_hub.py": (
            "def emit_global_span(operation: str) -> None:\n    _ = operation\n"
        ),
        # Multi-hop pair 1: OAuth bearer header rotation -> Ed25519 curve + HKDF ratchet
        src / "crypto_ed25519.py": (
            "def ed25519_curve_point_verify(wire_bytes: bytes, pub_scalar: bytes) -> bool:\n"
            "    return wire_bytes[:4] == pub_scalar[:4]\n\n"
            "def hkdf_sha512_subkey_derive(master_seed: bytes, epoch: int) -> bytes:\n"
            "    return master_seed + epoch.to_bytes(4, 'big')\n"
        ),
        src / "auth_gateway.py": (
            "from .crypto_ed25519 import ed25519_curve_point_verify, hkdf_sha512_subkey_derive\n"
            "from .telemetry_hub import emit_global_span\n\n"
            "def validate_oauth_bearer_rotation("
            "header_value: bytes, root_material: bytes) -> bool:\n"
            "    emit_global_span('oauth_bearer')\n"
            "    derived = hkdf_sha512_subkey_derive(root_material, 1)\n"
            "    return ed25519_curve_point_verify(header_value, derived)\n"
        ),
        # Multi-hop pair 2: Stripe webhook replay protection -> HMAC constant-time digest
        src / "mac_primitives.py": (
            "def constant_time_xor_compare(digest_a: bytes, digest_b: bytes) -> bool:\n"
            "    acc = 0\n"
            "    for left, right in zip(digest_a, digest_b, strict=False):\n"
            "        acc |= left ^ right\n"
            "    return acc == 0\n"
        ),
        src / "webhook_ingress.py": (
            "from .mac_primitives import constant_time_xor_compare\n"
            "from .telemetry_hub import emit_global_span\n\n"
            "def handle_stripe_webhook_replay(body_mac: bytes, expected_mac: bytes) -> bool:\n"
            "    emit_global_span('stripe_webhook')\n"
            "    return constant_time_xor_compare(body_mac, expected_mac)\n"
        ),
        # Multi-hop pair 3: Tenant API rate limit burst -> Atomic CAS token bucket
        src / "cas_counter.py": (
            "def atomic_compare_and_swap_slots(slot_ring: list[int], cost: int) -> bool:\n"
            "    if slot_ring[0] < cost:\n"
            "        return False\n"
            "    slot_ring[0] -= cost\n"
            "    return True\n"
        ),
        src / "quota_middleware.py": (
            "from .cas_counter import atomic_compare_and_swap_slots\n"
            "from .telemetry_hub import emit_global_span\n\n"
            "def enforce_tenant_rate_limit_burst(bucket_state: list[int]) -> bool:\n"
            "    emit_global_span('tenant_quota')\n"
            "    return atomic_compare_and_swap_slots(bucket_state, 1)\n"
        ),
        # Multi-hop pair 4: Encrypted disaster recovery backup -> AES-GCM envelope seal
        src / "aead_envelope.py": (
            "def chacha20_poly1305_seal_block(plaintext_frame: bytes, nonce_iv: bytes) -> bytes:\n"
            "    return nonce_iv + plaintext_frame\n"
        ),
        src / "backup_coordinator.py": (
            "from .aead_envelope import chacha20_poly1305_seal_block\n"
            "from .telemetry_hub import emit_global_span\n\n"
            "def archive_disaster_recovery_snapshot(archive_blob: bytes, iv: bytes) -> bytes:\n"
            "    emit_global_span('disaster_backup')\n"
            "    return chacha20_poly1305_seal_block(archive_blob, iv)\n"
        ),
        # TypeScript barrel re-export cluster
        ts_pkg / "journal_writer.ts": (
            "export function commitDoubleEntryPosting("
            "debitCents: number, creditCents: number): boolean {\n"
            "    return debitCents === creditCents && debitCents > 0;\n"
            "}\n"
        ),
        ts_pkg / "fx_revaluation.ts": (
            "export function computeSpotRateBasisPoints("
            "notionalMinor: number, pipSpread: number): number {\n"
            "    return Math.trunc((notionalMinor * pipSpread) / 10000);\n"
            "}\n"
        ),
        ts_pkg / "index.ts": (
            "export { commitDoubleEntryPosting } from './journal_writer';\n"
            "export { computeSpotRateBasisPoints } from './fx_revaluation';\n"
        ),
        ts_pkg / "checkout_settlement.ts": (
            "import { commitDoubleEntryPosting, computeSpotRateBasisPoints } from './index';\n\n"
            "export function settleMerchantInvoicePayout(amount: number): boolean {\n"
            "    const fee = computeSpotRateBasisPoints(amount, 25);\n"
            "    return commitDoubleEntryPosting(amount - fee, amount - fee);\n"
            "}\n"
        ),
    }

    for idx in range(noise_files):
        files_to_write[src / f"a_distractor_{idx:03d}.py"] = (
            "from .telemetry_hub import emit_global_span\n\n"
            f"def background_worker_task_{idx:03d}(payload: dict[str, str]) -> str:\n"
            f"    emit_global_span('worker_{idx:03d}')\n"
            f"    return payload.get('status', 'idle_{idx:03d}')\n"
        )

    total_bytes = 0
    for path, content in sorted(files_to_write.items()):
        encoded = content.encode("utf-8")
        path.write_bytes(encoded)
        total_bytes += len(encoded)

    queries = [
        GraphRagQueryCase(
            category="multi_hop",
            query="validate oauth bearer rotation header",
            relevant_targets=(
                ("src/auth_gateway.py", "validate_oauth_bearer_rotation"),
                ("src/crypto_ed25519.py", "ed25519_curve_point_verify"),
                ("src/crypto_ed25519.py", "hkdf_sha512_subkey_derive"),
            ),
        ),
        GraphRagQueryCase(
            category="multi_hop",
            query="handle stripe webhook replay verification",
            relevant_targets=(
                ("src/webhook_ingress.py", "handle_stripe_webhook_replay"),
                ("src/mac_primitives.py", "constant_time_xor_compare"),
            ),
        ),
        GraphRagQueryCase(
            category="multi_hop",
            query="enforce tenant rate limit burst middleware",
            relevant_targets=(
                ("src/quota_middleware.py", "enforce_tenant_rate_limit_burst"),
                ("src/cas_counter.py", "atomic_compare_and_swap_slots"),
            ),
        ),
        GraphRagQueryCase(
            category="multi_hop",
            query="archive disaster recovery snapshot coordinator",
            relevant_targets=(
                ("src/backup_coordinator.py", "archive_disaster_recovery_snapshot"),
                ("src/aead_envelope.py", "chacha20_poly1305_seal_block"),
            ),
        ),
        GraphRagQueryCase(
            category="barrel_reexport",
            query="settle merchant invoice payout ledger",
            relevant_targets=(
                ("web/ledger/checkout_settlement.ts", "settleMerchantInvoicePayout"),
                ("web/ledger/journal_writer.ts", "commitDoubleEntryPosting"),
                ("web/ledger/fx_revaluation.ts", "computeSpotRateBasisPoints"),
            ),
        ),
        GraphRagQueryCase(
            category="hub_resistance",
            query="stripe webhook replay mac check",
            relevant_targets=(
                ("src/webhook_ingress.py", "handle_stripe_webhook_replay"),
                ("src/mac_primitives.py", "constant_time_xor_compare"),
            ),
        ),
        GraphRagQueryCase(
            category="exact_symbol",
            query="ed25519_curve_point_verify",
            relevant_targets=(("src/crypto_ed25519.py", "ed25519_curve_point_verify"),),
        ),
        GraphRagQueryCase(
            category="exact_symbol",
            query="commitDoubleEntryPosting",
            relevant_targets=(("web/ledger/journal_writer.ts", "commitDoubleEntryPosting"),),
        ),
        GraphRagQueryCase(
            category="exact_symbol",
            query="atomic_compare_and_swap_slots",
            relevant_targets=(("src/cas_counter.py", "atomic_compare_and_swap_slots"),),
        ),
        GraphRagQueryCase(
            category="exact_symbol",
            query="chacha20_poly1305_seal_block",
            relevant_targets=(("src/aead_envelope.py", "chacha20_poly1305_seal_block"),),
        ),
    ]
    return total_bytes, queries


def _score_graphrag_queries(
    app: Application,
    project_id: str,
    queries: Sequence[GraphRagQueryCase],
    *,
    iterations: int,
    limit: int = 10,
) -> dict[str, Any]:
    latency_samples: list[float] = []
    latest_responses: list[SearchResponse] = []
    for iteration in range(iterations):
        run_responses: list[SearchResponse] = []
        for case in queries:
            started = time.monotonic_ns()
            resp = app.search_code(case.query, projects=[project_id], limit=limit)
            elapsed_ms = (time.monotonic_ns() - started) / 1_000_000
            latency_samples.append(elapsed_ms)
            run_responses.append(resp)
        if iteration == iterations - 1:
            latest_responses = run_responses

    def _summarize_subset(
        pairs: Sequence[tuple[GraphRagQueryCase, SearchResponse]],
    ) -> dict[str, Any]:
        rr_scores: list[float] = []
        recalls_5: list[float] = []
        recalls_10: list[float] = []
        hub_contaminations_5: list[int] = []
        for case, resp in pairs:
            targets = set(case.relevant_targets)
            ranked = [(hit.path, hit.symbol or "") for hit in resp.hits]
            rr = 0.0
            for idx, key in enumerate(ranked[:10], start=1):
                if key in targets:
                    rr = 1.0 / idx
                    break
            rr_scores.append(rr)
            hits_5 = sum(1 for key in set(ranked[:5]) if key in targets)
            hits_10 = sum(1 for key in set(ranked[:10]) if key in targets)
            recalls_5.append(hits_5 / len(targets))
            recalls_10.append(hits_10 / len(targets))
            hub_in_top5 = any(sym == "emit_global_span" for _p, sym in ranked[:5])
            hub_contaminations_5.append(1 if hub_in_top5 else 0)
        return {
            "queries": len(pairs),
            "mrr_at_10": round(statistics.fmean(rr_scores), 4) if rr_scores else 0.0,
            "recall_at_5": round(statistics.fmean(recalls_5), 4) if recalls_5 else 0.0,
            "recall_at_10": round(statistics.fmean(recalls_10), 4) if recalls_10 else 0.0,
            "hub_in_top_5_rate": (
                round(statistics.fmean(hub_contaminations_5), 4) if hub_contaminations_5 else 0.0
            ),
        }

    paired = list(zip(queries, latest_responses, strict=True))
    categories: dict[str, dict[str, Any]] = {}
    for category in ("multi_hop", "barrel_reexport", "hub_resistance", "exact_symbol"):
        subset = [item for item in paired if item[0].category == category]
        if subset:
            categories[category] = _summarize_subset(subset)

    return {
        "overall": _summarize_subset(paired),
        "by_category": categories,
        "latency_ms": _duration_summary(latency_samples),
    }


def run_graphrag_benchmark(
    paths: RuntimePaths,
    workspace: Path,
    *,
    noise_files: int = GRAPHRAG_NOISE_FILES,
    iterations: int = GRAPHRAG_ITERATIONS,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Run an A/B retrieval and latency evaluation of Hybrid Code GraphRAG."""
    corpus_root = workspace / "corpus"
    source_bytes, queries = write_graphrag_benchmark_corpus(corpus_root, noise_files=noise_files)

    digest = hashlib.sha256()
    for file_path in sorted(corpus_root.rglob("*")):
        if file_path.is_file():
            digest.update(file_path.relative_to(corpus_root).as_posix().encode("utf-8"))
            digest.update(file_path.read_bytes())
    for case in queries:
        digest.update(case.query.encode("utf-8"))

    base_settings = IndexSettings.from_environment()
    runtime_paths = RuntimePaths(data=workspace / "data", cache=paths.cache)

    graph_settings = replace(
        base_settings,
        graph_search=True,
        index_execution="in-process",
        broker_mode="off",
    )
    baseline_settings = replace(
        base_settings,
        graph_search=False,
        index_execution="in-process",
        broker_mode="off",
    )

    graph_app = Application(
        runtime_paths,
        embedder=embedder,
        cwd=corpus_root,
        settings=graph_settings,
    )
    project = graph_app.init_project(corpus_root)
    index_report = graph_app.index_project(project.id, force=True)

    # Share the exact same embedder instance with the baseline application so
    # query-embedding setup cost does not skew A/B search latency.
    baseline_app = Application(
        runtime_paths,
        embedder=graph_app.embedder,
        cwd=corpus_root,
        settings=baseline_settings,
    )

    # Measure graph compilation, sidecar warm reload, and in-memory hit latency.
    graph_cache_dir = runtime_paths.data / "lancedb" / "graph-cache"
    if graph_cache_dir.exists():
        for bin_file in graph_cache_dir.rglob("*.bin"):
            bin_file.unlink()
    graph_app.graph.clear_memory_cache()

    t0 = time.monotonic_ns()
    compiled_graph = graph_app.graph.get_partition_graph(project.id)
    cold_compile_ms = round((time.monotonic_ns() - t0) / 1_000_000, 3)

    graph_app.graph.clear_memory_cache()
    t1 = time.monotonic_ns()
    warm_graph = graph_app.graph.get_partition_graph(project.id)
    warm_sidecar_ms = round((time.monotonic_ns() - t1) / 1_000_000, 3)

    t2 = time.monotonic_ns()
    _ = graph_app.graph.get_partition_graph(project.id)
    memory_hit_ms = round((time.monotonic_ns() - t2) / 1_000_000, 3)

    sidecar_bytes = (
        sum(entry.stat().st_size for entry in graph_cache_dir.rglob("*.bin"))
        if graph_cache_dir.exists()
        else 0
    )

    baseline_metrics = _score_graphrag_queries(
        baseline_app, project.id, queries, iterations=iterations
    )
    hybrid_metrics = _score_graphrag_queries(graph_app, project.id, queries, iterations=iterations)

    base_overall = baseline_metrics["overall"]
    hyb_overall = hybrid_metrics["overall"]
    base_mh = baseline_metrics["by_category"]["multi_hop"]
    hyb_mh = hybrid_metrics["by_category"]["multi_hop"]
    base_exact = baseline_metrics["by_category"]["exact_symbol"]
    hyb_exact = hybrid_metrics["by_category"]["exact_symbol"]

    graph_ref = warm_graph or compiled_graph
    return {
        "schema_version": 1,
        **_benchmark_metadata(graph_app),
        "corpus": {
            "files": index_report.indexed_files,
            "noise_files": noise_files,
            "source_bytes": source_bytes,
            "chunks": index_report.embedded_chunks,
            "references": index_report.staged_reference_rows,
            "graph_nodes": len(graph_ref.nodes) if graph_ref is not None else 0,
            "graph_edges": len(graph_ref.out_indices) if graph_ref is not None else 0,
            "barrel_files": len(graph_ref.barrel_file_ids) if graph_ref is not None else 0,
            "queries": len(queries),
            "digest": digest.hexdigest(),
        },
        "iterations": iterations,
        "graph_cache": {
            "cold_compile_ms": cold_compile_ms,
            "warm_sidecar_load_ms": warm_sidecar_ms,
            "memory_hit_ms": memory_hit_ms,
            "sidecar_bytes": sidecar_bytes,
        },
        "variants": {
            "baseline": baseline_metrics,
            "hybrid_graphrag": hybrid_metrics,
        },
        "deltas": {
            "mrr_at_10": round(hyb_overall["mrr_at_10"] - base_overall["mrr_at_10"], 4),
            "recall_at_5": round(hyb_overall["recall_at_5"] - base_overall["recall_at_5"], 4),
            "recall_at_10": round(hyb_overall["recall_at_10"] - base_overall["recall_at_10"], 4),
            "multi_hop_recall_at_5": round(hyb_mh["recall_at_5"] - base_mh["recall_at_5"], 4),
            "median_latency_ms": round(
                hybrid_metrics["latency_ms"]["median_ms"]
                - baseline_metrics["latency_ms"]["median_ms"],
                3,
            ),
        },
        "gates": {
            "overall_recall_improved": hyb_overall["recall_at_5"] > base_overall["recall_at_5"],
            "multi_hop_recall_improved": hyb_mh["recall_at_5"] > base_mh["recall_at_5"],
            "exact_symbol_no_regression": hyb_exact["mrr_at_10"] >= base_exact["mrr_at_10"],
        },
    }


def run_graphrag_benchmark_command(
    paths: RuntimePaths,
    *,
    noise_files: int = GRAPHRAG_NOISE_FILES,
    iterations: int = GRAPHRAG_ITERATIONS,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Create an isolated workspace and run the Hybrid GraphRAG A/B benchmark."""
    if not 0 <= noise_files <= 1_000:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark noise_files must be from 0 to 1000",
        )
    if not 1 <= iterations <= 20:
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Benchmark iterations must be from 1 to 20",
        )
    if work_dir is not None:
        workspace = work_dir.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if (workspace / "corpus").exists() or (workspace / "data").exists():
            raise CodeIndexingError(
                ErrorCode.INVALID_CONFIGURATION,
                f"Benchmark work directory is not fresh: {workspace}",
            )
        return run_graphrag_benchmark(
            paths,
            workspace,
            noise_files=noise_files,
            iterations=iterations,
        )
    with tempfile.TemporaryDirectory(prefix="code-indexing-mcp-graphrag-benchmark-") as temporary:
        return run_graphrag_benchmark(
            paths,
            Path(temporary),
            noise_files=noise_files,
            iterations=iterations,
        )
