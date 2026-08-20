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
from .accelerator_env import RECORD_FILENAME, load_environment, write_environment
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


def _benchmark_runtime_paths(paths: RuntimePaths, workspace: Path) -> RuntimePaths:
    """Keep a verified machine accelerator available to an isolated benchmark."""
    data = workspace / "data"
    environment = load_environment(paths.data).environment
    if environment is not None:
        write_environment(data / RECORD_FILENAME, environment)
    return RuntimePaths(data=data, cache=paths.cache)


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
    settings = replace(
        IndexSettings.from_environment(),
        embedding_batch_size=batch_size,
        embedding_batch_auto=False,
        # A benchmark measures the selected backend's complete worker path,
        # including model load. Deferral is a production latency policy, not a
        # performance result, and would turn small accelerator runs into CPU.
        embedding_crossover_characters=0,
        embedding_crossover_auto=False,
        broker_mode="off",
    )
    app = Application(
        _benchmark_runtime_paths(paths, workspace),
        cwd=root,
        settings=settings,
    )
    result = run_index_benchmark(app, root)
    result.update(
        {
            "model_id": app.embedder.model_id,
            "embedding_backend": app.effective_backend_selection.descriptor.accelerator.value,
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
        settings = replace(
            IndexSettings.from_environment(),
            index_execution="in-process",
            broker_mode="off",
        )
        app = Application(
            RuntimePaths(data=workspace / "data", cache=paths.cache),
            cwd=workspace,
            settings=settings,
        )
        roots = [workspace / f"project_{index:03d}" for index in range(projects)]
        result = run_search_benchmark(app, roots, iterations=iterations)
        result.update(
            {
                "model_id": app.embedder.model_id,
                "embedding_backend": app.effective_backend_selection.descriptor.accelerator.value,
                "revision": update_check.checkout_head(Path(__file__).resolve().parents[2]),
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
        settings = replace(
            IndexSettings.from_environment(),
            index_execution="in-process",
            broker_mode="off",
        )
        app = Application(
            RuntimePaths(data=workspace / "data", cache=paths.cache),
            cwd=workspace,
            settings=settings,
        )
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
                "model_id": app.embedder.model_id,
                "embedding_backend": app.effective_backend_selection.descriptor.accelerator.value,
                "revision": update_check.checkout_head(Path(__file__).resolve().parents[2]),
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
