#!/usr/bin/env python3
"""Measure approximate-vector-index recall and latency against real embeddings.

``LanceStore`` builds an ``IVF_HNSW_SQ`` index on the chunk vectors once a
partition holds ``VECTOR_INDEX_MIN_ROWS`` rows and the operator has set
``CODE_INDEXING_VECTOR_INDEX=hnsw``. The 2026-09-02 review deferred deciding
whether that gate is right, and whether ``hnsw`` should ever become the default,
until recall had been measured at scale on real vectors rather than the random
ones an ANN index handles worst.

This script takes the vectors already embedded into one or more index data
directories (the output of ``profile_query_path.py`` is convenient), builds an
exact float32 cosine reference over subsets of increasing size, and for each
size reports:

* flat (``bypass_vector_index``) search latency — what ``exact`` mode costs;
* ``IVF_HNSW_SQ`` build time, index bytes, and search latency plus recall@k
  against the exact reference for a grid of query-time settings: ``nprobes``
  (IVF partitions visited), ``ef`` (HNSW beam width), and ``refine_factor``
  (re-score ``factor x k`` candidates with the stored vectors, which undoes the
  scalar-quantisation loss at the price of reading those rows).

Queries are the kind the tool receives: a fixed list of natural-language code
searches plus short symbol-and-path phrases derived from randomly chosen chunks,
all embedded with the project's query embedder.

Sizes above the number of real vectors available are reached, when
``--augment-to`` is given, by perturbing real vectors with small Gaussian noise
and renormalising. Those rows are marked ``synthetic`` in the output and should
be read as an upper bound on how the manifold scales, not as a measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import HnswSq

from code_indexing_mcp.application import RuntimePaths
from code_indexing_mcp.embedding import FastEmbedder

HANDWRITTEN_QUERIES = [
    "validate form field values before saving a model",
    "render a template with a request context",
    "parse an HTTP date header",
    "build a database query with joins and filters",
    "run pending migrations against the database",
    "cache a view response for a number of seconds",
    "escape HTML in a template variable",
    "serialize a queryset to JSON",
    "authenticate a user from a session cookie",
    "hash a password with PBKDF2",
    "handle a file upload and store it on disk",
    "parse a URL pattern and resolve it to a view",
    "send an email with attachments",
    "convert a datetime to the current timezone",
    "middleware that sets security headers",
    "compute the SQL for a many-to-many relation",
    "signal fired after a model instance is saved",
    "collect static files into the deployment directory",
    "paginate a list of objects",
    "check CSRF token on a POST request",
    "lazy translation of a string",
    "load a fixture into the test database",
    "generate a random secret key",
    "stream a large HTTP response in chunks",
    "connection pooling for the database backend",
    "raise a 404 response when an object is missing",
    "format a number with thousands separators",
    "walk a directory and find template files",
    "create an index in a schema editor",
    "parse command line arguments for a management command",
    "compress and decompress geometry data",
    "check whether a field value is unique",
    "return the primary key of a related object",
    "register a model with the admin site",
    "reverse a named URL with keyword arguments",
    "decode a signed value and verify its signature",
    "truncate a string to a maximum number of words",
    "iterate rows from a database cursor",
    "apply default ordering to a queryset",
    "clone a queryset without evaluating it",
]

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _words(text: str) -> str:
    text = _CAMEL.sub(" ", text)
    return " ".join(part for part in re.split(r"[^A-Za-z0-9]+", text) if part).lower()


def _load_vectors(data_dirs: list[Path]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for data_dir in data_dirs:
        projects = data_dir / "lancedb" / "projects"
        if not projects.exists():
            continue
        for partition in sorted(projects.iterdir()):
            if not (partition / "chunks.lance").exists():
                continue
            table = lancedb.connect(partition).open_table("chunks")
            batch = table.search().select(["chunk_id", "path", "symbol", "vector"]).to_arrow()
            column = batch.column("vector")
            if column.num_chunks == 0:
                continue
            array = np.stack([np.asarray(row, dtype=np.float32) for row in column.to_pylist()])
            vectors.append(array)
            for row in batch.select(["chunk_id", "path", "symbol"]).to_pylist():
                row["source"] = str(partition)
                metadata.append(row)
    if not vectors:
        raise SystemExit("no chunk vectors found under the given data directories")
    stacked = np.concatenate(vectors, axis=0)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return stacked / norms, metadata


def _augment(real: np.ndarray, target: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    needed = target - len(real)
    picks = rng.integers(0, len(real), size=needed)
    noisy = real[picks] + rng.normal(0.0, sigma, size=(needed, real.shape[1])).astype(np.float32)
    noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
    return np.concatenate([real, noisy], axis=0)


def _latency(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median": round(statistics.median(ordered), 2),
        "p90": round(ordered[round(0.9 * (len(ordered) - 1))], 2),
        "max": round(ordered[-1], 2),
    }


def _directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _run_size(
    corpus: np.ndarray,
    queries: np.ndarray,
    *,
    top_k: int,
    variants: list[tuple[int | None, int | None, int | None]],
    workspace: Path,
) -> dict[str, Any]:
    size = len(corpus)
    truth = np.argsort(-(queries @ corpus.T), axis=1, kind="stable")[:, :top_k]

    table_dir = workspace / f"n{size}"
    if table_dir.exists():
        shutil.rmtree(table_dir)
    database = lancedb.connect(table_dir)
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float16(), corpus.shape[1])),
        ]
    )
    rows = pa.table(
        {
            "id": pa.array(np.arange(size, dtype=np.int64)),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(corpus.astype(np.float16).ravel(), type=pa.float16()), corpus.shape[1]
            ),
        },
        schema=schema,
    )
    table = database.create_table("vectors", rows)

    def run_queries(build: Any) -> tuple[list[float], float]:
        samples: list[float] = []
        hits = 0
        for index, query in enumerate(queries):
            started = time.perf_counter_ns()
            result = build(query.tolist()).select(["id"]).to_list()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            found = {row["id"] for row in result}
            hits += len(found & set(truth[index].tolist()))
        return samples, hits / (len(queries) * top_k)

    def flat(vector: list[float]) -> Any:
        return (
            table.search(vector, vector_column_name="vector")
            .distance_type("cosine")
            .limit(top_k)
            .bypass_vector_index()
        )

    flat_samples, flat_recall = run_queries(flat)
    result: dict[str, Any] = {
        "rows": size,
        "flat": {"latency_ms": _latency(flat_samples), "recall": round(flat_recall, 4)},
    }

    started = time.perf_counter()
    table.create_index("vector", config=HnswSq(distance_type="cosine"), replace=True)
    build_s = time.perf_counter() - started
    index_bytes = _directory_bytes(table_dir / "vectors.lance" / "_indices")
    result["hnsw"] = {"build_s": round(build_s, 2), "index_bytes": index_bytes, "variants": {}}
    for probes, ef, refine in variants:

        def ann(
            vector: list[float],
            probes: int | None = probes,
            ef: int | None = ef,
            refine: int | None = refine,
        ) -> Any:
            query = (
                table.search(vector, vector_column_name="vector")
                .distance_type("cosine")
                .limit(top_k)
            )
            if probes is not None:
                query = query.nprobes(probes)
            if ef is not None:
                query = query.ef(ef)
            if refine is not None:
                query = query.refine_factor(refine)
            return query

        samples, recall = run_queries(ann)
        key = f"nprobes={probes or 'default'},ef={ef or 'default'},refine={refine or 'none'}"
        result["hnsw"]["variants"][key] = {
            "nprobes": probes,
            "ef": ef,
            "refine_factor": refine,
            "latency_ms": _latency(samples),
            "recall": round(recall, 4),
        }
    shutil.rmtree(table_dir, ignore_errors=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--sizes", default="20000,50000,100000,all")
    parser.add_argument("--augment-to", type=int, action="append", default=[])
    parser.add_argument("--augment-sigma", type=float, default=0.05)
    parser.add_argument("--derived-queries", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nprobes", default="default,50")
    parser.add_argument("--ef", default="default,100,300")
    parser.add_argument("--refine", default="none,2,5,10")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    real, metadata = _load_vectors([path.resolve() for path in args.data_dir])
    order = rng.permutation(len(real))
    real = real[order]
    metadata = [metadata[index] for index in order]
    print(f"loaded {len(real)} real vectors of dimension {real.shape[1]}", file=sys.stderr)

    embedder = FastEmbedder(RuntimePaths.from_environment().cache / "models", offline=True)
    derived: list[str] = []
    for index in rng.choice(
        len(metadata), size=min(args.derived_queries, len(metadata)), replace=False
    ):
        row = metadata[index]
        phrase = _words(f"{Path(row['path']).stem} {row['symbol'] or ''}")
        if phrase:
            derived.append(phrase)
    texts = [*HANDWRITTEN_QUERIES, *derived]
    queries = np.asarray([embedder.embed_query(text) for text in texts], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    print(f"embedded {len(texts)} queries", file=sys.stderr)

    def parse(spec: str) -> list[int | None]:
        return [None if token in {"default", "none"} else int(token) for token in spec.split(",")]

    variants = [
        (probes, ef, refine)
        for probes in parse(args.nprobes)
        for ef in parse(args.ef)
        for refine in parse(args.refine)
    ]
    sizes: list[tuple[int, bool]] = []
    for token in args.sizes.split(","):
        if token == "all":
            sizes.append((len(real), False))
        elif int(token) <= len(real):
            sizes.append((int(token), False))
    for target in args.augment_to:
        if target > len(real):
            sizes.append((target, True))
    sizes = sorted(set(sizes))

    workspace = (args.workspace or Path(tempfile.mkdtemp(prefix="vector-recall-"))).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for size, synthetic in sizes:
        corpus = _augment(real, size, rng, args.augment_sigma) if synthetic else real[:size]
        print(f"size {size}{' (synthetic tail)' if synthetic else ''} ...", file=sys.stderr)
        entry = _run_size(corpus, queries, top_k=args.top_k, variants=variants, workspace=workspace)
        entry["synthetic"] = synthetic
        entry["real_rows"] = min(size, len(real))
        results.append(entry)
        flat = entry["flat"]["latency_ms"]["median"]
        print(
            f"  flat {flat:>8.2f} ms   hnsw build {entry['hnsw']['build_s']:>7.1f} s",
            file=sys.stderr,
        )
        for key, value in entry["hnsw"]["variants"].items():
            print(
                f"    {key:<40} {value['latency_ms']['median']:>7.2f} ms"
                f"  r@{args.top_k}={value['recall']:.3f}",
                file=sys.stderr,
            )

    output = {
        "schema_version": 1,
        "model_id": embedder.model_id,
        "dimension": int(real.shape[1]),
        "real_vectors": len(real),
        "sources": sorted({row["source"] for row in metadata}),
        "queries": {"handwritten": len(HANDWRITTEN_QUERIES), "derived": len(derived)},
        "top_k": args.top_k,
        "augment_sigma": args.augment_sigma,
        "sizes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
