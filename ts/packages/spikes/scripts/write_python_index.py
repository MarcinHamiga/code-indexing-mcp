"""Write a LanceDB index with the Python build, for the S1 spike to open.

S1 asks whether the JS SDK can open an index written by the Python build. That
question is only answered by an index the *Python* stack produced, so this
script runs under the project's own environment (`uv run` / `.venv`) and writes
the real thing: the partition layout `LanceStore` uses, the schemas from
`storage.py`, and -- the part most likely to differ across SDKs -- the FTS and
BTree indexes built by lancedb's Python bindings.

Deliberately built on raw `lancedb` + `pyarrow` rather than importing
`LanceStore`, so the spike depends on the storage *format* and not on the
indexing pipeline that will not exist in TypeScript until Phase 3.

Usage:
    .venv/bin/python write_python_index.py <output-directory>
"""

from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS, BTree

# Mirrors code_indexing_mcp.storage: 768 dimensions, float16 vectors since the
# schema version 5 bump.
VECTOR_DIMENSION = 768
VECTOR_DTYPE = pa.float16()
PROJECT_ID = "spike"
SCHEMA_VERSION = 5

CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("file_id", pa.string()),
        ("path", pa.string()),
        ("language", pa.string()),
        ("kind", pa.string()),
        ("symbol", pa.string()),
        ("qualified_symbol", pa.string()),
        ("parent_symbol", pa.string()),
        ("start_byte", pa.int64()),
        ("end_byte", pa.int64()),
        ("start_line", pa.int32()),
        ("end_line", pa.int32()),
        ("content", pa.string()),
        ("identifier_terms", pa.string()),
        ("content_hash", pa.string()),
        ("part_index", pa.int32()),
        ("vector", pa.list_(VECTOR_DTYPE, VECTOR_DIMENSION)),
    ]
)

PROJECT_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("name", pa.string()),
        ("root", pa.string()),
        ("payload", pa.string()),
        ("model_id", pa.string()),
        ("vector_dimension", pa.int32()),
        ("schema_version", pa.int32()),
        ("state", pa.string()),
        ("updated_at", pa.int64()),
    ]
)

# Content chosen so the S1 hybrid query has both a lexical hit (the literal
# token "tokenizer") and a semantically distinct decoy: an FTS index that
# silently failed to build would still return the decoy by vector alone, and
# the spike would not notice.
ROWS = [
    {
        "chunk_id": "spike:0",
        "path": "src/embedding/tokenizer.py",
        "symbol": "load_tokenizer",
        "content": "def load_tokenizer(directory):\n    return Tokenizer.from_file(directory)\n",
        "identifier_terms": "load tokenizer load_tokenizer embedding",
    },
    {
        "chunk_id": "spike:1",
        "path": "src/storage/partition.py",
        "symbol": "open_partition",
        "content": "def open_partition(root):\n    return connect(root / 'projects')\n",
        "identifier_terms": "open partition open_partition storage",
    },
    {
        "chunk_id": "spike:2",
        "path": "tests/test_tokenizer.py",
        "symbol": "test_tokenizer_roundtrip",
        "content": "def test_tokenizer_roundtrip():\n    assert load_tokenizer(path) is not None\n",
        "identifier_terms": "test tokenizer roundtrip test_tokenizer_roundtrip",
    },
]


def _vector(seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    raw = generator.standard_normal(VECTOR_DIMENSION).astype(np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float16).tolist()


def main(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    registry = lancedb.connect(destination / "registry", read_consistency_interval=timedelta(0))
    projects = registry.create_table("projects", schema=PROJECT_SCHEMA, exist_ok=True)
    projects.add(
        [
            {
                "id": PROJECT_ID,
                "name": "spike",
                "root": str(destination),
                "payload": "{}",
                "model_id": "jinaai/jina-embeddings-v2-base-code",
                "vector_dimension": VECTOR_DIMENSION,
                "schema_version": SCHEMA_VERSION,
                "state": "ready",
                "updated_at": time.time_ns(),
            }
        ]
    )

    partition = lancedb.connect(
        destination / "projects" / PROJECT_ID, read_consistency_interval=timedelta(0)
    )
    chunks = partition.create_table("chunks", schema=CHUNK_SCHEMA, exist_ok=True)
    chunks.add(
        [
            {
                "chunk_id": row["chunk_id"],
                "file_id": f"file:{index}",
                "path": row["path"],
                "language": "python",
                "kind": "function",
                "symbol": row["symbol"],
                "qualified_symbol": row["symbol"],
                "parent_symbol": "",
                "start_byte": 0,
                "end_byte": len(row["content"]),
                "start_line": 1,
                "end_line": 2,
                "content": row["content"],
                "identifier_terms": row["identifier_terms"],
                "content_hash": f"hash{index}",
                "part_index": 0,
                "vector": _vector(index),
            }
            for index, row in enumerate(ROWS)
        ]
    )

    # The index configuration the write path requires, built by the Python
    # bindings. Whether the JS SDK can *use* these is the interesting half of
    # S1: creating them from JS proves the API exists, reading Python-built
    # ones proves the on-disk index format is shared.
    for column in ("content", "identifier_terms"):
        chunks.create_index(
            column,
            config=FTS(lower_case=True, stem=False, remove_stop_words=False),
            replace=False,
        )
    for column in ("file_id", "language", "path", "symbol"):
        chunks.create_index(column, config=BTree(), replace=False)

    print(f"wrote {chunks.count_rows()} chunk rows to {destination}")
    print(f"lancedb {lancedb.__version__}, pyarrow {pa.__version__}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: write_python_index.py <output-directory>")
    main(Path(sys.argv[1]).resolve())
