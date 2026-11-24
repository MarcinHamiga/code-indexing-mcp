"""Unit and process-lifecycle tests for ``code_indexing_mcp.parse_pool``."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import (
    REFERENCE_SCHEMA_VERSION,
    ExtractedChunk,
    StoredFile,
    content_digest,
)
from code_indexing_mcp.parse_pool import (
    PARSE_MEMORY_RESERVE_BYTES,
    PARSE_WORKER_MEMORY_ESTIMATE_BYTES,
    ParseTask,
    ParseWorkerPool,
    build_reference_columns,
    content_rejection,
    resolve_parse_workers,
)
from code_indexing_mcp.staging import VARYING_REFERENCE_COLUMNS
from code_indexing_mcp.storage import LanceStore

_CONSTANT_REFERENCE_COLUMNS = (
    "file_id",
    "project_id",
    "path",
    "language",
    "content_hash",
    "schema_version",
)

_GIB = 1024 * 1024 * 1024


def _live_worker_children() -> list[psutil.Process]:
    """Return live child processes excluding Python's singleton resource_tracker."""
    live: list[psutil.Process] = []
    for child in psutil.Process().children(recursive=True):
        try:
            cmdline = " ".join(child.cmdline())
        except psutil.Error:
            continue
        if "multiprocessing.resource_tracker" in cmdline:
            continue
        live.append(child)
    return live


def _chunk_from_tuple(row: tuple[object, ...]) -> ExtractedChunk:
    return ExtractedChunk.model_construct(
        kind=row[0],  # type: ignore[arg-type]
        symbol=row[1],  # type: ignore[arg-type]
        qualified_symbol=row[2],  # type: ignore[arg-type]
        parent_symbol=row[3],  # type: ignore[arg-type]
        start_byte=row[4],  # type: ignore[arg-type]
        end_byte=row[5],  # type: ignore[arg-type]
        start_line=row[6],  # type: ignore[arg-type]
        end_line=row[7],  # type: ignore[arg-type]
        content=row[8],  # type: ignore[arg-type]
        embedding_text=row[9],  # type: ignore[arg-type]
        search_text=row[10],  # type: ignore[arg-type]
        part_index=row[11],  # type: ignore[arg-type]
        is_test=row[12],  # type: ignore[arg-type]
        embedding_prefix=row[13],  # type: ignore[arg-type]
        search_suffix=row[14],  # type: ignore[arg-type]
        source_start_byte=None,
        source_end_byte=None,
        original_end_byte=None,
        source_content_hash=None,
    )


def test_resolve_parse_workers_handles_explicit_cpu_and_memory_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_parse_workers(1, 512) == 1
    assert resolve_parse_workers(1, 16 * _GIB) == 1
    assert resolve_parse_workers(2, 512) == 2
    assert resolve_parse_workers(6, 2 * _GIB) == 6
    assert resolve_parse_workers(16, 2 * _GIB) == 16
    assert resolve_parse_workers(32, 16 * _GIB) == 16

    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert resolve_parse_workers(0, 16 * _GIB) == 1

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert resolve_parse_workers(0, 16 * _GIB) == 1

    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert resolve_parse_workers(0, 16 * _GIB) == 4

    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert resolve_parse_workers(0, 16 * _GIB) == 8
    assert resolve_parse_workers(-1, 16 * _GIB) == 8

    # Memory-capped auto sizing reserves 1 GiB and allocates 160 MiB per worker.
    assert resolve_parse_workers(0, 512) == 1
    assert resolve_parse_workers(0, PARSE_MEMORY_RESERVE_BYTES) == 1
    assert (
        resolve_parse_workers(
            0, PARSE_MEMORY_RESERVE_BYTES + 3 * PARSE_WORKER_MEMORY_ESTIMATE_BYTES
        )
        == 3
    )
    assert resolve_parse_workers(0, 2 * _GIB) == 6


def test_varying_reference_columns_matches_schema_and_build_reference_columns() -> None:
    expected_varying = (
        "reference_id",
        "record_kind",
        "kind",
        "source_qualified_symbol",
        "written_name",
        "target_name",
        "module_path",
        "imported_name",
        "alias",
        "receiver_text",
        "start_byte",
        "end_byte",
        "start_line",
        "end_line",
        "shape_json",
    )
    assert expected_varying == VARYING_REFERENCE_COLUMNS
    schema_names = LanceStore.reference_arrow_schema().names
    assert set(schema_names) == set(VARYING_REFERENCE_COLUMNS) | set(_CONSTANT_REFERENCE_COLUMNS)
    assert len(schema_names) == len(VARYING_REFERENCE_COLUMNS) + len(_CONSTANT_REFERENCE_COLUMNS)

    samples: list[tuple[str, str, bytes]] = [
        (
            "app/service.py",
            "python",
            (
                b"from pkg.util import helper as h\n\n"
                b"class Service:\n"
                b"    def run(self, count: int, *args: str, flag: bool = False) -> int:\n"
                b"        return h(count, *args, flag=flag)\n"
            ),
        ),
        (
            "web/client.ts",
            "typescript",
            (
                b"import { fetchUser } from './api';\n\n"
                b"export class Client {\n"
                b"  async load(id: string, verbose?: boolean): Promise<string> {\n"
                b"    return fetchUser(id, verbose);\n"
                b"  }\n"
                b"}\n"
            ),
        ),
        (
            "cmd/worker.go",
            "go",
            (
                b'package main\n\nimport "fmt"\n\n'
                b"func greet(name string, times int) string {\n"
                b'    return fmt.Sprintf("%s:%d", name, times)\n'
                b"}\n"
            ),
        ),
        (
            "src/lib.rs",
            "rust",
            (b"use std::cmp::max;\n\npub fn bound(a: i32, b: i32) -> i32 {\n    max(a, b)\n}\n"),
        ),
        ("empty.py", "python", b"# empty module\n"),
    ]

    extractor = TreeSitterExtractor()
    for rel_posix, language, source in samples:
        extraction = extractor.extract(Path(rel_posix), language, source)
        assert extraction.has_errors is False
        file_id = content_digest(f"proj\0{rel_posix}")
        content_hash = content_digest(source)
        record = StoredFile(
            file_id=file_id,
            project_id="proj",
            path=rel_posix,
            language=language,
            size=len(source),
            mtime_ns=12345,
            content_hash=content_hash,
            indexed_at=99999,
        )
        expected_rows = Indexer._reference_rows(
            "proj", record, extraction.references, extraction.declarations
        )
        varying_cols = build_reference_columns(
            file_id, extraction.references, extraction.declarations
        )
        assert len(varying_cols) == len(VARYING_REFERENCE_COLUMNS)
        row_count = len(varying_cols[0])
        assert row_count == len(expected_rows)
        assert all(len(col) == row_count for col in varying_cols)
        assert varying_cols[1][-1] == "coverage"

        for row_idx, expected_row in enumerate(expected_rows):
            for col_idx, col_name in enumerate(VARYING_REFERENCE_COLUMNS):
                assert varying_cols[col_idx][row_idx] == getattr(expected_row, col_name), (
                    rel_posix,
                    row_idx,
                    col_name,
                )
            assert expected_row.schema_version == REFERENCE_SCHEMA_VERSION


def test_content_rejection_and_parse_pool_extraction_parity_and_cleanup(tmp_path: Path) -> None:
    assert content_rejection(b"print('ok')\n") is None
    assert content_rejection(b"a\x00b") == "binary"
    assert content_rejection(b"# \xff\xfe") == "encoding"

    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def escaped():\n    return 1\n", encoding="utf-8")

    file_specs: list[tuple[str, str, bytes]] = []
    for idx in range(12):
        file_specs.append(
            (
                f"py/mod_{idx}.py",
                "python",
                (
                    f"def py_fn_{idx}(x: int) -> int:\n"
                    f"    return py_helper_{idx}(x + {idx})\n\n"
                    f"def py_helper_{idx}(y: int) -> int:\n    return y * 2\n"
                ).encode(),
            )
        )
    for idx in range(10):
        file_specs.append(
            (
                f"ts/mod_{idx}.ts",
                "typescript",
                (
                    f"export function tsFn{idx}(n: number): number {{\n  return n + {idx};\n}}\n"
                ).encode(),
            )
        )
    for idx in range(9):
        file_specs.append(
            (
                f"go/mod_{idx}.go",
                "go",
                (
                    f"package p{idx}\n\nfunc goFn{idx}(v int) int {{\n    return v + {idx}\n}}\n"
                ).encode(),
            )
        )
    for idx in range(9):
        file_specs.append(
            (
                f"rs/mod_{idx}.rs",
                "rust",
                (f"pub fn rs_fn_{idx}(v: i32) -> i32 {{\n    v + {idx}\n}}\n").encode(),
            )
        )
    assert len(file_specs) == 40

    tasks: list[ParseTask] = []
    for rel_posix, language, content in file_specs:
        target = repo / rel_posix
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        tasks.append((rel_posix, language, None, None, False, False))

    # Add special cases: binary, invalid UTF-8, symlink, directory, and missing path.
    (repo / "bad_binary.py").write_bytes(b"def b():\x00\n")
    (repo / "bad_encoding.py").write_bytes(b"# caf\xe9\n")
    (repo / "link.py").symlink_to(outside)
    (repo / "as_dir.py").mkdir()
    tasks.extend(
        [
            ("bad_binary.py", "python", None, None, False, False),
            ("bad_encoding.py", "python", None, None, False, False),
            ("link.py", "python", None, None, False, False),
            ("as_dir.py", "python", None, None, False, False),
            ("missing.py", "python", None, None, False, False),
        ]
    )

    project_id = "proj-1"
    max_file_bytes = 1_048_576
    single_extractor = TreeSitterExtractor()

    pool_outcomes: dict[str, tuple[object, ...]] = {}
    pool = ParseWorkerPool(workers=2)
    try:
        assert len(_live_worker_children()) == 2
        for offset in range(0, len(tasks), 8):
            pool.submit(repo, max_file_bytes, project_id, tasks[offset : offset + 8])
            while pool.pending_count >= pool.max_in_flight:
                batch = pool.drain_one()
                assert batch is not None
                for entry in batch:
                    pool_outcomes[entry[1]] = entry
        for batch in pool.drain_all():
            for entry in batch:
                pool_outcomes[entry[1]] = entry
    finally:
        pool.close()

    # Wait briefly if OS process reaping lags by a tick, then verify zero worker children remain.
    deadline = time.monotonic() + 2.0
    while _live_worker_children() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert multiprocessing.active_children() == []
    assert _live_worker_children() == []

    for rel_posix, language, source in file_specs:
        entry = pool_outcomes[rel_posix]
        assert entry[0] == "parsed"
        assert entry[2] == language
        assert entry[5] == content_digest(source)
        assert entry[6] == content_digest(f"{project_id}\0{rel_posix}")
        assert entry[7] == len(source)
        expected = single_extractor.extract(Path(rel_posix), language, source)
        assert entry[8] == expected.has_errors
        actual_chunks = [_chunk_from_tuple(row) for row in entry[9]]  # type: ignore[arg-type]
        assert actual_chunks == expected.chunks
        expected_cols = build_reference_columns(
            str(entry[6]), expected.references, expected.declarations
        )
        assert entry[10] == expected_cols
        assert entry[11] == len(expected.references)

    assert pool_outcomes["bad_binary.py"][0] == "rejected"
    assert pool_outcomes["bad_binary.py"][6] == "binary"
    assert pool_outcomes["bad_encoding.py"][0] == "rejected"
    assert pool_outcomes["bad_encoding.py"][6] == "encoding"
    assert pool_outcomes["link.py"][0] == "read_skip"
    assert pool_outcomes["link.py"][2] == "symlink"
    assert pool_outcomes["as_dir.py"][0] == "read_skip"
    assert pool_outcomes["missing.py"][0] == "read_skip"

    # Also test close(abort=True) and drain_one(stop_event) aborting with pending batches.
    stop_event = threading.Event()
    abort_pool = ParseWorkerPool(workers=2)
    try:
        assert len(_live_worker_children()) == 2
        abort_pool.submit(repo, max_file_bytes, project_id, tasks[:8])
        stop_event.set()
        assert abort_pool.drain_one(stop_event) is None
    finally:
        abort_pool.close(abort=True)

    deadline = time.monotonic() + 2.0
    while _live_worker_children() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert multiprocessing.active_children() == []
    assert _live_worker_children() == []


def test_parse_pool_import_is_lean_without_heavy_storage_deps() -> None:
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_src}{os.pathsep}{existing_pp}" if existing_pp else repo_src

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, code_indexing_mcp.parse_pool; "
                "forbidden = ['lancedb', 'pyarrow', 'filelock', 'mcp', 'numpy', "
                "'code_indexing_mcp.storage', 'code_indexing_mcp.indexing', "
                "'code_indexing_mcp.staging']; "
                "loaded = [m for m in forbidden if m in sys.modules]; "
                "assert not loaded, f'unexpectedly imported: {loaded}'"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
