#!/usr/bin/env python3
"""Profile the lazy-mode query path against a real repository.

The query-path track of the 2026-09-02 review remediation claims that a lazy-mode
tool call no longer walks the whole source tree on a dirty worktree, no longer
spawns git to check whether the repository moved, and no longer commits a
LanceDB write on read. ``tests/test_query_path_overhead.py`` asserts those
properties on synthetic trees; this script measures them on a real checkout so
the claims carry numbers.

It drives ``Application`` in-process with the broker off, reproducing what the
MCP server does for a lazy-mode tool call: ``project_status`` for every project
in scope (in parallel, as the server gathers them), ``index_project`` for any
that reports stale, then the query itself. Per scenario it records wall time,
how many ``git`` processes were spawned, and how many files under the data
directory changed, for every call:

* ``clean.*`` — the worktree matches the index; the steady-state cost of a call.
* ``dirty.first`` — one tracked file was edited; the call that notices and
  refreshes it.
* ``dirty.steady`` — the worktree is still dirty and nothing else changed; the
  review's headline finding was that every such call re-walked the tree.
* ``head.first`` / ``head.steady`` — the edit was committed so HEAD moved by
  one commit on the same branch; the call that notices and the ones after.
* ``multi.*`` — the same query over a scope of several small projects.

Both revisions cache a clean freshness verdict for a few seconds, so each
scenario is measured twice: ``burst`` calls follow each other immediately (what
an agent issuing several tool calls in a row sees), and ``gapped`` calls wait
``--gap-seconds`` first so the cache has expired and the call must decide
freshness again (what the first call after a pause sees). ``noticed`` records
whether the call that should have picked up an edit actually did, read from the
store's file records rather than inferred from timing.

Run it once per source revision with ``PYTHONPATH`` pointing at that revision's
``src``; the JSON output records which revision produced it. ``--copy-data-from``
seeds the data directory from a previous run so the repository is embedded once
and the later revision only performs its incremental refresh and migration.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.settings import IndexSettings

SEARCH_QUERY = "validate form field values before saving a model"
SYMBOL_NAME = "ModelForm"
OUTLINE_PATH = "django/forms/models.py"
DIRTY_TARGET = "django/utils/version.py"
MULTI_SUBTREE = "django/utils"


@dataclass
class GitSpawnCounter:
    """Count ``git`` subprocesses by wrapping ``subprocess.Popen``."""

    commands: list[list[str]] = field(default_factory=list)
    _original_popen: type[subprocess.Popen[Any]] | None = None

    def install(self) -> None:
        # ``subprocess.run`` builds a ``Popen`` internally, so wrapping only
        # ``Popen`` counts every spawn exactly once.
        self._original_popen = subprocess.Popen
        counter = self

        class CountingPopen(subprocess.Popen):  # type: ignore[type-arg]
            def __init__(self, args: Any, *rest: Any, **kwargs: Any) -> None:
                counter._record(args)
                super().__init__(args, *rest, **kwargs)

        subprocess.Popen = CountingPopen  # type: ignore[misc]

    def _record(self, args: Any) -> None:
        if isinstance(args, (list, tuple)) and args and Path(str(args[0])).name == "git":
            self.commands.append([str(part) for part in args])

    def take(self) -> list[list[str]]:
        taken, self.commands = self.commands, []
        return taken


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = Path(root) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path.relative_to(directory))] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _changed_files(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> int:
    return sum(1 for key, value in after.items() if before.get(key) != value)


@dataclass
class Sample:
    wall_ms: float
    git_spawns: int
    git_commands: list[str]
    data_files_changed: int


def _summary(samples: list[Sample]) -> dict[str, Any]:
    walls = [sample.wall_ms for sample in samples]
    ordered = sorted(walls)
    p90_index = min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))
    return {
        "calls": len(samples),
        "wall_ms": {
            "min": round(ordered[0], 2),
            "median": round(statistics.median(walls), 2),
            "mean": round(statistics.fmean(walls), 2),
            "p90": round(ordered[p90_index], 2),
            "max": round(ordered[-1], 2),
        },
        "git_spawns": {
            "min": min(sample.git_spawns for sample in samples),
            "max": max(sample.git_spawns for sample in samples),
            "total": sum(sample.git_spawns for sample in samples),
        },
        "data_files_changed": {
            "min": min(sample.data_files_changed for sample in samples),
            "max": max(sample.data_files_changed for sample in samples),
            "total": sum(sample.data_files_changed for sample in samples),
        },
        "git_commands_first_call": samples[0].git_commands,
    }


class Profiler:
    def __init__(self, app: Application, data_dir: Path, counter: GitSpawnCounter) -> None:
        self.app = app
        self.data_dir = data_dir
        self.counter = counter
        self.scenarios: dict[str, Any] = {}

    def measure(
        self, name: str, call: Callable[[], Any], *, iterations: int, gap: float = 0.0
    ) -> list[Sample]:
        samples: list[Sample] = []
        for _ in range(iterations):
            if gap > 0:
                time.sleep(gap)
            self.counter.take()
            before = _snapshot(self.data_dir)
            started = time.perf_counter_ns()
            call()
            wall_ms = (time.perf_counter_ns() - started) / 1_000_000
            after = _snapshot(self.data_dir)
            commands = self.counter.take()
            samples.append(
                Sample(
                    wall_ms=wall_ms,
                    git_spawns=len(commands),
                    git_commands=[" ".join(command[:3]) for command in commands],
                    data_files_changed=_changed_files(before, after),
                )
            )
        self.scenarios[name] = _summary(samples)
        print(
            f"  {name:<36} median {self.scenarios[name]['wall_ms']['median']:>9.2f} ms"
            f"  git {self.scenarios[name]['git_spawns']['max']:>3}"
            f"  data-files {self.scenarios[name]['data_files_changed']['max']:>3}",
            file=sys.stderr,
        )
        return samples

    def profile(
        self, name: str, call: Callable[[], Any], *, gap: float = 0.0, top: int = 30
    ) -> None:
        if gap > 0:
            time.sleep(gap)
        profiler = cProfile.Profile()
        profiler.enable()
        call()
        profiler.disable()
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer)
        stats.sort_stats("cumulative").print_stats(top)
        self.scenarios.setdefault(name, {})["cprofile_cumulative_top"] = buffer.getvalue()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def _source_revision() -> str:
    import code_indexing_mcp

    source_root = Path(code_indexing_mcp.__file__).resolve().parents[2]
    try:
        return _git(source_root, "rev-parse", "--short", "HEAD") + f" ({source_root})"
    except (OSError, subprocess.SubprocessError):
        return f"unknown ({source_root})"


def _make_small_projects(repo: Path, workspace: Path, count: int) -> list[Path]:
    roots: list[Path] = []
    for index in range(count):
        root = workspace / f"small-{index}"
        if not (root / ".git").exists():
            if root.exists():
                shutil.rmtree(root)
            shutil.copytree(repo / MULTI_SUBTREE, root / "pkg")
            _git(root, "init", "--quiet")
            _git(root, "add", ".")
            _git(
                root,
                "-c",
                "user.name=profile",
                "-c",
                "user.email=profile@example.invalid",
                "commit",
                "--quiet",
                "-m",
                f"small project {index}",
            )
        roots.append(root)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repo", type=Path, required=True, help="git checkout to profile")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--label", required=True, help="name for this run's data directory")
    parser.add_argument("--copy-data-from", type=Path, help="seed the data directory from here")
    parser.add_argument("--iterations", type=int, default=10, help="burst calls per scenario")
    parser.add_argument("--gap-iterations", type=int, default=5, help="gapped calls per scenario")
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=6.0,
        help="pause before each gapped call; must exceed the freshness cache window",
    )
    parser.add_argument("--small-projects", type=int, default=8)
    parser.add_argument("--cprofile", action="store_true", help="profile one dirty gapped call")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = workspace / f"data-{args.label}"
    if args.copy_data_from is not None and not data_dir.exists():
        shutil.copytree(args.copy_data_from.resolve(), data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    original_head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain"):
        print("refusing to profile a dirty repository; reset it first", file=sys.stderr)
        return 2

    settings = replace(
        IndexSettings.from_environment(),
        index_execution="in-process",
        broker_mode="off",
    )
    cache = RuntimePaths.from_environment().cache
    app = Application(RuntimePaths(data=data_dir, cache=cache), cwd=repo, settings=settings)
    counter = GitSpawnCounter()
    counter.install()
    profiler = Profiler(app, data_dir, counter)
    result: dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "source_revision": _source_revision(),
        "repository": str(repo),
        "repository_head": original_head,
        "index_mode": settings.mode.value,
        "vector_index": settings.vector_index,
        "iterations": args.iterations,
    }

    print(f"[{args.label}] source {result['source_revision']}", file=sys.stderr)
    project = app.init_project(repo)
    started = time.perf_counter()
    report = app.index_project(project.id)
    result["index"] = {
        "wall_s": round(time.perf_counter() - started, 1),
        "discovered_files": report.discovered_files,
        "indexed_files": report.indexed_files,
        "unchanged_files": report.unchanged_files,
        "embedded_chunks": report.embedded_chunks,
        "embedding_backend": report.embedding_backend,
        "errors": len(report.errors),
    }
    status = app.project_status(project.id)
    result["index"]["file_count"] = status.file_count
    result["index"]["chunk_count"] = status.chunk_count
    print(
        f"  indexed {status.file_count} files / {status.chunk_count} chunks"
        f" in {result['index']['wall_s']}s (embedded {report.embedded_chunks})",
        file=sys.stderr,
    )
    counter.take()

    executor = ThreadPoolExecutor(max_workers=8)

    def lazy_tool_call(project_ids: list[str], query: Callable[[], Any]) -> None:
        """What the server does before answering a lazy-mode query (server.py,
        ``_wait_for_startup_projects``): status per project, refresh the stale
        ones, then run the query."""
        statuses = list(executor.map(app.project_status, project_ids))
        for project_id, status in zip(project_ids, statuses, strict=True):
            if status.state not in {"ready", "partial"}:
                app.index_project(project_id, trigger="lazy-query")
        query()

    def status_only() -> None:
        app.project_status(project.id)

    def search() -> None:
        lazy_tool_call(
            [project.id], lambda: app.search_code(SEARCH_QUERY, projects=[project.id], limit=8)
        )

    def symbol() -> None:
        lazy_tool_call([project.id], lambda: app.find_symbol(SYMBOL_NAME, project.id))

    def outline() -> None:
        lazy_tool_call([project.id], lambda: app.file_outline(OUTLINE_PATH, project.id))

    def embed_only() -> None:
        app.embedder.embed_query(SEARCH_QUERY)

    iterations = args.iterations
    gap_iterations = args.gap_iterations
    gap = args.gap_seconds
    noticed: dict[str, bool] = {}

    def stored_hash(project_id: str, path: str) -> str | None:
        for record in app.store.list_files(project_id):
            if record.path == path:
                return record.content_hash
        return None

    # One warm-up call so model load and table-open costs are not attributed to
    # the clean scenario; the warm-up is reported on its own.
    profiler.measure("warmup.search_code", search, iterations=1)
    profiler.measure("clean.embed_query", embed_only, iterations=iterations)
    profiler.measure("clean.burst.status_only", status_only, iterations=iterations)
    profiler.measure("clean.gapped.status_only", status_only, iterations=gap_iterations, gap=gap)
    profiler.measure("clean.burst.search_code", search, iterations=iterations)
    profiler.measure("clean.gapped.search_code", search, iterations=gap_iterations, gap=gap)
    profiler.measure("clean.burst.find_symbol", symbol, iterations=iterations)
    profiler.measure("clean.burst.file_outline", outline, iterations=iterations)

    target = repo / DIRTY_TARGET
    original_text = target.read_text()
    hash_before = stored_hash(project.id, DIRTY_TARGET)
    target.write_text(
        original_text + "\n\ndef profile_query_path_marker():\n    return 'touched'\n"
    )
    try:
        profiler.measure("dirty.first.search_code", search, iterations=1, gap=gap)
        noticed["dirty.first.search_code"] = stored_hash(project.id, DIRTY_TARGET) != hash_before
        profiler.measure("dirty.burst.search_code", search, iterations=iterations)
        profiler.measure("dirty.gapped.search_code", search, iterations=gap_iterations, gap=gap)
        profiler.measure("dirty.burst.find_symbol", symbol, iterations=iterations)
        if args.cprofile:
            profiler.profile("dirty.gapped.search_code", search, gap=gap)
        _git(
            repo,
            "-c",
            "user.name=profile",
            "-c",
            "user.email=profile@example.invalid",
            "commit",
            "--quiet",
            "-am",
            "profile_query_path: move HEAD by one commit",
        )
        profiler.measure("head.first.search_code", search, iterations=1, gap=gap)
        profiler.measure("head.burst.search_code", search, iterations=iterations)
        profiler.measure("head.gapped.search_code", search, iterations=gap_iterations, gap=gap)
        profiler.measure("head.burst.find_symbol", symbol, iterations=iterations)
    finally:
        _git(repo, "reset", "--quiet", "--hard", original_head)
    profiler.measure("head_return.first.search_code", search, iterations=1, gap=gap)
    noticed["head_return.first.search_code"] = stored_hash(project.id, DIRTY_TARGET) == hash_before
    profiler.measure("head_return.burst.search_code", search, iterations=iterations)

    small_roots = _make_small_projects(repo, workspace, args.small_projects)
    small_ids: list[str] = []
    for root in small_roots:
        small = app.init_project(root)
        app.index_project(small.id)
        small_ids.append(small.id)
    counter.take()
    scope = f"multi{len(small_ids)}"

    def multi_search() -> None:
        lazy_tool_call(
            small_ids, lambda: app.search_code(SEARCH_QUERY, projects=small_ids, limit=8)
        )

    def multi_with_big() -> None:
        everything = [project.id, *small_ids]
        lazy_tool_call(
            everything, lambda: app.search_code(SEARCH_QUERY, projects=everything, limit=8)
        )

    profiler.measure(f"{scope}.burst.search_code", multi_search, iterations=iterations)
    profiler.measure(
        f"{scope}.gapped.search_code", multi_search, iterations=gap_iterations, gap=gap
    )
    profiler.measure(f"{scope}.with_repo.burst.search_code", multi_with_big, iterations=iterations)
    touched = small_roots[0] / "pkg" / "profile_touch.py"
    touched.write_text("touched = True\n")
    try:
        profiler.measure(f"{scope}.dirty.first.search_code", multi_search, iterations=1, gap=gap)
        noticed[f"{scope}.dirty.first.search_code"] = (
            stored_hash(small_ids[0], "pkg/profile_touch.py") is not None
        )
        profiler.measure(f"{scope}.dirty.burst.search_code", multi_search, iterations=iterations)
        profiler.measure(
            f"{scope}.dirty.gapped.search_code",
            multi_search,
            iterations=gap_iterations,
            gap=gap,
        )
    finally:
        touched.unlink()
    profiler.measure(f"{scope}.restored.first.search_code", multi_search, iterations=1, gap=gap)
    for name, value in noticed.items():
        profiler.scenarios[name]["noticed"] = value
        print(f"  {name:<32} noticed={value}", file=sys.stderr)
    result["gap_seconds"] = gap
    result["gap_iterations"] = gap_iterations

    result["scenarios"] = profiler.scenarios
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
