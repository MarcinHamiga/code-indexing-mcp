"""Regression guard for the query-path-overhead remediation track.

Indexes a real Git repository with many files, dirties exactly one of them,
and asserts that neither a status check nor the search that follows it
performs a whole-tree scan, more than one Git probe per project, or a
LanceDB write -- the three fixed per-query costs
docs/plans/2026-09-02-review-remediation-1-query-path-plan.md set out to
remove: the dirty-worktree full walk (Steps 2-3), the extra git spawns per
call (Steps 4-5), and the registry write on every read (Step 6).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import run_git

from code_indexing_mcp import application as application_module
from code_indexing_mcp.application import Application, RuntimePaths

FILE_COUNT = 200


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


def _large_git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    for index in range(FILE_COUNT):
        (root / f"module_{index:04d}.py").write_text(f"def f_{index}():\n    return {index}\n")
    run_git("add", "-A", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "initial", cwd=root)
    return root


def test_a_dirty_worktree_query_costs_no_full_scan_extra_probes_or_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole-track regression guard.

    A 200-file repository is indexed once, then one file is dirtied -- the
    scenario the track's findings describe: before this track, a status
    check on a dirty checkout walked every one of the 200 files, and every
    query re-probed Git from scratch. After it, the status check stats only
    the one dirty path, the post-operation check reads HEAD off disk instead
    of spawning Git again, and neither call writes to the slot registry.
    """
    root = _large_git_repo(tmp_path)
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=root,
    )
    project = app.init_project(root)
    app.index_project(project.id)
    monkeypatch.setattr("code_indexing_mcp.application.FRESHNESS_CACHE_SECONDS", 0.0)
    assert app.project_status(project.id).state == "ready"

    # Dirty exactly one of the 200 files.
    (root / "module_0000.py").write_text("def f_0():\n    return 999\n")

    iter_scan_calls: list[int] = []
    original_iter_scan = app.indexer.scanner.iter_scan

    def counted_iter_scan(*args, **kwargs):
        iter_scan_calls.append(1)
        return original_iter_scan(*args, **kwargs)

    monkeypatch.setattr(app.indexer.scanner, "iter_scan", counted_iter_scan)

    probe_calls: list[int] = []
    original_probe = application_module.probe_git_state

    def counted_probe(*args, **kwargs):
        probe_calls.append(1)
        return original_probe(*args, **kwargs)

    monkeypatch.setattr("code_indexing_mcp.application.probe_git_state", counted_probe)

    version_before = app.store._project_slots.version

    # The status check a lazy-mode tool call runs before every query.
    status = app.project_status(project.id)
    assert status.state == "stale"
    assert iter_scan_calls == []
    assert len(probe_calls) <= 1

    probe_calls.clear()

    # The query itself, run against the (still stale) index -- this track
    # does not reindex on query, only avoids re-walking and re-probing.
    app.search_code("def f_1", projects=[project.id])

    assert iter_scan_calls == []
    assert len(probe_calls) <= 1
    assert app.store._project_slots.version == version_before
