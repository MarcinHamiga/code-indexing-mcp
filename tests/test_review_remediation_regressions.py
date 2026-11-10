"""Regression tests for logic bugs and performance improvements from the indexed review."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import lancedb
import pytest
from conftest import run_git

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.changes import parse_unified_zero
from code_indexing_mcp.embedding import EmbeddedSegment, PassageCandidate, SegmentPlan
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.git_state import GitCommandResult, probe_git_state
from code_indexing_mcp.git_state import run_git as run_git_cmd
from code_indexing_mcp.installer.config_files import write_changed_configuration
from code_indexing_mcp.models import ChunkPreview, DeclarationSelector, RenameOperation
from code_indexing_mcp.search import _outline_items


class TinyEmbedder:
    def __init__(self, model_id: str = "test/remediation-v1") -> None:
        self.model_id = model_id
        self.dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.25, 0.5, float((len(text) % 17) + 1)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.25, 0.5, float((len(text) % 17) + 1)]


class WindowingEmbedder(TinyEmbedder):
    def plan_and_embed(
        self, candidates: Sequence[PassageCandidate], plan: SegmentPlan
    ) -> list[list[EmbeddedSegment]]:
        vec_bytes = struct.pack("<4f", 1.0, 0.25, 0.5, 2.0)
        segments: list[list[EmbeddedSegment]] = []
        for candidate in candidates:
            length = len(candidate.content)
            if length > 80:
                midpoint = length // 2
                segments.append(
                    [
                        EmbeddedSegment(0, min(length, midpoint + 20), 20, vec_bytes),
                        EmbeddedSegment(max(0, midpoint - 20), length, 20, vec_bytes),
                    ]
                )
            else:
                segments.append([EmbeddedSegment(0, length, 10, vec_bytes)])
        return segments


def _commit_all(repo: Path, message: str) -> None:
    run_git("add", ".", cwd=repo)
    run_git(
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Tests",
        "commit",
        "-qm",
        message,
        cwd=repo,
    )


def _init_git_repo(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(repo))
    return repo


def test_linked_worktree_chunk_lookup_and_scan_config_preservation(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "mod.py").write_text(
        "def shared_fn() -> str:\n    return 'from_main_branch'\n\n"
        "def caller() -> str:\n    return shared_fn()\n",
        encoding="utf-8",
    )
    _commit_all(repo, "init")

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    marker_file = repo / ".ci-mcp" / "project.toml"
    marker_file.write_text(
        marker_file.read_text(encoding="utf-8").replace("exclude = []", 'exclude = ["vendor/**"]'),
        encoding="utf-8",
    )
    app.init_project(repo)

    wt = tmp_path / "wt"
    run_git("worktree", "add", "-q", "-b", "feature", str(wt), cwd=repo)
    assert app.init_project(wt).id == project.id
    (wt / "mod.py").write_text(
        "def shared_fn() -> str:\n    return 'from_feature_worktree_unique'\n\n"
        "def caller() -> str:\n    return shared_fn()\n",
        encoding="utf-8",
    )
    (wt / "vendor").mkdir()
    (wt / "vendor" / "ignored.py").write_text("def vendor_only():\n    return 0\n")
    _commit_all(wt, "feature")

    app.index_project(project.id, roots=[repo])
    app.index_project(project.id, roots=[wt])

    # Indexing the linked worktree must keep canonical scan.exclude and enforce it.
    assert app.list_projects()[0].scan.exclude == ["vendor/**"]
    assert not app.find_symbol("vendor_only", project=str(wt)).hits

    # Merged multi-root search hits from the secondary worktree resolve in get_chunk
    # and reference/refactor tools, both with roots=[repo, wt] and via active-slot lookup.
    merged = app.search_code("from_feature_worktree_unique", roots=[repo, wt])
    wt_hit = next(h for h in merged.hits if "from_feature_worktree_unique" in h.snippet)
    chunk = app.get_chunk(wt_hit.chunk_id, roots=[repo, wt])
    assert "from_feature_worktree_unique" in chunk.content
    assert "from_feature_worktree_unique" in app.get_chunk(wt_hit.chunk_id, roots=[repo]).content

    refs = app.find_references(DeclarationSelector(chunk_id=wt_hit.chunk_id), roots=[repo, wt])
    assert len(refs.hits) == 1
    assert refs.hits[0].path == "mod.py"

    # Single-project tools reject multi-checkout roots with deduplicated project id + checkouts.
    with pytest.raises(CodeIndexingError) as exc_info:
        app.project_status(roots=[repo, wt])
    assert exc_info.value.code is ErrorCode.AMBIGUOUS_PROJECT
    assert exc_info.value.details["projects"] == [project.id]
    assert len(exc_info.value.details["checkouts"]) == 2

    # Switching the linked worktree away from `feature` retires that slot's chunk_id.
    run_git("checkout", "-q", "-b", "other_branch", cwd=wt)
    app._recent_git_probes.clear()
    with pytest.raises(CodeIndexingError) as missing_exc:
        app.get_chunk(wt_hit.chunk_id, roots=[repo, wt])
    assert missing_exc.value.code is ErrorCode.CHUNK_NOT_FOUND


def test_incompatible_branch_slot_reports_rebuild_required_across_branches(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "a.py").write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    _commit_all(repo, "main")

    paths = RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache")
    app_v1 = Application(paths, embedder=TinyEmbedder("test/model-v1"), cwd=repo)
    project = app_v1.init_project(repo)
    app_v1.index_project(project.id)

    run_git("checkout", "-q", "-b", "feature", cwd=repo)
    (repo / "a.py").write_text("def alpha() -> int:\n    return 2\n", encoding="utf-8")
    _commit_all(repo, "feature")
    app_v1.index_project(project.id)
    run_git("checkout", "-q", "main", cwd=repo)

    app_v2 = Application(paths, embedder=TinyEmbedder("test/model-v2"), cwd=repo)
    status_before = app_v2.project_status(project.id)
    assert status_before.state == "rebuild_required"
    assert status_before.branch_build_pending is True
    assert app_v2.project_is_stale(project.id) is True

    app_v2.discover_project(repo)
    app_v2.index_project(project.id)
    assert app_v2.project_status(project.id).state == "ready"

    run_git("checkout", "-q", "feature", cwd=repo)
    app_v2._recent_git_probes.clear()
    app_v2.discover_project(repo)
    feat_status = app_v2.project_status(project.id)
    assert feat_status.state == "rebuild_required"
    assert feat_status.branch_build_pending is True
    assert app_v2.project_is_stale(project.id) is True


def test_split_declaration_with_stripped_blank_line_merges_cleanly(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    long_literal = "x" * 4045
    code = (
        "def single_target_fn() -> int:\n"
        f"    payload = '{long_literal}'\n"
        "\n"
        "    return len(payload)\n\n"
        "def caller() -> int:\n"
        "    return single_target_fn()\n"
    )
    (repo / "mod.py").write_text(code, encoding="utf-8")
    _commit_all(repo, "init")

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)

    refs = app.find_references(
        DeclarationSelector(project=project.id, path="mod.py", qualified_symbol="single_target_fn")
    )
    assert len(refs.hits) == 1

    (repo / "mod.py").write_text(
        code.replace("return len(payload)", "return len(payload) + 1"), encoding="utf-8"
    )
    changed = app.changed_symbols(project.id)
    assert [(s.qualified_symbol, s.start_line, s.end_line) for s in changed.files[0].symbols] == [
        ("single_target_fn", 1, 4)
    ]

    # Two distinct split overloads restarting at part_index 0 still stay separate.
    overload_previews = [
        ChunkPreview(
            chunk_id="p:1",
            project_id="p",
            path="over.ts",
            language="typescript",
            kind="function_part",
            symbol="fn",
            qualified_symbol="fn",
            start_line=1,
            end_line=10,
            part_index=0,
        ),
        ChunkPreview(
            chunk_id="p:2",
            project_id="p",
            path="over.ts",
            language="typescript",
            kind="function_part",
            symbol="fn",
            qualified_symbol="fn",
            start_line=12,
            end_line=20,
            part_index=1,
        ),
        ChunkPreview(
            chunk_id="p:3",
            project_id="p",
            path="over.ts",
            language="typescript",
            kind="function_part",
            symbol="fn",
            qualified_symbol="fn",
            start_line=21,
            end_line=30,
            part_index=0,
        ),
    ]
    merged_items = _outline_items(overload_previews, "over.ts", span_parts=True)
    assert [(i.start_line, i.end_line) for i in merged_items] == [(1, 20), (21, 30)]


def test_windowed_chunks_coalesce_in_changed_symbols_and_reuse_db_on_get_chunk(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    body = "\n".join(f"    step_{idx} = {idx} * 2" for idx in range(10))
    code = f"def multi_window_fn() -> int:\n{body}\n    return step_9\n"
    (repo / "mod.py").write_text(code, encoding="utf-8")
    _commit_all(repo, "init")

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=WindowingEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)

    (repo / "mod.py").write_text(code.replace("step_5 = 5 * 2", "step_5 = 5 * 3"), encoding="utf-8")
    changed = app.changed_symbols(project.id)
    assert [(s.qualified_symbol, s.start_line, s.end_line) for s in changed.files[0].symbols] == [
        ("multi_window_fn", 1, 12)
    ]

    # Windowed get_chunk reuses the cached partition database instead of reconnecting.
    hit = app.find_symbol("multi_window_fn", project=project.id).hits[0]
    with patch.object(lancedb, "connect", wraps=lancedb.connect) as spy_connect:
        chunk = app.get_chunk(hit.chunk_id)
    assert spy_connect.call_count == 0
    assert "return step_9" in chunk.content


def test_reference_backfill_preserves_target_slot_partial_state_across_branches(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "good.py").write_text("def ok() -> int:\n    return 1\n", encoding="utf-8")
    _commit_all(repo, "main")

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)

    run_git("checkout", "-q", "-b", "partial_branch", cwd=repo)
    (repo / "broken.py").write_text("def broken() -> int:\n    return 2\n", encoding="utf-8")
    _commit_all(repo, "partial")

    orig_extract = app.indexer.extractor.extract

    def failing_extract(path: Path, language: str, source: bytes):
        if path.name == "broken.py":
            raise RuntimeError("simulated parse crash")
        return orig_extract(path, language, source)

    with patch.object(app.indexer.extractor, "extract", side_effect=failing_extract):
        app.index_project(project.id)

    partial_target = app._resolve_active_target(project)
    tables = app.store._project_tables(project.id, partition_id=partial_target.partition_id)
    assert tables.references is not None
    tables.references.delete("path = 'good.py'")

    run_git("checkout", "-q", "main", cwd=repo)
    app._recent_git_probes.clear()
    app.index_project(project.id)
    assert app.store.project_state(project.id) == "ready"

    run_git("checkout", "-q", "partial_branch", cwd=repo)
    app._recent_git_probes.clear()
    app.find_references(
        DeclarationSelector(project=project.id, path="good.py", qualified_symbol="ok")
    )
    slot_after = app.store.get_slot(partial_target.slot.slot_id)
    assert slot_after is not None and slot_after.state == "partial"


def test_dead_code_report_batches_reference_table_in_constant_queries(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "defs.py").write_text(
        "\n".join(
            [
                "def used_direct() -> int:\n    return 1\n",
                "def used_aliased() -> int:\n    return 2\n",
                "def unused_fn() -> int:\n    return 3\n",
                *(f"def extra_unused_{idx}() -> int:\n    return {idx}\n" for idx in range(15)),
            ]
        ),
        encoding="utf-8",
    )
    (repo / "use.py").write_text(
        "from defs import used_aliased as alias_fn, used_direct\n\n"
        "def entry() -> int:\n    return used_direct() + alias_fn()\n",
        encoding="utf-8",
    )
    _commit_all(repo, "init")

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)

    calls = 0
    real_list_refs = app.store.list_reference_records

    def counting_list_refs(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_list_refs(*args, **kwargs)  # type: ignore[arg-type]

    with patch.object(app.store, "list_reference_records", side_effect=counting_list_refs):
        report = app.dead_code_report(project.id)

    assert calls == 4
    unused_names = {item.declaration.symbol for item in report.review}
    assert "unused_fn" in unused_names
    assert "used_direct" not in unused_names
    assert "used_aliased" not in unused_names


def test_btree_indexes_and_explain_hits_reuses_known_vector_distances(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "mod.py").write_text(
        "def alpha_search_target() -> int:\n    return 42\n", encoding="utf-8"
    )
    _commit_all(repo, "init")

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)

    target = app._resolve_active_target(project)
    tables = app.store._project_tables(project.id, partition_id=target.partition_id)
    chunk_indexed = {col for idx in tables.chunks.list_indices() for col in idx.columns}
    assert "chunk_id" in chunk_indexed
    assert tables.references is not None
    ref_indexed = {col for idx in tables.references.list_indices() for col in idx.columns}
    assert "reference_id" in ref_indexed

    response = app.search_code("alpha_search_target", projects=[project.id])
    assert response.hits and response.hits[0].explanation is not None
    assert response.hits[0].explanation.vector_score is not None


def test_language_reclassification_converges_and_minor_edge_cases(tmp_path: Path) -> None:
    # Finding #8: language mismatch converges to ready on reindex.
    nongit = tmp_path / "nongit"
    nongit.mkdir()
    (nongit / "mod.py").write_text("x = 1\n", encoding="utf-8")
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=nongit,
    )
    project = app.init_project(nongit)
    app.index_project(project.id)
    target = app._resolve_active_target(project)
    tables = app.store._project_tables(project.id, partition_id=target.partition_id)
    tables.files.update(where="path = 'mod.py'", values={"language": "javascript"})
    app.invalidate_freshness(project.id)
    assert app.project_status(project.id).state == "stale"

    report = app.index_project(project.id)
    assert report.indexed_files == 1
    app.invalidate_freshness(project.id)
    assert app.project_status(project.id).state == "ready"

    # Finding #9: unknown or mangled history cursor raises INVALID_CURSOR.
    with pytest.raises(CodeIndexingError) as cursor_exc:
        app.index_history(project.id, cursor="9999-12-31T23:59:59Z|missing_run_id")
    assert cursor_exc.value.code is ErrorCode.INVALID_CURSOR

    # Finding #10: parse_unified_zero ignores form-feeds / CR inside diff payload lines.
    patch_text = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,0 @@\n"
        "-old\f@@ -1,0 +99,5 @@\rdiff --git a/injected.py b/injected.py\n"
    )
    parsed = parse_unified_zero(patch_text)
    assert list(parsed) == ["foo.py"]
    assert parsed["foo.py"].ranges == []

    # Finding #11: write_changed_configuration preserves dotfile symlinks.
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real_cfg = dotfiles / "mcp.json"
    real_cfg.write_text('{"v": 1}\n', encoding="utf-8")
    symlink_cfg = tmp_path / "client_mcp.json"
    symlink_cfg.symlink_to(real_cfg)
    assert write_changed_configuration(symlink_cfg, '{"v": 1}\n', '{"v": 2}\n') is True
    assert symlink_cfg.is_symlink()
    assert real_cfg.read_text(encoding="utf-8") == '{"v": 2}\n'
    assert (tmp_path / "client_mcp.json.bak").read_text(encoding="utf-8") == '{"v": 1}\n'


def test_cached_git_status_probe_spawns_only_git_status_and_ts_rename_scans_once(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path)
    (repo / "lib.ts").write_text(
        "export function greet(name: string): string {\n    return name;\n}\n",
        encoding="utf-8",
    )
    (repo / "use.ts").write_text(
        "import { greet } from './lib';\n"
        "import { greet as g2 } from './lib';\n"
        "export { greet } from './lib';\n",
        encoding="utf-8",
    )
    _commit_all(repo, "init")

    cached = probe_git_state(repo, include_status=False)
    spawned: list[list[str]] = []

    def counting_runner(cmd: list[str], cwd: Path) -> GitCommandResult:
        spawned.append(cmd)
        return run_git_cmd(cmd, cwd)

    probed = probe_git_state(repo, include_status=True, runner=counting_runner, cached=cached)
    assert len(spawned) == 1
    assert spawned[0][:2] == ["git", "status"]
    assert probed.head_oid == cached.head_oid

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)
    analysis = app.analyze_refactor(
        DeclarationSelector(project=project.id, path="lib.ts", qualified_symbol="greet"),
        RenameOperation(new_name="welcome"),
    )
    assert analysis.counts.must_change >= 2
