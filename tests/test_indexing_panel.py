"""Tests for real-time indexing progress panel, CLI runner, and process visibility."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from support import DeterministicEmbedder
from test_indexing import RecordingEmbedder, make_indexer
from test_tui import _make_app, _wait_for_workers
from test_tui_service import FakeApplication, _sample_project
from textual.widgets import Input, ProgressBar, Static

from code_indexing_mcp import cli
from code_indexing_mcp.application import Application, ApplicationLike
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import (
    HistoryPage,
    IndexIssue,
    IndexProgress,
    IndexReport,
    ModelStatus,
    RunAudit,
)
from code_indexing_mcp.progress import ProgressPublisher, read_all_progress
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.tui.app import CodeIndexingApp
from code_indexing_mcp.tui.indexing_panel import (
    IndexingActivityPanel,
    IndexRunApp,
    format_bytes,
    format_compact_progress_strip,
    format_duration_ms,
)
from code_indexing_mcp.tui.service import TuiService


def test_formatting_helpers_and_compact_strip() -> None:
    assert format_bytes(None) == "—"
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.0 KiB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MiB"

    assert format_duration_ms(None) == "—"
    assert format_duration_ms(420) == "420ms"
    assert format_duration_ms(3450) == "3.5s"
    assert format_duration_ms(125_000) == "2m 05.0s"

    determinate = IndexProgress(
        project_id="proj-1",
        phase="embedding",
        trigger="watcher",
        candidates_seen=50,
        candidates_total=100,
        parsed_files=12,
        chunks_extracted=40,
        chunks_embedded=20,
        reused_candidates=8,
        embedding_backend="cuda",
    )
    strip = format_compact_progress_strip(
        determinate, prefix="Indexing repo", elapsed_seconds=4.2, tick=2
    )
    assert "50%" in strip
    assert "20/40 chunks" in strip
    assert "8 cached" in strip
    assert "cuda" in strip
    assert "watcher" in strip
    assert "[a] Activity" in strip


def test_live_progress_carries_backend_timings_and_errors(tmp_path: Path) -> None:
    class FailingOnBadFileEmbedder(RecordingEmbedder):
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            if any("explode_here" in text for text in texts):
                raise ValueError("embedding exploded on bad file")
            return super().embed_passages(texts)

    root = tmp_path / "repo"
    root.mkdir()
    (root / "good.py").write_text("def ok():\n    return 1\n")
    (root / "bad.py").write_text("def explode_here():\n    return 2\n")
    project = initialize_project(root)
    indexer, _ = make_indexer(tmp_path, FailingOnBadFileEmbedder())
    seen: list[IndexProgress] = []

    report = indexer.index(project, on_progress=lambda snap: seen.append(snap.model_copy()))

    assert report.failed_files == 1
    assert seen
    final = seen[-1]
    assert final.embedding_backend == "cpu"
    assert "scan" in final.phase_durations_ms
    assert "parse" in final.phase_durations_ms
    assert final.parsed_files == 2
    assert final.chunks_extracted >= 2
    assert len(final.recent_errors) == 1
    assert final.recent_errors[0].path == "bad.py"


def test_read_all_progress_returns_fresh_snapshots_newest_first(tmp_path: Path) -> None:
    progress_dir = tmp_path / "progress"
    older = ProgressPublisher("proj-old", run_id="run-old", directory=progress_dir)
    older.state.started_at = 100.0
    older.update(phase="scanning", candidates_seen=3, force=True)

    newer = ProgressPublisher("proj-new", run_id="run-new", directory=progress_dir)
    newer.state.started_at = 200.0
    newer.update(phase="embedding", candidates_seen=10, force=True)

    stale_path = progress_dir / "proj-stale.json"
    stale = IndexProgress(
        project_id="proj-stale",
        started_at=50.0,
        updated_at=time.time() - 3600.0,
    )
    stale_path.write_text(stale.model_dump_json())

    active = read_all_progress(progress_dir)
    assert [item.project_id for item in active] == ["proj-new", "proj-old"]


@pytest.mark.asyncio
async def test_index_run_app_streams_live_progress_and_holds_when_requested() -> None:
    allow_finish = threading.Event()
    progress_sent = threading.Event()

    def fake_run(on_progress: Callable[[IndexProgress], None]) -> IndexReport:
        on_progress(
            IndexProgress(
                project_id="proj-1",
                run_id="12345678abcdef",
                trigger="manual",
                phase="embedding",
                candidates_seen=25,
                candidates_total=50,
                eligible_files=20,
                changed_files=10,
                parsed_files=10,
                skipped_total=5,
                skipped_by_reason={"ignored": 4, "unsupported": 1},
                bytes_read=65_536,
                chunks_extracted=40,
                chunks_embedded=20,
                chunks_staged=20,
                staged_bytes=16_384,
                reused_candidates=6,
                reused_segments=8,
                embedding_backend="cuda",
                current_path="src/pkg/engine.py",
                started_at=time.time() - 2.0,
                updated_at=time.time(),
                phase_durations_ms={"scan": 120, "parse": 340, "embed": 900, "commit": 10},
                recent_errors=[IndexIssue(path="src/broken.py", message="syntax issue")],
            )
        )
        progress_sent.set()
        allow_finish.wait(timeout=5)
        return IndexReport(
            project_id="proj-1",
            run_id="12345678abcdef",
            discovered_files=20,
            indexed_files=10,
            parsed_files=10,
            unchanged_files=10,
            embedded_chunks=40,
            chunks_extracted=40,
            chunks_staged=40,
            staged_bytes=32_768,
            bytes_read=65_536,
            skipped_files=5,
            skip_reasons={"ignored": 4, "unsupported": 1},
            reused_candidates=6,
            reused_segments=8,
            embedding_backend="cuda",
            duration_ms=1500,
            scan_ms=120,
            parse_ms=340,
            embed_ms=950,
            commit_ms=90,
        )

    runner = IndexRunApp(fake_run, project_label="alpha-service", hold_open=True)
    async with runner.run_test(size=(120, 32)) as pilot:
        assert await asyncio.to_thread(progress_sent.wait, 2)
        await pilot.pause(0.1)

        banner = str(runner.query_one("#activity-banner", Static).render())
        assert "LIVE" in banner
        assert "Embedding code passages" in banner
        assert "cuda" in banner

        bar = runner.query_one("#activity-progress-bar", ProgressBar)
        assert bar.percentage == pytest.approx(0.5)

        current = str(runner.query_one("#activity-current-file", Static).render())
        assert "src/pkg/engine.py" in current

        chunks_card = str(runner.query_one("#activity-chunks-card", Static).render())
        assert "6 reused" in chunks_card

        diagnostics = str(runner.query_one("#activity-diagnostics", Static).render())
        assert "ignored: 4" in diagnostics
        assert "src/broken.py" in diagnostics

        allow_finish.set()
        await runner.workers.wait_for_complete()
        await pilot.pause()

        assert runner.report is not None
        assert runner.return_code is None  # Held open until dismissed
        done_banner = str(runner.query_one("#activity-banner", Static).render())
        assert "COMPLETED" in done_banner
        await pilot.press("q")
        await pilot.pause()

    assert runner.return_code == 0


@pytest.mark.asyncio
async def test_index_run_app_auto_exits_and_surfaces_failures() -> None:
    def failing_run(_on_progress: Callable[[IndexProgress], None]) -> IndexReport:
        raise CodeIndexingError(ErrorCode.INDEX_BUSY, "Another indexing job is active")

    runner = IndexRunApp(failing_run, project_label="busy-repo", hold_open=False)
    async with runner.run_test(size=(80, 24)) as pilot:
        await runner.workers.wait_for_complete()
        await pilot.pause()

    assert runner.return_code == 2
    assert isinstance(runner.error, CodeIndexingError)


@pytest.mark.asyncio
async def test_tui_activity_view_shows_process_trace_history_and_force_reindex() -> None:
    fake = FakeApplication()

    def fake_history(
        project: str | None = None,
        *,
        roots: list[Path] | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> HistoryPage:
        return HistoryPage(
            project=_sample_project(),
            runs=[
                RunAudit(
                    run_id="deadbeef12345678",
                    project_id="proj-1",
                    trigger="watcher",
                    state="completed",
                    started_at="2026-11-11T13:55:00+00:00",
                    finished_at="2026-11-11T13:55:02+00:00",
                    phase_durations={"scan": 100, "parse": 250, "embed": 1400, "commit": 120},
                    eligible_files=42,
                    changed_files=4,
                    unchanged_files=38,
                    parsed_files=4,
                    chunks_extracted=18,
                    chunks_embedded=18,
                    chunks_staged=18,
                    reused_candidates=12,
                    embedding_backend="mlx",
                )
            ],
        )

    def fake_model_status() -> ModelStatus:
        return ModelStatus(
            embedding_model="BAAI/bge-base-en-v1.5",
            dimension=768,
            requested_accelerator="auto",
            resolved_accelerator="mlx",
            device="gpu",
            execution_provider="MLX",
            available_providers=["MLX", "CPUExecutionProvider"],
            stability="stable",
            precision="fp16",
            runtime_version="0.32.0",
            batch_size=16,
            batch_calibration="measured",
            probe_cache_state="hit",
            strict=False,
        )

    fake.index_history = fake_history  # type: ignore[attr-defined]
    fake.model_status = fake_model_status  # type: ignore[attr-defined]
    fake.ping = lambda: {"pid": 42424}  # type: ignore[attr-defined]

    app = CodeIndexingApp(service=TuiService(cast(ApplicationLike, fake)))
    async with app.run_test(size=(120, 32)) as pilot:
        await _wait_for_workers(app, pilot)
        await pilot.pause()

        # Typing 'a' in the search box types into the input rather than switching views
        assert isinstance(app.focused, Input)
        await pilot.press("a")
        assert app.query_one("#query-input", Input).value == "a"
        assert not app.query_one("#activity-pane").display

        # Move focus out of input and open Activity with 'a'
        await pilot.press("escape")
        await pilot.press("a")
        await _wait_for_workers(app, pilot)
        await pilot.pause()

        panel = app.query_one("#activity-pane", IndexingActivityPanel)
        assert panel.display
        assert not app.query_one("#results-pane").display

        diagnostics = str(app.query_one("#activity-diagnostics", Static).render())
        assert "Shared daemon (PID 42424)" in diagnostics
        assert "mlx" in diagnostics
        assert "deadbeef" in diagnostics
        assert "watcher" in diagnostics

        # Trigger Force Reindex from the Activity panel button
        await pilot.click("#activity-force-btn")
        await _wait_for_workers(app, pilot)
        await pilot.pause()
        assert fake.index_calls[-1]["force"] is True

        # Escape returns cleanly to search view
        await pilot.press("escape")
        await pilot.pause()
        assert not panel.display
        assert app.query_one("#results-pane").display


@pytest.mark.asyncio
async def test_tui_detects_external_background_index_run() -> None:
    fake = FakeApplication()
    live_snap: IndexProgress | None = IndexProgress(
        project_id="proj-1",
        run_id="bg9876543210",
        trigger="watcher",
        phase="embedding",
        candidates_seen=30,
        candidates_total=60,
        chunks_extracted=50,
        chunks_embedded=25,
        embedding_backend="cuda",
        pid=99999,
        started_at=time.time() - 3.0,
        updated_at=time.time(),
    )
    fake.index_progress = lambda _pid: live_snap  # type: ignore[method-assign]

    app = _make_app(fake)
    async with app.run_test(size=(80, 24)) as pilot:
        await _wait_for_workers(app, pilot)
        await pilot.pause()

        # Advance poll until the idle external check picks up the live snapshot
        for _ in range(5):
            app._poll_activity()
            await _wait_for_workers(app, pilot)
            await pilot.pause()

        bar = app.query_one("#progress-bar", Static)
        assert bar.display
        rendered = str(bar.render())
        assert "Background index · test-repo" in rendered
        assert "50%" in rendered
        assert "watcher" in rendered

        # Compact 80x24 pane switcher exposes the Activity button
        await pilot.click("#activity-view")
        await _wait_for_workers(app, pilot)
        await pilot.pause()
        assert app.query_one("#activity-pane").display
        banner = str(app.query_one("#activity-banner", Static).render())
        assert "PID 99999 (external/daemon)" in banner

        # Finish the external run and verify strip clears automatically
        live_snap = None
        app._poll_activity()
        await _wait_for_workers(app, pilot)
        await pilot.pause()
        assert not bar.display


def test_cli_index_no_tui_flag_preserves_json_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        cli,
        "Application",
        lambda paths, cwd: Application(paths, embedder=DeterministicEmbedder(), cwd=cwd),
    )

    assert cli.main(["init", str(root)]) == 0
    capsys.readouterr()
    assert cli.main(["index", str(root), "--no-tui"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed_files"] == 1


def test_cli_index_uses_tui_runner_when_interactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n")
    monkeypatch.setenv("CODE_INDEXING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CODE_INDEXING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        cli,
        "Application",
        lambda paths, cwd: Application(paths, embedder=DeterministicEmbedder(), cwd=cwd),
    )
    assert cli.main(["init", str(root)]) == 0
    capsys.readouterr()

    monkeypatch.setattr(cli, "_use_index_tui", lambda *, no_tui: not no_tui)
    ran_tui = False

    def fake_run(self: IndexRunApp) -> int:
        nonlocal ran_tui
        ran_tui = True
        self.report = self._run_index(lambda _p: None)
        return 0

    monkeypatch.setattr(IndexRunApp, "run", fake_run)
    assert cli.main(["index", str(root)]) == 0
    assert ran_tui
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed_files"] == 1
