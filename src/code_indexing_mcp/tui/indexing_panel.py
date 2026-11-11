"""Real-time Textual indexing progress panel and standalone CLI runner."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import Any, ClassVar

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, Footer, Label, ProgressBar, Static
from textual.worker import Worker, WorkerState

from ..models import IndexIssue, IndexProgress, IndexReport, ProjectInfo, RunAudit
from .service import ProcessSnapshot

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_PHASE_LABELS: dict[str, str] = {
    "scanning": "Scanning & parsing repository",
    "embedding": "Embedding code passages",
    "extracting_references": "Extracting structural references",
    "committing": "Committing staged tables",
}


def format_bytes(num_bytes: int | float | None) -> str:
    """Format a byte count using binary IEC units."""
    if num_bytes is None or num_bytes < 0:
        return "—"
    value = float(num_bytes)
    if value < 1024:
        return f"{int(value)} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        value /= 1024.0
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TiB"


def format_duration_ms(duration_ms: int | float | None) -> str:
    """Format milliseconds into a concise human-readable duration."""
    if duration_ms is None or duration_ms < 0:
        return "—"
    ms = float(duration_ms)
    if ms < 1000:
        return f"{round(ms)}ms"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m {rem:04.1f}s"


def format_compact_progress_strip(
    progress: IndexProgress | None,
    *,
    prefix: str,
    elapsed_seconds: float,
    tick: int = 0,
    include_shortcut: bool = True,
) -> str:
    """Build a single-line real-time progress summary for status strips."""
    spinner = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
    hint = " · [a] Activity" if include_shortcut else ""
    if progress is None:
        return f"{spinner} {prefix} · {elapsed_seconds:.1f}s{hint}"

    fraction = progress.fraction
    if fraction is not None:
        filled = round(fraction * 10)
        bar = "█" * filled + "░" * (10 - filled)
        meter = f"[{bar}] {fraction * 100:3.0f}%"
    else:
        meter = spinner

    phase_name = progress.phase.replace("_", " ")
    counts: list[str] = [f"{progress.candidates_seen:,} seen"]
    if progress.changed_files or progress.parsed_files:
        active_files = max(progress.changed_files, progress.parsed_files)
        counts.append(f"{active_files:,} parsed")
    if progress.chunks_embedded or progress.chunks_extracted:
        if progress.chunks_extracted:
            counts.append(f"{progress.chunks_embedded:,}/{progress.chunks_extracted:,} chunks")
        else:
            counts.append(f"{progress.chunks_embedded:,} chunks")
    if progress.reused_candidates:
        counts.append(f"{progress.reused_candidates:,} cached")
    if progress.embedding_backend:
        counts.append(progress.embedding_backend)
    if progress.trigger != "manual":
        counts.append(progress.trigger)

    summary = " · ".join(counts)
    return f"{meter} {prefix} · {phase_name} · {elapsed_seconds:.1f}s · {summary}{hint}"


def _truncate_left(path: str, max_chars: int = 72) -> str:
    if len(path) <= max_chars:
        return path
    return "…" + path[-(max_chars - 1) :]


def _short_timestamp(iso_ts: str) -> str:
    if not iso_ts:
        return "—"
    # Convert '2026-11-11T14:00:25.123456+00:00' -> '11-11 14:00:25'
    cleaned = iso_ts.replace("T", " ")
    if len(cleaned) >= 19:
        return cleaned[5:19]
    return cleaned


class IndexingActivityPanel(Vertical):
    """Polished real-time indexing progress and process traceability panel."""

    class IndexRequested(Message):
        """Emitted when the user triggers an index or force-reindex from the panel."""

        def __init__(self, *, force: bool = False) -> None:
            super().__init__()
            self.force = force

    class RefreshRequested(Message):
        """Emitted when the user requests a fresh process/history snapshot."""

    class CloseRequested(Message):
        """Emitted when the user closes the activity view to return to search."""

    def __init__(
        self,
        *,
        show_actions: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.show_actions = show_actions
        self._project: ProjectInfo | None = None
        self._project_label: str = ""
        self._live_progress: IndexProgress | None = None
        self._starting_phase: str | None = None
        self._run_started_monotonic: float | None = None
        self._last_report: IndexReport | None = None
        self._last_error: str | None = None
        self._process_snapshot: ProcessSnapshot | None = None
        self._tick: int = 0

    def compose(self) -> ComposeResult:
        with Horizontal(id="activity-header-row"):
            yield Label("Indexing & Process Trace", id="activity-title", markup=False)
            if self.show_actions:
                with Horizontal(id="activity-actions"):
                    yield Button("Index F5", id="activity-index-btn")
                    yield Button("Force Reindex", id="activity-force-btn")
                    yield Button("Refresh", id="activity-refresh-btn")
                    yield Button("Back Esc", id="activity-close-btn")

        yield Static("", id="activity-banner")
        with Vertical(id="activity-pipeline-card"):
            yield Static("", id="activity-phases")
            with Horizontal(id="activity-bar-row"):
                yield ProgressBar(
                    total=100.0,
                    show_percentage=True,
                    show_eta=False,
                    id="activity-progress-bar",
                )
                yield Label("Idle", id="activity-bar-caption", markup=False)
            yield Static("", id="activity-current-file")

        with VerticalScroll(id="activity-scroll"):
            with Horizontal(id="activity-metrics-grid"):
                yield Static("", id="activity-files-card", classes="activity-card")
                yield Static("", id="activity-chunks-card", classes="activity-card")
            yield Static("", id="activity-diagnostics", classes="activity-card")

    def on_mount(self) -> None:
        self._render_all()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "activity-index-btn":
            event.stop()
            self.post_message(self.IndexRequested(force=False))
        elif button_id == "activity-force-btn":
            event.stop()
            self.post_message(self.IndexRequested(force=True))
        elif button_id == "activity-refresh-btn":
            event.stop()
            self.post_message(self.RefreshRequested())
        elif button_id == "activity-close-btn":
            event.stop()
            self.post_message(self.CloseRequested())

    def set_project(self, project: ProjectInfo | None, *, label: str = "") -> None:
        """Bind the panel to the active project, clearing run state on project switch."""
        changed = (project.id if project else None) != (self._project.id if self._project else None)
        self._project = project
        self._project_label = label or (project.name if project else "")
        if changed:
            self._live_progress = None
            self._starting_phase = None
            self._run_started_monotonic = None
            self._last_report = None
            self._last_error = None
        if self.is_mounted:
            self._render_all()

    def mark_starting(self, phase: str = "Starting index") -> None:
        """Show immediate startup feedback before the first progress snapshot lands."""
        self._starting_phase = phase
        self._live_progress = None
        self._last_error = None
        self._run_started_monotonic = time.monotonic()
        self._set_index_buttons_disabled(True)
        if self.is_mounted:
            self._render_all()

    def set_live_progress(self, progress: IndexProgress | None) -> None:
        """Update the panel from a live IndexProgress snapshot."""
        if progress is not None:
            self._live_progress = progress
            self._starting_phase = None
            self._last_error = None
            if self._run_started_monotonic is None:
                if progress.started_at > 0:
                    elapsed = max(0.0, time.time() - progress.started_at)
                    self._run_started_monotonic = time.monotonic() - elapsed
                else:
                    self._run_started_monotonic = time.monotonic()
            self._set_index_buttons_disabled(True)
        else:
            if self._live_progress is not None and self._starting_phase is None:
                self._run_started_monotonic = None
            self._live_progress = None
            if self._starting_phase is None:
                self._set_index_buttons_disabled(False)
        if self.is_mounted:
            self._render_all()

    def set_completed_report(self, report: IndexReport) -> None:
        """Transition from live progress to the completed IndexReport summary."""
        self._live_progress = None
        self._starting_phase = None
        self._run_started_monotonic = None
        self._last_report = report
        self._last_error = None
        self._set_index_buttons_disabled(False)
        if self.is_mounted:
            self._render_all()

    def set_failed(self, error_message: str) -> None:
        """Record an aborted or failed indexing run."""
        self._live_progress = None
        self._starting_phase = None
        self._run_started_monotonic = None
        self._last_error = error_message
        self._set_index_buttons_disabled(False)
        if self.is_mounted:
            self._render_all()

    def set_process_snapshot(self, snapshot: ProcessSnapshot) -> None:
        """Update daemon/process status, backend info, and durable run history."""
        self._process_snapshot = snapshot
        if self.is_mounted:
            self._render_all()

    def tick(self) -> None:
        """Advance live spinner and elapsed clock while an operation is active."""
        self._tick += 1
        if self.is_mounted and (self._live_progress is not None or self._starting_phase):
            self._render_all()

    def _set_index_buttons_disabled(self, disabled: bool) -> None:
        if not self.is_mounted or not self.show_actions:
            return
        for selector in ("#activity-index-btn", "#activity-force-btn"):
            for btn in self.query(selector):
                if isinstance(btn, Button):
                    btn.disabled = disabled

    def _elapsed_seconds(self) -> float:
        if self._live_progress is not None and self._live_progress.started_at > 0:
            return max(0.0, time.time() - self._live_progress.started_at)
        if self._run_started_monotonic is not None:
            return max(0.0, time.monotonic() - self._run_started_monotonic)
        return 0.0

    def _latest_history_run(self) -> RunAudit | None:
        if self._process_snapshot and self._process_snapshot.recent_runs:
            return self._process_snapshot.recent_runs[0]
        return None

    def _render_all(self) -> None:
        project_title = self._project_label or (
            self._project.name if self._project is not None else "No project selected"
        )
        self.query_one("#activity-title", Label).update(
            f"Indexing & Process Trace · {project_title}"
        )
        self._render_banner()
        self._render_pipeline()
        self._render_metrics()
        self._render_diagnostics()

    def _render_banner(self) -> None:
        text = Text()
        progress = self._live_progress
        report = self._last_report
        latest_audit = self._latest_history_run()

        if progress is not None or self._starting_phase is not None:
            spinner = _SPINNER_FRAMES[self._tick % len(_SPINNER_FRAMES)]
            elapsed = self._elapsed_seconds()
            headline = (
                _PHASE_LABELS.get(progress.phase, progress.phase.replace("_", " ").capitalize())
                if progress is not None
                else (self._starting_phase or "Starting")
            )
            text.append(f" {spinner} LIVE ", style="bold black on cyan")
            text.append(f"  {headline}", style="bold cyan")
            text.append(f"  ·  {elapsed:.1f}s elapsed\n", style="dim")
            if progress is not None:
                owner = (
                    f"PID {progress.pid} (this process)"
                    if progress.pid == os.getpid()
                    else f"PID {progress.pid} (external/daemon)"
                )
                meta: list[str] = []
                if progress.run_id:
                    meta.append(f"Run {progress.run_id[:8]} ({progress.trigger})")
                else:
                    meta.append(f"Trigger: {progress.trigger}")
                meta.append(owner)
                if progress.selector:
                    head = f" @ {progress.expected_head[:7]}" if progress.expected_head else ""
                    meta.append(f"Slot: {progress.selector}{head}")
                elif progress.slot_id:
                    meta.append(f"Slot: {progress.slot_id[:8]}")
                if progress.embedding_backend:
                    meta.append(f"Backend: {progress.embedding_backend}")
                if progress.peak_memory_bytes:
                    mem = format_bytes(progress.peak_memory_bytes)
                    if progress.memory_budget_bytes:
                        mem += f" / {format_bytes(progress.memory_budget_bytes)}"
                    meta.append(f"RSS: {mem}")
                text.append(" · ".join(meta), style="dim")
                if progress.embedding_fallback_reason:
                    text.append(
                        f"\n⚠ Backend fallback: {progress.embedding_fallback_reason}",
                        style="yellow",
                    )
            else:
                text.append("Acquiring project lock and initializing scanner…", style="dim")
        elif self._last_error:
            text.append(" ✗ FAILED ", style="bold white on red")
            text.append(f"  {self._last_error}", style="bold red")
        elif report is not None:
            if report.errors:
                text.append(" ⚠ PARTIAL ", style="bold black on yellow")
                text.append(
                    f"  Completed with {len(report.errors)} file error(s) in "
                    f"{format_duration_ms(report.duration_ms)}\n",
                    style="bold yellow",
                )
            else:
                text.append(" ✓ COMPLETED ", style="bold black on green")
                text.append(
                    f"  Finished in {format_duration_ms(report.duration_ms)}  ·  "
                    f"{report.indexed_files:,} indexed, {report.unchanged_files:,} unchanged, "
                    f"{report.embedded_chunks:,} chunks embedded\n",
                    style="bold green",
                )
            meta = []
            if report.run_id:
                meta.append(f"Run {report.run_id[:8]} ({report.trigger})")
            else:
                meta.append(f"Trigger: {report.trigger}")
            meta.append(
                f"Backend: {report.embedding_backend} (batch {report.embedding_batch_size})"
            )
            if report.peak_memory_bytes:
                mem = format_bytes(report.peak_memory_bytes)
                if report.memory_budget_bytes:
                    mem += f" / {format_bytes(report.memory_budget_bytes)}"
                meta.append(f"Peak RSS: {mem}")
            if report.embedding_cache_status:
                meta.append(f"Cache: {report.embedding_cache_status}")
            text.append(" · ".join(meta), style="dim")
            if report.embedding_fallback_reason:
                text.append(
                    f"\n⚠ Backend fallback: {report.embedding_fallback_reason}",
                    style="yellow",
                )
        elif latest_audit is not None:
            state_upper = latest_audit.state.upper()
            badge_style = (
                "bold black on green"
                if latest_audit.state == "completed" and not latest_audit.failed_files
                else "bold black on yellow"
                if latest_audit.state == "completed"
                else "bold white on red"
            )
            text.append(f" ○ IDLE (LAST: {state_upper}) ", style=badge_style)
            total_ms = sum(latest_audit.phase_durations.values())
            text.append(
                f"  Last run {latest_audit.run_id[:8]} ({latest_audit.trigger}) at "
                f"{_short_timestamp(latest_audit.started_at)} · "
                f"{format_duration_ms(total_ms) if total_ms else 'completed'}\n",
                style="bold",
            )
            text.append(
                f"Backend: {latest_audit.embedding_backend} · "
                f"PID {latest_audit.pid} · "
                f"{latest_audit.changed_files:,} changed, "
                f"{latest_audit.chunks_embedded:,} embedded, "
                f"{latest_audit.reused_candidates:,} cache hits",
                style="dim",
            )
        else:
            text.append(" ○ IDLE ", style="bold black on bright_black")
            text.append(
                "  No active indexing job. Press F5 to index or watch background runs here.",
                style="dim",
            )

        self.query_one("#activity-banner", Static).update(text)

    def _render_pipeline(self) -> None:
        progress = self._live_progress
        report = self._last_report
        latest_audit = self._latest_history_run()

        durations: dict[str, int] = {}
        active_stage: str | None = None
        completed_all = False

        if progress is not None:
            durations = dict(progress.phase_durations_ms)
            phase = progress.phase
            if phase == "scanning":
                active_stage = "scan_parse"
            elif phase == "embedding":
                active_stage = "embed"
            elif phase == "extracting_references":
                active_stage = "refs"
            elif phase == "committing":
                active_stage = "commit"
            else:
                active_stage = "scan_parse"
        elif self._starting_phase is not None:
            active_stage = "scan_parse"
        elif report is not None:
            completed_all = True
            durations = {
                "scan": report.scan_ms or report.scan_duration_ms or 0,
                "parse": report.parse_ms or report.parse_duration_ms or 0,
                "embed": report.embed_ms or report.embed_duration_ms or 0,
                "reference_extraction": report.reference_extraction_duration_ms or 0,
                "commit": report.commit_ms or report.commit_duration_ms or 0,
            }
        elif latest_audit is not None and latest_audit.state == "completed":
            completed_all = True
            durations = dict(latest_audit.phase_durations)

        stage_order = ("scan_parse", "embed", "refs", "commit")
        stage_defs = (
            (
                "scan_parse",
                "1. Scan & Parse",
                durations.get("scan", 0) + durations.get("parse", 0),
            ),
            ("embed", "2. Embed", durations.get("embed", 0)),
            ("refs", "3. References", durations.get("reference_extraction", 0)),
            ("commit", "4. Commit", durations.get("commit", 0)),
        )

        active_idx = stage_order.index(active_stage) if active_stage in stage_order else -1
        pipeline_text = Text()
        for idx, (key, title, ms) in enumerate(stage_defs):
            if idx > 0:
                pipeline_text.append("  ──▶  ", style="dim")
            timing = f" ({format_duration_ms(ms)})" if ms > 0 else ""
            if completed_all or (active_idx >= 0 and idx < active_idx):
                pipeline_text.append(f"✓ {title}{timing}", style="green")
            elif key == active_stage:
                pipeline_text.append(f"▶ {title}{timing}", style="bold cyan")
            else:
                pipeline_text.append(f"· {title}{timing}", style="dim")

        self.query_one("#activity-phases", Static).update(pipeline_text)

        bar = self.query_one("#activity-progress-bar", ProgressBar)
        caption = self.query_one("#activity-bar-caption", Label)
        current_file = self.query_one("#activity-current-file", Static)

        if progress is not None:
            fraction = progress.fraction
            if fraction is not None:
                bar.update(total=100.0, progress=fraction * 100.0)
                caption_str = (
                    f"{progress.candidates_seen:,} / ~{progress.candidates_total:,} candidates"
                )
            else:
                bar.update(total=None)
                if progress.phase == "embedding" and progress.chunks_extracted > 0:
                    caption_str = (
                        f"Batch: {progress.chunks_embedded:,} / "
                        f"{progress.chunks_extracted:,} chunks"
                    )
                else:
                    caption_str = f"Streaming ({progress.candidates_seen:,} seen)"
            caption.update(caption_str)

            file_line = Text("In flight: ", style="dim")
            if progress.current_path:
                file_line.append(_truncate_left(progress.current_path), style="cyan")
            elif progress.phase == "committing":
                file_line.append("Writing Arrow batches and updating Lance indexes…", style="cyan")
            elif progress.phase == "embedding":
                file_line.append("Preparing embedding batch on worker…", style="cyan")
            else:
                file_line.append("Walking repository tree…", style="dim")
            current_file.update(file_line)
        elif self._starting_phase is not None:
            bar.update(total=None)
            caption.update("Starting…")
            current_file.update(Text(f"Status: {self._starting_phase}…", style="cyan"))
        elif report is not None:
            bar.update(total=100.0, progress=100.0)
            caption.update(f"{report.discovered_files:,} eligible files")
            current_file.update(
                Text(
                    f"Last completed: {report.indexed_files:,} files updated, "
                    f"{report.removed_files:,} removed, {report.skipped_files:,} skipped.",
                    style="dim",
                )
            )
        elif self._last_error is not None:
            bar.update(total=100.0, progress=0.0)
            caption.update("Failed")
            current_file.update(Text("Run stopped before commit.", style="red"))
        elif latest_audit is not None:
            bar.update(
                total=100.0,
                progress=100.0 if latest_audit.state == "completed" else 0.0,
            )
            caption.update(f"Last: {latest_audit.state}")
            current_file.update(
                Text(
                    f"Ready · {latest_audit.eligible_files:,} eligible files tracked in last run.",
                    style="dim",
                )
            )
        else:
            bar.update(total=100.0, progress=0.0)
            caption.update("Idle")
            current_file.update(Text("Idle · Waiting for indexing activity.", style="dim"))

    def _render_metrics(self) -> None:
        progress = self._live_progress
        report = self._last_report
        latest = self._latest_history_run()
        elapsed = max(0.001, self._elapsed_seconds())

        files_text = Text()
        files_text.append("Files & Repository Scan\n", style="bold underline")

        chunks_text = Text()
        chunks_text.append("Chunks, Embeddings & Cache\n", style="bold underline")

        if progress is not None:
            cand_str = (
                f"{progress.candidates_seen:,} / ~{progress.candidates_total:,}"
                if progress.candidates_total
                else f"{progress.candidates_seen:,}"
            )
            files_rate = progress.candidates_seen / elapsed
            bytes_rate = progress.bytes_read / elapsed
            files_text.append(f"Candidates seen : {cand_str} ({files_rate:,.1f}/s)\n")
            files_text.append(f"Eligible files  : {progress.eligible_files:,}\n")
            files_text.append(
                f"Changed / Unchg : {progress.changed_files:,} indexed · "
                f"{progress.unchanged_files:,} unchanged\n"
            )
            files_text.append(f"Parsed files    : {progress.parsed_files:,}\n")
            files_text.append(f"Skipped / Failed: {progress.skipped_total:,} skipped · ")
            files_text.append(
                f"{progress.failed_files:,} failed\n",
                style="bold red" if progress.failed_files else "",
            )
            files_text.append(
                f"Source I/O read : {format_bytes(progress.bytes_read)} "
                f"({format_bytes(bytes_rate)}/s)"
            )

            chunk_rate = progress.chunks_embedded / elapsed
            chunks_text.append(f"Chunks extracted: {progress.chunks_extracted:,}\n")
            chunks_text.append(
                f"Chunks embedded : {progress.chunks_embedded:,} ({chunk_rate:,.1f}/s)\n"
            )
            chunks_text.append(
                f"Passage cache   : {progress.reused_candidates:,} reused "
                f"({progress.reused_segments:,} segments)\n"
            )
            chunks_text.append(f"Chunks staged   : {progress.chunks_staged:,}\n")
            chunks_text.append(f"Staged payload  : {format_bytes(progress.staged_bytes)}")
        elif report is not None:
            dur_s = max(0.001, report.duration_ms / 1000.0)
            files_text.append(f"Eligible files  : {report.discovered_files:,}\n")
            files_text.append(
                f"Changed / Unchg : {report.indexed_files:,} indexed · "
                f"{report.unchanged_files:,} unchanged\n"
            )
            files_text.append(
                f"Parsed / Removed: {report.parsed_files:,} parsed · "
                f"{report.removed_files:,} removed\n"
            )
            files_text.append(f"Skipped / Failed: {report.skipped_files:,} skipped · ")
            files_text.append(
                f"{report.failed_files:,} failed\n",
                style="bold red" if report.failed_files else "",
            )
            files_text.append(
                f"Source I/O read : {format_bytes(report.bytes_read)} "
                f"({format_bytes(report.bytes_read / dur_s)}/s)"
            )

            chunks_text.append(f"Chunks extracted: {report.chunks_extracted:,}\n")
            chunks_text.append(
                f"Chunks embedded : {report.embedded_chunks:,} "
                f"({report.embedded_chunks / dur_s:,.1f}/s)\n"
            )
            chunks_text.append(
                f"Passage cache   : {report.reused_candidates:,} reused "
                f"({report.reused_segments:,} segments)\n"
            )
            chunks_text.append(
                f"Refs staged     : {report.staged_reference_rows:,} structural rows\n"
            )
            chunks_text.append(
                f"Staged payload  : {report.chunks_staged:,} chunks "
                f"({format_bytes(report.staged_bytes)})"
            )
        elif latest is not None:
            files_text.append(f"Eligible files  : {latest.eligible_files:,}\n")
            files_text.append(
                f"Changed / Unchg : {latest.changed_files:,} indexed · "
                f"{latest.unchanged_files:,} unchanged\n"
            )
            files_text.append(
                f"Parsed / Removed: {latest.parsed_files:,} parsed · "
                f"{latest.removed_files:,} removed\n"
            )
            files_text.append(f"Skipped / Failed: {latest.skipped_total:,} skipped · ")
            files_text.append(
                f"{latest.failed_files:,} failed\n",
                style="bold red" if latest.failed_files else "",
            )
            files_text.append(f"Source I/O read : {format_bytes(latest.bytes_read)}")

            chunks_text.append(f"Chunks extracted: {latest.chunks_extracted:,}\n")
            chunks_text.append(f"Chunks embedded : {latest.chunks_embedded:,}\n")
            chunks_text.append(
                f"Passage cache   : {latest.reused_candidates:,} reused "
                f"({latest.reused_segments:,} segments)\n"
            )
            chunks_text.append(f"Chunks staged   : {latest.chunks_staged:,}\n")
            chunks_text.append(f"Staged payload  : {format_bytes(latest.staged_bytes)}")
        else:
            status = self._process_snapshot.project_status if self._process_snapshot else None
            if status is not None:
                files_text.append(f"Index state     : {status.state}\n")
                files_text.append(f"Committed files : {status.file_count:,}\n")
                if status.git_selector_value:
                    head = f" ({status.git_head[:7]})" if status.git_head else ""
                    files_text.append(f"Active branch   : {status.git_selector_value}{head}\n")
                chunks_text.append(f"Committed chunks: {status.chunk_count:,}\n")
                chunks_text.append("No run telemetry recorded in this session yet.", style="dim")
            else:
                files_text.append("Waiting for first indexing run…", style="dim")
                chunks_text.append("Waiting for first indexing run…", style="dim")

        self.query_one("#activity-files-card", Static).update(files_text)
        self.query_one("#activity-chunks-card", Static).update(chunks_text)

    def _render_diagnostics(self) -> None:
        text = Text()
        progress = self._live_progress
        report = self._last_report
        latest = self._latest_history_run()

        skip_reasons: dict[str, int] = {}
        errors: list[IndexIssue] = []
        if progress is not None:
            skip_reasons = progress.skipped_by_reason
            errors = progress.recent_errors
        elif report is not None:
            skip_reasons = report.skip_reasons
            errors = report.errors
        elif latest is not None:
            skip_reasons = latest.skip_reasons
            errors = latest.errors

        if skip_reasons:
            text.append("Skipped Candidates Breakdown\n", style="bold underline")
            ordered = sorted(skip_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
            breakdown = "  ·  ".join(f"{reason}: {count:,}" for reason, count in ordered)
            text.append(f"{breakdown}\n\n", style="dim")

        if errors:
            text.append(f"File Issues ({len(errors)})\n", style="bold red underline")
            for issue in errors[:10]:
                text.append(f"  ✗ {issue.path}: ", style="bold red")
                text.append(f"{issue.message}\n", style="yellow")
            if len(errors) > 10:
                text.append(f"  … and {len(errors) - 10} more error(s)\n", style="dim")
            text.append("\n")

        snap = self._process_snapshot
        if snap is not None:
            other_runs = [
                run
                for run in snap.active_runs
                if self._project is None or run.project_id != self._project.id
            ]
            if other_runs:
                text.append("Concurrent Runs Across Projects\n", style="bold cyan underline")
                for run in other_runs:
                    text.append(
                        f"  ● {run.project_id} · {run.phase} · "
                        f"{run.candidates_seen:,} seen, {run.chunks_embedded:,} embedded "
                        f"(PID {run.pid}, {run.trigger})\n",
                        style="cyan",
                    )
                text.append("\n")

            text.append("Process & Backend Runtime\n", style="bold underline")
            if snap.mode == "daemon":
                daemon_label = (
                    f"Shared daemon (PID {snap.daemon_pid})" if snap.daemon_pid else "Shared daemon"
                )
            else:
                daemon_label = "Direct in-process engine"
            text.append(f"  Execution  : {daemon_label}  ·  UI client PID {snap.client_pid}\n")
            if snap.model_status is not None:
                ms = snap.model_status
                text.append(
                    f"  Accelerator: {ms.resolved_accelerator} "
                    f"(requested: {ms.requested_accelerator}, provider: {ms.execution_provider})\n"
                )
                text.append(
                    f"  Model      : {ms.embedding_model} ({ms.precision}, "
                    f"batch {ms.batch_size} [{ms.batch_calibration}])\n"
                )
                if ms.fallback_reason:
                    text.append(f"  Fallback   : {ms.fallback_reason}\n", style="yellow")
            text.append("\n")

            if snap.recent_runs:
                text.append("Recent Run Audit Trail\n", style="bold underline")
                for audit_run in snap.recent_runs[:8]:
                    dur_ms = sum(audit_run.phase_durations.values())
                    dur_label = format_duration_ms(dur_ms) if dur_ms else "—"
                    icon = (
                        "✓"
                        if audit_run.state == "completed" and not audit_run.failed_files
                        else "⚠"
                        if audit_run.state == "completed"
                        else "●"
                        if audit_run.state == "running"
                        else "✗"
                    )
                    style = (
                        "green"
                        if icon == "✓"
                        else "yellow"
                        if icon == "⚠"
                        else "cyan"
                        if icon == "●"
                        else "red"
                    )
                    text.append(f"  {icon} ", style=style)
                    text.append(
                        f"{_short_timestamp(audit_run.started_at)}  "
                        f"{audit_run.run_id[:8]}  "
                        f"{audit_run.trigger:<18}  "
                        f"{audit_run.state:<11}  "
                        f"{dur_label:>7}  "
                        f"chg:{audit_run.changed_files:,} emb:{audit_run.chunks_embedded:,} "
                        f"cache:{audit_run.reused_candidates:,} [{audit_run.embedding_backend}]"
                    )
                    if audit_run.failed_files:
                        text.append(f" err:{audit_run.failed_files}", style="red")
                    text.append("\n")

        if not text.plain.strip():
            text.append("No warnings, file skips, or errors reported.", style="dim")

        self.query_one("#activity-diagnostics", Static).update(text)


class IndexRunApp(App[int]):
    """Standalone Textual live dashboard for interactive `index` CLI runs."""

    TITLE = "Code Indexing MCP · Indexer"

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
        color: $text;
    }
    #cli-activity-panel {
        height: 1fr;
        padding: 0 1;
    }
    #activity-header-row {
        height: 2;
        background: $panel;
        padding: 0 1;
    }
    #activity-title {
        text-style: bold;
        width: 1fr;
    }
    #activity-banner {
        padding: 0 1;
        margin-top: 1;
        border: round $accent;
        height: auto;
    }
    #activity-pipeline-card {
        padding: 0 1;
        border: round $secondary;
        height: auto;
    }
    #activity-bar-row {
        height: 1;
        margin-top: 0;
    }
    #activity-progress-bar {
        width: 1fr;
        margin-right: 2;
    }
    #activity-bar-caption {
        width: auto;
        color: $text-muted;
    }
    #activity-current-file {
        height: 1;
    }
    #activity-scroll {
        height: 1fr;
    }
    #activity-metrics-grid {
        height: auto;
    }
    .compact #activity-metrics-grid {
        layout: vertical;
    }
    .activity-card {
        width: 1fr;
        height: auto;
        border: solid $secondary;
        padding: 0 1;
        margin-right: 1;
    }
    #activity-chunks-card, #activity-diagnostics {
        margin-right: 0;
    }
    #cli-status-bar {
        height: 1;
        background: $panel;
        padding: 0 1;
        color: $text-muted;
    }
    Footer {
        dock: bottom;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("p", "toggle_hold", "Hold open", show=True),
        Binding("q,escape", "close_when_done", "Close", show=True),
        Binding("ctrl+q", "quit", "Abort UI", show=False, priority=True),
    ]

    def __init__(
        self,
        run_index: Callable[[Callable[[IndexProgress], None]], IndexReport],
        *,
        project_label: str = "repository",
        hold_open: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._run_index = run_index
        self._project_label = project_label
        self.hold_open = hold_open
        self.report: IndexReport | None = None
        self.error: BaseException | None = None
        self._finished = False
        self._ticker: Timer | None = None

    def compose(self) -> ComposeResult:
        yield IndexingActivityPanel(show_actions=False, id="cli-activity-panel")
        yield Static("", id="cli-status-bar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._update_compact(self.size.width)
        panel = self.query_one("#cli-activity-panel", IndexingActivityPanel)
        panel.set_project(None, label=self._project_label)
        panel.mark_starting("Scanning & indexing")
        self._update_status_line()
        self._ticker = self.set_interval(0.15, self._on_tick)
        self._execute_index()

    def on_resize(self, event: events.Resize) -> None:
        self._update_compact(event.size.width)

    def _update_compact(self, width: int) -> None:
        self.set_class(width < 100, "compact")

    def _on_tick(self) -> None:
        if self.is_running and self.query("#cli-activity-panel"):
            self.query_one("#cli-activity-panel", IndexingActivityPanel).tick()

    def _update_status_line(self) -> None:
        if not self.query("#cli-status-bar"):
            return
        bar = self.query_one("#cli-status-bar", Static)
        if not self._finished:
            hold_state = (
                "Holding panel open after completion (press p to auto-exit)."
                if self.hold_open
                else "Auto-exits when finished (press p to hold panel open)."
            )
            bar.update(f"Indexing {self._project_label}… · {hold_state}")
        elif self.error is not None:
            bar.update("Indexing failed · Press q or Esc to exit.")
        else:
            bar.update("Indexing complete · Press q or Esc to exit.")

    def action_toggle_hold(self) -> None:
        if self._finished:
            return
        self.hold_open = not self.hold_open
        self._update_status_line()

    def action_close_when_done(self) -> None:
        if self._finished:
            code = 2 if self.error is not None else 0
            self.exit(result=code, return_code=code)

    def exit(
        self,
        result: int | None = None,
        return_code: int = 0,
        message: Any | None = None,
    ) -> None:
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        super().exit(result, return_code, message)

    def _deliver(self, callback: Callable[..., None], *args: Any) -> None:
        if not self.is_running:
            return

        def apply() -> None:
            if self.is_running:
                callback(*args)

        try:
            self.call_from_thread(apply)
        except (CancelledError, RuntimeError):
            if self.is_running:
                raise

    @work(thread=True, exclusive=True)
    def _execute_index(self) -> tuple[IndexReport | None, BaseException | None]:
        def on_progress(snapshot: IndexProgress) -> None:
            self._deliver(self._on_progress, snapshot)

        try:
            report = self._run_index(on_progress)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return None, exc
        return report, None

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "_execute_index" or event.state is not WorkerState.SUCCESS:
            return
        result: tuple[IndexReport | None, BaseException | None] | None = event.worker.result
        if result is None:
            return
        report, exc = result
        if exc is not None:
            self._on_failed(exc)
        elif report is not None:
            self._on_completed(report)

    def _on_progress(self, snapshot: IndexProgress) -> None:
        if self.query("#cli-activity-panel"):
            self.query_one("#cli-activity-panel", IndexingActivityPanel).set_live_progress(snapshot)

    def _on_completed(self, report: IndexReport) -> None:
        self.report = report
        self._finished = True
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        if self.query("#cli-activity-panel"):
            self.query_one("#cli-activity-panel", IndexingActivityPanel).set_completed_report(
                report
            )
        self._update_status_line()
        if not self.hold_open:
            self.exit(0)

    def _on_failed(self, exc: BaseException) -> None:
        self.error = exc
        self._finished = True
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None
        if self.query("#cli-activity-panel"):
            self.query_one("#cli-activity-panel", IndexingActivityPanel).set_failed(str(exc))
        self._update_status_line()
        if not self.hold_open:
            self.exit(result=2, return_code=2)
