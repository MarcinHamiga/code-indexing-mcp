"""Textual user interface application for Code Indexing MCP."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from ..errors import CodeIndexingError
from ..models import (
    CodeChunk,
    ImpactRadiusResponse,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    ReferenceResponse,
    SearchHit,
)
from .service import TuiService, create_tui_service


class CodeIndexingApp(App[int]):
    """Terminal user interface for exploring indexed codebases."""

    TITLE = "Code Indexing MCP"
    CSS_PATH = "app.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("/", "focus_query", "Search", show=True, priority=False),
        Binding("enter", "open_selected", "Open", show=True, priority=False),
        Binding("o", "show_outline", "Outline", show=True, priority=False),
        Binding("r", "show_references", "References", show=True, priority=False),
        Binding("i", "show_impact", "Impact", show=True, priority=False),
        Binding("f5", "trigger_index", "Index", show=True, priority=False),
        Binding("escape", "escape_action", "Back", show=True, priority=False),
        Binding("q", "quit_app", "Quit", show=True, priority=False),
    ]

    def __init__(
        self,
        service: TuiService | None = None,
        *,
        cwd: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.service = service if service is not None else create_tui_service(cwd=cwd)
        self._hits: list[SearchHit] = []
        self._highlighted_index: int | None = None
        self._search_request_id: int = 0
        self._detail_request_id: int = 0
        self._is_indexing: bool = False
        self._projects: dict[str, ProjectInfo] = {}
        self._project_request_id = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="header-bar"):
            yield Label("Code Indexing MCP", id="header-title", markup=False)
            yield Label("Initializing...", id="header-status", markup=False)

        with Horizontal(id="query-bar"):
            yield Select[str]([], prompt="Select project", id="project-select", allow_blank=True)
            yield Select[str](
                [("Semantic", "semantic"), ("Symbol", "symbol")],
                value="semantic",
                id="mode-select",
                allow_blank=False,
            )
            yield Input(
                placeholder="Search query... (/ to focus, Enter to run)",
                id="query-input",
            )
            yield Button("Index [F5]", id="index-button", variant="primary")

        with Horizontal(id="main-container"):
            with Vertical(id="results-pane"):
                yield Label("Results", id="results-title", markup=False)
                yield OptionList(id="results-list")

            with Vertical(id="detail-pane"):
                yield Label("Code Preview", id="detail-title", markup=False)
                with VerticalScroll(id="detail-scroll"):
                    yield Static(
                        "Run a query, then select a result and press:\n\n"
                        "  Enter  Preview chunk source code\n"
                        "  o      Show file outline\n"
                        "  r      Find references\n"
                        "  i      Show impact radius\n"
                        "  F5     Index project\n"
                        "  q      Quit",
                        id="detail-content",
                    )

        yield Static("Ready", id="status-bar", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.run_discovery()

    @work(thread=True, exclusive=True)
    def run_discovery(self) -> None:
        try:
            current = self.service.selected_project or self.service.discover_current_project()
            projects = self.service.list_projects()
            status = self.service.project_status() if current else None
            self.call_from_thread(self._setup_projects_ui, projects, current, status)
        except CodeIndexingError as exc:
            self.call_from_thread(self._show_error, str(exc))
        except Exception as exc:
            self.call_from_thread(self._show_error, f"Discovery failed: {exc}")

    def _setup_projects_ui(
        self,
        projects: Sequence[ProjectInfo],
        selected: ProjectInfo | None,
        status: ProjectStatus | None,
    ) -> None:
        self._projects = {p.id: p for p in projects}
        select = self.query_one("#project-select", Select)
        options = [(f"{p.name} ({p.id})", p.id) for p in projects]
        select.set_options(options)
        if selected is not None:
            select.value = selected.id
            self._update_header(selected, status)
            self._set_status("Ready")
        else:
            self._set_status("No project registered or discovered.")

    def _update_header(self, project: ProjectInfo, status: ProjectStatus | None) -> None:
        title = self.query_one("#header-title", Label)
        status_label = self.query_one("#header-status", Label)

        title.update(f"Project: {project.name} [{project.id}]")
        if status is not None:
            branch_info = ""
            if status.git_selector_value:
                branch_info = f" | Branch: {status.git_selector_value}"
            status_text = (
                f"State: {status.state} "
                f"({status.file_count} files, {status.chunk_count} chunks){branch_info}"
            )
            status_label.update(status_text)
        else:
            status_label.update(f"Root: {project.root}")

    def _set_status(self, text: str, *, error: bool = False) -> None:
        bar = self.query_one("#status-bar", Static)
        bar.remove_class("error")
        if error:
            bar.add_class("error")
        bar.update(text)

    def _show_error(self, message: str) -> None:
        self._set_status(message, error=True)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "project-select" and event.value != Select.BLANK:
            self._on_project_selected(str(event.value))

    def _clear_details(self, message: str) -> None:
        self._detail_request_id += 1
        self.query_one("#detail-title", Label).update("Code Preview")
        self.query_one("#detail-content", Static).update(message)
        self.query_one("#detail-scroll", VerticalScroll).scroll_home(animate=False)

    def _on_project_selected(self, project_id: str) -> None:
        project = self._projects.get(project_id)
        if project is None:
            return
        self.service.select_project(project)
        self._project_request_id += 1
        self._search_request_id += 1
        self._hits = []
        self._highlighted_index = None
        self.query_one("#results-list", OptionList).clear_options()
        self.query_one("#results-title", Label).update("Results")
        self._clear_details(
            f"Active project changed to {project.name}.\n\n"
            "Run a query to search, or press F5 to index."
        )
        self._update_header(project, None)
        self._set_status(f"Selected project: {project.name}")
        self._load_project_status(self._project_request_id, project)

    @work(thread=True, exclusive=True, group="project-status")
    def _load_project_status(self, request_id: int, project: ProjectInfo) -> None:
        try:
            status = self.service.project_status(project)
            self.call_from_thread(self._apply_project_status, request_id, project, status)
        except Exception as exc:
            self.call_from_thread(self._project_error, request_id, str(exc))

    def _apply_project_status(
        self, request_id: int, project: ProjectInfo, status: ProjectStatus
    ) -> None:
        if request_id == self._project_request_id:
            self._update_header(project, status)

    def _project_error(self, request_id: int, message: str) -> None:
        if request_id == self._project_request_id:
            self._show_error(message)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input":
            self.action_submit_query()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "index-button":
            self.action_trigger_index()

    def action_focus_query(self) -> None:
        self.query_one("#query-input", Input).focus()

    def action_escape_action(self) -> None:
        focused = self.focused
        if isinstance(focused, Input) or (focused and focused.id == "detail-scroll"):
            self.query_one("#results-list", OptionList).focus()
        else:
            self.query_one("#query-input", Input).focus()

    def action_quit_app(self) -> None:
        if isinstance(self.focused, Input):
            return
        self.exit(0)

    def action_submit_query(self) -> None:
        query_input = self.query_one("#query-input", Input)
        query = query_input.value.strip()
        if not query:
            self._set_status("Enter a query to search.")
            return

        mode_select = self.query_one("#mode-select", Select)
        mode = str(mode_select.value) if mode_select.value != Select.BLANK else "semantic"

        self._clear_details("Searching… Select a result to preview its source.")
        self._search_request_id += 1
        req_id = self._search_request_id
        self._set_status(f"Searching ({mode})...")
        self._run_search_worker(req_id, mode, query, self.service.selected_project)

    @work(thread=True, exclusive=True, group="search")
    def _run_search_worker(
        self, request_id: int, mode: str, query: str, project: ProjectInfo | None
    ) -> None:
        try:
            hits = (
                self.service.find_symbol(query, project=project).hits
                if mode == "symbol"
                else self.service.search_code(query, project=project).hits
            )

            if request_id == self._search_request_id:
                self.call_from_thread(self._render_search_results, request_id, hits, query)
        except CodeIndexingError as exc:
            if request_id == self._search_request_id:
                self.call_from_thread(self._search_error, request_id, str(exc))
        except Exception as exc:
            if request_id == self._search_request_id:
                self.call_from_thread(self._search_error, request_id, f"Search failed: {exc}")

    def _search_error(self, request_id: int, message: str) -> None:
        if request_id == self._search_request_id:
            self._show_error(message)

    def _render_search_results(self, request_id: int, hits: list[SearchHit], query: str) -> None:
        if request_id != self._search_request_id:
            return

        self._hits = hits
        self._highlighted_index = 0 if hits else None
        option_list = self.query_one("#results-list", OptionList)
        option_list.clear_options()

        if not hits:
            self._clear_details("No matches. Try a broader query or a different project.")
            self._set_status(f"No results found for '{query}'.")
            self.query_one("#results-title", Label).update("Results (0)")
            return

        self.query_one("#results-title", Label).update(f"Results ({len(hits)})")
        options: list[Option] = []
        for i, hit in enumerate(hits):
            text = Text()
            text.append(f"{i + 1:2d}. ", style="dim")
            text.append(f"{hit.path}:{hit.start_line}", style="bold cyan")
            if hit.symbol:
                text.append(f"  {hit.symbol}", style="green")
            text.append(f" ({hit.kind})", style="yellow")
            text.append(f" [{hit.score:.2f}]", style="magenta")
            options.append(Option(prompt=text, id=str(i)))

        option_list.add_options(options)
        self._set_status(f"Found {len(hits)} hit(s).")
        option_list.focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._highlighted_index = event.option_index

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._highlighted_index = event.option_index
        self._load_detail_for_selected("chunk")

    def _get_selected_hit(self) -> SearchHit | None:
        if self._highlighted_index is not None and 0 <= self._highlighted_index < len(self._hits):
            return self._hits[self._highlighted_index]
        option_list = self.query_one("#results-list", OptionList)
        if option_list.highlighted is not None and 0 <= option_list.highlighted < len(self._hits):
            return self._hits[option_list.highlighted]
        return None

    def action_open_selected(self) -> None:
        if isinstance(self.focused, Input):
            self.action_submit_query()
            return
        self._load_detail_for_selected("chunk")

    def action_show_outline(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._load_detail_for_selected("outline")

    def action_show_references(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._load_detail_for_selected("references")

    def action_show_impact(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._load_detail_for_selected("impact")

    def _load_detail_for_selected(self, action: str) -> None:
        hit = self._get_selected_hit()
        if hit is None:
            self._set_status("No hit selected. Run a search and select a result first.")
            return

        self._detail_request_id += 1
        req_id = self._detail_request_id
        self._set_status(f"Loading {action} for {hit.path}...")
        self._run_detail_worker(req_id, action, hit, self.service.selected_project)

    @work(thread=True, exclusive=True, group="detail")
    def _run_detail_worker(
        self, request_id: int, action: str, hit: SearchHit, project: ProjectInfo | None
    ) -> None:
        try:
            if action == "chunk":
                chunk = self.service.get_chunk(hit.chunk_id)
                if request_id == self._detail_request_id:
                    self.call_from_thread(self._apply_detail, request_id, self._render_chunk, chunk)
            elif action == "outline":
                outline = self.service.file_outline(hit.path, project)
                if request_id == self._detail_request_id:
                    self.call_from_thread(
                        self._apply_detail, request_id, self._render_outline, outline
                    )
            elif action == "references":
                refs = self.service.find_references(hit, project=project)
                if request_id == self._detail_request_id:
                    self.call_from_thread(
                        self._apply_detail, request_id, self._render_references, refs
                    )
            elif action == "impact":
                impact = self.service.impact_radius(hit, project=project)
                if request_id == self._detail_request_id:
                    self.call_from_thread(
                        self._apply_detail, request_id, self._render_impact, impact
                    )
        except CodeIndexingError as exc:
            if request_id == self._detail_request_id:
                self.call_from_thread(self._detail_error, request_id, str(exc))
        except Exception as exc:
            if request_id == self._detail_request_id:
                self.call_from_thread(
                    self._detail_error, request_id, f"Failed to load {action}: {exc}"
                )

    def _apply_detail(self, request_id: int, render: Callable[..., None], value: Any) -> None:
        if request_id == self._detail_request_id:
            render(value)

    def _detail_error(self, request_id: int, message: str) -> None:
        if request_id == self._detail_request_id:
            self._show_error(message)

    def _render_chunk(self, chunk: CodeChunk) -> None:
        self.query_one("#detail-title", Label).update(
            f"Preview: {chunk.path}:{chunk.start_line}-{chunk.end_line} ({chunk.kind})"
        )
        syntax = Syntax(
            chunk.content,
            chunk.language or "text",
            line_numbers=True,
            start_line=chunk.start_line,
            word_wrap=True,
            theme="monokai",
        )
        self.query_one("#detail-content", Static).update(syntax)
        self._set_status(f"Loaded chunk {chunk.chunk_id[:8]} for {chunk.path}")

    def _render_outline(self, outline: OutlineResponse) -> None:
        self.query_one("#detail-title", Label).update(
            f"Outline: {outline.path} ({len(outline.items)} items)"
        )
        text = Text()
        text.append(f"File Outline for {outline.path}\n\n", style="bold underline")
        if not outline.items:
            text.append("No outline symbols discovered in this file.", style="dim")
        for item in outline.items:
            text.append(f"  L{item.start_line:4d}-{item.end_line:4d} ", style="dim")
            text.append(f"{item.kind:<12} ", style="yellow")
            text.append(f"{item.qualified_symbol}\n", style="green bold")

        self.query_one("#detail-content", Static).update(text)
        self._set_status(f"Loaded outline with {len(outline.items)} items.")

    def _render_references(self, refs: ReferenceResponse) -> None:
        target_name = refs.selected.path
        self.query_one("#detail-title", Label).update(
            f"References: {target_name} ({len(refs.hits)} hits)"
        )
        text = Text()
        text.append(
            f"References to declaration in {refs.selected.path}\n\n",
            style="bold underline",
        )
        if not refs.hits:
            text.append("No references found for this declaration.", style="dim")
        for hit in refs.hits:
            text.append(f"  {hit.path}:{hit.start_line} ", style="cyan bold")
            text.append(f"[{hit.resolution}] ", style="magenta")
            text.append(f"{hit.snippet.strip()}\n", style="white")

        if refs.limitations:
            text.append("\nLimitations:\n", style="yellow bold")
            for lim in refs.limitations:
                text.append(f"  - [{lim.code}] {lim.explanation}\n", style="yellow")

        self.query_one("#detail-content", Static).update(text)
        self._set_status(f"Loaded {len(refs.hits)} reference(s).")

    def _render_impact(self, impact: ImpactRadiusResponse) -> None:
        title_text = (
            f"Impact Radius: {impact.selected.path} "
            f"({len(impact.layers)} layers, {impact.visited} visited)"
        )
        self.query_one("#detail-title", Label).update(title_text)
        text = Text()
        text.append(
            f"Impact Radius for {impact.selected.path} (visited: {impact.visited})\n\n",
            style="bold underline",
        )
        if not impact.layers:
            text.append("No downstream impact detected within search depth.", style="dim")
        for layer in impact.layers:
            text.append(f"Depth {layer.depth}:\n", style="yellow bold")
            for edge in layer.edges:
                kinds = ", ".join(edge.kinds)
                text.append(f"  -> {edge.target.path} ", style="cyan")
                text.append(f"({kinds})\n", style="dim")

        self.query_one("#detail-content", Static).update(text)
        self._set_status(f"Impact graph loaded: {impact.visited} declaration(s) analyzed.")

    def action_trigger_index(self) -> None:
        if self._is_indexing:
            self._set_status("Index already in progress...")
            return
        proj = self.service.selected_project
        if proj is None:
            self._show_error("No project selected to index.")
            return

        self._is_indexing = True
        self._set_status(f"Starting index for {proj.name}...")
        self._run_index_worker(proj)

    @work(thread=True, exclusive=True, group="index")
    def _run_index_worker(self, project: ProjectInfo) -> None:
        stop_polling = False

        def poll_progress() -> None:
            while not stop_polling:
                prog = self.service.index_progress(project)
                if prog is not None:
                    msg = (
                        f"Indexing {project.name}: {prog.phase} "
                        f"({prog.candidates_seen} files, {prog.chunks_extracted} chunks)"
                    )
                    self.call_from_thread(self._set_status, msg)
                time.sleep(0.25)

        poller = threading.Thread(target=poll_progress, daemon=True)
        poller.start()

        try:
            report = self.service.index_project(project)
            stop_polling = True
            poller.join(timeout=1.0)
            status = self.service.project_status(project)
            self.call_from_thread(self._on_index_complete, project, status, report)
        except CodeIndexingError as exc:
            stop_polling = True
            self.call_from_thread(self._on_index_failed, str(exc))
        except Exception as exc:
            stop_polling = True
            self.call_from_thread(self._on_index_failed, f"Indexing failed: {exc}")

    def _on_index_complete(self, project: ProjectInfo, status: ProjectStatus, report: Any) -> None:
        self._is_indexing = False
        if self.service.selected_project == project:
            self._update_header(project, status)
        files_indexed = getattr(report, "indexed_files", getattr(report, "files_indexed", 0))
        chunks_embedded = getattr(report, "embedded_chunks", getattr(report, "chunks_staged", 0))
        self._set_status(
            f"Indexing complete: {files_indexed} files indexed, {chunks_embedded} chunks embedded."
        )

    def _on_index_failed(self, error_message: str) -> None:
        self._is_indexing = False
        self._show_error(error_message)
