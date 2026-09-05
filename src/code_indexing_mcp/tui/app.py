"""Textual user interface application for Code Indexing MCP."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from rich.syntax import Syntax
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Footer, Input, Label, OptionList, Select, Static, Tab, Tabs
from textual.widgets.option_list import Option

from ..errors import CodeIndexingError, ErrorCode
from ..models import (
    CodeChunk,
    DeclarationSelector,
    ImpactRadiusResponse,
    OutlineResponse,
    ProjectInfo,
    ProjectStatus,
    ReferenceResponse,
    SearchHit,
)
from .navigation import SourceLocation, SourcePreview
from .service import TuiService, create_tui_service


@dataclass
class DetailState:
    target: SourceLocation | None
    mode: str
    title: str
    content: Any
    entries: list[tuple[Text, SourceLocation]]
    highlighted: int | None
    scroll_y: float
    list_scroll_y: float


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
        self._detail_visible = False

        self._active_target: SourceLocation | None = None
        self._detail_mode = "chunk"
        self._detail_entries: list[tuple[Text, SourceLocation]] = []
        self._history: list[DetailState] = []
        self._preview_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="header-bar"):
            yield Label("Code Indexing MCP", id="header-title", markup=False)
            yield Label("Initializing...", id="header-status", markup=False)

        with Horizontal(id="project-bar"):
            yield Select[str]([], prompt="Select project", id="project-select", allow_blank=True)
            yield Select[str](
                [("Describe code", "semantic"), ("Find symbol", "symbol")],
                value="semantic",
                id="mode-select",
                allow_blank=False,
            )
            yield Select[str](
                [("Exact", "exact"), ("Starts with", "prefix"), ("Contains", "contains")],
                value="exact",
                id="match-select",
                allow_blank=False,
            )
            yield Button("Index [F5]", id="index-button")

        with Horizontal(id="query-bar"):
            yield Input(
                placeholder="Describe code: where are expired tokens rejected?",
                id="query-input",
            )
        with Horizontal(id="pane-switch"):
            yield Button("Results", id="results-view")
            yield Button("Details", id="details-view")

        with Horizontal(id="main-container"):
            with Vertical(id="results-pane"):
                yield Label("Results", id="results-title", markup=False)
                yield OptionList(id="results-list")

            with Vertical(id="detail-pane"):
                yield Label("Code Preview", id="detail-title", markup=False)
                yield Tabs(
                    Tab("Code", id="chunk-tab"),
                    Tab("Outline", id="outline-tab"),
                    Tab("References", id="references-tab"),
                    Tab("Impact", id="impact-tab"),
                    id="detail-tabs",
                )
                yield OptionList(id="detail-list")
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
        self.query_one("#detail-list").display = False
        self._update_search_mode()
        self._update_layout()
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
        options = [(p.name, p.id) for p in projects]
        select.set_options(options)
        if selected is not None:
            select.value = selected.id
            self._update_header(selected, status)
            self._set_status("Ready")
            self.action_focus_query()
        else:
            self._set_status("No project registered or discovered.")

    def _update_header(self, project: ProjectInfo, status: ProjectStatus | None) -> None:
        title = self.query_one("#header-title", Label)
        status_label = self.query_one("#header-status", Label)

        title.update(f"Syndex · {project.name}")
        title.tooltip = f"{project.root}\nProject ID: {project.id}"
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
        if event.select.id == "mode-select":
            self._update_search_mode()
        if event.select.id == "project-select" and event.value != Select.BLANK:
            self._on_project_selected(str(event.value))

    def _update_search_mode(self) -> None:
        symbol = self.query_one("#mode-select", Select).value == "symbol"
        self.query_one("#match-select", Select).display = symbol
        self.query_one("#query-input", Input).placeholder = (
            "Find symbol: TokenValidator (choose how the name should match)"
            if symbol
            else "Describe code: where are expired tokens rejected?"
        )

    def on_resize(self, event: events.Resize) -> None:
        if self.query("#pane-switch"):
            self._update_layout(event.size.width)

    def _update_layout(self, width: int | None = None) -> None:
        compact = (self.size.width if width is None else width) < 100
        self.set_class(compact, "compact")
        self.query_one("#pane-switch").display = compact
        self.query_one("#results-pane").display = not compact or not self._detail_visible
        self.query_one("#detail-pane").display = not compact or self._detail_visible

    def _show_pane(self, details: bool) -> None:
        self._detail_visible = details
        self._update_layout()
        if details:
            entries = self.query_one("#detail-list", OptionList)
            (
                entries if entries.display else self.query_one("#detail-scroll", VerticalScroll)
            ).focus()
        else:
            self.query_one("#results-list", OptionList).focus()

    def _clear_details(self, message: str) -> None:
        self._detail_request_id += 1
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._active_target = None
        self._history.clear()
        self._set_detail_entries([])
        self.query_one("#detail-title", Label).update("Code Preview")
        self.query_one("#detail-content", Static).update(message)
        self.query_one("#detail-scroll", VerticalScroll).scroll_home(animate=False)

    def _on_project_selected(self, project_id: str) -> None:
        project = self._projects.get(project_id)
        if project is None:
            return
        self._detail_visible = False
        self._update_layout()
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
        if event.button.id in {"results-view", "details-view"}:
            self._show_pane(event.button.id == "details-view")
        if event.button.id == "index-button":
            self.action_trigger_index()

    def action_focus_query(self) -> None:
        self.query_one("#query-input", Input).focus()

    def action_escape_action(self) -> None:
        if self._history and not isinstance(self.focused, Input):
            self.action_detail_back()
            return
        if self._detail_visible:
            self._show_pane(False)
            return
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

        self._detail_visible = False
        self._update_layout()
        self._clear_details("Searching… Select a result to preview its source.")
        self._search_request_id += 1
        req_id = self._search_request_id
        self._set_status(f"Searching ({mode})...")
        self._run_search_worker(
            req_id,
            mode,
            query,
            self.service.selected_project,
            str(self.query_one("#match-select", Select).value),
        )

    @work(thread=True, exclusive=True, group="search")
    def _run_search_worker(
        self, request_id: int, mode: str, query: str, project: ProjectInfo | None, match: str
    ) -> None:
        try:
            hits = (
                self.service.find_symbol(query, project=project, match=match).hits
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
            self._clear_details(
                "No matches. Try a broader query, Contains symbol matching, or a different project."
            )
            self._set_status(f"No results found for '{query}'.")
            self.query_one("#results-title", Label).update("Results (0)")
            return

        self.query_one("#results-title", Label).update(f"Results ({len(hits)})")
        options: list[Option] = []
        for i, hit in enumerate(hits):
            text = Text()
            text.append(f"{i + 1:2d}. ", style="dim")
            text.append(hit.qualified_symbol or hit.symbol or Path(hit.path).name, style="bold")
            text.append(f"  {hit.kind}\n", style="dim")
            text.append(f"    {hit.path}:{hit.start_line}-{hit.end_line}", style="dim")
            options.append(Option(prompt=text, id=str(i)))

        option_list.add_options(options)
        option_list.highlighted = 0
        self._set_status(
            f"Found {len(hits)} hit(s)."
            if len(hits) < 20
            else "Showing up to 20 matches. Refine the query to narrow results."
        )
        option_list.focus()

    @staticmethod
    def _hit_location(hit: SearchHit) -> SourceLocation:
        return SourceLocation(
            hit.path,
            hit.start_line,
            hit.end_line,
            hit.qualified_symbol or hit.symbol,
            hit.language,
            hit.kind,
            hit.chunk_id,
        )

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "results-list":
            return
        self._highlighted_index = event.option_index
        self._detail_request_id += 1
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._preview_timer = self.set_timer(0.15, self._preview_highlighted)

    def _preview_highlighted(self) -> None:
        if self._get_selected_hit() is not None:
            self._load_detail_for_selected("chunk")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "detail-list":
            if event.option_index < len(self._detail_entries):
                self._remember_detail()
                self._open_target(self._detail_entries[event.option_index][1], "chunk")
            return
        if event.option_list.id == "results-list":
            self._highlighted_index = event.option_index
            self._load_detail_for_selected("chunk")
            self._show_pane(True)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tabs.id != "detail-tabs":
            return
        mode = (event.tab.id or "chunk-tab").removesuffix("-tab")
        if mode != self._detail_mode:
            self._navigate_mode(mode)

    def _navigate_mode(self, mode: str) -> None:
        if self._active_target is None:
            self._load_detail_for_selected(mode)
        else:
            self._remember_detail()
            self._open_target(self._active_target, mode)
        if self._active_target is not None:
            self._show_pane(True)

    def _remember_detail(self) -> None:
        self._history.append(
            DetailState(
                self._active_target,
                self._detail_mode,
                str(self.query_one("#detail-title", Label).render()),
                self.query_one("#detail-content", Static).content,
                list(self._detail_entries),
                self.query_one("#detail-list", OptionList).highlighted,
                self.query_one("#detail-scroll", VerticalScroll).scroll_y,
                self.query_one("#detail-list", OptionList).scroll_y,
            )
        )
        self._history = self._history[-50:]

    def action_detail_back(self) -> None:
        if not self._history:
            self._show_pane(False)
            return
        self._detail_request_id += 1
        state = self._history.pop()
        self._active_target = state.target
        self._detail_mode = state.mode
        self.query_one("#detail-tabs", Tabs).active = f"{state.mode}-tab"
        self.query_one("#detail-title", Label).update(state.title)
        self.query_one("#detail-content", Static).update(state.content)
        self._set_detail_entries(state.entries)
        entries = self.query_one("#detail-list", OptionList)
        entries.highlighted = state.highlighted
        self.call_after_refresh(entries.scroll_to, y=state.list_scroll_y, animate=False)
        scroll = self.query_one("#detail-scroll", VerticalScroll)
        self.call_after_refresh(scroll.scroll_to, y=state.scroll_y, animate=False)
        (entries if state.entries else scroll).focus()
        self._set_status("Returned to previous view.")

    def _set_detail_entries(self, entries: list[tuple[Text, SourceLocation]]) -> None:
        self._detail_entries = entries
        widget = self.query_one("#detail-list", OptionList)
        widget.clear_options()
        widget.add_options([Option(text) for text, _ in entries])
        widget.display = bool(entries)
        widget.highlighted = 0 if entries else None
        self.query_one("#detail-pane").set_class(bool(entries), "has-entries")

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
        if self._get_selected_hit() is not None:
            self._show_pane(True)

    def action_show_outline(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._navigate_mode("outline")

    def action_show_references(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._navigate_mode("references")

    def action_show_impact(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._navigate_mode("impact")

    def _load_detail_for_selected(self, action: str) -> None:
        hit = self._get_selected_hit()
        if hit is None:
            self._set_status("No hit selected. Run a search and select a result first.")
            return

        self._history.clear()
        self._open_target(self._hit_location(hit), action)

    def _open_target(self, target: SourceLocation, action: str) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
        self._active_target = target
        self._detail_mode = action
        self.query_one("#detail-tabs", Tabs).active = f"{action}-tab"
        self._detail_request_id += 1
        req_id = self._detail_request_id
        self._set_status(f"Loading {action} for {target.path}...")
        self._run_detail_worker(req_id, action, target, self.service.selected_project)

    @work(thread=True, exclusive=True, group="detail")
    def _run_detail_worker(
        self, request_id: int, action: str, hit: SourceLocation, project: ProjectInfo | None
    ) -> None:
        try:
            if action == "chunk":
                chunk = (
                    self.service.get_chunk(hit.chunk_id)
                    if hit.chunk_id
                    else self.service.source_preview(
                        hit.path,
                        hit.start_line,
                        end_line=hit.end_line,
                        project=project,
                        language=hit.language,
                        symbol=hit.symbol,
                    )
                )
                if request_id == self._detail_request_id:
                    self.call_from_thread(self._apply_detail, request_id, self._render_chunk, chunk)
            elif action == "outline":
                outline = self.service.file_outline(hit.path, project)
                if request_id == self._detail_request_id:
                    self.call_from_thread(
                        self._apply_detail, request_id, self._render_outline, outline
                    )
            elif action == "references":
                refs = self.service.find_references(
                    self._target_selector(hit, project), project=project
                )
                if request_id == self._detail_request_id:
                    self.call_from_thread(
                        self._apply_detail, request_id, self._render_references, refs
                    )
            elif action == "impact":
                impact = self.service.impact_radius(
                    self._target_selector(hit, project), project=project
                )
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

    def _target_selector(
        self, target: SourceLocation, project: ProjectInfo | None
    ) -> DeclarationSelector:
        if target.chunk_id:
            return DeclarationSelector(chunk_id=target.chunk_id)
        if target.symbol and project:
            return DeclarationSelector(
                project=project.id, path=target.path, qualified_symbol=target.symbol
            )
        raise CodeIndexingError(
            ErrorCode.INVALID_CONFIGURATION,
            "Choose a declaration in Outline before finding references or impact.",
        )

    def _apply_detail(self, request_id: int, render: Callable[..., None], value: Any) -> None:
        if request_id == self._detail_request_id:
            self._set_detail_entries([])
            render(value)
            self.query_one("#detail-scroll", VerticalScroll).scroll_home(animate=False)

    def _detail_error(self, request_id: int, message: str) -> None:
        if request_id == self._detail_request_id:
            self._show_error(message)

    def _render_chunk(self, chunk: CodeChunk | SourcePreview) -> None:
        prefix = "Working tree" if isinstance(chunk, SourcePreview) else "Preview"
        self.query_one("#detail-title", Label).update(
            f"{prefix}: {chunk.path}:{chunk.start_line}-{chunk.end_line} ({chunk.kind})"
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
        self._set_status(
            f"Showing {chunk.symbol or chunk.path} · lines {chunk.start_line}-{chunk.end_line}"
        )

    def _render_outline(self, outline: OutlineResponse) -> None:
        self.query_one("#detail-title", Label).update(
            f"Outline: {outline.path} ({len(outline.items)} items)"
        )
        text = Text()
        text.append(f"File Outline for {outline.path}\n\n", style="bold underline")
        if not outline.items:
            text.append("No outline symbols discovered in this file.", style="dim")
        entries: list[tuple[Text, SourceLocation]] = []
        for item in outline.items:
            text.append(f"  L{item.start_line:4d}-{item.end_line:4d} ", style="dim")
            text.append(f"{item.kind:<12} ", style="yellow")
            text.append(f"{item.qualified_symbol}\n", style="green bold")
            entries.append(
                (
                    Text(f"{item.qualified_symbol}  · {item.kind}  · L{item.start_line}"),
                    SourceLocation(
                        outline.path,
                        item.start_line,
                        item.end_line,
                        item.qualified_symbol,
                        self._active_target.language if self._active_target else "text",
                        item.kind,
                    ),
                )
            )

        self.query_one("#detail-content", Static).update(text)
        self._set_detail_entries(entries)
        if entries:
            self.query_one("#detail-content", Static).update(
                "Select a declaration and press Enter to open its source."
            )
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
        entries: list[tuple[Text, SourceLocation]] = []
        for hit in refs.hits:
            text.append(f"  {hit.path}:{hit.start_line} ", style="cyan bold")
            text.append(f"[{hit.resolution}] ", style="magenta")
            text.append(f"{hit.snippet.strip()}\n", style="white")
            entries.append(
                (
                    Text(f"{hit.path}:{hit.start_line}  [{hit.resolution}]\n{hit.snippet.strip()}"),
                    SourceLocation(hit.path, hit.start_line, hit.end_line, language=hit.language),
                )
            )

        if entries:
            text = Text("Select a reference and press Enter to open its source.\n")
        if refs.limitations:
            text.append("\nLimitations:\n", style="yellow bold")
            for lim in refs.limitations:
                text.append(f"  - [{lim.code}] {lim.explanation}\n", style="yellow")

        self.query_one("#detail-content", Static).update(text)
        self._set_detail_entries(entries)
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
        entries: list[tuple[Text, SourceLocation]] = []
        for layer in impact.layers:
            text.append(f"Depth {layer.depth}:\n", style="yellow bold")
            for edge in layer.edges:
                kinds = ", ".join(edge.kinds)
                text.append(f"  -> {edge.target.path} ", style="cyan")
                text.append(f"({kinds})\n", style="dim")
                target = edge.target
                entries.append(
                    (
                        Text(
                            f"Depth {layer.depth} · {target.qualified_symbol}\n"
                            f"{target.path}:{target.start_line} ({kinds})"
                        ),
                        SourceLocation(
                            target.path,
                            target.start_line,
                            target.end_line,
                            target.qualified_symbol,
                            target.language,
                            target.kind,
                            target.chunk_id,
                        ),
                    )
                )

        self.query_one("#detail-content", Static).update(text)
        self._set_detail_entries(entries)
        if entries:
            self.query_one("#detail-content", Static).update(
                "Select a dependent declaration and press Enter to open its source."
            )
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
