"""The Textual installer wizard application."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, ContentSwitcher, Footer, Header

from ..orchestrator import InstallResult
from ..wizard import WizardState
from .panels import (
    AcceleratorPanel,
    DonePanel,
    HarnessesPanel,
    LocationPanel,
    ProgressPanel,
    SettingsPanel,
    SummaryPanel,
    WelcomePanel,
)

PANEL_ORDER = (
    "welcome",
    "location",
    "accelerator",
    "harnesses",
    "indexing",
    "embedding",
    "summary",
    "progress",
    "done",
)


class InstallerApp(App[None]):
    TITLE = "Code Indexing MCP Installer"
    CSS = """
    #screens { height: 1fr; }
    .panel { padding: 1 2; height: 1fr; overflow-y: auto; }
    .help { color: $text-muted; }
    .field { height: auto; margin-bottom: 1; }
    .error { color: $error; }
    #nav { height: auto; dock: bottom; padding: 0 1; }
    #nav Button { margin: 0 1; }
    """

    def __init__(self, state: WizardState) -> None:
        super().__init__()
        self.state = state
        # None while the wizard is incomplete; the CLI maps that to 130.
        self.done_code: int | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with ContentSwitcher(id="screens", initial="welcome"):
            yield WelcomePanel(self.state, id="welcome")
            yield LocationPanel(self.state, id="location")
            yield AcceleratorPanel(self.state, id="accelerator")
            yield HarnessesPanel(self.state, id="harnesses")
            yield SettingsPanel(self.state, "Indexing", id="indexing")
            yield SettingsPanel(self.state, "Embedding", id="embedding")
            yield SummaryPanel(self.state, id="summary")
            yield ProgressPanel(self.state, id="progress")
            yield DonePanel(id="done")
        with Horizontal(id="nav"):
            yield Button("Back", id="back", disabled=True)
            yield Button("Next", id="next", variant="primary")
            yield Button("Cancel", id="cancel", variant="error")
        yield Footer()

    def _order(self) -> tuple[str, ...]:
        if self.state.mode == "reconfigure":
            return tuple(panel for panel in PANEL_ORDER if panel != "location")
        return PANEL_ORDER

    @property
    def current(self) -> str:
        return str(self.query_one("#screens", ContentSwitcher).current)

    def show_panel(self, name: str) -> None:
        self.query_one("#screens", ContentSwitcher).current = name
        order = self._order()
        index = order.index(name)
        locked = name in {"progress", "done"}
        self.query_one("#back", Button).disabled = locked or index == 0
        next_button = self.query_one("#next", Button)
        next_button.disabled = locked
        next_button.label = "Confirm" if name == "summary" else "Next"
        self.query_one("#cancel", Button).disabled = locked
        panel = self.query_one(f"#{name}")
        became_visible = getattr(panel, "on_became_visible", None)
        if became_visible is not None:
            became_visible()

    def advance(self) -> None:
        panel = self.query_one(f"#{self.current}")
        commit = getattr(panel, "commit", None)
        if commit is not None and not commit():
            return  # validation failed; the panel displayed the reason
        if self.current == "summary":
            self.show_panel("progress")
            self.query_one("#progress", ProgressPanel).start()
            return
        self.show_panel(self._order()[self._order().index(self.current) + 1])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "next":
            self.advance()
        elif button == "back":
            order = self._order()
            self.show_panel(order[order.index(self.current) - 1])
        elif button == "cancel":
            self.exit(return_code=130)
        elif button == "exit":
            self.exit(return_code=self.done_code if self.done_code is not None else 0)

    def finish(
        self,
        result: InstallResult | None,
        *,
        error: Exception | None = None,
        cancelled: bool = False,
    ) -> None:
        if cancelled:
            self.done_code = 130
        elif error is not None or result is None or result.failures:
            self.done_code = 1
        else:
            self.done_code = 0
        self.query_one("#done", DonePanel).show_result(result, error=error, cancelled=cancelled)
        self.show_panel("done")
