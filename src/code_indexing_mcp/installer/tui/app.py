"""The Textual installer wizard application."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Button, ContentSwitcher, Footer, Header

from ..orchestrator import InstallResult
from ..wizard import WizardState
from .panels import (
    AcceleratorPanel,
    DonePanel,
    HarnessesPanel,
    LocationPanel,
    PathPanel,
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
    "path",
    "indexing",
    "embedding",
    "summary",
    "progress",
    "done",
)

# What the header calls each step. The two panels the user cannot navigate away
# from are deliberately absent from the step count below.
PANEL_TITLES = {
    "welcome": "Welcome",
    "location": "Install location",
    "accelerator": "Accelerator",
    "harnesses": "MCP clients",
    "path": "Command-line access",
    "indexing": "Indexing settings",
    "embedding": "Embedding settings",
    "summary": "Summary",
    "progress": "Installing",
    "done": "Done",
}


class InstallerApp(App[None]):
    TITLE = "Code Indexing MCP Installer"
    # Chords rather than bare letters: the settings panels are full of Input
    # widgets, where a plain "n" belongs to whatever the user is typing.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+n", "next", "Next", show=True),
        Binding("ctrl+b", "previous", "Back", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
    ]
    CSS = """
    #screens { height: 1fr; }
    .panel { padding: 1 2; height: 1fr; overflow-y: auto; }
    .help { color: $text-muted; }
    .field { height: auto; margin-bottom: 1; }
    .error { color: $error; }
    #nav { height: auto; dock: bottom; padding: 0 1; }
    #nav Button { margin: 0 1; }
    /* Docked so the retry/continue buttons stay reachable however long the
       detail log grows. */
    #progress-buttons { dock: bottom; height: auto; }
    #progress-buttons Button { margin: 0 1 0 0; }
    /* Docked so a long summary never pushes the jump-back row off-screen. */
    #summary-jumps { dock: bottom; height: auto; }
    #summary-jumps Button { margin: 0 1 0 0; min-width: 0; }
    #progress-steps { margin: 1 0; }
    #progress-log { height: 12; }
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
            yield PathPanel(self.state, id="path")
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

    def on_mount(self) -> None:
        # ContentSwitcher's `initial` shows the first panel without going through
        # show_panel, so the header and focus would start out unset.
        self.show_panel(self._order()[0])

    def _order(self) -> tuple[str, ...]:
        if self.state.mode == "reconfigure":
            return tuple(panel for panel in PANEL_ORDER if panel != "location")
        return PANEL_ORDER

    @property
    def current(self) -> str:
        return str(self.query_one("#screens", ContentSwitcher).current)

    @property
    def locked(self) -> bool:
        """True on the panels the wizard drives itself; navigation is not the user's."""

        return self.current in {"progress", "done"}

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
        # Steps are counted over the panels the user actually walks, so the
        # count does not jump when the wizard takes over at the end.
        walked = [panel for panel in order if panel not in {"progress", "done"}]
        title = PANEL_TITLES.get(name, name)
        if name in walked:
            self.sub_title = f"Step {walked.index(name) + 1} of {len(walked)} - {title}"
        else:
            self.sub_title = title
        panel = self.query_one(f"#{name}")
        became_visible = getattr(panel, "on_became_visible", None)
        if became_visible is not None:
            became_visible()
        self._focus_first_control(panel)

    @staticmethod
    def _focus_first_control(panel: Widget) -> None:
        """Land focus on the panel's first control, if it has one.

        A panel of prose (welcome, summary) has nothing to focus, and leaving
        focus on the Next button there is exactly right. Widgets inside a
        collapsed Collapsible are not focusable, so "Advanced" stays advanced.
        """

        for widget in panel.query(Widget):
            if widget.focusable:
                widget.focus()
                return

    def action_next(self) -> None:
        if not self.locked:
            self.advance()

    def action_previous(self) -> None:
        if self.locked:
            return
        order = self._order()
        index = order.index(self.current)
        if index > 0:
            self.show_panel(order[index - 1])

    def action_cancel(self) -> None:
        if not self.locked:
            self.exit(return_code=130)

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
            self.action_previous()
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
