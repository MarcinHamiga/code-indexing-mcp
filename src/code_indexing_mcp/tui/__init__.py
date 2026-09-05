"""Terminal user interface package for Code Indexing MCP."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any


def _launch_tui(project: str | None) -> int:
    """Open the interactive TUI, optionally preselecting one project."""
    # Lazily import Textual and TUI components so standard CLI / MCP server
    # never pay for Textual import overhead.
    from ..errors import CodeIndexingError
    from .app import CodeIndexingApp
    from .service import create_tui_service

    service = create_tui_service()
    if project:
        try:
            service.select_project(project)
        except CodeIndexingError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    app = CodeIndexingApp(service=service)
    result = app.run()
    return result if isinstance(result, int) else (app.return_code or 0)


def _is_command(name: str) -> bool:
    """Whether *name* is a top-level CLI subcommand.

    Imports the CLI lazily so the TUI fast path never pays for it unless
    disambiguation actually needs the command list.
    """
    from ..cli import COMMAND_NAMES

    return name in COMMAND_NAMES


def main(argv: Sequence[str] | None = None) -> int:
    """Run the syndex command: the full CLI, defaulting to the TUI.

    Bare ``syndex`` (or ``syndex <project>``) keeps its legacy meaning and
    opens the interactive TUI. Anything starting with a known subcommand name
    or a flag is the full ``code-indexing-mcp`` command surface under the
    syndex program name.

    A project whose name collides with a subcommand (``status``, ``index``,
    ...) cannot use the bare shorthand — ``syndex status`` always runs the
    command. Open such a project explicitly with ``syndex tui <project>``.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return _launch_tui(None)
    first = raw[0]
    if first == "tui" and len(raw) <= 2 and all(not arg.startswith("-") for arg in raw[1:]):
        return _launch_tui(raw[1] if len(raw) == 2 else None)
    if len(raw) == 1 and not first.startswith("-") and not _is_command(first):
        return _launch_tui(first)
    from ..cli import main as cli_main

    return cli_main(raw, prog="syndex")


def __getattr__(name: str) -> Any:
    """Lazily re-export the TUI components advertised in ``__all__``.

    Keeps ``from code_indexing_mcp.tui import CodeIndexingApp`` working
    without making every ``import code_indexing_mcp.tui`` pay for Textual.
    """
    if name == "CodeIndexingApp":
        from .app import CodeIndexingApp

        return CodeIndexingApp
    if name == "create_tui_service":
        from .service import create_tui_service

        return create_tui_service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CodeIndexingApp", "create_tui_service", "main"]
