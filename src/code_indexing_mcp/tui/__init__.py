"""Terminal user interface package for Code Indexing MCP."""

from __future__ import annotations

import sys
from collections.abc import Sequence


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the syndex command: the full CLI, defaulting to the TUI.

    Bare ``syndex`` (or ``syndex <project>``) keeps its legacy meaning and
    opens the interactive TUI. Anything starting with a known subcommand name
    or a flag is the full ``code-indexing-mcp`` command surface under the
    syndex program name.
    """
    from ..cli import COMMAND_NAMES
    from ..cli import main as cli_main

    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        return _launch_tui(None)
    first = raw[0]
    if len(raw) == 1 and not first.startswith("-") and first not in COMMAND_NAMES:
        return _launch_tui(first)
    return cli_main(raw, prog="syndex")


__all__ = ["CodeIndexingApp", "create_tui_service", "main"]
