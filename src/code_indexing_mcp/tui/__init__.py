"""Terminal user interface package for Code Indexing MCP."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the terminal user interface."""
    parser = argparse.ArgumentParser(
        prog="syndex",
        description="Terminal user interface for Code Indexing MCP",
    )
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Project name, id, or path to open in the TUI",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # Lazily import Textual and TUI components so standard CLI / MCP server
    # never pay for Textual import overhead.
    from ..errors import CodeIndexingError
    from .app import CodeIndexingApp
    from .service import create_tui_service

    service = create_tui_service()
    if args.project:
        try:
            service.select_project(args.project)
        except CodeIndexingError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    app = CodeIndexingApp(service=service)
    result = app.run()
    return result if isinstance(result, int) else (app.return_code or 0)


__all__ = ["CodeIndexingApp", "create_tui_service", "main"]
