"""Small immutable values for TUI source destinations and previews."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLocation:
    path: str
    start_line: int
    end_line: int | None = None
    symbol: str | None = None
    language: str = "text"
    kind: str = "source"
    chunk_id: str | None = None


@dataclass(frozen=True)
class SourcePreview:
    path: str
    start_line: int
    end_line: int
    content: str
    language: str = "text"
    symbol: str | None = None
    kind: str = "source"
