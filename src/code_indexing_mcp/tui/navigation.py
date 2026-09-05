"""Small immutable values for TUI source destinations and previews."""

import shlex
from dataclasses import dataclass
from pathlib import Path


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


def editor_command(editor: str, path: Path, line: int) -> list[str]:
    """Build argv for common editors; never interpret shell metacharacters."""
    args = shlex.split(editor)
    if not args:
        raise ValueError("Set VISUAL or EDITOR to your editor command first.")
    name = Path(args[0]).name.lower()
    if name in {"code", "code-insiders", "codium", "cursor"}:
        return [*args, "--goto", f"{path}:{line}"]
    if name in {"vi", "vim", "nvim", "nano", "emacs", "emacsclient"}:
        return [*args, f"+{line}", str(path)]
    return [*args, str(path)]
