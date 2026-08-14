"""Project marker creation and resolution."""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from pathlib import Path
from uuid import UUID, uuid4

import tomli_w
from pydantic import ValidationError

from .errors import CodeIndexingError, ErrorCode
from .models import (
    DEFAULT_INCLUDES,
    LEGACY_DEFAULT_INCLUDES_V1,
    LEGACY_DEFAULT_INCLUDES_V2,
    LEGACY_DEFAULT_INCLUDES_V3,
    ProjectInfo,
    ScanConfig,
)

MARKER_DIRECTORY = ".ci-mcp"
LEGACY_MARKER_DIRECTORY = ".code-indexing-mcp"
MARKER_FILE = "project.toml"


def same_project_root(left: Path, right: Path) -> bool:
    """Return whether two path spellings identify the same project directory."""
    left = left.expanduser()
    right = right.expanduser()
    try:
        return left.samefile(right)
    except OSError:
        return left.resolve() == right.resolve()


def rooted_under(parent: Path, child: Path) -> bool:
    """Return whether *child* names a directory strictly inside *parent*.

    The boundary directory is compared with ``samefile`` rather than string
    equality, so differently-cased spellings of one directory (common on macOS
    and Windows) count as containment exactly like ``same_project_root``
    counts them as equality. Both paths must be resolved; a missing boundary
    directory means no containment.
    """
    if len(child.parts) <= len(parent.parts):
        return False
    boundary = Path(*child.parts[: len(parent.parts)])
    try:
        return boundary.samefile(parent)
    except OSError:
        return False


def project_root_identity(root: Path) -> str:
    """Return a cross-process identity for an existing project directory."""
    resolved = root.expanduser().resolve()
    try:
        info = resolved.stat()
    except OSError:
        return f"path:{resolved}"
    return f"inode:{info.st_dev}:{info.st_ino}"


def marker_path(root: Path) -> Path:
    return root / MARKER_DIRECTORY / MARKER_FILE


def legacy_marker_path(root: Path) -> Path:
    return root / LEGACY_MARKER_DIRECTORY / MARKER_FILE


def existing_marker_path(root: Path) -> Path | None:
    current = marker_path(root)
    if current.is_file():
        return current
    legacy = legacy_marker_path(root)
    if legacy.is_file():
        return legacy
    return None


def initialize_project(
    root: Path, *, name: str | None = None, force_new_id: bool = False
) -> ProjectInfo:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise CodeIndexingError(
            ErrorCode.PROJECT_NOT_FOUND, f"Project directory does not exist: {root}"
        )
    if existing_marker_path(root) is not None and not force_new_id:
        return read_project_marker(root)

    existing_path = marker_path(root)
    project = ProjectInfo(id=str(uuid4()), name=name or root.name, root=root)
    directory = existing_path.parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    data = {
        "version": project.version,
        "id": project.id,
        "name": project.name,
        "scan": project.scan.model_dump(),
    }
    existing_path.write_text(tomli_w.dumps(data), encoding="utf-8")
    return project


def read_project_marker(root: Path) -> ProjectInfo:
    root = root.expanduser().resolve()
    path = existing_marker_path(root) or marker_path(root)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError("unsupported marker version")
        UUID(str(raw["id"]))
        scan = ScanConfig.model_validate(raw.get("scan", {}))
        # A marker still carrying an older default include list is upgraded to the
        # current one, so a project written before a language was supported picks
        # it up. An include list the user has edited is left exactly as written.
        if scan.include in (
            LEGACY_DEFAULT_INCLUDES_V1,
            LEGACY_DEFAULT_INCLUDES_V2,
            LEGACY_DEFAULT_INCLUDES_V3,
        ):
            scan = scan.model_copy(update={"include": list(DEFAULT_INCLUDES)})
        return ProjectInfo(
            version=raw["version"],
            id=str(raw["id"]),
            name=str(raw["name"]),
            root=root,
            scan=scan,
        )
    except (OSError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise CodeIndexingError(
            ErrorCode.PROJECT_NOT_FOUND,
            f"Invalid or missing project marker: {path}",
            path=str(path),
        ) from exc


def find_project_root(start: Path) -> Path | None:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if existing_marker_path(candidate) is not None:
            return candidate
    return None


class ProjectResolver:
    def __init__(self, projects: Iterable[ProjectInfo]) -> None:
        self._projects = list(projects)

    def resolve(
        self,
        *,
        explicit: str | None = None,
        roots: Iterable[Path] = (),
        cwd: Path | None = None,
    ) -> ProjectInfo:
        if explicit:
            return self._resolve_explicit(explicit)

        roots = list(roots)
        marked = self._marked_projects(roots)
        if len(marked) == 1:
            return marked[0]
        if len(marked) > 1:
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_PROJECT,
                "Multiple MCP roots contain initialized projects",
                projects=[project.id for project in marked],
            )

        if cwd is not None and (root := find_project_root(cwd)) is not None:
            return self._by_root_or_marker(root)
        raise CodeIndexingError(
            ErrorCode.PROJECT_NOT_FOUND,
            "No active CodeIndexing project was detected; pass an explicit project id, name, or "
            "path, or run init_project for this directory",
            searched_roots=[str(root) for root in roots],
        )

    def _resolve_explicit(self, explicit: str) -> ProjectInfo:
        direct = [
            project
            for project in self._projects
            if project.id == explicit or project.name == explicit
        ]
        if len(direct) == 1:
            return direct[0]
        if len(direct) > 1:
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_PROJECT,
                f"Project name is ambiguous: {explicit}",
                projects=[project.id for project in direct],
            )
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            root = find_project_root(candidate)
            if root is not None:
                return self._by_root_or_marker(root)
        raise CodeIndexingError(ErrorCode.PROJECT_NOT_FOUND, f"Unknown project: {explicit}")

    def _marked_projects(self, roots: Iterable[Path]) -> list[ProjectInfo]:
        found: dict[str, ProjectInfo] = {}
        for candidate in roots:
            root = find_project_root(candidate)
            if root is not None:
                project = self._by_root_or_marker(root)
                found[project.id] = project
        return list(found.values())

    def _by_root_or_marker(self, root: Path) -> ProjectInfo:
        resolved = root.resolve()
        for project in self._projects:
            if same_project_root(project.root, resolved):
                return project
        return read_project_marker(resolved)
