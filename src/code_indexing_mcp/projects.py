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
    _write_marker(existing_path, project)
    return project


def initialize_checkout(
    root: Path, registration: ProjectInfo, *, name: str | None = None
) -> ProjectInfo:
    """Write this checkout's marker carrying an existing registration's identity.

    A linked worktree joins its repository's shared project rather than
    minting a new id: the marker adopts the registration's id and scan
    configuration, while the returned ProjectInfo stays checkout-local so
    probes and scans observe this checkout. Passing *name* renames the whole
    shared registration.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise CodeIndexingError(
            ErrorCode.PROJECT_NOT_FOUND, f"Project directory does not exist: {root}"
        )
    project = ProjectInfo(
        version=registration.version,
        id=registration.id,
        name=name or registration.name,
        root=root,
        scan=registration.scan,
    )
    _write_marker(marker_path(root), project)
    return project


def _write_marker(path: Path, project: ProjectInfo) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    (path.parent / ".gitignore").write_text("*\n", encoding="utf-8")
    data = {
        "version": project.version,
        "id": project.id,
        "name": project.name,
        "scan": project.scan.model_dump(),
    }
    path.write_text(tomli_w.dumps(data), encoding="utf-8")


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
        """Resolve exactly one project or raise.

        See ``resolve_scope`` for the rules; this method additionally raises
        ``AMBIGUOUS_PROJECT`` when a request's roots bind several distinct
        projects at once.
        """
        scope = self.resolve_scope(explicit=explicit, roots=roots, cwd=cwd)
        if len(scope) > 1:
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_PROJECT,
                "Multiple MCP roots contain initialized projects",
                projects=[project.id for project in scope],
            )
        return scope[0]

    def resolve_scope(
        self,
        *,
        explicit: str | None = None,
        roots: Iterable[Path] = (),
        cwd: Path | None = None,
    ) -> list[ProjectInfo]:
        """Resolve the checkouts a request is scoped to.

        Checkouts of one repository share a single registration: each entry
        of the result carries the checkout root the request arrived through
        when one exists (the marker under an MCP root or *cwd*), and they may
        repeat a project id once per live checkout so callers can search
        every requested branch slot together.
        """
        if explicit:
            project = self._resolve_explicit(explicit)
            return [self._bind_checkout(project, roots=roots, cwd=cwd)]

        marked = self._marked_checkouts(roots)
        if not marked and cwd is not None and (root := find_project_root(cwd)) is not None:
            marked = [self._by_root_or_marker(root)]
        ids = {project.id for project in marked}
        if len(ids) > 1:
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_PROJECT,
                "Multiple MCP roots contain initialized projects",
                projects=[project.id for project in self._dedupe_by_id(marked)],
            )
        if marked:
            return marked
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

    def _bind_checkout(
        self,
        project: ProjectInfo,
        *,
        roots: Iterable[Path],
        cwd: Path | None,
    ) -> ProjectInfo:
        """Re-root *project* at the request's own checkout when one matches.

        Explicit id and name selection resolves to the registered payload
        whose root is the canonical checkout. When the request itself arrives
        from another checkout of the same shared registration -- a linked
        worktree's marker with the same id -- that local marker wins so probes
        and scans observe the requesting worktree rather than the canonical
        root.
        """
        candidates: list[Path] = []
        for root in roots:
            candidates.append(Path(root))
        if cwd is not None:
            candidates.append(Path(cwd))
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                marker_root = find_project_root(candidate.expanduser())
            except OSError:
                continue
            if marker_root is None or marker_root in seen:
                continue
            seen.add(marker_root)
            try:
                marker = read_project_marker(marker_root)
            except CodeIndexingError:
                continue
            if marker.id == project.id and not same_project_root(marker.root, project.root):
                return marker
        return project

    @staticmethod
    def _dedupe_by_id(projects: list[ProjectInfo]) -> list[ProjectInfo]:
        unique: dict[str, ProjectInfo] = {}
        for project in projects:
            unique.setdefault(project.id, project)
        return list(unique.values())

    def _marked_projects(self, roots: Iterable[Path]) -> list[ProjectInfo]:
        """Return each distinct registered project found under *roots*."""
        return self._dedupe_by_id(self._marked_checkouts(roots))

    def _marked_checkouts(self, roots: Iterable[Path]) -> list[ProjectInfo]:
        """Return every marked checkout under *roots*, one entry per checkout.

        Two roots can legally carry markers with the same id -- that is one
        shared registration observed through two of its checkouts -- so only
        exact (id, root) duplicates collapse here.
        """
        found: list[ProjectInfo] = []
        seen: set[tuple[str, str]] = set()
        for candidate in roots:
            root = find_project_root(candidate)
            if root is None:
                continue
            project = self._by_root_or_marker(root)
            key = (project.id, project_root_identity(root))
            if key in seen:
                continue
            seen.add(key)
            found.append(project)
        return found

    def _by_root_or_marker(self, root: Path) -> ProjectInfo:
        resolved = root.resolve()
        for project in self._projects:
            if same_project_root(project.root, resolved):
                return project
        return read_project_marker(resolved)
