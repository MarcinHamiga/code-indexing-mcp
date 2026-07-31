"""Filesystem scanning with local ignore rules and safety constraints."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from pathspec import GitIgnoreSpec

from .models import ProjectInfo, ScanConfig, ScannedFile, ScanResult, SkippedFile, StoredFile

LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".cs": "csharp",
    ".csx": "csharp",
    ".gd": "gdscript",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

HARD_EXCLUDED_DIRECTORIES = {
    ".git",
    ".ci-mcp",
    ".incode",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "coverage",
    "htmlcov",
}


class SourceScanner:
    def has_supported_source(self, root: Path, scan: ScanConfig) -> bool:
        """Return whether *root* contains an eligible source file without reading it."""
        root = root.expanduser().resolve()
        config_excludes = GitIgnoreSpec.from_lines(scan.exclude)
        include_spec = GitIgnoreSpec.from_lines(scan.include)
        inherited_specs: dict[Path, list[tuple[Path, GitIgnoreSpec]]] = {root: []}

        for dirpath, dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in HARD_EXCLUDED_DIRECTORIES and not (base / name).is_symlink()
            )
            ignore_specs = inherited_specs.get(base, [])
            gitignore = base / ".gitignore"
            if ".gitignore" in filenames:
                ignore_specs = [
                    *ignore_specs,
                    *self._load_ignore_specs(root, [gitignore]),
                ]
            for name in dirnames:
                inherited_specs[base / name] = ignore_specs

            for name in sorted(filenames):
                absolute = base / name
                relative = absolute.relative_to(root)
                language, _ = self._classify(
                    relative,
                    absolute,
                    include_spec=include_spec,
                    config_excludes=config_excludes,
                    ignore_specs=ignore_specs,
                )
                if language is None:
                    continue
                try:
                    if absolute.stat().st_size <= scan.max_file_bytes:
                        return True
                except OSError:
                    continue
        return False

    def scan(
        self, project: ProjectInfo, known_files: dict[str, StoredFile] | None = None
    ) -> ScanResult:
        """Collect stat-only scan results without retaining or reading source bytes."""
        files: list[ScannedFile] = []
        skipped: list[SkippedFile] = []
        for item in self.iter_scan(project, known_files, read_contents=False):
            if isinstance(item, ScannedFile):
                files.append(item)
            else:
                skipped.append(item)
        return ScanResult(files=files, skipped=skipped)

    def iter_scan(
        self,
        project: ProjectInfo,
        known_files: dict[str, StoredFile] | None = None,
        *,
        read_contents: bool = True,
    ) -> Iterator[ScannedFile | SkippedFile]:
        """Yield scan results one file at a time.

        When *read_contents* is true, changed files carry their raw source bytes
        so the indexer never reads a file twice. Binary and encoding validation
        belongs to the indexer, where those bytes are already consumed. The
        bytes die with the yielded item, so at most one file's source is live at
        any moment.
        """
        known_files = known_files or {}
        root = project.root.resolve()
        config_excludes = GitIgnoreSpec.from_lines(project.scan.exclude)
        include_spec = GitIgnoreSpec.from_lines(project.scan.include)
        candidates, gitignores = self._walk(root)
        ignore_specs = self._load_ignore_specs(root, gitignores)

        for absolute in candidates:
            relative = absolute.relative_to(root)
            language, skip_reason = self._classify(
                relative,
                absolute,
                include_spec=include_spec,
                config_excludes=config_excludes,
                ignore_specs=ignore_specs,
            )
            if language is None:
                if skip_reason is not None:
                    yield SkippedFile(path=relative, reason=skip_reason)
                continue
            try:
                stat = absolute.stat()
                if stat.st_size > project.scan.max_file_bytes:
                    yield SkippedFile(path=relative, reason="oversized")
                    continue
            except OSError as exc:
                yield SkippedFile(path=relative, reason="unreadable", detail=str(exc))
                continue
            previous = known_files.get(relative.as_posix())
            content: bytes | None = None
            if read_contents and (
                previous is None
                or previous.size != stat.st_size
                or previous.mtime_ns != stat.st_mtime_ns
            ):
                try:
                    content = absolute.read_bytes()
                except OSError as exc:
                    yield SkippedFile(path=relative, reason="unreadable", detail=str(exc))
                    continue
            yield ScannedFile(
                path=relative,
                absolute_path=absolute,
                language=language,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                content=content,
            )

    @staticmethod
    def _walk(root: Path) -> tuple[list[Path], list[Path]]:
        """Collect candidate files and .gitignore files without descending into
        hard-excluded or symlinked directories."""
        candidates: list[Path] = []
        gitignores: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in HARD_EXCLUDED_DIRECTORIES and not (base / name).is_symlink()
            )
            for name in filenames:
                path = base / name
                candidates.append(path)
                if name == ".gitignore":
                    gitignores.append(path)
        candidates.sort()
        gitignores.sort()
        return candidates, gitignores

    @staticmethod
    def _classify(
        relative: Path,
        absolute: Path,
        *,
        include_spec: GitIgnoreSpec,
        config_excludes: GitIgnoreSpec,
        ignore_specs: list[tuple[Path, GitIgnoreSpec]],
    ) -> tuple[str | None, str | None]:
        """Decide whether a candidate file is eligible for indexing.

        Returns ``(language, skip_reason)``. ``language`` is set only when the
        file passes every path-based eligibility check (not in a hard-excluded
        directory, not a symlink, a regular file, a supported suffix that
        matches the include spec, and not matched by config excludes or
        gitignore rules); callers still need to apply their own size/content
        checks on top. ``skip_reason`` carries the reason string ``scan``
        records as a :class:`SkippedFile` (``"symlink"``, ``"unsupported"``,
        or ``"ignored"``); it is ``None`` for rejections ``scan`` does not
        record (hard-excluded directories, non-files, and symlinks whose
        suffix is not supported to begin with).
        """
        if SourceScanner._in_hard_excluded_directory(relative):
            return None, None
        if absolute.is_symlink():
            if absolute.suffix.lower() in LANGUAGES:
                return None, "symlink"
            return None, None
        if not absolute.is_file():
            return None, None
        language = LANGUAGES.get(absolute.suffix.lower())
        if language is None or not include_spec.match_file(relative.as_posix()):
            return None, "unsupported"
        if config_excludes.match_file(relative.as_posix()) or SourceScanner._is_ignored(
            relative, ignore_specs
        ):
            return None, "ignored"
        return language, None

    @staticmethod
    def _in_hard_excluded_directory(path: Path) -> bool:
        return any(part in HARD_EXCLUDED_DIRECTORIES for part in path.parts[:-1])

    @staticmethod
    def _load_ignore_specs(root: Path, gitignores: list[Path]) -> list[tuple[Path, GitIgnoreSpec]]:
        specs: list[tuple[Path, GitIgnoreSpec]] = []
        for path in gitignores:
            relative = path.relative_to(root)
            if SourceScanner._in_hard_excluded_directory(relative):
                continue
            try:
                specs.append(
                    (
                        path.parent.relative_to(root),
                        GitIgnoreSpec.from_lines(path.read_text(encoding="utf-8").splitlines()),
                    )
                )
            except (OSError, UnicodeDecodeError):
                continue
        return specs

    @staticmethod
    def _is_ignored(path: Path, specs: list[tuple[Path, GitIgnoreSpec]]) -> bool:
        ignored = False
        for base, spec in specs:
            try:
                candidate = path if base == Path(".") else path.relative_to(base)
                result = spec.check_file(candidate.as_posix())
                if result.include is not None:
                    ignored = result.include
            except ValueError:
                continue
        return ignored
