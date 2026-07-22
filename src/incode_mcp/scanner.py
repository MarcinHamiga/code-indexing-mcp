"""Filesystem scanning with local ignore rules and safety constraints."""

from __future__ import annotations

import os
from pathlib import Path

from pathspec import GitIgnoreSpec

from .models import ProjectInfo, ScannedFile, ScanResult, SkippedFile, StoredFile

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
}

HARD_EXCLUDED_DIRECTORIES = {
    ".git",
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
    def scan(
        self, project: ProjectInfo, known_files: dict[str, StoredFile] | None = None
    ) -> ScanResult:
        root = project.root.resolve()
        known_files = known_files or {}
        config_excludes = GitIgnoreSpec.from_lines(project.scan.exclude)
        include_spec = GitIgnoreSpec.from_lines(project.scan.include)
        candidates, gitignores = self._walk(root)
        ignore_specs = self._load_ignore_specs(root, gitignores)
        files: list[ScannedFile] = []
        skipped: list[SkippedFile] = []

        for absolute in candidates:
            relative = absolute.relative_to(root)
            if self._in_hard_excluded_directory(relative):
                continue
            if absolute.is_symlink():
                if absolute.suffix.lower() in LANGUAGES:
                    skipped.append(SkippedFile(path=relative, reason="symlink"))
                continue
            if not absolute.is_file():
                continue
            language = LANGUAGES.get(absolute.suffix.lower())
            if language is None or not include_spec.match_file(relative.as_posix()):
                skipped.append(SkippedFile(path=relative, reason="unsupported"))
                continue
            if config_excludes.match_file(relative.as_posix()) or self._is_ignored(
                relative, ignore_specs
            ):
                skipped.append(SkippedFile(path=relative, reason="ignored"))
                continue
            try:
                stat = absolute.stat()
                if stat.st_size > project.scan.max_file_bytes:
                    skipped.append(SkippedFile(path=relative, reason="oversized"))
                    continue
            except OSError as exc:
                skipped.append(SkippedFile(path=relative, reason="unreadable", detail=str(exc)))
                continue
            previous = known_files.get(relative.as_posix())
            content: bytes | None = None
            if (
                previous is None
                or previous.size != stat.st_size
                or previous.mtime_ns != stat.st_mtime_ns
            ):
                try:
                    content = absolute.read_bytes()
                except OSError as exc:
                    skipped.append(SkippedFile(path=relative, reason="unreadable", detail=str(exc)))
                    continue
                if b"\x00" in content:
                    skipped.append(SkippedFile(path=relative, reason="binary"))
                    continue
                try:
                    content.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    skipped.append(SkippedFile(path=relative, reason="encoding", detail=str(exc)))
                    continue
            files.append(
                ScannedFile(
                    path=relative,
                    absolute_path=absolute,
                    language=language,
                    size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    content=content,
                )
            )
        return ScanResult(files=files, skipped=skipped)

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
