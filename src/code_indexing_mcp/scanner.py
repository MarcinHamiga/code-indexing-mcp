"""Filesystem scanning with local ignore rules and safety constraints.

Two enumeration strategies share one classification pipeline:

- Inside a Git worktree the scanner asks Git for the truth up front --
  ``git ls-files --cached --others --exclude-standard`` returns every tracked
  file plus every untracked non-ignored file in one bounded stream. Ignore
  rules (``.gitignore``, ``info/exclude``, global excludes) are applied by Git
  itself, so the scanner never re-runs ``check-ignore`` on those candidates
  and never stats a file whose suffix or include pattern already excludes it.
  Tracked-but-ignored files stay eligible because Git's own rule is that the
  index wins. Submodules and nested repositories appear as single non-file
  entries and are not descended into.
- Outside Git, ``os.walk`` streams per-directory, sorted, with nested
  ``.gitignore`` files loaded incrementally and candidate batches bounded in
  memory. ``git check-ignore`` runs only on a rare fallback path (a worktree
  whose ``ls-files`` enumeration failed) and only over pre-filtered batches.
"""

from __future__ import annotations

import os
import select
import subprocess
import time
from collections.abc import Iterable, Iterator
from contextlib import suppress
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
    ".gdshader": "gdshader",
    ".gdshaderinc": "gdshader",
    ".tres": "godot_resource",
    ".tscn": "godot_resource",
    ".godot": "godot_resource",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".go": "go",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".lua": "lua",
}

HARD_EXCLUDED_DIRECTORIES = {
    ".git",
    ".ci-mcp",
    ".code-indexing-mcp",
    # `.godot` is both an extension this scanner indexes and the name of Godot's
    # own cache directory, which holds a generated copy of every imported asset.
    # Excluding the directory does not exclude a `project.godot` file: only
    # directory names are matched here.
    ".godot",
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

GIT_IGNORE_DISCOVERY_BATCH_SIZE = 256
GIT_CHECK_IGNORE_TIMEOUT_SECONDS = 10
GIT_LS_FILES_TIMEOUT_SECONDS = 10
SCAN_STREAM_READ_SIZE = 65536


class _GitEnumerationError(Exception):
    """Git enumeration failed or timed out; the walk path takes over."""


class SourceScanner:
    def has_supported_source(self, root: Path, scan: ScanConfig) -> bool:
        """Return whether *root* contains an eligible source file without reading it."""
        root = root.expanduser().resolve()
        config_excludes = GitIgnoreSpec.from_lines(scan.exclude)
        include_spec = GitIgnoreSpec.from_lines(scan.include)
        inherited_specs: dict[Path, list[tuple[Path, GitIgnoreSpec]]] = {root: []}
        eligible: list[Path] = []

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
                    if absolute.stat().st_size > scan.max_file_bytes:
                        continue
                except OSError:
                    continue
                eligible.append(absolute)
                if len(eligible) >= GIT_IGNORE_DISCOVERY_BATCH_SIZE:
                    batch, eligible = eligible, []
                    ignored = self._git_ignored_paths(root, batch)
                    if any(path.relative_to(root) not in ignored for path in batch):
                        return True
        ignored = self._git_ignored_paths(root, eligible)
        return any(path.relative_to(root) not in ignored for path in eligible)

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

        Git worktrees are enumerated with ``git ls-files``, which applies every
        ignore rule in one pass; everything else streams ``os.walk`` with
        incremental nested ignore rules. Either way only files whose suffix and
        include pattern already admit them are ever statted or passed to Git,
        and candidate batches stay bounded in memory.
        """
        known_files = known_files or {}
        root = project.root.resolve()
        config_excludes = GitIgnoreSpec.from_lines(project.scan.exclude)
        include_spec = GitIgnoreSpec.from_lines(project.scan.include)
        in_worktree = self._in_git_worktree(root)
        if in_worktree:
            try:
                for batch in self._iter_git_batches(root):
                    for item in self._prefilter_git_batch(batch, root, include_spec):
                        if isinstance(item, SkippedFile):
                            yield item
                            continue
                        yield from self._scan_candidates(
                            item,
                            root=root,
                            include_spec=include_spec,
                            config_excludes=config_excludes,
                            max_file_bytes=project.scan.max_file_bytes,
                            known_files=known_files,
                            read_contents=read_contents,
                            run_check_ignore=False,
                        )
                return
            except _GitEnumerationError:
                # A failed or timed-out git process must not silently produce
                # an empty index: the streaming walk covers the same tree.
                pass
        for walk_batch in self._iter_walk_batches(root, include_spec):
            if isinstance(walk_batch, SkippedFile):
                yield walk_batch
                continue
            yield from self._scan_candidates(
                walk_batch,
                root=root,
                include_spec=include_spec,
                config_excludes=config_excludes,
                max_file_bytes=project.scan.max_file_bytes,
                known_files=known_files,
                read_contents=read_contents,
                run_check_ignore=in_worktree,
            )

    def _prefilter_git_batch(
        self,
        batch: list[Path],
        root: Path,
        include_spec: GitIgnoreSpec,
    ) -> Iterator[SkippedFile | list[tuple[Path, list[tuple[Path, GitIgnoreSpec]]]]]:
        """Split one git-enumerated batch into recordable skips and candidates.

        Git already applied every ignore rule, but the scanner's own safety and
        include filters still apply before any stat: hard-excluded directories
        are dropped silently and unsupported suffixes are recorded as skips,
        so ``_classify`` (and its stat calls) never see them.
        """
        items: list[tuple[Path, list[tuple[Path, GitIgnoreSpec]]]] = []
        for absolute in batch:
            relative = absolute.relative_to(root)
            if SourceScanner._in_hard_excluded_directory(relative):
                continue
            if (
                LANGUAGES.get(absolute.suffix.lower()) is None
                or not include_spec.match_file(relative.as_posix())
            ):
                yield SkippedFile(path=relative, reason="unsupported")
                continue
            items.append((absolute, []))
        if items:
            yield items

    def _iter_walk_batches(
        self, root: Path, include_spec: GitIgnoreSpec
    ) -> Iterator[SkippedFile | list[tuple[Path, list[tuple[Path, GitIgnoreSpec]]]]]:
        """Stream the non-Git tree in deterministic per-directory order.

        Nested ``.gitignore`` files are loaded as the walk reaches their
        directory, so only the directories on the current path are ever held.
        Candidates are pre-filtered by suffix and include pattern before any
        stat or ignore work; batches stay bounded in memory.
        """
        inherited_specs: dict[Path, list[tuple[Path, GitIgnoreSpec]]] = {root: []}
        batch: list[tuple[Path, list[tuple[Path, GitIgnoreSpec]]]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            base = Path(dirpath)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in HARD_EXCLUDED_DIRECTORIES and not (base / name).is_symlink()
            )
            ignore_specs = inherited_specs.get(base, [])
            if ".gitignore" in filenames:
                ignore_specs = [
                    *ignore_specs,
                    *self._load_ignore_specs(root, [base / ".gitignore"]),
                ]
            for name in dirnames:
                inherited_specs[base / name] = ignore_specs

            for name in sorted(filenames):
                absolute = base / name
                relative = absolute.relative_to(root)
                if SourceScanner._in_hard_excluded_directory(relative):
                    continue
                if (
                    LANGUAGES.get(absolute.suffix.lower()) is None
                    or not include_spec.match_file(relative.as_posix())
                ):
                    yield SkippedFile(path=relative, reason="unsupported")
                    continue
                batch.append((absolute, ignore_specs))
                if len(batch) >= GIT_IGNORE_DISCOVERY_BATCH_SIZE:
                    yield batch
                    batch = []
        if batch:
            yield batch

    @staticmethod
    def _in_git_worktree(root: Path) -> bool:
        """Return whether *root* sits inside a Git worktree.

        A worktree may carry its repository as a ``.git`` file or directory,
        so the authoritative check is Git itself, not the presence of a
        directory. Outside any repository Git exits 128 and the scanner falls
        back to the streaming walk.
        """
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=GIT_CHECK_IGNORE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.strip() == b"true"

    @staticmethod
    def _iter_git_batches(root: Path) -> Iterator[list[Path]]:
        """Stream ``git ls-files`` output in bounded, sorted batches.

        Reads are chunked so candidate memory stays proportional to one batch,
        and the whole enumeration has a hard deadline so a hung git process
        cannot wedge a scan. A non-zero exit (which cannot normally happen for
        a worktree the probe just accepted) falls back to the walk.
        """
        try:
            process = subprocess.Popen(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise _GitEnumerationError(str(exc)) from exc
        assert process.stdout is not None
        batch: list[Path] = []
        buffer = bytearray()
        start = 0
        deadline = time.monotonic() + GIT_LS_FILES_TIMEOUT_SECONDS
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _GitEnumerationError("git ls-files timed out")
                readable, _, _ = select.select([process.stdout], [], [], remaining)
                if not readable:
                    continue
                chunk = process.stdout.read(SCAN_STREAM_READ_SIZE)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    separator = buffer.find(b"\0", start)
                    if separator < 0:
                        # Compact the consumed prefix once per chunk; without
                        # this the bytearray shift would be O(n) per path.
                        del buffer[:start]
                        start = 0
                        break
                    raw = bytes(buffer[start:separator])
                    start = separator + 1
                    if raw:
                        batch.append(root / os.fsdecode(raw))
                    if len(batch) >= GIT_IGNORE_DISCOVERY_BATCH_SIZE:
                        batch.sort()
                        yield batch
                        batch = []
            if process.wait() != 0:
                raise _GitEnumerationError(f"git ls-files exited with status {process.returncode}")
        except OSError as exc:
            raise _GitEnumerationError(str(exc)) from exc
        finally:
            # A consumer that stops early or an exception path must not leave
            # the process running; an already-exited process makes kill a no-op.
            with suppress(ProcessLookupError):
                process.kill()
            process.stdout.close()
            process.wait()
        if batch:
            batch.sort()
            yield batch

    def _scan_candidates(
        self,
        items: Iterable[tuple[Path, list[tuple[Path, GitIgnoreSpec]]]],
        *,
        root: Path,
        include_spec: GitIgnoreSpec,
        config_excludes: GitIgnoreSpec,
        max_file_bytes: int,
        known_files: dict[str, StoredFile],
        read_contents: bool,
        run_check_ignore: bool,
    ) -> Iterator[ScannedFile | SkippedFile]:
        """Classify, stat, and optionally read one pre-filtered candidate batch.

        ``run_check_ignore`` is the rare walk fallback inside a worktree whose
        ``ls-files`` enumeration failed; regular walks outside Git skip the
        subprocess entirely because there is nothing for it to apply.
        """
        ignored_paths: set[Path] = set()
        if run_check_ignore and items:
            ignored_paths = self._git_ignored_paths(root, [absolute for absolute, _ in items])
        for absolute, ignore_specs in items:
            relative = absolute.relative_to(root)
            language, skip_reason = self._classify(
                relative,
                absolute,
                include_spec=include_spec,
                config_excludes=config_excludes,
                ignore_specs=ignore_specs,
                standard_ignored=relative in ignored_paths,
            )
            if language is None:
                if skip_reason is not None:
                    yield SkippedFile(path=relative, reason=skip_reason)
                continue
            try:
                stat = absolute.stat()
                if stat.st_size > max_file_bytes:
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
    def _classify(
        relative: Path,
        absolute: Path,
        *,
        include_spec: GitIgnoreSpec,
        config_excludes: GitIgnoreSpec,
        ignore_specs: list[tuple[Path, GitIgnoreSpec]],
        standard_ignored: bool = False,
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
        if (
            standard_ignored
            or config_excludes.match_file(relative.as_posix())
            or SourceScanner._is_ignored(relative, ignore_specs)
        ):
            return None, "ignored"
        return language, None

    @staticmethod
    def _git_ignored_paths(root: Path, candidates: list[Path]) -> set[Path]:
        """Return paths ignored by Git's complete standard exclude stack.

        ``git check-ignore`` applies nested ``.gitignore`` files, the repository's
        ``info/exclude``, and the user's configured global excludes in one batch.
        Non-Git projects and environments without Git keep using the in-process
        ``.gitignore`` fallback loaded by :meth:`_load_ignore_specs`.
        """
        relative = [path.relative_to(root) for path in candidates]
        if not relative:
            return set()
        payload = b"\0".join(os.fsencode(path.as_posix()) for path in relative) + b"\0"
        try:
            result = subprocess.run(
                ["git", "-C", str(root), "check-ignore", "--no-index", "--stdin", "-z"],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=GIT_CHECK_IGNORE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return set()
        if result.returncode not in {0, 1}:
            return set()
        return {Path(os.fsdecode(path)) for path in result.stdout.split(b"\0") if path}

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
