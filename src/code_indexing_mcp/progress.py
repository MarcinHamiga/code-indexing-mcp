"""Live progress for an indexing run, in process and across processes.

An index can run for minutes, and the caller who has to wait it out is rarely
the process running it: the MCP server usually delegates to the per-user daemon,
and the CLI can be watching a run someone else started. So progress is both
handed to an in-process listener and published as a small JSON snapshot under
``<data>/progress/<project-id>.json`` that any process can read.

The snapshot is a status file, not a queue: every update replaces it, readers
only ever see the latest state, and a stale file left behind by a killed process
is ignored once it stops being refreshed.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

# How long a snapshot stays trustworthy without a refresh. Comfortably above the
# publish interval, and above the pauses a single slow file can cause between
# updates, so a live run is never mistaken for a dead one.
STALE_AFTER_SECONDS = 60.0
PUBLISH_INTERVAL_SECONDS = 0.25


class IndexProgress(BaseModel):
    """A point-in-time snapshot of one project's indexing run."""

    project_id: str
    phase: str = "scanning"
    files_seen: int = 0
    # None until a run knows better than to guess: the scanner streams, so on a
    # first index there is no total to compare against.
    files_total: int | None = None
    files_indexed: int = 0
    files_unchanged: int = 0
    chunks_embedded: int = 0
    current_path: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    pid: int = Field(default_factory=os.getpid)

    @property
    def fraction(self) -> float | None:
        """Completion in ``[0, 1]``, or None when the total is unknown."""

        if not self.files_total:
            return None
        return min(1.0, self.files_seen / self.files_total)

    def describe(self) -> str:
        """Render a one-line status suitable for a progress bar or a log line."""

        if self.phase == "committing":
            return "Committing the index"
        if not self.files_seen:
            return "Scanning for changed files"
        if self.files_total:
            scanned = f"{self.files_seen}/~{self.files_total} files"
        else:
            scanned = f"{self.files_seen} files"
        parts = [f"{self.phase.capitalize()} {scanned}"]
        if self.files_indexed:
            parts.append(f"{self.files_indexed} indexed")
        if self.files_unchanged:
            parts.append(f"{self.files_unchanged} unchanged")
        if self.chunks_embedded:
            parts.append(f"{self.chunks_embedded} chunks embedded")
        return ", ".join(parts)


def progress_path(directory: Path, project_id: str) -> Path:
    return directory / f"{project_id}.json"


class ProgressPublisher:
    """Throttled writer for one run's snapshot file and in-process listener."""

    def __init__(
        self,
        project_id: str,
        *,
        directory: Path | None = None,
        listener: Callable[[IndexProgress], None] | None = None,
        interval_seconds: float = PUBLISH_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = directory
        self.listener = listener
        self.interval_seconds = interval_seconds
        self._clock = clock
        self._last_published: float | None = None
        self.state = IndexProgress(project_id=project_id, started_at=time.time())

    @property
    def enabled(self) -> bool:
        return self.directory is not None or self.listener is not None

    def update(self, *, force: bool = False, **fields: object) -> None:
        """Merge *fields* into the snapshot, publishing at most once per interval.

        Publishing is never allowed to break an index: a full disk or a racing
        reader costs an update, not the run.
        """

        if not self.enabled:
            return
        self.state = self.state.model_copy(update=fields)
        now = self._clock()
        if (
            not force
            and self._last_published is not None
            and now - self._last_published < self.interval_seconds
        ):
            return
        self._last_published = now
        self.state.updated_at = time.time()
        if self.listener is not None:
            self.listener(self.state)
        if self.directory is not None:
            with contextlib.suppress(OSError):
                self._write(self.state)

    def clear(self) -> None:
        """Remove the snapshot once the run is over."""

        if self.directory is None:
            return
        with contextlib.suppress(OSError):
            progress_path(self.directory, self.state.project_id).unlink(missing_ok=True)

    def _write(self, state: IndexProgress) -> None:
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json()
        # Written whole and renamed into place so a reader polling the file
        # never parses a half-written snapshot.
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(payload)
            os.replace(temporary, progress_path(self.directory, state.project_id))
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def read_progress(
    directory: Path, project_id: str, *, stale_after_seconds: float = STALE_AFTER_SECONDS
) -> IndexProgress | None:
    """Return the latest snapshot for *project_id*, or None when there is none.

    A snapshot older than *stale_after_seconds* is treated as absent: it was
    left behind by a process that died rather than by one still working.
    """

    path = progress_path(directory, project_id)
    try:
        payload = path.read_text()
    except (OSError, ValueError):
        return None
    try:
        progress = IndexProgress.model_validate_json(payload)
    except ValueError:
        return None
    if time.time() - progress.updated_at > stale_after_seconds:
        return None
    return progress
