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

from .models import IndexTrigger

# How long a snapshot stays trustworthy without a refresh. Comfortably above the
# publish interval, and above the pauses a single slow file can cause between
# updates, so a live run is never mistaken for a dead one.
STALE_AFTER_SECONDS = 60.0
PUBLISH_INTERVAL_SECONDS = 0.25


class IndexProgress(BaseModel):
    """A point-in-time snapshot of one project's indexing run.

    Every counter's name matches what it counts: ``candidates_*`` cover every
    path the scanner examined, ``eligible_files`` the files that passed the
    scan, and the ``*_files`` counters the eligible ones. Totals stay unset
    while the scanner streams and the run genuinely does not know them, and a
    candidate count is never compared with an eligible-file total.
    """

    project_id: str
    run_id: str = ""
    trigger: IndexTrigger = "manual"
    phase: str = "scanning"
    # Every path the scanner examined, whether it became eligible or was
    # skipped. A first index has no honest candidates_total: the scanner
    # streams, so the total is only known once the walk has finished.
    candidates_seen: int = 0
    candidates_total: int | None = None
    eligible_files: int = 0
    unchanged_files: int = 0
    changed_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    skipped_total: int = 0
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    bytes_read: int = 0
    chunks_extracted: int = 0
    chunks_embedded: int = 0
    chunks_staged: int = 0
    staged_bytes: int = 0
    current_path: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    phase_started_at: float = 0.0
    pid: int = Field(default_factory=os.getpid)

    @property
    def fraction(self) -> float | None:
        """Completion in ``[0, 1]``, or None when the total is unknown.

        Only candidate counts may feed this: comparing candidates seen with an
        eligible-file total would overstate (or understate) progress whenever
        the repository contains skipped paths.
        """

        if not self.candidates_total:
            return None
        return min(1.0, self.candidates_seen / self.candidates_total)

    def describe(self) -> str:
        """Render a one-line status suitable for a progress bar or a log line."""

        if self.phase == "committing":
            return "Committing the index"
        if self.phase == "extracting_references":
            return "Extracting structural references"
        if not self.candidates_seen:
            return "Scanning for changed files"
        if self.candidates_total:
            scanned = f"{self.candidates_seen}/~{self.candidates_total} candidates"
        else:
            scanned = f"{self.candidates_seen} candidates"
        parts = [f"{self.phase.capitalize()} {scanned}"]
        if self.eligible_files:
            parts.append(f"{self.eligible_files} eligible")
        if self.changed_files:
            parts.append(f"{self.changed_files} changed")
        if self.unchanged_files:
            parts.append(f"{self.unchanged_files} unchanged")
        if self.failed_files:
            parts.append(f"{self.failed_files} failed")
        if self.skipped_total:
            parts.append(f"{self.skipped_total} skipped")
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
        run_id: str = "",
        trigger: IndexTrigger = "manual",
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
        self.state = IndexProgress(
            project_id=project_id,
            run_id=run_id,
            trigger=trigger,
            started_at=time.time(),
            phase_started_at=time.time(),
        )

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
        phase = fields.get("phase")
        if phase is not None and phase != self.state.phase:
            fields["phase_started_at"] = time.time()
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
