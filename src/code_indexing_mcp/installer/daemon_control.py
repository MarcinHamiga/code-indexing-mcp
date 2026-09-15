"""Stop the per-user daemon on behalf of the installer.

Both ``update`` (a code change invalidates a running daemon's in-memory
model and settings) and ``configure`` (D6: writing a setting the daemon reads
at startup does too) need the same "stop it, then wait until it has actually
exited" sequence and want to report it the same way, so it lives here once
rather than twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from ..application import RuntimePaths
from ..daemon import BrokerApplication, daemon_status, daemon_supported, stop_daemon_and_wait

DaemonStopStatus = Literal["skipped", "warning", "ok"]


def stop_daemon(paths: RuntimePaths, *, reason: str) -> tuple[DaemonStopStatus, str]:
    """Stop a running daemon so the next client respawns one on current code and settings.

    ``reason`` names what changed (``"code"`` for ``update``, ``"settings"``
    for ``configure``) and is folded into the success detail, so both
    commands report the same shape of status line for what is, from the
    daemon's point of view, the same operation: something it was started
    with is now stale.
    """
    if not daemon_supported():
        return "skipped", "this platform has no shared daemon"
    try:
        current = daemon_status(paths)
        if not current["running"]:
            return "skipped", "no daemon is running"
        stopped = stop_daemon_and_wait(
            paths,
            status=daemon_status,
            stop=lambda: BrokerApplication(paths).stop(),
            initial_running=True,
        )
        if not stopped:
            return "warning", "the daemon did not stop; run `code-indexing-mcp daemon stop`"
    except Exception as exc:
        return "warning", f"the daemon could not be stopped: {exc}"
    return "ok", f"stopped; it restarts on the updated {reason} with the next client"


def daemon_relevant_settings_changed(env_updates: Mapping[str, str | None]) -> bool:
    """Whether *env_updates* sets or unsets a setting the daemon reads at startup.

    Every setting the installer manages (``installer.settings_spec.BY_NAME``,
    which is the only source of keys ``--set``/``--unset`` accept) is a
    setting the daemon actually consumes: index and embedding behaviour, the
    data/cache directories, offline mode. Installer-only knobs such as
    ``CODE_INDEXING_MCP_BIN_DIR`` are read from the process environment, not
    written through this path, so they never appear in *env_updates* -- there
    is currently nothing to exclude. If an installer-only setting is ever
    added to the managed catalog, exclude its name here explicitly.
    """
    return bool(env_updates)
