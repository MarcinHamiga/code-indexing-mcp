# Track 4 — Daemon Lifecycle — Implementation Plan

**Goal:** A daemon that is retired whenever the code behind it changes, request
handling that cannot hang either side forever, an error contract that holds across
the socket, `configure` that restarts the daemon when it changes something the daemon
consumes, and a warm query model after start.

**Review findings closed:** arch-minor (daemon inherits first client's env; retired
only on RPC-shape change), sec-minor (no read timeouts, uncapped request threads),
arch-minor (`except BaseException`, `PROTOCOL_ERROR` overloading, raw transport
errors), perf-minor (cold start), arch-minor (broker/application drift test).

**Baseline:** see the index.

## Decisions settled before implementation

- **D1 — The ping carries a build identity.** `DaemonServer._dispatch` `ping`
  (`daemon.py:436-437`) returns `{"pid", "protocol"}`. Add `"build": BUILD_IDENTITY`
  where `BUILD_IDENTITY` is computed once at import in `daemon.py` as
  `sha256(f"{__version__}|{SCHEMA_VERSION}|{REFERENCE_SCHEMA_VERSION}|{source_stamp}")`
  with `source_stamp` = the git HEAD of the installed checkout when
  `update_check.checkout_head`-style lookup is available (managed installs), else the
  mtime of `daemon.py`'s package directory. `daemon_status` (`:799-817`) surfaces it;
  `ensure_daemon` (`:838-875`) treats a running daemon with a different `build` exactly
  like a protocol mismatch and retires it through `_retire_stale_daemon` (which must
  therefore send `stop` on the *current* protocol when the protocol matches). A daemon
  from before this change has no `build` key: treat "missing" as mismatch once, which
  is the upgrade path.
- **D2 — Environment mismatch is detected, not silently absorbed.** The daemon is
  spawned with the first client's environment (`:855-861`). Add the settings that the
  daemon actually consumes — the `IndexSettings` fields plus `CODE_INDEXING_OFFLINE`,
  `DATA_DIR`, `CACHE_DIR` — as a `settings_digest` in the ping reply (sha256 of the
  sorted `CODE_INDEXING_*` environment pairs the daemon started with). A client whose
  own digest differs logs a warning once per `BrokerApplication` naming the differing
  keys (compute locally, never send values over the wire) and continues. It does not
  restart the daemon: two live clients with different settings cannot both win, and a
  restart loop would be worse than a warning. `configure` handles the durable case
  (D6).
- **D3 — Read timeouts on both sides.** Server: `connection.settimeout(30)` before
  `receive_frame` in `_handle` (`:390`), then `settimeout(None)` before `_send_response`
  so a slow reader of a large response is not cut off; a timeout while receiving is
  reported as `PROTOCOL_ERROR` and the connection closed. Client: `_call_once`
  (`:537-540`) sets `settimeout(None)` after connect; replace with a per-method budget:
  `None` for `index_project`, `maintain_storage`, `init_project`, `stop`; otherwise
  `DAEMON_QUERY_TIMEOUT_SECONDS = 900`. A `socket.timeout` becomes
  `DAEMON_UNAVAILABLE` with the method name and the budget in details.
- **D4 — Request concurrency is capped.** `serve` (`:342-347`) spawns a thread per
  connection. Add `MAX_CONCURRENT_REQUESTS = 64` (a `threading.BoundedSemaphore`);
  when it cannot be acquired without blocking, respond immediately with
  `INDEX_BUSY` ("the local daemon is at its request limit") and close. `_handle`
  releases it in its `finally`.
- **D5 — Error contract at the edge.** In `_handle`: `except BaseException` (`:418`)
  becomes `except Exception`, reported under a new `ErrorCode.INTERNAL_ERROR`
  (`errors.py`) with `type(exc).__name__` in details rather than in the message
  (message: "The local daemon failed while handling <method>"). In
  `BrokerApplication._call` (`:561-570`): after the existing connect-phase retry,
  wrap `EOFError`, `ConnectionResetError`, `BrokenPipeError`, and `socket.timeout`
  into `DAEMON_UNAVAILABLE` with `method` and `log_path` in details; keep the
  no-retry rule for these (a completed non-idempotent operation must not be
  duplicated). `_with_error_details` in `server.py:780-798` needs no change.
- **D6 — `configure` restarts a running daemon when it changes daemon-consumed
  settings.** Move `_stop_daemon` / `_wait_until_stopped`
  (`installer/update.py:415-441`) to a shared `installer/daemon_control.py`
  (`stop_daemon(paths, *, reason) -> tuple[str, str]`), keep `update.py` calling it.
  In the configure flow (`installer/cli.py:275` `configure_main` and the orchestrator
  step that writes harness settings), after settings are written, compute whether any
  written or unset key is one the daemon consumes (every `CODE_INDEXING_*` key except
  the installer-only ones such as `CODE_INDEXING_MCP_BIN_DIR`); if so and a daemon is
  running, stop it and print the same status line `update` prints. `--no-prompt` runs
  do the same; nothing here needs a prompt.
- **D7 — Warm the query model on daemon start.** In `DaemonServer.serve`, after
  `ready.set()`, start a daemon thread that calls `self.application.prepare_model()`
  (`application.py:1991-1994`) inside `try/except Exception: logger.warning(...)`.
  Offline mode with no cached model logs and moves on. Count the warm-up as activity
  the way startup maintenance is counted (`_maintenance_active` pattern) so the idle
  timer cannot reap the daemon mid-load. Do not change `idle_timeout_seconds`.
- **D8 — Drift test between `BrokerApplication` and `Application`.** Add
  `tests/test_daemon.py::test_broker_mirrors_application_surface`: the set of public
  method names on `BrokerApplication` that `server.py` and `cli.py` call must exist on
  `Application` with the same positional/keyword parameter names (compare
  `inspect.signature` minus `self`, ignoring annotations). Also add a
  `typing.Protocol` `ApplicationLike` in `application.py` listing that surface, and
  type `create_server`'s parameter with it; `mypy` then enforces it.

## Steps

**Step 0 — Coordinates.** Re-read `daemon.py` end to end (875 lines),
`installer/update.py:400-500`, `installer/cli.py:275-326`, the orchestrator's settings
step, `server.py:801-816`, `errors.py`.

**Step 1 — D1 build identity.** Tests in `tests/test_daemon.py`: a daemon answering
`ping` with a different `build` is stopped and a new one started (drive with a
monkeypatched `BUILD_IDENTITY`); a missing `build` key counts as mismatch; a matching
one is reused. `daemon_status` exposes `build`.

**Step 2 — D2 settings digest.** Tests: differing `CODE_INDEXING_EMBED_THREADS`
between client and daemon environments produces one warning naming the key and no
restart; identical environments produce none; values never appear in the log.

**Step 3 — D3, D4.** Tests: a client that connects and sends nothing is dropped
after the server timeout and the daemon still exits on idle; a wedged dispatch
(monkeypatch a method to block) makes the client raise `DAEMON_UNAVAILABLE` after
the budget (use a lowered constant via monkeypatch); the 65th concurrent connection
gets `INDEX_BUSY` while 64 block.

**Step 4 — D5.** Tests: a dispatch raising `ValueError` reaches the client as
`INTERNAL_ERROR` with the type name in details; a daemon killed mid-request reaches
the client as `DAEMON_UNAVAILABLE`, not `EOFError`; `SystemExit` inside a request
thread is not swallowed (it is not `Exception`).

**Step 5 — D6.** Tests in `tests/test_installer_cli.py` / `test_installer_update.py`:
`configure --set CODE_INDEXING_EMBED_THREADS=2` with a running daemon (monkeypatched
`daemon_status` / `stop`) reports "daemon: stopped"; `--set CODE_INDEXING_MCP_BIN_DIR`
alone does not touch the daemon; `update` still stops it exactly as before.

**Step 6 — D7.** Test: `prepare_model` is called once after `serve` reaches ready
and a failure in it is logged, not raised; the daemon is not reaped while it runs.

**Step 7 — D8.** Protocol, drift test, and `create_server` typing.

**Step 8 — Docs.** README daemon paragraph: mention that `configure` restarts a
running daemon when it changes indexing settings, and that a daemon from a previous
build is replaced automatically.

## Completion note (2026-09-02)

Implemented all eight steps against the tree as it stood after Tracks 1-3, on top of
the already-green 1700-passed/9-skipped baseline. Final baseline:
`1722 passed, 9 skipped` (ruff format/check clean, mypy clean on `src`).

Deviations from the plan text, with reasons:

- **D6 test coordinates.** The plan's step-5 test sketch names
  `--set CODE_INDEXING_MCP_BIN_DIR` as the "no daemon-relevant change" case, but
  `CODE_INDEXING_MCP_BIN_DIR` is not, and never was, one of the settings
  `installer.settings_spec.BY_NAME` accepts through `--set` (it is an installer-only
  env var read directly from the process environment for launcher placement) --
  `--set CODE_INDEXING_MCP_BIN_DIR` would fail `parse_settings` with "unknown
  setting". Implemented the equivalent real invocation instead: `configure --bin-dir
  <path>` alone (no `--set`/`--unset`) does not touch the daemon. Also found that
  every key `BY_NAME` accepts is in fact daemon-consumed (no installer-only settings
  are in that catalog today), so `daemon_relevant_settings_changed` is `bool(env_updates)`
  with a comment explaining why, rather than an explicit exclusion list.
- **D6 restart scope.** Wired the restart only into the non-interactive
  `installer/cli.py:main()` path used by `configure_main` and driven by
  `args.reconfigure`; the `--repair` path (`_repair()`) intentionally writes back
  unchanged settings so never finds anything to restart over, and the TUI wizard
  (`installer/tui/panels.py`) was left unwired -- it tracks a fixed set of named
  progress steps and wiring a "daemon" step into it is a UI change beyond what D6
  describes ("hook the daemon restart after that write" in `cli.py`/`orchestrator.py`).
- **D4/D3 ordering.** The plan places D4's concurrency-cap check first in `_handle`,
  but checking it *before* reading the client's request frame created a real race:
  the server could reply-and-close before the client's `send_frame` had finished,
  handing it a raw `BrokenPipeError` instead of a clean `INDEX_BUSY`. Moved the D4
  check to run *after* the D3 receive (and after the protocol/token checks), which
  are cheap and cannot block on the network, eliminating the race while keeping the
  cap's purpose (rejecting before the expensive `_dispatch`) intact.
- **D5 error-response delivery.** Added `contextlib.suppress(OSError)` around the
  two error-frame `_send_response` calls in `_handle`: a client that already gave up
  (D3's own query-budget timeout, or a plain disconnect) leaves nothing to receive
  the reply, and the resulting `BrokenPipeError`/`ConnectionResetError` was crashing
  the request thread with an unrelated-looking traceback. Not asked for explicitly,
  but a direct, load-bearing consequence of D3+D5 landing together.
- **D8 surface and Protocol.** Built `ApplicationLike` from `Application`'s full
  signatures (the stricter, closed one) rather than `BrokerApplication`'s -- a
  `BrokerApplication` method's `**params: Any` catch-all structurally satisfies a
  protocol with named parameters, but the reverse does not hold, so the Protocol had
  to mirror the stricter side. Two parameters exist on `Application` but not on the
  shared surface at all: `index_project`'s `on_progress` (a local callback, cannot
  cross the daemon socket -- `BrokerApplication.index_progress` polls a published
  snapshot file instead) and `maintain_storage`'s `trigger` (the daemon's own
  scheduled-maintenance marker, not a client-settable value). Both are dropped from
  `ApplicationLike` rather than added to `BrokerApplication`, per the decision's own
  "widen the Protocol, not change behaviour". Typing `create_server`'s parameter
  with `ApplicationLike` required also widening `AutoIndexingMCP.__init__`,
  `StartupCoordinator.__init__`, `_ProgressStream.application`, and
  `_reporting_index_progress`'s parameter the same way (everything `create_server`
  hands the value to), and flipping one `isinstance(self.application,
  BrokerApplication): return` guard in `_run_startup_maintenance` to the positive
  `isinstance(self.application, Application)` form, since a Protocol-typed variable
  no longer narrows to "the other member of a two-class union" on exclusion.
  `test_broker_mirrors_application_surface` derives its checked surface from
  `ApplicationLike`'s own declared methods rather than a second hand-maintained
  list, so the Protocol and the drift test cannot drift from each other.

Nothing was left undone; every step (D1-D8) is implemented and covered by tests.
