# Index Freshness Monitoring Design

## Goal

Keep indexed projects consistent with their source trees: eager mode refreshes at startup and
continues monitoring changes, lazy mode refreshes when a code query observes drift, manual mode
indexes only on explicit request, and status reports stale indexes honestly.

## Architecture

`Application.project_is_stale` performs the same stat-level comparison used by incremental
indexing: the eligible path set and each stored file's size and nanosecond mtime are compared with
a fresh scanner pass. `project_status` maps a stored ready/partial state to `stale` when that
comparison finds drift. `BrokerApplication` exposes the check through the existing daemon RPC.

`StartupCoordinator.schedule` remains the single entry point for automatic jobs. Completed jobs
are reused only when their project is still fresh. In eager mode the coordinator also starts one
`watchfiles` producer and one coalescing consumer per discovered root. The producer never waits for
indexing; it places a single dirty token in a bounded queue. The consumer schedules and waits for a
refresh. Changes that arrive during a refresh leave a token queued, guaranteeing a follow-up pass
without starting concurrent jobs. Lazy mode starts no watcher and relies on the stale check already
performed by every project-scoped code query. Manual mode bypasses all automatic scheduling.

## Git exclusions

For Git worktrees, the scanner batches candidate paths through `git check-ignore --no-index`, which
applies repository `.gitignore` files, `$GIT_DIR/info/exclude`, and the user's configured global
exclude file. The existing in-process pathspec implementation remains the fallback for non-Git
projects and environments where Git is unavailable. Project-specific include/exclude rules and hard
directory exclusions remain authoritative in both paths.

## Errors and shutdown

Watcher failures are logged without terminating the MCP server. Index failures remain visible to
waiting queries and later changes can retry them. The existing lifespan task group owns all watcher
and refresh tasks, so shutdown cancellation continues to wait only for non-abandonable active index
writes. Query-time stale checks provide a correctness fallback if an OS event is missed.

## Testing

Tests cover create/modify/delete refreshes, a change arriving during an active eager refresh, lazy
query refresh, manual-mode stability, stale status, daemon freshness RPC, and `.git/info/exclude`.
Existing startup, lock, cancellation, scanner, and daemon tests remain regression coverage.
