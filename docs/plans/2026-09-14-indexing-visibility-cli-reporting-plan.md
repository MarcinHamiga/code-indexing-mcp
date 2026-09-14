## Goal

Make indexing runs fully observable and CLI output human-legible without breaking the existing JSON/MCP contracts: live progress that keeps moving through the longest phases, reports and audit records whose counters reconcile, and readable `status` / `history` / `scan` / `storage` / `model` / `daemon` output — implemented on a fresh worktree branched from current `main`.

## Success Criteria

- A large index run emits visible, advancing progress through scanning, embedding, reference extraction, and committing — never minutes of silence on either a TTY or a redirected log.
- Every counter in `index` output, `status`, `history`, and the progress line can be reconciled: no two same-named numbers disagree, truncated lists say they are truncated, and phase timings document why they do not sum to the total.
- A human can read `status`, `history`, `scan`, `storage status`, `model status`, and `daemon status` without parsing JSON, while `stdout` JSON output stays byte-compatible in shape for existing consumers.
- Any run can be attributed after the fact: run id, trigger, branch slot / selector / HEAD, and checkout are visible in history and status.
- Automatic (watcher/startup) indexing leaves the same audit trail as manual runs and is identifiable as automatic.
- Full gate passes: `ruff format --check .`, `ruff check .`, `mypy src`, `uv run pytest -n auto`.

## Context And Current Facts

Verified on `main` at `6f5ec68` (in sync with `origin/main`, clean except pre-existing untracked review/plan files):

- **Progress goes dark in the longest phases.** `IndexProgress.describe()` returns static strings with zero counters for `committing` and `extracting_references` (`models.py:1111-1114`), and never shows `current_path`, `run_id`, `trigger`, or slot/selector. The embedding loop publishes a counter-less update per group (`indexing.py:1231`) and only sets `chunks_embedded`/`chunks_staged` at the very end (`indexing.py:1289-1296`), so the longest phase looks stalled. The CLI's non-TTY logger additionally only prints on phase change or every 5s (`cli.py:319-331`).
- **Report counters are misleading or duplicated.** `discovered_files` actually counts eligible files (`indexing.py:248`), not scanned candidates; durations exist twice (`scan_duration_ms` and `scan_ms`, `models.py:741-754`) with total-vs-phase mismatch documented only in a code comment; `embedded_segments` can exceed `embedded_chunks` on worker runs with the explanation only in a comment (`models.py:768-772`).
- **History is branch-blind.** `RunAudit` / the `runs` table (`history.py:38-82`, `models.py:960-1015`) record no slot id, selector, indexed HEAD, or checkout root, so runs from different worktrees/branches are indistinguishable after the fact. Errors and skipped samples are truncated to 20 (`indexing.py:484-485`, `history.py:34-35`) with no total-vs-sample marker. Cursor pagination (`history.py:290-329`) and `recent()` (`history.py:331-352`) are sound and stay.
- **CLI is JSON-only and JSON-hostile.** Every read command prints `_json()` with `sort_keys=True` (`cli.py:350-357`), i.e. alphabetical, deeply nested, raw bytes/milliseconds/ISO timestamps. There is no `progress` command, no `--format`, no `--watch`, no `--quiet/--verbose` (`logging` is unconditionally INFO, `cli.py:362`), and all failures exit `2` (`cli.py:552-554`).
- **Background work is nearly invisible.** Auto-indexing logs only "N files indexed" (`server.py:337-342`) with no run id, trigger, duration, or outcome; `daemon status` answers only `{running, ...ping}` (`daemon.py:1163-1181`); progress snapshots are deliberately per-project-id files (`progress.py:38-39`, `server.py:1100-1101`) and the daemon's indexing thread must not serve RPC (`daemon.py:983-990`).
- **Conventions to reuse:** additive optional model fields with `schema_version` bumps (`HistoryPage`, `StorageStatus` v2); SQLite `ALTER TABLE` migration guards (`history.py:166-182`); `stdout`-pure JSON with progress on `stderr` (`cli.py:293-299`); `.worktrees/<name>` worktree layout with `feat/...` branches; `COMMAND_NAMES` parity pin between `code-indexing-mcp` and `syndex` (`cli.py:41-58`, `tests/test_syndex_parity.py`).

## Constraints And Non-goals

- JSON field renames/removals are forbidden; `stdout` JSON for existing commands must keep validating old consumers (additive optional fields only, duplicates documented, deprecation via docs not deletion).
- No new daemon RPC served by the indexing thread; cross-process visibility stays snapshot-file based.
- No behavior changes to ranking, passage cache, accelerators, reference resolution, or the TUI beyond what CLI parity requires.
- No roadmap/docs consolidation beyond the operator-facing reporting docs this work needs (that is review item 16, separate).
- Assumption (reversible): JSON remains the default `--format`; human rendering is opt-in. If you prefer TTY auto-detection, say so at approval and Phase 3 flips the default.

## Key Decisions

| Decision | Recommended | Why / rejected alternative |
|---|---|---|
| Human output mechanism | `--format human|json` (default `json`) on read commands + a one-line stderr summary after `index` | Keeps pipes/MCP stable; rejected TTY auto-detection (surprising under pipes, harder to test) |
| Counter contract | Document one canonical contract in `models.py` + operator doc; keep field names, fix meanings via new clarifying fields where the name lies | Renames would break MCP clients; a documented contract plus `*_total` companions fixes correctness without breakage |
| Duration duplication | Keep both spellings, populate identically, document as aliases, deprecate one in docs | Matches the codebase's own "older clients keep validating" pattern |
| Branch attribution | Add `slot_id`/`selector`/`indexed_head`/`checkout_root` to `RunAudit`/report via guarded `ALTER TABLE` migration | Follows the existing `pid`/`rebuild_reason` migration pattern; required to make history useful with worktrees |
| Live-run CLI surface | New `progress` command + `status --watch` polling the snapshot file; no daemon RPC | Honors the "indexing thread must not serve RPC" constraint; snapshot polling is what the MCP path already does (`server.py:596-616`) |
| Exit codes | Keep `2` for usage/client errors; add distinct codes for `INDEX_BUSY` and backend-unavailable | Small, additive, script-friendly; rejected a full exit-code taxonomy as scope creep |

## Recommended Approach

One worktree, one branch, five sequential PRs (each independently reviewable and revertible). All model changes are additive; all CLI changes keep `stdout` JSON shape-compatible. Work proceeds from the run itself outward: first fix what a live run reports, then what a finished run records, then how humans read it, then background-work visibility, then docs and acceptance.

- Phase 0 — worktree + baseline (no code changes beyond the plan file).
- Phase 1 — live progress transparency (`models.py` describe/fraction, `indexing.py` embedding/reference/commit updates, `cli.py` printer).
- Phase 2 — reporting and audit correctness (counter contract, run attribution migration, truncation markers, duration aliases documented).
- Phase 3 — CLI legibility (`--format human` renderers, `progress` command, `--watch`, exit codes, log verbosity flags).
- Phase 4 — background-work visibility (structured auto-index completion log with run id/trigger/duration/outcome, richer `daemon status` from lock/queue/snapshot state readable without RPC, `status` correlating live `progress` with `last_run`).
- Phase 5 — operator docs + acceptance (docs page, README pointers, focused + end-to-end tests, full gate).

## Work Plan

**Phase 0 — Setup (no behavior change).**
Create `.worktrees/indexing-visibility-cli-reporting` from `origin/main` on branch `feat/indexing-visibility-cli-reporting`; copy this plan to `docs/plans/2026-09-14-indexing-visibility-cli-reporting-plan.md` on that branch; capture baseline CLI transcripts (`index`, `status`, `history`, `scan`, `storage status`, `model status`, `daemon status`) against an isolated `CODE_INDEXING_DATA_DIR`/`CACHE_DIR` fixture repo for before/after comparison.

**Phase 1 — Live progress transparency.**
`models.py`: `describe()` includes counters, current path tail, run id/trigger, and slot/selector in every phase (no more counter-less phases); keep `fraction` candidate-scoped per the pinned contract (`tests/test_progress.py:60-83`). `indexing.py`: publish `chunks_embedded`/`chunks_staged`/file counters per embedding group (not only at the end), per-file counters during reference backfill without clobbering scan totals, and committing-phase totals. `cli.py`: TTY printer shows a stable single-line status with phase + counts; non-TTY logger gains periodic count lines during embedding (not only phase changes). `server.py`: reuse the improved `describe()` for MCP progress messages (no protocol change).

**Phase 2 — Reporting and audit correctness.**
Write the canonical counter contract (candidates vs eligible vs changed/indexed/parsed/failed vs skipped; chunks extracted vs embedded vs staged; segments vs chunks) as the module docstring in `models.py` and mirror it in the operator doc. `IndexReport`: always set `run_id`/`trigger`; add `candidates_seen`/`candidates_total`, `slot_id`/`selector`/`indexed_head`/`checkout_root`, `error_total`/`skipped_total` companions to the sampled lists, and one documented duration-alias pair. `history.py`: guarded migration adding the attribution columns; `finish()` allowlist extended; `recent()`/`list_runs()` unchanged in shape. Cap `report.errors` growth for failure storms (bounded list + total count) while keeping `failed_files` exact.

**Phase 3 — CLI legibility.**
Add `--format human|json` to `status`, `history`, `scan`, `storage status`, `model status`, `daemon status`, and `index` (human goes to `stdout` replacing JSON only when requested; JSON default unchanged). Human renderers: aligned tables, human bytes/durations/ages, last-run + live-progress correlation line in `status`, one-line-per-run history with trigger/state/duration, skip-reason breakdown in `scan`. New `progress <project>` command printing the current snapshot (or "no live run"); `status --watch[=seconds]` polling loop. Exit codes: distinct code for `INDEX_BUSY`; `--quiet`/`--verbose` controlling stderr logging; keep `COMMAND_NAMES`/`syndex` parity and extend `tests/test_syndex_parity.py` + `tests/test_cli.py`.

**Phase 4 — Background-work visibility.**
`server.py`: auto-index completion log becomes structured (project, run id, trigger, slot/selector, duration, outcome, changed/failed/skipped counts) at INFO with errors summarized, not just "N files indexed". `daemon status`: add lock-holder/queue-depth/progress-file presence derived from lock + snapshot files without RPC to the busy thread. `status`: when a live `progress` snapshot and `last_run` disagree (run id/trigger/slot), say so explicitly instead of showing both silently.

**Phase 5 — Docs and acceptance.**
New `docs/indexing-visibility.md` operator page (progress semantics, counter contract, history attribution, CLI cookbook); short README pointers. Tests: progress-advance tests for embedding/backfill/commit phases, contract tests asserting report/history/status counter reconciliation, golden human-format transcripts, migration test (old DB → new columns, old clients validate new payloads), CLI exit-code/verbosity tests. Run the full gate per AGENTS.md after every phase.

## Validation Plan

- Per phase: `uv run ruff format .`, `uv run ruff check .`, `uv run mypy src`, `uv run pytest -n auto` (highest-risk gate: Phase 2's migration + contract tests — a wrong default or missed allowlist entry breaks audit writes or old clients; run `tests/test_history.py`, `tests/test_indexing.py`, `tests/test_application.py` first, then the full suite).
- Phase 1: new tests asserting `describe()` contains counters in all four phases and per-group embedding advancement on a multi-group fixture; manual `index` on a ~500-file fixture showing movement on TTY and in redirected stderr.
- Phase 2: contract test indexing a fixture with skips + failures and asserting `index` report vs `history` vs `status.last_run` reconcile; migration test opening a pre-change `runs.sqlite`.
- Phase 3: golden-transcript tests for each human renderer; `progress` against a live daemon-owned run; exit-code tests for busy/invalid; `syndex` parity tests green.
- Phase 4: watcher-triggered auto-index on a temp worktree asserting structured log + attributed history row with `trigger != manual`; `daemon status` human/JSON before/during/after a run.
- Before/after: rerun the Phase 0 baseline transcripts and diff JSON shapes (additive-only) plus eyeball human output.

## Risks / Rollback

- History migration on user databases: mitigated by guarded `ALTER TABLE` (existing pattern), additive nullable columns, and a migration test over a realistic DB; rollback is `git revert` of the phase PR — old code ignores new columns, new code tolerates missing columns until migration runs.
- `describe()`/progress text is consumed by MCP clients as message strings: message content may change but the progress protocol fields are untouched; client-visible risk is cosmetic.
- Human-format scope creep: contained by one renderer module with golden tests; no TUI changes.
- Branch-slot progress overwrite (one snapshot per project): documented as a known limitation; Phase 4 surfaces the owning slot so overwrites are at least visible rather than silent. Per-slot snapshots are explicitly deferred.

## Open Questions

None — all material choices were resolved from workspace evidence with reversible defaults noted above.
