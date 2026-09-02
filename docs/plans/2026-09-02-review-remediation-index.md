# Review Remediation — Index

**Source:** architecture, performance, and security review of `main` at `4e7a8b4`
(2026-09-02). No critical findings; ten major, twelve minor. The review's ranked
findings and the fixes it proposed are what these plans implement, in the order the
review suggested.

**Baseline (holds before and after every track):**
`uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest -n auto`
— green at 1618 passed, 8 skipped on 2026-09-02.

**Ground rules for every track**

- No behaviour change beyond what the track names. Public tool schemas, error codes,
  and response shapes stay identical unless a step says otherwise.
- Existing tests are the contract. A test may be edited only where it reaches into a
  private that the track moves; its assertion must survive.
- Every step ends with the baseline command green. A step that cannot get there is
  reverted and reported, not left half-done.
- All line coordinates below were read from the tree on 2026-09-02; re-verify before
  editing, and follow the code where it has moved.

## Tracks

| # | Plan | Findings closed |
|---|------|-----------------|
| 1 | [Query-path overhead](2026-09-02-review-remediation-1-query-path-plan.md) | Dirty-worktree full walk; ~13 git spawns per call; LanceDB write on every read; unsynchronized freshness cache |
| 2 | [Marker trust and data-directory mode](2026-09-02-review-remediation-2-trust-boundary-plan.md) | Unbounded `max_file_bytes`; silent id takeover; index dir at default umask; uninstall name short-circuit; patch containment; README trust boundary |
| 3 | [Reference query pushdown](2026-09-02-review-remediation-3-reference-pushdown-plan.md) | Whole reference table per call; per-node reload in `impact_radius`; full chunk scan in `_select` |
| 4 | [Daemon lifecycle](2026-09-02-review-remediation-4-daemon-lifecycle-plan.md) | Stale daemon after non-protocol changes; no read timeouts; `except BaseException`; transport errors bypass the error contract; `configure` leaves the daemon on old settings; cold start |
| 5 | [Application split](2026-09-02-review-remediation-5-application-split-plan.md) | God facade; sideways imports; store private reach-through; env reads outside `IndexSettings`; broker/application drift |

**Deferred by the review itself, now planned separately:**

- Per-language rules table for `extractor.py` and `reference_service.py`:
  [plan](2026-09-02-language-rules-table-plan.md). Runs on its own branch from `main`
  after PR #51.
- Vector-index size gate: [plan](2026-09-02-vector-index-gate-plan.md), measured in
  [results](2026-09-02-vector-index-gate-shipped.md).
- Query-path gains measured on a real repository (the release gate for track 1):
  [plan](2026-09-02-query-path-profiling-plan.md),
  [results](2026-09-02-query-path-profiling-shipped.md).
