# MCP server review fixes

Approved implementation plan, based on `e9768f7` and the September 8 review.

Goal: fix all fourteen numbered review findings in an isolated worktree.
Architecture: share bounded source access, validate rename bindings conservatively,
and enforce resource budgets at the producer and transport boundaries. Preserve
existing public response models and pinned partition semantics where possible.

- [x] 1–3: safe marker writes and bounded regular-file reads across scanning and references.
- [x] 4–5: imported-name shadowing and rename destination collisions.
- [x] 6–8: bounded derived context, extraction memory, supervised worker communication,
  and complete model-input token accounting.
- [x] 9: bounded Unix socket endpoint paths and robust discovery.
- [x] 10–11: complete literal symbol lookup and global search fusion.
- [x] 12: scan-configuration guard on the status fast path.
- [x] 13: Git enumeration timeout excludes consumer suspension.
- [x] 14: adjacency-based override discovery and bulk declarations.
- [x] Remove duplicate hash normalization and correct touched invariant comments.
- [x] Run formatter, lint, strict typecheck, and the full parallel test suite.

Each behavioral fix gets a failing regression before implementation. Tests extend
the existing pytest fixtures, simulated workers and emitted-patch checks. Larger
service splitting and general search preparation refactoring remain follow-ups.

Verification: `uv run ruff format .`, `uv run ruff format --check .`,
`uv run ruff check .`, `uv run mypy src`, `uv run pytest -n auto`.
Smoke coverage includes actual Git patch application and long-path daemon reconnect.


Implementation notes:

- Filesystem safety includes both later Indexer reads (forced validation and structural
  backfill), as well as scanner/reference/patch reads. POSIX opens pin directories;
  Windows uses native directory pins and final-handle validation. Unsupported safe
  opening mechanisms fail closed.
- Rename collisions withhold the whole edit set. Unsupported lexical scopes report
  `unsupported_binding`, including non-Python scoped uses; proven declaration/import
  cases remain available. Python dynamic/generic scope handling is conservative.
- Search fuses globally comparable cosine distances with ordinal lexical evidence.
  Raw partition BM25 scores are not treated as comparable. Candidate retrieval is
  still bounded per modality and partition; literal symbol filtering pages until
  sufficient real matches or exhaustion.
- Extraction preserves exact symbol identity but bounds repeated derived context.
  Parent budget checkpoints share a baseline with workers created later. Worker
  initialization and all request I/O have a supervised deadline.
- Windows native behavior has mocked coverage on macOS; real-model/accelerator gates
  require separately configured runners and are not claimed by the default suite.


Final verification (macOS, Python 3.12.11; CPU and TUI extras installed):

- `uv run ruff format .` and `uv run ruff format --check .`: 140 files formatted.
- `uv run ruff check .`: passed.
- `uv run mypy src`: passed, 70 source files.
- `uv run pytest -n auto --tb=short`: 2,026 passed, 10 skipped, 95.85 seconds.
- `git diff --check`: passed.

The full suite includes real Git patch application, long-runtime-root daemon
binding/discovery/reconnect, all fourteen regression groups, and checkout-aware
search deduplication. Existing optional model/accelerator/platform fixtures remain
skipped; the default suite does not establish real-device memory or latency limits.

Integration with `main` at `25afbf9`:

- Preserve upstream exact-length batching, passage embedding reuse, cache telemetry,
  and configure wizard changes alongside the review fixes.
- Advance the passage cache embedding contract to version 2 so windows generated
  before separator/special-token accounting are recomputed. A regression test first
  demonstrated the incompatible cache hit, then passed with the version change.
- Make the stale-daemon test wait for listening readiness; socket-path existence
  alone races with `listen()` under parallel test load.

Post-merge verification: formatter and format check passed (147 files), Ruff passed,
and mypy passed (71 source files). The full parallel suite passed with 2,087 passed
and 10 skipped in 195.94 seconds. `git diff --check` passed.
