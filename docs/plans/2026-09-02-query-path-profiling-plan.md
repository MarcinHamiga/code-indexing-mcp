# Query-Path Profiling on a Real Repository — Plan

**Goal:** Put numbers behind the query-path track of the review remediation
(`2026-09-02-review-remediation-1-query-path-plan.md`). The track's tests assert its
properties on synthetic trees; before the next release the same properties are measured
on a real large checkout, on the pre-remediation baseline and on the remediated tree,
and the comparison is recorded next to the plan.

**Claims under test** (review findings perf-major 1–3, arch-minor freshness cache):

1. A lazy-mode call on a dirty worktree no longer walks the whole source tree.
2. The post-operation "did the repository move" check no longer spawns git.
3. Read queries no longer commit a LanceDB write.
4. Multi-project scopes resolve in parallel rather than sequentially.

**Baseline:** the normal repo baseline is unaffected (`uv run ruff format --check . &&
uv run ruff check . && uv run mypy src && uv run pytest -n auto`); this plan adds a script
under `scripts/`, which ruff covers and mypy does not.

## Decisions settled before implementation

- **D1 — One harness, run once per source revision.** `scripts/profile_query_path.py`
  drives `Application` in-process (`index_execution="in-process"`, `broker_mode="off"`)
  so the numbers are the application layer's own, not the daemon transport's. The same
  script runs against the review's baseline commit (`4e7a8b4`, a detached worktree) with
  `PYTHONPATH=<worktree>/src`, and against the remediated tree. Only the public
  `Application` surface is used, which both revisions share.
- **D2 — Three instruments per call.** Wall time (`perf_counter_ns`), git spawns (a
  `subprocess.Popen` subclass that records argv when `argv[0]` is `git`; `subprocess.run`
  goes through `Popen`, so each spawn counts once), and the number of files under the
  data directory whose mtime or size changed across the call (a write-on-read detector
  that needs no knowledge of the store's layout).
- **D3 — Scenarios mirror the review's findings.** `clean.*` (worktree matches the index),
  `dirty.first` then `dirty.steady` (one tracked file edited; the call that refreshes it
  and every call after while the tree stays dirty), `head.first` then `head.steady` (the
  edit committed on the same branch so HEAD moves by one commit), `head_return.*` (reset
  back), and `multi8.*` (eight small projects cut from one subtree of the repository, plus
  the same scope with the large repository added). `clean.embed_query` times the query
  embedding alone so the irreducible part of a search call is visible.
- **D4 — Embed once, refresh twice.** The baseline run indexes the repository from
  scratch. The remediated run seeds its data directory with `--copy-data-from` the
  baseline's, then runs `index_project`, so it pays only the registry migration and an
  incremental refresh. This keeps the comparison to the query path and halves the
  wall-clock of the experiment. Index timings are recorded but are not what this plan
  compares.
- **D5 — Repository choice.** Django at a pinned commit: ~2,900 Python files, one
  supported language, a real git history, and a `django/utils` subtree that makes
  plausible small projects. The script does not hard-code the repository, only the
  paths inside it it touches (`DIRTY_TARGET`, `OUTLINE_PATH`, `MULTI_SUBTREE`); a second
  repository in another language is a parameter change plus those three constants.
- **D6 — The repository must be clean before and after.** The script refuses a dirty
  checkout, restores the touched file by `git reset --hard <original HEAD>` in a
  `finally`, and never touches anything outside the given workspace.

## Steps

**Step 1 — Harness.** Write `scripts/profile_query_path.py` per D1–D3, ruff-clean.
Output is one JSON document per run: source revision, repository HEAD, index summary,
and per-scenario `wall_ms` (min/median/mean/p90/max), `git_spawns` (min/max/total),
`data_files_changed` (min/max/total), and the git argv list of the first call.
`--cprofile` attaches `cProfile` to one `dirty.steady` call and stores the top 30
cumulative entries so a regression has a stack, not only a number.

**Step 2 — Baseline run.** Clone Django shallowly (`--depth 50`) into the scratch
workspace, add a detached worktree at `4e7a8b4`, run the harness with that worktree on
`PYTHONPATH`, ten iterations per scenario.

**Step 3 — Remediated run.** Run the harness from this branch with `--copy-data-from`
the baseline data directory (D4).

**Step 4 — Record.** Write `2026-09-02-query-path-profiling-shipped.md` with the two
runs side by side per scenario and a verdict per claim in the list above. Keep the raw
JSON under `docs/plans/data/` so the next release can re-run and diff.

**Step 5 — Release gate.** The shipped note names the numbers that must hold on the
next release: `dirty.steady.search_code` spawns zero git processes and changes zero data
files, and its median is within a small factor of `clean.search_code`. Re-run Steps 2–3
with the release candidate on `PYTHONPATH` before tagging.
