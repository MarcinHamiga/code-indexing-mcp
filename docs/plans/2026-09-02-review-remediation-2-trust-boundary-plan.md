# Track 2 — Marker Trust and Data-Directory Mode — Implementation Plan

**Goal:** Bound what a repository-shipped `.ci-mcp/project.toml` can make the server
do, stop a known project id from silently moving to a new root, keep indexed source
text private on disk, and remove the one uninstall path that deletes by directory
name. Small, contained changes with a clear safety payoff.

**Review findings closed:** sec-major (marker trust), sec-major (index dir umask),
sec-minor (uninstall name short-circuit), info (patch containment guard), info (trust
boundary undocumented).

**Baseline:** see the index.

## Decisions settled before implementation

- **D1 — `max_file_bytes` gets a ceiling.** `ScanConfig.max_file_bytes`
  (`models.py:188`) is `Field(default=1_048_576, gt=0)`. Add
  `le=MAX_FILE_BYTES_CEILING` with `MAX_FILE_BYTES_CEILING = 16 * 1024 * 1024` as a
  module constant next to `DEFAULT_INCLUDES`. A marker above the ceiling fails
  validation in `read_project_marker` (`projects.py:144-147`), which already maps
  `ValidationError` to `PROJECT_NOT_FOUND` with the marker path in the message; make
  that message name the field when the cause is a `ValidationError` so the user can
  fix the file. `init_project` cannot produce an over-limit config today (no parameter
  for it); keep it that way.
- **D2 — A known id at a new root is a move only when the old root is gone.**
  `upsert_project` (`storage.py:513-530`) raises `PROJECT_ID_CONFLICT` only while the
  registered root still has a marker; a deleted marker lets any root claim the id.
  Change the guard to `registered_root.exists()`: a directory that still exists but has
  lost its marker is ambiguous and is treated as a conflict (unless
  `_shares_repository` says it is a linked worktree, exactly as now). A root that no
  longer exists at all is the legitimate "the user moved the directory" case and keeps
  today's behaviour. The conflict message gains the recovery hint: "run remove_project
  on the registered root, or init_project with force_new_id here". Verify `remove_project`
  (`application.py:1639-1655`) still lets the user resolve this.
- **D3 — Discovery walk stays as is.** `find_project_root` walking to `/`
  (`projects.py:173-180`) is what lets a nested checkout join a registered parent.
  Restricting it is a behaviour change without a clear win once D1 and D2 hold. Not
  changed; the README sentence in D6 explains the trust model instead.
- **D4 — Runtime directories are created private.** Add
  `RuntimePaths.ensure_private()` (`application.py:185-196`) that creates `data` and
  `cache` with `mode=0o700` and, when `os.getuid` exists and the directory is owned by
  the current user and `st_mode & 0o077` is set, chmods it to `0o700` (log at debug;
  never raise on chmod failure — a user-provided `CODE_INDEXING_DATA_DIR` on an odd
  filesystem must still work). Mirror `_private_directory` (`daemon.py:97-124`) but do
  not refuse foreign ownership; only skip the chmod. Call it from
  `Application.__init__` before anything under `paths` is touched
  (`application.py:219-254`), from `DaemonServer.serve` (`daemon.py:289`), and from
  `ensure_daemon` (`daemon.py:843`). Subdirectories keep default modes: a 0700 parent
  denies traversal. Windows: `mkdir(mode=)` is ignored there and there is no `getuid`;
  the method must be a no-op beyond creation.
- **D5 — Uninstall requires a marker.** Remove the
  `if resolved.name == "code-indexing-mcp": return None` short-circuit
  (`installer/uninstall.py:108-109`). Check `_DATA_MARKERS` (`:71-79`) still matches a
  freshly installed data directory (it lists `lancedb`, `locks`, `daemon.token`,
  `daemon.log`, ...) and the cache directory (it does not: add the markers a cache
  directory reliably has — `models`, `backend-probes.json` — or write a sentinel
  `.code-indexing-mcp` file from `RuntimePaths.ensure_private()` and accept it as a
  marker for both). Prefer the sentinel: it is explicit and survives a cache that was
  never populated.
- **D6 — README states the trust boundary.** One short section, "Trust boundary",
  after the intro: the MCP client is fully trusted and can register, index, search, and
  remove any directory the user can read; a checked-in `.ci-mcp/project.toml` is honoured
  within the size ceiling and the id-conflict rule; git is executed inside the
  repositories being indexed, so repository-local git configuration applies; the index
  stores chunk text under the user-private data directory. Four sentences, no more.
- **D7 — Patch emission containment.** Where `emit_refactor_patch` reads
  `root / path` (`reference_service.py:3159`), skip the file into `conflicted` with a
  reason when `(root / path).resolve()` is not relative to `root.resolve()`. The path
  comes from index rows so this cannot trigger today; the guard makes the invariant
  explicit and testable.

## Steps

**Step 0 — Coordinates.** Re-read the sites above plus `projects.py:130-160`,
`application.py:644-700` (`init_project`), `installer/uninstall.py:60-115`,
`reference_service.py:3140-3175`.

**Step 1 — D1.** Tests in `tests/test_projects.py`: a marker with
`max_file_bytes = 17 * 1024 * 1024` raises `PROJECT_NOT_FOUND` whose message names
`max_file_bytes`; a marker at the ceiling loads. `tests/test_scanner.py` unchanged.

**Step 2 — D2.** Tests in `tests/test_storage.py`: (a) registered root deleted from
disk, same id at a new root → accepted, root rewritten (existing behaviour, now
explicit); (b) registered root still exists, marker deleted, same id at a new
unrelated root → `PROJECT_ID_CONFLICT` with the hint; (c) linked worktree of the
registered repository → accepted as today; (d) `remove_project` then re-register at
the new root → accepted.

**Step 3 — D4 and D5.** Tests in `tests/test_application.py` (POSIX-only, skip on
Windows): after constructing an `Application` on a fresh `tmp_path`, the data and cache
directories are mode `0700`; a pre-existing `0755` directory owned by the user is
tightened; the sentinel file exists. `tests/test_installer_uninstall.py`: a directory
named `code-indexing-mcp` with no markers is refused; one with the sentinel is
accepted; every existing acceptance case still passes.

**Step 4 — D7.** Test in `tests/test_refactors.py`: monkeypatch the declaration path
to `../outside.py` and assert the file lands in `conflicted` with the containment
reason and no read happens outside the root.

**Step 5 — D6.** README section. Also add the ceiling to the `index_project` tool
description's size sentence only if it mentions the default cap as fixed.

## Completion note (2026-09-02)

All seven decisions implemented as designed; no deviations from D1–D7. Sentinel filename
constant is intentionally duplicated as a literal in `installer/uninstall.py` rather than
imported from `application.py`, to keep the installer's uninstall path from pulling in the
embedding/vector-store stack — noted in a comment at both ends. Baseline green: 1651 passed,
8 skipped (11 new tests over the 1640-passed baseline).
