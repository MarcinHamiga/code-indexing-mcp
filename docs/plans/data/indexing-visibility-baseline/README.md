# Baseline transcripts — indexing-visibility-cli-reporting (Phase 0)

Taken on `feat/indexing-visibility-cli-reporting` at `origin/main` (`6f5ec68`),
before any Phase 1 code changes.

Fixture: `/tmp/iv/repo` — 150 generated `pkg/mod_*.py` files (2 functions each),
plus `notes.txt` (unsupported) and `blob.bin` (binary) to exercise skips.
Isolated dirs: `CODE_INDEXING_DATA_DIR=/tmp/iv/data`,
`CODE_INDEXING_CACHE_DIR=/tmp/iv/cache` (with `models/` seeded from the user
cache so no download), `CODE_INDEXING_UPDATE_CHECK=off`.

Regenerate (from the worktree root, same env):

```sh
bash /tmp/iv/capture.sh <python> \
  docs/plans/data/indexing-visibility-baseline /tmp/iv/repo
```

(`capture.sh` runs `init`, `index`, `status`, `history`, `scan`,
`storage status`, `model status`, `daemon status`, one `.out`/`.err`/`.exit`
per command. The `.err` of `02-index` is the progress narration under test.)

Notable "before" observations (`02-index.err`, 6.8 s run, 300 chunks):

- Only 4 stderr lines; the two `Embedding` lines are byte-identical —
  chunk counters never advance during the longest phase.
- `Committing the index` carries zero counters.
