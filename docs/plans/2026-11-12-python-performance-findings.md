# Python Performance Findings — Query and Index Hot Paths (2026-11-12)

This profiling pass looked at where the serving and indexing hot paths spend their time,
and what can be won without leaving Python. It covers four findings:

- **F1.** An index run stalls every query served by the same process (GIL convoy).
- **F2.** Extraction time is ~90% Python per-node logic.
- **F3.** Git subprocess waits are the largest share of `project_status` and of no-op
  index runs.
- **F4.** No-op watcher runs write a new storage version (observed, cause not traced).

F1 has the largest user-visible effect.

Related: the algorithmic findings in `2026-11-10-performance-and-bug-review.md` (REF-7,
EXT-1, SRV-5, IDX-10, STO-12, DMN-5) are not repeated here. Where a finding below
touches one of them, it says so.

## Setup

- Apple M4 Pro, Python 3.12.11, lancedb 0.34.0.
- Branch `feat/f2llm-v2-160m` between `f0ddb78` and `0d3d6f8`. The commits in between
  change only the embedding stack (plus a three-line default in `storage.py`).
  `extractor.py`, `indexing.py`, `server.py`, `application.py` and `scanner.py` are
  identical across that range.
- Index phase timings come from this repo's own `index_history`.
- In-process measurements drive `Application` against a local clone of this repo
  (311 indexable files, 4.6 MB). They use `tests/support.py`'s `DeterministicEmbedder`
  (4-dimensional vectors), so they exclude inference and understate LanceDB write volume.
- Time attribution uses cProfile:
  - waits in `select.kqueue` are LanceDB answering a sync call;
  - waits in `select.poll` / `fork_exec` are `git` subprocesses.

## Where time goes

| Path | Time | Dominated by |
|---|---:|---|
| `search_code` | 21 ms | ~18 ms inside LanceDB, ~2–3 ms project Python |
| `find_symbol` | 12 ms | LanceDB |
| `project_status` (precedes every lazy query) | 44 ms | ~27 ms waiting on one `git` spawn per call |
| No-op index run | 306 ms | ~15 `git` spawns per run, ~130 ms waiting on them |
| One-file edit (`index_history`) | ~1 s | embed 600–975 ms, commit ~200 ms, scan ~80 ms, parse 3 ms |
| Full index, warm passage cache (`index_history`, 7,147 chunks, 97% reused) | 7.1 s | parse 2.8 s, embed 2.7 s, commit 1.3 s, scan 0.1 s |
| Full index, cold passage cache | ~40 s MLX / ~140 s CPU | model inference |

The cold-cache row combines this repo's 7,147 chunks with F2LLM-v2-160M throughput from
`2026-11-12-f2llm-v2-160m-migration-plan.md`: 182.9 chunks/s on MLX FP16 and 52.2
chunks/s on CPU (`t=2`).

On the query path, project Python is only a few milliseconds. The remaining cost is
storage and `git`. On the index path, parsing is the largest Python-owned phase once
the passage cache is warm.

## F1 — An index run stalls every query in the same process (GIL convoy)

**Where.** Index runs execute via `anyio.to_thread.run_sync` (`server.py:229`). Query
tools run via `asyncio.to_thread` (`server.py:1231` and siblings). Both are threads in
the same process, and the daemon serves all clients from it.

**Measured.** Latency of one `search_code` + `find_symbol` pair, 30 samples each:

| Condition | p50 | p90 | max |
|---|---:|---:|---:|
| idle | 25.6 ms | 26.9 ms | 28.7 ms |
| another thread running extraction | **6,377 ms** | 6,689 ms | 7,021 ms |
| same, `sys.setswitchinterval(0.0005)` | 671 ms | 723 ms | 824 ms |

**Cause.** A query releases and reacquires the GIL about 1,270 times, mostly through
LanceDB's sync-over-async bridge (one `kqueue` wait per event-loop turn). While a
CPU-bound thread holds the GIL, each reacquire waits up to the switch interval (5 ms by
default). Cutting the interval tenfold cut the slowdown tenfold, which confirms the
mechanism.

**Real-world size.** The test runs extraction continuously. A real index run also
spends time waiting on the embedding worker, when the GIL is free, so production stalls
are shorter. They still last for the whole parse phase, and every daemon client sees
them, whichever project it queries. Lazy refreshes, eager watcher runs and reference
backfills all count.

**Fix options, cheapest first:**
1. **Stopgap:** call `sys.setswitchinterval(0.0005)` at server and daemon startup.
   That's one line. It cuts the stall about tenfold but still leaves queries 25× slower
   during extraction, and costs some CPU throughput in switching.
2. **Make fewer GIL handoffs per query:**
   - make the explanation probes opt-in or derive them from scores already fetched
     (STO-12a), which removes about half of `search_code`'s storage calls;
   - batch lookups that are issued one by one.
   The stall shrinks in proportion to the handoffs removed.
3. **Run index work outside the serving process.** This is the real fix. Either run the
   whole index run in a child process (file locks are already multi-process safe), or
   send extraction to a worker the way embedding already is. Points to design for:
   - **Child startup:** it pays the import cost (0.76 s / 143 MB for
     `code_indexing_mcp.server`; see DMN-5) unless it starts from a lean entry module.
   - **Memory budget:** `indexing_memory_budget` and the 2 GB ceiling currently measure
     the parent process.
   - **Progress and cancellation:** these need to cross the process boundary.
   - **Result transfer:** shipping extraction results back costs pickling (this repo:
     6,978 chunks and 128k reference rows).
4. Make extraction faster (F2). This shortens the stall window but does not remove it.

**Gate for a fix.** Rerun the contention test below with a real index run instead of
the synthetic loop. Query p50 during the run should stay within 2× of idle.

## F2 — Extraction is ~90% Python per-node logic

**Measured.** Tree-sitter's parse is a small share. The reference pass dominates:

| Corpus | Files | Tree-sitter parse | Full extract | Reference pass |
|---|---:|---:|---:|---:|
| this repo (Python) | 309 | 233 ms | 2,648 ms | 1,970 ms |
| pocketworld-nestjs (TypeScript) | 163 | 27 ms | 299 ms | 137 ms |
| manusail-django (Python) | 51 | 14 ms | 147 ms | 110 ms |

Building `_ParentMap` adds 133 ms on this repo.

**Profile (this repo, cProfile):**

| Function | Calls | Self | Cumulative |
|---|---:|---:|---:|
| `_identifier_record` | 192,700 | 0.86 s | 1.90 s |
| `_ParentMap.parent` | 1,952,428 | 0.24 s | 0.39 s |
| `_structural_records` | 294 | 0.28 s | 3.90 s |
| `_ParentMap.build` | 309 | 0.27 s | 0.48 s |
| `_enclosing_symbol` | 128,338 | 0.20 s | 0.39 s |
| `add` (closure in `_structural_records`) | 155,829 | 0.21 s | 0.98 s |
| `Node.child_by_field_name` | 1,139,746 | 0.16 s | 0.16 s |
| pydantic `validate_python` | 200,758 | 0.14 s | 0.14 s |
| `staging.stage_references` (in a full index run) | 309 | 0.32 s | 0.49 s |

**Candidates.** These are unmeasured. Each must keep
`tests/test_extractor_equivalence.py` (the committed snapshot) green.

1. **`_identifier_record` climbs to the root for every identifier.** Each identifier
   averages ~10 `parents.parent()` steps, each with several set lookups and
   `child_by_field_name` calls. The loop only exits early to *exclude* an identifier,
   so every real read walks to the root. Two options:
   - stop the climb at ancestor types where no exclusion rule can fire above them
     (statement or block boundaries, per language);
   - memoize per-ancestor facts that do not depend on the identifier.
   This is the largest single target.
2. **`_enclosing_symbol` re-climbs for each of 128k references.** References that share
   ancestors repeat the same walk. Options:
   - memoize the result per visited node id within a file;
   - do the byte-range sweep over sorted declarations that EXT-1's fix notes describe.
3. **Method-call overhead on `_ParentMap.parent`** (1.95 M calls). In the hot loops,
   bind `parents._parents.get` to a local and look up `node.id` directly. Same for
   `rules` attribute reads inside `_identifier_record`'s loop.
4. **pydantic validation on internal records.** Each `ExtractedReference` (a
   `FrozenModel`) is validated on construction. These are built from trusted extractor
   output, so `model_construct` or a slotted dataclass skips the validation. Pydantic
   is about 3% of a full index run.
5. **`stage_references` transposes rows with `getattr` per field per row** (128k rows ×
   every schema field). Transpose once with `operator.attrgetter`, or build the column
   lists directly while emitting references.

**Ceiling.** Tree-sitter parse plus the parent map is about 0.37 s of the 2.65 s on this
repo; the rest is the target. The phase matters for warm-cache full indexes, reference
backfills and large repos. It does not matter for incremental edits (parse is 3 ms) or
cold indexes, which are inference-bound.

## F3 — Git subprocess waits dominate status and no-op index runs

**Measured.**
- `project_status`: 44 ms per call, of which ~27 ms is waiting on one `git` spawn.
- No-op index run: 306 ms, with about 15 `git` spawns and ~130 ms waiting on them.
- Earlier Django measurements (`2026-09-02-query-path-profiling-shipped.md`) put each
  spawn at ~29 ms.

**Candidates:**
- **No-op runs:** count which `git` commands a no-op run issues. Collapse repeated
  probes the way the 2026-09-25 change did for status-then-query (7 → 4 spawns), and
  skip the run entirely when the watcher's change set contains nothing indexable.
- **`project_status`:** see SRV-9 (`list_files` materialised for a count) and the
  freshness cache. The spawn itself is the floor, so avoiding the call is the only win.

## F4 — No-op watcher runs write a new storage version (to investigate)

**Observed in `index_history`.** At 15:15 on 2026-11-12 the watcher triggered six runs
within 30 s, each with `changed_files: 0` and `parsed_files: 0`. Each one:
- read 30–34 KB;
- spent 160–235 ms in `commit`;
- advanced the `files` table version by one (`storage_before.files` 2 → 3 → … → 13).

**Likely path.** `indexing.py` stages a metadata-only row when stat changes but the
content hash does not, and restamps rows considered racy. That is a write per run even
when nothing is re-indexed. Whether the restamp repeats on files that are not being
touched is not traced.

**Why it matters.** Every no-op run costs ~200 ms of commit and adds a version that
maintenance has to compact later.

## Reproducing

**Extraction split.** Build `items` as `(path, language, bytes)` from `git ls-files`
filtered through `scanner.LANGUAGES`. Time `Parser(language).parse` alone, then
`TreeSitterExtractor().extract` per file, and read
`ExtractionResult.reference_extraction_ns`.

**GIL convoy:**

```python
app = Application(RuntimePaths(data=..., cache=...), embedder=DeterministicEmbedder(), cwd=repo)
app.init_project(repo, force_new_id=True)
app.index_project(str(repo), force=True)

def burn():  # CPU-bound extraction loop on a second thread
    ex = TreeSitterExtractor()
    while not stop.is_set():
        for path, language, source in items:
            ex.extract(path, language, source)

# time app.search_code("parse configuration file", projects=[str(repo)])
#    + app.find_symbol("main", str(repo)), 30 times, idle vs. with burn() running;
# repeat after sys.setswitchinterval(0.0005)
```

**Bucketed path profile.** Run cProfile around `index_project(force=True)`, a no-op
`index_project`, `project_status`, `search_code` and `find_symbol`. Bucket self time by
filename: project code, LanceDB/pyarrow, tree-sitter, pydantic, subprocess.

## Caveats

- The corpora are small; the largest is 311 files. Confirm F2's split on a Django-sized
  repo before investing in it.
- The test embedder's 4-dimensional vectors understate commit and storage time.
- Measurements ran in-process, without the MCP transport or the daemon broker. F1 in the
  daemon also involves the asyncio loop thread taking the GIL to read requests.
