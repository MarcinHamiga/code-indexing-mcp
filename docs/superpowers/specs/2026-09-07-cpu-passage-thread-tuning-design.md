# CPU passage thread tuning: next indexing-performance PR

Status: proposed for review; implementation has not started.

Analyzed revision: `dfdc6d0fa0a7ee1daabde20c7912d261fefc92fe`, branched from
`tabbed-configure-wizard` into `plan/indexing-speed`. This is the user's current
checkout, not a claim about the latest remote main branch. Rebase and recheck the
affected symbols before implementation if the PR will target a different base.

## Recommendation

Make CPU passage inference choose a measured thread count instead of always
using the conservative two-thread default. Keep the existing batch planner,
model, token windows, worker isolation, and storage format. Tune only when a CPU
indexing run is large enough to justify measurement; reuse the result thereafter.

The measured opportunity is **1.44× end-to-end speed on a small real-source
corpus using four threads instead of two**. That is a configuration experiment,
not a measured speedup from an implemented automatic tuner. The proposed tuner
must earn its own cold-cache and warm-cache acceptance results.

## Evidence and indexing-path analysis

The indexed source tree matched the clean checkout when analysis began. Code
queries used the existing main-checkout index; the worktree contains identical
source at the analyzed revision. Markdown, Git history, runtime dependency
metadata, model-cache paths, and exact setting literals were inspected locally.

| Stage | Current implementation | Implication for the next PR |
| --- | --- | --- |
| Scan and change detection | `indexing.py:1251`, `indexing.py:1396` | Existing metadata and hash checks avoid unchanged-file work. |
| Extraction | `indexing.py:1251`; previous extractor performance work in `docs/plans/2026-07-27-extractor-performance.md` | Parsing is already a small fraction of the observed full build. |
| Cross-file aggregation | `indexing.py:121`, `indexing.py:1148` | Already groups candidates into bounded 256-item / 256-KiB-content requests; proposing this again would duplicate shipped work. |
| Token windowing and packing | `token_batching.py:132`, `token_batching.py:166`, `embedding.py:206` | Already windows and groups by length. Larger batches are not automatically faster. |
| CPU inference | `embedding_worker.py:133`, `embedding_worker.py:181`; `settings.py:245` | Worker passes the configured thread count into FastEmbed; default is at most two. |
| Backend tuning | `passage_backend.py:302`, `passage_backend.py:349`, `passage_backend.py:392`, `passage_backend.py:425` | CPU-only selection bypasses accelerator verification/calibration. CPU reference measurement runs only after a verified accelerator. |
| Commit | `indexing.py:1534`; shipped storage remediation document dated 2026-08-11 | Bounded batched commits already exist; the sampled full build spends little time here. |

All source paths above are relative to `src/code_indexing_mcp/` unless prefixed
with `docs/`. Detailed baseline observations are saved beside the plan in
`../plans/data/2026-09-07-index-history.json`.

Historical full CPU build `405630dfc2794f8eb4a026f2ce8cd9e4`:

- 261 files, 5,295 embedded chunks, approximately 783.675 seconds wall time.
- Embedding 759.514 s (96.9% of wall time), parsing 2.601 s, scan 0.104 s,
  commit 1.154 s. These phase totals do not cover all wall time.
- The audit identifies running server revision `715efee`, not the newly created
  worktree revision. It does not preserve every effective setting or host-load
  condition, so it establishes priority, not a controlled baseline.
- Removing all measured scan, parse, and commit time would save only 3.859 s,
  about 0.49% of this run. Even eliminating every non-embedding cost would save
  only about 3.1%. Pipelining those stages cannot explain a large speedup here.

### Controlled configuration probes

Host: AMD Ryzen 7 9800X3D, Linux x86_64, Python 3.12.13, FastEmbed 0.8.0,
ONNX Runtime 1.24.4. Cached model:
`jinaai/jina-embeddings-v2-base-code`, snapshot
`516f4baf13dec4ddddda8631e019b5737c8bc250`. CPU arena disabled, 2 GiB worker
ceiling, 1,024-token windows, 64-token overlap, 4,096 token-product budget.
No model downloads, accelerator, serving-index writes, or source-code changes.

The worker probe uses the existing 16-candidate calibration corpus (15,744
characters including prefixes), one untimed warmup and three measured requests
per setting. Each configuration starts a fresh worker. Medians:

| Threads | Max items | Request time | Speedup vs initial 2/1 |
| ---: | ---: | ---: | ---: |
| 2 | 1 | 2.714 s | 1.00× |
| 4 | 1 | 1.641 s | 1.65× |
| 8 | 1 | 1.760 s | 1.54× |
| 2 | 2 | 3.221 s | 0.84× |
| 4 | 2 | 1.944 s | 1.40× |
| 8 | 2 | 1.515 s | 1.79× |
| 2 | 1, final control | 2.789 s | 0.97× |

The final control differs by 2.8%, so results are approximate, not a hardware
leaderboard. Some baseline tests overlapped this exploratory probe. Every
configuration returned the same segment count, no retries, and minimum
float32 cosine similarity reported as 1.0 against the initial baseline.
This is a small parity check, not a substitute for retrieval acceptance.
Worker sampled peak RSS stayed below 0.95 GB. Telemetry counts include the
warmup and three measured requests; the timed numerator is 16 candidates.

Full-path check: copy the unmodified `embedding.py`, `token_batching.py`, and
`calibration.py` into a temporary project. Use a fresh store and disposable
worker per run, batch size one, and no calibration. Run thread counts
`2, 4, 4, 2, 2, 4`; take the median of each three runs.

| Metric | Two threads | Four threads |
| --- | ---: | ---: |
| Index call wall time | 7.576 s | 5.263 s |
| Reported embedding time | 7.295 s | 4.988 s |
| Embedded chunks / tokens | 44 / 7,813 | 44 / 7,813 |
| Staged structural rows | 823 | 823 |
| Maximum reported peak RSS | 1,404,145,664 B | 1,484,214,272 B |
| Errors / retries | 0 / 0 | 0 / 0 |

This is **30.5% less wall time, or 1.44× throughput**. It includes worker startup,
parsing, embedding, staging, and commit; it excludes Application construction,
project registration, and model download. OS/model file caches are warm and
stores are empty. The initial outside-sandbox indexing-test run overlapped the
early samples; the result is directional and must be repeated under release
benchmark conditions. Raw JSONL and both reproducer scripts are in
`../plans/data/2026-09-07-cpu-thread-{probe,e2e}.{py,jsonl}`.

### Alternatives

1. **CPU passage thread tuning — recommended.** Targets the dominant stage with
   a measured gain and no embedding-model migration. The main engineering risk
   is calibration overhead and lifecycle/cache correctness.
2. **Joint thread/batch tuning and tighter padding.** The 8-thread/2-item result
   is promising, but larger batches slowed both 2-thread and 4-thread probes.
   A two-dimensional sweep needs more measurements and memory validation. Keep
   the current batch policy for this PR; revisit after the thread-only change.
3. **Vector reuse, parallel extraction, or streaming parse/embed overlap.**
   Vector reuse primarily helps edits and rebuilds with repeated inputs; it
   does not accelerate inference for novel passages and requires contextual
   cache identity/invalidation. Parse/storage parallelism has a low observed
   ceiling here. These are separate follow-ups, not part of this PR.

Accelerator backends already exist. Promoting a new provider or changing the
model/quantization would need different hardware and parity evidence; neither
is necessary for the measured CPU improvement.

## Proposed behavior

### Settings and scope

- Extend `CODE_INDEXING_EMBED_THREADS` to accept `auto|1..64`; unset means auto.
- Retain `IndexSettings.embedding_threads: int` as the conservative resolved
  value and add `embedding_threads_auto: bool = False`. Environment parsing sets
  the flag for unset/auto; direct dataclass construction stays compatible.
- Query inference keeps the current conservative resolution. Only CPU passage
  workers adopt a tuned value. An explicit integer continues to control both
  as it does today. Explicit integers never initiate thread calibration.
- `CODE_INDEXING_EMBED_CALIBRATE=0` prevents new measurements. A compatible
  cached result may still be used, matching the distinction between measuring
  and reusing a result. With no record, use the conservative value.
- First PR scope is CPU-only worker selection. Accelerator, provisional CPU,
  accelerator-fallback, and `index_execution=in-process` behavior remain on
  their existing path. Their factories do not opt into this tuner.
- Keep automatic batch sizing at its existing value. Pin the effective
  `SegmentPlan.max_items` during a thread sweep; include it in cache identity.

### Bounded tuning

- On a CPU-only run with no compatible record, keep using the baseline until
  successfully embedded candidate characters plus the next request reach
  512 KiB. Count prefixes consistently with `characters_embedded`. No-op and
  short jobs never start an extra worker to tune.
- Attempt tuning once per run. Candidate counts are 1, 2, 4, and 8, capped by
  the effective CPU allowance (logical availability, process affinity where
  supported, and readable Linux CPU quota). Include the conservative baseline.
- Use eight deterministic, stratified candidates from the existing calibration
  corpus, one warmup and three timed requests per configuration. Keep sample,
  batch size, token budgets, arena setting, and memory ceiling fixed.
- Measure through `EmbeddingWorkerSession.plan_and_embed`, including IPC.
  Only one passage model may be resident: close the production CPU worker
  before opening a trial; close every trial before the next. Recreate the
  selected worker for real work after measurement.
- A monotonic 15-second overall budget bounds the sweep, including loads and
  warmup. Add an absolute request deadline so a slow in-flight request cannot
  exceed this merely because the next between-trial check has not run. The
  existing memory limit and worker teardown still apply.
- Only fully completed three-sample trials compete. Prefer the baseline unless
  a candidate improves median request time by at least 10%; prefer fewer
  threads on ties. Persist baseline too, so an unhelpful sweep is not repeated.
- Failures, OOM, or deadline expiry during a trial never commit trial output or
  become content failures. Retain a previously completed safe winner, or the
  baseline when no winner exists. Verify real work still succeeds after the
  failed trial. Cancellation of the indexing run propagates and closes workers.
- Apply the winner to the same run's remaining requests and future CPU-only
  runs. Keep tuning counters separate from corpus counters and keep the actual
  overall peak RSS; report tuning time separately instead of hiding its cost.

The 512-KiB trigger, 15-second cap, and 10% threshold are initial design choices,
not measured optima. Task 4 must reject or adjust them if cold-cache benefit
does not repay the sweep. A single small request is not evidence to change a
machine-wide default.

### Cache and observability

Use a separate bounded CPU-thread-tuning record instead of treating the current
accelerator probe record as a thread measurement. Its identity includes model
artifact, model ID, runtime/FastEmbed versions, CPU identity, effective CPU
allowance, arena flag, memory ceiling, token window/overlap/product limits,
effective batch size, and a tuning-policy version. Validate records on read;
corruption or a changed identity means a cache miss. Write atomically under a
lock and retain at most 64 newest records. Store finite measured durations,
chosen threads, measured sample size, and termination/selection reason.

Do not overwrite the CPU reference rate used by accelerator crossover: it was
measured with a different worker configuration. Keeping this PR CPU-only avoids
feeding a four-thread rate into a two-thread fallback decision.

Add optional/defaulted report/status fields for effective passage threads,
selection source (`explicit`, `default`, `cache`, `measured`), and tuning duration.
An unstarted CPU worker reports no effective measurement. Query thread settings
and actual passage thread settings must be distinguishable. Update the shared
installer setting specification to `auto_int`, so both installer surfaces use
the same validation and explanation.

## Non-negotiable constraints

- Python >=3.12,<3.14; no new runtime dependencies.
- No model, tokenizer, prefix, window-boundary, vector-dimension, precision,
  storage-schema, or retrieval-contract change.
- Preserve bounded worker memory, disposal, cancellation, per-file failure
  retention, and transactional commit/recovery.
- Explicit thread counts 1..64 remain authoritative.
- Calibration never adds indexed chunks, reference rows, or user-file errors.
- At most one passage model is resident during tuning and indexing.
- Benchmark fresh indexes and forced rebuilds as well as no-op and small edits.
- Run the repository formatter, lint, typecheck, and full CI test gate before push.

## Impact and coverage limitations

Structural references confirm the existing `calibrate` calls in
`passage_backend.py:411` and `:448`; `ProbeKey` construction is in
`backend_coordinator.py:170` and `:192`. These remain relevant regression
boundaries, not invitations to refactor the accelerator path. Tests using
direct imports were reported as likely/unproven reexports by the index; their
source and outlines were also inspected. JSON, YAML, and Godot-resource files
are not structurally covered. Exact thread-setting literals were checked in
the installer, README, and settings tests. Dynamic uses are not exhaustively
proven by the structural index.

Release evidence is still required on another CPU architecture and a constrained
CPU allocation. The current results do not establish GPU performance, retrieval
parity on a full corpus, or a speedup for the proposed automatic policy.
