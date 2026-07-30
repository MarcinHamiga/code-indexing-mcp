# Phase 5 Adaptive Selection and Hardening: Design

## Problem

An accelerator is now prepared by default on two platforms: CUDA on a supported NVIDIA machine and
MLX on macOS 14+ Apple Silicon. Selection is still static — if a backend is prepared and eligible,
every index run pays for it, including the run that re-indexes one edited file.

That cost is real and it is fixed. Starting a passage worker means spawning an interpreter, loading
a 600 MB model onto a device, and running a verification inference. On the machine Phase 4B was
measured on, that is seconds; the run it is being paid for may be a fraction of a second of actual
embedding. The long-term plan's Phase 3 gate already stated the requirement and Phase 4B shipped
without it: *small incremental jobs remain on CPU whenever accelerator startup would make them
slower.*

Three other Phase 5 items are outstanding for the same reason — nothing has ever measured a backend
beyond "it works":

- Batch size is configured, never calibrated. `INCODE_EMBED_BATCH_SIZE=auto` resolves to 1, the
  value CPU indexing has always used, on every backend. `ProbeRecord.batch_size` exists and is
  written as 0, and `PassageBackendSession` carries a `calibrated_batch_size` parameter that
  `Application` explicitly passes 0 to with a comment naming this phase.
- Microbatches are bounded by item count and by a fixed `max_token_product` of 4,096, which was
  measured against the default 2 GiB ceiling and does not move when the ceiling does.
- An out-of-memory batch is halved up to twice within a request and the smaller size is then
  forgotten, so the next run rediscovers the same limit by overrunning the ceiling again.

## Approach

### Measure both sides, once per configuration

Calibration runs from the parent through the existing `plan_and_embed` command, against synthetic
candidates, at a small set of candidate batch sizes. Nothing is added to the worker protocol: the
measurement then covers exactly what indexing experiences, IPC included, and works unchanged for
the FastEmbed, direct-ONNX, and MLX workers.

Two numbers come out of it per backend — the throughput it sustains warm, and the batch size that
sustained it — and one comes from the session that produced them: the cold cost of having the
worker at all, measured from spawn through model load and verification.

Throughput is measured in **characters per second**. Segments are not known in the parent before
embedding, tokens are known only after it, and characters are exactly known for every candidate the
indexer is about to hand over. The unit has to be one the crossover decision can be made in.
Character density varies between corpora — that is the same caveat `token_batching` documents — so
the calibration corpus is code-shaped rather than filler, and the crossover it feeds is a coarse
decision about whether a run is seconds or minutes, not a precise one.

Results are stored in the existing probe cache, whose key already folds in model artifact,
accelerator, provider, runtime version, platform, device, and driver version — exactly the
"per device/model/runtime combination" the plan asks for. The schema version is bumped so records
written before this phase are treated as absent rather than read as uncalibrated zeroes.

CPU is calibrated too, because a crossover needs both rates. It is calibrated **only when an
accelerator was selected**, so a CPU-only machine — where there is no decision to make — pays
nothing.

### Cross over lazily, in the middle of the run

The crossover point follows directly from the three measurements. CPU costs `n / R_cpu`; the
accelerator costs `L + n / R_accel`; they meet at

```
n* = L / (1/R_cpu - 1/R_accel)
```

characters. Below `n*` the run is faster on CPU, above it on the accelerator. If `R_accel <= R_cpu`
there is no crossover at all and the answer is CPU for any size — which is a diagnostic worth
reporting rather than a case to hide.

The indexer streams: it scans, parses, and embeds interleaved, so the size of a run is not known
before the run is over. The decision is therefore made **while embedding**, not before it. A session
that selected an accelerator begins on CPU and keeps a running total of the characters it has
embedded; when that total passes `n*` it closes the CPU worker and brings the accelerator up for the
remainder.

The cost of deciding late rather than perfectly is exactly one accelerator load, `L`, on runs that
turn out to be large — the below-threshold characters are embedded at CPU speed, and by the
definition of `n*` that is the same time the accelerator would have taken to start plus embed them.
The benefit on runs that turn out to be small is the same `L`, and small runs are the common case
for a daemon that indexes on save. Any scheme that avoided the penalty would have to forecast the
size of a run before parsing it, which this pipeline deliberately does not do.

A deferred start is not a fallback. It is not counted in `fallback_count`, does not set
`embedding_fallback_reason`, and does not pin the process to CPU the way a real degradation does —
the next run makes the decision again from its own size.

### Bound batches by memory as well as by count

`max_token_product` becomes a function of the configured indexing ceiling rather than a constant
tied to the ceiling it happened to be measured at, floored so that a single longest-window sequence
always forms a batch and capped so a large ceiling cannot ask for a padded matrix no measurement
supports.

### Remember the limit an overrun found

When a batch retry succeeds at a reduced `max_items`, that size is recorded as the configuration's
calibrated batch size, marked as reduced rather than measured. The next run starts there instead of
rediscovering the ceiling by overrunning it. Nothing raises the limit again on its own; a changed
model, runtime, driver, platform, or device invalidates the key and re-calibrates, and an explicit
`INCODE_EMBED_BATCH_SIZE` overrides it outright. `model status` reports which of the two it is
looking at, so a machine pinned low by one bad run says so.

### Report all of it

`model status` gains the measured rates, the cold-load cost, the crossover, and — when the numbers
say something actionable, such as an accelerator that lost to CPU — a recommended override.
`IndexReport` gains the characters embedded, the crossover in force, and the reason the run used
the backend it used, so a run that stayed on CPU by design is distinguishable from one that fell
back to it after a failure.

## Non-goals

- No forecasting of run size before parsing, and no persistence of "this project is large".
- No change to vectors, model identity, pooling, normalisation, or index compatibility. Crossing
  over mid-run mixes CPU and accelerator vectors within one run, which the parity gates already
  cover: a backend is only promoted when its vectors match CPU, and the existing fallback has always
  mixed them.
- No new download, install, or driver action of any kind. Calibration embeds through a worker that
  is already running.
