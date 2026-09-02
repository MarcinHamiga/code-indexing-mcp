# Vector-Index Size Gate — Plan

**Goal:** Decide, from measured recall and latency on real embeddings, where the
approximate vector index should switch on and whether it should ever switch on by
default. Today `LanceStore` builds an `IVF_HNSW_SQ` index once a partition holds 20,000
chunk rows (`storage.py`, the `chunks.count_rows() >= 20_000` branch in the index
maintenance path), but only when `CODE_INDEXING_VECTOR_INDEX=hnsw`; the default `exact`
mode drops any such index and every search bypasses it. The review deferred the gate
question until recall had been benchmarked at scale.

**Review finding closed:** the deferred "vector-index size gate needs a recall benchmark
at scale first" item.

**Baseline:** `uv run ruff format --check . && uv run ruff check . && uv run mypy src &&
uv run pytest -n auto`. Green before Step 0 and after every step.

## Decisions settled before implementation

- **D1 — Real vectors, not random ones.** Random vectors are the worst case for a graph
  index and would understate recall. The benchmark reads the `vector` column straight
  out of existing index partitions (`<data>/lancedb/projects/<partition>/chunks.lance`),
  which is what the profiling plan leaves behind for Django. Stored vectors are float16;
  the exact reference is computed over the same float16-derived values in float32 so the
  measurement isolates the approximate index from the storage precision that
  `benchmark precision` already covers.
- **D2 — Queries are the tool's queries.** Forty hand-written natural-language code
  searches plus 160 short phrases derived from randomly chosen chunks (`path stem` and
  `symbol`, camel/snake split into words), all embedded with the project's query embedder.
  Recall@10 is overlap with the exact cosine top-10, averaged over queries.
- **D3 — Sizes.** Real subsets at 20k, 50k, 100k, and every real vector available.
  Larger sizes, if wanted, are reached by perturbing real vectors with Gaussian noise
  (σ = 0.05) and renormalising; those rows are marked `synthetic` in the output and are
  an indication of how the manifold scales, not a measurement. More real vectors means
  indexing more repositories; do that before trusting any synthetic point.
- **D4 — What gets measured per size.** Flat search latency with
  `bypass_vector_index()` (what `exact` costs today); `IVF_HNSW_SQ` build seconds and
  index bytes; and for `nprobes` in {default, 50, 100} the approximate search latency and
  recall. The store issues no `nprobes`, so "default" is the production setting.
- **D5 — Decision rule.** The gate moves to the smallest size at which flat median
  latency exceeds a budget the search path can feel (the profiling plan reports what a
  whole `search_code` call costs; the vector scan should not dominate it) **and** the
  approximate index at the default `nprobes` keeps recall@10 at or above 0.95. If no
  measured size satisfies both, the gate stays where it is and `exact` stays the default;
  the shipped note says so with the numbers. If a size does, the constant becomes a named
  `VECTOR_INDEX_MIN_ROWS` in `storage.py`, and whether `hnsw` becomes the default above
  it is decided by the recall number alone.
- **D6 — Tunables stay out of the schema.** If `nprobes` turns out to matter, the store
  sets it from a module constant chosen by the benchmark, not from a new setting. One
  environment variable (`CODE_INDEXING_VECTOR_INDEX`) is enough surface.

## Steps

**Step 0 — Harness.** `scripts/benchmark_vector_recall.py` per D1–D4, ruff-clean, JSON
output with model id, dimension, number of real vectors, sources, query counts, and one
entry per size.

**Step 1 — Run on Django's vectors.** Point it at the profiling run's data directory.
Add `--augment-to 200000 --augment-to 500000` for the synthetic tail.

**Step 2 — Decide (D5).** Record the table and the decision in
`2026-09-02-vector-index-gate-shipped.md`.

**Step 3 — Apply.** If the gate moves: replace the literal with `VECTOR_INDEX_MIN_ROWS`,
update the test that pins the threshold (`tests/test_storage.py`, search for `20_000`),
and note the number in the README's `CODE_INDEXING_VECTOR_INDEX` paragraph. If it does
not: still name the constant, so the next measurement has one place to change.

**Step 4 — Docs.** README: one paragraph on when to set `hnsw`, with the measured recall
at the sizes it was measured.
