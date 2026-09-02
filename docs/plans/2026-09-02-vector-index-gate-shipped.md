# Vector-Index Size Gate — Measurements and Decision (2026-09-02)

Plan: `2026-09-02-vector-index-gate-plan.md`. Harness: `scripts/benchmark_vector_recall.py`.
Raw output: `data/2026-09-02-vector-recall-django-sizes.json` (size sweep, `nprobes`
only) and `data/2026-09-02-vector-recall-django-grid.json` (query-setting grid).

**Setup.** 56,266 real `jina-embeddings-v2-base-code` vectors (768-d, float16 as
stored) read from the Django index left by the profiling run; 200 queries (40
hand-written natural-language searches, 160 symbol-and-path phrases) embedded with the
project's query embedder; recall@10 against exact float32 cosine over the same vectors.
Apple M4 Pro. Sizes above 56k are real vectors plus Gaussian-perturbed copies and are
marked synthetic.

## Size sweep (production query settings: default `nprobes`, default `ef`, no refine)

| Rows | Flat median ms | HNSW build s | Index MB | HNSW median ms | recall@10 |
|---:|---:|---:|---:|---:|---:|
| 20,000 | 3.2 | 0.6 | 17.8 | 0.73 | 0.860 |
| 50,000 | 6.9 | 2.0 | 44.3 | 0.81 | 0.854 |
| 56,266 (all real) | 6.4 | 2.1 | 49.8 | 0.75 | 0.847 |
| 100,000 (synthetic) | 10.3 | 4.9 | 84.3 | 0.77 | 0.842 |
| 200,000 (synthetic) | 20.1 | 10.2 | 163.3 | 0.78 | 0.871 |

`nprobes` 50 and 100 changed nothing at any size: the loss is not from IVF partitions.

## Query-setting grid at 56,266 real rows (flat: 6.4 ms)

| `ef` | `refine_factor` | median ms | recall@10 |
|---|---|---:|---:|
| default | none | 0.78 | 0.869 |
| default | 2 | 1.04 | 0.958 |
| default | 5 | 1.26 | 0.996 |
| 100 | none | 0.83 | 0.973 |
| **100** | **2** | **1.10** | **0.999** |
| 300 | 2 | 1.33 | 1.000 |

The same grid at 20,000 rows lands within 0.01 of every cell. The loss is scalar
quantisation plus the default beam width; a small refine over the stored vectors
removes it for about 0.3 ms.

## Decision (plan D5)

- **The gate does not move and `exact` stays the default.** A flat cosine scan over
  every real chunk in a 3,100-file repository costs 6 ms; the tool call around it costs
  about 230 ms on the remediated branch, nearly all of it in git spawns
  (`2026-09-02-query-path-profiling-shipped.md`). Even the synthetic 200k point is
  20 ms. No measured size makes the scan worth an approximate index by the plan's rule.
- **`hnsw` mode as shipped loses 13% of the top ten.** With the store's current query
  (no `ef`, no `refine_factor`) recall@10 is 0.85–0.87 at every size. That is the
  finding to act on: when the operator opts into `hnsw`, the store must issue
  `ef=100` and `refine_factor=2`, which restores recall to 0.999 at 1.1 ms. These are
  module constants, not settings (plan D6).
- **The threshold gets a name.** `chunks.count_rows() >= 20_000` in `storage.py`
  becomes `VECTOR_INDEX_MIN_ROWS = 20_000`; the number is unchanged because nothing
  measured argues for another one, and the constant gives the next measurement one
  place to edit.

## Applied (plan Step 3–4, done 2026-09-03)

1. `storage.py`: `VECTOR_INDEX_MIN_ROWS = 20_000`, `VECTOR_INDEX_EF = 100`,
   `VECTOR_INDEX_REFINE_FACTOR = 2` as module constants with the measurement in
   their comments; `_hybrid_search_rows` issues `.ef(VECTOR_INDEX_EF)`
   `.refine_factor(VECTOR_INDEX_REFINE_FACTOR)` whenever the mode is not
   `exact`.
2. `tests/test_storage.py`:
   `test_hybrid_query_vector_knobs_follow_the_index_mode` spies the query
   builder and asserts exact mode bypasses while hnsw mode issues the measured
   pair; `test_vector_index_gate_reads_vector_index_min_rows` asserts the index
   is built only in hnsw mode and only at or above the constant. (The plan
   assumed an existing test pinned the 20k literal; none did, so the gate test
   is new.)
3. README, `CODE_INDEXING_VECTOR_INDEX`: the storage section states `exact` is
   the default because a full scan of the 56k-chunk measurement index costs
   ~6 ms, that `hnsw` is for indexes well past 20,000 chunks, and quotes
   recall@10 0.999 at the settings the store now issues. The two older
   "HNSW-SQ8 gates did not pass" / "safer default" mentions were updated to
   match.

## When to re-measure

When a real index passes about 500k chunks, or when the embedding model changes.
Index more repositories into one data directory and pass every `--data-dir` to the
harness; do not trust the synthetic tail for the decision.
