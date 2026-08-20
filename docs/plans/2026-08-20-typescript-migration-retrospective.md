# TypeScript migration retrospective and retirement

Date: 2026-08-20
Status: retired; retained on the `ts-migration` branch for reference
Pull request: #38
Measured revision: `c64f50965b9d9fc8fb294d4cd27aac47cbb581b0`

## Decision

The TypeScript/Bun migration is retired because it did not meet its primary
product objective: reduce end-to-end indexing time on moderate-to-large
codebases. It reached functional parity and performed well on synthetic CPU
benchmarks, no-op scans, and search, but a matched accelerator run on a real
repository was substantially slower than the shipping Python implementation.

The branch is intentionally preserved rather than deleted. It contains useful
parity fixtures, platform findings, soak tooling, and a complete alternative
implementation, but it is not a release candidate and should not be merged or
published. Future performance work targets the Python implementation.

## Representative benchmark

### Environment

- Host: Linux 7.0.0-29-generic x86_64, glibc 2.43.
- GPU: AMD Navi 31 (`1002:744c`).
- Repository: `easycode` at
  `26f650ef9f379ba106eb87f0e2e258111cf458e5`, with the same dirty worktree
  presented to both implementations.
- Workload: 1,768 eligible files and about 20,200 chunks.
- Backend: strict WebGPU, batch size 8, worker execution required, no CPU
  fallback, and no small-workload accelerator deferral.
- Python: 3.12.13, ONNX Runtime 1.24.4 with a verified
  `WebGpuExecutionProvider` accelerator record.
- TypeScript: Bun 1.3.14 and `onnxruntime-node` 1.27.0.
- Each implementation used an isolated data directory and the same model
  artifact/cache.

The source worktree had these pre-existing changes during both runs:

```text
 M bun.lock
 M package.json
 M specs/code-index.md
?? packages/code-index-probes/
```

### End-to-end results

| Scenario | Python | TypeScript | TS / Python |
|---|---:|---:|---:|
| Cold index | 222.747 s | 325.472 s | 1.461x |
| No-op index | 176.8 ms | 101.0 ms | 0.571x |
| Forced reindex | 231.437 s | 318.640 s | 1.377x |
| Semantic search | 584 ms | 542 ms | 0.928x |
| Cold-run peak memory | 1.10 GB | 1.45 GB | 1.318x |

The cold-run history records show where the 102.725 second end-to-end gap
came from:

| Phase | Python | TypeScript | TS minus Python |
|---|---:|---:|---:|
| Scan | 0.340 s | 0.022 s | -0.318 s |
| Parse | 8.591 s | 16.879 s | +8.288 s |
| Embed | 209.151 s | 294.866 s | +85.715 s |
| Commit | 4.315 s | 12.567 s | +8.252 s |

The forced-reindex history confirms the same shape:

| Phase | Python | TypeScript | TS minus Python |
|---|---:|---:|---:|
| Scan | 0.212 s | 0.291 s | +0.079 s |
| Parse | 8.645 s | 16.786 s | +8.141 s |
| Embed | 217.596 s | 288.471 s | +70.875 s |
| Commit | 4.640 s | 12.206 s | +7.566 s |

Python completed all 1,768 files and wrote 20,193 chunks. TypeScript completed
1,766 files and wrote 20,190 chunks; two SQL migrations failed because
`@derekstride/tree-sitter-sql` had no Linux x64 addon in the local install.
That three-chunk difference is too small to explain TypeScript's slower run.

Both histories recorded `embedding_backend = webgpu`, `worker_used = 1`, and
no fallback reason. The top five semantic search hits matched, as did all eight
structural hits for `Provider`. Both Lance partitions reported consistent
storage.

The retained data directories are:

```text
/tmp/opencode/easycode-webgpu-python.vrIA2z
/tmp/opencode/easycode-webgpu-typescript.zmTEwi
```

They are temporary machine-local evidence, not permanent project artifacts;
the tables above are the durable record.

## Why the earlier benchmark was insufficient

The Phase 9 generated CPU benchmark passed its 15% gate. At 128 generated
files, TypeScript was 1.043x Python on cold indexing and 1.047x on forced
reindexing, while large-scope search was faster. Those results remain valid for
that synthetic workload, but they did not predict accelerator throughput on a
moderate real repository. The generated files were too small to expose the
native-to-JavaScript tensor conversion, pure-JavaScript tokenization, worker
transport, and Arrow construction costs that dominate at scale.

The final benchmark harness changes retained in the retirement commit are
important for reproducibility. They make the requested batch size
authoritative, preserve the verified accelerator record inside isolated
benchmark workspaces, run the real worker path, and disable the production
policy that defers small accelerator workloads to CPU. Earlier accelerator
comparisons without those properties were not comparable.

## Technical findings worth retaining

- Python is already an orchestration layer around compiled ONNX Runtime,
  tokenizer, NumPy, PyArrow, Tree-sitter, and LanceDB code. Replacing that
  orchestration with TypeScript does not make the native kernels faster.
- The TypeScript direct-ONNX path materializes a `[batch, sequence, 768]`
  tensor as JavaScript numbers and performs mean pooling and normalization in
  JavaScript. Python keeps the same operation in vectorized NumPy.
- TypeScript used the pure-JavaScript `@huggingface/tokenizers` package rather
  than Python's compiled tokenizer implementation.
- The worker protocol serializes vectors as base64 inside JSON, and the
  storage path performs additional JavaScript-to-Arrow conversions.
- Moving masked mean pooling and L2 normalization into the ONNX graph remains
  a credible optimization because it reduces GPU-to-host output to
  `[batch, 768]`. It should be evaluated against Python first because it can
  improve either runtime; no graph-rewrite implementation was made.
- Native Rust/C bindings could remove TypeScript-specific overhead, but would
  mostly restore parity. Since embedding consumed about 94% of Python's cold
  run, improving end-to-end time requires improving inference, transfer, batch
  utilization, or pipeline overlap rather than changing the orchestration
  language.
- Float16 persisted vectors remain a requirement. Model execution precision
  and graph-side pooling can be investigated independently without changing
  the `FixedSizeList[768]<Float16>` storage schema.

## Reusable artifacts

The retired branch preserves:

- cross-language extraction, schema, storage, and vector parity fixtures;
- a dual-run soak comparator and benchmark comparison harness;
- platform compatibility findings for Bun, native Tree-sitter grammars,
  LanceDB, and ONNX Runtime;
- a complete TypeScript implementation that can be consulted when changing
  wire or on-disk contracts; and
- the benchmark correction that prevents accelerator runs from silently
  measuring CPU execution.

Code should be cherry-picked from this branch only when it independently
benefits the Python implementation. The migration itself is not to be resumed
without new evidence that it can beat the Python end-to-end baseline.
