# Phase 0 spikes

The six experiments §6 of
[the migration plan](../../../docs/plans/2026-08-17-typescript-migration.md)
asks for before committing to anything. Verdicts are recorded in
[the results document](../../../docs/plans/2026-08-17-phase-0-spike-results.md).

Each spike is a standalone program: run one directly, or all of them through
the runner.

```sh
bun install                       # from ts/
bun run src/run.ts                # every spike
bun run src/run.ts s1 s2          # a subset
bun run src/s0-native-modules.ts  # one, directly
```

Every spike also runs under Node, and several are only meaningful *because*
they do — S0 in particular tells a Bun N-API gap apart from a broken package
by running the same checks under both:

```sh
node src/s0-native-modules.ts
```

Each run writes `results/<spike>-<runtime>-<platform>-<arch>.json` and exits
non-zero if any check failed, so a spike can be promoted to a CI gate the day
it starts passing.

## Spikes needing a fixture

Two spikes compare against artifacts the *Python* build produces, because a
comparison against something this tree generated itself would not answer the
question. Both skip loudly rather than passing quietly when the fixture is
absent.

**S1** — a real Python-written index:

```sh
.venv/bin/python scripts/write_python_index.py /tmp/py-index
S1_PYTHON_INDEX=/tmp/py-index bun run src/s1-lancedb-parity.ts
```

**S3** — Python-produced embedding vectors, plus the model artifact. The model
directory is the HuggingFace snapshot the Python build already downloaded, so
this normally costs nothing extra:

```sh
.venv/bin/python scripts/write_python_vectors.py /tmp/reference.json --offline
S3_REFERENCE_VECTORS=/tmp/reference.json \
S3_MODEL_DIR="$HOME/Library/Caches/code-indexing-mcp/models/models--jinaai--jina-embeddings-v2-base-code/snapshots/<revision>" \
  bun run src/s3-embedding-parity.ts
```

## What outlives Phase 0

Most of this directory is throwaway, but three modules are drafts of real
Phase 1–2 code and should be moved rather than rewritten:

| Module | Becomes |
|---|---|
| `grammar-loader.ts` | the extractor's grammar loading (Phase 2) — it works around packaging bugs that will not be fixed by then |
| `acceptance.ts` | `acceptance.py`'s port (Phase 1), already verified against the Python original |
| `onnx-model.ts` | a test helper for exercising ONNX Runtime without a 640 MB download |
