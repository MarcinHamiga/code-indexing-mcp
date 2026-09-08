# Indexing batching and embedding reuse results

The implementation is on branch `feat/indexing-batching-embedding-reuse`.

The exact token ordering regression reduces the four item example from 2,042
padded token slots under the old power of two ordering to 1,538 slots. The
candidate reuse integration covers cold cache, partial hits, all hits after a
Unicode insertion, cache reopen persistence, forced bypass, absent tokenizer
bypass, cache failure fallback, current source offsets, and history fields.

The durable cache is stored at
`<cache-directory>/passage-embeddings.sqlite3`. Its logical payload is capped
at 256 MiB, individual entries at 64 KiB, and the SQLite page file at 512 MiB.
It uses rollback journaling and disables itself for the current run after an
I/O, locking, or database failure.

Validation completed during implementation:

- focused cache, indexing, worker, backend, and history tests: passed;
- application, benchmark, and CLI tests: passed;
- Ruff formatting, Ruff checks, and mypy: passed.

Representative large repository timing comparisons and real CPU/MLX model
acceptance runs require a prepared local model and accelerator environment. They
were not run in this workspace, so the cold-cache overhead and warm-edit wall
time acceptance thresholds remain open for a machine with those environments.
