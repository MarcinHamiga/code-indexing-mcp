# Phase 0 spike results

Date: 2026-08-17
Status: S0–S5 run; all pass
Branch: `ts-migration`
Plan: [2026-08-17-typescript-migration.md](2026-08-17-typescript-migration.md) §6

The follow-up §6 of the migration plan asks for. Each spike is a runnable
program under [`ts/packages/spikes/`](../../ts/packages/spikes/); re-run any of
them to reproduce a verdict, and see that directory's README for the two that
need a Python-produced fixture.

## Verdicts

| Spike | Verdict | Consequence |
|---|---|---|
| S0 — Bun native-module matrix | **Pass** | All four addons load and run under Bun. No process needs to fall back to Node. |
| S1 — LanceDB Node parity | **Pass** | Not blocking. The JS SDK expresses every index configuration the write path uses, and reads Python-written indexes including their FTS indexes. |
| S2 — Grammar coverage | **Pass** on macOS and Linux; **skip** on Windows | All 18 languages load and every committed `.scm` compiles unchanged. `gdshader` comes from a language pack, as it does in Python — but that pack does not load on Windows, the one gap Phase 0 found. |
| S3 — Embedding parity | **Pass** | **Resolves D1: keep indexes.** Vectors are identical to Python's to 9 significant figures. |
| S4 — Memory ceiling | **Pass** | The polling/kill/retry mechanism ports on all three platforms, and `pidusage` agrees with `psutil` exactly. |
| S5 — Watcher and scanner | **Pass** | `@parcel/watcher` holds the properties the auto-indexer relies on. |

Measured on macOS 15 / arm64, Bun 1.3.14, Node 26.7.0, and confirmed on
ubuntu-latest and windows-latest / x64 through the CI matrix
(`.github/workflows/ci-ts.yml`).
S1, S3, and S4 skip on CI because their fixtures (a Python-written index, the
model artifact, the project virtualenv) are not present there.

Linux confirmed S0 and S2 in full — every native addon loads, all 18 grammars
resolve, `@derekstride/tree-sitter-sql` compiles from source on the runner, and
the language pack downloads `gdshader` — and surfaced two real behavioral
differences in S5, recorded under that spike.

Windows confirmed S0 and S5 in full, and S2 for 17 of 18 languages. It found
one genuine gap: **the language pack cannot load its `win32-x64` binding**, so
`gdshader` is unavailable there. See "The one platform gap" below.

## S0 — Bun native-module matrix — PASS

Every addon loaded and completed a real operation under **both** Bun and Node:

| Module | Real call exercised | Bun | Node |
|---|---|---|---|
| `@lancedb/lancedb` 0.37.1 | create a table, reopen it, count rows | pass | pass |
| `tree-sitter` 0.25.1 + grammar | parse a file, compile and run a query | pass | pass |
| `onnxruntime-node` 1.27.0 | run an inference | pass | pass |
| `@parcel/watcher` 2.6.0 | subscribe and observe a write | pass | pass |

The ONNX check runs a genuine inference rather than merely loading the addon:
`src/onnx-model.ts` encodes a one-node `Identity` graph as ONNX protobuf in
memory, so ORT's native execution path is exercised without a model download.

**Consequence for the plan.** The first row of the risk register (§10, "a
native addon hits a Bun N-API gap") does not fire. §3's Node-compat discipline
stays worth keeping as insurance, but no process needs to run under Node today.
Decision D5 (`bun build --compile`) is unaffected and still opens at Phase 8.

## S1 — LanceDB Node parity — PASS

The blocking spike clears. All seven checks pass:

- The chunks schema round-trips, `vector` included, as
  `FixedSizeList[768]<Float16>` — the float16 storage default from schema
  version 5.
- `Index.fts({ lowercase: true, stem: false, removeStopWords: false })` is the
  exact analogue of `FTS(lower_case=True, stem=False, remove_stop_words=False)`.
- `Index.btree()` on all four scalar filter columns; `Index.hnswSq({
  distanceType: "cosine" })` builds as `IvfHnswSq`.
- A hybrid query spanning both FTS columns via `MultiMatchQuery` with
  `Operator.Or`, RRF-reranked.
- The `regexp_like(path, …)` predicate `path_filter.py` emits pushes into the
  scan and does not leak non-matching rows.
- **An index written by the Python build (lancedb 0.34.0, pyarrow 23.0.1) opens
  through the JS SDK**, exposing all 6 Python-built indexes, and a full-text
  query against the *Python-built* FTS index returns the right rows.

**Consequence.** The §10 "LanceDB JS lacks the index-config API" risk is
retired; no upstream contribution window is needed, and the schedule buffer it
represented can be reclaimed.

### Two API differences Phase 3 must carry

1. **`apache-arrow` is pinned by LanceDB.** `@lancedb/lancedb@0.37.1`
   peer-requires `apache-arrow >=15.0.0 <=18.1.0`; the current release is
   21.2.0. §4's dependency table should read **18.1.0**, not "apache-arrow JS".
   This also constrains the Arrow IPC staging format in Phase 3.
2. **Hybrid query builder order is not Python's.** `nearestToText` is declared
   on `Query` and `nearestTo` returns a `VectorQuery`, so the text leg must be
   attached *first*. Reversing them is a runtime `TypeError`, not a type error.
   Python's `.rerank()` also defaults to RRF while JS requires an explicit
   `rerankers.RRFReranker.create()` — confirm the default `k` matches when
   ranking parity is measured in Phase 9.

Noted for later: LanceDB warns that `_distance`/`_score` autoprojection into
selected columns is deprecated. Phase 3 should call
`disable_scoring_autoprojection` and select those columns explicitly.

## S2 — Grammar coverage — PASS (18 of 18 languages)

All 18 languages in `extractor.py::_languages` were loaded, the committed
`queries/*.scm` compiled against each, and each was run over its fixture from
`tests/fixtures/extractor_corpus/`. **Every committed `.scm` compiled without
edits**, chunk queries and reference queries alike — the query packs port as-is.

| Language | npm source | Status |
|---|---|---|
| python, java, javascript, typescript, tsx, go, rust, c, cpp, json | `tree-sitter-*` (unscoped) | pass, via package entrypoint |
| csharp | `tree-sitter-c-sharp` | pass, via loader fallback |
| terraform | `@tree-sitter-grammars/tree-sitter-hcl@1.2.0` | pass, via loader fallback |
| yaml | `@tree-sitter-grammars/tree-sitter-yaml@0.7.1` | pass, via loader fallback |
| lua | `@tree-sitter-grammars/tree-sitter-lua@0.4.1` | pass, via loader fallback |
| sql | `@derekstride/tree-sitter-sql@0.3.11` | pass, builds from source |
| gdscript | `tree-sitter-gdscript@6.1.0` (prestonknopp) | pass |
| godot_resource | `tree-sitter-godot-resource@0.7.0` (prestonknopp) | pass |
| gdshader | `@kreuzberg/tree-sitter-language-pack@1.10.9` | pass, via the language pack |

Sourcing notes, against §4's "GDScript/gdshader/godot-resource and HCL/SQL need
sourcing":

- **HCL and YAML resolve exactly.** The unscoped `tree-sitter-hcl` on npm is a
  `0.0.1-security` placeholder, not a grammar; the real one lives under
  `@tree-sitter-grammars/`, at the same 1.2.0 the PyPI package pins. Same for
  YAML at 0.7.1.
- **SQL resolves exactly** to `@derekstride/tree-sitter-sql@0.3.11`, the same
  version and upstream as PyPI's `tree-sitter-sql`.
- **GDScript and godot-resource resolve** to the prestonknopp grammars, which
  is what `tree-sitter-language-pack` builds from — so no pack analogue is
  needed for two of the three Godot formats.
- **Lua drifts**: npm 0.4.1 against PyPI 0.5.x, same upstream org. The
  committed query compiles and captures, so this is a version note rather than
  a gap; confirm capture parity when Phase 2 diffs chunk output.
- **gdshader has no dedicated npm package**, and is taken from
  `@kreuzberg/tree-sitter-language-pack` instead — the npm analogue of the PyPI
  `tree-sitter-language-pack` the Python build already uses, and for exactly
  the same reason. It publishes 306 grammars, ships native addons for all six
  platform triples in one package, and fetches each grammar's shared library on
  first use (so CI warms its cache, as the Python workflow already does for the
  Godot grammars).

  Verified end to end rather than assumed: the pack's `getLanguage("gdshader")`
  is accepted directly by node-tree-sitter's `Parser`, our committed
  `queries/gdshader.scm` compiles against it, and it yields **14 captures over
  10 named children on `water.gdshader` — byte-identical to what the Python
  pack produces on the same fixture**.

  It is used *only* for gdshader. Every other language keeps its dedicated
  package so its version moves independently, which is the rule
  `pyproject.toml` already records for the PyPI side — and gdscript and
  godot_resource both have dedicated npm packages, so only one of the three
  Godot formats needs the pack here (versus all three in Python).

### Grammar loading needs a shim under Bun (and Phase 2 inherits it)

Five languages failed under Bun but passed under Node. Neither cause is a Bun
N-API gap — both are bugs in the grammar packages' own
`process.versions.bun` fast paths:

- **Scoped packages build the wrong filename.** `@tree-sitter-grammars/*`
  require `prebuilds/<platform>/tree-sitter-yaml.node` while shipping
  `@tree-sitter-grammars+tree-sitter-yaml.node`.
- **ESM-flavored packages use `import` on a `.node` file.**
  `tree-sitter-c-sharp` does `await import(…".node")`, which Bun rejects:
  "To load Node-API modules, use require() or process.dlopen instead of import."

`src/grammar-loader.ts` resolves both by falling back to requiring whatever
`.node` file actually sits in the platform's prebuild directory (or in
`build/Release` for source-built grammars). With it, **Bun matches Node exactly**.
That module should move into the extractor in Phase 2 rather than be rewritten.

Separately, **Bun does not run install scripts for untrusted packages**.
`@derekstride/tree-sitter-sql` compiles its parser in an `install` script, so it
needs a `trustedDependencies` entry — already added to the spikes package, and
required for any grammar that builds from source.

### Scope note

The plan's S2 also asks to diff chunk output against `extractor_snapshot.json`.
That is not a Phase 0 artifact: the snapshot records the *chunking algorithm's*
output, and that algorithm is 1,583 lines of Phase 2 work. The spike records
per-language capture counts instead, as the baseline Phase 2 diffs against.

## S3 — Embedding parity — PASS — resolves D1

Two checks, both passing:

1. **The ported acceptance metrics match the Python originals.**
   `ts/packages/spikes/src/acceptance.ts` reproduces `cosine_rows` to within
   **8.5e-8** of `acceptance.py` on a committed fixture, and `top_k_overlap`
   identically at k=1 and k=5. This is checked first because a parity verdict
   computed by a broken metric is worse than no verdict.

2. **TypeScript vectors match Python vectors.** Embedding a 10-document probe
   corpus through `onnxruntime-node` + `@huggingface/tokenizers` with
   `direct_onnx.py`'s mean-pool-and-normalize, against vectors produced by the
   Python `FastEmbedder` on the same pinned artifact:

   | Metric | Measured | Promotion gate |
   |---|---|---|
   | min cosine | **1.00000000** | ≥ 0.999 |
   | top-5 overlap | **1.0000** | ≥ 0.99 |

   Scored with the gates `test_accelerator_acceptance.py` already enforces,
   not a metric invented for the migration.

**Consequence.** §5.2 resolves in the good direction and **D1 resolves to "keep
indexes"**: migrated installs need no rebuild. The result is unsurprising in
hindsight — same ONNX graph, same tokenizer, same pooling, float32 throughout —
which is the point of §5.1's "the direct ONNX path becomes the only path".

This also confirms `@huggingface/tokenizers@0.1.3` reproduces the Python
tokenizer's output; §4's row can name it instead of "HF `tokenizers` bindings".
Note its `Encoding` fields are `ids`/`attention_mask`, not `input_ids`.

**Still open: D2 (Apple Metal).** S3 measured the CPU provider only. The
CoreML-EP comparison the plan hoped S3 would supply has not been run, and D2
remains open for Phase 7.

## S4 — Memory ceiling — PASS

The enforcement mechanism ports intact:

- `pidusage` tracks a child's growing resident set (1 MB → 128 MB over 8 samples).
- A ceiling trips and the child is killed short of its target (150 MB ceiling,
  tripped at 160 MB, ended on SIGKILL).
- The halving retry converges: 512 → 256 → 128 MB, completing on the third
  attempt under a 200 MB ceiling — `embedding_worker.py`'s retry shape.
- **`pidusage` and `psutil` agree to 0.00%** (288.2 MB vs 288.2 MB) when both
  measure the same quiesced child.

**One caveat Phase 4 inherits: polling has a sampling race.** Windows allocates
fast enough that a child could run to completion between two 25 ms polls. That
showed up twice — as a check reporting zero samples, and more insidiously as a
256 MB child appearing to *fit* under a 200 MB ceiling because its peak was
never observed. An unobserved peak reads exactly like a peak that never
happened.

The spike now holds each child at its peak so the mechanism is measured rather
than the scheduler, and refuses to trust a ceiling verdict it took no samples
for. Phase 4 needs the same property in earnest: `embedding_worker.py`'s real
batches allocate over a long enough window that this is unlikely to bite, but
"unlikely" is not the same as "checked", and a ceiling that silently fails open
is worse than no ceiling.

**Consequence.** The §10 risk "child RSS accounting differs from `psutil`
semantics" is retired: the ceilings in `settings.py` carry over numerically
unchanged. Comparing *peak* RSS at `test_memory_acceptance.py`'s fixture scales
stays a Phase 4 task, since it measures the model's allocation behavior rather
than the mechanism.

## S5 — Watcher and scanner semantics — PASS

Every property the auto-indexing monitor relies on holds under
`@parcel/watcher`:

- Recursive coverage reaches a file created three levels below the watch root.
- A 50-write burst coalesces to **2 events**, comparable to watchfiles'
  debounced batch.
- A rename surfaces as `delete` on the old path plus `create` on the new.
- `git ls-files -z` streams as NUL-delimited chunks through
  `child_process.spawn`, parsed incrementally rather than buffered (249 paths).

### The one platform gap

`@kreuzberg/tree-sitter-language-pack` fails to load on Windows:

    Failed to load native binding for win32-x64.
    LoadLibrary failed: The specified module could not be found.

Both the bundled `ts-pack-core-node.win32-x64-msvc.node` and the separate
`@kreuzberg/tree-sitter-language-pack-win32-x64-msvc` optional dependency fail
identically. "The specified module could not be found" from `LoadLibrary`
signals a missing *transitive* DLL — almost certainly an MSVC runtime the
addon links against — rather than a missing addon, so the file is present and
its dependencies are not.

`gdshader` is therefore unavailable on Windows, and only `gdshader`: the other
17 languages pass there. It is a regression against the Python build, whose
PyPI `tree-sitter-language-pack` ships working Windows wheels.

**Decision (2026-08-17): accepted, not fixed.** Shader files are a marginal
part of the corpus, this is one format on one platform, and the two Godot
formats that matter more — `.gd` and `.tscn`/`.tres` — work everywhere from
dedicated packages. The routes to a fix (ship the VC++ redistributable, load
the grammar from the pack's WASM build through `web-tree-sitter` on Windows
only, or vendor the grammar's C sources as `@derekstride/tree-sitter-sql`
already does) all cost more than the capability is worth.

Two things follow from accepting it, and both are Phase 2's to carry:

- **A missing grammar must be a supported state, not an error.** The extractor
  builds its language table eagerly today; on Windows it will now have a hole
  in it. Files matching `.gdshader`/`.gdshaderinc` must be skipped the way an
  unsupported extension is, not raise — an index that fails on a Godot
  repository would be a far worse outcome than one that quietly omits its
  shaders.
- **It belongs in the release notes.** A Windows user indexing a Godot project
  gets their scripts and scenes but not their shaders, and that differs from
  what the Python build does today, so §12's cutover material has to say so
  rather than let them find out by searching for a shader and getting nothing.

S2 keeps the gap in its `KNOWN_GAPS` table and reports it as a **skip**, so the
spike's verdict stays visibly incomplete on Windows rather than falsely green.
The entry re-checks the failure each run and fails loudly if it stops
reproducing — if upstream ever fixes the binding, the capability comes back for
free and the entry can be deleted.

### Two platform differences Phase 6 must encode

**1. `ignore` entries are names and paths, not globs — and the glob behavior
differs by platform.** `"node_modules"` suppresses correctly everywhere. The
glob form `"**/node_modules/**"` suppresses nothing on macOS and everything on
Linux. Since `pathspec` patterns in `scanner.py` *are* globs, Phase 6 must
translate them into the plain directory names this API wants rather than pass
them through. A mistranslated pattern fails silently, putting the churn it was
meant to suppress back into the event stream with no error to notice — and it
would fail differently on each platform.

**2. Linux misses files written into a just-created directory.** macOS watches
a tree through FSEvents, so `mkdir -p a/b/c && write a/b/c/deep.py` is fully
observed. Linux inotify has no recursive mode: `@parcel/watcher` adds a watch
per directory as it learns of them, and a file created inside the window
between the mkdir and the watch registration is missed outright. The Linux run
saw the directory events and no event for the file.

This is **not a regression against the Python build** — `watchfiles` carries the
identical inotify constraint — and it is a latency issue rather than a
correctness one: `application.py::_project_is_stale` rescans the tree and
compares the whole path set on the next query, so a dropped event delays
proactive auto-indexing but cannot leave a file permanently unindexed. Phase 6
should not try to close the race in the watcher; it should preserve the lazy
freshness check as the actual correctness mechanism, which is what it already
is.

The spike records this per-platform rather than asserting it, since the answer
legitimately differs. The hard assertion it keeps is the property the monitor
genuinely depends on: a write into an already-watched nested directory is
always seen.

## Open items entering Phase 1

1. **D2 — Apple Metal** is still open; S3 did not measure CoreML.
2. **Correct §4 of the plan**: `apache-arrow` pins to 18.1.0, the tokenizer is
   `@huggingface/tokenizers`, and the grammar row needs the language pack.
3. **Correct §2.2/§7 of the plan**: the extractor registers **18** languages,
   not 19.
4. **Handle the accepted Windows `gdshader` gap in Phase 2** (see above):
   skip the files rather than fail, and note the difference in the release
   material. The gap itself is a decided non-goal, not an open question.
5. **Pin the language pack's grammar revisions** before Phase 2 relies on it.
   The pack resolves grammars from a remote manifest at its own version, so a
   pack upgrade can move a grammar underneath the extractor — the same hazard
   the PyPI pack carries, and worth the same treatment.
