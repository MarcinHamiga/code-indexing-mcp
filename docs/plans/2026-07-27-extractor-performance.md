# Extractor Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove three superlinear costs from `TreeSitterExtractor` so extraction scales with file
size instead of with the square of a file's definition count, and stop recompiling the Tree-sitter
query for every file.

**Architecture:** Four passes over one file, each precomputing something the current code rebuilds
per definition: a cached compiled `Query` per language, a definition index built once per file, and
a newline-position index built once per file. **No output changes.** Task 1 installs a committed
fingerprint snapshot over a fixture corpus first, so every later task is proven output-identical
rather than assumed to be.

**Tech Stack:** `tree_sitter` (`Language`, `Query`, `QueryCursor`), `bisect`, `threading.Lock`.

## Global Constraints

See [the plan index](2026-07-27-review-followups-index.md#global-constraints). Additionally:

- **Chunk output must not change — at all.** Chunk identity is
  `sha256(file_id, kind, qualified_symbol, start_byte, end_byte, part_index)`
  (`indexing.py:387-396`). Any drift in offsets or kinds silently invalidates every stored chunk id
  and corrupts incremental indexing. The Task 1 snapshot is the gate; do not weaken it to make a
  later task pass.
- **The extractor is shared across threads.** `Indexer` holds one `TreeSitterExtractor`
  (`application.py:102-105`) and the daemon serves each client on its own thread
  (`daemon.py:222-227`). Any new cache must be safe under that, following the double-checked
  `threading.Lock` pattern already used for the lazy model load at `embedding.py:196-199`.
- **No behaviour change for unsupported input.** Files that currently raise or produce
  `has_errors=True` must keep doing exactly that.

---

## Problem

Three independent superlinear costs, all in `extractor.py`, all measured in this repository.

### 1. Quadratic in definitions per file

`_has_definition_ancestor` (`extractor.py:132-140`) rebuilds a set of **all** definition node ids,
and `_symbol_context` (`extractor.py:142-164`) rebuilds a dict of **all** definitions — on every
call, once per definition. `_content_range` (`extractor.py:172-178`) scans all definitions for every
container. Profiled on a 2,000-definition file, the two dict/set rebuilds are **74% of extraction
time**:

```
ncalls  tottime  function
  2000    0.169  extractor.py:142(_symbol_context)
  2000    0.152  extractor.py:132(_has_definition_ancestor)
     1    0.036  extractor.py:65(extract)
  2000    0.028  {method 'count' of 'bytes' objects}
     1    0.008  {method 'parse' of 'tree_sitter.Parser' objects}   <- the actual parsing
```

Scaling is cleanly quadratic — **~4× the time for 2× the definitions**:

| definitions | bytes | extract |
|---|---|---|
| 250 | 9,779 | 9.3 ms |
| 500 | 19,779 | 26.7 ms (2.86×) |
| 1,000 | 39,779 | 104.9 ms (3.93×) |
| 2,000 | 81,779 | 409.2 ms (3.90×) |

At the scanner's own 1 MiB ceiling this is severe: **a 699 KB generated file with 16,384 definitions
takes 31.3 seconds to extract**, against **8 ms** of Tree-sitter parsing. Protobuf stubs, ORM
models, and generated API clients have exactly this shape.

### 2. The compiled query is rebuilt for every file

`extractor.py:101-102` re-reads the `.scm` from `importlib.resources` and recompiles
`Query(language, query_text)` on **every** `extract()` call:

```python
query_text = files("code_indexing_mcp.queries").joinpath(f"{language_name}.scm").read_text()
matches = QueryCursor(Query(language, query_text)).matches(root)
```

Measured over 35 repo files, caching the compiled query alone:

```
baseline:            72.9 ms/pass
cached Query:        40.8 ms/pass    -> 44% faster
```

The `Language` objects are already built once in `__init__` (`extractor.py:63`); the queries were
missed.

### 3. `start_line` rescans the file per chunk

`extractor.py:196`:

```python
start_line = source[:start].count(b"\n") + 1
```

That is O(file size) per chunk, so O(chunks × file size) per file — the third quadratic, and part of
the 31.3 s above. `_chunks_for_range` already precomputes a cumulative `line_offsets` array for
exactly this reason at `extractor.py:216-219`; the file-level equivalent is missing.

### Honest scoping

The README's own benchmark reports that embedding dominates a cold index — 141 s of 147 s at batch
size 1. So these fixes do **not** speed up a first index of ordinary source. They matter for
definition-dense generated files, where item 1 alone can cost 31 s for a single file, and for warm
re-indexes where embedding is skipped and extraction is what remains. Item 2's 44% applies to every
run.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `tests/fixtures/extractor_corpus/` | **New.** Six committed source files covering the shapes the extractor treats specially. Stable input, so the snapshot never needs regenerating for unrelated code changes. |
| `tests/fixtures/extractor_snapshot.json` | **New.** Committed fingerprint of the extractor's output over that corpus. |
| `tests/test_extractor_equivalence.py` | **New.** Asserts output against the snapshot, plus the scaling guards. |
| `src/code_indexing_mcp/extractor.py` | Gains `_query()` with a lock-guarded cache, `_DefinitionIndex`, and `_LineIndex`. `_has_definition_ancestor`, `_symbol_context`, and `_content_range` take the index instead of the list. |

Task order matters: **Task 1 must land before Tasks 2–4.** It is the only thing standing between a
performance refactor and silently corrupted chunk ids.

---

### Task 1: Freeze the extractor's output behind a snapshot

**Files:**
- Create: `tests/fixtures/extractor_corpus/nested.py`
- Create: `tests/fixtures/extractor_corpus/oversized.py`
- Create: `tests/fixtures/extractor_corpus/Service.java`
- Create: `tests/fixtures/extractor_corpus/exports.ts`
- Create: `tests/fixtures/extractor_corpus/widget.tsx`
- Create: `tests/fixtures/extractor_corpus/legacy.js`
- Create: `tests/fixtures/extractor_snapshot.json` (generated in Step 3)
- Create: `tests/test_extractor_equivalence.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fingerprint(result: ExtractionResult) -> list[list[object]]` and
  `CORPUS_DIRECTORY` / `SNAPSHOT_PATH` constants in the new test module. Tasks 2–4 re-run this test
  unchanged.

- [ ] **Step 1: Create the corpus**

`tests/fixtures/extractor_corpus/nested.py` — nested scopes, decorators, methods inside functions,
module-level code between definitions:

```python
"""Nested definition shapes."""

IMPORTANT = 1


def module_level(value):
    def inner(other):
        return other + 1

    return inner(value)


class Outer:
    CONSTANT = 2

    def method(self, value):
        def closure():
            return value

        return closure

    class Inner:
        def deep_method(self):
            return self.CONSTANT


TRAILING = Outer()


def after_class():
    return TRAILING
```

`tests/fixtures/extractor_corpus/oversized.py` — drives the line-fragment path and the long-function
windowing path. The single long line must exceed the 4,096-char default:

```python
def long_body():
    total = 0
    total += 1  # repeated below to force line-window splitting
```

Then append 400 identical `    total += 1` lines and one oversized line, deterministically:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
target = Path("tests/fixtures/extractor_corpus/oversized.py")
lines = [
    "def long_body():",
    "    total = 0",
]
lines += ["    total += 1" for _ in range(400)]
lines.append("    minified = " + repr("x" * 9000))
lines.append("    return total, minified")
lines.append("")
lines.append("")
lines.append("def after_oversized():")
lines.append("    return long_body()")
lines.append("")
target.write_text("\n".join(lines))
print(f"wrote {target} ({target.stat().st_size} bytes)")
PY
```

`tests/fixtures/extractor_corpus/Service.java`:

```java
package com.example.service;

public interface Marker {
    String name();
}

public enum Status {
    ACTIVE,
    CLOSED
}

public record Point(int x, int y) {}

public class Service implements Marker {
    private static final int LIMIT = 10;

    public Service() {
        this.value = LIMIT;
    }

    private int value;

    @Override
    public String name() {
        return "service";
    }

    static class Helper {
        int scaled(int input) {
            return input * LIMIT;
        }
    }
}
```

`tests/fixtures/extractor_corpus/exports.ts`:

```typescript
export const VERSION = "1.0.0";

export interface Options {
  retries: number;
}

export function configure(options: Options): Options {
  function normalize(value: number): number {
    return Math.max(0, value);
  }
  return { retries: normalize(options.retries) };
}

export default class Client {
  constructor(private readonly options: Options) {}

  async send(payload: string): Promise<string> {
    return payload;
  }
}

const anonymous = () => VERSION;
export { anonymous };
```

`tests/fixtures/extractor_corpus/widget.tsx`:

```tsx
import type { ReactNode } from "react";

export interface WidgetProps {
  label: string;
  children?: ReactNode;
}

export function Widget({ label, children }: WidgetProps) {
  return (
    <div className="widget">
      <span>{label}</span>
      {children}
    </div>
  );
}

export default Widget;
```

`tests/fixtures/extractor_corpus/legacy.js`:

```javascript
const CONSTANT = 42;

function classic(value) {
  return value + CONSTANT;
}

const arrow = (value) => classic(value);

class Legacy {
  constructor(seed) {
    this.seed = seed;
  }

  compute() {
    return arrow(this.seed);
  }
}

module.exports = { classic, arrow, Legacy };
```

- [ ] **Step 2: Write the equivalence test**

Create `tests/test_extractor_equivalence.py`:

```python
"""Output-equivalence gate for extractor refactors.

Chunk identity is a digest of kind, qualified symbol, byte offsets, and part index
(indexing.py). A refactor that shifts any of those silently invalidates every stored
chunk id and breaks incremental indexing, so performance work is gated on a
committed fingerprint rather than on review.

Regenerate deliberately, never to make a failing test pass:
    .venv/bin/python -m tests.test_extractor_equivalence
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.models import ExtractionResult
from code_indexing_mcp.scanner import LANGUAGES

CORPUS_DIRECTORY = Path(__file__).parent / "fixtures" / "extractor_corpus"
SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "extractor_snapshot.json"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def fingerprint(result: ExtractionResult) -> list[list[object]]:
    """Everything about a chunk that a consumer or a chunk id depends on."""
    return [
        [
            chunk.kind,
            chunk.symbol,
            chunk.qualified_symbol,
            chunk.parent_symbol,
            chunk.start_byte,
            chunk.end_byte,
            chunk.start_line,
            chunk.end_line,
            chunk.part_index,
            _digest(chunk.content),
            _digest(chunk.embedding_text),
            _digest(chunk.search_text),
            _digest(chunk.embedding_prefix),
            _digest(chunk.search_suffix),
        ]
        for chunk in result.chunks
    ]


def corpus_fingerprints() -> dict[str, object]:
    extractor = TreeSitterExtractor()
    snapshot: dict[str, object] = {}
    for path in sorted(CORPUS_DIRECTORY.iterdir()):
        language = LANGUAGES[path.suffix.lower()]
        result = extractor.extract(Path(path.name), language, path.read_bytes())
        snapshot[path.name] = {
            "has_errors": result.has_errors,
            "chunks": fingerprint(result),
        }
    return snapshot


def test_corpus_is_present_and_covers_every_language() -> None:
    languages = {LANGUAGES[path.suffix.lower()] for path in CORPUS_DIRECTORY.iterdir()}

    assert languages == {"python", "java", "javascript", "typescript", "tsx"}


def test_extractor_output_matches_the_committed_snapshot() -> None:
    expected = json.loads(SNAPSHOT_PATH.read_text())

    actual = corpus_fingerprints()

    assert set(actual) == set(expected)
    for name in sorted(expected):
        assert actual[name] == expected[name], f"extractor output changed for {name}"


def _generated_source(definitions: int) -> bytes:
    return "\n".join(
        f"def f{index}(a, b):\n    return a + b + {index}\n" for index in range(definitions)
    ).encode()


@pytest.mark.parametrize("definitions", [500, 2000])
def test_extraction_stays_within_a_linear_time_budget(definitions: int) -> None:
    """Guard against a return to quadratic scaling in definition count.

    The bounds are deliberately loose — roughly 40x the post-fix measurement — so
    they survive a slow or loaded CI machine while still failing hard if the
    per-definition rebuilds come back. Before the fix, 2,000 definitions took 409 ms
    and 16,384 took 31.3 s.
    """
    extractor = TreeSitterExtractor()
    source = _generated_source(definitions)
    extractor.extract(Path("warm.py"), "python", source)  # warm the query cache

    started = time.perf_counter()
    result = extractor.extract(Path("generated.py"), "python", source)
    elapsed = time.perf_counter() - started

    assert len(result.chunks) == definitions
    assert elapsed < definitions / 1000, (
        f"{definitions} definitions took {elapsed:.3f}s; expected sublinear-ish scaling"
    )


def test_definition_dense_file_at_the_scan_ceiling_is_not_quadratic() -> None:
    """A generated file just under the scanner's 1 MiB cap must not take ~30 s."""
    extractor = TreeSitterExtractor()
    source = _generated_source(16_384)
    assert len(source) < 1_048_576

    started = time.perf_counter()
    result = extractor.extract(Path("huge.py"), "python", source)
    elapsed = time.perf_counter() - started

    assert len(result.chunks) == 16_384
    assert elapsed < 5.0, f"extraction took {elapsed:.1f}s; quadratic behaviour is back"


if __name__ == "__main__":
    SNAPSHOT_PATH.write_text(json.dumps(corpus_fingerprints(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {SNAPSHOT_PATH}")
```

- [ ] **Step 3: Generate the snapshot from the current, unmodified extractor**

This must run **before** any change to `extractor.py`, so the snapshot records today's behaviour:

```bash
git diff --stat src/code_indexing_mcp/extractor.py   # must be empty
.venv/bin/python -m tests.test_extractor_equivalence
```

Expected: `wrote tests/fixtures/extractor_snapshot.json`. If `git diff` shows changes to
`extractor.py`, stash them first — a snapshot taken after a refactor proves nothing.

- [ ] **Step 4: Verify the snapshot test passes and the timing tests fail**

Run: `.venv/bin/python -m pytest tests/test_extractor_equivalence.py -v`

Expected: the snapshot and corpus tests PASS.
`test_definition_dense_file_at_the_scan_ceiling_is_not_quadratic` **FAILS** with roughly
`extraction took 31.3s; quadratic behaviour is back`, and
`test_extraction_stays_within_a_linear_time_budget[2000]` fails or is marginal (409 ms against a
2.0 s budget it may pass; the 16,384 case is the real gate). This failure is the point: it is the
red test that Tasks 2–4 turn green.

- [ ] **Step 5: Commit the gate**

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add tests/fixtures tests/test_extractor_equivalence.py
git commit -m "test: freeze extractor output and add scaling guards"
```

Commit with the timing test red, and note it in the commit body. Tasks 2–4 make it green.

---

### Task 2: Cache the compiled Tree-sitter query per language

**Files:**
- Modify: `src/code_indexing_mcp/extractor.py:52-63` (constructor), `:97-116` (`_definitions`)
- Test: `tests/test_extractor.py`

**Interfaces:**
- Consumes: the Task 1 snapshot test.
- Produces: `TreeSitterExtractor._query(language_name: str) -> Query`. `_definitions` becomes an
  instance method rather than a `@staticmethod`, taking `(language_name, root, source)` — the
  `Language` argument is no longer needed at the call site because `_query` resolves it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extractor.py`:

```python
def test_compiled_query_is_built_once_per_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """The .scm files are package data and Language objects are built in __init__.

    Re-reading and recompiling per file cost 44% of extraction time over 35 files.
    """
    import code_indexing_mcp.extractor as extractor_module

    compiled: list[str] = []
    original = extractor_module.Query

    def counting_query(language: object, text: str) -> object:
        compiled.append(text[:40])
        return original(language, text)

    monkeypatch.setattr(extractor_module, "Query", counting_query)
    extractor = extractor_module.TreeSitterExtractor()
    source = b"def one():\n    return 1\n"

    for _ in range(5):
        extractor.extract(Path("a.py"), "python", source)
    extractor.extract(Path("b.ts"), "typescript", b"export const x = 1;\n")

    assert len(compiled) == 2, f"compiled {len(compiled)} times, expected one per language"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extractor.py -k built_once -v`

Expected: FAIL — `compiled 6 times, expected one per language`.

- [ ] **Step 3: Add the cache**

In `src/code_indexing_mcp/extractor.py`, add `import threading` and the cache to `__init__`:

```python
        self._languages = _languages()
        self._queries: dict[str, Query] = {}
        # Indexer holds one extractor and the daemon serves each client on its own
        # thread, so the lazy compile must not build two queries concurrently. Same
        # double-checked shape as FastEmbedder's model load.
        self._queries_lock = threading.Lock()
```

Add the accessor:

```python
    def _query(self, language_name: str) -> Query:
        """Return the compiled query for *language_name*, compiling once per process.

        The .scm files are package data and never change at runtime, but the previous
        code re-read and recompiled one per extracted file, which measured at 44% of
        extraction time across a 35-file pass.
        """
        cached = self._queries.get(language_name)
        if cached is not None:
            return cached
        with self._queries_lock:
            cached = self._queries.get(language_name)
            if cached is not None:
                return cached
            text = files("code_indexing_mcp.queries").joinpath(f"{language_name}.scm").read_text()
            compiled = Query(self._languages[language_name], text)
            self._queries[language_name] = compiled
            return compiled
```

Change `_definitions` from a `@staticmethod` to an instance method and use the cache. Replace lines
97-116:

```python
    def _definitions(self, language_name: str, root: Node, source: bytes) -> list[_Definition]:
        matches = QueryCursor(self._query(language_name)).matches(root)
        found: dict[tuple[int, int, str], _Definition] = {}
        for _, captures in matches:
            name_nodes = captures.get("name", [])
            if not name_nodes:
                continue
            name_node = name_nodes[0]
            name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
            for capture, nodes in captures.items():
                if not capture.startswith("definition."):
                    continue
                kind = capture.removeprefix("definition.")
                node = nodes[0]
                found[(node.start_byte, node.end_byte, kind)] = _Definition(node, kind, name)
        return sorted(found.values(), key=lambda item: (item.node.start_byte, -item.node.end_byte))
```

And update the call in `extract` (line 69):

```python
        definitions = self._definitions(language, tree.root_node, normalized_source)
```

`language_impl` is still needed for the `Parser`, so keep the `self._languages[language]` lookup at
line 66.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_extractor.py tests/test_extractor_equivalence.py -v -k "built_once or snapshot"`

Expected: both PASS. The snapshot test passing is the proof that caching changed no output.

- [ ] **Step 5: Measure the win**

```bash
.venv/bin/python - <<'PY'
import time
from pathlib import Path
from code_indexing_mcp.extractor import TreeSitterExtractor
files = sorted(Path("src/code_indexing_mcp").glob("*.py")) + sorted(Path("tests").glob("*.py"))
data = [(p, p.read_bytes()) for p in files]
ex = TreeSitterExtractor()
for p, b in data: ex.extract(p, "python", b)
start = time.perf_counter()
for _ in range(3):
    for p, b in data: ex.extract(p, "python", b)
print(f"{len(data)} files: {(time.perf_counter()-start)/3*1000:.1f} ms/pass  (baseline was 72.9)")
PY
```

Expected: about **41 ms/pass**, down from 72.9.

- [ ] **Step 6: Commit**

```bash
.venv/bin/python -m pytest -q tests/test_extractor.py
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/extractor.py tests/test_extractor.py
git commit -m "perf: compile each Tree-sitter query once instead of per file"
```

---

### Task 3: Build the definition index once per file

**Files:**
- Modify: `src/code_indexing_mcp/extractor.py:45-50` (add `_DefinitionIndex`), `:65-95` (`extract`),
  `:132-140`, `:142-164`, `:166-180`
- Test: `tests/test_extractor_equivalence.py` (no new test — the Task 1 timing gate turns green)

**Interfaces:**
- Consumes: `_Definition` (existing), the Task 1 snapshot.
- Produces: `_DefinitionIndex` with `.definitions: list[_Definition]`, `.by_node_id: dict[int,
  _Definition]`, `.starts: list[int]`, and `_DefinitionIndex.build(definitions)`. The three helpers
  take `_DefinitionIndex` where they previously took `list[_Definition]`.

- [ ] **Step 1: Confirm the timing gate is red**

Run: `.venv/bin/python -m pytest tests/test_extractor_equivalence.py -k quadratic -v`

Expected: FAIL, `extraction took ~31s`. This is the test Task 3 fixes.

- [ ] **Step 2: Add the index type**

In `src/code_indexing_mcp/extractor.py`, add `from bisect import bisect_right` and, after the `_Definition`
dataclass:

```python
@dataclass(frozen=True)
class _DefinitionIndex:
    """Per-file lookups the definition walk needs, built once instead of per definition.

    ``_has_definition_ancestor``, ``_symbol_context``, and ``_content_range`` each
    rebuilt a whole-file dict or set on every call, making extraction quadratic in
    definition count: a 699 KB generated file with 16,384 definitions spent 31 s
    here against 8 ms of parsing.
    """

    definitions: list[_Definition]
    by_node_id: dict[int, _Definition]
    starts: list[int]

    @classmethod
    def build(cls, definitions: list[_Definition]) -> _DefinitionIndex:
        return cls(
            definitions=definitions,
            by_node_id={definition.node.id: definition for definition in definitions},
            # _definitions returns rows sorted by (start_byte, -end_byte), so this
            # is ascending and safe to bisect.
            starts=[definition.node.start_byte for definition in definitions],
        )
```

- [ ] **Step 3: Build it once in `extract` and thread it through**

Replace the body of `extract` from the `definitions =` line through the loop:

```python
        definitions = self._definitions(language, tree.root_node, normalized_source)
        index = _DefinitionIndex.build(definitions)
        chunks: list[ExtractedChunk] = []
        covered: list[tuple[int, int]] = []

        for definition in definitions:
            outer = self._outer_node(definition.node)
            if not self._has_definition_ancestor(definition.node, index):
                covered.append((outer.start_byte, outer.end_byte))
            kind, parent, qualified = self._symbol_context(definition, index)
            start, end = self._content_range(outer, definition.node, kind, index)
```

- [ ] **Step 4: Rewrite the three helpers**

```python
    @staticmethod
    def _has_definition_ancestor(node: Node, index: _DefinitionIndex) -> bool:
        parent = node.parent
        while parent is not None:
            if parent.id in index.by_node_id:
                return True
            parent = parent.parent
        return False

    @staticmethod
    def _symbol_context(
        definition: _Definition, index: _DefinitionIndex
    ) -> tuple[str, str | None, str]:
        chain: list[_Definition] = []
        parent = definition.node.parent
        while parent is not None:
            candidate = index.by_node_id.get(parent.id)
            if candidate is not None:
                chain.append(candidate)
            parent = parent.parent
        chain.reverse()
        if any(item.kind in _CALLABLE_KINDS for item in chain):
            scope = chain
        else:
            scope = [item for item in chain if item.kind in _CONTAINER_KINDS]
        parent_name = ".".join(item.name for item in scope) or None
        qualified = f"{parent_name}.{definition.name}" if parent_name else definition.name
        kind = definition.kind
        if scope and scope[-1].kind in _CONTAINER_KINDS and kind == "function":
            kind = "method"
        return kind, parent_name, qualified

    @staticmethod
    def _content_range(
        outer: Node, node: Node, kind: str, index: _DefinitionIndex
    ) -> tuple[int, int]:
        if kind not in _CONTAINER_KINDS:
            return outer.start_byte, outer.end_byte
        # The old code scanned every definition to take the minimum qualifying start.
        # Definitions are start-ascending, so the first qualifying one after
        # outer.start_byte *is* that minimum. A definition starting at or after
        # outer.end_byte cannot end inside outer, which bounds the scan.
        position = bisect_right(index.starts, outer.start_byte)
        while position < len(index.definitions):
            candidate = index.definitions[position].node
            if candidate.start_byte >= outer.end_byte:
                break
            if candidate.id != node.id and candidate.end_byte <= outer.end_byte:
                return outer.start_byte, candidate.start_byte
            position += 1
        return outer.start_byte, outer.end_byte
```

The `candidate.id != node.id` check replaces the original `item.node != node`. It must compare ids,
not spans: `_definitions` keys by `(start_byte, end_byte, kind)`, so one node legitimately appears
under two kinds and both entries must be excluded.

- [ ] **Step 5: Run the equivalence and timing tests**

Run: `.venv/bin/python -m pytest tests/test_extractor_equivalence.py -v`

Expected: **all pass now**, including
`test_definition_dense_file_at_the_scan_ceiling_is_not_quadratic`. The snapshot test passing proves
the rewrite is output-identical; the timing test passing proves the quadratic is gone.

If the snapshot test fails, `_content_range` is the likely culprit — it is the only helper whose
logic changed shape rather than just its lookup source. Compare against the original list
comprehension on the offending fixture before touching the snapshot.

- [ ] **Step 6: Confirm the scaling is now linear**

```bash
.venv/bin/python - <<'PY'
import time
from pathlib import Path
from code_indexing_mcp.extractor import TreeSitterExtractor
ex = TreeSitterExtractor()
previous = None
for count in (250, 500, 1000, 2000, 4000):
    src = "\n".join(f"def f{i}(a, b):\n    return a + b + {i}\n" for i in range(count)).encode()
    ex.extract(Path("warm.py"), "python", src)
    start = time.perf_counter(); ex.extract(Path("big.py"), "python", src)
    elapsed = (time.perf_counter() - start) * 1000
    ratio = f"  ({elapsed/previous:.2f}x for 2x input)" if previous else ""
    print(f"{count:5d} defs, {len(src):8,} bytes: {elapsed:8.1f} ms{ratio}")
    previous = elapsed
PY
```

Expected: ratios near **2.0×**, not the 3.9× measured before. The 2,000-definition case should be
tens of milliseconds rather than 409 ms.

- [ ] **Step 7: Commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/extractor.py
git commit -m "perf: build the definition index once per file instead of per definition"
```

---

### Task 4: Index newline positions once per file

**Files:**
- Modify: `src/code_indexing_mcp/extractor.py` (add `_LineIndex`, `:182-196`, `:305-340`)
- Test: `tests/test_extractor.py`

**Interfaces:**
- Consumes: the Task 1 snapshot.
- Produces: `_LineIndex` with `_LineIndex(source: bytes)` and `.line_at(byte_offset: int) -> int`
  (1-based). `_chunks_for_range` and `_module_chunks` take a `lines: _LineIndex` keyword argument.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extractor.py`:

```python
def test_line_index_matches_a_naive_newline_count() -> None:
    from code_indexing_mcp.extractor import _LineIndex

    source = b"alpha\nbeta\n\ngamma\r\ndelta"
    index = _LineIndex(source)

    for offset in range(len(source) + 1):
        assert index.line_at(offset) == source[:offset].count(b"\n") + 1, f"offset {offset}"


def test_line_index_handles_empty_and_newline_only_sources() -> None:
    from code_indexing_mcp.extractor import _LineIndex

    assert _LineIndex(b"").line_at(0) == 1
    assert _LineIndex(b"\n\n\n").line_at(3) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_extractor.py -k line_index -v`

Expected: FAIL — `ImportError: cannot import name '_LineIndex'`.

- [ ] **Step 3: Add the index**

Add `from bisect import bisect_left, bisect_right` (extending the Task 3 import) and, next to
`_DefinitionIndex`:

```python
class _LineIndex:
    """Byte offsets of every newline in one file, for O(log n) line lookups.

    ``source[:start].count(b"\\n")`` is O(file size) per chunk, so computing a start
    line per chunk was O(chunks x file size). ``bytes.find`` scans at C speed, so
    building this costs one pass and one append per line.
    """

    __slots__ = ("_newlines",)

    def __init__(self, source: bytes) -> None:
        newlines: list[int] = []
        position = source.find(b"\n")
        while position != -1:
            newlines.append(position)
            position = source.find(b"\n", position + 1)
        self._newlines = newlines

    def line_at(self, byte_offset: int) -> int:
        """Return the 1-based line number containing *byte_offset*."""
        return bisect_left(self._newlines, byte_offset) + 1
```

`bisect_left` is what matches the original: it counts newline positions strictly below
`byte_offset`, exactly as `source[:byte_offset].count(b"\n")` did.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_extractor.py -k line_index -v`

Expected: 2 passed.

- [ ] **Step 5: Use it in the two call sites**

In `extract`, build it once beside the definition index:

```python
        index = _DefinitionIndex.build(definitions)
        lines = _LineIndex(normalized_source)
```

Pass it into both `_chunks_for_range` calls in `extract` and `_module_chunks`, adding
`lines=lines` to each keyword-argument list. Change `_chunks_for_range`'s signature to accept it:

```python
    def _chunks_for_range(
        self,
        *,
        path: Path,
        language: str,
        kind: str,
        symbol: str | None,
        qualified: str | None,
        parent: str | None,
        source: bytes,
        start: int,
        end: int,
        lines: _LineIndex,
    ) -> list[ExtractedChunk]:
```

and replace line 196:

```python
        start_line = lines.line_at(start)
```

Change `_module_chunks`'s signature to `(self, path, language, source, covered, lines)` and forward
`lines=lines` to each of its two `_chunks_for_range` calls. Update its call in `extract`:

```python
        chunks.extend(self._module_chunks(path, language, normalized_source, covered, lines))
```

- [ ] **Step 6: Verify no output changed**

Run: `.venv/bin/python -m pytest tests/test_extractor_equivalence.py tests/test_extractor.py -q`

Expected: all pass. The snapshot test is the proof.

- [ ] **Step 7: Confirm the end-to-end gain on the pathological file**

```bash
.venv/bin/python - <<'PY'
import time
from pathlib import Path
from code_indexing_mcp.extractor import TreeSitterExtractor
ex = TreeSitterExtractor()
src = "\n".join(f"def f{i}(a, b):\n    return a + b + {i}\n" for i in range(16_384)).encode()
ex.extract(Path("warm.py"), "python", src)
start = time.perf_counter(); result = ex.extract(Path("huge.py"), "python", src)
print(f"{len(src):,} bytes, {len(result.chunks):,} chunks: {time.perf_counter()-start:.2f} s "
      f"(was 31.3 s)")
PY
```

Expected: well under a second, against 31.3 s on `main`.

- [ ] **Step 8: Full suite, lint, type-check, commit**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src tests
git add src/code_indexing_mcp/extractor.py tests/test_extractor.py
git commit -m "perf: index newline offsets once per file instead of rescanning per chunk"
```

- [ ] **Step 9: Record the measurements**

In `README.md`, in the indexing-performance section (around line 221-234, where the batch-size table
lives), add:

```markdown
Extraction is linear in file size and in definition count. Each Tree-sitter query is compiled once
per language per process, and the definition and newline indexes are built once per file. A
definition-dense generated file near the 1 MiB scan cap — 699 KB, 16,384 definitions — extracts in
well under a second; earlier releases took roughly 31 seconds on the same shape because those
indexes were rebuilt per definition.
```

---

## Self-Review

**Spec coverage.** Review items C1 (Task 3), C2 (Task 2), and C3 (Task 4) are each covered by a task
with its own measurement step. C5 (`get_chunk` scanning every project) and C6 (`find_symbol`
over-fetching with content) are **not** in this plan: C5 is documented as inherent to the chunk-id
format at `storage.py:184-188` and needs an id change plus a re-index, and C6 is a small constant
factor. Neither is a scaling defect.

**Type consistency.** `_DefinitionIndex.build` is defined in Task 3 Step 2 and used in Step 3; its
three attributes (`definitions`, `by_node_id`, `starts`) are consumed by the helpers rewritten in
Step 4. `_LineIndex.line_at` is defined in Task 4 Step 3 and used in Step 5. `_definitions` changes
from a `@staticmethod` taking `(language_name, language, root, source)` to an instance method taking
`(language_name, root, source)` in Task 2 Step 3, and the single call site is updated in the same
step.

**Ordering is not optional.** Task 1 must be committed with the snapshot generated from unmodified
`extractor.py`. Tasks 2, 3, and 4 are then independent of each other and may be reordered, but each
must leave the Task 1 snapshot test green.

**Concurrency note for reviewers.** The query cache is the only new shared mutable state. It is
guarded by a double-checked `threading.Lock`, mirroring `FastEmbedder._get_model`. Worth knowing:
indexing is already serialized machine-wide by `index-global.lock` (`indexing.py:121`), so
contention here is theoretical rather than real — the lock is for correctness under the daemon's
thread-per-connection model, not for throughput. `_DefinitionIndex` and `_LineIndex` are per-call
locals and need no synchronisation.
