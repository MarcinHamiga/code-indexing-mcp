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


def _call_shape_summary(call_shape: object) -> list[object] | None:
    if call_shape is None:
        return None
    return [
        call_shape.positional_count,
        call_shape.keywords,
        call_shape.has_positional_spread,
        call_shape.has_keyword_spread,
        call_shape.type_argument_count,
        call_shape.constructor,
    ]


def reference_fingerprint(result: ExtractionResult) -> list[list[object]]:
    """Everything a resolver or a rename depends on for one structural reference.

    This is the fingerprint the silent-miss defects (E1-E14) hide behind: a
    dropped, mis-shaped, or noise-carrying reference row changes nothing about
    `result.chunks`, so a snapshot gate that only covers chunks lets every one
    of those defects ship green. Covering `references`/`declarations` here is
    the whole point of Task 0.1.
    """
    return [
        [
            reference.kind,
            reference.target_name,
            reference.written_name,
            reference.module_path,
            reference.imported_name,
            reference.alias,
            reference.start_byte,
            reference.end_byte,
            reference.source_qualified_symbol,
            _call_shape_summary(reference.call_shape),
        ]
        for reference in result.references
    ]


def declaration_fingerprint(result: ExtractionResult) -> list[list[object]]:
    """Everything a rename/refactor depends on for one declaration shape."""
    return [
        [
            declaration.qualified_symbol,
            declaration.kind,
            [
                [parameter.name, parameter.kind, parameter.required, parameter.position]
                for parameter in declaration.parameters
            ],
        ]
        for declaration in result.declarations
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
            "references": reference_fingerprint(result),
            "declarations": declaration_fingerprint(result),
        }
    return snapshot


def test_corpus_is_present_and_covers_every_language() -> None:
    languages = {LANGUAGES[path.suffix.lower()] for path in CORPUS_DIRECTORY.iterdir()}

    assert languages == set(LANGUAGES.values())


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
