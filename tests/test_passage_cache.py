from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from code_indexing_mcp.embedding import EmbeddedSegment, PassageCandidate, SegmentPlan
from code_indexing_mcp.passage_cache import (
    PassageCacheNamespace,
    PassageEmbeddingCache,
    PassageReuseContext,
    artifact_digest,
    candidate_key,
    decode_segments,
    encode_segments,
    validate_segments,
)

DIMENSION = 4
PLAN = SegmentPlan(max_tokens=8, overlap_tokens=2, max_windows=4)


def _namespace(**overrides: object) -> PassageCacheNamespace:
    values: dict[str, object] = {
        "project_id": "project-a",
        "artifact_digest": "artifact-a",
        "tokenizer_digest": "tokenizer-a",
        "producer": "cpu-float32",
        "runtime_version": "python-3.12",
        "dimension": DIMENSION,
        "precision": "float32-le",
    }
    values.update(overrides)
    return PassageCacheNamespace(**values)


def _candidate(
    prefix: str = "kind: function", content: str = "héllo🙂 world!!"
) -> PassageCandidate:
    return PassageCandidate(prefix, content)


def _segments() -> list[EmbeddedSegment]:
    return [
        EmbeddedSegment(0, 8, 3, np.asarray([1, 2, 3, 4], dtype="<f4").tobytes()),
        EmbeddedSegment(6, 14, 3, np.asarray([4, 3, 2, 1], dtype="<f4").tobytes()),
    ]


def _larger_segments() -> list[EmbeddedSegment]:
    return [
        *_segments(),
        EmbeddedSegment(12, 14, 1, np.asarray([1, 1, 1, 1], dtype="<f4").tobytes()),
    ]


def test_candidate_key_is_stable_and_ignores_batch_scheduling() -> None:
    candidate = _candidate()

    first = candidate_key(_namespace(), candidate, PLAN)
    second = candidate_key(
        _namespace(),
        candidate,
        SegmentPlan(
            max_tokens=8, overlap_tokens=2, max_windows=4, max_items=99, max_token_product=1
        ),
    )

    assert first == second
    assert first != candidate_key(_namespace(project_id="project-b"), candidate, PLAN)
    assert first != candidate_key(_namespace(tokenizer_digest="tokenizer-b"), candidate, PLAN)
    assert first != candidate_key(_namespace(precision="float16"), candidate, PLAN)
    assert first != candidate_key(_namespace(), _candidate(content="changed"), PLAN)
    assert first != candidate_key(_namespace(), _candidate(prefix="kind: class"), PLAN)
    assert first != candidate_key(
        _namespace(), candidate, SegmentPlan(max_tokens=16, overlap_tokens=2, max_windows=4)
    )


def test_segment_encoding_round_trips_unicode_and_overlapping_windows() -> None:
    segments = _segments()
    candidate = _candidate()
    validate_segments(segments, content_length=len(candidate.content), plan=PLAN)

    encoded = encode_segments(segments, dimension=DIMENSION, max_windows=PLAN.max_windows)
    decoded = decode_segments(
        encoded.metadata,
        encoded.vectors,
        encoded.checksum,
        dimension=DIMENSION,
        max_windows=PLAN.max_windows,
    )

    assert decoded == segments


@pytest.mark.parametrize(
    "metadata, vectors, checksum",
    [
        ("{", b"", ""),
        (json.dumps([[0, 1, 1]]), b"short", "bad"),
    ],
)
def test_malformed_entries_are_rejected(metadata: str, vectors: bytes, checksum: str) -> None:
    with pytest.raises(ValueError):
        decode_segments(metadata, vectors, checksum, dimension=DIMENSION, max_windows=4)


def test_nonfinite_or_zero_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        encode_segments(
            [
                EmbeddedSegment(
                    0,
                    1,
                    1,
                    np.asarray([np.nan, 1, 2, 3], dtype="<f4").tobytes(),
                )
            ],
            dimension=DIMENSION,
        )

    with pytest.raises(ValueError, match="nonzero"):
        encode_segments([EmbeddedSegment(0, 1, 1, bytes(DIMENSION * 4))], dimension=DIMENSION)


def test_segment_validation_rejects_gaps_and_excessive_windows() -> None:
    with pytest.raises(ValueError, match="cover"):
        validate_segments(
            [EmbeddedSegment(0, 2, 1, b""), EmbeddedSegment(4, 6, 1, b"")],
            content_length=6,
            plan=PLAN,
        )

    with pytest.raises(ValueError, match="window"):
        validate_segments(
            [EmbeddedSegment(index, index + 1, 1, b"") for index in range(5)],
            content_length=5,
            plan=PLAN,
        )


def test_artifact_digest_detects_a_same_size_replacement(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"aaaa")
    first = artifact_digest(artifact)
    artifact.write_bytes(b"bbbb")

    assert first is not None
    assert artifact_digest(artifact) != first


def test_cache_persists_bulk_entries_and_returns_only_hits(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        cache.put_many({"a": _segments(), "b": _segments()})
        assert set(cache.get_many(["a", "missing", "b"])) == {"a", "b"}

    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        assert cache.get_many(["a"]) == {"a": _segments()}


def test_cache_evicts_old_entries_to_stay_within_the_logical_quota(tmp_path: Path) -> None:
    cache = PassageEmbeddingCache(
        tmp_path / "passage.sqlite3", dimension=DIMENSION, max_payload_bytes=80
    )
    with cache:
        cache.put_many({"a": _segments()})
        cache.put_many({"b": _segments()})
        assert cache.payload_bytes <= 160
        assert set(cache.get_many(["a", "b"])) == {"b"}


def test_cache_rejects_entries_larger_than_the_entry_limit(tmp_path: Path) -> None:
    with PassageEmbeddingCache(
        tmp_path / "passage.sqlite3", dimension=DIMENSION, max_entry_bytes=32
    ) as cache:
        cache.put_many({"too-large": _segments()})
        assert cache.get_many(["too-large"]) == {}


def test_a_too_large_replacement_preserves_the_previous_entry(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    with PassageEmbeddingCache(
        path, dimension=DIMENSION, max_payload_bytes=70, max_entry_bytes=100
    ) as cache:
        cache.put_many({"a": _segments()})
        cache.put_many({"a": _larger_segments()})

        assert cache.get_many(["a"]) == {"a": _segments()}


def test_corrupt_database_rows_are_treated_as_misses(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        cache.put_many({"a": _segments()})

    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE entries SET vectors = ? WHERE cache_key = ?", (b"bad", "a"))

    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        assert cache.get_many(["a"]) == {}


def test_replacing_an_entry_does_not_drift_payload_accounting(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        cache.put_many({"a": _segments()})
        first_size = cache.payload_bytes
        cache.put_many({"a": _segments()})

        assert cache.payload_bytes == first_size


def test_cache_disables_itself_after_a_database_error(tmp_path: Path) -> None:
    cache = PassageEmbeddingCache(tmp_path / "passage.sqlite3", dimension=DIMENSION)
    with cache:
        assert cache._connection is not None
        cache._connection.close()
        assert cache.get_many(["a"]) == {}
        assert cache.disabled is True


def test_contending_writer_disables_only_its_cache_connection(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    first = PassageEmbeddingCache(path, dimension=DIMENSION)
    second = PassageEmbeddingCache(path, dimension=DIMENSION, busy_timeout_ms=1)
    with first, second:
        assert first._connection is not None
        first._connection.execute("BEGIN IMMEDIATE")
        second.put_many({"a": _segments()})
        assert second.disabled is True
        first._connection.execute("ROLLBACK")


def test_cache_uses_rollback_journaling_and_respects_the_page_ceiling(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    with PassageEmbeddingCache(path, dimension=DIMENSION, max_database_bytes=128 * 1024) as cache:
        assert cache._connection is not None
        journal_mode = cache._connection.execute("PRAGMA journal_mode").fetchone()[0]
        page_size = cache._connection.execute("PRAGMA page_size").fetchone()[0]
        max_pages = cache._connection.execute("PRAGMA max_page_count").fetchone()[0]

        cache.put_many({str(index): _segments() for index in range(10)})

        page_count = cache._connection.execute("PRAGMA page_count").fetchone()[0]
        assert journal_mode == "delete"
        assert page_count <= max_pages
        assert page_count * page_size <= 128 * 1024


def test_reuse_context_bulk_resolves_hits_and_writes_misses(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    namespace = _namespace()
    candidates = [_candidate(), _candidate(content="changed text!!")]
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        cache.put_many({candidate_key(namespace, candidates[0], PLAN): _segments()})

    class TokenizerAvailable:
        tokenizer_available = True

    producer = TokenizerAvailable()
    with PassageReuseContext(path, namespace) as reuse:
        hits, misses = reuse.lookup(candidates, PLAN, producer=producer)
        assert hits == {0: _segments()}
        assert misses == [1]
        reuse.store(candidates, {1: _segments()}, PLAN, producer=producer)
        assert reuse.status == "active"
        assert reuse.lookup(candidates, PLAN, producer=producer)[0].keys() == {0, 1}
        assert reuse.reused_candidates == 3
        assert reuse.reused_segments == 6
        assert reuse.lookup_duration_ms >= 0
        assert reuse.write_duration_ms >= 0


def test_reuse_context_skips_hits_when_the_runtime_lacks_a_tokenizer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "passage.sqlite3"
    namespace = _namespace()
    candidate = _candidate()
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        cache.put_many({candidate_key(namespace, candidate, PLAN): _segments()})

    class NoTokenizer:
        tokenizer_available = False

    with PassageReuseContext(path, namespace) as reuse:
        assert reuse.lookup([candidate], PLAN, producer=NoTokenizer()) == ({}, [0])


def test_reuse_context_bypasses_reads_in_strict_mode(tmp_path: Path) -> None:
    with PassageReuseContext(tmp_path / "passage.sqlite3", _namespace(), strict=True) as reuse:
        assert reuse.status == "bypassed"
        assert reuse.lookup([_candidate()], PLAN) == ({}, [0])


def test_interleaved_writers_share_authoritative_payload_accounting(tmp_path: Path) -> None:
    path = tmp_path / "passage.sqlite3"
    size = encode_segments(_segments(), dimension=DIMENSION).payload_bytes
    with (
        PassageEmbeddingCache(path, dimension=DIMENSION, max_payload_bytes=size) as first,
        PassageEmbeddingCache(path, dimension=DIMENSION, max_payload_bytes=size) as second,
    ):
        first.put_many({"a": _segments()})
        second.put_many({"b": _segments()})
        first.put_many({"c": _segments()})
        with sqlite3.connect(path) as connection:
            actual = connection.execute("SELECT SUM(payload_bytes) FROM entries").fetchone()[0]
            recorded = connection.execute("SELECT payload_bytes FROM cache_meta").fetchone()[0]
        assert actual == recorded == size
        assert set(first.get_many(["a", "b", "c"])) == {"c"}


@pytest.mark.parametrize(
    "producer", [None, type("UnknownTokenizer", (), {"tokenizer_available": None})()]
)
def test_unknown_tokenizer_bypasses_cache_reads_and_writes(
    tmp_path: Path, producer: object
) -> None:
    path = tmp_path / "passage.sqlite3"
    namespace = _namespace()
    candidates = [_candidate(), _candidate(content="changed text!!")]
    key = candidate_key(namespace, candidates[0], PLAN)
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        cache.put_many({key: _segments()})
    with PassageReuseContext(path, namespace) as reuse:
        assert reuse.lookup(candidates, PLAN, producer=producer) == ({}, [0, 1])
        reuse.store(candidates, {1: _segments()}, PLAN, producer=producer)
    with PassageEmbeddingCache(path, dimension=DIMENSION) as cache:
        assert cache.get_many([candidate_key(namespace, candidates[1], PLAN)]) == {}
