from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from filelock import Timeout

from code_indexing_mcp.probe_cache import (
    CACHE_SCHEMA_VERSION,
    MAX_RECORDS,
    PACKING_POLICY_VERSION,
    ProbeCache,
    ProbeKey,
    ProbeRecord,
    model_artifact_fingerprint,
)


@pytest.mark.parametrize("failure", [Timeout, OSError])
def test_a_cache_lock_failure_skips_the_optional_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    cache = ProbeCache(tmp_path / "probes.json")

    def acquire(*args: object, **kwargs: object) -> object:
        raise failure("cache lock unavailable")

    monkeypatch.setattr("code_indexing_mcp.probe_cache.FileLock.acquire", acquire)

    cache.store(_key(), batch_size=8, dimension=768)

    assert not cache.path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("dimension", 0),
        ("dimension", -1),
        ("characters_per_second", float("nan")),
        ("characters_per_second", float("inf")),
    ],
)
def test_invalid_probe_measurements_are_dropped(field: str, value: object) -> None:
    record = {
        "fingerprint": "fingerprint",
        "batch_size": 8,
        "dimension": 768,
        "recorded_at_ns": 1,
        "characters_per_second": 1.0,
        "load_ns": 1,
    }
    record[field] = value

    assert ProbeRecord.from_json(record) is None


def _key(**overrides: str) -> ProbeKey:
    base = ProbeKey(
        model_id="jinaai/jina-embeddings-v2-base-code",
        model_artifact="artifact-a",
        accelerator="cuda",
        provider="CUDAExecutionProvider",
        runtime_version="1.20.0",
        platform="darwin-arm64-25.5.0",
        device="cuda:0",
        driver_version="550.54",
    )
    return replace(base, **overrides)


def test_a_backend_that_changes_precision_is_measured_again(tmp_path: Path) -> None:
    """Half precision is a different speed and a different best batch size on
    the same device and runtime, so the float32 measurement must not answer for
    it -- while every float32 record written before precision was part of the
    key has to stay readable."""
    cache = ProbeCache(tmp_path / "probes.json")
    cache.store(_key(), batch_size=2, dimension=768, characters_per_second=46_000.0)

    assert cache.load(_key(precision="float16")) is None
    assert cache.load(_key(precision="float32")) is not None
    # The fingerprint of a float32 key is the one it had before the field.
    legacy = "\0".join(
        (
            "jinaai/jina-embeddings-v2-base-code",
            "artifact-a",
            "cuda",
            "CUDAExecutionProvider",
            "1.20.0",
            "darwin-arm64-25.5.0",
            "cuda:0",
            "550.54",
            str(PACKING_POLICY_VERSION),
        )
    )
    assert _key().fingerprint() == sha256(legacy.encode()).hexdigest()


def test_a_stored_probe_is_found_again(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    key = _key()

    assert cache.state(key) == "miss"
    cache.store(key, batch_size=16, dimension=768, detail="CUDAExecutionProvider")

    record = cache.load(key)
    assert record is not None
    assert record.batch_size == 16
    assert record.dimension == 768
    assert cache.state(key) == "hit"


def test_a_stored_calibration_survives_the_round_trip(tmp_path: Path) -> None:
    """The measurement is the whole point of storing anything beyond "it works":
    a rate that came back as zero would put the crossover at CPU forever."""
    cache = ProbeCache(tmp_path / "probes.json")
    key = _key()

    cache.store(
        key,
        batch_size=8,
        dimension=768,
        characters_per_second=12_345.5,
        load_ns=2_500_000_000,
        limited_by="memory",
    )

    record = cache.load(key)
    assert record is not None
    assert record.characters_per_second == 12_345.5
    assert record.load_ns == 2_500_000_000
    assert record.limited_by == "memory"


def test_a_record_written_before_calibration_is_not_read_as_uncalibrated(
    tmp_path: Path,
) -> None:
    """A version-1 record has no rate at all. Reading its absence as a measured
    zero would mean an accelerator that never crosses over."""
    path = tmp_path / "probes.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "fingerprint": _key().fingerprint(),
                        "batch_size": 8,
                        "dimension": 768,
                        "recorded_at_ns": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert ProbeCache(path).load(_key()) is None


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "model_artifact",
        "accelerator",
        "provider",
        "runtime_version",
        "platform",
        "device",
        "driver_version",
    ],
)
def test_every_key_component_invalidates_the_record(tmp_path: Path, field: str) -> None:
    """A cached "this backend works" must not outlive what it was measured on."""
    cache = ProbeCache(tmp_path / "probes.json")
    cache.store(_key(), batch_size=16, dimension=768)

    assert cache.load(_key(**{field: "changed"})) is None


def test_a_calibration_from_the_old_packing_policy_is_ignored(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    old_policy = replace(_key(), packing_policy_version=1)
    cache.store(old_policy, batch_size=16, dimension=768)

    assert cache.load(_key()) is None


def test_restoring_the_original_configuration_finds_the_record_again(tmp_path: Path) -> None:
    cache = ProbeCache(tmp_path / "probes.json")
    cache.store(_key(), batch_size=8, dimension=768)
    cache.store(_key(runtime_version="1.21.0"), batch_size=4, dimension=768)

    original = cache.load(_key())
    assert original is not None
    assert original.batch_size == 8


def test_storing_the_same_key_twice_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "probes.json"
    cache = ProbeCache(path)
    cache.store(_key(), batch_size=8, dimension=768)
    cache.store(_key(), batch_size=32, dimension=768)

    record = cache.load(_key())
    assert record is not None
    assert record.batch_size == 32
    assert len(json.loads(path.read_text())["records"]) == 1


def test_a_corrupt_cache_reads_as_empty_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "probes.json"
    path.write_text("{ not json")

    assert ProbeCache(path).load(_key()) is None


def test_a_cache_from_another_schema_version_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "probes.json"
    cache = ProbeCache(path)
    cache.store(_key(), batch_size=8, dimension=768)
    payload = json.loads(path.read_text())
    payload["schema_version"] = CACHE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))

    assert cache.load(_key()) is None


def test_a_partial_record_is_dropped_without_taking_the_others_with_it(tmp_path: Path) -> None:
    path = tmp_path / "probes.json"
    cache = ProbeCache(path)
    cache.store(_key(), batch_size=8, dimension=768)
    payload = json.loads(path.read_text())
    payload["records"].insert(0, {"fingerprint": "orphan"})
    path.write_text(json.dumps(payload))

    record = cache.load(_key())
    assert record is not None
    assert record.batch_size == 8


def test_the_cache_is_trimmed_to_its_bound(tmp_path: Path) -> None:
    path = tmp_path / "probes.json"
    cache = ProbeCache(path)
    for index in range(MAX_RECORDS + 5):
        cache.store(_key(device=f"cuda:{index}"), batch_size=index + 1, dimension=768)

    stored = json.loads(path.read_text())["records"]
    assert len(stored) == MAX_RECORDS
    # The newest survive; the earliest configurations are the ones dropped.
    assert cache.load(_key(device="cuda:0")) is None
    assert cache.load(_key(device=f"cuda:{MAX_RECORDS + 4}")) is not None


def test_a_missing_cache_file_is_simply_a_miss(tmp_path: Path) -> None:
    assert ProbeCache(tmp_path / "absent" / "probes.json").state(_key()) == "miss"


def test_an_unwritable_cache_directory_does_not_fail_the_run(tmp_path: Path) -> None:
    # Losing the cache costs a re-probe next time; it must never cost this run.
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")

    ProbeCache(blocked / "probes.json").store(_key(), batch_size=8, dimension=768)


def test_the_artifact_fingerprint_notices_a_changed_model_file(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "jina").mkdir(parents=True)
    artifact = models / "jina" / "model.onnx"
    artifact.write_bytes(b"x" * 128)

    before = model_artifact_fingerprint(models, "jina")
    artifact.write_bytes(b"x" * 256)
    after = model_artifact_fingerprint(models, "jina")

    assert before != after


def test_the_artifact_fingerprint_is_stable_across_calls(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.onnx").write_bytes(b"x" * 64)

    assert model_artifact_fingerprint(models, "jina") == model_artifact_fingerprint(models, "jina")


def test_the_artifact_fingerprint_survives_a_missing_cache_directory(tmp_path: Path) -> None:
    assert model_artifact_fingerprint(tmp_path / "absent", "jina")


def _fastembed_layout(cache: Path, model_id: str) -> Path:
    artifact = cache / f"models--{model_id.replace('/', '--')}" / "blobs" / "model.onnx"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x" * 128)
    return artifact


def test_the_fingerprint_ignores_the_rest_of_a_shared_model_cache(tmp_path: Path) -> None:
    """A probe is about one model; nothing else in the cache should void it.

    FastEmbed keeps a sibling ``.locks`` directory and a scratch ``models``
    directory that churn on their own, and a second model may be pulled at any
    time. Charging any of that to this model's key would re-probe every backend
    for reasons that have nothing to do with the artifact being probed.
    """
    cache = tmp_path / "models"
    _fastembed_layout(cache, "jinaai/jina-embeddings-v2-base-code")
    before = model_artifact_fingerprint(cache, "jinaai/jina-embeddings-v2-base-code")

    (cache / ".locks" / "models--jinaai--jina-embeddings-v2-base-code").mkdir(parents=True)
    (cache / ".locks" / "models--jinaai--jina-embeddings-v2-base-code" / "a.lock").write_text("1")
    _fastembed_layout(cache, "someone/another-model")
    (cache / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172")

    assert model_artifact_fingerprint(cache, "jinaai/jina-embeddings-v2-base-code") == before


def test_the_fingerprint_still_notices_the_models_own_artifact_changing(tmp_path: Path) -> None:
    cache = tmp_path / "models"
    artifact = _fastembed_layout(cache, "jinaai/jina-embeddings-v2-base-code")
    before = model_artifact_fingerprint(cache, "jinaai/jina-embeddings-v2-base-code")

    artifact.write_bytes(b"x" * 256)

    assert model_artifact_fingerprint(cache, "jinaai/jina-embeddings-v2-base-code") != before


def test_two_models_in_one_cache_do_not_share_a_fingerprint(tmp_path: Path) -> None:
    cache = tmp_path / "models"
    _fastembed_layout(cache, "jinaai/jina-embeddings-v2-base-code")
    _fastembed_layout(cache, "someone/another-model")

    assert model_artifact_fingerprint(
        cache, "jinaai/jina-embeddings-v2-base-code"
    ) != model_artifact_fingerprint(cache, "someone/another-model")


def test_an_unrecognised_layout_falls_back_to_the_whole_cache(tmp_path: Path) -> None:
    """Over-invalidating costs a re-probe; under-invalidating vouches for a lie."""
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "model.onnx").write_bytes(b"x" * 64)
    before = model_artifact_fingerprint(cache, "jina")

    (cache / "model.onnx").write_bytes(b"x" * 65)

    assert model_artifact_fingerprint(cache, "jina") != before


def test_a_record_without_a_context_stays_readable_but_an_old_reduction_does_not(
    tmp_path: Path,
) -> None:
    """EMB-7: records written before the context field existed must load."""
    from code_indexing_mcp.probe_cache import CACHE_SCHEMA_VERSION, ProbeCache, ProbeKey

    key = ProbeKey("m", "a", "cuda", "p", "1", "plat", "d")
    other = ProbeKey("m", "a", "cpu", "p", "1", "plat", "d")
    path = tmp_path / "probes.json"
    legacy = {"dimension": 4, "recorded_at_ns": 1, "characters_per_second": 5.0, "load_ns": 2}
    path.write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "records": [
                    {**legacy, "fingerprint": key.fingerprint(), "batch_size": 8},
                    {
                        **legacy,
                        "fingerprint": other.fingerprint(),
                        "batch_size": 1,
                        "limited_by": "memory",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cache = ProbeCache(path)

    assert cache.load(key, context="mem=1") is not None
    assert cache.load(other, context="mem=1") is None
    assert cache.load(other) is not None


def test_survived_round_trips_and_gates_contextless_records(tmp_path: Path) -> None:
    from code_indexing_mcp.probe_cache import CACHE_SCHEMA_VERSION, ProbeCache, ProbeKey

    key = ProbeKey("m", "a", "cuda", "p", "1", "plat", "d")
    legacy_key = ProbeKey("m", "a", "webgpu", "p", "1", "plat", "d")
    path = tmp_path / "probes.json"
    cache = ProbeCache(path)
    cache.store(
        key,
        batch_size=2,
        dimension=4,
        characters_per_second=100.0,
        load_ns=5,
        survived=4,
        context="mem=1",
    )

    loaded = cache.load(key, context="mem=1")
    assert loaded is not None
    assert loaded.survived == 4
    assert loaded.limited_by == ""

    # Legacy records lacking a context are rejected under context validation if they
    # carry a survived ceiling, just like limited_by reductions.
    path.write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "records": [
                    {
                        "fingerprint": legacy_key.fingerprint(),
                        "batch_size": 2,
                        "dimension": 4,
                        "recorded_at_ns": 1,
                        "characters_per_second": 5.0,
                        "load_ns": 2,
                        "survived": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert cache.load(legacy_key, context="mem=1") is None
    assert cache.load(legacy_key) is not None
