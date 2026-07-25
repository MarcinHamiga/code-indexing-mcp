#!/usr/bin/env python3
"""Measure indexing memory against a corpus shaped to drive the peak.

Peak resident memory tracks the largest *single* file, not repository size, so
the generated corpus deliberately contains a file just under the 1 MiB scanner
cap, a single-line minified file near that cap, and a long blank run next to an
oversized line - the shapes that drive the extractor's fragment path and the
widest embedding batch. Ordinary modules scale with ``--corpus-scale`` to give a
baseline and to expose parent-side growth.

``evaluate_result`` is pure and importable, so the acceptance criteria are
testable without a model. Only ``run_benchmark`` starts a real index, and it
never downloads a model unless ``--model-cache`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

MIB = 1024**2

# Acceptance thresholds. They mirror the release criteria in
# docs/plans/2026-07-24-indexing-memory-hardening-completion.md Task 8.
OVERSHOOT_ALLOWANCE_BYTES = 256 * MIB
MAXIMUM_BREACH_SECONDS = 1.0
MAXIMUM_WORKER_EXIT_SECONDS = 2.0
MAXIMUM_PARENT_GROWTH_BYTES = 128 * MIB
MINIMUM_SAMPLES = 2

SCANNER_MAX_FILE_BYTES = 1_048_576
EXTRACTOR_MAX_CHARS = 4_096
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1
ORDINARY_FILES_PER_SCALE = 20


@dataclass(frozen=True)
class Sample:
    """One process-tree observation. Parent and worker stay separate.

    Combined RSS alone hides which side grew, which is what made an earlier
    memory-accounting bug invisible.
    """

    at: float
    parent_bytes: int
    worker_bytes: int
    available_bytes: int

    @property
    def combined_bytes(self) -> int:
        return self.parent_bytes + self.worker_bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "at": round(self.at, 4),
            "parent_bytes": self.parent_bytes,
            "worker_bytes": self.worker_bytes,
            "available_bytes": self.available_bytes,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Sample:
        return cls(
            at=float(payload["at"]),
            parent_bytes=int(payload["parent_bytes"]),
            worker_bytes=int(payload["worker_bytes"]),
            available_bytes=int(payload["available_bytes"]),
        )


@dataclass(frozen=True)
class BenchmarkRun:
    corpus_scale: int
    corpus_bytes: int
    batch_size: int
    configured_ceiling_bytes: int
    effective_ceiling_bytes: int
    samples: tuple[Sample, ...]
    exit_code: int
    duration_seconds: float
    worker_exit_seconds: float
    report: dict[str, Any] | None = None
    stderr_tail: str = ""

    @property
    def peak_parent_bytes(self) -> int:
        return max((sample.parent_bytes for sample in self.samples), default=0)

    @property
    def peak_worker_bytes(self) -> int:
        return max((sample.worker_bytes for sample in self.samples), default=0)

    @property
    def peak_combined_bytes(self) -> int:
        return max((sample.combined_bytes for sample in self.samples), default=0)

    @property
    def minimum_available_bytes(self) -> int:
        return min((sample.available_bytes for sample in self.samples), default=0)

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus_scale": self.corpus_scale,
            "corpus_bytes": self.corpus_bytes,
            "batch_size": self.batch_size,
            "configured_ceiling_bytes": self.configured_ceiling_bytes,
            "effective_ceiling_bytes": self.effective_ceiling_bytes,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 3),
            "worker_exit_seconds": round(self.worker_exit_seconds, 3),
            "peak_parent_bytes": self.peak_parent_bytes,
            "peak_worker_bytes": self.peak_worker_bytes,
            "peak_combined_bytes": self.peak_combined_bytes,
            "minimum_available_bytes": self.minimum_available_bytes,
            "report": self.report,
            "stderr_tail": self.stderr_tail,
            "samples": [sample.to_json() for sample in self.samples],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> BenchmarkRun:
        return cls(
            corpus_scale=int(payload["corpus_scale"]),
            corpus_bytes=int(payload["corpus_bytes"]),
            batch_size=int(payload["batch_size"]),
            configured_ceiling_bytes=int(payload["configured_ceiling_bytes"]),
            effective_ceiling_bytes=int(payload["effective_ceiling_bytes"]),
            samples=tuple(Sample.from_json(item) for item in payload["samples"]),
            exit_code=int(payload["exit_code"]),
            duration_seconds=float(payload["duration_seconds"]),
            worker_exit_seconds=float(payload["worker_exit_seconds"]),
            report=payload.get("report"),
            stderr_tail=str(payload.get("stderr_tail", "")),
        )


@dataclass(frozen=True)
class Verdict:
    passed: bool
    valid: bool
    reasons: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"passed": self.passed, "valid": self.valid, "reasons": list(self.reasons)}


def _breach_seconds(run: BenchmarkRun) -> float:
    """Return the longest contiguous stretch spent above the allowed ceiling."""
    allowed = run.effective_ceiling_bytes + OVERSHOOT_ALLOWANCE_BYTES
    longest = 0.0
    start: float | None = None
    for sample in run.samples:
        if sample.combined_bytes > allowed:
            if start is None:
                start = sample.at
            longest = max(longest, sample.at - start)
        else:
            start = None
    return longest


def evaluate_result(run: BenchmarkRun, *, baseline: BenchmarkRun | None = None) -> Verdict:
    """Judge a run against the acceptance criteria. Pure: no processes, no clock.

    *baseline* is a smaller-scale run of the same configuration. Parent growth is
    only meaningful as a difference between scales, because a single run cannot
    distinguish a fixed footprint from one that tracks corpus size.
    """
    if len(run.samples) < MINIMUM_SAMPLES:
        return Verdict(
            passed=False,
            valid=False,
            reasons=(
                f"invalid benchmark: {len(run.samples)} RSS samples, "
                f"at least {MINIMUM_SAMPLES} required",
            ),
        )

    reasons: list[str] = []
    if run.exit_code != 0:
        reasons.append(f"indexing exited with code {run.exit_code}")

    allowed = run.effective_ceiling_bytes + OVERSHOOT_ALLOWANCE_BYTES
    breach = _breach_seconds(run)
    if breach > MAXIMUM_BREACH_SECONDS:
        reasons.append(
            f"combined RSS stayed above {allowed // MIB} MiB for {breach:.1f}s "
            f"(limit {MAXIMUM_BREACH_SECONDS:.1f}s); peak "
            f"{run.peak_combined_bytes // MIB} MiB"
        )

    if run.worker_exit_seconds > MAXIMUM_WORKER_EXIT_SECONDS:
        reasons.append(
            f"embedding worker was still alive {run.worker_exit_seconds:.1f}s after "
            f"completion (limit {MAXIMUM_WORKER_EXIT_SECONDS:.1f}s)"
        )

    if baseline is not None:
        growth = run.peak_parent_bytes - baseline.peak_parent_bytes
        if growth > MAXIMUM_PARENT_GROWTH_BYTES:
            reasons.append(
                f"parent RSS grew {growth // MIB} MiB from corpus scale "
                f"{baseline.corpus_scale} to {run.corpus_scale} "
                f"(limit {MAXIMUM_PARENT_GROWTH_BYTES // MIB} MiB)"
            )

    return Verdict(passed=not reasons, valid=True, reasons=tuple(reasons))


def _ordinary_module(index: int) -> str:
    lines = [
        f'"""Deterministic benchmark module {index}."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    for item in range(12):
        lines += [
            "",
            f"def operation_{index}_{item}(value: int) -> int:",
            f'    """Scale *value* by {item + 1} and fold in a bounded loop."""',
            f"    total = value * {item + 1}",
            "    for step in range(4):",
            f"        total += step * {item + 1}",
            "    return total",
        ]
    lines += [
        "",
        "",
        f"class Service{index}:",
        '    """A small class so container chunking is exercised too."""',
        "",
        f"    name = 'service-{index}'",
        "",
        "    def handle(self, value: int) -> int:",
        f"        return operation_{index}_0(value)",
        "",
    ]
    return "\n".join(lines)


def _sized_module(target_bytes: int, *, prefix: str) -> str:
    """Return valid Python of just under *target_bytes*, built deterministically."""
    parts = [f'"""{prefix}: a file sized to sit just under the scanner cap."""', ""]
    size = len("\n".join(parts).encode())
    index = 0
    while True:
        block = _ordinary_module(index).split("\n", 1)[1]
        encoded = len(block.encode()) + 1
        if size + encoded > target_bytes:
            break
        parts.append(block)
        size += encoded
        index += 1
    return "\n".join(parts)


def _minified_module(target_bytes: int) -> str:
    """One very long line: the shape that drives the extractor's fragment path."""
    values: list[str] = []
    size = len(b"DATA = []\n")
    index = 0
    while size < target_bytes:
        token = str(index % 977)
        values.append(token)
        size += len(token) + 2
        index += 1
    return "DATA = [" + ", ".join(values) + "]\n"


def _blank_run_module() -> str:
    """A long blank run adjacent to an oversized line.

    The blank run makes the extractor's line window stop on an oversized line
    rather than fill up, and the fragments of that line are mostly whitespace -
    the branch that skips empty fragments.
    """
    blanks = "\n" * 400
    oversized = "PAYLOAD = '" + ("x" * (EXTRACTOR_MAX_CHARS * 3)) + "'"
    return (
        '"""A blank run next to an oversized line."""\n'
        f"{blanks}"
        f"{oversized}\n"
        f"{blanks}"
        "def tail(value: int) -> int:\n"
        "    return value + 1\n"
    )


_CAP_MARGIN_BYTES = 4 * 1024

# The shapes that drive the peak, as opposed to the ordinary modules that only
# drive corpus size. Selectable so a run can attribute a breach to one of them.
PEAK_SHAPES: dict[str, Any] = {
    "near_cap": lambda: _sized_module(
        SCANNER_MAX_FILE_BYTES - _CAP_MARGIN_BYTES, prefix="near_cap"
    ),
    "minified": lambda: _minified_module(SCANNER_MAX_FILE_BYTES - _CAP_MARGIN_BYTES),
    "blank_run": _blank_run_module,
}


def write_corpus(root: Path, *, scale: int, shapes: Sequence[str] | None = None) -> int:
    """Write the benchmark corpus under *root* and return its total byte size.

    *shapes* defaults to every peak-driving shape. Narrow it to attribute a
    ceiling breach to one shape, or to gate the shapes that pass separately from
    one that is known to fail.
    """
    selected = list(PEAK_SHAPES) if shapes is None else list(shapes)
    unknown = set(selected) - set(PEAK_SHAPES)
    if unknown:
        raise ValueError(f"Unknown corpus shapes: {sorted(unknown)}")

    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'benchmark'\n", encoding="utf-8")
    package = root / "src" / "benchmark"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text('"""Benchmark package."""\n', encoding="utf-8")

    for index in range(ORDINARY_FILES_PER_SCALE * scale):
        (package / f"module_{index:04d}.py").write_text(_ordinary_module(index), encoding="utf-8")

    # Written once at every scale: the peak tracks the largest single file, so
    # repeating these would not raise it.
    for shape in selected:
        (package / f"{shape}.py").write_text(PEAK_SHAPES[shape](), encoding="utf-8")

    return sum(path.stat().st_size for path in root.rglob("*.py"))


def _sample(tracker: psutil.Process, descendants: set[int], started: float) -> Sample | None:
    try:
        parent_bytes = int(tracker.memory_info().rss)
        worker_bytes = 0
        for child in tracker.children(recursive=True):
            try:
                worker_bytes += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            descendants.add(child.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return Sample(
        at=time.monotonic() - started,
        parent_bytes=parent_bytes,
        worker_bytes=worker_bytes,
        available_bytes=int(psutil.virtual_memory().available),
    )


def _await_descendant_exit(pids: set[int], *, timeout: float) -> float:
    """Return how long after completion the last worker took to disappear.

    PID reuse could in principle keep this above zero; the window is far shorter
    than the two-second criterion, so it is not worth defending against here.
    """
    if not pids:
        return 0.0
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if not any(psutil.pid_exists(pid) for pid in pids):
            return time.monotonic() - started
        time.sleep(0.05)
    return timeout


def _environment(
    *,
    data_directory: Path,
    cache_directory: Path,
    batch_size: int,
    memory_mb: int | None,
    offline: bool,
) -> dict[str, str]:
    environment = dict(os.environ)
    environment["INCODE_DATA_DIR"] = str(data_directory)
    environment["INCODE_CACHE_DIR"] = str(cache_directory)
    environment["INCODE_EMBED_BATCH_SIZE"] = str(batch_size)
    environment["INCODE_BROKER"] = "off"
    if memory_mb is not None:
        environment["INCODE_INDEX_MEMORY_MB"] = str(memory_mb)
    if offline:
        environment["INCODE_OFFLINE"] = "1"
    else:
        environment.pop("INCODE_OFFLINE", None)
    return environment


def run_benchmark(
    *,
    root: Path,
    data_directory: Path,
    cache_directory: Path,
    corpus_scale: int,
    corpus_bytes: int,
    batch_size: int,
    memory_mb: int | None,
    offline: bool,
    sample_interval: float,
    command_prefix: Sequence[str],
) -> BenchmarkRun:
    environment = _environment(
        data_directory=data_directory,
        cache_directory=cache_directory,
        batch_size=batch_size,
        memory_mb=memory_mb,
        offline=offline,
    )
    subprocess.run(
        [*command_prefix, "init", str(root)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    configured = memory_mb * MIB if memory_mb is not None else 0
    with tempfile.TemporaryDirectory() as streams:
        out_path = Path(streams) / "stdout"
        err_path = Path(streams) / "stderr"
        started = time.monotonic()
        with out_path.open("w") as out, err_path.open("w") as err:
            process = subprocess.Popen(
                [*command_prefix, "index", str(root)],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
            )
            tracker = psutil.Process(process.pid)
            samples: list[Sample] = []
            descendants: set[int] = set()
            while process.poll() is None:
                sample = _sample(tracker, descendants, started)
                if sample is not None:
                    samples.append(sample)
                time.sleep(sample_interval)
        duration = time.monotonic() - started
        worker_exit = _await_descendant_exit(descendants, timeout=MAXIMUM_WORKER_EXIT_SECONDS + 1.0)
        stdout = out_path.read_text(errors="replace")
        stderr = err_path.read_text(errors="replace")

    report: dict[str, Any] | None = None
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            report = parsed
    except json.JSONDecodeError:
        report = None

    # The child computes the true ceiling from memory available at worker start,
    # so prefer what it reported over the value asked for.
    budget = report.get("memory_budget_bytes") if report else None
    effective = int(budget) if budget else configured
    return BenchmarkRun(
        corpus_scale=corpus_scale,
        corpus_bytes=corpus_bytes,
        batch_size=batch_size,
        configured_ceiling_bytes=configured,
        effective_ceiling_bytes=effective,
        samples=tuple(samples),
        exit_code=process.returncode,
        duration_seconds=duration,
        worker_exit_seconds=worker_exit,
        report=report,
        stderr_tail=stderr[-4000:],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_index_memory",
        description="Measure indexing memory against a peak-driving corpus",
    )
    parser.add_argument("--corpus-scale", type=int, default=1)
    parser.add_argument(
        "--shapes",
        default=",".join(PEAK_SHAPES),
        help=f"Comma-separated peak-driving shapes to include: {', '.join(PEAK_SHAPES)}",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=None)
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=None,
        help="Reuse (and populate) this model cache. Without it the run is offline "
        "and no model is downloaded.",
    )
    parser.add_argument("--offline", action="store_true", help="Stay offline even with a cache")
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="A smaller-scale run's JSON, used to judge parent-side growth",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Outside the repository on purpose: the corpus must not be picked up by the
    # project's own scanner, .gitignore rules, or editors.
    workspace = args.work_dir or Path(tempfile.mkdtemp(prefix="incode-benchmark-"))
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace / "corpus"
    shapes = [shape for shape in args.shapes.split(",") if shape]
    corpus_bytes = write_corpus(root, scale=args.corpus_scale, shapes=shapes)
    cache_directory = args.model_cache or (workspace / "cache")
    offline = args.offline or args.model_cache is None

    run = run_benchmark(
        root=root,
        data_directory=workspace / "data",
        cache_directory=cache_directory,
        corpus_scale=args.corpus_scale,
        corpus_bytes=corpus_bytes,
        batch_size=args.batch_size,
        memory_mb=args.memory_mb,
        offline=offline,
        sample_interval=args.sample_interval,
        command_prefix=[sys.executable, "-m", "incode_mcp.cli"],
    )

    baseline = None
    if args.baseline is not None:
        baseline = BenchmarkRun.from_json(json.loads(args.baseline.read_text())["run"])
    verdict = evaluate_result(run, baseline=baseline)

    document = {
        "configuration": {
            "corpus_scale": args.corpus_scale,
            "shapes": shapes,
            "batch_size": args.batch_size,
            "memory_mb": args.memory_mb,
            "offline": offline,
            "sample_interval": args.sample_interval,
            "workspace": str(workspace),
        },
        "run": run.to_json(),
        "verdict": verdict.to_json(),
    }
    serialized = json.dumps(document, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
