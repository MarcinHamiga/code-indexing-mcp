import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pathspec import GitIgnoreSpec

from code_indexing_mcp.models import DEFAULT_INCLUDES, ScanConfig, ScannedFile
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import LANGUAGES, SourceScanner, _GitEnumerationError


def test_scanner_honors_languages_gitignore_and_hard_exclusions(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    (root / "main.py").write_text("print('ok')\n")
    (root / "component.tsx").write_text("export const App = () => <div />;\n")
    (root / "notes.md").write_text("not source\n")
    (root / "ignored.py").write_text("ignored = True\n")
    (root / ".gitignore").write_text("ignored.py\n")
    vendor = root / "node_modules"
    vendor.mkdir()
    (vendor / "vendor.js").write_text("export default 1\n")

    result = SourceScanner().scan(project)

    assert [(item.path.as_posix(), item.language) for item in result.files] == [
        ("component.tsx", "tsx"),
        ("main.py", "python"),
    ]
    assert {skip.reason for skip in result.skipped} >= {"unsupported", "ignored"}


def test_scanner_honors_git_info_exclude(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    (root / "main.py").write_text("value = 1\n")
    (root / "local_only.py").write_text("local = True\n")
    (root / ".git" / "info" / "exclude").write_text("local_only.py\n")

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["main.py"]
    # Git's own enumeration applies info/exclude before the scanner sees the
    # file, so an excluded file is absent from both files and skipped.
    assert not any(item.path.as_posix() == "local_only.py" for item in result.skipped)


def test_git_repo_scan_mixes_tracked_and_untracked_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    (root / "tracked.py").write_text("tracked = True\n")
    (root / "untracked.ts").write_text("export const x = 1\n")
    (root / "notes.md").write_text("not source\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=root, check=True)

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["tracked.py", "untracked.ts"]
    assert any(
        item.path.as_posix() == "notes.md" and item.reason == "unsupported"
        for item in result.skipped
    )


def test_tracked_but_ignored_file_stays_eligible(tmp_path: Path) -> None:
    """Git's own rule is that the index wins: a file that is ignored but was
    force-added stays tracked, so Git enumerates it and the scanner indexes it.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    (root / ".gitignore").write_text("ignored.py\n")
    (root / "ignored.py").write_text("value = 1\n")
    (root / "main.py").write_text("value = 2\n")
    subprocess.run(["git", "add", "main.py"], cwd=root, check=True)
    subprocess.run(["git", "add", "-f", "ignored.py"], cwd=root, check=True)

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["ignored.py", "main.py"]


def test_git_submodule_and_nested_repository_are_opaque(tmp_path: Path) -> None:
    """Submodules (gitlinks) and nested repositories are single non-file entries
    in Git's enumeration, so their contents are not indexed from the parent.
    """
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True)
    sub = tmp_path / "sub"
    sub.mkdir()
    subprocess.run(["git", "init", "-q", str(sub)], check=True)
    (sub / "sub.py").write_text("def sub_symbol():\n    return 1\n")
    subprocess.run(["git", "add", "sub.py"], cwd=sub, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "sub"],
        cwd=sub,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(sub),
            "vendor/sub",
        ],
        cwd=outer,
        check=True,
    )
    nested = outer / "nested"
    nested.mkdir()
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    (nested / "nested.py").write_text("def nested_symbol():\n    return 2\n")
    (outer / "main.py").write_text("def main_symbol():\n    return 3\n")
    project = initialize_project(outer)

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["main.py"]
    assert not any("vendor/sub" in item.path.as_posix() for item in result.files)
    assert not any("nested" in item.path.as_posix() for item in result.files)


def test_git_worktree_is_scanned_as_a_normal_checkout(tmp_path: Path) -> None:
    main_root = tmp_path / "repo"
    main_root.mkdir()
    subprocess.run(["git", "init", "-q", str(main_root)], check=True)
    (main_root / "main.py").write_text("value = 1\n")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "main.py"],
        cwd=main_root,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=main_root,
        check=True,
    )
    worktree = tmp_path / "wt"
    subprocess.run(["git", "worktree", "add", "-q", str(worktree)], cwd=main_root, check=True)
    project = initialize_project(worktree)

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["main.py"]


def test_git_enumeration_order_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    for name in ("c.py", "a.py", "b.py"):
        (root / name).write_text("value = 1\n")

    first = SourceScanner().scan(project)
    second = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in first.files] == ["a.py", "b.py", "c.py"]
    assert [item.path.as_posix() for item in first.files] == [
        item.path.as_posix() for item in second.files
    ]


def test_scan_stats_only_files_whose_suffix_is_supported(tmp_path: Path) -> None:
    """A repository dominated by unsupported files must not pay a stat per file
    (the 100,000-file gate, at small scale): only pre-filtered candidates reach
    the filesystem. Git mode never even passes unsupported paths to Git.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    for index in range(100):
        (root / f"doc{index:03d}.md").write_text("not source\n")
    for index in range(5):
        (root / f"module{index}.py").write_text("value = 1\n")
    statted: list[Path] = []
    original_stat = Path.stat

    def counting_stat(path: Path, *args, **kwargs):
        statted.append(path)
        return original_stat(path, *args, **kwargs)

    with patch.object(Path, "stat", counting_stat):
        result = SourceScanner().scan(project)

    assert len(result.files) == 5
    # Only supported-suffix files are ever statted (the root directory itself
    # is statted once by resolve(), which is not a per-file cost).
    assert not any(path.suffix == ".md" for path in statted)


def test_walk_mode_streams_per_directory_in_deterministic_order(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    project = initialize_project(root)
    (root / "a" / "z.py").write_text("value = 1\n")
    (root / "b" / "a.py").write_text("value = 2\n")

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["a/z.py", "b/a.py"]


def test_walk_batches_stay_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    for index in range(10):
        (root / f"file{index:02d}.py").write_text("value = 1\n")
    monkeypatch.setattr("code_indexing_mcp.scanner.GIT_IGNORE_DISCOVERY_BATCH_SIZE", 4)
    include_spec = GitIgnoreSpec.from_lines(project.scan.include)

    batches = list(SourceScanner()._iter_walk_batches(root, include_spec))

    sizes = [len(batch) for batch in batches if isinstance(batch, list)]
    assert sizes == [4, 4, 2]


def test_walk_mode_treats_nested_repositories_as_opaque(tmp_path: Path) -> None:
    """A non-Git walk must not descend into a directory carrying a ``.git``
    entry: a nested repository or submodule is opaque, matching what
    ``git ls-files`` reports on the git path.
    """
    root = tmp_path / "repo"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: ../.git/modules/nested\n")
    (nested / "nested.py").write_text("def nested_symbol():\n    return 2\n")
    (root / "main.py").write_text("def main_symbol():\n    return 3\n")
    project = initialize_project(root)

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["main.py"]
    assert not any("nested" in item.path.as_posix() for item in result.files)


def test_walk_fallback_keeps_tracked_but_ignored_files_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk fallback inside a worktree consults the index (no
    ``--no-index``), so a force-added file that is both tracked and ignored
    stays eligible exactly as on the git path: the index wins.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    (root / ".gitignore").write_text("ignored.py\n")
    (root / "ignored.py").write_text("value = 1\n")
    (root / "main.py").write_text("value = 2\n")
    subprocess.run(["git", "add", "main.py"], cwd=root, check=True)
    subprocess.run(["git", "add", "-f", "ignored.py"], cwd=root, check=True)
    scanner = SourceScanner()

    def failing_enumeration(_: Path):
        raise _GitEnumerationError("simulated git failure")
        yield  # pragma: no cover

    monkeypatch.setattr(scanner, "_iter_git_batches", failing_enumeration)

    result = scanner.scan(project)

    assert [item.path.as_posix() for item in result.files] == ["ignored.py", "main.py"]


def test_failed_git_enumeration_falls_back_to_the_streaming_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    (root / "main.py").write_text("value = 1\n")
    scanner = SourceScanner()

    def broken_enumeration(_: Path):
        raise _GitEnumerationError("simulated git failure")
        yield  # pragma: no cover

    monkeypatch.setattr(scanner, "_iter_git_batches", broken_enumeration)

    result = scanner.scan(project)

    assert [item.path.as_posix() for item in result.files] == ["main.py"]


def test_walk_fallback_after_partial_git_enumeration_never_repeats_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A git enumeration that fails after streaming some batches must not let
    the walk fallback yield those files again: the indexer queues pending work
    per yielded file, and a repeat would stage one file's chunk rows under two
    owners, crashing the staging contiguity invariant."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    (root / "a.py").write_text("value = 1\n")
    (root / "b.py").write_text("value = 2\n")
    (root / "notes.md").write_text("not source\n")
    scanner = SourceScanner()

    def partially_failing_enumeration(_: Path):
        yield [root / "a.py", root / "notes.md"]
        raise _GitEnumerationError("simulated mid-stream git failure")

    monkeypatch.setattr(scanner, "_iter_git_batches", partially_failing_enumeration)

    result = scanner.scan(project)

    assert [item.path.as_posix() for item in result.files] == ["a.py", "b.py"]
    assert [skip.path.as_posix() for skip in result.skipped] == ["notes.md"]


def test_walk_mode_passes_only_supported_files_to_check_ignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rare walk-inside-a-worktree fallback must not hand unsupported
    files to `git check-ignore`: the pre-filter happens before the batch.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    project = initialize_project(root)
    for index in range(20):
        (root / f"doc{index:03d}.md").write_text("not source\n")
    for index in range(5):
        (root / f"module{index}.py").write_text("value = 1\n")
    scanner = SourceScanner()
    batches: list[list[Path]] = []

    def recording_ignored(_: Path, candidates: list[Path]) -> set[Path]:
        batches.append(candidates)
        return set()

    def failing_enumeration(_: Path):
        raise _GitEnumerationError("simulated git failure")
        yield  # pragma: no cover

    monkeypatch.setattr(scanner, "_git_ignored_paths", recording_ignored)
    monkeypatch.setattr(scanner, "_in_git_worktree", lambda _: True)
    monkeypatch.setattr(scanner, "_iter_git_batches", failing_enumeration)

    result = scanner.scan(project)

    assert len(result.files) == 5
    assert batches and all(path.suffix == ".py" for batch in batches for path in batch)


def test_default_includes_and_the_extension_map_describe_the_same_languages() -> None:
    """The two lists are edited separately, and either one alone is useless.

    An extension the scanner can classify but no default pattern matches is
    never offered a file; a default pattern with no extension entry matches
    files the scanner then rejects as unsupported.
    """
    assert {pattern.removeprefix("**/*") for pattern in DEFAULT_INCLUDES} == set(LANGUAGES)


def test_next_language_extensions_have_stable_language_names() -> None:
    expected = {
        ".go": "go",
        ".tf": "terraform",
        ".tfvars": "terraform",
        ".rs": "rust",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".hh": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".lua": "lua",
    }
    assert all(LANGUAGES.get(extension) == language for extension, language in expected.items())


def test_scanner_discovers_every_default_language(tmp_path: Path) -> None:
    """Every extension in the map is discovered under the default include list.

    Driven off `LANGUAGES` rather than a hand-written list so a newly mapped
    extension cannot quietly go undiscovered: the file is written from the map,
    so it is missing from the result until the default patterns cover it too.
    """
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    for extension in LANGUAGES:
        (root / f"sample{extension}").write_text("sample\n")

    result = SourceScanner().scan(project)

    assert [(item.path.as_posix(), item.language) for item in result.files] == sorted(
        (f"sample{extension}", language) for extension, language in LANGUAGES.items()
    )


def test_godot_cache_directory_is_excluded_without_excluding_the_project_file(
    tmp_path: Path,
) -> None:
    """`.godot` names both an indexed extension and Godot's asset cache.

    A Godot project that has been opened in the editor carries a `.godot`
    directory holding a generated copy of every imported asset, including
    scenes. Nothing there is source, and the project's own `project.godot` has
    to survive the exclusion.
    """
    root = tmp_path / "game"
    root.mkdir()
    project = initialize_project(root)
    (root / "project.godot").write_text("config_version=5\n")
    (root / "level.tscn").write_text('[node name="Player" type="Node2D"]\n')
    cache = root / ".godot" / "imported"
    cache.mkdir(parents=True)
    (cache / "level.tscn").write_text('[node name="Generated" type="Node2D"]\n')

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["level.tscn", "project.godot"]


def test_scanner_applies_nested_gitignore_and_config_excludes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package = root / "package"
    package.mkdir(parents=True)
    project = initialize_project(root)
    (package / ".gitignore").write_text("generated.py\n")
    (package / "generated.py").write_text("generated = True\n")
    (package / "keep.py").write_text("keep = True\n")
    project = project.model_copy(
        update={"scan": project.scan.model_copy(update={"exclude": ["package/keep.py"]})}
    )

    result = SourceScanner().scan(project)

    assert result.files == []


def test_nested_gitignore_can_reinclude_a_file(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package = root / "package"
    package.mkdir(parents=True)
    project = initialize_project(root)
    (root / ".gitignore").write_text("package/*.py\n")
    (package / ".gitignore").write_text("!keep.py\n")
    (package / "keep.py").write_text("keep = True\n")
    (package / "drop.py").write_text("drop = True\n")

    result = SourceScanner().scan(project)

    assert [item.path.as_posix() for item in result.files] == ["package/keep.py"]


def test_scanner_rejects_oversized_and_symlink_files_without_reading(tmp_path: Path) -> None:
    """Size and symlink checks are stat-only; content checks belong to the indexer.

    The scanner used to read every changed file to test for NUL bytes and UTF-8
    validity, then discard the bytes, so the indexer read the same file again.
    """
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    project = project.model_copy(
        update={"scan": project.scan.model_copy(update={"max_file_bytes": 8})}
    )
    (root / "large.py").write_text("0123456789")
    (root / "binary.py").write_bytes(b"a\x00b")
    target = tmp_path / "target.py"
    target.write_text("x = 1\n")
    (root / "link.py").symlink_to(target)

    result = SourceScanner().scan(project)

    reasons = {skip.reason for skip in result.skipped}
    assert reasons >= {"oversized", "symlink"}
    assert "binary" not in reasons
    assert "encoding" not in reasons
    # binary.py is 3 bytes, so it now passes the stat-only scan.
    assert {item.path.as_posix() for item in result.files} == {"binary.py"}


def test_scan_paths_classifies_exactly_the_listed_candidates(tmp_path: Path) -> None:
    """scan_paths answers the freshness fast path: stat only the named paths.

    Mirrors what iter_scan would report for each of these paths individually,
    without walking the rest of the tree.
    """
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    project = project.model_copy(
        update={
            "scan": project.scan.model_copy(
                update={"max_file_bytes": 9, "exclude": ["excluded.py"]}
            )
        }
    )
    # Bytes, not text: write_text would translate the newline to CRLF on
    # Windows and the size assertion below is about bytes on disk.
    (root / "ok.py").write_bytes(b"v = 1\n")
    (root / "large.py").write_text("0123456789")
    (root / "notes.md").write_text("not source\n")
    (root / "excluded.py").write_text("value = 2\n")
    target = tmp_path / "target.py"
    target.write_text("x = 1\n")
    (root / "link.py").symlink_to(target)

    results = list(
        SourceScanner().scan_paths(
            project,
            [
                "ok.py",
                "large.py",
                "notes.md",
                "excluded.py",
                "link.py",
                "missing.py",
            ],
        )
    )

    by_path = {item.path.as_posix(): item for item in results}
    assert set(by_path) == {"ok.py", "large.py", "notes.md", "excluded.py", "link.py"}
    ok = by_path["ok.py"]
    assert isinstance(ok, ScannedFile)
    assert ok.size == len(b"v = 1\n")
    assert ok.mtime_ns == (root / "ok.py").stat().st_mtime_ns
    assert ok.content is None
    assert by_path["large.py"].reason == "oversized"
    assert by_path["notes.md"].reason == "unsupported"
    assert by_path["excluded.py"].reason == "ignored"
    assert by_path["link.py"].reason == "symlink"


def test_scan_paths_yields_nothing_for_a_missing_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)

    results = list(SourceScanner().scan_paths(project, ["does-not-exist.py"]))

    assert results == []


def test_scanner_does_not_read_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ok.py").write_text("def ok():\n    return 1\n")
    project = initialize_project(root)

    def reject_read(self: Path) -> bytes:
        raise AssertionError(f"scan must not read {self}")

    monkeypatch.setattr(Path, "read_bytes", reject_read)

    assert len(SourceScanner().scan(project).files) == 1


def test_iter_scan_reads_one_file_source_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming means laziness: a file's bytes are read only as it is yielded."""
    root = tmp_path / "repo"
    root.mkdir()
    # Written as bytes: the assertions below are byte-exact, and write_text
    # would turn the newlines into CRLF on Windows.
    (root / "a.py").write_bytes(b"a = 1\n")
    (root / "b.py").write_bytes(b"b = 2\n")
    project = initialize_project(root)
    reads: list[Path] = []
    original = Path.read_bytes

    def tracking_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracking_read_bytes)
    stream = SourceScanner().iter_scan(project)

    first = next(stream)
    assert len(reads) == 1
    second = next(stream)
    assert len(reads) == 2

    # The streaming path hands the source to the caller instead of re-reading.
    assert first.content == b"a = 1\n"
    assert second.content == b"b = 2\n"


def test_scanner_never_walks_hard_excluded_directories(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    project = initialize_project(root)
    (root / "main.py").write_text("print('ok')\n")
    vendor = root / "node_modules"
    vendor.mkdir()
    (vendor / "vendor.js").write_text("export default 1\n")
    git = root / ".git"
    git.mkdir()
    (git / "hook.py").write_text("hook = True\n")
    for marker_directory in (".code-indexing-mcp", ".ci-mcp"):
        marker = root / marker_directory
        marker.mkdir(exist_ok=True)
        (marker / "private.py").write_text("private = True\n")

    stat_failures = []
    original_stat = Path.stat

    def fail_if_excluded_is_statted(path: Path, *args, **kwargs):
        if any(
            excluded in path.parts
            for excluded in ("node_modules", ".git", ".code-indexing-mcp", ".ci-mcp")
        ):
            stat_failures.append(path)
            raise AssertionError(f"excluded path was statted: {path}")
        return original_stat(path, *args, **kwargs)

    with patch.object(Path, "stat", fail_if_excluded_is_statted):
        result = SourceScanner().scan(project)

    assert stat_failures == []
    assert [item.path.as_posix() for item in result.files] == ["main.py"]


def test_has_supported_source_respects_ignore_and_hard_exclusion_rules(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("ignored.py\n")
    (root / "ignored.py").write_text("value = 1\n")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "vendor.js").write_text("export default 1\n")

    assert SourceScanner().has_supported_source(root, ScanConfig()) is False

    (root / "main.ts").write_text("export const answer = 42\n")

    assert SourceScanner().has_supported_source(root, ScanConfig()) is True


def test_has_supported_source_applies_nested_gitignore_rules(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package = root / "package"
    package.mkdir(parents=True)
    (root / ".gitignore").write_text("package/*.py\n")
    (package / ".gitignore").write_text("!keep.py\n")
    keep = package / "keep.py"
    keep.write_text("keep = True\n")
    (package / "drop.py").write_text("drop = True\n")
    scanner = SourceScanner()

    assert scanner.has_supported_source(root, ScanConfig()) is True

    keep.unlink()

    assert scanner.has_supported_source(root, ScanConfig()) is False


def test_has_supported_source_batches_git_ignore_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for index in range(300):
        (root / f"ignored_{index:03d}.py").write_text("ignored = True\n")
    scanner = SourceScanner()
    batches: list[list[Path]] = []

    def all_ignored(_: Path, candidates: list[Path]) -> set[Path]:
        batches.append(candidates)
        return {path.relative_to(root) for path in candidates}

    monkeypatch.setattr(scanner, "_git_ignored_paths", all_ignored)

    assert scanner.has_supported_source(root, ScanConfig()) is False
    assert len(batches) == 2
    assert max(map(len, batches)) == 256


def test_git_ignore_timeout_falls_back_to_local_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "main.py"
    source.write_text("value = 1\n")

    def time_out(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired("git check-ignore", timeout=10)

    monkeypatch.setattr(subprocess, "run", time_out)

    assert SourceScanner._git_ignored_paths(root, [source]) == set()


def test_language_name_literal_matches_scanner_languages() -> None:
    from typing import get_args

    from code_indexing_mcp.models import LanguageName
    from code_indexing_mcp.scanner import LANGUAGES

    assert set(get_args(LanguageName)) == set(LANGUAGES.values())
