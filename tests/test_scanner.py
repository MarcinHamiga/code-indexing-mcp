from pathlib import Path
from unittest.mock import patch

import pytest

from code_indexing_mcp.models import DEFAULT_INCLUDES, ScanConfig
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.scanner import LANGUAGES, SourceScanner


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


def test_default_includes_and_the_extension_map_describe_the_same_languages() -> None:
    """The two lists are edited separately, and either one alone is useless.

    An extension the scanner can classify but no default pattern matches is
    never offered a file; a default pattern with no extension entry matches
    files the scanner then rejects as unsupported.
    """
    assert {pattern.removeprefix("**/*") for pattern in DEFAULT_INCLUDES} == set(LANGUAGES)


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


def test_language_name_literal_matches_scanner_languages() -> None:
    from typing import get_args

    from code_indexing_mcp.models import LanguageName
    from code_indexing_mcp.scanner import LANGUAGES

    assert set(get_args(LanguageName)) == set(LANGUAGES.values())
