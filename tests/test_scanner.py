from pathlib import Path
from unittest.mock import patch

from incode_mcp.models import ScanConfig
from incode_mcp.projects import initialize_project
from incode_mcp.scanner import SourceScanner


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


def test_scanner_rejects_oversized_binary_and_symlink_files(tmp_path: Path) -> None:
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

    assert result.files == []
    assert {skip.reason for skip in result.skipped} >= {"oversized", "binary", "symlink"}


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

    stat_failures = []
    original_stat = Path.stat

    def fail_if_excluded_is_statted(path: Path, *args, **kwargs):
        if "node_modules" in path.parts or ".git" in path.parts:
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
