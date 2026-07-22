from pathlib import Path

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
