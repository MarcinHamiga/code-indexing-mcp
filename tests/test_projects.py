from pathlib import Path

import pytest
import tomli_w

from incode_mcp.errors import ErrorCode, IncodeError
from incode_mcp.projects import (
    ProjectResolver,
    find_project_root,
    initialize_project,
    read_project_marker,
)


def test_initialize_project_creates_local_marker(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()

    project = initialize_project(root)

    marker = root / ".ci-mcp" / "project.toml"
    assert marker.exists()
    assert (root / ".ci-mcp" / ".gitignore").read_text() == "*\n"
    assert project.root == root.resolve()
    assert project.name == "demo"
    assert read_project_marker(root) == project


def test_legacy_marker_remains_readable(tmp_path: Path) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    project = initialize_project(root)
    current_directory = root / ".ci-mcp"
    legacy_directory = root / ".incode"
    if current_directory.exists():
        current_directory.rename(legacy_directory)

    assert find_project_root(root / "src") == root.resolve()
    assert read_project_marker(root) == project


def test_initialize_project_is_idempotent_unless_forced(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()

    first = initialize_project(root)
    second = initialize_project(root)
    replacement = initialize_project(root, force_new_id=True)

    assert second.id == first.id
    assert replacement.id != first.id


def test_legacy_default_marker_adds_java_without_rewriting_file(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    marker = root / ".incode" / "project.toml"
    marker.parent.mkdir()
    contents = tomli_w.dumps(
        {
            "version": 1,
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "demo",
            "scan": {
                "include": [
                    "**/*.py",
                    "**/*.pyi",
                    "**/*.js",
                    "**/*.jsx",
                    "**/*.mjs",
                    "**/*.cjs",
                    "**/*.ts",
                    "**/*.tsx",
                    "**/*.mts",
                    "**/*.cts",
                ],
                "exclude": [],
                "max_file_bytes": 1_048_576,
            },
        }
    )
    marker.write_text(contents)

    project = read_project_marker(root)

    assert project.scan.include[-1] == "**/*.java"
    assert marker.read_text() == contents


def test_custom_marker_includes_are_preserved(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    root.mkdir()
    marker = root / ".incode" / "project.toml"
    marker.parent.mkdir()
    marker.write_text(
        tomli_w.dumps(
            {
                "version": 1,
                "id": "00000000-0000-0000-0000-000000000001",
                "name": "demo",
                "scan": {"include": ["src/**/*.py"], "exclude": [], "max_file_bytes": 1_048_576},
            }
        )
    )

    project = read_project_marker(root)

    assert project.scan.include == ["src/**/*.py"]


def test_resolver_prefers_explicit_project(tmp_path: Path) -> None:
    one_root = tmp_path / "one"
    two_root = tmp_path / "two"
    one_root.mkdir()
    two_root.mkdir()
    one = initialize_project(one_root)
    two = initialize_project(two_root)
    resolver = ProjectResolver([one, two])

    assert resolver.resolve(explicit=two.id, roots=[one_root], cwd=one_root) == two
    assert resolver.resolve(explicit=str(two_root), roots=[one_root], cwd=one_root) == two


def test_resolver_uses_single_marked_root_then_cwd(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "src" / "pkg"
    nested.mkdir(parents=True)
    project = initialize_project(root)
    resolver = ProjectResolver([project])

    assert resolver.resolve(roots=[root], cwd=tmp_path) == project
    assert resolver.resolve(roots=[], cwd=nested) == project


def test_resolver_rejects_ambiguous_roots(tmp_path: Path) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
    projects = [initialize_project(root) for root in roots]
    resolver = ProjectResolver(projects)

    with pytest.raises(IncodeError) as raised:
        resolver.resolve(roots=roots, cwd=tmp_path)

    assert raised.value.code is ErrorCode.AMBIGUOUS_PROJECT
