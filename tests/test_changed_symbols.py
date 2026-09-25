"""changed_symbols: Git change sets mapped onto indexed declarations."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import run_git
from support import DeterministicEmbedder

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.changes import FileChange, parse_name_status, parse_unified_zero
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.models import ChunkPreview, OutlineItem
from code_indexing_mcp.search import _outline_items

SOURCE = """\
def first():
    return 1


def second():
    value = 2
    return value


class Holder:
    def method(self):
        return 3
"""


def _commit(root: Path, message: str) -> None:
    run_git("add", "-A", cwd=root)
    run_git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", message, cwd=root)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    run_git("init", "-q", "--initial-branch", "main", str(root))
    (root / "main.py").write_text(SOURCE)
    (root / "other.py").write_text("def other():\n    return 0\n")
    (root / ".gitignore").write_text(".ci-mcp/\n")
    _commit(root, "initial")
    return root


def _app(tmp_path: Path, root: Path) -> Application:
    return Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=DeterministicEmbedder(),
        cwd=root,
    )


def _indexed(tmp_path: Path) -> tuple[Path, Application, str]:
    root = _repo(tmp_path)
    app = _app(tmp_path, root)
    project = app.init_project(root)
    app.index_project(project.id)
    return root, app, project.id


def _item(name: str, start: int, end: int) -> OutlineItem:
    return OutlineItem(
        kind="function", symbol=name, qualified_symbol=name, start_line=start, end_line=end
    )


# -- parsing ---------------------------------------------------------------


def test_unified_zero_patch_yields_new_side_ranges_and_deletion_points() -> None:
    patch = (
        "diff --git a/src/a.py b/src/a.py\n"
        "index 1..2 100644\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -3 +3 @@ def f():\n"
        "-    return 1\n"
        "+    return 2\n"
        "@@ -10,2 +9,0 @@\n"
        "--- a removed SQL-style comment line\n"
        "-+++ and one that looks like a header\n"
        "@@ -20,0 +21,3 @@\n"
        "+a\n+b\n+c\n"
    )

    hunks = parse_unified_zero(patch)

    assert list(hunks) == ["src/a.py"]
    assert hunks["src/a.py"].ranges == [(3, 3), (21, 23)]
    assert hunks["src/a.py"].deletion_points == [9]


def test_patch_paths_with_spaces_quotes_binary_and_mode_changes() -> None:
    patch = (
        "diff --git a/with space.py b/with space.py\n"
        "--- a/with space.py\t\n"
        "+++ b/with space.py\t\n"
        "@@ -1 +1 @@\n-x\n+y\n"
        'diff --git "a/tab\\there.py" "b/tab\\there.py"\n'
        '--- "a/tab\\there.py"\n'
        '+++ "b/tab\\there.py"\n'
        "@@ -2 +2,2 @@\n-x\n+y\n+z\n"
        "diff --git a/image.png b/image.png\n"
        "index 1..2 100644\n"
        "Binary files a/image.png and b/image.png differ\n"
        "diff --git a/run.sh b/run.sh\n"
        "old mode 100644\n"
        "new mode 100755\n"
    )

    hunks = parse_unified_zero(patch)

    assert hunks["with space.py"].ranges == [(1, 1)]
    assert hunks["tab\there.py"].ranges == [(2, 3)]
    assert hunks["image.png"].binary is True
    assert hunks["run.sh"].ranges == [] and hunks["run.sh"].binary is False


def test_name_status_maps_git_letters_to_change_kinds() -> None:
    output = "M\0a.py\0A\0b.py\0D\0c.py\0T\0d.py\0"

    assert parse_name_status(output) == {
        "a.py": "modified",
        "b.py": "added",
        "c.py": "deleted",
        "d.py": "modified",
    }


def test_touched_matches_overlapping_ranges_and_enclosed_deletions() -> None:
    items = [_item("a", 1, 3), _item("b", 5, 9), _item("c", 11, 12)]

    assert FileChange("x.py", "modified", ranges=((3, 5),)).touched(items) == items[:2]
    # Lines removed between current lines 6 and 7 fell inside b; lines
    # removed between 9 and 10 fell between declarations and touch neither.
    assert FileChange("x.py", "modified", deletion_points=(6, 9)).touched(items) == [items[1]]
    assert FileChange("x.py", "added", whole_file=True).touched(items) == items
    assert FileChange("x.py", "deleted").touched(items) == []


def test_changed_outline_spans_every_part_of_a_split_declaration() -> None:
    def part(start: int, end: int) -> ChunkPreview:
        return ChunkPreview(
            chunk_id=f"c{start}",
            project_id="p",
            path="big.py",
            language="python",
            kind="function_part",
            symbol="big",
            qualified_symbol="big",
            start_line=start,
            end_line=end,
        )

    chunks = [part(1, 40), part(41, 80)]

    assert _outline_items(chunks, "big.py")[0].end_line == 40
    assert _outline_items(chunks, "big.py", span_parts=True)[0].end_line == 80


# -- application -----------------------------------------------------------


def test_uncommitted_edit_reports_only_the_declaration_it_touches(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    (root / "main.py").write_text(SOURCE.replace("value = 2", "value = 22"))
    app.index_project(project_id)

    response = app.changed_symbols(project_id)

    assert response.base == response.head
    assert [file.path for file in response.files] == ["main.py"]
    changed = response.files[0]
    assert changed.change == "modified"
    assert changed.indexed is True
    assert changed.index_current is True
    assert [(line.start_line, line.end_line) for line in changed.changed_lines] == [(6, 6)]
    assert [symbol.qualified_symbol for symbol in changed.symbols] == ["second"]


def test_untracked_and_deleted_files(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    (root / "fresh.py").write_text("def fresh():\n    return 1\n\n\ndef newer():\n    return 2\n")
    (root / "other.py").unlink()
    app.index_project(project_id)

    response = app.changed_symbols(project_id)
    files = {file.path: file for file in response.files}

    assert set(files) == {"fresh.py", "other.py"}
    assert files["fresh.py"].change == "untracked"
    assert [symbol.symbol for symbol in files["fresh.py"].symbols] == ["fresh", "newer"]
    assert files["other.py"].change == "deleted"
    assert files["other.py"].indexed is False
    assert files["other.py"].symbols == []

    tracked_only = app.changed_symbols(project_id, include_untracked=False)
    assert [file.path for file in tracked_only.files] == ["other.py"]


def test_since_a_commit_includes_committed_changes(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    base = app.project_status(project_id).git_head
    (root / "main.py").write_text(SOURCE + "\n\ndef third():\n    return 4\n")
    _commit(root, "add third")
    app.index_project(project_id)

    assert app.changed_symbols(project_id).files == []
    response = app.changed_symbols(project_id, since="HEAD~1")

    assert response.base == base
    assert [symbol.symbol for symbol in response.files[0].symbols] == ["third"]
    assert app.changed_symbols(project_id, since=base).files == response.files


def test_since_time_uses_the_last_commit_before_it(tmp_path: Path) -> None:
    _, app, project_id = _indexed(tmp_path)

    response = app.changed_symbols(project_id, since_time="2090-01-01T00:00:00Z")

    assert response.base == app.project_status(project_id).git_head
    with pytest.raises(CodeIndexingError) as raised:
        app.changed_symbols(project_id, since_time="1990-01-01")
    assert raised.value.code is ErrorCode.INVALID_FILTER


def test_stale_index_is_flagged_per_file(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    (root / "main.py").write_text(SOURCE.replace("return 1", "return 11"))

    stale = app.changed_symbols(project_id)
    assert stale.files[0].index_current is False

    app.index_project(project_id)
    assert app.changed_symbols(project_id).files[0].index_current is True


@pytest.mark.parametrize(
    ("arguments", "code"),
    [
        ({"since": "no-such-branch"}, ErrorCode.INVALID_FILTER),
        ({"since": "--output=/tmp/x"}, ErrorCode.INVALID_FILTER),
        ({"since": "HEAD", "since_time": "yesterday"}, ErrorCode.INVALID_FILTER),
        ({"limit": 0}, ErrorCode.INVALID_FILTER),
    ],
)
def test_invalid_bases_and_limits_are_rejected(
    tmp_path: Path, arguments: dict[str, object], code: ErrorCode
) -> None:
    _, app, project_id = _indexed(tmp_path)

    with pytest.raises(CodeIndexingError) as raised:
        app.changed_symbols(project_id, **arguments)  # type: ignore[arg-type]

    assert raised.value.code is code


def test_limit_truncates_in_path_order(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    for name in ("c.py", "a.py", "b.py"):
        (root / name).write_text(f"def {name[0]}():\n    return 1\n")

    response = app.changed_symbols(project_id, limit=2)

    assert [file.path for file in response.files] == ["a.py", "b.py"]
    assert response.total_files == 3
    assert response.truncated is True


def test_subdirectory_project_sees_only_its_subtree(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    service = root / "service"
    service.mkdir()
    (service / "api.py").write_text("def handler():\n    return 1\n")
    _commit(root, "service")
    app = _app(tmp_path, service)
    project = app.init_project(service)
    app.index_project(project.id)
    (service / "api.py").write_text("def handler():\n    return 2\n")
    (root / "main.py").write_text(SOURCE.replace("return 1", "return 11"))

    response = app.changed_symbols(project.id)

    assert [file.path for file in response.files] == ["api.py"]
    assert [symbol.symbol for symbol in response.files[0].symbols] == ["handler"]


def test_non_git_project_is_unsupported(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    root.mkdir()
    (root / "main.py").write_text(SOURCE)
    app = _app(tmp_path, root)
    project = app.init_project(root)

    with pytest.raises(CodeIndexingError) as raised:
        app.changed_symbols(project.id)

    assert raised.value.code is ErrorCode.UNSUPPORTED_OPERATION
