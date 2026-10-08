"""changed_symbols: Git change sets mapped onto indexed declarations."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import run_git
from support import DeterministicEmbedder

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.changes import Deletion, FileChange, parse_name_status, parse_unified_zero
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

SHAPES = """\
class Shapes {
    int area(int side) {
        return side * side;
    }

    int perimeter(int side) {
        return 4 * side;
    }

    int area(int width, int height) {
        return width * height;
    }
}
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


def _chunk(kind: str, name: str, start: int, end: int) -> ChunkPreview:
    return ChunkPreview(
        chunk_id=f"{name}:{start}",
        project_id="p",
        path="big.py",
        language="python",
        kind=kind,
        symbol=name.rsplit(".", 1)[-1],
        qualified_symbol=name,
        start_line=start,
        end_line=end,
    )


def _spans(items: list[OutlineItem]) -> list[tuple[str | None, int, int]]:
    return [(item.qualified_symbol, item.start_line, item.end_line) for item in items]


# -- parsing ---------------------------------------------------------------


def test_unified_zero_patch_yields_new_side_ranges_and_deletions() -> None:
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
        "@@ -30,2 +30,0 @@\n"
        "-\n"
        "-        tail()\n"
        "@@ -40 +37,0 @@\n"
        "-\n"
    )

    hunks = parse_unified_zero(patch)

    assert list(hunks) == ["src/a.py"]
    assert hunks["src/a.py"].ranges == [(3, 3), (21, 23)]
    # Each deletion carries its first non-blank removed line's indentation.
    assert hunks["src/a.py"].deletions == [
        Deletion(after=9, indent=0),
        Deletion(after=30, indent=8),
        Deletion(after=37),
    ]


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
    deletions = (Deletion(after=6), Deletion(after=9))
    assert FileChange("x.py", "modified", deletions=deletions).touched(items) == [items[1]]
    assert FileChange("x.py", "added", whole_file=True).touched(items) == items
    assert FileChange("x.py", "deleted").touched(items) == []


def test_removing_the_tail_of_an_indentation_scoped_body_touches_it() -> None:
    lines = ["def a():", "    x = 1", "", "", "def b():", "    return 2", ""]
    items = [_item("a", 1, 2), _item("b", 5, 6)]

    # a's last statement went: the removal ends a and is indented deeper
    # than a's first line, so it was a's own.
    tail = FileChange("x.py", "modified", deletions=(Deletion(after=2, indent=4),))
    assert tail.touched(items, lines) == [items[0]]
    # A removed top-level sibling, or only blank lines, belonged to neither.
    sibling = FileChange(
        "x.py", "modified", deletions=(Deletion(after=2, indent=0), Deletion(after=2))
    )
    assert sibling.touched(items, lines) == []
    # Without the file text only a removal both neighbours enclose matches.
    assert tail.touched(items) == []


def test_changed_outline_spans_every_part_of_a_split_declaration() -> None:
    chunks = [_chunk("function_part", "big", 1, 40), _chunk("function_part", "big", 41, 80)]

    assert _outline_items(chunks, "big.py")[0].end_line == 40
    assert _outline_items(chunks, "big.py", span_parts=True)[0].end_line == 80


def test_changed_outline_keeps_same_named_declarations_apart() -> None:
    overloads = [
        _chunk("method", "Shapes.area", 2, 4),
        _chunk("method", "Shapes.perimeter", 6, 8),
        _chunk("method", "Shapes.area", 10, 12),
    ]

    assert _spans(_outline_items(overloads, "big.py", span_parts=True)) == [
        ("Shapes.area", 2, 4),
        ("Shapes.perimeter", 6, 8),
        ("Shapes.area", 10, 12),
    ]
    # file_outline keeps its one entry per name.
    assert _spans(_outline_items(overloads, "big.py")) == [
        ("Shapes.area", 2, 4),
        ("Shapes.perimeter", 6, 8),
    ]

    # Overlapping parts merge; two split overloads do not merge across the
    # declaration between them.
    split = [
        _chunk("method_part", "Shapes.area", 2, 30),
        _chunk("method_part", "Shapes.area", 21, 50),
        _chunk("method", "Shapes.perimeter", 52, 54),
        _chunk("method_part", "Shapes.area", 56, 90),
        _chunk("method_part", "Shapes.area", 81, 120),
    ]
    assert _spans(_outline_items(split, "big.py", span_parts=True)) == [
        ("Shapes.area", 2, 50),
        ("Shapes.perimeter", 52, 54),
        ("Shapes.area", 56, 120),
    ]


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


def test_removing_a_python_body_tail_touches_only_that_function(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    (root / "main.py").write_text(SOURCE.replace("    return value\n", ""))
    app.index_project(project_id)

    changed = app.changed_symbols(project_id).files[0]

    assert changed.changed_lines == []
    assert [symbol.qualified_symbol for symbol in changed.symbols] == ["second"]

    # Removing a whole function touches neither neighbour.
    (root / "main.py").write_text(
        SOURCE.replace("def second():\n    value = 2\n    return value\n\n\n", "")
    )
    app.index_project(project_id)
    assert app.changed_symbols(project_id).files[0].symbols == []


def test_overloads_are_matched_one_by_one(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    (root / "Shapes.java").write_text(SHAPES)
    _commit(root, "shapes")
    (root / "Shapes.java").write_text(SHAPES.replace("return 4 * side;", "return side * 4;"))
    app.index_project(project_id)

    changed = app.changed_symbols(project_id).files[0]

    assert [symbol.qualified_symbol for symbol in changed.symbols] == ["Shapes.perimeter"]


def test_files_that_are_not_utf8_are_still_diffed(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    (root / "legacy.properties").write_bytes("greeting=caf\xe9\n".encode("latin-1"))
    _commit(root, "legacy")
    (root / "legacy.properties").write_bytes("greeting=d\xe9j\xe0 vu\n".encode("latin-1"))
    (root / "staged.properties").write_bytes("name=Jos\xe9\n".encode("latin-1"))
    run_git("add", "staged.properties", cwd=root)

    files = {file.path: file for file in app.changed_symbols(project_id).files}

    assert files["staged.properties"].change == "added"
    assert files["legacy.properties"].change == "modified"
    assert [
        (line.start_line, line.end_line) for line in files["legacy.properties"].changed_lines
    ] == [(1, 1)]


def test_a_file_dropped_from_the_index_but_kept_on_disk_is_untracked(tmp_path: Path) -> None:
    root, app, project_id = _indexed(tmp_path)
    run_git("rm", "-q", "--cached", "other.py", cwd=root)

    files = {file.path: file for file in app.changed_symbols(project_id).files}

    assert files["other.py"].change == "untracked"
    assert [symbol.symbol for symbol in files["other.py"].symbols] == ["other"]
    tracked_only = app.changed_symbols(project_id, include_untracked=False)
    assert [(file.path, file.change) for file in tracked_only.files] == [("other.py", "deleted")]


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
        # `--before` alone would read these as now and answer with HEAD.
        ({"since_time": "garbage"}, ErrorCode.INVALID_FILTER),
        ({"since_time": "yesterdy"}, ErrorCode.INVALID_FILTER),
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
