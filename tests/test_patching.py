"""Byte-exactness contract for the patch renderer, on inline fixtures only."""

import pytest

import code_indexing_mcp.patching as patching
from code_indexing_mcp.patching import ByteEdit, apply_edits, render_unified_diff

_BOM = b"\xef\xbb\xbf"


def test_identical_bytes_render_nothing() -> None:
    assert render_unified_diff("a.py", b"same\n", b"same\n") is None


def test_a_single_mid_file_edit_renders_one_hunk() -> None:
    original = b"def authorize(user):\n    return user\n"
    edit = ByteEdit(4, 13, b"validate")
    edited = apply_edits(original, [edit])

    assert edited == b"def validate(user):\n    return user\n"
    assert render_unified_diff("auth.py", original, edited) == (
        "diff --git a/auth.py b/auth.py\n"
        "--- a/auth.py\n"
        "+++ b/auth.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def authorize(user):\n"
        "+def validate(user):\n"
        "     return user\n"
    )


def test_adjacent_edits_collapse_into_one_hunk() -> None:
    original = b"alpha();\nbeta();\ngamma();\n"
    edited = apply_edits(original, [ByteEdit(0, 5, b"ALPHA"), ByteEdit(9, 13, b"BETA")])

    assert edited == b"ALPHA();\nBETA();\ngamma();\n"
    assert render_unified_diff("lib.py", original, edited) == (
        "diff --git a/lib.py b/lib.py\n"
        "--- a/lib.py\n"
        "+++ b/lib.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-alpha();\n"
        "-beta();\n"
        "+ALPHA();\n"
        "+BETA();\n"
        " gamma();\n"
    )


def test_edits_at_byte_zero_and_at_the_end_of_file() -> None:
    original = b"authorize(user);\nreturn 0;\nauthorize(u);"
    edited = apply_edits(original, [ByteEdit(0, 9, b"validate"), ByteEdit(27, 36, b"validate")])

    assert edited == b"validate(user);\nreturn 0;\nvalidate(u);"
    assert render_unified_diff("main.c", original, edited) == (
        "diff --git a/main.c b/main.c\n"
        "--- a/main.c\n"
        "+++ b/main.c\n"
        "@@ -1,3 +1,3 @@\n"
        "-authorize(user);\n"
        "+validate(user);\n"
        " return 0;\n"
        "-authorize(u);"
        "\n\\ No newline at end of file\n"
        "+validate(u);"
        "\n\\ No newline at end of file\n"
    )


def test_a_gap_within_twice_the_context_merges_two_edits_into_one_hunk() -> None:
    original = b"aaa\nBBB\nccc\nddd\nBBB\neee\n"
    edited = apply_edits(original, [ByteEdit(4, 7, b"XXX"), ByteEdit(16, 19, b"XXX")])

    patch = render_unified_diff("gaps.py", original, edited, context_lines=3)
    assert patch is not None
    assert patch.count("@@ -") == 1
    assert patch == (
        "diff --git a/gaps.py b/gaps.py\n"
        "--- a/gaps.py\n"
        "+++ b/gaps.py\n"
        "@@ -1,6 +1,6 @@\n"
        " aaa\n"
        "-BBB\n"
        "+XXX\n"
        " ccc\n"
        " ddd\n"
        "-BBB\n"
        "+XXX\n"
        " eee\n"
    )


def test_a_gap_beyond_twice_the_context_splits_into_two_hunks() -> None:
    original = b"aaa\nBBB\n" + b"m\n" * 7 + b"BBB\nzzz\n"
    edited = apply_edits(original, [ByteEdit(4, 7, b"XXX"), ByteEdit(22, 25, b"XXX")])

    patch = render_unified_diff("gaps.py", original, edited, context_lines=3)
    assert patch is not None
    assert patch.count("@@ -") == 2
    assert patch.splitlines()[3] == "@@ -1,5 +1,5 @@"


def test_zero_context_keeps_only_changed_lines() -> None:
    original = b"a\nb\nc\n"
    edited = apply_edits(original, [ByteEdit(4, 5, b"C")])

    patch = render_unified_diff("tight.py", original, edited, context_lines=0)
    assert patch == (
        "diff --git a/tight.py b/tight.py\n--- a/tight.py\n+++ b/tight.py\n@@ -3 +3 @@\n-c\n+C\n"
    )


def test_crlf_terminators_survive_the_round_trip() -> None:
    original = b"def authorize(user):\r\n    return user\r\n"
    edited = apply_edits(original, [ByteEdit(4, 13, b"validate")])

    assert edited == b"def validate(user):\r\n    return user\r\n"
    assert render_unified_diff("win.py", original, edited) == (
        "diff --git a/win.py b/win.py\n"
        "--- a/win.py\n"
        "+++ b/win.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def authorize(user):\r\n"
        "+def validate(user):\r\n"
        "     return user\r\n"
    )


def test_a_bom_prefixed_file_keeps_the_marker_in_its_first_line() -> None:
    original = _BOM + b"x = authorize\n"
    edited = apply_edits(original, [ByteEdit(4 + len(_BOM), 13 + len(_BOM), b"validate")])

    assert edited == _BOM + b"x = validate\n"
    patch = render_unified_diff("bom.py", original, edited)
    assert patch is not None
    assert patch.startswith("diff --git a/bom.py b/bom.py\n")
    # The marker stays part of the line content, byte for byte.
    assert "-\ufeffx = authorize\n" in patch
    assert "+\ufeffx = validate\n" in patch


def test_a_minus_side_missing_final_newline_is_marked() -> None:
    original = b"answer\nauthorize"
    edited = apply_edits(original, [ByteEdit(7, 16, b"validate")])

    assert edited == b"answer\nvalidate"
    assert render_unified_diff("tail.py", original, edited) == (
        "diff --git a/tail.py b/tail.py\n"
        "--- a/tail.py\n"
        "+++ b/tail.py\n"
        "@@ -1,2 +1,2 @@\n"
        " answer\n"
        "-authorize\n"
        "\\ No newline at end of file\n"
        "+validate\n"
        "\\ No newline at end of file\n"
    )


def test_a_plus_side_missing_final_newline_is_marked() -> None:
    original = b"answer\nauthorize\n"
    # The replacement consumes the file's final newline, so the edited file
    # is the side that ends without a terminator.
    edited = apply_edits(original, [ByteEdit(7, 17, b"validate")])

    assert edited == b"answer\nvalidate"
    assert render_unified_diff("tail.py", original, edited) == (
        "diff --git a/tail.py b/tail.py\n"
        "--- a/tail.py\n"
        "+++ b/tail.py\n"
        "@@ -1,2 +1,2 @@\n"
        " answer\n"
        "-authorize\n"
        "+validate\n"
        "\\ No newline at end of file\n"
    )


def test_both_sides_missing_the_final_newline_are_marked() -> None:
    original = b"authorize"
    edited = apply_edits(original, [ByteEdit(0, 9, b"validate")])

    assert render_unified_diff("tail.py", original, edited) == (
        "diff --git a/tail.py b/tail.py\n"
        "--- a/tail.py\n"
        "+++ b/tail.py\n"
        "@@ -1 +1 @@\n"
        "-authorize\n"
        "\\ No newline at end of file\n"
        "+validate\n"
        "\\ No newline at end of file\n"
    )


def test_multibyte_content_before_the_edit_keeps_offsets_and_line_numbers() -> None:
    original = "# café setup\n".encode() + b"value = caf\xc3\xa9_name\n"
    edit_at = len("# café setup\n".encode()) + len("value = ")
    edited = apply_edits(original, [ByteEdit(edit_at, edit_at + 10, b"cafe_x")])

    assert edited == "# café setup\n".encode() + b"value = cafe_x\n"
    assert render_unified_diff("unicode.py", original, edited) == (
        "diff --git a/unicode.py b/unicode.py\n"
        "--- a/unicode.py\n"
        "+++ b/unicode.py\n"
        "@@ -1,2 +1,2 @@\n"
        " # café setup\n"
        "-value = café_name\n"
        "+value = cafe_x\n"
    )


def test_rendering_is_deterministic() -> None:
    original = b"def authorize(user):\n    return user\n"
    edit = ByteEdit(4, 13, b"validate")
    edited = apply_edits(original, [edit])

    assert apply_edits(original, [edit]) == edited
    assert render_unified_diff("auth.py", original, edited) == render_unified_diff(
        "auth.py", original, edited
    )


def test_overlapping_edits_raise_instead_of_merging() -> None:
    original = b"aaaa\nbbbb\ncccc\n"
    with pytest.raises(ValueError, match="overlaps"):
        apply_edits(original, [ByteEdit(0, 8, b"x"), ByteEdit(5, 9, b"y")])


def test_unsorted_edits_are_applied_in_offset_order() -> None:
    original = b"alpha();\nbeta();\n"
    edited = apply_edits(original, [ByteEdit(9, 13, b"BETA"), ByteEdit(0, 5, b"ALPHA")])

    assert edited == b"ALPHA();\nBETA();\n"


@pytest.mark.parametrize(
    "edit",
    [ByteEdit(-1, 2, b"x"), ByteEdit(1, 99, b"x"), ByteEdit(99, 100, b"x")],
)
def test_edits_outside_the_source_raise(edit: ByteEdit) -> None:
    with pytest.raises(ValueError, match="outside"):
        apply_edits(b"abcdef", [edit])


@pytest.mark.parametrize("context_lines", [-1, 51])
def test_context_lines_outside_the_public_bounds_raise(context_lines: int) -> None:
    with pytest.raises(ValueError, match="between 0 and 50"):
        render_unified_diff("a.py", b"old\n", b"new\n", context_lines=context_lines)


@pytest.mark.parametrize(
    ("path", "quoted"),
    [("tab\tname.py", '"a/tab\\tname.py"'), ("line\nbreak.py", '"a/line\\nbreak.py"')],
)
def test_unusual_paths_are_quoted_for_git(path: str, quoted: str) -> None:
    patch = render_unified_diff(path, b"old\n", b"new\n")

    assert patch is not None
    assert patch.startswith(f"diff --git {quoted} {quoted.replace('a/', 'b/', 1)}\n")


def test_repeated_lines_keep_difflibs_popularity_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    original_matcher = patching.difflib.SequenceMatcher
    settings: list[bool] = []

    def matcher(*args: object, **kwargs: object) -> object:
        settings.append(bool(kwargs.get("autojunk", True)))
        return original_matcher(*args, **kwargs)

    monkeypatch.setattr(patching.difflib, "SequenceMatcher", matcher)
    original = b"same\n" * 1_000 + b"authorize\n"
    edited = b"same\n" * 1_000 + b"validate\n"

    assert render_unified_diff("repeated.py", original, edited) is not None
    assert settings == [True]
