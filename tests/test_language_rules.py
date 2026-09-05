"""Regression tests for the per-language rules table."""

from pathlib import PurePosixPath

from code_indexing_mcp.extractor import STRUCTURAL_LANGUAGES
from code_indexing_mcp.language_rules import _DEFAULT, LANGUAGE_RULES


def test_every_structural_language_has_a_row() -> None:
    assert set(LANGUAGE_RULES.keys()) == set(STRUCTURAL_LANGUAGES)


def test_structural_languages_equals_table_keys() -> None:
    assert frozenset(LANGUAGE_RULES) == STRUCTURAL_LANGUAGES


def test_no_row_reserved_words_is_empty() -> None:
    for language, rules in LANGUAGE_RULES.items():
        assert len(rules.reserved_words) > 0, f"{language} reserved_words must not be empty"


def test_default_rules_are_empty_and_reject_all() -> None:
    assert _DEFAULT.import_owner_parents == frozenset()
    assert _DEFAULT.method_name_field_excluded is False
    assert _DEFAULT.name_and_type_parents == frozenset()
    assert _DEFAULT.name_and_field_parents == frozenset()
    assert _DEFAULT.name_only_parents == frozenset()
    assert _DEFAULT.type_only_parents == frozenset()
    assert _DEFAULT.parameters_parents == frozenset()
    assert _DEFAULT.left_and_type_parents == frozenset()
    assert _DEFAULT.function_and_type_parents == frozenset()
    assert _DEFAULT.pair_parents == frozenset()
    assert _DEFAULT.declarator_parents == frozenset()
    assert _DEFAULT.handler_owned_type_parents == frozenset()
    assert _DEFAULT.keyword_only_marker is None
    assert _DEFAULT.reserved_words == frozenset()
    assert _DEFAULT.identifier_valid("valid_name") is False
    assert _DEFAULT.bound_receivers == frozenset()
    assert (
        _DEFAULT.import_candidates(PurePosixPath("src/mod.py"), "other", frozenset(), None) == set()
    )


def test_identifier_valid_rules() -> None:
    # ECMAScript allows `$` in identifiers, rejects reserved words
    js_rules = LANGUAGE_RULES["javascript"]
    assert js_rules.identifier_valid("$state") is True
    assert js_rules.identifier_valid("state$") is True
    assert js_rules.identifier_valid("validName") is True
    assert js_rules.identifier_valid("await") is False
    assert js_rules.identifier_valid("class") is False
    assert js_rules.identifier_valid("123bad") is False

    # C# allows `@`-prefixed reserved words (verbatim identifiers)
    cs_rules = LANGUAGE_RULES["csharp"]
    assert cs_rules.identifier_valid("validName") is True
    assert cs_rules.identifier_valid("@class") is True
    assert cs_rules.identifier_valid("@event") is True
    assert cs_rules.identifier_valid("class") is False
    assert cs_rules.identifier_valid("event") is False
    assert cs_rules.identifier_valid("@123") is False

    # Python rejects keywords
    py_rules = LANGUAGE_RULES["python"]
    assert py_rules.identifier_valid("valid_name") is True
    assert py_rules.identifier_valid("def") is False
    assert py_rules.identifier_valid("class") is False

    # Go rejects Go keywords
    go_rules = LANGUAGE_RULES["go"]
    assert go_rules.identifier_valid("ValidName") is True
    assert go_rules.identifier_valid("package") is False
    assert go_rules.identifier_valid("func") is False

    # Rust rejects Rust keywords
    rs_rules = LANGUAGE_RULES["rust"]
    assert rs_rules.identifier_valid("valid_name") is True
    assert rs_rules.identifier_valid("fn") is False
    assert rs_rules.identifier_valid("impl") is False

    # Java rejects Java keywords
    java_rules = LANGUAGE_RULES["java"]
    assert java_rules.identifier_valid("validName") is True
    assert java_rules.identifier_valid("assert") is False
    assert java_rules.identifier_valid("package") is False

    # C rejects C keywords
    c_rules = LANGUAGE_RULES["c"]
    assert c_rules.identifier_valid("add") is True
    assert c_rules.identifier_valid("struct") is False
    assert c_rules.identifier_valid("typedef") is False
    assert c_rules.identifier_valid("123bad") is False

    # C++ rejects both C and C++-only keywords
    cpp_rules = LANGUAGE_RULES["cpp"]
    assert cpp_rules.identifier_valid("render") is True
    assert cpp_rules.identifier_valid("struct") is False
    assert cpp_rules.identifier_valid("class") is False
    assert cpp_rules.identifier_valid("namespace") is False

    # Lua rejects Lua keywords
    lua_rules = LANGUAGE_RULES["lua"]
    assert lua_rules.identifier_valid("helper") is True
    assert lua_rules.identifier_valid("function") is False
    assert lua_rules.identifier_valid("local") is False

    # SQL rejects keywords case-insensitively
    sql_rules = LANGUAGE_RULES["sql"]
    assert sql_rules.identifier_valid("username") is True
    assert sql_rules.identifier_valid("select") is False
    assert sql_rules.identifier_valid("SELECT") is False
    assert sql_rules.identifier_valid("from") is False

    # GDScript rejects Godot keywords
    gdscript_rules = LANGUAGE_RULES["gdscript"]
    assert gdscript_rules.identifier_valid("take_damage") is True
    assert gdscript_rules.identifier_valid("func") is False
    assert gdscript_rules.identifier_valid("self") is False

    # GDShader rejects shader keywords
    gdshader_rules = LANGUAGE_RULES["gdshader"]
    assert gdshader_rules.identifier_valid("brightness") is True
    assert gdshader_rules.identifier_valid("uniform") is False
    assert gdshader_rules.identifier_valid("shader_type") is False

    # Terraform rejects only the literal names
    terraform_rules = LANGUAGE_RULES["terraform"]
    assert terraform_rules.identifier_valid("image_id") is True
    assert terraform_rules.identifier_valid("true") is False
    assert terraform_rules.identifier_valid("null") is False


def test_import_candidates_for_new_languages() -> None:
    c = LANGUAGE_RULES["c"].import_candidates
    known = frozenset({"src/util.h", "src/main.c", "include/util.h"})
    assert c(PurePosixPath("src/main.c"), "util.h", known, None) == {
        PurePosixPath("src/util.h"),
        PurePosixPath("include/util.h"),
    }
    assert c(PurePosixPath("src/main.c"), "stdio.h", known, None) == set()

    cpp = LANGUAGE_RULES["cpp"].import_candidates
    assert cpp(PurePosixPath("app/main.cpp"), "widget.h", known, None) == set()

    lua = LANGUAGE_RULES["lua"].import_candidates
    assert lua(PurePosixPath("app/main.lua"), "a.b", known, None) == {
        PurePosixPath("app/a/b.lua"),
        PurePosixPath("app/a/b/init.lua"),
        PurePosixPath("a/b.lua"),
        PurePosixPath("a/b/init.lua"),
    }
    assert lua(PurePosixPath("app/main.lua"), "./side.lua", known, None) == {
        PurePosixPath("app/side.lua"),
        PurePosixPath("side.lua"),
    }

    gdscript = LANGUAGE_RULES["gdscript"].import_candidates
    assert gdscript(PurePosixPath("player.gd"), "res://ui/hud.gd", known, None) == {
        PurePosixPath("ui/hud.gd")
    }
    assert gdscript(PurePosixPath("actors/player.gd"), "base.gd", known, None) == {
        PurePosixPath("actors/base.gd")
    }

    terraform = LANGUAGE_RULES["terraform"].import_candidates
    assert terraform(PurePosixPath("main.tf"), "./vpc", known, None) == {
        PurePosixPath("vpc/main.tf")
    }
    assert terraform(PurePosixPath("env/main.tf"), "../vpc", known, None) == {
        PurePosixPath("vpc/main.tf")
    }
    assert (
        terraform(PurePosixPath("main.tf"), "terraform-aws-modules/vpc/aws", known, None) == set()
    )

    sql = LANGUAGE_RULES["sql"].import_candidates
    assert sql(PurePosixPath("schema.sql"), "other", known, None) == set()


def test_bound_receivers_per_language() -> None:
    assert LANGUAGE_RULES["python"].bound_receivers == frozenset({"self", "cls"})
    for language, rules in LANGUAGE_RULES.items():
        if language != "python":
            assert rules.bound_receivers == frozenset(), (
                f"{language} should have no bound_receivers"
            )
