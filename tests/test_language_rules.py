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


def test_bound_receivers_per_language() -> None:
    assert LANGUAGE_RULES["python"].bound_receivers == frozenset({"self", "cls"})
    for language, rules in LANGUAGE_RULES.items():
        if language != "python":
            assert rules.bound_receivers == frozenset(), (
                f"{language} should have no bound_receivers"
            )
