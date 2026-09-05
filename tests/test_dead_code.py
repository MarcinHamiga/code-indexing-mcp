"""Dead-code findings are review candidates, never proof of unreachability."""

from pathlib import Path

import pytest
from test_references import _indexed_service

from code_indexing_mcp.errors import CodeIndexingError, ErrorCode


def test_dead_code_reports_only_exports_without_exact_uses(tmp_path: Path) -> None:
    service, project = _indexed_service(
        tmp_path,
        {
            "lib.ts": (
                "export function unused() { return 1; }\n"
                "export function used() { return 2; }\n"
                "function privateHelper() { return 3; }\n"
            ),
            "main.ts": "import { used } from './lib';\nused();\n",
        },
    )

    report = service.dead_code_report(project)

    assert report.exported_symbols == 2
    assert [item.declaration.symbol for item in report.review] == ["unused"]
    finding = report.review[0]
    assert finding.status == "possibly_dead"
    assert (finding.exact_references, finding.likely_references, finding.unresolved_references) == (
        0,
        0,
        0,
    )
    assert report.snapshot_version > 0
    assert any(item.code == "external_consumers" for item in report.limitations)
    assert report.completeness.state == "complete_with_dynamic_limitations"


@pytest.mark.parametrize(
    ("caller", "resolution"),
    [("answer()\n", "likely"), ("from lib import *\nanswer()\n", "unresolved")],
)
def test_dead_code_preserves_uncertain_references(
    tmp_path: Path, caller: str, resolution: str
) -> None:
    service, project = _indexed_service(
        tmp_path,
        {"lib.py": "def answer():\n    return 42\n", "main.py": caller},
    )

    report = service.dead_code_report(project)

    finding = next(item for item in report.review if item.declaration.symbol == "answer")
    assert finding.status == "possibly_dead"
    assert getattr(finding, f"{resolution}_references") > 0
    assert finding.reason_code == "only_uncertain_references"


def test_dead_code_python_public_and_explicit_exports(tmp_path: Path) -> None:
    service, project = _indexed_service(
        tmp_path,
        {
            "lib.py": (
                "__all__ = ['_api']\n"
                "def _api():\n    pass\n"
                "def _private():\n    pass\n"
                "def public():\n"
                "    def nested():\n        pass\n"
                "class Public:\n    def method(self):\n        pass\n"
            ),
        },
    )

    report = service.dead_code_report(project)

    assert {item.declaration.symbol for item in report.review} == {"_api", "public", "Public"}


def test_dead_code_reexports_and_alias_imports_count_as_uses(tmp_path: Path) -> None:
    service, project = _indexed_service(
        tmp_path,
        {
            "lib.ts": "export function answer() { return 42; }\n",
            "barrel.ts": "export { answer as result } from './lib';\n",
            "main.ts": "import { result as value } from './barrel';\nvalue();\n",
        },
    )

    assert service.dead_code_report(project).review == []


def test_dead_code_reports_coverage_gaps_even_without_exports(tmp_path: Path) -> None:
    service, project = _indexed_service(tmp_path, {"config.yaml": "enabled: true\n"})

    report = service.dead_code_report(project)

    assert report.review == []
    assert report.completeness.state == "incomplete"
    assert any(item.code == "unsupported_language" for item in report.limitations)


def test_dead_code_checks_stale_files_without_previous_references(tmp_path: Path) -> None:
    service, project = _indexed_service(
        tmp_path, {"lib.py": "def answer():\n    return 42\n", "main.py": "value = 1\n"}
    )
    (tmp_path / "repo" / "main.py").write_text("from lib import answer\nanswer()\n")

    report = service.dead_code_report(project)

    assert report.completeness.state == "incomplete"
    assert any(item.code == "stale_file" for item in report.limitations)


def test_dead_code_rejects_missing_reference_index(tmp_path: Path) -> None:
    service, project = _indexed_service(tmp_path, {"lib.py": "def answer():\n    pass\n"})
    service.store.has_reference_table = lambda *args, **kwargs: False  # type: ignore[method-assign]

    with pytest.raises(CodeIndexingError) as error:
        service.dead_code_report(project)

    assert error.value.code == ErrorCode.REFERENCE_INDEX_UNAVAILABLE


@pytest.mark.parametrize(
    ("path", "source", "expected"),
    [
        ("main.go", "package main\nfunc Public() {}\nfunc private() {}\n", {"Public"}),
        ("lib.rs", "pub fn public() {}\nfn private() {}\n", {"public"}),
        ("Public.java", "public class Public {}\nclass Private {}\n", {"Public"}),
        ("Public.cs", "namespace Example;\npublic class Public {}\n", {"Public"}),
        ("lib.js", "function api() {}\nexport { api as publicApi };\n", {"api"}),
        ("default.js", "function api() {}\nexport default api;\n", {"api"}),
        ("named.js", "export default function api() {}\n", {"api"}),
        ("lib.tsx", "export function View() { return <div />; }\n", {"View"}),
    ],
)
def test_dead_code_uses_structural_export_rules(
    tmp_path: Path, path: str, source: str, expected: set[str]
) -> None:
    service, project = _indexed_service(tmp_path, {path: source})

    report = service.dead_code_report(project)

    assert {item.declaration.symbol for item in report.review} == expected


def test_dead_code_counts_all_references_beyond_a_reference_page(tmp_path: Path) -> None:
    service, project = _indexed_service(
        tmp_path,
        {
            "lib.py": "def answer():\n    pass\n",
            "a.py": "answer()\n" * 110,
            "z.py": "from lib import answer\nanswer()\n",
        },
    )

    assert service.dead_code_report(project).review == []


def test_dead_code_inline_export_does_not_export_a_same_named_declaration(
    tmp_path: Path,
) -> None:
    service, project = _indexed_service(
        tmp_path,
        {"lib.ts": "namespace N { export function api() {} }\nfunction api() {}\n"},
    )

    report = service.dead_code_report(project)

    assert report.exported_symbols == 1
    assert [item.declaration.start_line for item in report.review] == [1]
