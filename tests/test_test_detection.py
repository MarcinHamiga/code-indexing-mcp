"""Tests for test/benchmark detection and search scoping in C#, Java, Python, and Rust."""

from __future__ import annotations

from pathlib import Path

import pytest
from support import DeterministicEmbedder

from code_indexing_mcp.application import Application, RuntimePaths
from code_indexing_mcp.errors import CodeIndexingError, ErrorCode
from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.models import DeclarationSelector, RenameOperation
from code_indexing_mcp.test_detection import is_test_file_path


@pytest.mark.parametrize(
    ("language", "rel_path", "expected"),
    [
        # Python
        ("python", "tests/helpers/factory.py", True),
        ("python", "test/unit/sample.py", True),
        ("python", "src/pkg/conftest.py", True),
        ("python", "src/pkg/test_orders.py", True),
        ("python", "src/pkg/orders_test.py", True),
        ("python", "benchmarks/bench_parser.py", True),
        ("python", "src/pkg/bench_hash.py", True),
        ("python", "src/pkg/testing/runner.py", False),
        ("python", "src/pkg/contest.py", False),
        ("python", "src/pkg/latest.py", False),
        # Rust
        ("rust", "tests/integration.rs", True),
        ("rust", "tests/common/mod.rs", True),
        ("rust", "benches/throughput.rs", True),
        ("rust", "crates/core/src/parser/tests.rs", True),
        ("rust", "crates/core/src/parser_test.rs", True),
        ("rust", "crates/core/src/parser_bench.rs", True),
        ("rust", "crates/core/src/lib.rs", False),
        ("rust", "crates/core/src/contest.rs", False),
        # Java
        ("java", "src/test/java/com/acme/UserBuilder.java", True),
        ("java", "javatests/com/acme/UserBuilder.java", True),
        ("java", "src/integrationTest/java/com/acme/ApiSpec.java", True),
        ("java", "src/testFixtures/java/com/acme/Fixtures.java", True),
        ("java", "src/jmh/java/com/acme/Throughput.java", True),
        ("java", "src/main/java/com/acme/OrderTest.java", True),
        ("java", "src/main/java/com/acme/OrderTests.java", True),
        ("java", "src/main/java/com/acme/OrderIT.java", True),
        ("java", "src/main/java/com/acme/TestOrder.java", True),
        ("java", "src/main/java/com/acme/OrderBenchmark.java", True),
        ("java", "src/main/java/com/acme/Contest.java", False),
        ("java", "src/main/java/com/acme/Latest.java", False),
        ("java", "src/main/java/com/acme/Testable.java", False),
        ("java", "src/main/java/com/acme/Testing.java", False),
        ("java", "src/main/java/com/acme/BIT.java", False),
        # C#
        ("csharp", "tests/Acme.Core/FakeClock.cs", True),
        ("csharp", "src/Acme.Orders.Tests/UserBuilder.cs", True),
        ("csharp", "src/Acme.Orders.UnitTests/Helper.cs", True),
        ("csharp", "src/Acme.Orders.IntegrationTests/Fixture.cs", True),
        ("csharp", "src/Acme.Orders.Specs/StepDefs.cs", True),
        ("csharp", "src/Acme.Benchmarks/HashBench.cs", True),
        ("csharp", "src/AcmeOrdersTests/Helper.cs", True),
        ("csharp", "src/Acme/OrderTests.cs", True),
        ("csharp", "src/Acme/OrderBenchmark.cs", True),
        ("csharp", "src/Acme/TestOrder.cs", True),
        ("csharp", "src/Acme/Contest.cs", False),
        ("csharp", "src/Acme/Latest.cs", False),
        ("csharp", "src/Acme/Testable.cs", False),
        ("csharp", "src/Acme/TestingService.cs", False),
        ("csharp", "src/Acme.Domain/Specs/ActiveOrder.cs", False),
        ("csharp", "src/Acme.Domain/OrderSpecs/ByDate.cs", False),
        ("csharp", "src/Acme.Domain/Spec.cs", False),
        # Non-target languages stay False
        ("typescript", "tests/auth.test.ts", False),
        ("go", "pkg/auth_test.go", False),
    ],
)
def test_is_test_file_path_conventions_and_false_positive_guards(
    language: str, rel_path: str, expected: bool
) -> None:
    assert is_test_file_path(language, Path(rel_path)) is expected


def test_rust_inline_cfg_test_and_attributes_separate_prod_from_tests() -> None:
    extractor = TreeSitterExtractor()
    source = b"""use std::fmt;

macro_rules! prod_macro {
    () => { 1 };
}

#[cfg_attr(test, derive(Clone))]
pub struct ProductionConfig {
    pub retries: usize,
}

#[cfg(not(test))]
pub fn clock_now() -> u64 {
    100
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeHarness;

    fn make_config() -> ProductionConfig {
        ProductionConfig { retries: 3 }
    }

    #[tokio::test]
    async fn retries_on_timeout() {
        let cfg = make_config();
        assert_eq!(cfg.retries, 3);
    }

    #[bench]
    fn bench_config() {}
}

#[cfg(all(unix, test))]
fn unix_only_test_helper() -> u64 {
    0
}
"""
    result = extractor.extract(Path("src/lib.rs"), "rust", source)
    by_symbol = {chunk.qualified_symbol: chunk for chunk in result.chunks if chunk.qualified_symbol}

    assert by_symbol["ProductionConfig"].is_test is False
    assert by_symbol["clock_now"].is_test is False
    assert by_symbol["FakeHarness"].is_test is True
    assert by_symbol["make_config"].is_test is True
    assert by_symbol["retries_on_timeout"].is_test is True
    assert by_symbol["bench_config"].is_test is True
    assert by_symbol["unix_only_test_helper"].is_test is True

    module_chunks = [chunk for chunk in result.chunks if chunk.kind == "module"]
    prod_modules = [chunk for chunk in module_chunks if not chunk.is_test]
    test_modules = [chunk for chunk in module_chunks if chunk.is_test]
    assert any("prod_macro" in chunk.content for chunk in prod_modules)
    assert all("use super::*" not in chunk.content for chunk in prod_modules)
    assert any("use super::*" in chunk.content for chunk in test_modules)


def test_rust_criterion_benchmark_and_crate_inner_cfg_test() -> None:
    extractor = TreeSitterExtractor()
    criterion_src = b"""use criterion::{Criterion, criterion_group, criterion_main};

fn parse_throughput(c: &mut Criterion) {
    c.bench_function("parse", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, parse_throughput);
criterion_main!(benches);
"""
    bench_result = extractor.extract(Path("src/perf_suite.rs"), "rust", criterion_src)
    assert bench_result.chunks
    assert all(chunk.is_test for chunk in bench_result.chunks)

    inner_cfg_src = b"""#![cfg(test)]
pub fn shared_fixture() -> usize { 42 }
"""
    inner_result = extractor.extract(Path("src/support.rs"), "rust", inner_cfg_src)
    assert inner_result.chunks
    assert all(chunk.is_test for chunk in inner_result.chunks)


def test_python_ast_detects_pytest_unittest_fixtures_and_benchmarks() -> None:
    extractor = TreeSitterExtractor()
    mixed_source = b"""import unittest
import pytest


class PatternMatcher:
    def test(self, text: str) -> bool:
        return bool(text)


def compute_invoice(amount: int) -> int:
    return amount * 2


@pytest.fixture
def sample_invoice() -> int:
    return compute_invoice(5)


def test_compute_invoice(sample_invoice: int) -> None:
    assert sample_invoice == 10


def bench_compute_invoice() -> None:
    compute_invoice(100)


class LegacyInvoiceChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.base = 7

    def helper_value(self) -> int:
        return self.base
"""
    result = extractor.extract(Path("src/billing.py"), "python", mixed_source)
    by_symbol = {chunk.qualified_symbol: chunk for chunk in result.chunks if chunk.qualified_symbol}

    assert by_symbol["PatternMatcher"].is_test is False
    assert by_symbol["PatternMatcher.test"].is_test is False
    assert by_symbol["compute_invoice"].is_test is False

    assert by_symbol["sample_invoice"].is_test is True
    assert by_symbol["test_compute_invoice"].is_test is True
    assert by_symbol["bench_compute_invoice"].is_test is True
    assert by_symbol["LegacyInvoiceChecks"].is_test is True
    assert by_symbol["LegacyInvoiceChecks.setUp"].is_test is True
    assert by_symbol["LegacyInvoiceChecks.helper_value"].is_test is True


def test_java_ast_promotes_junit_and_jmh_classes_including_helpers() -> None:
    extractor = TreeSitterExtractor()
    java_source = b"""package com.acme.orders;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.annotations.Benchmark;

class OrderVerification {
    private String buildFixture() {
        return "order-1";
    }

    @BeforeEach
    void prepare() {}

    @Test
    void rejectsExpiredOrder() {
        buildFixture();
    }
}

class HashingPerf {
    @Benchmark
    public int measureHash() {
        return 42;
    }
}
"""
    result = extractor.extract(
        Path("src/main/java/com/acme/OrderVerification.java"), "java", java_source
    )
    assert result.chunks
    assert all(chunk.is_test for chunk in result.chunks)


def test_csharp_ast_promotes_xunit_nunit_mstest_and_benchmarkdotnet() -> None:
    extractor = TreeSitterExtractor()
    csharp_source = b"""using Xunit;
using BenchmarkDotNet.Attributes;

namespace Acme.Billing;

public class InvoiceChecks
{
    private static int CreateSample() => 42;

    [FactAttribute]
    public void CalculatesTotal()
    {
        var x = CreateSample();
    }
}

public class InvoiceThroughput
{
    [Benchmark]
    public int Run() => 1;
}
"""
    result = extractor.extract(Path("src/Acme.Billing/InvoiceChecks.cs"), "csharp", csharp_source)
    assert result.chunks
    assert all(chunk.is_test for chunk in result.chunks)


def _build_app(tmp_path: Path) -> tuple[Application, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")

    # Python prod + test file
    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text(
        "def verify_session_token(token: str) -> bool:\n    return token.startswith('valid_')\n",
        encoding="utf-8",
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_auth.py").write_text(
        "from src.auth import verify_session_token\n\n"
        "def make_fake_token() -> str:\n"
        "    return 'valid_ session token test'\n\n"
        "def test_verify_session_token() -> None:\n"
        "    assert verify_session_token(make_fake_token())\n",
        encoding="utf-8",
    )

    # Rust mixed file with inline #[cfg(test)]
    (repo / "src" / "lib.rs").write_text(
        "pub fn dispatch_order(order_id: u64) -> bool {\n"
        "    order_id > 0\n"
        "}\n\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n\n"
        "    fn order_fixture() -> u64 { 7 }\n\n"
        "    #[test]\n"
        "    fn test_dispatch_order() {\n"
        "        let ok = dispatch_order(order_fixture());\n"
        "        assert!(ok);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    # C# production + test project
    (repo / "src" / "Billing.cs").write_text(
        "namespace Demo;\n"
        "public class LedgerService {\n"
        "    public int ComputeBalance(int cents) => cents;\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "Demo.Tests").mkdir()
    (repo / "Demo.Tests" / "LedgerBuilder.cs").write_text(
        "using Xunit;\n"
        "namespace Demo.Tests;\n"
        "public class LedgerBuilder {\n"
        "    public int ComputeBalance(int cents) => cents;\n"
        "}\n"
        "public class LedgerSpecs {\n"
        "    [Fact]\n"
        "    public void ChecksLedgerBalance() {}\n"
        "}\n",
        encoding="utf-8",
    )

    # Java production + Maven test source set
    java_main = repo / "src" / "main" / "java" / "demo"
    java_main.mkdir(parents=True)
    (java_main / "Inventory.java").write_text(
        "package demo;\n"
        "public class Inventory {\n"
        "    public int reserveStock(int count) { return count; }\n"
        "}\n",
        encoding="utf-8",
    )
    java_test = repo / "src" / "test" / "java" / "demo"
    java_test.mkdir(parents=True)
    (java_test / "InventoryTest.java").write_text(
        "package demo;\n"
        "import org.junit.jupiter.api.Test;\n"
        "public class InventoryTest {\n"
        "    int reserveStock(int count) { return count; }\n"
        "    @Test void reservesStock() {}\n"
        "}\n",
        encoding="utf-8",
    )

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=DeterministicEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)
    return app, repo


def test_search_code_example_and_find_symbol_respect_scope_while_refactor_stays_unfiltered(
    tmp_path: Path,
) -> None:
    app, _ = _build_app(tmp_path)

    # Default scope="code" excludes Python tests/, Rust #[cfg(test)], C# *.Tests/, Java src/test/
    code_hits = app.search_code("verify_session_token", limit=20).hits
    assert code_hits[0].path == "src/auth.py"
    assert all(not hit.is_test for hit in code_hits)
    assert {hit.path for hit in code_hits} == {
        "src/auth.py",
        "src/lib.rs",
        "src/Billing.cs",
        "src/main/java/demo/Inventory.java",
    }
    assert all("order_fixture" not in hit.snippet for hit in code_hits)

    test_hits = app.search_code("verify_session_token", scope="tests", limit=20).hits
    assert test_hits[0].path == "tests/test_auth.py"
    assert all(hit.is_test for hit in test_hits)
    assert {hit.path for hit in test_hits} == {
        "tests/test_auth.py",
        "src/lib.rs",
        "Demo.Tests/LedgerBuilder.cs",
        "src/test/java/demo/InventoryTest.java",
    }

    all_hits = app.search_code("verify_session_token", scope="all", limit=50).hits
    assert any(hit.is_test for hit in all_hits)
    assert any(not hit.is_test for hit in all_hits)
    assert "tests/test_auth.py" in {hit.path for hit in all_hits}
    assert "src/auth.py" in {hit.path for hit in all_hits}

    # Rust inline #[cfg(test)] helpers are excluded from code symbol lookup, included in tests/all
    assert app.find_symbol("dispatch_order").hits
    assert app.find_symbol("order_fixture").hits == []
    rust_test_sym = app.find_symbol("order_fixture", scope="tests").hits
    assert len(rust_test_sym) == 1
    assert rust_test_sym[0].is_test is True

    # C# & Java duplicate method names in test helpers vs production classes
    cs_code = app.find_symbol("ComputeBalance").hits
    assert [hit.path for hit in cs_code] == ["src/Billing.cs"]
    cs_tests = app.find_symbol("ComputeBalance", scope="tests").hits
    assert [hit.path for hit in cs_tests] == ["Demo.Tests/LedgerBuilder.cs"]
    cs_all = app.find_symbol("ComputeBalance", scope="all").hits
    assert {hit.path for hit in cs_all} == {"src/Billing.cs", "Demo.Tests/LedgerBuilder.cs"}

    java_code = app.find_symbol("reserveStock").hits
    assert [hit.path for hit in java_code] == ["src/main/java/demo/Inventory.java"]
    java_tests = app.find_symbol("reserveStock", scope="tests").hits
    assert [hit.path for hit in java_tests] == ["src/test/java/demo/InventoryTest.java"]

    # search_by_example obeys scope too
    snippet = (
        "def verify_session_token(token: str) -> bool:\n    return token.startswith('valid_')\n"
    )
    ex_code = app.search_by_example(snippet, limit=20).hits
    assert all(not hit.is_test for hit in ex_code)
    ex_tests = app.search_by_example(snippet, scope="tests", limit=20).hits
    assert ex_tests
    assert all(hit.is_test for hit in ex_tests)

    # Invalid scope raises INVALID_FILTER
    with pytest.raises(CodeIndexingError) as exc_info:
        app.search_code("token", scope="invalid")  # type: ignore[arg-type]
    assert exc_info.value.code is ErrorCode.INVALID_FILTER

    # analyze_refactor stays unfiltered so renaming production code still updates test callers!
    prod_chunk_id = app.find_symbol("dispatch_order").hits[0].chunk_id
    analysis = app.analyze_refactor(
        DeclarationSelector(chunk_id=prod_chunk_id),
        RenameOperation(new_name="route_order"),
    )
    edited_lines = {finding.start_line for finding in analysis.findings}
    assert 1 in edited_lines
    assert 13 in edited_lines

    py_chunk_id = app.find_symbol("verify_session_token").hits[0].chunk_id
    py_analysis = app.analyze_refactor(
        DeclarationSelector(chunk_id=py_chunk_id),
        RenameOperation(new_name="check_session_token"),
    )
    assert "tests/test_auth.py" in {finding.path for finding in py_analysis.must_change}


@pytest.mark.asyncio
async def test_mcp_search_across_projects_and_find_symbol_scope_filtering(tmp_path: Path) -> None:
    from mcp import types
    from mcp.shared.memory import create_connected_server_and_client_session

    from code_indexing_mcp.server import create_server

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=DeterministicEmbedder(),
        cwd=tmp_path,
    )
    alpha_root = tmp_path / "alpha"
    beta_root = tmp_path / "beta"
    (alpha_root / "src").mkdir(parents=True)
    (alpha_root / "tests").mkdir(parents=True)
    (beta_root / "src").mkdir(parents=True)
    (beta_root / "benches").mkdir(parents=True)

    (alpha_root / "src" / "payment.py").write_text(
        "def settle_ledger_entry(amount: int) -> int:\n    return amount\n",
        encoding="utf-8",
    )
    (alpha_root / "tests" / "test_payment.py").write_text(
        "def test_settle_ledger_entry() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    (beta_root / "src" / "lib.rs").write_text(
        "pub fn settle_ledger_entry(amount: i64) -> i64 { amount }\n",
        encoding="utf-8",
    )
    (beta_root / "benches" / "ledger_bench.rs").write_text(
        "pub fn bench_settle_ledger_entry() {}\n",
        encoding="utf-8",
    )

    alpha = app.init_project(alpha_root, "alpha")
    beta = app.init_project(beta_root, "beta")
    app.index_project(alpha.id)
    app.index_project(beta.id)

    server = create_server(app, auto_index=False)

    async def list_roots(_: types.ListRootsRequest) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[])

    async with create_connected_server_and_client_session(
        server, list_roots_callback=list_roots
    ) as client:
        default_res = await client.call_tool(
            "search_across_projects",
            {"projects": [alpha.id, beta.id], "query": "settle_ledger_entry", "limit": 10},
        )
        assert not default_res.isError
        assert default_res.structuredContent is not None
        default_paths = {hit["path"] for hit in default_res.structuredContent["hits"]}
        assert default_paths == {"src/payment.py", "src/lib.rs"}

        tests_res = await client.call_tool(
            "search_across_projects",
            {
                "projects": [alpha.id, beta.id],
                "query": "settle_ledger_entry",
                "scope": "tests",
                "limit": 10,
            },
        )
        assert not tests_res.isError
        assert tests_res.structuredContent is not None
        test_paths = {hit["path"] for hit in tests_res.structuredContent["hits"]}
        assert test_paths == {"tests/test_payment.py", "benches/ledger_bench.rs"}

        sym_code = await client.call_tool(
            "find_symbol",
            {"name": "test_settle_ledger_entry", "project": alpha.id},
        )
        assert sym_code.structuredContent is not None
        assert sym_code.structuredContent["hits"] == []

        sym_tests = await client.call_tool(
            "find_symbol",
            {"name": "test_settle_ledger_entry", "project": alpha.id, "scope": "tests"},
        )
        assert sym_tests.structuredContent is not None
        assert len(sym_tests.structuredContent["hits"]) == 1


@pytest.mark.asyncio
async def test_tui_scope_selector_forwards_scope_and_renders_test_badge() -> None:
    from typing import cast

    from test_tui import _wait_for_workers
    from test_tui_service import FakeApplication, _sample_hit, _sample_project
    from textual.widgets import Input, OptionList, Select, Static

    from code_indexing_mcp.application import ApplicationLike
    from code_indexing_mcp.models import SearchResponse
    from code_indexing_mcp.tui.app import CodeIndexingApp
    from code_indexing_mcp.tui.service import TuiService

    class ScopedFakeApp(FakeApplication):
        def search_code(self, query: str, **kwargs: object) -> SearchResponse:
            scope = str(kwargs.get("scope", "code"))
            self.search_calls.append({"query": query, "scope": scope})
            if scope == "code":
                return SearchResponse(query=query, hits=[])
            hit = _sample_hit(path="tests/test_main.py", symbol="test_main").model_copy(
                update={"is_test": True}
            )
            return SearchResponse(query=query, hits=[hit])

    fake = ScopedFakeApp(projects=[_sample_project(project_id="proj-1", name="repo-alpha")])
    app = CodeIndexingApp(
        service=TuiService(cast(ApplicationLike, fake), cwd=Path("/workspace/test-repo"))
    )
    async with app.run_test() as pilot:
        await _wait_for_workers(app, pilot)
        await pilot.pause()

        scope_select = app.query_one("#scope-select", Select)
        assert scope_select.value == "code"

        query_input = app.query_one("#query-input", Input)
        query_input.value = "main"
        app.action_submit_query()
        await _wait_for_workers(app, pilot)
        await pilot.pause()

        assert fake.search_calls[-1]["scope"] == "code"
        detail_text = str(app.query_one("#detail-content", Static).render())
        assert "Tests only" in detail_text

        # Changing #scope-select with an active query automatically re-runs the search
        scope_select.value = "tests"
        await _wait_for_workers(app, pilot)
        await pilot.pause()

        assert fake.search_calls[-1]["scope"] == "tests"
        results_list = app.query_one("#results-list", OptionList)
        assert results_list.option_count == 1
        prompt = str(results_list.get_option_at_index(0).prompt)
        assert "[test]" in prompt


def test_production_files_and_domain_types_with_test_prefixes_stay_code() -> None:
    extractor = TreeSitterExtractor()

    # Production module under src/ starting with `test_` and defining `TestClassification`
    own_module = Path("src/code_indexing_mcp/test_detection.py")
    own_res = extractor.extract(own_module, "python", own_module.read_bytes())
    assert own_res.chunks
    assert all(not chunk.is_test for chunk in own_res.chunks)

    # Co-located Python test file under src/ WITH actual tests still promotes helpers whole-file
    colocated_py = b"""def make_order() -> dict[str, int]:
    return {"id": 1}

def test_creates_order() -> None:
    assert make_order()["id"] == 1
"""
    colocated_res = extractor.extract(Path("src/pkg/test_orders.py"), "python", colocated_py)
    assert colocated_res.chunks
    assert all(chunk.is_test for chunk in colocated_res.chunks)

    # Production Test*/Benchmark* dataclasses, __init__ classes, and helpers without test methods
    py_models = b"""from dataclasses import dataclass

@dataclass
class TestConfig:
    timeout: int = 30

@dataclass
class BenchmarkResult:
    ops_per_sec: float = 0.0

class TestClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def test_endpoint(self) -> bool:
        return True

class TestOrderFlow:
    def make_order(self) -> int:
        return 1

    def test_rejects_negative(self) -> None:
        assert self.make_order() > 0
"""
    py_res = extractor.extract(Path("src/models.py"), "python", py_models)
    by_sym = {c.qualified_symbol: c.is_test for c in py_res.chunks if c.qualified_symbol}
    assert by_sym["TestConfig"] is False
    assert by_sym["BenchmarkResult"] is False
    assert by_sym["TestClient"] is False
    assert by_sym["TestClient.test_endpoint"] is False
    assert by_sym["TestOrderFlow"] is True
    assert by_sym["TestOrderFlow.make_order"] is True
    assert by_sym["TestOrderFlow.test_rejects_negative"] is True

    # Java and C# Test*/Benchmark*/Specs domain classes without test attributes stay False
    java_runner = b"""package com.acme;
public class TestRunner {
    public void execute() {}
}
"""
    java_res = extractor.extract(
        Path("src/main/java/com/acme/TestRunner.java"), "java", java_runner
    )
    assert all(not chunk.is_test for chunk in java_res.chunks)

    cs_spec = b"""namespace Acme.Domain.Specs;
public class ActiveOrderSpecs {
    public bool IsSatisfiedBy(int total) => total > 0;
}
"""
    cs_res = extractor.extract(Path("src/Acme.Domain/Specs/ActiveOrderSpecs.cs"), "csharp", cs_spec)
    assert all(not chunk.is_test for chunk in cs_res.chunks)


def test_python_non_test_class_methods_and_module_boilerplate() -> None:
    extractor = TreeSitterExtractor()

    # Non-test class with `def test_connection(self)` and nested `def test_ping()` stay code
    prod_py = b"""class DatabaseClient:
    def connect(self) -> None:
        pass

    def query(self, sql: str) -> list[str]:
        return [sql]

    def test_connection(self) -> bool:
        return True

def create_client() -> DatabaseClient:
    def test_ping() -> bool:
        return True
    test_ping()
    return DatabaseClient()
"""
    prod_res = extractor.extract(Path("src/db/client.py"), "python", prod_py)
    assert prod_res.chunks
    assert all(not chunk.is_test for chunk in prod_res.chunks)

    # Test file with TYPE_CHECKING, pytestmark, unittest.main(), and direct @mark/@skipIf
    suite_py = b"""from __future__ import annotations
from typing import TYPE_CHECKING
import unittest
from unittest import expectedFailure, skipIf
from pytest import mark

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = mark.integration

@mark.parametrize("qty", [1, 2])
def check_order(qty: int) -> None:
    assert qty > 0

class OrderSuite(unittest.TestCase):
    def build_order(self) -> dict[str, int]:
        return {"qty": 1}

    @skipIf(False, "enabled")
    @expectedFailure
    def test_qty(self) -> None:
        self.assertEqual(self.build_order()["qty"], 1)

if __name__ == "__main__":
    unittest.main()
"""
    suite_res = extractor.extract(Path("src/order_suite.py"), "python", suite_py)
    assert suite_res.chunks
    assert all(chunk.is_test for chunk in suite_res.chunks)


def test_java_and_csharp_nested_types_aliases_enums_jmh_state_and_junit3() -> None:
    extractor = TreeSitterExtractor()

    # Java prod class with nested test class keeps outer class/methods as production code
    java_nested = b"""package com.acme;
import org.junit.jupiter.api.Test;

public class OrderService {
    public int placeOrder(int qty) { return qty * 10; }
    public void cancelOrder(String id) {}

    static class SanityTest {
        @Test void ok() {}
    }
}
"""
    jn_res = extractor.extract(
        Path("src/main/java/com/acme/OrderService.java"), "java", java_nested
    )
    jn_syms = {c.qualified_symbol: c.is_test for c in jn_res.chunks if c.qualified_symbol}
    assert jn_syms["OrderService"] is False
    assert jn_syms["OrderService.placeOrder"] is False
    assert jn_syms["OrderService.cancelOrder"] is False
    assert jn_syms["OrderService.SanityTest"] is True
    assert jn_syms["OrderService.SanityTest.ok"] is True

    # Java enum @Benchmark (`enum_body_declarations`), JMH @State/@Setup, and JUnit 3 TestCase
    java_quirks = b"""package com.acme;
import junit.framework.TestCase;
import org.openjdk.jmh.annotations.*;

public enum HashImpl {
    SHA256;
    @Benchmark
    public int measure() { return 1; }
}

@State(Scope.Benchmark)
class OrderState {
    @Setup
    public void setUp() {}
}

class LegacyChecks extends TestCase {
    public void testLegacy() {}
}
"""
    jq_res = extractor.extract(
        Path("src/main/java/com/acme/PerfAndLegacy.java"), "java", java_quirks
    )
    assert jq_res.chunks
    assert all(chunk.is_test for chunk in jq_res.chunks)

    # C# prod class with nested benchmark class keeps outer class/methods as production code
    cs_nested = b"""namespace Acme;
public class OrderService {
    public int PlaceOrder(int qty) => qty * 10;

    public class OrderServiceBenchmarks {
        [BenchmarkDotNet.Attributes.Benchmark]
        public int Measure() => 1;
    }
}
"""
    csn_res = extractor.extract(Path("src/Acme/OrderService.cs"), "csharp", cs_nested)
    csn_syms = {c.qualified_symbol: c.is_test for c in csn_res.chunks if c.qualified_symbol}
    assert csn_syms["OrderService"] is False
    assert csn_syms["OrderService.PlaceOrder"] is False
    assert csn_syms["OrderService.OrderServiceBenchmarks"] is True
    assert csn_syms["OrderService.OrderServiceBenchmarks.Measure"] is True

    # C# xUnit nested class-per-method test suite AND `::` alias-qualified attributes
    cs_suite = b"""extern alias xunit;
namespace Acme;
public class OrderServiceTests {
    private static int BuildSample() => 7;

    public class PlaceOrder {
        [xunit::Fact]
        public void AcceptsPositive() => _ = BuildSample();

        [global::Fact]
        public void AcceptsGlobal() {}
    }
}
"""
    css_res = extractor.extract(Path("src/Acme/OrderServiceSuite.cs"), "csharp", cs_suite)
    assert css_res.chunks
    assert all(chunk.is_test for chunk in css_res.chunks)


def test_rust_pub_use_facade_criterion_custom_config_and_inner_mod_cfg_test() -> None:
    extractor = TreeSitterExtractor()

    # `pub use` re-export facade with inline `#[cfg(test)]` keeps facade module chunk in `code`
    facade_src = b"""pub use crate::auth::{Session, Token};
pub use crate::billing::Invoice;

#[cfg(test)]
mod tests {
    #[test]
    fn reexports_visible() {}
}
"""
    facade_res = extractor.extract(Path("src/lib.rs"), "rust", facade_src)
    prod_modules = [c for c in facade_res.chunks if c.kind == "module" and not c.is_test]
    assert any("pub use crate::auth" in c.content for c in prod_modules)
    by_sym = {c.qualified_symbol: c.is_test for c in facade_res.chunks if c.qualified_symbol}
    assert by_sym["reexports_visible"] is True

    # Criterion custom-config syntax (`name = ...; config = ...; targets = ...`)
    criterion_custom = b"""use criterion::{Criterion, criterion_group, criterion_main};

fn bench_alpha(c: &mut Criterion) {}
fn bench_beta(c: &mut Criterion) {}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_alpha, self::bench_beta
}
criterion_main!(benches);
"""
    crit_res = extractor.extract(Path("src/perf_custom.rs"), "rust", criterion_custom)
    assert crit_res.chunks
    assert all(chunk.is_test for chunk in crit_res.chunks)

    # Inline `mod tests { #![cfg(test)] ... }`, `#[fixture]`, and `#[cfg(doctest)]`
    inner_mod_src = b"""pub fn compute() -> i32 { 7 }

#[rstest::fixture]
fn sample_value() -> i32 { 3 }

#[cfg(doctest)]
fn doctest_only_stub() {}

mod tests {
    #![cfg(test)]
    struct Fixture;
    fn make_fixture() -> Fixture { Fixture }
    #[test]
    fn check() { let _ = make_fixture(); }
}
"""
    inner_res = extractor.extract(Path("src/compute.rs"), "rust", inner_mod_src)
    inner_syms = {c.qualified_symbol: c.is_test for c in inner_res.chunks if c.qualified_symbol}
    assert inner_syms["compute"] is False
    assert inner_syms["sample_value"] is True
    assert inner_syms["doctest_only_stub"] is True
    assert inner_syms["Fixture"] is True
    assert inner_syms["make_fixture"] is True
    assert inner_syms["check"] is True
