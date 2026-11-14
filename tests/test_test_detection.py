"""Tests for test/benchmark detection and search scoping.

Covers unambiguous test/benchmark directories across all indexed languages plus
filename and AST rules for C#, Java, Python, Rust, and TypeScript/TSX.
"""

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
        # TypeScript / TSX
        ("typescript", "tests/helpers/factory.ts", True),
        ("tsx", "__tests__/Button.tsx", True),
        ("typescript", "src/components/__mocks__/api.ts", True),
        ("typescript", "src/__fixtures__/users.ts", True),
        ("typescript", "e2e/checkout.ts", True),
        ("typescript", "playwright/login.ts", True),
        ("typescript", "cypress/support/commands.ts", True),
        ("typescript", "src/auth.test.ts", True),
        ("tsx", "src/Button.spec.tsx", True),
        ("typescript", "src/hash.bench.mts", True),
        ("typescript", "src/orders.integration.test.cts", True),
        ("typescript", "src/api.e2e-spec.ts", True),
        ("typescript", "src/schema.test-d.ts", True),
        ("typescript", "src/parser_test.ts", True),
        ("typescript", "src/hash_bench.ts", True),
        ("typescript", "src/orders/spec.ts", True),
        ("typescript", "src/orders/test.ts", True),
        ("typescript", "src/contest.ts", False),
        ("tsx", "src/latest.tsx", False),
        ("typescript", "src/testable.ts", False),
        ("typescript", "src/testRunner.ts", False),
        ("typescript", "src/testing/runner.ts", False),
        ("typescript", "src/domain/specs/activeOrder.ts", False),
        ("typescript", "src/types/jest.d.ts", False),
        ("typescript", "src/types/test.d.ts", False),
        # Other languages: unambiguous test/benchmark directories classify as True,
        # while paths outside test directories stay False (no filename/AST rules yet)
        ("javascript", "tests/auth.test.js", True),
        ("javascript", "__tests__/widget.js", True),
        ("javascript", "__mocks__/api.js", True),
        ("javascript", "src/auth.test.js", False),
        ("javascript", "e2e/checkout.js", False),
        ("json", "tests/fixtures/extractor_snapshot.json", True),
        ("json", "src/__fixtures__/users.json", True),
        ("json", "package.json", False),
        ("yaml", "tests/fixtures/compose.yaml", True),
        ("yaml", ".github/workflows/ci.yml", False),
        ("go", "tests/fixtures/resolver_corpus/go/sample.go", True),
        ("go", "pkg/auth_test.go", False),
        ("kotlin", "app/src/test/kotlin/FooTest.kt", True),
        ("kotlin", "app/src/main/kotlin/Foo.kt", False),
        ("sql", "benches/query.sql", True),
        ("sql", "src/schema.sql", False),
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


def test_typescript_and_tsx_suites_fixtures_type_tests_and_colocated_specs() -> None:
    extractor = TreeSitterExtractor()

    # Vitest / Jest / Mocha / Bun / node:test suites in a non-test path promote whole file
    suite_src = b"""import {
    assertType,
    bench,
    beforeEach,
    describe,
    expectTypeOf,
    it,
    test,
} from "vitest";

describe("orderSuite", () => {
    const buildOrder = (qty: number) => ({ qty });

    beforeEach(() => {
        buildOrder(1);
    });

    it.each([1, 2])("creates order %i", (qty) => {
        const order = buildOrder(qty);
        expectTypeOf(order.qty).toEqualTypeOf<number>();
        assertType<{ qty: number }>(order);
    });

    test.only("single order", () => {
        buildOrder(3);
    });

    bench("buildOrder", () => {
        buildOrder(10);
    });
});
"""
    suite_res = extractor.extract(Path("src/orderSuite.ts"), "typescript", suite_src)
    assert suite_res.chunks
    assert all(chunk.is_test for chunk in suite_res.chunks)

    # TSX component tests in both unambiguous test path and non-test path
    tsx_test_path_src = b"""import { Button } from "./Button";

const renderDefault = () => <Button label="ok" />;
"""
    tsx_test_res = extractor.extract(
        Path("src/components/Button.test.tsx"), "tsx", tsx_test_path_src
    )
    assert tsx_test_res.chunks
    assert all(chunk.is_test for chunk in tsx_test_res.chunks)

    tsx_colocated_src = b"""import { render } from "@testing-library/react";
import { describe, it } from "vitest";
import { Button } from "./Button";

describe("Button", () => {
    const renderButton = () => <Button label="ok" />;

    it("renders label", () => {
        render(renderButton());
    });
});
"""
    tsx_colocated_res = extractor.extract(
        Path("src/components/ButtonChecks.tsx"), "tsx", tsx_colocated_src
    )
    assert tsx_colocated_res.chunks
    assert all(chunk.is_test for chunk in tsx_colocated_res.chunks)
    tsx_syms = {
        c.qualified_symbol: c.is_test for c in tsx_colocated_res.chunks if c.qualified_symbol
    }
    assert tsx_syms["renderButton"] is True

    # Playwright / Vitest custom fixture file using base.extend(...)
    fixtures_src = b"""import { test as base } from "@playwright/test";

export const test = base.extend<{ api: string }>({
    api: async ({}, use) => {
        await use("https://api.example.test");
    },
});
"""
    fixtures_res = extractor.extract(Path("src/fixtures.ts"), "typescript", fixtures_src)
    assert fixtures_res.chunks
    assert all(chunk.is_test for chunk in fixtures_res.chunks)

    # Co-located spec.ts: production OpenAPI spec stays code, AST-confirmed orders/spec.ts promotes
    openapi_spec_src = b"""export interface OpenApiSpec {
    openapi: string;
}

export function loadSpec(): OpenApiSpec {
    return { openapi: "3.1.0" };
}
"""
    openapi_res = extractor.extract(Path("src/openapi/spec.ts"), "typescript", openapi_spec_src)
    assert openapi_res.chunks
    assert all(not chunk.is_test for chunk in openapi_res.chunks)

    orders_spec_src = b"""function makeOrder() {
    return { id: 1 };
}

describe("orders", () => {
    it("creates order", () => {
        makeOrder();
    });
});
"""
    orders_spec_res = extractor.extract(Path("src/orders/spec.ts"), "typescript", orders_spec_src)
    assert orders_spec_res.chunks
    assert all(chunk.is_test for chunk in orders_spec_res.chunks)

    # Ambient .d.ts stays code while pure-type .test-d.ts classifies whole file as test
    ambient_dts_src = b"""export interface TestEnv {
    retries: number;
}
"""
    ambient_res = extractor.extract(Path("src/types/test.d.ts"), "typescript", ambient_dts_src)
    assert ambient_res.chunks
    assert all(not chunk.is_test for chunk in ambient_res.chunks)

    type_test_src = b"""export type OrderId = string;
type AssertOrderIdIsString = OrderId extends string ? true : false;
"""
    type_test_res = extractor.extract(
        Path("src/types/order.test-d.ts"), "typescript", type_test_src
    )
    assert type_test_res.chunks
    assert all(chunk.is_test for chunk in type_test_res.chunks)


def test_typescript_inline_vitest_deno_and_decorated_suites_separate_prod_from_tests() -> None:
    extractor = TreeSitterExtractor()

    # Mixed production + Vitest in-source if (import.meta.vitest) + false-positive guards
    validator_src = b"""export class RegexValidator {
    test(input: string): boolean {
        return /^ok$/.test(input);
    }
}

export function validateOrder(s: string): boolean {
    if (!import.meta.vitest) {
        return s.length > 0;
    }
    return true;
}

if (import.meta.vitest) {
    const makeSample = () => "ok";
    it("validates sample", () => {
        validateOrder(makeSample());
    });
}
"""
    val_res = extractor.extract(Path("src/validator.ts"), "typescript", validator_src)
    val_syms = {c.qualified_symbol: c.is_test for c in val_res.chunks if c.qualified_symbol}
    assert val_syms["RegexValidator"] is False
    assert val_syms["RegexValidator.test"] is False
    assert val_syms["validateOrder"] is False
    assert val_syms["makeSample"] is True

    val_prod_modules = [c for c in val_res.chunks if c.kind == "module" and not c.is_test]
    val_test_modules = [c for c in val_res.chunks if c.kind == "module" and c.is_test]
    assert val_test_modules
    assert any("validates sample" in c.content for c in val_test_modules)
    assert all("validates sample" not in c.content for c in val_prod_modules)
    assert all("makeSample" not in c.content for c in val_prod_modules)

    # Mixed production + inline Deno.test / Deno.bench
    deno_src = b"""export function add(a: number, b: number): number {
    return a + b;
}

Deno.test("adds numbers", () => {
    add(1, 2);
});

Deno.bench("add throughput", () => {
    add(10, 20);
});
"""
    deno_res = extractor.extract(Path("src/deno_math.ts"), "typescript", deno_src)
    deno_syms = {c.qualified_symbol: c.is_test for c in deno_res.chunks if c.qualified_symbol}
    assert deno_syms["add"] is False

    deno_prod_modules = [c for c in deno_res.chunks if c.kind == "module" and not c.is_test]
    deno_test_modules = [c for c in deno_res.chunks if c.kind == "module" and c.is_test]
    assert any("Deno.test" in c.content for c in deno_test_modules)
    assert any("Deno.bench" in c.content for c in deno_test_modules)
    assert all("Deno.test" not in c.content for c in deno_prod_modules)
    assert all("Deno.bench" not in c.content for c in deno_prod_modules)

    # Decorated @Suite / @Test class promotes whole class & helper methods
    decorated_src = b"""@Suite()
export class OrderSuite {
    helper() {}

    @Test()
    createsOrder() {
        this.helper();
    }
}
"""
    dec_res = extractor.extract(Path("src/orderDecorated.ts"), "typescript", decorated_src)
    assert dec_res.chunks
    assert all(chunk.is_test for chunk in dec_res.chunks)
    dec_syms = {c.qualified_symbol: c.is_test for c in dec_res.chunks if c.qualified_symbol}
    assert dec_syms["OrderSuite"] is True
    assert dec_syms["OrderSuite.helper"] is True
    assert dec_syms["OrderSuite.createsOrder"] is True

    # Outer production class + nested @Suite class in namespace keeps outer prod code False
    nested_src = b"""export class OrderService {
    placeOrder() {
        return 1;
    }
}

export namespace OrderService {
    @Suite()
    export class SanitySuite {
        helper() {}

        @Test()
        ok() {
            this.helper();
        }
    }
}
"""
    nested_res = extractor.extract(Path("src/orderServiceNested.ts"), "typescript", nested_src)
    nested_syms = {c.qualified_symbol: c.is_test for c in nested_res.chunks if c.qualified_symbol}
    assert nested_syms["OrderService"] is False
    assert nested_syms["OrderService.placeOrder"] is False
    assert nested_syms["SanitySuite"] is True
    assert nested_syms["SanitySuite.helper"] is True
    assert nested_syms["SanitySuite.ok"] is True


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

    # TypeScript inline if (import.meta.vitest) + TSX production & test components
    (repo / "src" / "checkout.ts").write_text(
        "export function computeOrderTotal(price: number, qty: number): number {\n"
        "    if (!import.meta.vitest) {\n"
        "        return price * qty;\n"
        "    }\n"
        "    return price * qty;\n"
        "}\n\n"
        "if (import.meta.vitest) {\n"
        "    const makeCartSample = () => computeOrderTotal(10, 2);\n"
        '    it("computes total", () => {\n'
        "        makeCartSample();\n"
        "    });\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "src" / "Widget.tsx").write_text(
        'import { computeOrderTotal } from "./checkout";\n\n'
        "export function WidgetSummary({ qty }: { qty: number }) {\n"
        "    const total = computeOrderTotal(5, qty);\n"
        "    return <span>{total}</span>;\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "__tests__").mkdir()
    (repo / "__tests__" / "Widget.test.tsx").write_text(
        'import { WidgetSummary } from "../src/Widget";\n'
        'import { computeOrderTotal } from "../src/checkout";\n\n'
        "const makeWidgetProps = () => ({ qty: computeOrderTotal(2, 3) });\n\n"
        'describe("WidgetSummary", () => {\n'
        '    it("renders total", () => {\n'
        "        return <WidgetSummary qty={makeWidgetProps().qty} />;\n"
        "    });\n"
        "});\n",
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

    # Default scope="code" excludes Python tests/, Rust #[cfg(test)], C# *.Tests/, Java src/test/,
    # TS inline import.meta.vitest, and TSX __tests__/
    code_hits = app.search_code("verify_session_token", limit=50).hits
    assert code_hits[0].path == "src/auth.py"
    assert all(not hit.is_test for hit in code_hits)
    assert {hit.path for hit in code_hits} == {
        "src/auth.py",
        "src/lib.rs",
        "src/Billing.cs",
        "src/main/java/demo/Inventory.java",
        "src/checkout.ts",
        "src/Widget.tsx",
    }
    assert all("order_fixture" not in hit.snippet for hit in code_hits)
    assert all("makeCartSample" not in hit.snippet for hit in code_hits)

    test_hits = app.search_code("verify_session_token", scope="tests", limit=50).hits
    assert test_hits[0].path == "tests/test_auth.py"
    assert all(hit.is_test for hit in test_hits)
    assert {hit.path for hit in test_hits} == {
        "tests/test_auth.py",
        "src/lib.rs",
        "Demo.Tests/LedgerBuilder.cs",
        "src/test/java/demo/InventoryTest.java",
        "src/checkout.ts",
        "__tests__/Widget.test.tsx",
    }

    all_hits = app.search_code("verify_session_token", scope="all", limit=50).hits
    assert any(hit.is_test for hit in all_hits)
    assert any(not hit.is_test for hit in all_hits)
    assert "tests/test_auth.py" in {hit.path for hit in all_hits}
    assert "src/auth.py" in {hit.path for hit in all_hits}
    assert "__tests__/Widget.test.tsx" in {hit.path for hit in all_hits}
    assert "src/Widget.tsx" in {hit.path for hit in all_hits}

    # Rust inline #[cfg(test)] helpers are excluded from code symbol lookup, included in tests/all
    assert app.find_symbol("dispatch_order").hits
    assert app.find_symbol("order_fixture").hits == []
    rust_test_sym = app.find_symbol("order_fixture", scope="tests").hits
    assert len(rust_test_sym) == 1
    assert rust_test_sym[0].is_test is True

    # TS inline import.meta.vitest and TSX test symbols obey scope
    ts_code_sym = app.find_symbol("computeOrderTotal").hits
    assert [hit.path for hit in ts_code_sym] == ["src/checkout.ts"]
    assert ts_code_sym[0].is_test is False
    assert app.find_symbol("makeCartSample").hits == []
    ts_test_sym = app.find_symbol("makeCartSample", scope="tests").hits
    assert len(ts_test_sym) == 1
    assert ts_test_sym[0].path == "src/checkout.ts"
    assert ts_test_sym[0].is_test is True

    assert [hit.path for hit in app.find_symbol("WidgetSummary").hits] == ["src/Widget.tsx"]
    assert app.find_symbol("makeWidgetProps").hits == []
    tsx_test_sym = app.find_symbol("makeWidgetProps", scope="tests").hits
    assert [hit.path for hit in tsx_test_sym] == ["__tests__/Widget.test.tsx"]
    assert tsx_test_sym[0].is_test is True
    assert [hit.path for hit in app.find_symbol("makeWidgetProps", scope="all").hits] == [
        "__tests__/Widget.test.tsx"
    ]

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
    ex_code = app.search_by_example(snippet, limit=50).hits
    assert all(not hit.is_test for hit in ex_code)
    ex_tests = app.search_by_example(snippet, scope="tests", limit=50).hits
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

    ts_chunk_id = ts_code_sym[0].chunk_id
    ts_analysis = app.analyze_refactor(
        DeclarationSelector(chunk_id=ts_chunk_id),
        RenameOperation(new_name="calculateOrderTotal"),
    )
    ts_finding_paths = {finding.path for finding in ts_analysis.findings}
    assert "src/checkout.ts" in ts_finding_paths
    assert "src/Widget.tsx" in ts_finding_paths
    assert "__tests__/Widget.test.tsx" in ts_finding_paths
    ts_checkout_lines = {
        finding.start_line for finding in ts_analysis.findings if finding.path == "src/checkout.ts"
    }
    assert 1 in ts_checkout_lines
    assert 9 in ts_checkout_lines


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
    (alpha_root / "src" / "ledger.ts").write_text(
        "export function settleLedgerEntry(amount: number): number {\n"
        "    return amount;\n"
        "}\n\n"
        "if (import.meta.vitest) {\n"
        "    const settle_ledger_entry_sample = () => settleLedgerEntry(1);\n"
        '    it("checks ledger", () => {\n'
        "        settle_ledger_entry_sample();\n"
        "    });\n"
        "}\n",
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
    (beta_root / "src" / "LedgerView.tsx").write_text(
        "export function LedgerView({ amount }: { amount: number }) {\n"
        "    return <span>{amount}</span>;\n"
        "}\n",
        encoding="utf-8",
    )
    (beta_root / "src" / "LedgerView.test.tsx").write_text(
        'import { LedgerView } from "./LedgerView";\n'
        "export function renderLedgerTest() {\n"
        "    return <LedgerView amount={1} />;\n"
        "}\n",
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
            {"projects": [alpha.id, beta.id], "query": "settle_ledger_entry", "limit": 20},
        )
        assert not default_res.isError
        assert default_res.structuredContent is not None
        default_hits = default_res.structuredContent["hits"]
        assert all(hit["is_test"] is False for hit in default_hits)
        default_paths = {hit["path"] for hit in default_hits}
        assert default_paths == {
            "src/payment.py",
            "src/ledger.ts",
            "src/lib.rs",
            "src/LedgerView.tsx",
        }

        tests_res = await client.call_tool(
            "search_across_projects",
            {
                "projects": [alpha.id, beta.id],
                "query": "settle_ledger_entry",
                "scope": "tests",
                "limit": 20,
            },
        )
        assert not tests_res.isError
        assert tests_res.structuredContent is not None
        test_hits = tests_res.structuredContent["hits"]
        assert all(hit["is_test"] is True for hit in test_hits)
        test_paths = {hit["path"] for hit in test_hits}
        assert test_paths == {
            "tests/test_payment.py",
            "src/ledger.ts",
            "benches/ledger_bench.rs",
            "src/LedgerView.test.tsx",
        }

        all_res = await client.call_tool(
            "search_across_projects",
            {
                "projects": [alpha.id, beta.id],
                "query": "settle_ledger_entry",
                "scope": "all",
                "limit": 30,
            },
        )
        assert not all_res.isError
        assert all_res.structuredContent is not None
        all_hits = all_res.structuredContent["hits"]
        assert any(hit["is_test"] is False for hit in all_hits)
        assert any(hit["is_test"] is True for hit in all_hits)
        assert "src/LedgerView.tsx" in {hit["path"] for hit in all_hits}
        assert "src/LedgerView.test.tsx" in {hit["path"] for hit in all_hits}

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
        assert sym_tests.structuredContent["hits"][0]["is_test"] is True

        ts_sym_code = await client.call_tool(
            "find_symbol",
            {"name": "settle_ledger_entry_sample", "project": alpha.id},
        )
        assert ts_sym_code.structuredContent is not None
        assert ts_sym_code.structuredContent["hits"] == []

        ts_sym_tests = await client.call_tool(
            "find_symbol",
            {"name": "settle_ledger_entry_sample", "project": alpha.id, "scope": "tests"},
        )
        assert ts_sym_tests.structuredContent is not None
        assert len(ts_sym_tests.structuredContent["hits"]) == 1
        assert ts_sym_tests.structuredContent["hits"][0]["path"] == "src/ledger.ts"
        assert ts_sym_tests.structuredContent["hits"][0]["is_test"] is True

        tsx_sym_code = await client.call_tool(
            "find_symbol",
            {"name": "renderLedgerTest", "project": beta.id},
        )
        assert tsx_sym_code.structuredContent is not None
        assert tsx_sym_code.structuredContent["hits"] == []

        tsx_sym_tests = await client.call_tool(
            "find_symbol",
            {"name": "renderLedgerTest", "project": beta.id, "scope": "tests"},
        )
        assert tsx_sym_tests.structuredContent is not None
        assert len(tsx_sym_tests.structuredContent["hits"]) == 1
        assert tsx_sym_tests.structuredContent["hits"][0]["path"] == "src/LedgerView.test.tsx"
        assert tsx_sym_tests.structuredContent["hits"][0]["is_test"] is True


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


def test_typescript_default_and_commonjs_single_fn_runner_imports() -> None:
    extractor = TreeSitterExtractor()

    ava_src = b"""import test from "ava";

const buildPayload = () => ({ ok: true });

test("accepts payload", (t) => {
    t.truthy(buildPayload());
});

test.serial("runs serially", (t) => {
    t.truthy(buildPayload());
});
"""
    for rel_path in ("src/avaChecks.ts", "src/orders/test.ts"):
        ava_res = extractor.extract(Path(rel_path), "typescript", ava_src)
        assert ava_res.chunks
        assert all(chunk.is_test for chunk in ava_res.chunks), rel_path

    node_test_src = b"""import test from "node:test";

test("basic", () => {});
test.only("focused", () => {});
test.describe("nested suite", () => {
    test("inner", () => {});
});
"""
    for rel_path in ("src/nodeChecks.ts", "src/orders/test.ts"):
        node_res = extractor.extract(Path(rel_path), "typescript", node_test_src)
        assert node_res.chunks
        assert all(chunk.is_test for chunk in node_res.chunks), rel_path

    tape_src = b"""import tape from "tape";

tape("tape assertion", (t) => {
    t.end();
});
"""
    tape_res = extractor.extract(Path("src/tapeChecks.ts"), "typescript", tape_src)
    assert tape_res.chunks
    assert all(chunk.is_test for chunk in tape_res.chunks)

    cjs_node_src = b"""const test = require("node:test");

test("cjs node test", () => {});
"""
    cjs_res = extractor.extract(Path("src/cjsNodeChecks.ts"), "typescript", cjs_node_src)
    assert cjs_res.chunks
    assert all(chunk.is_test for chunk in cjs_res.chunks)

    ts_require_ava_src = b"""import test = require("ava");

test("ts import-require ava", (t) => {
    t.pass();
});
"""
    ts_req_res = extractor.extract(Path("src/orders/test.ts"), "typescript", ts_require_ava_src)
    assert ts_req_res.chunks
    assert all(chunk.is_test for chunk in ts_req_res.chunks)

    # Negative guard: namespace-only default import called directly is NOT a test runner
    playwright_neg_src = b"""import playwright from "@playwright/test";

playwright("not a runner");
"""
    for rel_path in ("src/playwrightCaller.ts", "src/orders/test.ts"):
        neg_res = extractor.extract(Path(rel_path), "typescript", playwright_neg_src)
        assert neg_res.chunks
        assert all(not chunk.is_test for chunk in neg_res.chunks), rel_path


def test_typescript_type_only_imports_do_not_trigger_test_classification() -> None:
    extractor = TreeSitterExtractor()

    pw_type_src = b"""import type { Page, BrowserContext } from "@playwright/test";

export function currentUrl(page: Page, defaultBrowser: BrowserContext): string {
    context(defaultBrowser);
    return page.url();
}
"""
    pw_type_res = extractor.extract(Path("src/pageHelpers.ts"), "typescript", pw_type_src)
    assert pw_type_res.chunks
    assert all(not chunk.is_test for chunk in pw_type_res.chunks)

    pw_tsx_res = extractor.extract(Path("src/PageHelpers.tsx"), "tsx", pw_type_src)
    assert pw_tsx_res.chunks
    assert all(not chunk.is_test for chunk in pw_tsx_res.chunks)

    vitest_type_src = b"""import type { Mock } from "vitest";

export function validateSpec(mockFn: Mock, defaultInput: string): boolean {
    return Boolean(mockFn) && test(defaultInput);
}
"""
    spec_res = extractor.extract(Path("src/orders/spec.ts"), "typescript", vitest_type_src)
    assert spec_res.chunks
    assert all(not chunk.is_test for chunk in spec_res.chunks)

    inline_type_alias_src = b"""import { type test as TestFn } from "vitest";

TestFn(123);
"""
    inline_res = extractor.extract(
        Path("src/typeAliasHelper.ts"), "typescript", inline_type_alias_src
    )
    assert inline_res.chunks
    assert all(not chunk.is_test for chunk in inline_res.chunks)

    type_default_and_ns_src = b"""import type test from "ava";
import type * as vt from "vitest";
import type avaReq = require("ava");

export type RunnerAlias = typeof test | typeof avaReq | vt.Suite;
context(defaultInput);
"""
    type_ns_res = extractor.extract(
        Path("src/typeOnlyVariants.ts"), "typescript", type_default_and_ns_src
    )
    assert type_ns_res.chunks
    assert all(not chunk.is_test for chunk in type_ns_res.chunks)

    # Mixed value + type import still detects runtime test bindings and sets has_test_import=True
    mixed_src = b"""import { type Suite, describe, it } from "vitest";

const makeSuiteName = (s?: Suite) => (s ? "named" : "default");

describe(makeSuiteName(), () => {
    it("runs with mixed import", () => {});
});
"""
    mixed_res = extractor.extract(Path("src/mixedVitest.ts"), "typescript", mixed_src)
    assert mixed_res.chunks
    assert all(chunk.is_test for chunk in mixed_res.chunks)

    mixed_hook_src = b"""import { type Suite, expect } from "vitest";

const label = (_s?: Suite) => "suite";

context(label(), () => {
    expect(1).toBe(1);
});
"""
    mixed_hook_res = extractor.extract(Path("src/mixedVitestHook.ts"), "typescript", mixed_hook_src)
    assert mixed_hook_res.chunks
    assert all(chunk.is_test for chunk in mixed_hook_res.chunks)


def test_typescript_vitest_else_branches_and_demorgan_guards() -> None:
    extractor = TreeSitterExtractor()

    if_else_src = b"""if (import.meta.vitest) {
    const sample = () => "sample";
    it("runs inline vitest", () => {
        sample();
    });
} else {
    function bootstrapCli() {
        return "cli";
    }
    bootstrapCli();
}
"""
    if_else_res = extractor.extract(Path("src/cliEntry.ts"), "typescript", if_else_src)
    if_else_syms = {c.qualified_symbol: c.is_test for c in if_else_res.chunks if c.qualified_symbol}
    assert if_else_syms["sample"] is True
    assert if_else_syms["bootstrapCli"] is False

    if_else_prod_mods = [c for c in if_else_res.chunks if c.kind == "module" and not c.is_test]
    if_else_test_mods = [c for c in if_else_res.chunks if c.kind == "module" and c.is_test]
    assert if_else_test_mods
    assert any("runs inline vitest" in c.content for c in if_else_test_mods)
    assert all("runs inline vitest" not in c.content for c in if_else_prod_mods)
    assert any("bootstrapCli" in c.content for c in if_else_prod_mods)
    assert all("bootstrapCli" not in c.content for c in if_else_test_mods)

    neg_if_else_src = b"""if (!import.meta.vitest) {
    function bootstrapProd() {
        return "prod";
    }
    bootstrapProd();
} else {
    function makeVitestSample() {
        return "vitest";
    }
    it("runs in else branch", () => {
        makeVitestSample();
    });
}
"""
    neg_res = extractor.extract(Path("src/prodBootstrap.ts"), "typescript", neg_if_else_src)
    neg_syms = {c.qualified_symbol: c.is_test for c in neg_res.chunks if c.qualified_symbol}
    assert neg_syms["bootstrapProd"] is False
    assert neg_syms["makeVitestSample"] is True

    neg_prod_mods = [c for c in neg_res.chunks if c.kind == "module" and not c.is_test]
    neg_test_mods = [c for c in neg_res.chunks if c.kind == "module" and c.is_test]
    assert neg_test_mods
    assert any("runs in else branch" in c.content for c in neg_test_mods)
    assert all("runs in else branch" not in c.content for c in neg_prod_mods)
    assert any("bootstrapProd" in c.content for c in neg_prod_mods)
    assert all("bootstrapProd" not in c.content for c in neg_test_mods)

    # De Morgan false-positive guards must keep all chunks as production code
    demorgan_not_src = b"""declare const strictMode: boolean;

if (!(!import.meta.vitest && strictMode)) {
    function devFallback() {
        return "fallback";
    }
    devFallback();
}
"""
    dm_not_res = extractor.extract(Path("src/orders/test.ts"), "typescript", demorgan_not_src)
    assert dm_not_res.chunks
    assert all(not chunk.is_test for chunk in dm_not_res.chunks)

    demorgan_typeof_src = b"""declare const isProd: boolean;

if (!(typeof import.meta.vitest === "undefined" && isProd)) {
    function enableDebugTelemetry() {
        return true;
    }
    enableDebugTelemetry();
}
"""
    dm_typeof_res = extractor.extract(Path("src/runtimeInit.ts"), "typescript", demorgan_typeof_src)
    assert dm_typeof_res.chunks
    assert all(not chunk.is_test for chunk in dm_typeof_res.chunks)


def test_typescript_shadowed_ambient_globals_and_local_declarations() -> None:
    extractor = TreeSitterExtractor()

    router_hook_src = b"""import { beforeEach } from "./router";

beforeEach((to, from) => {
    return Boolean(to && from);
});
"""
    router_res = extractor.extract(Path("src/navigation.ts"), "typescript", router_hook_src)
    assert router_res.chunks
    assert all(not chunk.is_test for chunk in router_res.chunks)

    validator_import_src = b"""import { test } from "./schemaValidator";

test("order-id", (v: string) => v.length > 0);
"""
    validator_res = extractor.extract(
        Path("src/orders/test.ts"), "typescript", validator_import_src
    )
    assert validator_res.chunks
    assert all(not chunk.is_test for chunk in validator_res.chunks)

    hoisted_local_decl_src = b"""test("before-declaration", () => true);

export function test(rule: string, fn: () => boolean): boolean {
    return Boolean(rule) && fn();
}

test("non-empty", () => true);
"""
    hoisted_res = extractor.extract(
        Path("src/orders/test.ts"), "typescript", hoisted_local_decl_src
    )
    assert hoisted_res.chunks
    assert all(not chunk.is_test for chunk in hoisted_res.chunks)

    # Extended test fixture binding via base.extend(...) must still classify as test
    extended_src = b"""import { test as base } from "@playwright/test";

const test = base.extend<{ api: string }>({
    api: async ({}, use) => {
        await use("https://api.example.test");
    },
});

test("uses extended fixture", async ({ api }) => {
    return api;
});
"""
    extended_res = extractor.extract(Path("src/extendedChecks.ts"), "typescript", extended_src)
    assert extended_res.chunks
    assert all(chunk.is_test for chunk in extended_res.chunks)


def test_typescript_void_floating_promise_runner_calls() -> None:
    extractor = TreeSitterExtractor()

    nested_void_src = b"""import { describe, test } from "node:test";

const makeOrder = () => ({ id: 1 });

void describe("suite", () => {
    void test("case", async () => {
        makeOrder();
    });
});
"""
    for rel_path, lang in (
        ("src/ordersVoid.ts", "typescript"),
        ("src/orders/test.ts", "typescript"),
        ("src/OrderVoid.tsx", "tsx"),
    ):
        res = extractor.extract(Path(rel_path), lang, nested_void_src)
        assert res.chunks
        assert all(chunk.is_test for chunk in res.chunks), rel_path

    standalone_void_src = b"""import { test } from "node:test";

void test("standalone async check", async () => {});
"""
    standalone_res = extractor.extract(
        Path("src/orders/test.ts"), "typescript", standalone_void_src
    )
    assert standalone_res.chunks
    assert all(chunk.is_test for chunk in standalone_res.chunks)


def test_typescript_playwright_fixture_module_with_local_export_expect() -> None:
    extractor = TreeSitterExtractor()

    fixtures_src = b"""import { test as base, expect } from "@playwright/test";

function createTestUser() {
    return { id: "user-1" };
}

export const test = base.extend<{ user: { id: string } }>({
    user: async ({}, use) => {
        await use(createTestUser());
    },
});

export { expect };
"""
    fixtures_res = extractor.extract(Path("src/fixtures.ts"), "typescript", fixtures_src)
    assert fixtures_res.chunks
    assert all(chunk.is_test for chunk in fixtures_res.chunks)
    syms = {c.qualified_symbol: c.is_test for c in fixtures_res.chunks if c.qualified_symbol}
    assert syms["createTestUser"] is True


def test_non_ast_languages_in_test_and_benchmark_directories_are_classified_and_scoped(
    tmp_path: Path,
) -> None:
    extractor = TreeSitterExtractor()

    fixture_cases: list[tuple[str, str, bytes]] = [
        ("tests/fixtures/extractor_snapshot.json", "json", b'{"verifyFixtureToken": true}\n'),
        (
            "tests/fixtures/compose.yaml",
            "yaml",
            b"services:\n  verify_fixture_token:\n    image: alpine\n",
        ),
        (
            "__tests__/widget.js",
            "javascript",
            b"export function verifyFixtureToken() { return true; }\n",
        ),
        (
            "app/src/test/kotlin/FixtureToken.kt",
            "kotlin",
            b"package demo\nfun verifyFixtureToken(): Boolean = true\n",
        ),
        ("benches/query.sql", "sql", b"CREATE VIEW verify_fixture_token AS SELECT 1;\n"),
    ]
    for rel_path, language, source in fixture_cases:
        res = extractor.extract(Path(rel_path), language, source)
        assert res.chunks, rel_path
        assert all(chunk.is_test for chunk in res.chunks), rel_path

    prod_cases: list[tuple[str, str, bytes]] = [
        ("package.json", "json", b'{"verifyFixtureToken": false}\n'),
        (".github/workflows/ci.yml", "yaml", b"name: ci\n"),
        (
            "src/auth.test.js",
            "javascript",
            b"export function verifyFixtureToken() { return false; }\n",
        ),
        (
            "pkg/auth_test.go",
            "go",
            b"package auth\nfunc VerifyFixtureToken() bool { return false }\n",
        ),
        ("src/schema.sql", "sql", b"CREATE TABLE prod_tokens (id INT);\n"),
    ]
    for rel_path, language, source in prod_cases:
        res = extractor.extract(Path(rel_path), language, source)
        assert res.chunks, rel_path
        assert all(not chunk.is_test for chunk in res.chunks), rel_path

    # End-to-end indexing and search/symbol scoping on non-AST fixture vs production files.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "token.js").write_text(
        "export function verifyFixtureToken() { return true; }\n",
        encoding="utf-8",
    )
    (repo / "package.json").write_text(
        '{"name": "demo", "verifyFixtureToken": "prod"}\n',
        encoding="utf-8",
    )
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "snapshot.json").write_text(
        '{"verifyFixtureToken": "fixture"}\n',
        encoding="utf-8",
    )
    (repo / "tests" / "fixtures" / "helper.go").write_text(
        "package fixtures\nfunc verifyFixtureToken() bool { return true }\n",
        encoding="utf-8",
    )
    (repo / "__tests__").mkdir()
    (repo / "__tests__" / "token.js").write_text(
        "export function verifyFixtureToken() { return true; }\n",
        encoding="utf-8",
    )

    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=DeterministicEmbedder(),
        cwd=repo,
    )
    project = app.init_project(repo)
    app.index_project(project.id)

    code_hits = app.search_code("verifyFixtureToken", limit=50).hits
    assert code_hits
    assert all(not hit.is_test for hit in code_hits)
    assert {hit.path for hit in code_hits} == {"src/token.js", "package.json"}

    test_hits = app.search_code("verifyFixtureToken", scope="tests", limit=50).hits
    assert all(hit.is_test for hit in test_hits)
    assert {hit.path for hit in test_hits} == {
        "tests/fixtures/snapshot.json",
        "tests/fixtures/helper.go",
        "__tests__/token.js",
    }

    all_hits = app.search_code("verifyFixtureToken", scope="all", limit=50).hits
    assert {hit.path for hit in all_hits} == {
        "src/token.js",
        "package.json",
        "tests/fixtures/snapshot.json",
        "tests/fixtures/helper.go",
        "__tests__/token.js",
    }

    code_syms = app.find_symbol("verifyFixtureToken").hits
    assert [hit.path for hit in code_syms] == ["src/token.js"]
    assert code_syms[0].is_test is False

    test_syms = app.find_symbol("verifyFixtureToken", scope="tests").hits
    assert {hit.path for hit in test_syms} == {"tests/fixtures/helper.go", "__tests__/token.js"}
    assert all(hit.is_test for hit in test_syms)

    example_snippet = "export function verifyFixtureToken() { return true; }\n"
    ex_code = app.search_by_example(example_snippet, language="javascript", limit=50).hits
    assert all(not hit.is_test for hit in ex_code)
    assert {hit.path for hit in ex_code}.isdisjoint(
        {"tests/fixtures/snapshot.json", "tests/fixtures/helper.go", "__tests__/token.js"}
    )
    ex_tests = app.search_by_example(
        example_snippet, language="javascript", scope="tests", limit=50
    ).hits
    assert ex_tests
    assert all(hit.is_test for hit in ex_tests)
