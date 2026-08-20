/** Unit tests for the Phase 9 cross-build benchmark comparison. */

import { describe, expect, test } from "bun:test";
import {
  buildBenchmarkComparisonReport,
  compareIndexReports,
  compareSearchReports,
} from "../src/benchmark-compare.ts";

function indexReport(durations: Record<string, number>): Record<string, unknown> {
  return {
    schema_version: 2,
    revision: "p".repeat(40),
    scenarios: Object.fromEntries(
      Object.entries(durations).map(([name, reportedMs]) => [
        name,
        { wall_ms: reportedMs + 50, reported_duration_ms: reportedMs },
      ]),
    ),
  };
}

function indexReportWithMaintenance(): Record<string, unknown> {
  const report = indexReport({ cold_start: 1000 });
  const scenarios = report.scenarios as Record<string, Record<string, unknown>>;
  // post_maintenance times a maintenance pass: wall time only.
  scenarios.post_maintenance = { wall_ms: 12.5 };
  return report;
}

function searchReport(
  latencies: Record<string, { median: number; p95: number }>,
): Record<string, unknown> {
  return {
    schema_version: 1,
    revision: "p".repeat(40),
    scopes: Object.fromEntries(
      Object.entries(latencies).map(([name, values]) => [
        name,
        { latency_ms: { median_ms: values.median, p95_ms: values.p95 } },
      ]),
    ),
  };
}

describe("compareIndexReports", () => {
  test("a regression within tolerance passes", () => {
    const comparison = compareIndexReports(
      indexReport({ cold_start: 1000, no_op: 200 }),
      indexReport({ cold_start: 1149, no_op: 230 }),
    );
    expect(comparison.passed).toBe(true);
    const coldStart = comparison.scenarios.find((scenario) => scenario.scenario === "cold_start");
    expect(coldStart?.ratio).toBeCloseTo(1.149, 12);
    expect(coldStart?.gated).toBe(true);
  });

  test("a regression beyond tolerance fails only the offending scenario", () => {
    const comparison = compareIndexReports(
      indexReport({ cold_start: 1000, forced_reindex: 500 }),
      indexReport({ cold_start: 1001, forced_reindex: 700 }),
    );
    expect(comparison.passed).toBe(false);
    expect(
      comparison.scenarios.find((scenario) => scenario.scenario === "cold_start")?.within_tolerance,
    ).toBe(true);
    expect(
      comparison.scenarios.find((scenario) => scenario.scenario === "forced_reindex")
        ?.within_tolerance,
    ).toBe(false);
  });

  test("faster TypeScript runs pass with ratios under one", () => {
    const comparison = compareIndexReports(
      indexReport({ cold_start: 1000 }),
      indexReport({ cold_start: 600 }),
    );
    expect(comparison.passed).toBe(true);
    expect(comparison.scenarios[0]?.ratio).toBeCloseTo(0.6, 12);
  });

  test("scenarios below the minimum duration are recorded but not gated", () => {
    const comparison = compareIndexReports(indexReport({ no_op: 10 }), indexReport({ no_op: 500 }));
    expect(comparison.passed).toBe(true);
    expect(comparison.scenarios[0]?.gated).toBe(false);
    expect(comparison.scenarios[0]?.ratio).toBeCloseTo(50, 12);
  });

  test("a zero Python duration yields a null ratio rather than a bogus one", () => {
    const comparison = compareIndexReports(indexReport({ no_op: 60 }), indexReport({ no_op: 60 }), {
      minimumMs: 50,
    });
    expect(comparison.scenarios[0]?.ratio).toBeCloseTo(1, 12);
    const zero = compareIndexReports(indexReport({ no_op: 0 }), indexReport({ no_op: 0 }));
    expect(zero.scenarios[0]?.ratio).toBe(null);
    expect(zero.passed).toBe(true);
  });

  test("scenario sets must match", () => {
    expect(() =>
      compareIndexReports(
        indexReport({ cold_start: 1000 }),
        indexReport({ cold_start: 1000, extra: 5 }),
      ),
    ).toThrow("unknown scenarios: extra");
  });

  test("post_maintenance reports no pipeline duration and is not gated", () => {
    const comparison = compareIndexReports(
      indexReportWithMaintenance(),
      indexReportWithMaintenance(),
    );
    const maintenance = comparison.scenarios.find(
      (scenario) => scenario.scenario === "post_maintenance",
    );
    expect(maintenance?.python_reported_ms).toBe(null);
    expect(maintenance?.gated).toBe(false);
    expect(comparison.passed).toBe(true);
  });
});

describe("compareSearchReports", () => {
  test("median and p95 are gated per scope", () => {
    const comparison = compareSearchReports(
      searchReport({ "1": { median: 10, p95: 20 }, "50": { median: 30, p95: 60 } }),
      searchReport({ "1": { median: 11, p95: 21 }, "50": { median: 31, p95: 70 } }),
      { tolerance: 0.1 },
    );
    expect(comparison.scopes).toHaveLength(4);
    expect(comparison.passed).toBe(false);
    expect(
      comparison.scopes.find((scope) => scope.scope === "50" && scope.metric === "p95_ms")
        ?.within_tolerance,
    ).toBe(false);
    expect(
      comparison.scopes.find((scope) => scope.scope === "1" && scope.metric === "median_ms")
        ?.within_tolerance,
    ).toBe(true);
  });

  test("a zero Python latency is a floor, not a target", () => {
    const comparison = compareSearchReports(
      searchReport({ "1": { median: 0, p95: 0 } }),
      searchReport({ "1": { median: 0, p95: 5 } }),
    );
    expect(comparison.scopes.find((scope) => scope.metric === "median_ms")?.within_tolerance).toBe(
      true,
    );
    expect(comparison.scopes.find((scope) => scope.metric === "p95_ms")?.within_tolerance).toBe(
      false,
    );
  });

  test("missing scopes are an input error", () => {
    expect(() =>
      compareSearchReports(
        searchReport({ "1": { median: 1, p95: 2 }, "8": { median: 1, p95: 2 } }),
        searchReport({ "1": { median: 1, p95: 2 } }),
      ),
    ).toThrow("missing scopes: 8");
  });
});

describe("buildBenchmarkComparisonReport", () => {
  test("aggregates both kinds and carries the revisions", () => {
    const report = buildBenchmarkComparisonReport({
      pythonIndex: indexReport({ cold_start: 1000 }),
      typescriptIndex: indexReport({ cold_start: 1100 }),
      pythonSearch: searchReport({ "1": { median: 10, p95: 20 } }),
      typescriptSearch: searchReport({ "1": { median: 11, p95: 21 } }),
    });
    expect(report.schema_version).toBe(1);
    expect(report.python_revision).toBe("p".repeat(40));
    expect(report.typescript_revision).toBe("p".repeat(40));
    expect(report.passed).toBe(true);
    expect(report.targets).toEqual({ index_within: 0.15, search_within: 0.1 });
  });

  test("one kind failing fails the report", () => {
    const report = buildBenchmarkComparisonReport({
      pythonIndex: indexReport({ cold_start: 1000 }),
      typescriptIndex: indexReport({ cold_start: 2000 }),
    });
    expect(report.search).toBe(null);
    expect(report.passed).toBe(false);
  });

  test("half a pair is rejected", () => {
    expect(() =>
      buildBenchmarkComparisonReport({ pythonIndex: indexReport({ cold_start: 1 }) }),
    ).toThrow("both builds");
    expect(() => buildBenchmarkComparisonReport({})).toThrow("at least one benchmark kind");
  });
});
