/**
 * Cross-build benchmark comparison: the Phase 9 performance gate of the
 * migration plan. §8 bounds the TypeScript build's regression against the
 * Python one -- within 15% on index time, no regression on search latency --
 * and §11 brings that comparison into CI.
 *
 * Both CLIs print benchmark reports with identical JSON shapes, so this
 * module compares two parsed reports and evaluates those gates. The index
 * gate uses each scenario's `reported_duration_ms` (the indexer's own
 * pipeline clock) rather than `wall_ms`, which would charge the comparison
 * process startup and one-time model warmup differences to indexing;
 * scenarios whose Python duration is below `minimumMs` are recorded but not
 * gated, because 15% of a near-zero duration is scheduler noise.
 */

export const DEFAULT_INDEX_TOLERANCE = 0.15;
export const DEFAULT_SEARCH_TOLERANCE = 0.1;
export const DEFAULT_INDEX_MINIMUM_MS = 50;

type Report = Record<string, unknown>;

export interface IndexScenarioComparison {
  scenario: string;
  /** Null for scenarios that report no pipeline duration (post_maintenance). */
  python_reported_ms: number | null;
  typescript_reported_ms: number | null;
  /** `typescript / python`; null when the Python duration is zero or absent. */
  ratio: number | null;
  /** False when the Python duration is below the gate's minimum or absent. */
  gated: boolean;
  within_tolerance: boolean;
}

export interface IndexBenchmarkComparison {
  tolerance: number;
  minimum_ms: number;
  scenarios: IndexScenarioComparison[];
  passed: boolean;
}

export interface SearchMetricComparison {
  scope: string;
  metric: "median_ms" | "p95_ms";
  python_ms: number;
  typescript_ms: number;
  ratio: number | null;
  within_tolerance: boolean;
}

export interface SearchBenchmarkComparison {
  tolerance: number;
  scopes: SearchMetricComparison[];
  passed: boolean;
}

export interface BenchmarkComparisonReport {
  schema_version: 1;
  python_revision: string | null;
  typescript_revision: string | null;
  targets: { index_within: number; search_within: number };
  index: IndexBenchmarkComparison | null;
  search: SearchBenchmarkComparison | null;
  passed: boolean;
}

function scenarioDurations(report: Report): Map<string, number | null> {
  const scenarios = report.scenarios;
  if (scenarios === undefined || typeof scenarios !== "object" || scenarios === null) {
    throw new Error("benchmark report has no scenarios object");
  }
  const durations = new Map<string, number | null>();
  for (const [name, value] of Object.entries(scenarios)) {
    // `post_maintenance` times a maintenance pass, not an index run, so it
    // reports no `reported_duration_ms`; it stays informational.
    const duration = (value as Report).reported_duration_ms;
    durations.set(name, typeof duration === "number" ? duration : null);
  }
  return durations;
}

function requireScenarios(
  python: Map<string, number | null>,
  typescript: Map<string, number | null>,
): void {
  const missing = [...python.keys()].filter((name) => !typescript.has(name));
  if (missing.length > 0) {
    throw new Error(`TypeScript index report is missing scenarios: ${missing.join(", ")}`);
  }
  const extra = [...typescript.keys()].filter((name) => !python.has(name));
  if (extra.length > 0) {
    throw new Error(`TypeScript index report has unknown scenarios: ${extra.join(", ")}`);
  }
}

export function compareIndexReports(
  python: Report,
  typescript: Report,
  {
    tolerance = DEFAULT_INDEX_TOLERANCE,
    minimumMs = DEFAULT_INDEX_MINIMUM_MS,
  }: { tolerance?: number; minimumMs?: number } = {},
): IndexBenchmarkComparison {
  const pythonDurations = scenarioDurations(python);
  const typescriptDurations = scenarioDurations(typescript);
  requireScenarios(pythonDurations, typescriptDurations);
  const scenarios: IndexScenarioComparison[] = [...pythonDurations.entries()].map(
    ([name, pythonMs]) => {
      const typescriptMs = typescriptDurations.get(name) ?? null;
      const gated = pythonMs !== null && pythonMs >= minimumMs && typescriptMs !== null;
      const ratio = pythonMs !== null && pythonMs > 0 ? (typescriptMs ?? 0) / pythonMs : null;
      return {
        scenario: name,
        python_reported_ms: pythonMs,
        typescript_reported_ms: typescriptMs,
        ratio,
        gated,
        within_tolerance: !gated || (ratio !== null && ratio <= 1 + tolerance),
      };
    },
  );
  return {
    tolerance,
    minimum_ms: minimumMs,
    scenarios,
    passed: scenarios.every((scenario) => scenario.within_tolerance),
  };
}

function searchScopeLatencies(report: Report): Map<string, { median_ms: number; p95_ms: number }> {
  const scopes = report.scopes;
  if (scopes === undefined || typeof scopes !== "object" || scopes === null) {
    throw new Error("benchmark report has no scopes object");
  }
  const latencies = new Map<string, { median_ms: number; p95_ms: number }>();
  for (const [name, value] of Object.entries(scopes)) {
    const latency = (value as Report).latency_ms as Report | undefined;
    const median = latency?.median_ms;
    const p95 = latency?.p95_ms;
    if (typeof median !== "number" || typeof p95 !== "number") {
      throw new Error(`search scope ${name} has no numeric median_ms/p95_ms`);
    }
    latencies.set(name, { median_ms: median, p95_ms: p95 });
  }
  return latencies;
}

export function compareSearchReports(
  python: Report,
  typescript: Report,
  { tolerance = DEFAULT_SEARCH_TOLERANCE }: { tolerance?: number } = {},
): SearchBenchmarkComparison {
  const pythonScopes = searchScopeLatencies(python);
  const typescriptScopes = searchScopeLatencies(typescript);
  const missing = [...pythonScopes.keys()].filter((name) => !typescriptScopes.has(name));
  if (missing.length > 0) {
    throw new Error(`TypeScript search report is missing scopes: ${missing.join(", ")}`);
  }
  const scopes: SearchMetricComparison[] = [];
  for (const [name, pythonLatency] of pythonScopes) {
    const typescriptLatency = typescriptScopes.get(name);
    if (typescriptLatency === undefined) continue;
    for (const metric of ["median_ms", "p95_ms"] as const) {
      const pythonMs = pythonLatency[metric];
      const typescriptMs = typescriptLatency[metric];
      const ratio = pythonMs > 0 ? typescriptMs / pythonMs : null;
      scopes.push({
        scope: name,
        metric,
        python_ms: pythonMs,
        typescript_ms: typescriptMs,
        ratio,
        // A zero Python latency is a floor effect, not a target: gate only on
        // the direction, so an equally zero-cost TypeScript run passes and a
        // measurable one is flagged rather than charged a bogus ratio.
        within_tolerance: ratio !== null ? ratio <= 1 + tolerance : typescriptMs === 0,
      });
    }
  }
  return { tolerance, scopes, passed: scopes.every((scope) => scope.within_tolerance) };
}

function revision(report: Report): string | null {
  const value = report.revision;
  return typeof value === "string" ? value : null;
}

export function buildBenchmarkComparisonReport({
  pythonIndex,
  typescriptIndex,
  pythonSearch,
  typescriptSearch,
  indexTolerance = DEFAULT_INDEX_TOLERANCE,
  searchTolerance = DEFAULT_SEARCH_TOLERANCE,
  indexMinimumMs = DEFAULT_INDEX_MINIMUM_MS,
}: {
  pythonIndex?: Report | undefined;
  typescriptIndex?: Report | undefined;
  pythonSearch?: Report | undefined;
  typescriptSearch?: Report | undefined;
  indexTolerance?: number;
  searchTolerance?: number;
  indexMinimumMs?: number;
}): BenchmarkComparisonReport {
  if (
    (pythonIndex === undefined) !== (typescriptIndex === undefined) ||
    (pythonSearch === undefined) !== (typescriptSearch === undefined)
  ) {
    throw new Error("each benchmark kind needs both builds' reports");
  }
  const index =
    pythonIndex !== undefined && typescriptIndex !== undefined
      ? compareIndexReports(pythonIndex, typescriptIndex, {
          tolerance: indexTolerance,
          minimumMs: indexMinimumMs,
        })
      : null;
  const search =
    pythonSearch !== undefined && typescriptSearch !== undefined
      ? compareSearchReports(pythonSearch, typescriptSearch, { tolerance: searchTolerance })
      : null;
  if (index === null && search === null) {
    throw new Error("at least one benchmark kind must be compared");
  }
  return {
    schema_version: 1,
    python_revision: revision(pythonIndex ?? pythonSearch ?? {}),
    typescript_revision: revision(typescriptIndex ?? typescriptSearch ?? {}),
    targets: { index_within: indexTolerance, search_within: searchTolerance },
    index,
    search,
    passed: (index?.passed ?? true) && (search?.passed ?? true),
  };
}
