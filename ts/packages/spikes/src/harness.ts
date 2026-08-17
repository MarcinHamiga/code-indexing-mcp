/**
 * Shared pass/fail harness for the Phase 0 spikes.
 *
 * Section 6 of the migration plan asks each spike to be "a small throwaway
 * program with a pass/fail written into this document's follow-up". These
 * programs are the throwaway part; this module is what makes the verdicts
 * uniform enough to paste into the follow-up without editing by hand.
 *
 * Deliberately `node:`-only so every spike runs under both runtimes. Several
 * spikes exist precisely to answer "does this work under Bun", and a harness
 * that only ran under Bun could not report the Node column of that answer.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export type Status = "pass" | "fail" | "skip";

export interface CheckRecord {
  readonly name: string;
  readonly status: Status;
  /** What was observed -- the sentence that ends up in the results table. */
  readonly detail: string;
  readonly durationMs: number;
}

export interface SpikeReport {
  readonly spike: string;
  readonly title: string;
  readonly runtime: string;
  readonly runtimeVersion: string;
  readonly platform: string;
  readonly arch: string;
  readonly checks: readonly CheckRecord[];
  readonly verdict: Status;
}

/** A check may return a detail sentence, or throw to fail, or return a skip. */
export type CheckOutcome = string | { readonly skip: string };

const RESULTS_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "results");

function runtimeOf(): { name: string; version: string } {
  const bun = (globalThis as { Bun?: { version?: string } }).Bun;
  return typeof bun?.version === "string"
    ? { name: "bun", version: bun.version }
    : { name: "node", version: process.versions.node };
}

const SYMBOL: Record<Status, string> = {
  pass: "PASS",
  fail: "FAIL",
  skip: "SKIP",
};

export class Spike {
  private readonly checks: CheckRecord[] = [];
  private readonly id: string;
  private readonly title: string;

  // Assigned in the body rather than declared as parameter properties: Node's
  // strip-only TypeScript support rejects those outright, and these programs
  // must run under both runtimes to be worth anything.
  constructor(id: string, title: string) {
    this.id = id;
    this.title = title;
  }

  /**
   * Run one check. A thrown error is a failure, never a crash: a spike that
   * dies on its first missing addon tells us about one module, and the point
   * of S0 is to learn about all of them in a single run.
   */
  async check(name: string, body: () => Promise<CheckOutcome> | CheckOutcome): Promise<void> {
    const started = performance.now();
    try {
      const outcome = await body();
      const durationMs = performance.now() - started;
      if (typeof outcome === "object") {
        this.record({ name, status: "skip", detail: outcome.skip, durationMs });
      } else {
        this.record({ name, status: "pass", detail: outcome, durationMs });
      }
    } catch (error) {
      const durationMs = performance.now() - started;
      this.record({
        name,
        status: "fail",
        detail: describe(error),
        durationMs,
      });
    }
  }

  private record(entry: CheckRecord): void {
    this.checks.push(entry);
    const seconds = (entry.durationMs / 1000).toFixed(2);
    process.stdout.write(`  ${SYMBOL[entry.status]}  ${entry.name}  (${seconds}s)\n`);
    process.stdout.write(`        ${entry.detail}\n`);
  }

  /**
   * Print the verdict, persist the report, and exit non-zero on any failure so
   * the spike can be wired into CI as a gate the day it starts passing.
   */
  finish(): never {
    const runtime = runtimeOf();
    const failed = this.checks.filter((entry) => entry.status === "fail").length;
    const skipped = this.checks.filter((entry) => entry.status === "skip").length;
    const passed = this.checks.length - failed - skipped;
    const verdict: Status = failed > 0 ? "fail" : skipped > 0 ? "skip" : "pass";

    const report: SpikeReport = {
      spike: this.id,
      title: this.title,
      runtime: runtime.name,
      runtimeVersion: runtime.version,
      platform: process.platform,
      arch: process.arch,
      checks: this.checks,
      verdict,
    };

    mkdirSync(RESULTS_DIR, { recursive: true });
    const file = join(
      RESULTS_DIR,
      `${this.id}-${runtime.name}-${process.platform}-${process.arch}.json`,
    );
    writeFileSync(file, `${JSON.stringify(report, null, 2)}\n`);

    process.stdout.write(
      `\n${this.id} verdict: ${SYMBOL[verdict]}  (${passed} passed, ${failed} failed, ${skipped} skipped)\n`,
    );
    process.stdout.write(`report: ${file}\n`);
    process.exit(failed > 0 ? 1 : 0);
  }

  header(): void {
    const runtime = runtimeOf();
    process.stdout.write(
      `\n${this.id} -- ${this.title}\n` +
        `runtime: ${runtime.name} ${runtime.version} on ${process.platform}/${process.arch}\n\n`,
    );
  }
}

export function describe(error: unknown): string {
  if (error instanceof Error) {
    // Native addon load failures put the useful part (the dlopen message, the
    // missing symbol) in the message, and the stack is noise.
    return `${error.name}: ${error.message.split("\n")[0] ?? error.message}`;
  }
  return String(error);
}

/** Repository root, so spikes can read the committed queries and fixtures. */
export function repoRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "..");
}
