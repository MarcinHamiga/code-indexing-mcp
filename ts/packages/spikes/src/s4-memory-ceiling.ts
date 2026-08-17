/**
 * S4 -- Memory ceiling.
 *
 * Reproduce the token-batching memory model's enforcement mechanism in the TS
 * runtime: poll a child's RSS, kill it at the ceiling, retry the batch smaller.
 *
 * The risk this retires is the register's "`worker_threads`/child RSS
 * accounting differs from `psutil` semantics". `embedding_worker.py` does not
 * merely want *a* memory number -- it wants the same number `psutil` reports,
 * because the ceilings in the settings were calibrated against psutil's
 * definition of RSS. So the third check measures one child with both
 * `pidusage` and `psutil` at the same moment and compares them directly,
 * rather than assuming two libraries mean the same thing by "RSS".
 *
 * Comparing *peak* RSS at the fixture scales `test_memory_acceptance.py` uses
 * is left to Phase 4, when there is a real embedding worker to measure; that
 * comparison is about the model's allocation behavior, not about whether the
 * enforcement mechanism works, which is what Phase 0 needs to know.
 */

import { spawn } from "node:child_process";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import pidusage from "pidusage";
import { Spike, repoRoot } from "./harness.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const CHILD = join(HERE, "s4-child.ts");
const MEGABYTE = 1024 * 1024;

const spike = new Spike("s4", "Memory ceiling");
spike.header();

interface ChildHandle {
  readonly pid: number;
  readonly kill: () => void;
  readonly exited: Promise<{ code: number | null; signal: string | null }>;
}

/** Start the allocator child under the same runtime as this process. */
function startChild(targetMb: number, stepMb: number, delayMs: number, holdMs = 0): ChildHandle {
  const runtime = process.execPath;
  const child = spawn(
    runtime,
    [CHILD, String(targetMb), String(stepMb), String(delayMs), String(holdMs)],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  if (child.pid === undefined) throw new Error("the child did not start");
  return {
    pid: child.pid,
    kill: () => child.kill("SIGKILL"),
    exited: new Promise((resolve) => {
      child.on("close", (code, signal) => resolve({ code, signal }));
    }),
  };
}

/** Poll until `predicate` accepts a sample, returning the samples taken. */
async function pollRss(
  pid: number,
  predicate: (rssMb: number) => boolean,
  timeoutMs: number,
): Promise<number[]> {
  const samples: number[] = [];
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    let rssMb: number;
    try {
      rssMb = (await pidusage(pid)).memory / MEGABYTE;
    } catch {
      break; // the process exited underneath the poll
    }
    samples.push(rssMb);
    if (predicate(rssMb)) return samples;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  return samples;
}

await spike.check("pidusage tracks a child's growing resident set", async () => {
  const child = startChild(192, 16, 30);
  const samples = await pollRss(child.pid, (rss) => rss > 120, 20_000);
  child.kill();
  await child.exited;

  if (samples.length < 3) throw new Error(`only ${samples.length} samples taken`);
  const first = samples[0] ?? 0;
  const peak = Math.max(...samples);
  if (peak <= first) throw new Error(`RSS never grew: started ${first}, peaked ${peak}`);
  return `${samples.length} samples, ${first.toFixed(0)} MB -> ${peak.toFixed(0)} MB`;
});

await spike.check("a ceiling kills the child before it reaches its target", async () => {
  const ceilingMb = 150;
  // The child would climb to 512 MB unimpeded; the ceiling must stop it well
  // short, the way the worker's own ceiling stops a batch that is too large.
  const child = startChild(512, 16, 30);
  const samples = await pollRss(child.pid, (rss) => rss >= ceilingMb, 30_000);
  const observed = Math.max(...samples, 0);
  const tripped = observed >= ceilingMb;
  child.kill();
  const outcome = await child.exited;

  if (!tripped) throw new Error(`never reached the ${ceilingMb} MB ceiling (peaked ${observed})`);
  if (outcome.signal !== "SIGKILL" && outcome.code === 0) {
    throw new Error("the child ran to completion instead of being killed at the ceiling");
  }
  return `tripped at ${observed.toFixed(0)} MB (ceiling ${ceilingMb} MB), child ended with ${outcome.signal ?? `code ${outcome.code}`}`;
});

await spike.check("a halved retry completes under the same ceiling", async () => {
  const ceilingMb = 200;
  let targetMb = 512;
  let attempts = 0;

  // `embedding_worker.py`'s retry shape: on a ceiling kill, halve the batch and
  // try again, until it fits or there is nothing left to halve.
  while (targetMb >= 16) {
    attempts += 1;
    const child = startChild(targetMb, 16, 15);
    const samples = await pollRss(child.pid, (rss) => rss >= ceilingMb, 60_000);
    const exceeded = Math.max(...samples, 0) >= ceilingMb;
    if (exceeded) {
      child.kill();
      await child.exited;
      targetMb = Math.floor(targetMb / 2);
      continue;
    }
    const outcome = await child.exited;
    if (outcome.code === 0) {
      return `completed at ${targetMb} MB after ${attempts} attempt(s) under a ${ceilingMb} MB ceiling`;
    }
    throw new Error(`child failed with code ${outcome.code} at ${targetMb} MB`);
  }
  throw new Error("halving exhausted without fitting under the ceiling");
});

await spike.check("pidusage RSS agrees with psutil RSS on the same process", async () => {
  // The ceilings in settings.py were tuned against psutil's number. If these
  // two disagree materially, the ported ceilings are wrong by that margin.
  const python = join(repoRoot(), ".venv", "bin", "python");
  if (!existsSync(python)) {
    return {
      skip: `no interpreter at ${python}; run the project's own env to compare`,
    };
  }
  try {
    execFileSync(python, ["-c", "import psutil"], { stdio: "ignore" });
  } catch {
    return { skip: "psutil is not installed in the project environment" };
  }

  // Allocate, then hold: both readings are taken while the child is idle at a
  // stable resident set, so any difference is the metric and not the clock.
  const child = startChild(256, 16, 20, 4_000);
  await pollRss(child.pid, (rss) => rss > 240, 20_000);
  await new Promise((resolve) => setTimeout(resolve, 300));

  const fromPidusage = (await pidusage(child.pid)).memory / MEGABYTE;
  const fromPsutil =
    Number(
      execFileSync(
        python,
        ["-c", `import psutil; print(psutil.Process(${child.pid}).memory_info().rss)`],
        { encoding: "utf8" },
      ).trim(),
    ) / MEGABYTE;

  child.kill();
  await child.exited;

  const difference = Math.abs(fromPidusage - fromPsutil);
  const relative = difference / Math.max(fromPsutil, 1);
  // Sampled through two different mechanisms on an idle process, so the bar is
  // tight: anything beyond a few percent means they disagree about what RSS is,
  // and every ceiling in settings.py would need re-tuning by that margin.
  if (relative > 0.05) {
    throw new Error(
      `pidusage reported ${fromPidusage.toFixed(1)} MB, psutil ${fromPsutil.toFixed(1)} MB ` +
        `(${(relative * 100).toFixed(1)}% apart) -- different RSS semantics`,
    );
  }
  return (
    `pidusage ${fromPidusage.toFixed(1)} MB vs psutil ${fromPsutil.toFixed(1)} MB ` +
    `(${(relative * 100).toFixed(2)}% apart on a quiesced child)`
  );
});

spike.finish();
