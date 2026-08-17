/**
 * The disposable child S4 measures and kills.
 *
 * Stands in for `embedding_worker.py`: it grows its resident set in visible
 * steps and reports progress on stdout, so the parent's polling loop has
 * something with a known shape to observe. Allocation is deliberately not
 * releasable -- buffers are retained and touched -- because an allocation the
 * collector can reclaim would measure the GC rather than the ceiling.
 *
 * Arguments: <target-megabytes> <step-megabytes> <step-delay-ms> [hold-ms]
 */

const [targetArgument, stepArgument, delayArgument, holdArgument] = process.argv.slice(2);
const targetMb = Number(targetArgument ?? 256);
const stepMb = Number(stepArgument ?? 16);
const delayMs = Number(delayArgument ?? 40);
const holdMs = Number(holdArgument ?? 0);

const retained: Uint8Array[] = [];

async function main(): Promise<void> {
  let allocatedMb = 0;
  while (allocatedMb < targetMb) {
    const block = new Uint8Array(stepMb * 1024 * 1024);
    // Touch every page: an untouched allocation may never become resident, and
    // resident set size is exactly what the parent is polling.
    for (let offset = 0; offset < block.length; offset += 4096) {
      block[offset] = 1;
    }
    retained.push(block);
    allocatedMb += stepMb;
    process.stdout.write(`${allocatedMb}\n`);
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  process.stdout.write("done\n");
  // Hold the allocation briefly so the parent can sample a process that is no
  // longer moving. Comparing two RSS readings taken milliseconds apart on a
  // still-growing process measures the delay between them, not the metric.
  await new Promise((resolve) => setTimeout(resolve, holdMs));
}

await main();
