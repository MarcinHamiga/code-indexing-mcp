/**
 * S5 -- Watcher and scanner semantics.
 *
 * `@parcel/watcher` event coalescing vs `watchfiles`, and `git ls-files`
 * streaming through a spawned child.
 *
 * The auto-indexing monitor in `server.py` is built on watchfiles' contract:
 * a debounced batch of (change, path) pairs, deduplicated within the window.
 * What matters for the port is not that the two libraries agree event for
 * event -- they will not -- but that the properties the monitor relies on hold:
 * recursive coverage, coalescing of a burst into few events, moves visible as
 * delete plus create, and an ignore mechanism that keeps `.git` churn out of
 * the stream.
 *
 * The scanner half checks that `git ls-files -z` can be consumed as a stream
 * of NUL-delimited paths rather than one buffered string, because
 * `scanner.py` deliberately streams it: a monorepo's file list is large enough
 * that buffering it whole is a memory decision, not a style one.
 */

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Spike, repoRoot } from "./harness.ts";

const spike = new Spike("s5", "Watcher and scanner semantics");
spike.header();

const scratch = mkdtempSync(join(tmpdir(), "ci-mcp-s5-"));
const watcher = await import("@parcel/watcher");

interface Observed {
  readonly type: string;
  readonly path: string;
}

/**
 * Subscribe, run `act`, and collect every event seen in `settleMs`.
 *
 * The settle window is what makes coalescing observable: returning on the
 * first event would measure latency instead.
 */
async function observe(
  directory: string,
  act: () => void,
  options: { settleMs?: number; ignore?: string[] } = {},
): Promise<Observed[]> {
  const events: Observed[] = [];
  const subscription = await watcher.subscribe(
    directory,
    (error, batch) => {
      if (error) return;
      for (const event of batch) events.push({ type: event.type, path: event.path });
    },
    options.ignore === undefined ? undefined : { ignore: options.ignore },
  );
  try {
    act();
    await new Promise((resolve) => setTimeout(resolve, options.settleMs ?? 600));
  } finally {
    await subscription.unsubscribe();
  }
  return events;
}

await spike.check("recursive coverage reaches an existing nested directory", async () => {
  // The property the monitor actually depends on: a write anywhere in the
  // established tree is seen. The directory exists and is being watched before
  // the write happens, which is the ordinary steady state.
  const root = join(scratch, "recursive");
  const nested = join(root, "a", "b", "c");
  mkdirSync(nested, { recursive: true });
  await new Promise((resolve) => setTimeout(resolve, 250));

  const events = await observe(root, () => {
    writeFileSync(join(nested, "deep.py"), "value = 1\n");
  });
  const deep = events.filter((event) => event.path.endsWith("deep.py"));
  if (deep.length === 0) {
    throw new Error(`no event for a file three levels down; saw ${events.length} events`);
  }
  return `${events.length} events, including ${deep[0]?.type} for the nested file`;
});

await spike.check("a directory created and populated in one burst", async () => {
  // Recorded rather than asserted, because the answer is platform-dependent
  // and the plan needs the fact, not a green tick.
  //
  // macOS watches a tree through FSEvents, so a file written into a
  // just-created directory is seen. Linux inotify has no recursive mode:
  // @parcel/watcher must add a watch per directory as it learns of them, and a
  // file created inside the window between mkdir and watch-add is missed
  // outright. watchfiles carries the identical constraint, so this is not a
  // regression against the Python build.
  //
  // It is a latency issue rather than a correctness one:
  // `application.py::_project_is_stale` rescans the tree and compares the whole
  // path set on the next query, so a dropped event delays proactive
  // auto-indexing but cannot leave a file permanently unindexed.
  const root = join(scratch, "burst-create");
  mkdirSync(root, { recursive: true });
  await new Promise((resolve) => setTimeout(resolve, 250));

  const events = await observe(root, () => {
    mkdirSync(join(root, "a", "b", "c"), { recursive: true });
    writeFileSync(join(root, "a", "b", "c", "deep.py"), "value = 1\n");
  });
  const deep = events.filter((event) => event.path.endsWith("deep.py"));
  return deep.length > 0
    ? `the nested file was seen (${deep[0]?.type}) -- no watch-registration race here`
    : `the nested file was MISSED (${events.length} other events) -- ` +
        `new directories race the watch registration on this platform; ` +
        `proactive indexing waits for the next lazy freshness check`;
});

await spike.check("a burst of writes coalesces", async () => {
  const root = join(scratch, "coalesce");
  mkdirSync(root, { recursive: true });
  const target = join(root, "hot.py");
  writeFileSync(target, "0\n");

  const events = await observe(root, () => {
    // watchfiles debounces on a 50ms window by default; this is the same
    // shape of burst the auto-indexer sees when a formatter rewrites a file.
    for (let index = 0; index < 50; index += 1) {
      writeFileSync(target, `value = ${index}\n`);
    }
  });
  const forTarget = events.filter((event) => event.path.endsWith("hot.py"));
  if (forTarget.length === 0) throw new Error("50 writes produced no event at all");
  if (forTarget.length > 10) {
    throw new Error(`50 writes produced ${forTarget.length} events -- no useful coalescing`);
  }
  return `50 writes -> ${forTarget.length} event(s) (${[...new Set(forTarget.map((e) => e.type))].join(", ")})`;
});

await spike.check("a rename surfaces as delete plus create", async () => {
  const root = join(scratch, "rename");
  mkdirSync(root, { recursive: true });
  const before = join(root, "before.py");
  writeFileSync(before, "x = 1\n");
  await new Promise((resolve) => setTimeout(resolve, 200));

  const events = await observe(root, () => {
    renameSync(before, join(root, "after.py"));
  });
  const kinds = new Map<string, string>();
  for (const event of events) {
    if (event.path.endsWith("before.py")) kinds.set("before", event.type);
    if (event.path.endsWith("after.py")) kinds.set("after", event.type);
  }
  if (!kinds.has("before") || !kinds.has("after")) {
    throw new Error(
      `rename reported ${JSON.stringify(events.map((e) => `${e.type}:${e.path.split("/").pop()}`))}`,
    );
  }
  return `before.py -> ${kinds.get("before")}, after.py -> ${kinds.get("after")}`;
});

await spike.check("ignore patterns keep churn out of the stream", async () => {
  const root = join(scratch, "ignored");
  mkdirSync(join(root, "node_modules"), { recursive: true });
  // Let the directory creation drain before subscribing: an event already in
  // flight is delivered regardless of the ignore list, and reading that as a
  // leak would be a bug in the spike rather than in the watcher.
  await new Promise((resolve) => setTimeout(resolve, 250));

  const events = await observe(
    root,
    () => {
      writeFileSync(join(root, "node_modules", "noise.js"), "noise\n");
      writeFileSync(join(root, "real.py"), "value = 1\n");
    },
    { ignore: ["node_modules"] },
  );
  const leaked = events.filter((event) => event.path.includes("node_modules"));
  if (leaked.length > 0) {
    throw new Error(`${leaked.length} ignored-path events leaked through`);
  }
  const real = events.filter((event) => event.path.endsWith("real.py"));
  if (real.length === 0) throw new Error("the ignore pattern suppressed the tracked file too");
  return `node_modules suppressed, ${real.length} event(s) for the tracked file`;
});

await spike.check("ignore entries are names and paths, not globs", async () => {
  // Worth pinning down rather than assuming: `ignore` looks glob-shaped, and a
  // pattern that silently matches nothing would put every `node_modules` write
  // back into the auto-indexer's event stream without any error to notice.
  const root = join(scratch, "ignore-globs");
  mkdirSync(join(root, "node_modules"), { recursive: true });
  await new Promise((resolve) => setTimeout(resolve, 250));

  const observedFor = async (pattern: string): Promise<number> => {
    const events = await observe(
      root,
      () => writeFileSync(join(root, "node_modules", `noise-${pattern.length}.js`), "noise\n"),
      { ignore: [pattern] },
    );
    return events.filter((event) => event.path.includes("node_modules")).length;
  };

  const bare = await observedFor("node_modules");
  const globbed = await observedFor("**/node_modules/**");
  if (bare > 0) throw new Error("a bare directory name failed to suppress anything");
  const note = globbed > 0 ? "glob form does NOT suppress" : "glob form also suppresses";
  return `bare name suppresses (${bare} leaked); ${note} (${globbed} leaked)`;
});

await spike.check("git ls-files streams NUL-delimited paths", async () => {
  const root = repoRoot();
  const paths: string[] = [];
  let pending = "";

  await new Promise<void>((resolve, reject) => {
    // `-z` because paths may contain anything but NUL; splitting on newline is
    // what scanner.py deliberately avoids.
    const child = spawn("git", ["ls-files", "-z"], {
      cwd: root,
      stdio: ["ignore", "pipe", "pipe"],
    });
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (piece: string) => {
      pending += piece;
      const parts = pending.split("\0");
      // The final element is either an empty string or a partial path that the
      // next chunk completes -- carrying it forward is the whole point of
      // streaming rather than buffering.
      pending = parts.pop() ?? "";
      for (const part of parts) if (part !== "") paths.push(part);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (pending !== "") paths.push(pending);
      code === 0 ? resolve() : reject(new Error(`git ls-files exited ${code}`));
    });
  });

  if (paths.length === 0) throw new Error("git ls-files produced nothing");
  const suspicious = paths.filter((path) => path.includes("\n") || path === "");
  if (suspicious.length > 0) throw new Error(`${suspicious.length} malformed paths parsed`);
  return `streamed ${paths.length} tracked paths (e.g. ${paths[0]})`;
});

rmSync(scratch, { recursive: true, force: true });
spike.finish();
