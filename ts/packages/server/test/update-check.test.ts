/** The throttled update notice for managed installations. */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import {
  CACHE_FILENAME,
  CHECK_INTERVAL_SECONDS,
  checkoutHead,
  checkRemote,
  DISABLE_VARIABLE,
  type GitRunner,
  installContext,
  notice,
  readCache,
  refreshIfDue,
  startBackgroundRefresh,
  updateAvailable,
  type UpdateStatus,
  writeCache,
} from "../src/update-check.ts";
import { resolvePath } from "../src/paths.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

const LOCAL_SHA = "1111111aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const REMOTE_SHA = "2222222bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

let root: string;
let cache: string;

beforeEach(() => {
  root = temporaryDirectory();
  cache = path.join(root, "cache");
});

afterEach(() => {
  removeDirectory(root);
});

/** A directory that looks like a git checkout, without invoking git. */
function checkout(
  location: string,
  { head = "ref: refs/heads/main\n", ref = LOCAL_SHA as string | null } = {},
): string {
  const git = path.join(location, ".git");
  fs.mkdirSync(git, { recursive: true });
  fs.writeFileSync(path.join(git, "HEAD"), head, "utf8");
  if (ref !== null) {
    fs.mkdirSync(path.join(git, "refs", "heads"), { recursive: true });
    fs.writeFileSync(path.join(git, "refs", "heads", "main"), `${ref}\n`, "utf8");
  }
  return location;
}

/** Make `<root>/install` look like the install this process was loaded from. */
function managed(): {
  installDirectory: string;
  environment: Record<string, string>;
  runtimeRoot: string;
} {
  const installDirectory = checkout(path.join(root, "install"));
  return {
    installDirectory,
    environment: { CODE_INDEXING_MCP_INSTALL_DIR: installDirectory },
    // Python asked whether `sys.prefix` sat inside the install; the equivalent
    // question here is whether the running code does.
    runtimeRoot: path.join(installDirectory, "packages", "server", "src"),
  };
}

function runner({
  sha = REMOTE_SHA,
  error = null,
  calls,
}: {
  sha?: string;
  error?: Error | null;
  calls?: unknown[];
} = {}): GitRunner {
  return async (command, cwd, timeoutSeconds) => {
    calls?.push([[...command], cwd, timeoutSeconds]);
    if (error !== null) throw error;
    return { stdout: `${sha}\trefs/heads/main\n` };
  };
}

describe("install context", () => {
  test("returns the directory of a managed install", () => {
    const { installDirectory, environment, runtimeRoot } = managed();

    expect(installContext({ environment, runtimeRoot })).toBe(resolvePath(installDirectory));
  });

  test("is null without a git directory", () => {
    const installDirectory = path.join(root, "install");
    fs.mkdirSync(installDirectory);

    expect(
      installContext({
        environment: { CODE_INDEXING_MCP_INSTALL_DIR: installDirectory },
        runtimeRoot: path.join(installDirectory, "src"),
      }),
    ).toBeNull();
  });

  test("is null when the running code lives elsewhere", () => {
    const { environment } = managed();

    expect(
      installContext({ environment, runtimeRoot: path.join(root, "elsewhere", "src") }),
    ).toBeNull();
  });
});

describe("head reading", () => {
  test("reads a detached head", () => {
    const repo = checkout(path.join(root, "repo"), { head: `${LOCAL_SHA}\n`, ref: null });

    expect(checkoutHead(repo)).toBe(LOCAL_SHA);
  });

  test("follows a symbolic ref", () => {
    expect(checkoutHead(checkout(path.join(root, "repo")))).toBe(LOCAL_SHA);
  });

  test("falls back to packed refs", () => {
    const repo = checkout(path.join(root, "repo"), { ref: null });
    fs.writeFileSync(
      path.join(repo, ".git", "packed-refs"),
      `# pack-refs with: peeled fully-peeled sorted\n${LOCAL_SHA} refs/heads/main\n`,
      "utf8",
    );

    expect(checkoutHead(repo)).toBe(LOCAL_SHA);
  });

  test("follows a worktree's gitdir file", () => {
    const real = checkout(path.join(root, "repo"));
    const worktree = path.join(root, "worktree");
    fs.mkdirSync(worktree);
    fs.writeFileSync(path.join(worktree, ".git"), `gitdir: ${path.join(real, ".git")}\n`, "utf8");

    expect(checkoutHead(worktree)).toBe(LOCAL_SHA);
  });

  test("is null without a checkout", () => {
    expect(checkoutHead(root)).toBeNull();
  });
});

describe("the cache", () => {
  test("round-trips through a write", () => {
    const nested = path.join(cache, "nested");
    const status: UpdateStatus = {
      checkedAt: 1000,
      localSha: LOCAL_SHA,
      remoteSha: REMOTE_SHA,
    };

    writeCache(nested, status);

    expect(readCache(nested)).toEqual(status);
    expect(fs.readdirSync(nested)).toEqual([CACHE_FILENAME]);
  });

  test("is written with the snake-case keys the Python build reads", () => {
    writeCache(cache, { checkedAt: 1000, localSha: LOCAL_SHA, remoteSha: REMOTE_SHA });

    expect(JSON.parse(fs.readFileSync(path.join(cache, CACHE_FILENAME), "utf8"))).toEqual({
      checked_at: 1000,
      local_sha: LOCAL_SHA,
      remote_sha: REMOTE_SHA,
      schema_version: 1,
    });
  });

  test.each([
    ["not json at all", "not json at all"],
    ["a record from another schema version", JSON.stringify({ schema_version: 99, checked_at: 1 })],
    ["a list", JSON.stringify([])],
    ["a record missing its fields", JSON.stringify({ schema_version: 1 })],
  ])("treats %s as absent", (_label, payload) => {
    fs.mkdirSync(cache, { recursive: true });
    fs.writeFileSync(path.join(cache, CACHE_FILENAME), payload, "utf8");

    expect(readCache(cache)).toBeNull();
  });

  test("is null when nothing was written", () => {
    expect(readCache(cache)).toBeNull();
  });
});

describe("the remote check", () => {
  test("asks origin for the main branch", async () => {
    const repo = checkout(path.join(root, "repo"));
    const calls: unknown[] = [];

    const status = await checkRemote(repo, {
      timeoutSeconds: 3,
      runCommand: runner({ calls }),
    });

    expect(calls).toEqual([[["git", "ls-remote", "origin", "refs/heads/main"], repo, 3]]);
    expect(status.localSha).toBe(LOCAL_SHA);
    expect(status.remoteSha).toBe(REMOTE_SHA);
    expect(updateAvailable(status)).toBe(true);
  });

  test("rejects when the remote has no such branch", async () => {
    const repo = checkout(path.join(root, "repo"));

    await expect(checkRemote(repo, { runCommand: async () => ({ stdout: "\n" }) })).rejects.toThrow(
      /no remote branch/,
    );
  });
});

describe("refreshing", () => {
  test("writes a status when no cache exists", async () => {
    const repo = checkout(path.join(root, "repo"));

    await refreshIfDue(repo, cache, { now: 1000, runCommand: runner(), environment: {} });

    expect(readCache(cache)?.remoteSha).toBe(REMOTE_SHA);
  });

  test("honours the throttle", async () => {
    const repo = checkout(path.join(root, "repo"));
    writeCache(cache, { checkedAt: 990, localSha: "a", remoteSha: "a" });
    const calls: unknown[] = [];

    await refreshIfDue(repo, cache, {
      now: 1000,
      runCommand: runner({ calls }),
      environment: {},
    });

    expect(calls).toEqual([]);
    expect(readCache(cache)?.checkedAt).toBe(990);
  });

  test("rechecks once the interval expired", async () => {
    const repo = checkout(path.join(root, "repo"));
    writeCache(cache, { checkedAt: 1000, localSha: "a", remoteSha: "a" });

    await refreshIfDue(repo, cache, {
      now: 1000 + CHECK_INTERVAL_SECONDS + 1,
      runCommand: runner(),
      environment: {},
    });

    expect(readCache(cache)?.remoteSha).toBe(REMOTE_SHA);
  });

  test.each(["off", "OFF", "0", "false", "No", " off "])(
    "is disabled by %s in the environment",
    async (value) => {
      const repo = checkout(path.join(root, "repo"));
      const calls: unknown[] = [];

      await refreshIfDue(repo, cache, {
        now: 1000,
        runCommand: runner({ calls }),
        environment: { [DISABLE_VARIABLE]: value },
      });

      expect(calls).toEqual([]);
      expect(fs.existsSync(cache)).toBe(false);
    },
  );

  test("stays silent when git fails", async () => {
    const repo = checkout(path.join(root, "repo"));

    await refreshIfDue(repo, cache, {
      now: 1000,
      runCommand: runner({ error: new Error("spawn git ENOENT") }),
      environment: {},
    });

    expect(readCache(cache)).toBeNull();
  });

  test("overwrites a corrupt cache", async () => {
    const repo = checkout(path.join(root, "repo"));
    fs.mkdirSync(cache, { recursive: true });
    fs.writeFileSync(path.join(cache, CACHE_FILENAME), "{not json", "utf8");

    await refreshIfDue(repo, cache, { now: 1000, runCommand: runner(), environment: {} });

    expect(readCache(cache)?.remoteSha).toBe(REMOTE_SHA);
  });
});

describe("the background refresh", () => {
  test("is null when disabled", () => {
    const { environment, runtimeRoot } = managed();

    expect(
      startBackgroundRefresh(cache, {
        environment: { ...environment, [DISABLE_VARIABLE]: "off" },
        runtimeRoot,
      }),
    ).toBeNull();
  });

  test("is null without a managed install", () => {
    expect(
      startBackgroundRefresh(cache, {
        environment: { CODE_INDEXING_MCP_INSTALL_DIR: path.join(root, "missing") },
        runtimeRoot: root,
      }),
    ).toBeNull();
  });

  test("is null when the cache is fresh", () => {
    const { environment, runtimeRoot } = managed();
    writeCache(cache, {
      checkedAt: Date.now() / 1000,
      localSha: LOCAL_SHA,
      remoteSha: LOCAL_SHA,
    });

    expect(startBackgroundRefresh(cache, { environment, runtimeRoot })).toBeNull();
  });

  test("runs off the hot path when due, and swallows its own failure", async () => {
    const { environment, runtimeRoot } = managed();

    const pending = startBackgroundRefresh(cache, { environment, runtimeRoot });

    expect(pending).not.toBeNull();
    // The fake checkout has no "origin", so the real git call fails -- silently.
    await expect(pending).resolves.toBeUndefined();
  });
});

describe("the notice", () => {
  test("is silent when the live head matches the cached remote", () => {
    const { environment, runtimeRoot } = managed();
    writeCache(cache, { checkedAt: 1000, localSha: "0".repeat(40), remoteSha: LOCAL_SHA });

    expect(notice(cache, { environment, runtimeRoot })).toBeNull();
  });

  test("reports a newer remote", () => {
    const { environment, runtimeRoot } = managed();
    writeCache(cache, { checkedAt: 1000, localSha: LOCAL_SHA, remoteSha: REMOTE_SHA });

    expect(notice(cache, { environment, runtimeRoot })).toBe(
      `A code-indexing-mcp update is available (${LOCAL_SHA.slice(0, 7)} -> ` +
        `${REMOTE_SHA.slice(0, 7)}). Run: code-indexing-mcp update`,
    );
  });

  test("is silent without a cache", () => {
    const { environment, runtimeRoot } = managed();

    expect(notice(cache, { environment, runtimeRoot })).toBeNull();
  });

  test("is silent without a managed install", () => {
    writeCache(cache, { checkedAt: 1000, localSha: LOCAL_SHA, remoteSha: REMOTE_SHA });

    expect(
      notice(cache, {
        environment: { CODE_INDEXING_MCP_INSTALL_DIR: path.join(root, "missing") },
        runtimeRoot: root,
      }),
    ).toBeNull();
  });
});
