/** Authenticated dial-back launch and handshake rejection. */

import { expect, test } from "bun:test";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { workerConfig } from "../src/embedding-worker.ts";
import { isCodeIndexingError } from "../src/errors.ts";
import { answerChallenge, deliverChallenge } from "../src/worker-channel.ts";
import { ChildProcessLauncher } from "../src/worker-launcher.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

test("a missing interpreter is reported as an unavailable backend", async () => {
  const directory = temporaryDirectory();
  try {
    const launcher = new ChildProcessLauncher({
      executable: path.join(directory, "no-such-python"),
    });
    let caught: unknown;
    try {
      await launcher.launch(
        workerConfig({
          cacheDirectory: path.join(directory, "models"),
          offline: true,
          threads: 1,
          enableCpuMemArena: false,
          dimension: 4,
        }),
      );
    } catch (error) {
      caught = error;
    }
    expect(isCodeIndexingError(caught)).toBe(true);
    if (!isCodeIndexingError(caught)) return;
    expect(caught.code).toBe("BACKEND_UNAVAILABLE");
  } finally {
    removeDirectory(directory);
  }
});

test("an environment that cannot start the worker is reported not awaited", async () => {
  if (process.platform === "win32") return;
  const directory = temporaryDirectory();
  try {
    const broken = path.join(directory, "broken-python");
    fs.writeFileSync(broken, "#!/bin/sh\nexit 3\n");
    fs.chmodSync(broken, 0o755);
    const launcher = new ChildProcessLauncher({
      executable: broken,
      timeoutSeconds: 30,
    });
    let caught: unknown;
    try {
      await launcher.launch(
        workerConfig({
          cacheDirectory: path.join(directory, "models"),
          offline: true,
          threads: 1,
          enableCpuMemArena: false,
          dimension: 4,
        }),
      );
    } catch (error) {
      caught = error;
    }
    expect(isCodeIndexingError(caught)).toBe(true);
    if (!isCodeIndexingError(caught)) return;
    expect(caught.code).toBe("BACKEND_UNAVAILABLE");
    expect(String(caught)).toContain("exited with status 3");
  } finally {
    removeDirectory(directory);
  }
});

test("a child that never connects gives up at the timeout", async () => {
  if (process.platform === "win32") return;
  const directory = temporaryDirectory();
  try {
    const stalled = path.join(directory, "stalled-python");
    fs.writeFileSync(stalled, "#!/bin/sh\nexec sleep 30\n");
    fs.chmodSync(stalled, 0o755);
    const launcher = new ChildProcessLauncher({
      executable: stalled,
      timeoutSeconds: 0.5,
    });
    let caught: unknown;
    try {
      await launcher.launch(
        workerConfig({
          cacheDirectory: path.join(directory, "models"),
          offline: true,
          threads: 1,
          enableCpuMemArena: false,
          dimension: 4,
        }),
      );
    } catch (error) {
      caught = error;
    }
    expect(isCodeIndexingError(caught)).toBe(true);
    if (!isCodeIndexingError(caught)) return;
    expect(caught.code).toBe("EMBEDDING_WORKER_FAILED");
    expect(String(caught)).toContain("did not connect");
  } finally {
    removeDirectory(directory);
  }
});

test("a peer with the right key is accepted", async () => {
  const [left, right] = await connectedPair();
  const key = Buffer.alloc(32, 97);
  const peer = answerThenDeliver(right, key);
  expect(await deliverThenAnswer(left, key)).toBe(true);
  await peer;
  left.destroy();
  right.destroy();
});

test("a peer with the wrong key is dropped rather than raised", async () => {
  const [left, right] = await connectedPair();
  const peer = answerThenDeliver(right, Buffer.alloc(32, 98));
  expect(await deliverThenAnswer(left, Buffer.alloc(32, 97))).toBe(false);
  await peer;
  left.destroy();
  right.destroy();
});

test("a peer that connects and goes quiet is dropped at the timeout", async () => {
  const [left, right] = await connectedPair();
  const finished = Promise.race([
    deliverChallenge(left, Buffer.alloc(32, 97)),
    Bun.sleep(800).then(() => false),
  ]);
  expect(await finished).toBe(false);
  left.destroy();
  right.destroy();
});

async function connectedPair(): Promise<[net.Socket, net.Socket]> {
  const server = net.createServer();
  const incoming = new Promise<net.Socket>((resolve) => server.once("connection", resolve));
  await new Promise<void>((resolve, reject) => {
    server.listen(0, "127.0.0.1", () => resolve());
    server.once("error", reject);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("expected tcp");
  const client = net.connect(address.port, address.address);
  const accepted = await incoming;
  if (client.connecting) {
    await new Promise<void>((resolve, reject) => {
      client.once("connect", () => resolve());
      client.once("error", reject);
    });
  }
  server.close();
  return [client, accepted];
}

async function deliverThenAnswer(socket: net.Socket, key: Buffer): Promise<boolean> {
  const delivered = await deliverChallenge(socket, key);
  if (!delivered) return false;
  return await answerChallenge(socket, key);
}

async function answerThenDeliver(socket: net.Socket, key: Buffer): Promise<boolean> {
  const answered = await answerChallenge(socket, key);
  if (!answered) return false;
  return await deliverChallenge(socket, key);
}
