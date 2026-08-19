/**
 * How an embedding worker is started.
 *
 * Per the migration plan §5.3 every worker is a child of this process that
 * dials back over an authenticated socket. The two-environment Python design
 * collapses: one install, provider selected at runtime, the process boundary
 * kept only for memory isolation.
 */

import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { WorkerConfig } from "./embedding-worker.ts";
import { CodeIndexingError } from "./errors.ts";
import { authenticateClient, authenticatePeer, type WorkerConnection } from "./worker-channel.ts";

export const HANDSHAKE_TIMEOUT_SECONDS = 30;
export const HANDSHAKE_POLL_SECONDS = 0.1;
export const DEFAULT_WORKER_TARGET = "embedding-worker:workerMain";

export interface WorkerProcess {
  readonly pid: number | undefined;
  isAlive(): boolean;
  terminate(): void;
  kill(): void;
  join(timeout?: number): Promise<void>;
}

export interface LaunchedWorker {
  readonly process: WorkerProcess;
  readonly connection: WorkerConnection;
}

export interface WorkerLauncher {
  launch(config: WorkerConfig): Promise<LaunchedWorker>;
  readonly description: string;
}

export type WorkerTarget = (
  connection: WorkerConnection,
  config: WorkerConfig,
) => Promise<void> | void;

export class FunctionLauncher implements WorkerLauncher {
  readonly description = "serving environment";
  private readonly target: WorkerTarget;

  constructor(target: WorkerTarget) {
    this.target = target;
  }

  async launch(config: WorkerConfig): Promise<LaunchedWorker> {
    const { linkedQueues } = await import("./worker-channel.ts");
    const { parent, child } = linkedQueues();
    const handle = new InProcessWorker();
    void Promise.resolve()
      .then(() => this.target(child, config))
      .catch(() => {
        child.close();
      })
      .finally(() => {
        handle.finish();
      });
    return { process: handle, connection: parent };
  }
}

class InProcessWorker implements WorkerProcess {
  readonly pid = process.pid;
  private done = false;
  private waiters: Array<() => void> = [];

  isAlive(): boolean {
    return !this.done;
  }

  terminate(): void {
    this.finish();
  }

  kill(): void {
    this.finish();
  }

  async join(_timeout?: number): Promise<void> {
    if (this.done) return;
    await new Promise<void>((resolve) => {
      this.waiters.push(resolve);
    });
  }

  finish(): void {
    if (this.done) return;
    this.done = true;
    for (const waiter of this.waiters) waiter();
    this.waiters = [];
  }
}

export class ChildProcessWorker implements WorkerProcess {
  private readonly child: ChildProcess;

  constructor(child: ChildProcess) {
    this.child = child;
  }

  get pid(): number | undefined {
    return this.child.pid;
  }

  isAlive(): boolean {
    return this.child.exitCode === null && this.child.signalCode === null;
  }

  terminate(): void {
    this.child.kill("SIGTERM");
  }

  kill(): void {
    this.child.kill("SIGKILL");
  }

  async join(timeout?: number): Promise<void> {
    if (!this.isAlive()) return;
    await new Promise<void>((resolve) => {
      const timer =
        timeout === undefined
          ? undefined
          : setTimeout(() => {
              this.child.off("exit", onExit);
              resolve();
            }, timeout * 1000);
      const onExit = (): void => {
        if (timer !== undefined) clearTimeout(timer);
        resolve();
      };
      this.child.once("exit", onExit);
    });
  }
}

export class ChildProcessLauncher implements WorkerLauncher {
  readonly executable: string;
  readonly timeoutSeconds: number;
  private readonly environmentName: string;
  private readonly target: string;
  private readonly extraEnvironment: Record<string, string>;

  constructor({
    executable,
    environmentName = "accelerator environment",
    timeoutSeconds = HANDSHAKE_TIMEOUT_SECONDS,
    target = DEFAULT_WORKER_TARGET,
    extraEnvironment = {},
  }: {
    executable: string;
    environmentName?: string;
    timeoutSeconds?: number;
    target?: string;
    extraEnvironment?: Record<string, string>;
  }) {
    this.executable = executable;
    this.environmentName = environmentName;
    this.timeoutSeconds = timeoutSeconds;
    this.target = target;
    this.extraEnvironment = extraEnvironment;
  }

  get description(): string {
    return `${this.environmentName} (${this.executable})`;
  }

  async launch(config: WorkerConfig): Promise<LaunchedWorker> {
    const channel = await WorkerChannel.open();
    let child: ChildProcess;
    try {
      child = this.start(channel);
    } catch (error) {
      channel.close();
      throw error;
    }
    let connection: WorkerConnection;
    try {
      connection = await channel.accept(child, this.timeoutSeconds);
    } catch (error) {
      await reap(child);
      channel.close();
      throw error;
    }
    channel.close();
    try {
      connection.send(["configure", workerConfigWire(config)]);
    } catch (error) {
      connection.close();
      await reap(child);
      throw error;
    }
    return { process: new ChildProcessWorker(child), connection };
  }

  private start(channel: WorkerChannel): ChildProcess {
    if (!fs.existsSync(this.executable) || !fs.statSync(this.executable).isFile()) {
      throw new CodeIndexingError(
        "BACKEND_UNAVAILABLE",
        `The recorded accelerator interpreter is missing: ${this.executable}`,
        { interpreter: this.executable },
      );
    }
    let child: ChildProcess;
    try {
      child = spawn(this.executable, launcherArgs(), {
        stdio: ["pipe", "ignore", "inherit"],
        env: { ...process.env, ...this.extraEnvironment },
      });
    } catch (error) {
      throw new CodeIndexingError(
        "BACKEND_UNAVAILABLE",
        `Could not start the accelerator interpreter: ${error}`,
        { interpreter: this.executable },
      );
    }
    try {
      writeHandshake(child, channel.handshakePayload(this.target));
    } catch (error) {
      void reap(child);
      throw new CodeIndexingError(
        "EMBEDDING_WORKER_FAILED",
        `Could not hand the accelerator worker its connection details: ${error}`,
      );
    }
    return child;
  }
}

export class SpawnLauncher implements WorkerLauncher {
  readonly description = "serving environment";
  private readonly target: WorkerTarget;

  constructor(target: WorkerTarget) {
    this.target = target;
  }

  launch(config: WorkerConfig): Promise<LaunchedWorker> {
    return new ChildProcessLauncher({
      executable: process.execPath,
      environmentName: this.description,
      target: targetReference(this.target),
    }).launch(config);
  }
}

class WorkerChannel {
  private readonly server: net.Server;
  readonly address: string | { host: string; port: number };
  readonly authkey: Buffer;
  readonly directory: string | undefined;

  constructor(fields: {
    server: net.Server;
    address: string | { host: string; port: number };
    authkey: Buffer;
    directory?: string;
  }) {
    this.server = fields.server;
    this.address = fields.address;
    this.authkey = fields.authkey;
    this.directory = fields.directory;
  }

  static open(): Promise<WorkerChannel> {
    return new Promise((resolve, reject) => {
      const authkey = Buffer.from(
        Array.from({ length: 32 }, () => Math.floor(Math.random() * 256)),
      );
      if (process.platform === "win32") {
        const server = net.createServer();
        server.listen(0, "127.0.0.1", () => {
          const address = server.address();
          if (address === null || typeof address === "string") {
            reject(new Error("expected a TCP address"));
            return;
          }
          resolve(
            new WorkerChannel({
              server,
              address: { host: address.address, port: address.port },
              authkey,
            }),
          );
        });
        server.on("error", reject);
        return;
      }
      const directory = fs.mkdtempSync(path.join(os.tmpdir(), "code-indexing-mcp-worker-"));
      const socketPath = path.join(directory, "worker.sock");
      const server = net.createServer();
      server.listen(socketPath, () => {
        resolve(new WorkerChannel({ server, address: socketPath, authkey, directory }));
      });
      server.on("error", reject);
    });
  }

  async accept(child: ChildProcess, timeoutSeconds: number): Promise<WorkerConnection> {
    const deadline = Date.now() + timeoutSeconds * 1000;
    let rejected = 0;
    for (;;) {
      stillExpected(child, deadline, timeoutSeconds, rejected);
      const client = await acceptOnce(this.server, HANDSHAKE_POLL_SECONDS);
      if (client === undefined) continue;
      const remaining = Math.max((deadline - Date.now()) / 1000, HANDSHAKE_POLL_SECONDS);
      const connection = await authenticatePeer(client, this.authkey, {
        timeoutSeconds: remaining,
      });
      if (connection !== undefined) return connection;
      rejected += 1;
      client.destroy();
    }
  }

  handshakePayload(target: string): string {
    const address =
      typeof this.address === "string" ? this.address : [this.address.host, this.address.port];
    return JSON.stringify({ address, authkey: this.authkey.toString("hex"), target });
  }

  close(): void {
    this.server.close();
    if (this.directory !== undefined) {
      fs.rmSync(this.directory, { recursive: true, force: true });
    }
  }
}

function stillExpected(
  child: ChildProcess,
  deadline: number,
  timeoutSeconds: number,
  rejected: number,
): void {
  if (child.exitCode !== null) {
    throw new CodeIndexingError(
      "BACKEND_UNAVAILABLE",
      `The accelerator worker exited with status ${child.exitCode} before it ` +
        "could be reached; its environment is most likely incomplete",
      { exit_code: child.exitCode },
    );
  }
  if (Date.now() >= deadline) {
    const detail = rejected ? `; ${rejected} connection(s) failed the handshake` : "";
    throw new CodeIndexingError(
      "EMBEDDING_WORKER_FAILED",
      `The accelerator worker did not connect within ${timeoutSeconds.toFixed(0)}s${detail}`,
    );
  }
}

function acceptOnce(server: net.Server, timeoutSeconds: number): Promise<net.Socket | undefined> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      server.off("connection", onConnection);
      server.off("error", onError);
      resolve(undefined);
    }, timeoutSeconds * 1000);
    const onConnection = (socket: net.Socket): void => {
      clearTimeout(timer);
      server.off("error", onError);
      resolve(socket);
    };
    const onError = (error: Error): void => {
      clearTimeout(timer);
      server.off("connection", onConnection);
      reject(
        new CodeIndexingError(
          "EMBEDDING_WORKER_FAILED",
          `Could not accept the accelerator worker connection: ${error}`,
        ),
      );
    };
    server.once("connection", onConnection);
    server.once("error", onError);
  });
}

async function reap(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null) return;
  child.kill("SIGTERM");
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve();
    }, 2000);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function writeHandshake(child: ChildProcess, payload: string): void {
  const stdin = child.stdin;
  if (stdin === null) throw new Error("worker stdin is not available");
  stdin.write(`${payload}\n`);
  stdin.end();
}

function launcherArgs(): string[] {
  const here = fileURLToPath(import.meta.url);
  return [here];
}

function workerConfigWire(config: WorkerConfig): Record<string, unknown> {
  return {
    cacheDirectory: config.cacheDirectory,
    offline: config.offline,
    threads: config.threads,
    enableCpuMemArena: config.enableCpuMemArena,
    dimension: config.dimension,
    modelId: config.modelId,
    providers: [...config.providers],
    accelerator: config.accelerator,
  };
}

export function parseWorkerConfig(raw: Record<string, unknown>): WorkerConfig {
  return {
    cacheDirectory: String(raw.cacheDirectory ?? raw.cache_directory ?? ""),
    offline: Boolean(raw.offline),
    threads: Number(raw.threads ?? 1),
    enableCpuMemArena: Boolean(raw.enableCpuMemArena ?? raw.enable_cpu_mem_arena),
    dimension: Number(raw.dimension),
    modelId: String(raw.modelId ?? raw.model_id ?? "jinaai/jina-embeddings-v2-base-code"),
    providers: Array.isArray(raw.providers) ? raw.providers.map((name) => String(name)) : [],
    accelerator: String(raw.accelerator ?? "cpu"),
  };
}

const targets = new Map<string, WorkerTarget>();

export function registerWorkerTarget(name: string, target: WorkerTarget): void {
  targets.set(name, target);
}

function targetReference(target: WorkerTarget): string {
  for (const [reference, registered] of targets) {
    if (registered === target) return reference;
  }
  throw new Error("Worker target must be registered before it can be spawned");
}

export function resolveTarget(reference: string): WorkerTarget {
  const registered = targets.get(reference);
  if (registered !== undefined) return registered;
  const [moduleName, attribute] = reference.split(":");
  if (!moduleName || !attribute) {
    throw new Error(`Malformed worker target: ${JSON.stringify(reference)}`);
  }
  const loaded = targets.get(`${moduleName}:${attribute}`);
  if (loaded === undefined) {
    throw new Error(`Unknown worker target: ${reference}`);
  }
  return loaded;
}

export async function childMain(stream: NodeJS.ReadableStream = process.stdin): Promise<number> {
  const line = await readLine(stream);
  if (line === undefined) return 2;
  const payload = JSON.parse(line) as {
    address: string | [string, number];
    authkey: string;
    target?: string;
  };
  const socket = await connectBack(payload.address);
  const connection = await authenticateClient(socket, Buffer.from(payload.authkey, "hex"));
  if (connection === undefined) return 2;
  try {
    const message = (await connection.recv()) as [string, Record<string, unknown>];
    const [command, rawConfig] = message;
    if (command !== "configure") return 2;
    const reference = String(payload.target ?? DEFAULT_WORKER_TARGET);
    if (reference === DEFAULT_WORKER_TARGET && !targets.has(reference)) {
      await import("./embedding-worker.ts");
    }
    const target = resolveTarget(reference);
    await target(connection, parseWorkerConfig(rawConfig));
  } finally {
    connection.close();
  }
  return 0;
}

function connectBack(address: string | [string, number]): Promise<net.Socket> {
  return new Promise((resolve, reject) => {
    const socket =
      typeof address === "string" ? net.connect(address) : net.connect(address[1], address[0]);
    socket.once("connect", () => resolve(socket));
    socket.once("error", reject);
  });
}

function readLine(stream: NodeJS.ReadableStream): Promise<string | undefined> {
  return new Promise((resolve) => {
    let buffer = "";
    const onData = (chunk: Buffer | string): void => {
      buffer += chunk.toString();
      const index = buffer.indexOf("\n");
      if (index >= 0) {
        stream.off("data", onData);
        stream.off("end", onEnd);
        resolve(buffer.slice(0, index));
      }
    };
    const onEnd = (): void => {
      stream.off("data", onData);
      resolve(buffer.length > 0 ? buffer : undefined);
    };
    stream.on("data", onData);
    stream.on("end", onEnd);
  });
}

if (import.meta.main) {
  void childMain().then((code) => process.exit(code));
}
