/**
 * Length-prefixed JSON frames and HMAC challenge-response for worker sockets.
 */

import { createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import type { Socket } from "node:net";

export const WELCOME = "#WELCOME#";
export const FAILURE = "#FAILURE#";
const CHALLENGE_BYTES = 20;

export type WireMessage = readonly [string, unknown];

export interface WorkerConnection {
  send(message: unknown): void;
  poll(timeoutSeconds: number): Promise<boolean>;
  recv(): Promise<unknown>;
  close(): void;
}

export function encodeBytes(value: Uint8Array): { $bytes: string } {
  return { $bytes: Buffer.from(value).toString("base64") };
}

export function decodeBytes(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (Buffer.isBuffer(value)) return new Uint8Array(value);
  if (value !== null && typeof value === "object" && "$bytes" in value) {
    return new Uint8Array(Buffer.from(String((value as { $bytes: string }).$bytes), "base64"));
  }
  throw new Error("expected packed bytes");
}

export function reviveBytes(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(reviveBytes);
  if (value !== null && typeof value === "object") {
    if ("$bytes" in value && Object.keys(value).length === 1) return decodeBytes(value);
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) out[key] = reviveBytes(entry);
    return out;
  }
  return value;
}

export function replaceBytes(value: unknown): unknown {
  if (value instanceof Uint8Array || Buffer.isBuffer(value)) return encodeBytes(value);
  if (Array.isArray(value)) return value.map(replaceBytes);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) out[key] = replaceBytes(entry);
    return out;
  }
  return value;
}

export class QueueConnection implements WorkerConnection {
  private readonly inbox: unknown[] = [];
  private waiters: Array<(ready: boolean) => void> = [];
  peer: QueueConnection | undefined;
  closed = false;

  send(message: unknown): void {
    if (this.closed) throw new Error("closed");
    const peer = this.peer;
    if (peer === undefined || peer.closed)
      throw Object.assign(new Error("closed"), { code: "EPIPE" });
    peer.inbox.push(structuredClone(replaceBytes(message)));
    peer.wake(true);
  }

  async poll(timeoutSeconds: number): Promise<boolean> {
    if (this.inbox.length > 0) return true;
    if (this.closed) throw Object.assign(new Error("closed"), { code: "EOF" });
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((waiter) => waiter !== onReady);
        resolve(false);
      }, timeoutSeconds * 1000);
      const onReady = (ready: boolean): void => {
        clearTimeout(timer);
        if (this.closed && this.inbox.length === 0) {
          reject(Object.assign(new Error("closed"), { code: "EOF" }));
          return;
        }
        resolve(ready);
      };
      this.waiters.push(onReady);
    });
  }

  async recv(): Promise<unknown> {
    if (this.inbox.length === 0) {
      const ready = await this.poll(60);
      if (!ready) throw Object.assign(new Error("closed"), { code: "EOF" });
    }
    const next = this.inbox.shift();
    if (next === undefined) throw Object.assign(new Error("closed"), { code: "EOF" });
    return reviveBytes(next);
  }

  close(): void {
    this.closed = true;
    this.wake(false);
  }

  private wake(ready: boolean): void {
    const waiters = this.waiters;
    this.waiters = [];
    for (const waiter of waiters) waiter(ready);
  }
}

export function linkedQueues(): { parent: QueueConnection; child: QueueConnection } {
  const parent = new QueueConnection();
  const child = new QueueConnection();
  parent.peer = child;
  child.peer = parent;
  return { parent, child };
}

export class SocketConnection implements WorkerConnection {
  private leftover = Buffer.alloc(0);
  private readonly inbox: unknown[] = [];
  private waiters: Array<(ready: boolean) => void> = [];
  private closed = false;

  private readonly socket: Socket;

  constructor(socket: Socket) {
    this.socket = socket;
    this.socket.on("data", (chunk: Buffer | string) => this.onData(Buffer.from(chunk)));
    this.socket.on("end", () => this.shutdown());
    this.socket.on("error", () => this.shutdown());
    this.socket.on("close", () => this.shutdown());
  }

  send(message: unknown): void {
    if (this.closed) throw Object.assign(new Error("closed"), { code: "EPIPE" });
    this.socket.write(frame(replaceBytes(message)));
  }

  async poll(timeoutSeconds: number): Promise<boolean> {
    if (this.inbox.length > 0) return true;
    if (this.closed) throw Object.assign(new Error("closed"), { code: "EOF" });
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((waiter) => waiter !== onReady);
        resolve(false);
      }, timeoutSeconds * 1000);
      const onReady = (ready: boolean): void => {
        clearTimeout(timer);
        if (this.closed && this.inbox.length === 0) {
          reject(Object.assign(new Error("closed"), { code: "EOF" }));
          return;
        }
        resolve(ready);
      };
      this.waiters.push(onReady);
    });
  }

  async recv(): Promise<unknown> {
    if (this.inbox.length === 0) {
      const ready = await this.poll(60);
      if (!ready) throw Object.assign(new Error("closed"), { code: "EOF" });
    }
    const next = this.inbox.shift();
    if (next === undefined) throw Object.assign(new Error("closed"), { code: "EOF" });
    return reviveBytes(next);
  }

  close(): void {
    this.shutdown();
    this.socket.destroy();
  }

  private onData(chunk: Buffer): void {
    this.leftover = Buffer.concat([this.leftover, chunk]);
    while (this.leftover.length >= 4) {
      const size = this.leftover.readUInt32BE(0);
      if (this.leftover.length < 4 + size) break;
      const payload = this.leftover.subarray(4, 4 + size).toString("utf8");
      this.leftover = this.leftover.subarray(4 + size);
      this.inbox.push(JSON.parse(payload));
      this.wake(true);
    }
  }

  private shutdown(): void {
    if (this.closed) return;
    this.closed = true;
    this.wake(false);
  }

  private wake(ready: boolean): void {
    const waiters = this.waiters;
    this.waiters = [];
    for (const waiter of waiters) waiter(ready);
  }
}

export function frame(value: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(value), "utf8");
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32BE(body.length);
  return Buffer.concat([header, body]);
}

export function hmacDigest(authkey: Buffer, message: Buffer): Buffer {
  return createHmac("sha256", authkey).update(message).digest();
}

export function deliverChallenge(socket: Socket, authkey: Buffer): Promise<boolean> {
  return exchange(socket, authkey, "deliver");
}

export function answerChallenge(socket: Socket, authkey: Buffer): Promise<boolean> {
  return exchange(socket, authkey, "answer");
}

async function exchange(
  socket: Socket,
  authkey: Buffer,
  role: "deliver" | "answer",
): Promise<boolean> {
  try {
    if (role === "deliver") {
      const message = randomBytes(CHALLENGE_BYTES);
      await writeRaw(socket, message);
      const response = await readExact(socket, 32);
      const expected = hmacDigest(authkey, message);
      if (!safeEqual(response, expected)) {
        await writeRaw(socket, Buffer.from(FAILURE, "utf8"));
        return false;
      }
      await writeRaw(socket, Buffer.from(WELCOME, "utf8"));
      return true;
    }
    const message = await readExact(socket, CHALLENGE_BYTES);
    await writeRaw(socket, hmacDigest(authkey, message));
    const welcome = await readExact(socket, WELCOME.length);
    return welcome.toString("utf8") === WELCOME;
  } catch {
    return false;
  }
}

export async function authenticatePeer(
  socket: Socket,
  authkey: Buffer,
  { timeoutSeconds }: { timeoutSeconds: number },
): Promise<SocketConnection | undefined> {
  const timer = setTimeout(() => {
    socket.destroy();
  }, timeoutSeconds * 1000);
  try {
    const delivered = await deliverChallenge(socket, authkey);
    if (!delivered) return undefined;
    const answered = await answerChallenge(socket, authkey);
    if (!answered) return undefined;
    return new SocketConnection(socket);
  } catch {
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}

export async function authenticateClient(
  socket: Socket,
  authkey: Buffer,
): Promise<SocketConnection | undefined> {
  try {
    const answered = await answerChallenge(socket, authkey);
    if (!answered) return undefined;
    const delivered = await deliverChallenge(socket, authkey);
    if (!delivered) return undefined;
    return new SocketConnection(socket);
  } catch {
    return undefined;
  }
}

function safeEqual(left: Buffer, right: Buffer): boolean {
  if (left.length !== right.length) return false;
  return timingSafeEqual(left, right);
}

function writeRaw(socket: Socket, data: Buffer): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.write(data, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

function readExact(socket: Socket, size: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let received = 0;
    const onData = (chunk: Buffer): void => {
      chunks.push(chunk);
      received += chunk.length;
      if (received >= size) {
        cleanup();
        const all = Buffer.concat(chunks);
        const extra = all.subarray(size);
        if (extra.length > 0) socket.unshift(extra);
        resolve(all.subarray(0, size));
      }
    };
    const onError = (error: Error): void => {
      cleanup();
      reject(error);
    };
    const onClose = (): void => {
      cleanup();
      reject(Object.assign(new Error("closed"), { code: "EOF" }));
    };
    const cleanup = (): void => {
      socket.off("data", onData);
      socket.off("error", onError);
      socket.off("close", onClose);
    };
    socket.on("data", onData);
    socket.on("error", onError);
    socket.on("close", onClose);
  });
}
