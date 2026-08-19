/** Per-user local daemon and application-level JSON RPC client. */

import { createHash, randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Application, type RuntimePaths, runtimePathsFromEnvironment } from "./application.ts";
import { CodeIndexingError, isCodeIndexingError } from "./errors.ts";
import { acquireLock } from "./indexing.ts";
import { jsonable } from "./jsonable.ts";
import {
  CodeChunk,
  DeclarationSelector,
  HistoryPage,
  IndexReport,
  type IndexTrigger,
  MaintenanceReport,
  ModelStatus,
  OutlineResponse,
  ProjectInfo,
  ProjectStatus,
  RefactorAnalysis,
  RefactorOperation,
  ReferenceResponse,
  RemovalReport,
  ScanInspectionPage,
  SearchResponse,
  StorageStatus,
  SymbolResponse,
} from "./models.ts";
import { resolvePath } from "./paths.ts";
import { type IndexProgress, readProgress } from "./progress.ts";
import { indexSettingsFromEnvironment, type IndexSettings } from "./settings.ts";

export const PROTOCOL_VERSION = 2;
export const MAX_FRAME_BYTES = 16 * 1024 ** 2;

const logger = console;

export function daemonSupported(): boolean {
  return process.platform !== "win32";
}

export function requireDaemonSupport(): void {
  if (!daemonSupported()) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "The shared indexing daemon requires Unix domain sockets, which are " +
        "unavailable on this platform; set CODE_INDEXING_BROKER=off or run " +
        "'code-indexing-mcp serve --direct'",
      { platform: process.platform },
    );
  }
}

function privateDirectory(directory: string): string {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  if (typeof process.getuid !== "function") return directory;
  const info = fs.lstatSync(directory);
  if (!info.isDirectory()) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "The daemon runtime path is not a directory",
      { path: directory },
    );
  }
  if (info.uid !== process.getuid()) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      "The daemon runtime directory is not owned by the current user",
      { path: directory, owner_uid: info.uid },
    );
  }
  if ((info.mode & 0o77) !== 0) fs.chmodSync(directory, 0o700);
  return directory;
}

export function daemonEndpoint(paths: RuntimePaths): string {
  const identity =
    typeof process.getuid === "function"
      ? String(process.getuid())
      : (process.env.USERNAME ?? "user");
  const runtimeRoot = process.env.XDG_RUNTIME_DIR;
  const root = runtimeRoot && runtimeRoot !== "" ? runtimeRoot : os.tmpdir();
  const directory = privateDirectory(path.join(root, `code-indexing-mcp-${identity}`));
  const digest = createHash("sha256").update(resolvePath(paths.data)).digest("hex").slice(0, 16);
  return path.join(directory, `${digest}.sock`);
}

export async function sendFrame(
  socket: net.Socket,
  payload: Record<string, unknown>,
): Promise<void> {
  const encoded = Buffer.from(JSON.stringify(jsonable(payload)));
  if (encoded.length > MAX_FRAME_BYTES) {
    throw new CodeIndexingError(
      "PROTOCOL_ERROR",
      "Local daemon request exceeds the maximum frame size",
      { maximum_bytes: MAX_FRAME_BYTES },
    );
  }
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32BE(encoded.length, 0);
  await writeAll(socket, Buffer.concat([header, encoded]));
}

export function attachFrameReader(socket: net.Socket): {
  receive: () => Promise<Record<string, unknown>>;
} {
  let buffered = Buffer.alloc(0);
  const waiters: Array<() => void> = [];
  socket.on("data", (chunk: Buffer) => {
    buffered = Buffer.concat([buffered, chunk]);
    const waiter = waiters.shift();
    if (waiter !== undefined) waiter();
  });
  const read = async (size: number): Promise<Buffer> => {
    while (buffered.length < size) {
      await new Promise<void>((resolve, reject) => {
        if (buffered.length >= size) {
          resolve();
          return;
        }
        const onError = (error: Error): void => {
          socket.off("close", onClose);
          reject(error);
        };
        const onClose = (): void => {
          socket.off("error", onError);
          reject(new Error("Local daemon connection closed"));
        };
        waiters.push(() => {
          socket.off("error", onError);
          socket.off("close", onClose);
          resolve();
        });
        socket.once("error", onError);
        socket.once("close", onClose);
      });
    }
    const out = buffered.subarray(0, size);
    buffered = buffered.subarray(size);
    return out;
  };
  return {
    receive: async () => {
      const header = await read(4);
      const size = header.readUInt32BE(0);
      if (size > MAX_FRAME_BYTES) {
        throw new CodeIndexingError(
          "PROTOCOL_ERROR",
          "Local daemon frame exceeds the maximum size",
          { maximum_bytes: MAX_FRAME_BYTES },
        );
      }
      const value: unknown = JSON.parse((await read(size)).toString("utf8"));
      if (value === null || typeof value !== "object" || Array.isArray(value)) {
        throw new Error("Daemon frame must contain a JSON object");
      }
      return value as Record<string, unknown>;
    },
  };
}

export async function receiveFrame(socket: net.Socket): Promise<Record<string, unknown>> {
  return attachFrameReader(socket).receive();
}

function writeAll(socket: net.Socket, payload: Buffer): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.write(payload, (error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

export class DaemonServer {
  readonly paths: RuntimePaths;
  readonly application: Application;
  readonly idleTimeoutSeconds: number;
  readonly endpoint: string;
  readonly tokenPath: string;
  readonly ready: Promise<void>;
  #resolveReady: () => void = () => undefined;
  #rejectReady: (error: unknown) => void = () => undefined;
  #stop = false;
  #lastActivity = nowSeconds();
  #activeRequests = 0;
  #maintenanceActive = false;
  #token = "";
  #maintenance: Promise<void> | null = null;

  constructor(
    paths: RuntimePaths,
    {
      application,
      idleTimeoutSeconds = 300,
    }: { application?: Application; idleTimeoutSeconds?: number } = {},
  ) {
    this.paths = paths;
    this.application = application ?? new Application(paths);
    this.idleTimeoutSeconds = idleTimeoutSeconds;
    this.endpoint = daemonEndpoint(paths);
    this.tokenPath = path.join(paths.data, "daemon.token");
    this.ready = new Promise((resolve, reject) => {
      this.#resolveReady = resolve;
      this.#rejectReady = reject;
    });
  }

  async serve(): Promise<void> {
    try {
      await this.#serve();
    } catch (error) {
      this.#rejectReady(error);
      throw error;
    }
  }

  async #serve(): Promise<void> {
    fs.mkdirSync(this.paths.data, { recursive: true });
    const lockDirectory = path.join(this.paths.data, "locks");
    fs.mkdirSync(lockDirectory, { recursive: true });
    let release: (() => Promise<void>) | undefined;
    try {
      release = await acquireLock(path.join(lockDirectory, "daemon.lock"), false);
    } catch (error) {
      if (isCodeIndexingError(error) && error.code === "INDEX_BUSY") {
        throw new CodeIndexingError(
          "INDEX_BUSY",
          "The per-user indexing daemon is already running",
          {},
          { cause: error },
        );
      }
      throw error;
    }
    this.#token = this.#loadOrCreateToken();
    if (fs.existsSync(this.endpoint)) fs.unlinkSync(this.endpoint);
    const incoming: net.Socket[] = [];
    const waiters: Array<(socket: net.Socket | null) => void> = [];
    const listener = net.createServer((socket) => {
      const waiter = waiters.shift();
      if (waiter !== undefined) waiter(socket);
      else incoming.push(socket);
    });
    try {
      await listenUnix(listener, this.endpoint);
      if (process.platform !== "win32") fs.chmodSync(this.endpoint, 0o600);
      this.#resolveReady();
      this.#maintenanceActive = true;
      this.#maintenance = this.#runStartupMaintenance();
      while (!this.#stop) {
        const idle =
          this.#activeRequests === 0 &&
          !this.#maintenanceActive &&
          nowSeconds() - this.#lastActivity >= this.idleTimeoutSeconds;
        if (idle) break;
        const connection =
          incoming.shift() ??
          (await new Promise<net.Socket | null>((resolve) => {
            const timer = setTimeout(() => {
              const index = waiters.indexOf(resolve);
              if (index >= 0) waiters.splice(index, 1);
              resolve(null);
            }, 500);
            waiters.push((socket) => {
              clearTimeout(timer);
              resolve(socket);
            });
          }));
        if (connection === null) continue;
        this.#lastActivity = nowSeconds();
        this.#activeRequests += 1;
        void this.#handle(connection, attachFrameReader(connection));
      }
    } finally {
      listener.close();
      if (this.#maintenance !== null) await this.#maintenance;
      if (fs.existsSync(this.endpoint)) fs.unlinkSync(this.endpoint);
      await release();
      this.#resolveReady();
    }
  }

  stop(): void {
    this.#stop = true;
  }

  async #runStartupMaintenance(): Promise<void> {
    try {
      this.#lastActivity = nowSeconds();
      await this.application.maybeRunMaintenance();
    } catch (error) {
      logger.error("Scheduled maintenance after daemon startup failed", error);
    } finally {
      this.#maintenanceActive = false;
      this.#lastActivity = nowSeconds();
    }
  }

  async #handle(
    connection: net.Socket,
    reader: { receive: () => Promise<Record<string, unknown>> },
  ): Promise<void> {
    let requestId: unknown;
    try {
      const request = await reader.receive();
      requestId = request.id;
      if (request.protocol !== PROTOCOL_VERSION) {
        throw new CodeIndexingError("INVALID_CONFIGURATION", "Incompatible local daemon protocol", {
          expected: PROTOCOL_VERSION,
        });
      }
      const token = String(request.token ?? "");
      if (!safeEqual(token, this.#token)) {
        throw new CodeIndexingError("INVALID_CONFIGURATION", "Local daemon authentication failed");
      }
      const result = await this.#dispatch(
        String(request.method ?? ""),
        (request.params as Record<string, unknown> | undefined) ?? {},
      );
      await sendFrame(connection, {
        id: requestId,
        result: jsonable(result) as Record<string, unknown>,
      });
    } catch (error) {
      if (isCodeIndexingError(error)) {
        await sendFrame(connection, {
          id: requestId,
          error: { code: error.code, message: error.toString(), details: error.details },
        });
      } else {
        await sendFrame(connection, {
          id: requestId,
          error: {
            code: "PROTOCOL_ERROR",
            message: `${error instanceof Error ? error.name : "Error"}: ${error instanceof Error ? error.message : String(error)}`,
            details: {},
          },
        });
      }
    } finally {
      connection.end();
      this.#lastActivity = nowSeconds();
      this.#activeRequests -= 1;
    }
  }

  async #dispatch(method: string, params: Record<string, unknown>): Promise<unknown> {
    if (method === "ping") return { pid: process.pid, protocol: PROTOCOL_VERSION };
    if (method === "stop") {
      this.#stop = true;
      return { stopping: true };
    }
    const app = this.application;
    const roots = asStringList(params.roots);
    const rest = { ...params };
    delete rest.roots;
    if (method === "init_project") {
      const projectPath =
        rest.path === undefined || rest.path === null ? undefined : String(rest.path);
      const name = rest.name === undefined || rest.name === null ? undefined : String(rest.name);
      return app.initProject(projectPath, {
        ...(name === undefined ? {} : { name }),
        forceNewId: Boolean(rest.force_new_id),
        allowOverlap: Boolean(rest.allow_overlap),
        roots,
      });
    }
    if (method === "discover_project") return app.discoverProject(String(rest.root));
    if (method === "index_project") {
      return app.indexProject(optionalString(rest.project), {
        roots,
        force: Boolean(rest.force),
        waitForLock: Boolean(rest.wait_for_lock),
        trigger: (optionalString(rest.trigger) ?? "manual") as IndexTrigger,
      });
    }
    if (method === "project_status") {
      return app.projectStatus(optionalString(rest.project), { roots });
    }
    if (method === "index_history") {
      const cursor = optionalString(rest.cursor);
      return app.indexHistory(optionalString(rest.project), {
        roots,
        ...(cursor === undefined ? {} : { cursor }),
        limit: typeof rest.limit === "number" ? rest.limit : 20,
      });
    }
    if (method === "inspect_scan") {
      const outcome = optionalString(rest.outcome);
      const reason = optionalString(rest.reason);
      const cursor = optionalString(rest.cursor);
      return app.inspectScan(optionalString(rest.project), {
        roots,
        ...(outcome === undefined ? {} : { outcome }),
        ...(reason === undefined ? {} : { reason }),
        ...(cursor === undefined ? {} : { cursor }),
        limit: typeof rest.limit === "number" ? rest.limit : 50,
      });
    }
    if (method === "storage_status") {
      return app.storageStatus(optionalString(rest.project), { roots });
    }
    if (method === "maintain_storage") {
      return app.maintainStorage(optionalString(rest.project), {
        roots,
        dryRun: Boolean(rest.dry_run),
        waitForLock: rest.wait_for_lock === undefined ? true : Boolean(rest.wait_for_lock),
      });
    }
    if (method === "list_projects") return app.listProjects();
    if (method === "remove_project") return app.removeProject(String(rest.project));
    if (method === "resolve_project") {
      return app.resolveProject(optionalString(rest.explicit), roots);
    }
    if (method === "resolve_search_scope") {
      return app.resolveSearchScope(
        rest.projects === undefined || rest.projects === null
          ? undefined
          : asStringList(rest.projects),
        Boolean(rest.all_projects),
        roots,
      );
    }
    if (method === "search_code") {
      const projects =
        rest.projects === undefined || rest.projects === null
          ? undefined
          : asStringList(rest.projects);
      const languages = optionalStringList(rest.languages);
      const paths = optionalStringList(rest.paths);
      const kinds = optionalStringList(rest.kinds);
      return app.searchCode(String(rest.query), {
        roots,
        ...(projects === undefined ? {} : { projects }),
        allProjects: Boolean(rest.all_projects),
        ...(languages === undefined ? {} : { languages }),
        ...(paths === undefined ? {} : { paths }),
        ...(kinds === undefined ? {} : { kinds }),
        limit: typeof rest.limit === "number" ? rest.limit : 8,
      });
    }
    if (method === "find_symbol") {
      const kinds = optionalStringList(rest.kinds);
      return app.findSymbol(String(rest.name), optionalString(rest.project), {
        roots,
        match: (optionalString(rest.match) ?? "exact") as "exact" | "prefix" | "contains",
        ...(kinds === undefined ? {} : { kinds }),
        limit: typeof rest.limit === "number" ? rest.limit : 20,
      });
    }
    if (method === "file_outline") {
      return app.fileOutline(String(rest.path), optionalString(rest.project), { roots });
    }
    if (method === "get_chunk") return app.getChunk(String(rest.chunk_id));
    if (method === "find_references") {
      const selector = DeclarationSelector.parse(rest.selector);
      const kinds =
        rest.kinds === undefined || rest.kinds === null
          ? undefined
          : new Set(asStringList(rest.kinds));
      const cursor = optionalString(rest.cursor);
      return app.findReferences(selector, {
        roots,
        ...(kinds === undefined ? {} : { kinds }),
        limit: typeof rest.limit === "number" ? rest.limit : 100,
        ...(cursor === undefined ? {} : { cursor }),
      });
    }
    if (method === "analyze_refactor") {
      const selector = DeclarationSelector.parse(rest.selector);
      const operation = RefactorOperation.parse(rest.operation);
      const cursor = optionalString(rest.cursor);
      return app.analyzeRefactor(selector, operation, {
        roots,
        limit: typeof rest.limit === "number" ? rest.limit : 500,
        ...(cursor === undefined ? {} : { cursor }),
      });
    }
    if (method === "model_status") return app.modelStatus();
    throw new CodeIndexingError("PROTOCOL_ERROR", `Unknown daemon method: ${method}`);
  }

  #loadOrCreateToken(): string {
    if (fs.existsSync(this.tokenPath)) return fs.readFileSync(this.tokenPath, "utf8").trim();
    const token = randomBytes(32).toString("hex");
    try {
      const handle = fs.openSync(this.tokenPath, "wx", 0o600);
      try {
        fs.writeFileSync(handle, token);
      } finally {
        fs.closeSync(handle);
      }
      return token;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "EEXIST") {
        return fs.readFileSync(this.tokenPath, "utf8").trim();
      }
      throw error;
    }
  }
}

export class BrokerApplication {
  readonly paths: RuntimePaths;
  readonly cwd: string;
  readonly settings: IndexSettings;
  readonly endpoint: string;
  readonly tokenPath: string;

  constructor(paths: RuntimePaths, { cwd }: { cwd?: string } = {}) {
    this.paths = paths;
    this.cwd = resolvePath(cwd ?? process.cwd());
    this.settings = indexSettingsFromEnvironment();
    this.endpoint = daemonEndpoint(paths);
    this.tokenPath = path.join(paths.data, "daemon.token");
  }

  static fromEnvironment({ cwd }: { cwd?: string } = {}): BrokerApplication {
    return new BrokerApplication(runtimePathsFromEnvironment(), cwd === undefined ? {} : { cwd });
  }

  async callOnce(
    method: string,
    params: Record<string, unknown> = {},
    protocol = PROTOCOL_VERSION,
  ): Promise<unknown> {
    const token = fs.readFileSync(this.tokenPath, "utf8").trim();
    const connection = await connectUnix(this.endpoint, 5000);
    const reader = attachFrameReader(connection);
    try {
      await sendFrame(connection, {
        protocol,
        id: randomUUID().replaceAll("-", ""),
        token,
        method,
        params: jsonable(params) as Record<string, unknown>,
      });
      const response = await reader.receive();
      const error = response.error as
        | { code?: string; message?: string; details?: Record<string, unknown> }
        | undefined;
      if (error) {
        const code =
          typeof error.code === "string" && isErrorCode(error.code) ? error.code : "PROTOCOL_ERROR";
        throw new CodeIndexingError(code, String(error.message ?? ""), error.details ?? {});
      }
      return response.result;
    } finally {
      connection.end();
    }
  }

  async call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    try {
      return await this.callOnce(method, params);
    } catch (error) {
      if (isConnectionGone(error)) {
        await ensureDaemon(this.paths);
        return this.callOnce(method, params);
      }
      throw error;
    }
  }

  async pingOnce(): Promise<Record<string, unknown>> {
    return (await this.callOnce("ping")) as Record<string, unknown>;
  }

  async ping(): Promise<Record<string, unknown>> {
    return (await this.call("ping")) as Record<string, unknown>;
  }

  async stop(): Promise<Record<string, unknown>> {
    return (await this.call("stop")) as Record<string, unknown>;
  }

  async initProject(
    projectPath?: string | null,
    options: {
      name?: string;
      forceNewId?: boolean;
      allowOverlap?: boolean;
      roots?: readonly string[];
    } = {},
  ): Promise<ProjectInfo> {
    return ProjectInfo.parse(
      await this.call("init_project", {
        path: projectPath ?? null,
        name: options.name ?? null,
        force_new_id: options.forceNewId ?? false,
        allow_overlap: options.allowOverlap ?? false,
        roots: options.roots ?? [],
      }),
    );
  }

  async discoverProject(root: string): Promise<ProjectInfo | null> {
    const value = await this.call("discover_project", { root });
    return value === null ? null : ProjectInfo.parse(value);
  }

  async listProjects(): Promise<ProjectInfo[]> {
    return ((await this.call("list_projects")) as unknown[]).map((value) =>
      ProjectInfo.parse(value),
    );
  }

  async indexProject(
    project?: string | null,
    options: {
      roots?: readonly string[];
      force?: boolean;
      waitForLock?: boolean;
      trigger?: IndexTrigger;
    } = {},
  ): Promise<IndexReport> {
    return IndexReport.parse(
      await this.call("index_project", {
        project: project ?? null,
        roots: options.roots ?? [],
        force: options.force ?? false,
        wait_for_lock: options.waitForLock ?? false,
        trigger: options.trigger ?? "manual",
      }),
    );
  }

  indexProgress(projectId: string): IndexProgress | null {
    return readProgress(path.join(this.paths.data, "progress"), projectId);
  }

  async projectStatus(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<ProjectStatus> {
    return ProjectStatus.parse(
      await this.call("project_status", { project: project ?? null, roots: roots ?? [] }),
    );
  }

  async indexHistory(
    project?: string | null,
    options: { roots?: readonly string[]; cursor?: string | null; limit?: number } = {},
  ): Promise<HistoryPage> {
    return HistoryPage.parse(
      await this.call("index_history", {
        project: project ?? null,
        roots: options.roots ?? [],
        cursor: options.cursor ?? null,
        limit: options.limit ?? 20,
      }),
    );
  }

  async inspectScan(
    project?: string | null,
    options: {
      roots?: readonly string[];
      outcome?: string | null;
      reason?: string | null;
      cursor?: string | null;
      limit?: number;
    } = {},
  ): Promise<ScanInspectionPage> {
    return ScanInspectionPage.parse(
      await this.call("inspect_scan", {
        project: project ?? null,
        roots: options.roots ?? [],
        outcome: options.outcome ?? null,
        reason: options.reason ?? null,
        cursor: options.cursor ?? null,
        limit: options.limit ?? 50,
      }),
    );
  }

  async storageStatus(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<StorageStatus> {
    return StorageStatus.parse(
      await this.call("storage_status", { project: project ?? null, roots: roots ?? [] }),
    );
  }

  async maintainStorage(
    project?: string | null,
    options: { roots?: readonly string[]; dryRun?: boolean; waitForLock?: boolean } = {},
  ): Promise<MaintenanceReport> {
    return MaintenanceReport.parse(
      await this.call("maintain_storage", {
        project: project ?? null,
        roots: options.roots ?? [],
        dry_run: options.dryRun ?? false,
        wait_for_lock: options.waitForLock ?? true,
      }),
    );
  }

  async projectIsStale(
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<boolean> {
    return (
      (await this.projectStatus(project, roots === undefined ? {} : { roots })).state === "stale"
    );
  }

  async removeProject(project: string): Promise<RemovalReport> {
    return RemovalReport.parse(await this.call("remove_project", { project }));
  }

  async resolveProject(explicit?: string | null, roots?: readonly string[]): Promise<ProjectInfo> {
    return ProjectInfo.parse(
      await this.call("resolve_project", { explicit: explicit ?? null, roots: roots ?? [] }),
    );
  }

  async resolveSearchScope(
    projects: readonly string[] | undefined,
    allProjects: boolean,
    roots?: readonly string[],
  ): Promise<string[]> {
    return asStringList(
      await this.call("resolve_search_scope", {
        projects: projects ?? null,
        all_projects: allProjects,
        roots: roots ?? [],
      }),
    );
  }

  async searchCode(
    query: string,
    options: {
      projects?: readonly string[];
      allProjects?: boolean;
      languages?: readonly string[];
      paths?: readonly string[];
      kinds?: readonly string[];
      limit?: number;
      roots?: readonly string[];
    } = {},
  ): Promise<SearchResponse> {
    return SearchResponse.parse(
      await this.call("search_code", {
        query,
        projects: options.projects ?? null,
        all_projects: options.allProjects ?? false,
        languages: options.languages ?? null,
        paths: options.paths ?? null,
        kinds: options.kinds ?? null,
        limit: options.limit ?? 8,
        roots: options.roots ?? [],
      }),
    );
  }

  async findSymbol(
    name: string,
    project?: string | null,
    options: {
      match?: "exact" | "prefix" | "contains";
      kinds?: readonly string[];
      limit?: number;
      roots?: readonly string[];
    } = {},
  ): Promise<SymbolResponse> {
    return SymbolResponse.parse(
      await this.call("find_symbol", {
        name,
        project: project ?? null,
        match: options.match ?? "exact",
        kinds: options.kinds ?? null,
        limit: options.limit ?? 20,
        roots: options.roots ?? [],
      }),
    );
  }

  async fileOutline(
    sourcePath: string,
    project?: string | null,
    { roots }: { roots?: readonly string[] } = {},
  ): Promise<OutlineResponse> {
    return OutlineResponse.parse(
      await this.call("file_outline", {
        path: sourcePath,
        project: project ?? null,
        roots: roots ?? [],
      }),
    );
  }

  async getChunk(chunkId: string): Promise<CodeChunk> {
    return CodeChunk.parse(await this.call("get_chunk", { chunk_id: chunkId }));
  }

  async findReferences(
    selector: DeclarationSelector,
    options: {
      kinds?: ReadonlySet<string> | null;
      limit?: number;
      cursor?: string | null;
      roots?: readonly string[];
    } = {},
  ): Promise<ReferenceResponse> {
    return ReferenceResponse.parse(
      await this.call("find_references", {
        selector,
        kinds:
          options.kinds === undefined ? null : options.kinds === null ? null : [...options.kinds],
        limit: options.limit ?? 100,
        cursor: options.cursor ?? null,
        roots: options.roots ?? [],
      }),
    );
  }

  async analyzeRefactor(
    selector: DeclarationSelector,
    operation: RefactorOperation,
    options: { limit?: number; cursor?: string | null; roots?: readonly string[] } = {},
  ): Promise<RefactorAnalysis> {
    return RefactorAnalysis.parse(
      await this.call("analyze_refactor", {
        selector,
        operation,
        limit: options.limit ?? 500,
        cursor: options.cursor ?? null,
        roots: options.roots ?? [],
      }),
    );
  }

  async modelStatus(): Promise<ModelStatus> {
    return ModelStatus.parse(await this.call("model_status"));
  }
}

export async function daemonStatus(paths: RuntimePaths): Promise<Record<string, unknown>> {
  try {
    return { running: true, ...(await new BrokerApplication(paths).pingOnce()) };
  } catch (error) {
    if (isCodeIndexingError(error)) {
      const expected = error.details.expected;
      if (error.code === "INVALID_CONFIGURATION" && typeof expected === "number") {
        return { running: true, protocol: expected };
      }
      return { running: false };
    }
    if (error instanceof Error && /closed|ECONNREFUSED|ENOENT|EPIPE/i.test(error.message)) {
      return { running: false };
    }
    if (isConnectionGone(error)) return { running: false };
    throw error;
  }
}

export async function retireStaleDaemon(paths: RuntimePaths, protocol: number): Promise<void> {
  const broker = new BrokerApplication(paths);
  try {
    await broker.callOnce("stop", {}, protocol);
  } catch {
    // best-effort
  }
  const deadline = nowSeconds() + 5;
  while (nowSeconds() < deadline) {
    if (!(await daemonStatus(paths)).running) return;
    await sleep(50);
  }
}

export async function ensureDaemon(
  paths: RuntimePaths,
  { timeoutSeconds = 10 }: { timeoutSeconds?: number } = {},
): Promise<BrokerApplication> {
  requireDaemonSupport();
  let status = await daemonStatus(paths);
  if (status.running && status.protocol === PROTOCOL_VERSION) return new BrokerApplication(paths);
  fs.mkdirSync(paths.data, { recursive: true });
  fs.mkdirSync(path.join(paths.data, "locks"), { recursive: true });
  const logPath = path.join(paths.data, "daemon.log");
  const release = await acquireLock(path.join(paths.data, "locks", "daemon-start.lock"), true);
  try {
    status = await daemonStatus(paths);
    if (status.running) {
      if (status.protocol === PROTOCOL_VERSION) return new BrokerApplication(paths);
      await retireStaleDaemon(paths, typeof status.protocol === "number" ? status.protocol : 0);
    }
    const log = fs.openSync(logPath, "a");
    try {
      const child = spawn(process.execPath, [cliEntry(), "daemon", "run"], {
        stdio: ["ignore", "ignore", log],
        detached: true,
        env: process.env,
      });
      child.unref();
    } finally {
      fs.closeSync(log);
    }
    const deadline = nowSeconds() + timeoutSeconds;
    while (nowSeconds() < deadline) {
      const broker = new BrokerApplication(paths);
      try {
        await broker.pingOnce();
        return broker;
      } catch {
        await sleep(50);
      }
    }
  } finally {
    await release();
  }
  throw new CodeIndexingError(
    "DAEMON_UNAVAILABLE",
    `Timed out starting the local indexing daemon; see ${logPath}`,
    { log_path: logPath, timeout_seconds: timeoutSeconds },
  );
}

function cliEntry(): string {
  return path.join(path.dirname(fileURLToPath(import.meta.url)), "cli.ts");
}

function listenUnix(server: net.Server, endpoint: string): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(endpoint, () => {
      server.off("error", reject);
      resolve();
    });
  });
}

function connectUnix(endpoint: string, timeoutMs: number): Promise<net.Socket> {
  requireDaemonSupport();
  return new Promise((resolve, reject) => {
    const socket = net.connect(endpoint);
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new Error("Local daemon connection timed out"));
    }, timeoutMs);
    socket.once("connect", () => {
      clearTimeout(timer);
      resolve(socket);
    });
    socket.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function safeEqual(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left);
  const rightBytes = Buffer.from(right);
  if (leftBytes.length !== rightBytes.length) return false;
  return timingSafeEqual(leftBytes, rightBytes);
}

function nowSeconds(): number {
  return performance.now() / 1000;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function optionalString(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  return String(value);
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item));
}

function optionalStringList(value: unknown): string[] | undefined {
  if (value === undefined || value === null) return undefined;
  return asStringList(value);
}

function isErrorCode(value: string): value is import("./errors.ts").ErrorCode {
  return (
    value === "PROJECT_NOT_FOUND" ||
    value === "AMBIGUOUS_PROJECT" ||
    value === "PROJECT_ID_CONFLICT" ||
    value === "CHUNK_NOT_FOUND" ||
    value === "MODEL_UNAVAILABLE" ||
    value === "INDEX_INCOMPATIBLE" ||
    value === "INDEX_BUSY" ||
    value === "INDEX_RESOURCE_LIMIT" ||
    value === "INDEX_CANCELLED" ||
    value === "EMBEDDING_WORKER_FAILED" ||
    value === "BACKEND_UNAVAILABLE" ||
    value === "DAEMON_UNAVAILABLE" ||
    value === "PROTOCOL_ERROR" ||
    value === "INVALID_CONFIGURATION" ||
    value === "INVALID_FILTER" ||
    value === "STALE_CURSOR" ||
    value === "AMBIGUOUS_SYMBOL" ||
    value === "UNSUPPORTED_LANGUAGE" ||
    value === "REFERENCE_INDEX_UNAVAILABLE" ||
    value === "INVALID_REFACTOR" ||
    value === "INVALID_CURSOR" ||
    value === "UNSUPPORTED_RUNTIME" ||
    value === "OVERLAPPING_PROJECT"
  );
}

function isConnectionGone(error: unknown): boolean {
  if (error instanceof Error && (error as NodeJS.ErrnoException).code === "ENOENT") return true;
  if (error instanceof Error && (error as NodeJS.ErrnoException).code === "ECONNREFUSED")
    return true;
  return false;
}

export { jsonable };
