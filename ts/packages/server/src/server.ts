/** MCP stdio adapter. */

import { fileURLToPath } from "node:url";
import watcher from "@parcel/watcher";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { CallToolResult, Tool, ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { Application } from "./application.ts";
import { BrokerApplication } from "./daemon.ts";
import { CodeIndexingError, isCodeIndexingError } from "./errors.ts";
import { jsonable } from "./jsonable.ts";
import {
  ChunkKind,
  DeclarationSelector,
  type IndexReport,
  type IndexTrigger,
  LanguageName,
  describeProgress,
  RefactorOperation,
  ReferenceKind,
} from "./models.ts";
import { resolvePath } from "./paths.ts";
import { sameProjectRoot } from "./projects.ts";
import { type IndexMode, indexSettingsFromEnvironment } from "./settings.ts";

export const SERVER_INSTRUCTIONS =
  "Local Tree-sitter code indexing and hybrid search. " +
  "When exploring code, prefer these index tools over grep-style file reading: " +
  "search_code (semantic natural-language queries), find_symbol (definitions), " +
  "find_references (structural uses of a selected declaration), analyze_refactor " +
  "(rename or signature impact), file_outline (file structure before reading), " +
  "get_chunk (exact code for a " +
  "search hit). When correlating code across explicitly related services, use list_projects " +
  "to discover them and search_across_projects to search the selected repositories together. " +
  "Check list_projects/project_status for index freshness first and run index_project if the " +
  "index is missing or stale.";

export const TOOL_INSTRUCTIONS = `Local Tree-sitter code indexing with hybrid semantic and full-text search over repositories on \
this machine. No code leaves the machine: embeddings are computed locally and stored in a local \
LanceDB index.

search_code answers "where is the code that does X". find_symbol resolves a declaration whose name \
is already known. file_outline lists one file's structure without returning code. Both search \
tools return chunk_id values that get_chunk expands to full text.

Scope defaults to the active MCP root, or the nearest .ci-mcp/project.toml above the working \
directory. Searching every registered project requires all_projects=true, so cross-project results \
are never mixed in by accident.

For cross-repository debugging, use list_projects to discover related registrations, then prefer \
search_across_projects with at least two explicit project ids, names, or paths. It searches only \
that deliberate scope and globally ranks the combined results.

In the default lazy mode every project-scoped code query checks freshness and refreshes only when \
the source tree has changed. The initial refresh can take minutes on a large repository and \
reports progress while it runs.`;

export const INITIAL_RETRY_DELAY_SECONDS = { value: 0.05 };
export const MAXIMUM_RETRY_DELAY_SECONDS = { value: 1.0 };
export const PROGRESS_POLL_SECONDS = { value: 0.5 };
// Module-level holders so tests can substitute timings the way the Python
// suite monkeypatches the same constants.
export const EAGER_RECONCILE_SECONDS = { value: 30.0 };
export const WATCH_RETRY_INITIAL_SECONDS = { value: 1.0 };
export const WATCH_RETRY_MAXIMUM_SECONDS = { value: 30.0 };

const READ_ONLY: ToolAnnotations = {
  readOnlyHint: true,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
};
const READS_AND_REGISTERS: ToolAnnotations = {
  readOnlyHint: false,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
};
const INITIALIZES: ToolAnnotations = {
  readOnlyHint: false,
  destructiveHint: true,
  idempotentHint: false,
  openWorldHint: false,
};
const WRITES: ToolAnnotations = {
  readOnlyHint: false,
  destructiveHint: false,
  idempotentHint: true,
  openWorldHint: false,
};
const DESTRUCTIVE: ToolAnnotations = {
  readOnlyHint: false,
  destructiveHint: true,
  idempotentHint: true,
  openWorldHint: false,
};

type Surface = Application | BrokerApplication;

/** Sentinel thrown when the coordinator closes while a job waits for the index lock. */
const CLOSED = Symbol("coordinator-closed");

class ManualResetEvent {
  #waiters: Array<() => void> = [];
  #set = false;

  set(): void {
    this.#set = true;
    for (const waiter of this.#waiters) waiter();
    this.#waiters = [];
  }

  isSet(): boolean {
    return this.#set;
  }

  wait(): Promise<void> {
    if (this.#set) return Promise.resolve();
    return new Promise((resolve) => {
      this.#waiters.push(resolve);
    });
  }
}

class OneSlotQueue {
  #queued = false;
  #waiters: Array<() => void> = [];

  putNowait(): void {
    if (this.#queued) return;
    const waiter = this.#waiters.shift();
    if (waiter !== undefined) {
      waiter();
      return;
    }
    this.#queued = true;
  }

  async get(timeoutSeconds?: number): Promise<"item" | "timeout"> {
    if (this.#queued) {
      this.#queued = false;
      return "item";
    }
    return new Promise((resolve) => {
      const finish = (result: "item" | "timeout"): void => {
        const index = this.#waiters.indexOf(onItem);
        if (index >= 0) this.#waiters.splice(index, 1);
        if (timer !== undefined) clearTimeout(timer);
        resolve(result);
      };
      const onItem = (): void => {
        finish("item");
      };
      this.#waiters.push(onItem);
      const timer =
        timeoutSeconds === undefined
          ? undefined
          : setTimeout(() => {
              finish("timeout");
            }, timeoutSeconds * 1000);
    });
  }
}

class StartupJob {
  readonly discovered = new ManualResetEvent();
  readonly ready = new ManualResetEvent();
  projectId: string | null = null;
  discoveryError: unknown;
  indexingError: unknown;
  indexes: boolean;
  trigger: IndexTrigger;

  constructor(indexes: boolean, trigger: IndexTrigger) {
    this.indexes = indexes;
    this.trigger = trigger;
  }

  get failed(): boolean {
    return this.discoveryError !== undefined || this.indexingError !== undefined;
  }
}

export type WatchRoot = (root: string) => AsyncGenerator<unknown, void, unknown>;

export async function* defaultWatchRoot(root: string): AsyncGenerator<void> {
  const signals: Array<() => void> = [];
  let waiting: (() => void) | undefined;
  const subscription = await watcher.subscribe(root, (error, events) => {
    if (error !== null || events.length === 0) return;
    const notify = waiting;
    waiting = undefined;
    if (notify !== undefined) notify();
    else signals.push(() => undefined);
  });
  try {
    while (true) {
      if (signals.length === 0) {
        await new Promise<void>((resolve) => {
          waiting = resolve;
        });
      } else {
        signals.shift();
      }
      yield;
    }
  } finally {
    await subscription.unsubscribe();
  }
}

export const watchRoot: { current: WatchRoot } = { current: defaultWatchRoot };

export class StartupCoordinator {
  readonly application: Surface;
  readonly mode: IndexMode;
  readonly waitSeconds: number;
  readonly jobs = new Map<string, StartupJob>();
  readonly monitors = new Map<string, OneSlotQueue>();
  readonly dirtyRoots = new Set<string>();
  readonly dirtyGeneration = new Map<string, number>();
  readonly tasks: Promise<void>[] = [];
  #lock: Promise<void> = Promise.resolve();
  #slot: Promise<void> | null = null;
  readonly firstSchedule = new ManualResetEvent();
  #closed = false;

  constructor(
    application: Surface,
    { mode, waitSeconds = 300 }: { mode: IndexMode; waitSeconds?: number },
  ) {
    this.application = application;
    this.mode = mode;
    this.waitSeconds = waitSeconds;
  }

  close(): void {
    this.#closed = true;
  }

  async schedule(
    roots: readonly string[],
    { indexes, trigger = "startup" }: { indexes: boolean; trigger?: IndexTrigger },
  ): Promise<void> {
    if (this.mode === "manual" || this.#closed) return;
    await this.#withLock(async () => {
      for (const raw of roots) {
        const root = resolvePath(raw);
        const registered = [...this.jobs.keys()].find((candidate) =>
          sameProjectRoot(candidate, root),
        );
        const existing = registered === undefined ? undefined : this.jobs.get(registered);
        if (existing !== undefined) {
          if (!existing.ready.isSet()) continue;
          if (!existing.failed) {
            if (!indexes) continue;
            if (
              existing.indexes &&
              existing.projectId !== null &&
              !this.dirtyRoots.has(registered ?? root) &&
              !(await this.#isStale(existing.projectId))
            ) {
              continue;
            }
          }
        }
        const job = new StartupJob(indexes, trigger);
        const jobRoot = registered ?? root;
        this.jobs.set(jobRoot, job);
        this.#spawn(this.#run(jobRoot, job));
      }
    });
    this.firstSchedule.set();
  }

  async waitForStartupSettled(): Promise<void> {
    await this.firstSchedule.wait();
    while (true) {
      let pending = false;
      await this.#withLock(async () => {
        pending = [...this.jobs.values()].some((job) => job.indexes && !job.ready.isSet());
      });
      if (!pending) return;
      await sleep(250);
    }
  }

  async #isStale(projectId: string): Promise<boolean> {
    return this.application.projectIsStale(projectId);
  }

  async waitForDiscovery(roots: readonly string[]): Promise<void> {
    for (const job of await this.#jobsFor(roots)) {
      await job.discovered.wait();
      if (job.discoveryError !== undefined) throw job.discoveryError;
    }
  }

  async hasPendingIndexing(
    roots: readonly string[],
    projectIds: ReadonlySet<string>,
  ): Promise<boolean> {
    return (await this.#jobsFor(roots)).some(
      (job) =>
        job.projectId !== null &&
        projectIds.has(job.projectId) &&
        job.indexes &&
        !job.ready.isSet(),
    );
  }

  async waitForReady(roots: readonly string[], projectIds: ReadonlySet<string>): Promise<void> {
    for (const job of await this.#jobsFor(roots)) {
      if (job.projectId === null || !projectIds.has(job.projectId)) continue;
      await job.ready.wait();
      if (job.discoveryError !== undefined) throw job.discoveryError;
      if (job.indexingError !== undefined) throw job.indexingError;
    }
  }

  async #jobsFor(roots: readonly string[]): Promise<StartupJob[]> {
    return this.#withLock(async () => {
      const jobs: StartupJob[] = [];
      const seen = new Set<StartupJob>();
      for (const root of roots) {
        const job = [...this.jobs.entries()].find(([candidate]) =>
          sameProjectRoot(candidate, root),
        )?.[1];
        if (job !== undefined && !seen.has(job)) {
          jobs.push(job);
          seen.add(job);
        }
      }
      return jobs;
    });
  }

  async #run(root: string, job: StartupJob): Promise<void> {
    try {
      const project = await this.application.discoverProject(root);
      job.projectId = project === null ? null : project.id;
      job.discovered.set();
      if (project === null || !job.indexes) return;
      await this.#ensureMonitor(root, project.id);
      await this.#indexWhenFree(project.id, job.trigger);
    } catch (error) {
      // A close during a lock wait cancels the job the way session teardown
      // cancels it on the Python side: settled, with no failure recorded.
      if (error === CLOSED) return;
      if (!job.discovered.isSet()) {
        job.discoveryError = error;
        job.discovered.set();
      } else {
        job.indexingError = error;
      }
    } finally {
      job.ready.set();
    }
  }

  async #ensureMonitor(root: string, projectId: string): Promise<void> {
    if (this.mode !== "eager") return;
    const resolved = resolvePath(root);
    await this.#withLock(async () => {
      if ([...this.monitors.keys()].some((existing) => sameProjectRoot(existing, resolved))) {
        return;
      }
      const dirty = new OneSlotQueue();
      dirty.putNowait();
      this.monitors.set(resolved, dirty);
      this.#spawn(this.#watchRoot(resolved, projectId, dirty));
      this.#spawn(this.#refreshDirtyRoot(resolved, projectId, dirty));
    });
  }

  async #watchRoot(root: string, projectId: string, dirty: OneSlotQueue): Promise<void> {
    let retrySeconds = WATCH_RETRY_INITIAL_SECONDS.value;
    while (!this.#closed) {
      try {
        for await (const _changes of watchRoot.current(root)) {
          if (this.#closed) return;
          retrySeconds = WATCH_RETRY_INITIAL_SECONDS.value;
          this.dirtyRoots.add(root);
          this.dirtyGeneration.set(root, (this.dirtyGeneration.get(root) ?? 0) + 1);
          if (this.application instanceof Application) {
            this.application.invalidateFreshness(projectId);
          }
          dirty.putNowait();
        }
      } catch {
        // restart after backoff
      }
      await sleep(retrySeconds * 1000);
      retrySeconds = Math.min(retrySeconds * 2, WATCH_RETRY_MAXIMUM_SECONDS.value);
    }
  }

  async #refreshDirtyRoot(root: string, projectId: string, dirty: OneSlotQueue): Promise<void> {
    while (!this.#closed) {
      const result = await dirty.get(EAGER_RECONCILE_SECONDS.value);
      if (this.application instanceof Application) {
        this.application.invalidateFreshness(projectId);
      }
      const generation = this.dirtyGeneration.get(root) ?? 0;
      try {
        const passes = result === "timeout" ? 1 : 2;
        for (let index = 0; index < passes; index += 1) {
          await this.schedule([root], { indexes: true, trigger: "watcher" });
          await this.waitForReady([root], new Set([projectId]));
        }
        if ((this.dirtyGeneration.get(root) ?? 0) === generation) {
          this.dirtyRoots.delete(root);
          this.dirtyGeneration.delete(root);
        }
      } catch {
        // logged by the job
      }
    }
  }

  async #indexWhenFree(projectId: string, trigger: IndexTrigger): Promise<IndexReport> {
    const started = nowSeconds();
    const deadline = started + this.waitSeconds;
    await this.#acquireSlot(deadline, started);
    if (this.#closed) throw CLOSED;
    try {
      return await this.#indexWithBackoff(projectId, deadline, started, trigger);
    } finally {
      this.#slot = null;
    }
  }

  async #acquireSlot(deadline: number, started: number): Promise<void> {
    if (this.#slot === null) {
      this.#slot = Promise.resolve();
      return;
    }
    const remaining = deadline - nowSeconds();
    if (remaining > 0) {
      const held = this.#slot;
      let released = false;
      this.#slot = new Promise((resolve) => {
        const finish = (): void => {
          if (released) return;
          released = true;
          resolve();
        };
        void held.then(finish);
        setTimeout(finish, remaining * 1000);
      });
      await held;
      if (nowSeconds() <= deadline) return;
    }
    throw this.#busy(nowSeconds() - started);
  }

  async #indexWithBackoff(
    projectId: string,
    deadline: number,
    started: number,
    trigger: IndexTrigger,
  ): Promise<IndexReport> {
    let delay = INITIAL_RETRY_DELAY_SECONDS.value;
    while (true) {
      if (this.#closed) throw CLOSED;
      try {
        return await this.application.indexProject(projectId, { trigger });
      } catch (error) {
        if (this.#closed) throw CLOSED;
        if (!isCodeIndexingError(error) || error.code !== "INDEX_BUSY") throw error;
        const remaining = deadline - nowSeconds();
        if (remaining <= 0) throw this.#busy(nowSeconds() - started, error);
        await sleep(Math.min(delay, remaining) * 1000);
        delay = Math.min(delay * 2, MAXIMUM_RETRY_DELAY_SECONDS.value);
      }
    }
  }

  #busy(waited: number, cause?: CodeIndexingError): CodeIndexingError {
    const message = cause === undefined ? "Another indexing job is already active" : cause.message;
    return new CodeIndexingError(
      "INDEX_BUSY",
      `${message}; gave up after waiting ${waited.toFixed(1)}s`,
      {
        ...(cause === undefined ? {} : cause.details),
        waited_seconds: Math.round(waited * 1000) / 1000,
        wait_timeout_seconds: this.waitSeconds,
      },
      cause === undefined ? undefined : { cause },
    );
  }

  #spawn(task: Promise<void>): void {
    this.tasks.push(task.catch(() => undefined).then(() => undefined));
  }

  async #withLock<T>(body: () => Promise<T>): Promise<T> {
    const previous = this.#lock;
    let release: () => void = () => undefined;
    this.#lock = new Promise((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await body();
    } finally {
      release();
    }
  }
}

const SAFE_INTEGER_SENTINEL = 9007199254740991;

/**
 * Normalize what zod emits into what the Python surface advertises:
 * defaulted keys never appear in `required` at any level (pydantic omits
 * them), unbounded integers carry no bounds (zod adds ±MAX_SAFE_INTEGER),
 * and no `$schema` keyword is attached.
 */
function jsonSchema(schema: z.ZodType): Tool["inputSchema"] {
  return normalizeJsonSchema(z.toJSONSchema(schema)) as Tool["inputSchema"];
}

function normalizeJsonSchema(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(normalizeJsonSchema);
  if (node === null || typeof node !== "object") return node;
  const source = node as Record<string, unknown>;
  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(source)) {
    if (key === "$schema") continue;
    if (value === -SAFE_INTEGER_SENTINEL && key === "minimum") continue;
    if (value === SAFE_INTEGER_SENTINEL && key === "maximum") continue;
    normalized[key] = normalizeJsonSchema(value);
  }
  const properties = normalized.properties;
  if (
    Array.isArray(normalized.required) &&
    properties !== undefined &&
    typeof properties === "object"
  ) {
    const documented = properties as Record<string, unknown>;
    const remaining = (normalized.required as string[]).filter(
      (key) => documented[key] === undefined || !hasDefault(documented[key]),
    );
    if (remaining.length === 0) delete normalized.required;
    else normalized.required = remaining;
  }
  return normalized;
}

function hasDefault(spec: unknown): boolean {
  return (
    spec !== null && typeof spec === "object" && "default" in (spec as Record<string, unknown>)
  );
}

function uniqueProjectRoots(roots: readonly string[]): string[] {
  const unique: string[] = [];
  for (const root of roots) {
    const resolved = resolvePath(root);
    if (!unique.some((existing) => sameProjectRoot(resolved, existing))) unique.push(resolved);
  }
  return unique;
}

export interface ServerContext {
  coordinator: StartupCoordinator | null;
  listRoots: () => Promise<string[]>;
  reportProgress?: (progress: number, total: number | null, message: string) => Promise<void>;
}

async function startupRoots(
  ctx: ServerContext,
  { discover = false, indexes = false } = {},
): Promise<string[]> {
  const roots = await ctx.listRoots();
  const coordinator = ctx.coordinator;
  if (coordinator === null) return roots;
  await coordinator.schedule(roots, { indexes });
  if (discover || indexes) await coordinator.waitForDiscovery(roots);
  return roots;
}

async function waitForStartupProjects(
  ctx: ServerContext,
  roots: readonly string[],
  projectIds: readonly string[],
): Promise<void> {
  const coordinator = ctx.coordinator;
  if (coordinator === null || coordinator.mode === "manual") return;
  const projects = await Promise.all(
    projectIds.map((projectId) => coordinator.application.resolveProject(projectId)),
  );
  const selectedRoots = uniqueProjectRoots([...roots, ...projects.map((project) => project.root)]);
  const statuses = await Promise.all(
    projects.map((project) => coordinator.application.projectStatus(project.id)),
  );
  const refreshRoots = projects
    .filter((project, index) => {
      const status = statuses[index];
      return (
        status === undefined ||
        (status.state !== "ready" && status.state !== "partial") ||
        [...coordinator.dirtyRoots].some((dirty) => sameProjectRoot(dirty, project.root))
      );
    })
    .map((project) => project.root);
  await coordinator.schedule(refreshRoots, { indexes: true, trigger: "lazy-query" });
  await coordinator.waitForDiscovery(selectedRoots);
  const wanted = new Set(projectIds);
  const pending = await coordinator.hasPendingIndexing(selectedRoots, wanted);
  if (!pending) {
    await coordinator.waitForReady(selectedRoots, wanted);
    return;
  }
  const stream = startProgress(ctx, coordinator.application, [...projectIds], {
    message: statuses.every((status) => status.file_count === 0)
      ? "Building the initial index"
      : "Refreshing the stale index",
  });
  try {
    await coordinator.waitForReady(selectedRoots, wanted);
  } finally {
    stream.stop();
  }
  await stream.finish("Index ready");
}

function startProgress(
  ctx: ServerContext,
  application: Surface,
  projectIds: readonly string[],
  { message }: { message: string },
): { stop: () => void; finish: (done: string) => Promise<void> } {
  let highest = 0;
  let running = true;
  const report = ctx.reportProgress;
  if (report !== undefined) void report(0, null, message);
  const timer = setInterval(() => {
    if (!running || report === undefined) return;
    const snapshot = projectIds
      .map((projectId) => application.indexProgress(projectId))
      .find((item) => item !== null);
    if (snapshot === undefined || snapshot === null) return;
    highest = Math.max(highest, snapshot.candidates_seen);
    void report(highest, snapshot.candidates_total, describeProgress(snapshot));
  }, PROGRESS_POLL_SECONDS.value * 1000);
  return {
    stop: () => {
      running = false;
      clearInterval(timer);
    },
    finish: async (done) => {
      running = false;
      clearInterval(timer);
      const total = Math.max(highest, 1);
      if (report !== undefined) await report(total, total, done);
    },
  };
}

const PROJECT_SELECTOR = z
  .string()
  .nullable()
  .default(null)
  .describe(
    "Project id, name, or path. Defaults to the active MCP root or the nearest .ci-mcp/project.toml.",
  );

function toolResult(value: unknown): CallToolResult {
  const payload = jsonable(value);
  return {
    content: [{ type: "text", text: JSON.stringify(payload) }],
    structuredContent: payload as Record<string, unknown>,
  };
}

function fail(error: unknown): never {
  if (isCodeIndexingError(error)) throw new Error(error.forClient());
  throw error;
}

export type ProgressReporter = (
  progress: number,
  total: number | null,
  message: string,
) => Promise<void>;

export interface CreatedServer {
  readonly mcp: McpServer;
  readonly application: Surface;
  readonly mode: IndexMode;
  readonly instructions: string;
  readonly tools: Tool[];
  coordinator: StartupCoordinator | null;
  listRoots: () => Promise<string[]>;
  listTools(): Promise<Tool[]>;
  callTool(
    name: string,
    args?: Record<string, unknown>,
    options?: { onProgress?: (progress: number, total: number | null, message: string) => void },
  ): Promise<unknown>;
  startCoordinator(): StartupCoordinator;
  runStartupMaintenance(): Promise<void>;
  close(): void;
  run(): Promise<void>;
}

export function createServer(
  application?: Surface | null,
  { autoIndex }: { autoIndex?: boolean | null } = {},
): CreatedServer {
  const app = application ?? Application.fromEnvironment();
  const settings = indexSettingsFromEnvironment();
  const mode: IndexMode =
    autoIndex === true ? "eager" : autoIndex === false ? "manual" : settings.mode;
  const instructions = `${TOOL_INSTRUCTIONS}\n\n${SERVER_INSTRUCTIONS}`;
  const mcp = new McpServer({ name: "code-indexing-mcp", version: "0.0.0" }, { instructions });
  const tools: Tool[] = [];
  const handlers = new Map<
    string,
    (args: Record<string, unknown>, ctx?: ServerContext) => Promise<unknown>
  >();
  const created: CreatedServer = {
    mcp,
    application: app,
    mode,
    instructions,
    tools,
    coordinator: null,
    listRoots: async () => [],
    async listTools() {
      if (this.mode === "eager" && this.coordinator !== null) {
        await this.coordinator.schedule(await this.listRoots(), { indexes: true });
      }
      return this.tools;
    },
    async callTool(name, args = {}, options = {}) {
      const handler = handlers.get(name);
      if (handler === undefined) throw new Error(`Unknown tool: ${name}`);
      const onProgress = options.onProgress;
      const ctx = context();
      if (onProgress !== undefined) {
        ctx.reportProgress = async (progress, total, message) => {
          onProgress(progress, total, message);
        };
      }
      try {
        return await handler(args, ctx);
      } catch (error) {
        fail(error);
      }
    },
    startCoordinator() {
      this.coordinator ??= new StartupCoordinator(app, {
        mode,
        waitSeconds: settings.indexWaitSeconds,
      });
      return this.coordinator;
    },
    /**
     * Run overdue scheduled maintenance once, after startup indexing settles.
     *
     * Eager mode defers the pass until startup index jobs have settled so
     * optimize never competes with the initial build. Lazy mode runs it
     * immediately because indexing is intentionally deferred until a query.
     * When serving through the daemon, the daemon itself runs startup
     * maintenance, so only a real Application schedules here.
     */
    async runStartupMaintenance() {
      try {
        if (app instanceof BrokerApplication) return;
        if (mode === "eager") await this.coordinator?.waitForStartupSettled();
        if (!(app instanceof Application)) return;
        await app.maybeRunMaintenance();
      } catch {
        // best-effort, like the Python surface's logged exception
      }
    },
    close() {
      this.coordinator?.close();
    },
    async run() {
      this.startCoordinator();
      if (mode !== "manual" && app instanceof Application) {
        void this.runStartupMaintenance();
      }
      const transport = new StdioServerTransport();
      await mcp.connect(transport);
    },
  };

  const context = (): ServerContext => ({
    coordinator: created.coordinator,
    listRoots: () => created.listRoots(),
  });

  const register = (
    name: string,
    config: {
      title: string;
      description: string;
      annotations: ToolAnnotations;
      inputSchema: z.ZodType;
    },
    handler: (args: Record<string, unknown>, ctx: ServerContext) => Promise<unknown>,
  ): void => {
    const schema = config.inputSchema;
    const registered = mcp.registerTool.bind(mcp) as (
      toolName: string,
      toolConfig: {
        title: string;
        description: string;
        inputSchema: z.ZodType;
        annotations: ToolAnnotations;
      },
      callback: (
        args: Record<string, unknown>,
        extra: {
          _meta?: { progressToken?: number | string };
          sendNotification: (notification: {
            method: "notifications/progress";
            params: {
              progressToken: number | string;
              progress: number;
              total?: number;
              message?: string;
            };
          }) => Promise<void>;
        },
      ) => Promise<CallToolResult>,
    ) => void;
    registered(
      name,
      {
        title: config.title,
        description: config.description,
        inputSchema: schema,
        annotations: config.annotations,
      },
      async (args, extra) => {
        // The client opted into progress with _meta.progressToken; mirrors
        // FastMCP's ctx.report_progress, which the Python surface uses.
        const token = extra._meta?.progressToken;
        const reportProgress: ProgressReporter | undefined =
          token === undefined
            ? undefined
            : async (progress, total, message) => {
                await extra.sendNotification({
                  method: "notifications/progress",
                  params: {
                    progressToken: token,
                    progress,
                    ...(total === null ? {} : { total }),
                    ...(message === undefined ? {} : { message }),
                  },
                });
              };
        try {
          return toolResult(
            await handler(args, {
              ...context(),
              ...(reportProgress === undefined ? {} : { reportProgress }),
            }),
          );
        } catch (error) {
          fail(error);
        }
      },
    );
    handlers.set(name, async (args, ctx) => {
      const parsed = schema.parse(args) as Record<string, unknown>;
      return handler(parsed, ctx ?? context());
    });
    tools.push({
      name,
      title: config.title,
      description: config.description,
      inputSchema: jsonSchema(schema),
      annotations: config.annotations,
    });
  };

  register(
    "init_project",
    {
      title: "Initialize project",
      description:
        "Register a directory as an indexable project and write its local " +
        ".ci-mcp/project.toml marker, which holds a checkout-local id and the scan " +
        "configuration. Returns the project id, name, root, and scan settings. Building the " +
        "index is a separate operation (index_project). Re-running on an already-initialized " +
        "directory returns the existing project unless force_new_id is set. A new " +
        "registration whose root equals, contains, or is nested inside an existing " +
        "project's root is rejected unless allow_overlap is true.",
      annotations: INITIALIZES,
      inputSchema: z.object({
        path: z
          .string()
          .nullable()
          .default(null)
          .describe(
            "Directory to initialize. Defaults to the single MCP root when exactly one " +
              "is offered, otherwise to the server's working directory.",
          ),
        name: z
          .string()
          .nullable()
          .default(null)
          .describe("Display name for the project. Defaults to the directory name."),
        force_new_id: z
          .boolean()
          .default(false)
          .describe(
            "Mint a new project id even if a marker already exists, orphaning the " +
              "previous index for this directory.",
          ),
        allow_overlap: z
          .boolean()
          .default(false)
          .describe(
            "Register even when the directory equals, contains, or is nested inside " +
              "the root of an already registered project, which would index the same " +
              "sources twice. Set true only for an intentional duplicate registration.",
          ),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const name = args.name === null ? undefined : String(args.name);
      return app.initProject(args.path === null ? undefined : String(args.path), {
        ...(name === undefined ? {} : { name }),
        forceNewId: Boolean(args.force_new_id),
        allowOverlap: Boolean(args.allow_overlap),
        roots,
      });
    },
  );

  register(
    "index_project",
    {
      title: "Index project",
      description:
        "Incrementally index a project: scan for supported source files, parse changed files " +
        "with Tree-sitter, embed their chunks, and commit them. Files whose size, mtime, and " +
        "content hash are unchanged are skipped without being re-read. Returns per-phase " +
        "counts and durations plus any per-file errors. Indexes supported source files, " +
        "skipping symlinks, binaries, and files over 1 MiB.",
      annotations: WRITES,
      inputSchema: z.object({
        project: PROJECT_SELECTOR,
        force: z
          .boolean()
          .default(false)
          .describe(
            "Re-parse and re-embed every discovered file, ignoring change detection. " +
              "Use after changing the embedding model or chunking settings.",
          ),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const resolved = await app.resolveProject(
        args.project === null ? undefined : String(args.project),
        roots,
      );
      const stream = startProgress(ctx, app, [resolved.id], {
        message: `Indexing ${resolved.name}`,
      });
      try {
        const report = await app.indexProject(resolved.id, { roots, force: Boolean(args.force) });
        await stream.finish(
          `Indexed ${report.indexed_files} files, ${report.embedded_chunks} chunks embedded`,
        );
        return report;
      } finally {
        stream.stop();
      }
    },
  );

  register(
    "project_status",
    {
      title: "Project status",
      description:
        "Report one project's index state — pending, indexing, ready, partial, stale, " +
        "rebuild_required, or error — with its indexed file count and chunk count. Compares " +
        "eligible source metadata with the index but does not rebuild it; index_project does " +
        "that, including rebuilding a rebuild_required partition. A root that is not " +
        "registered yet is registered first, which writes its .ci-mcp/project.toml marker.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({ project: PROJECT_SELECTOR }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      return app.projectStatus(args.project === null ? undefined : String(args.project), { roots });
    },
  );

  register(
    "index_history",
    {
      title: "Indexing history",
      description:
        "One page of a project's durable indexing history — each run's id, trigger, " +
        "server and schema version, embedding model, force flag, start/finish timestamps, " +
        "final state, phase durations, file and chunk counts, skip counts by reason, bounded " +
        "error details, and storage table versions before and after. Newest first, paginated " +
        "with an opaque cursor; at most 100 runs per project are retained and history is " +
        "never loaded wholesale into project status.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        project: PROJECT_SELECTOR,
        cursor: z
          .string()
          .nullable()
          .default(null)
          .describe("Opaque cursor from a previous page; omit for the first page."),
        limit: z.int().min(1).max(100).default(20).describe("Maximum runs per page, up to 100."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const cursor = args.cursor === null ? undefined : String(args.cursor);
      return app.indexHistory(args.project === null ? undefined : String(args.project), {
        roots,
        ...(cursor === undefined ? {} : { cursor }),
        limit: Number(args.limit ?? 20),
      });
    },
  );

  register(
    "inspect_scan",
    {
      title: "Inspect scan",
      description:
        "One page of a stat-only dry-run scan of a project: what an index run would find, " +
        "before anything is embedded or written. Each item carries a repository-relative path " +
        "with the outcome ('eligible' with its language, or 'skipped' with its reason and " +
        "explanation). Filter by outcome or skip reason and paginate with the opaque cursor. " +
        "Read-only: never mutates the index and never persists a scan manifest.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        project: PROJECT_SELECTOR,
        // Plain strings, not enums: the Python schema leaves the filter value
        // open (an unknown value simply matches nothing), and downstream scan
        // rows are the authority on what outcomes and reasons exist.
        outcome: z
          .string()
          .nullable()
          .default(null)
          .describe("Keep only 'eligible' or 'skipped' items; omit for both."),
        reason: z
          .string()
          .nullable()
          .default(null)
          .describe(
            "Keep only skipped items with this reason: unsupported, ignored, symlink, oversized, or unreadable.",
          ),
        cursor: z
          .string()
          .nullable()
          .default(null)
          .describe("Opaque cursor from a previous page; omit for the first page."),
        limit: z.int().min(1).max(200).default(50).describe("Maximum items per page, up to 200."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const outcome = args.outcome === null ? undefined : String(args.outcome);
      const reason = args.reason === null ? undefined : String(args.reason);
      const cursor = args.cursor === null ? undefined : String(args.cursor);
      return app.inspectScan(args.project === null ? undefined : String(args.project), {
        roots,
        ...(outcome === undefined ? {} : { outcome }),
        ...(reason === undefined ? {} : { reason }),
        ...(cursor === undefined ? {} : { cursor }),
        limit: Number(args.limit ?? 50),
      });
    },
  );

  register(
    "index_storage_status",
    {
      title: "Index storage status",
      description:
        "Read-only storage statistics for one project or the whole installation — current " +
        "table versions, row counts, Lance-reported logical bytes, filesystem-reported " +
        "physical bytes, fragment and retained-version counts, index coverage, and an " +
        "installation total — plus advisory warnings for overlapping registered roots and " +
        "Git worktrees that share one repository. Never mutates the index: a registered " +
        "project with no partition reports zeroed tables instead of materializing one.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        project: z
          .string()
          .nullable()
          .default(null)
          .describe(
            "Project id, name, or path. Defaults to the active MCP root or the nearest " +
              ".ci-mcp/project.toml when exactly one project is in scope; omit for the " +
              "whole installation.",
          ),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      return app.storageStatus(args.project === null ? undefined : String(args.project), { roots });
    },
  );

  register(
    "index_storage_maintenance",
    {
      title: "Index storage maintenance",
      description:
        "Compact tables and remove verified old Lance versions for one project or the whole " +
        "installation. Dry-run by default: it reports the before statistics and a labelled " +
        "reclaimable-bytes estimate, leaving the after statistics null; only an explicit " +
        "dry_run=false performs cleanup and reports the after statistics, versions removed, " +
        "bytes reclaimed, duration, skipped projects, and busy projects. Never uses zero-age " +
        "retention or delete_unverified.",
      annotations: WRITES,
      inputSchema: z.object({
        project: z
          .string()
          .nullable()
          .default(null)
          .describe(
            "Project id, name, or path. Defaults to the active MCP root or the nearest " +
              ".ci-mcp/project.toml when exactly one project is in scope; omit for the whole " +
              "installation.",
          ),
        dry_run: z
          .boolean()
          .default(true)
          .describe(
            "True (default) reports statistics and a reclaimable-bytes estimate without " +
              "mutating the index; false performs the cleanup.",
          ),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      return app.maintainStorage(args.project === null ? undefined : String(args.project), {
        roots,
        dryRun: args.dry_run !== false,
        waitForLock: true,
      });
    },
  );

  register(
    "list_projects",
    {
      title: "List projects",
      description:
        "List every project registered with this server — id, name, root directory, and scan " +
        "configuration — sorted by name. Takes no arguments and returns registrations only, " +
        "not index state; project_status reports that.",
      annotations: READ_ONLY,
      inputSchema: z.object({}),
    },
    async () => app.listProjects(),
  );

  register(
    "remove_project",
    {
      title: "Remove project",
      description:
        "Permanently delete a project's registration and its entire on-disk index partition. " +
        "The .ci-mcp/project.toml marker in the working tree is left in place, so a later " +
        "init_project re-registers the same id with an empty index. Irreversible: the only " +
        "way back is a full re-index. Returns whether a registration existed.",
      annotations: DESTRUCTIVE,
      inputSchema: z.object({
        project: z.string().describe("Project id, name, or path to remove. Required — no default."),
      }),
    },
    async (args) => app.removeProject(String(args.project)),
  );

  const searchLanguages = z.array(LanguageName).nullable().default(null);
  const searchKinds = z.array(ChunkKind).nullable().default(null);
  const searchPaths = z
    .array(z.string())
    .nullable()
    .default(null)
    .describe(
      "Restrict to paths matching these glob patterns, relative to the project root, for example 'src/*' or '**/*.py'. Patterns match from the right, so '*.py' matches any Python file at any depth.",
    );

  register(
    "search_code",
    {
      title: "Search code",
      description:
        "Hybrid semantic and keyword search over indexed code chunks. Returns hits ranked by " +
        "relevance, each with a code snippet, file path, line range, and a chunk_id that " +
        "get_chunk expands to the full text. Searches indexed source only — not commit " +
        "history, not comments in unindexed files, and not files excluded by .gitignore or " +
        "the 1 MiB size cap. For a declaration whose name is already known, find_symbol is " +
        "direct; for one file's structure, file_outline is cheaper. A root that is not " +
        "registered yet is registered and indexed before the first query is answered; later " +
        "queries refresh selected projects when source metadata has changed.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        query: z
          .string()
          .describe(
            "What to look for, as natural language or keywords. Matched against chunk " +
              "text and against normalized identifier names.",
          ),
        projects: z
          .array(z.string())
          .nullable()
          .default(null)
          .describe(
            "Restrict the search to these project ids, names, or paths. Mutually " +
              "exclusive with all_projects.",
          ),
        all_projects: z
          .boolean()
          .default(false)
          .describe(
            "Search every registered project. Off by default so results from unrelated " +
              "repositories are never mixed in implicitly.",
          ),
        languages: searchLanguages.describe("Restrict to these languages."),
        paths: searchPaths,
        kinds: searchKinds.describe("Restrict to these chunk kinds."),
        limit: z
          .int()
          .min(1)
          .max(50)
          .default(8)
          .describe("Maximum hits to return. Hard cap of 50."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const projectIds =
        args.all_projects === true
          ? await app.resolveSearchScope(undefined, true, roots)
          : args.projects === null || args.projects === undefined
            ? [(await app.resolveProject(undefined, roots)).id]
            : await app.resolveSearchScope(args.projects as string[], false, roots);
      await waitForStartupProjects(ctx, roots, projectIds);
      return app.searchCode(String(args.query), {
        projects: projectIds,
        allProjects: false,
        ...optionalFilters(args),
        limit: Number(args.limit ?? 8),
        roots,
      });
    },
  );

  register(
    "search_across_projects",
    {
      title: "Search across projects",
      description:
        "Hybrid semantic and keyword search across an explicit set of related projects for " +
        "cross-repository debugging. Accepts project ids, unique names, or paths and requires " +
        "at least two distinct resolved projects. Returns one globally ranked hit list with " +
        "project metadata; use list_projects first to discover the intended repositories.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        query: z
          .string()
          .describe(
            "What to look for across the selected projects, as natural language or " +
              "keywords. Matched against chunk text and normalized identifier names.",
          ),
        projects: z
          .array(z.string())
          .min(2)
          .describe(
            "At least two project ids, unique names, or paths to search together. " +
              "Selectors must resolve to at least two distinct projects.",
          ),
        languages: searchLanguages.describe(
          "Restrict to these languages across the complete selected scope.",
        ),
        paths: z
          .array(z.string())
          .nullable()
          .default(null)
          .describe(
            "Restrict to paths matching these glob patterns relative to each selected " +
              "project root, for example 'src/*' or '**/*.py'. Patterns match from the " +
              "right, so '*.py' matches any Python file at any depth.",
          ),
        kinds: searchKinds.describe("Restrict to these chunk kinds across the selected projects."),
        limit: z
          .int()
          .min(1)
          .max(50)
          .default(8)
          .describe("Maximum globally ranked hits to return. Hard cap of 50."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const resolvedIds = await app.resolveSearchScope(args.projects as string[], false, roots);
      const projectIds = [...new Set(resolvedIds)];
      if (projectIds.length < 2) {
        throw new CodeIndexingError(
          "INVALID_FILTER",
          "search_across_projects requires at least two distinct projects",
        );
      }
      await waitForStartupProjects(ctx, roots, projectIds);
      return app.searchCode(String(args.query), {
        projects: projectIds,
        allProjects: false,
        ...optionalFilters(args),
        limit: Number(args.limit ?? 8),
        roots,
      });
    },
  );

  register(
    "find_symbol",
    {
      title: "Find symbol",
      description:
        "Look up indexed code chunks by symbol name, matching exactly, by prefix, or by " +
        "substring. Returns hits ordered by path and line, each with a snippet and a " +
        "chunk_id. Matches declaration names only — not call sites, imports, or other " +
        "references. For a conceptual query rather than a known name, search_code applies. A " +
        "root that is not registered yet is registered and indexed before the first query is " +
        "answered; later queries refresh it when source metadata has changed.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        name: z
          .string()
          .describe(
            "Symbol name to look up. Either the bare name or the dotted qualified name, " +
              "for example 'LanceStore' or 'LanceStore.upsert_project'.",
          ),
        project: PROJECT_SELECTOR,
        match: z
          .enum(["exact", "prefix", "contains"])
          .default("exact")
          .describe(
            "How to compare name against stored symbols. 'exact' requires a full match " +
              "on the bare or qualified name.",
          ),
        kinds: searchKinds.describe("Restrict to these chunk kinds."),
        limit: z
          .int()
          .min(1)
          .max(50)
          .default(20)
          .describe("Maximum hits to return. Hard cap of 50."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const resolved = await app.resolveProject(
        args.project === null ? undefined : String(args.project),
        roots,
      );
      await waitForStartupProjects(ctx, roots, [resolved.id]);
      const kinds = optionalList(args.kinds);
      return app.findSymbol(String(args.name), resolved.id, {
        match: (args.match as "exact" | "prefix" | "contains") ?? "exact",
        ...(kinds === undefined ? {} : { kinds }),
        limit: Number(args.limit ?? 20),
        roots,
      });
    },
  );

  register(
    "find_references",
    {
      title: "Find references",
      description:
        "Find structural uses of one Python, JavaScript, TypeScript, or TSX declaration; " +
        "other languages return UNSUPPORTED_LANGUAGE. Select it with a chunk_id or project, " +
        "path, and qualified_symbol. Results distinguish exact, likely, and unresolved " +
        "bindings and may trigger parse-only structural backfill; they never edit source " +
        "files. This is a syntax-only index: dynamic dispatch, reflection, and files in " +
        "other languages are invisible to it, so check `limitations` before concluding a " +
        "declaration is unused.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        selector: DeclarationSelector.describe(
          "Declaration selected by chunk id or stable source location.",
        ),
        kinds: z
          .array(ReferenceKind)
          .nullable()
          .default(null)
          .describe(
            "Optional reference kinds to keep. Omit for all kinds; an unknown kind is " +
              "rejected rather than silently returning nothing.",
          ),
        limit: z.int().min(1).max(500).default(100).describe("Maximum results per page."),
        cursor: z.string().nullable().default(null).describe("Opaque page cursor."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const kinds = optionalList(args.kinds);
      const cursor = args.cursor === null ? undefined : String(args.cursor);
      return app.findReferences(DeclarationSelector.parse(args.selector), {
        roots,
        ...(kinds === undefined ? {} : { kinds: new Set(kinds) }),
        limit: Number(args.limit ?? 100),
        ...(cursor === undefined ? {} : { cursor }),
      });
    },
  );

  register(
    "analyze_refactor",
    {
      title: "Analyze refactor impact",
      description:
        "Analyze a proposed rename or signature change without editing source files, for a " +
        "Python, JavaScript, TypeScript, or TSX declaration. Returns required edits, likely " +
        "changes, dynamic-review findings, and evidence: for a rename, resolved aliases that " +
        "need no spelling change; for a signature change, compatible call sites that need no " +
        "argument edit. Always read `completeness` and `limitations`: only the state " +
        "'complete' means every indexed file was analyzed, and edits should use " +
        "edit_start_byte/edit_end_byte, which cover just the identifier, rather than the " +
        "wider reference range.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        selector: DeclarationSelector.describe(
          "Declaration selected by chunk id or stable source location.",
        ),
        operation: RefactorOperation.describe(
          "Discriminated rename or signature-change operation.",
        ),
        limit: z.int().min(1).max(500).default(500).describe("Maximum findings per page."),
        cursor: z.string().nullable().default(null).describe("Opaque analysis page cursor."),
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      return app.analyzeRefactor(
        DeclarationSelector.parse(args.selector),
        RefactorOperation.parse(args.operation),
        {
          roots,
          limit: Number(args.limit ?? 500),
          ...(args.cursor === null ? {} : { cursor: String(args.cursor) }),
        },
      );
    },
  );

  register(
    "file_outline",
    {
      title: "File outline",
      description:
        "List the symbols declared in one indexed file, in source order, with kind, " +
        "qualified name, parent, and line range. Returns structure metadata only, never code " +
        "text, so it is the cheap way to understand a file before fetching parts of it. The " +
        "file must already be indexed; a root that is not registered yet is registered and " +
        "indexed first, and a changed index is refreshed before the outline is returned.",
      annotations: READS_AND_REGISTERS,
      inputSchema: z.object({
        path: z
          .string()
          .describe(
            "File path relative to the project root, using forward slashes, exactly as " +
              "reported in search_code hits.",
          ),
        project: PROJECT_SELECTOR,
      }),
    },
    async (args, ctx) => {
      const roots = await startupRoots(ctx, { discover: true });
      const resolved = await app.resolveProject(
        args.project === null ? undefined : String(args.project),
        roots,
      );
      await waitForStartupProjects(ctx, roots, [resolved.id]);
      return app.fileOutline(String(args.path), resolved.id, { roots });
    },
  );

  register(
    "get_chunk",
    {
      title: "Get chunk",
      description:
        "Fetch one indexed chunk's full stored text by the chunk_id returned from search_code " +
        "or find_symbol, with its path, symbol, and line range. Chunk ids are content-derived " +
        "and change when the file is re-indexed, so a stale id returns CHUNK_NOT_FOUND rather " +
        "than the wrong code.",
      annotations: READ_ONLY,
      inputSchema: z.object({
        chunk_id: z.string().describe("Chunk id from a search_code or find_symbol hit."),
      }),
    },
    async (args) => app.getChunk(String(args.chunk_id)),
  );

  return created;
}

export async function runServer(): Promise<void> {
  await createServer().run();
}

export function rootsFromUris(uris: readonly string[]): string[] {
  const roots: string[] = [];
  for (const uri of uris) {
    try {
      roots.push(resolvePath(fileURLToPath(uri)));
    } catch {
      // ignore non-file URIs
    }
  }
  return uniqueProjectRoots(roots);
}

function optionalList(value: unknown): string[] | undefined {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value) || value.length === 0) return undefined;
  return value.map((item) => String(item));
}

function optionalFilters(args: Record<string, unknown>): {
  languages?: readonly string[];
  paths?: readonly string[];
  kinds?: readonly string[];
} {
  const languages = optionalList(args.languages);
  const paths = optionalList(args.paths);
  const kinds = optionalList(args.kinds);
  return {
    ...(languages === undefined ? {} : { languages }),
    ...(paths === undefined ? {} : { paths }),
    ...(kinds === undefined ? {} : { kinds }),
  };
}

function nowSeconds(): number {
  return performance.now() / 1000;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
