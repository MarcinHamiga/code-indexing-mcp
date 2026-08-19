/**
 * Embedding backend contract: descriptors, capability probes, and selection.
 *
 * Selection is deliberately split from execution. This module answers "which
 * backend should this machine use, and why", using only the execution providers
 * the installed ONNX Runtime reports. Whether that backend actually works is
 * decided later by a real inference probe in a disposable worker -- hardware
 * detection nominates a backend, only the probe confirms it.
 */

import { createRequire } from "node:module";
import os from "node:os";
import { CodeIndexingError } from "./errors.ts";

const require = createRequire(import.meta.url);

/**
 * The execution targets a passage embedder can be pointed at.
 *
 * Ordered as the Python `StrEnum` is, because `parseAccelerator` lists the
 * members back to the operator when their spelling was wrong and that message
 * should not reorder itself between the two builds.
 */
export const ACCELERATORS = ["auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"] as const;

export type Accelerator = (typeof ACCELERATORS)[number];

export const RUNTIMES = ["onnxruntime", "onnxruntime-plugin", "mlx"] as const;

export type RuntimeKind = (typeof RUNTIMES)[number];

export const STABILITIES = ["experimental", "manual", "automatic"] as const;

export type Stability = (typeof STABILITIES)[number];

export const PRECISIONS = ["float32", "float16"] as const;

export type Precision = (typeof PRECISIONS)[number];

export const CPU_PROVIDER = "CPUExecutionProvider";
export const MLX_PROVIDER = "MlxMetalBackend";

export const DIRECT_MODEL_ACCELERATORS = new Set<Accelerator>(["webgpu", "migraphx", "mlx"]);

export class BackendDescriptor {
  readonly accelerator: Accelerator;
  readonly provider: string;
  readonly device: string;
  readonly stability: Stability;
  readonly precision: Precision;
  readonly runtimeVersion: string;
  readonly driverVersion: string;
  readonly runtime: RuntimeKind;

  constructor(fields: {
    accelerator: Accelerator;
    provider: string;
    device: string;
    stability: Stability;
    precision: Precision;
    runtimeVersion?: string;
    driverVersion?: string;
    runtime?: RuntimeKind;
  }) {
    this.accelerator = fields.accelerator;
    this.provider = fields.provider;
    this.device = fields.device;
    this.stability = fields.stability;
    this.precision = fields.precision;
    this.runtimeVersion = fields.runtimeVersion ?? "";
    this.driverVersion = fields.driverVersion ?? "";
    this.runtime = fields.runtime ?? "onnxruntime";
  }

  get isCpu(): boolean {
    return this.accelerator === "cpu";
  }

  get providers(): readonly string[] {
    if (this.isCpu) return [CPU_PROVIDER];
    if (!this.runsOnOnnx) return [this.provider];
    return [this.provider, CPU_PROVIDER];
  }

  get providerIsPreregistered(): boolean {
    return this.runtime === "onnxruntime";
  }

  get runsOnOnnx(): boolean {
    return this.runtime === "onnxruntime" || this.runtime === "onnxruntime-plugin";
  }

  get publishesExecutionProviders(): boolean {
    return this.runsOnOnnx;
  }

  get usesDirectModel(): boolean {
    return DIRECT_MODEL_ACCELERATORS.has(this.accelerator);
  }
}

export const CPU_BACKEND = new BackendDescriptor({
  accelerator: "cpu",
  provider: CPU_PROVIDER,
  device: "cpu",
  stability: "automatic",
  precision: "float32",
});

export const ACCELERATOR_BACKENDS: readonly BackendDescriptor[] = [
  new BackendDescriptor({
    accelerator: "cuda",
    provider: "CUDAExecutionProvider",
    device: "cuda:0",
    stability: "automatic",
    precision: "float32",
  }),
  new BackendDescriptor({
    accelerator: "mlx",
    provider: MLX_PROVIDER,
    device: "metal",
    stability: "automatic",
    precision: "float32",
    runtime: "mlx",
  }),
  new BackendDescriptor({
    accelerator: "webgpu",
    provider: "WebGpuExecutionProvider",
    device: "gpu",
    stability: "experimental",
    precision: "float32",
    runtime: "onnxruntime-plugin",
  }),
  new BackendDescriptor({
    accelerator: "migraphx",
    provider: "MIGraphXExecutionProvider",
    device: "gpu",
    stability: "experimental",
    precision: "float32",
  }),
  new BackendDescriptor({
    accelerator: "coreml",
    provider: "CoreMLExecutionProvider",
    device: "ane",
    stability: "manual",
    precision: "float32",
  }),
];

export const KNOWN_BACKENDS: readonly BackendDescriptor[] = [CPU_BACKEND, ...ACCELERATOR_BACKENDS];

export function backendFor(
  accelerator: Accelerator,
  registry: readonly BackendDescriptor[] = KNOWN_BACKENDS,
): BackendDescriptor | undefined {
  return registry.find((backend) => backend.accelerator === accelerator);
}

export class BackendSelection {
  readonly requested: Accelerator;
  readonly descriptor: BackendDescriptor;
  readonly availableProviders: readonly string[];
  readonly honored: boolean;
  readonly fallbackReason: string | null;

  constructor(fields: {
    requested: Accelerator;
    descriptor: BackendDescriptor;
    availableProviders: readonly string[];
    honored?: boolean;
    fallbackReason?: string | null;
  }) {
    this.requested = fields.requested;
    this.descriptor = fields.descriptor;
    this.availableProviders = fields.availableProviders;
    this.honored = fields.honored ?? true;
    this.fallbackReason = fields.fallbackReason ?? null;
  }

  get accelerator(): Accelerator {
    return this.descriptor.accelerator;
  }

  get usesAccelerator(): boolean {
    return !this.descriptor.isCpu;
  }

  fellBackTo(descriptor: BackendDescriptor, reason: string): BackendSelection {
    return new BackendSelection({
      requested: this.requested,
      descriptor,
      availableProviders: this.availableProviders,
      honored: false,
      fallbackReason: reason,
    });
  }

  describedAs(descriptor: BackendDescriptor): BackendSelection {
    return new BackendSelection({
      requested: this.requested,
      descriptor,
      availableProviders: this.availableProviders,
      honored: this.honored,
      fallbackReason: this.fallbackReason,
    });
  }

  diagnosed(reason: string): BackendSelection {
    const combined = this.fallbackReason ? `${this.fallbackReason}; ${reason}` : reason;
    return new BackendSelection({
      requested: this.requested,
      descriptor: this.descriptor,
      availableProviders: this.availableProviders,
      honored: this.honored,
      fallbackReason: combined,
    });
  }

  requireHonored(): void {
    if (this.honored) return;
    throw new CodeIndexingError(
      "BACKEND_UNAVAILABLE",
      `Requested embedding accelerator is unavailable: ${this.requested}`,
      {
        requested: this.requested,
        resolved: this.accelerator,
        reason: this.fallbackReason ?? "unavailable",
      },
    );
  }
}

export function parseAccelerator(value: string): Accelerator {
  const normalized = value.trim().toLowerCase();
  const member = ACCELERATORS.find((candidate) => candidate === normalized);
  if (member === undefined) {
    throw new CodeIndexingError(
      "INVALID_CONFIGURATION",
      `Unknown embedding accelerator: ${JSON.stringify(value)}; ` +
        `expected one of ${ACCELERATORS.join(", ")}`,
      { value },
    );
  }
  return member;
}

const PROVIDER_BY_BACKEND: Record<string, string> = {
  cpu: CPU_PROVIDER,
  cuda: "CUDAExecutionProvider",
  coreml: "CoreMLExecutionProvider",
  dml: "DmlExecutionProvider",
  webgpu: "WebGpuExecutionProvider",
  migraphx: "MIGraphXExecutionProvider",
};

export function availableExecutionProviders(): readonly string[] {
  try {
    const listed = listOnnxProviders();
    if (listed.includes(CPU_PROVIDER)) return listed;
    return [...listed, CPU_PROVIDER];
  } catch {
    return [CPU_PROVIDER];
  }
}

function listOnnxProviders(): string[] {
  const ort = loadOnnxRuntime();
  if (typeof ort.listSupportedBackends === "function") {
    const backends = ort.listSupportedBackends();
    const providers: string[] = [];
    for (const backend of backends) {
      const name = typeof backend === "string" ? backend : backend.name;
      const provider = PROVIDER_BY_BACKEND[name.toLowerCase()] ?? name;
      if (!providers.includes(provider)) providers.push(provider);
    }
    return providers;
  }
  return [CPU_PROVIDER];
}

interface OnnxRuntimeModule {
  listSupportedBackends?: () => ReadonlyArray<string | { name: string }>;
  version?: string;
}

let onnxRuntime: OnnxRuntimeModule | undefined | null;

function loadOnnxRuntime(): OnnxRuntimeModule {
  if (onnxRuntime === undefined) {
    try {
      onnxRuntime = requireOnnxRuntime();
    } catch {
      onnxRuntime = null;
    }
  }
  if (onnxRuntime === null) throw new Error("onnxruntime-node is not importable");
  return onnxRuntime;
}

function requireOnnxRuntime(): OnnxRuntimeModule {
  return require("onnxruntime-node") as OnnxRuntimeModule;
}

export function runtimeVersion(runtime: RuntimeKind = "onnxruntime"): string {
  try {
    if (runtime === "mlx") return "";
    const ort = loadOnnxRuntime();
    return typeof ort.version === "string" ? ort.version : "";
  } catch {
    return "";
  }
}

export function platformFingerprint(): string {
  return `${os.type()}-${os.machine()}-${os.release()}`.toLowerCase();
}

export function selectBackend(
  requested: Accelerator,
  {
    availableProviders,
    registry = KNOWN_BACKENDS,
  }: {
    availableProviders: readonly string[];
    registry?: readonly BackendDescriptor[];
  },
): BackendSelection {
  const providers = availableProviders.map((name) => String(name));
  const catalogue = byAccelerator(registry);
  const cpu = catalogue.get("cpu") ?? CPU_BACKEND;

  if (requested === "cpu") {
    return new BackendSelection({
      requested,
      descriptor: cpu,
      availableProviders: providers,
    });
  }

  if (requested === "auto") {
    for (const descriptor of registry) {
      if (descriptor.isCpu || descriptor.stability !== "automatic") continue;
      if (providers.includes(descriptor.provider)) {
        return new BackendSelection({
          requested,
          descriptor,
          availableProviders: providers,
        });
      }
    }
    return new BackendSelection({
      requested,
      descriptor: cpu,
      availableProviders: providers,
      fallbackReason:
        "no accelerator is prepared and eligible on this machine; reinstall " +
        "with --accelerator to prepare one, or set CODE_INDEXING_EMBED_ACCELERATOR " +
        "to force a backend this installation already offers",
    });
  }

  const explicit = catalogue.get(requested);
  if (explicit === undefined) {
    return new BackendSelection({
      requested,
      descriptor: cpu,
      availableProviders: providers,
      honored: false,
      fallbackReason: `no backend is registered for ${requested}`,
    });
  }
  if (!providers.includes(explicit.provider)) {
    return new BackendSelection({
      requested,
      descriptor: cpu,
      availableProviders: providers,
      honored: false,
      fallbackReason:
        `${explicit.provider} is not among the execution providers this ` +
        `installation offers (${providers.join(", ")}); reinstall with ` +
        `--accelerator ${explicit.accelerator} to prepare it`,
    });
  }
  return new BackendSelection({
    requested,
    descriptor: explicit,
    availableProviders: providers,
  });
}

export function describeEnvironment(descriptor: BackendDescriptor): BackendDescriptor {
  return new BackendDescriptor({
    accelerator: descriptor.accelerator,
    provider: descriptor.provider,
    device: descriptor.device,
    stability: descriptor.stability,
    precision: descriptor.precision,
    runtimeVersion: descriptor.runtimeVersion || runtimeVersion(descriptor.runtime),
    driverVersion: descriptor.driverVersion,
    runtime: descriptor.runtime,
  });
}

function byAccelerator(
  registry: readonly BackendDescriptor[],
): Map<Accelerator, BackendDescriptor> {
  return new Map(registry.map((descriptor) => [descriptor.accelerator, descriptor]));
}
