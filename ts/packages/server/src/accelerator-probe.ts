/** Run a real embedding before an installer records an accelerator backend. */

import {
  type Accelerator,
  availableExecutionProviders,
  backendFor,
  platformFingerprint,
  runtimeVersion,
} from "./backends.ts";
import { type AcceleratorEnvironmentRecord, runningRuntimeVersion } from "./accelerator-env.ts";
import {
  DEFAULT_DIMENSION,
  DEFAULT_MODEL,
  PROBE_TEXTS,
  packVector,
  resolveSessionProviders,
  validateProbeVectors,
} from "./embedding.ts";
import { loadModel, workerConfig } from "./embedding-worker.ts";

export interface AcceleratorProbeReport extends AcceleratorEnvironmentRecord {
  readonly ok: true;
  readonly resolvedProviders: readonly string[];
  readonly platform: string;
  readonly dimension: number;
  readonly modelId: string;
}

export async function probeAccelerator(
  accelerator: Accelerator,
  {
    cacheDirectory,
    offline,
    modelId = DEFAULT_MODEL,
    dimension = DEFAULT_DIMENSION,
  }: {
    cacheDirectory: string;
    offline: boolean;
    modelId?: string;
    dimension?: number;
  },
): Promise<AcceleratorProbeReport> {
  const descriptor = backendFor(accelerator);
  if (descriptor === undefined) throw new Error(`no backend is registered for ${accelerator}`);
  const available = descriptor.publishesExecutionProviders ? availableExecutionProviders() : [];
  if (descriptor.providerIsPreregistered && !available.includes(descriptor.provider)) {
    throw new Error(
      `${descriptor.provider} is not offered by this environment's ONNX Runtime ` +
        `(${available.join(", ")})`,
    );
  }
  const model = await loadModel(
    workerConfig({
      cacheDirectory,
      offline,
      threads: 1,
      enableCpuMemArena: false,
      dimension,
      modelId,
      providers: descriptor.providers,
      accelerator,
    }),
  );
  const resolvedProviders = [...new Set(resolveSessionProviders(model).map(String))];
  if (!resolvedProviders.includes(descriptor.provider)) {
    const actual = resolvedProviders.length === 0 ? "no providers" : resolvedProviders.join(", ");
    throw new Error(`${descriptor.provider} was requested but the session runs on ${actual}`);
  }
  const vectors = [...(await model.passageEmbed([...PROBE_TEXTS]))].map(packVector);
  validateProbeVectors(vectors, { dimension, count: PROBE_TEXTS.length });
  return {
    ok: true,
    accelerator,
    interpreter: process.execPath,
    providers: [...new Set([...available, ...resolvedProviders])],
    runtimeVersion: runtimeVersion(descriptor.runtime),
    driverVersion: "",
    device: descriptor.device,
    pythonVersion: runningRuntimeVersion(),
    recordedAtNs: (BigInt(Date.now()) * 1_000_000n).toString(),
    detail: `probed ${PROBE_TEXTS.length} passages on ${descriptor.provider}`,
    resolvedProviders,
    platform: platformFingerprint(),
    dimension,
    modelId,
  };
}
