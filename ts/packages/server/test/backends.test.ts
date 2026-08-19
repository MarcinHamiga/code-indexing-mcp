/** Backend descriptors, capability probes, and selection. */

import { expect, test } from "bun:test";
import {
  ACCELERATORS,
  type Accelerator,
  availableExecutionProviders,
  BackendDescriptor,
  backendFor,
  CPU_BACKEND,
  CPU_PROVIDER,
  KNOWN_BACKENDS,
  MLX_PROVIDER,
  parseAccelerator,
  type Stability,
  selectBackend,
} from "../src/backends.ts";
import { isCodeIndexingError } from "../src/errors.ts";

const CUDA_PROVIDER = "CUDAExecutionProvider";
const WEBGPU_PROVIDER = "WebGpuExecutionProvider";

function descriptor(
  accelerator: Accelerator,
  provider: string,
  stability: Stability,
): BackendDescriptor {
  return new BackendDescriptor({
    accelerator,
    provider,
    device: "gpu",
    stability,
    precision: "float32",
  });
}

function registry(...accelerators: BackendDescriptor[]): BackendDescriptor[] {
  return [CPU_BACKEND, ...accelerators];
}

test("every member parses back to itself", () => {
  for (const accelerator of ACCELERATORS) {
    expect(parseAccelerator(accelerator)).toBe(accelerator);
  }
});

test("case and surrounding whitespace are forgiven", () => {
  expect(parseAccelerator("  CUDA \n")).toBe("cuda");
});

test("an unknown name is a configuration error that lists the alternatives", () => {
  let caught: unknown;
  try {
    parseAccelerator("tpu");
  } catch (error) {
    caught = error;
  }

  expect(isCodeIndexingError(caught)).toBe(true);
  if (!isCodeIndexingError(caught)) return;
  expect(caught.code).toBe("INVALID_CONFIGURATION");
  expect(caught.details.value).toBe("tpu");
  expect(caught.message).toContain(ACCELERATORS.join(", "));
});

test("cpu is always selectable", () => {
  const selection = selectBackend("cpu", { availableProviders: [CPU_PROVIDER] });

  expect(selection.accelerator).toBe("cpu");
  expect(selection.usesAccelerator).toBe(false);
  expect(selection.honored).toBe(true);
  expect(selection.fallbackReason).toBeNull();
});

test("auto picks the first automatic backend whose provider exists", () => {
  const selection = selectBackend("auto", {
    availableProviders: [WEBGPU_PROVIDER, CUDA_PROVIDER, CPU_PROVIDER],
    registry: registry(
      descriptor("cuda", CUDA_PROVIDER, "automatic"),
      descriptor("webgpu", WEBGPU_PROVIDER, "automatic"),
    ),
  });

  expect(selection.accelerator).toBe("cuda");
});

test("auto skips an automatic backend the runtime does not offer", () => {
  const selection = selectBackend("auto", {
    availableProviders: [WEBGPU_PROVIDER, CPU_PROVIDER],
    registry: registry(
      descriptor("cuda", CUDA_PROVIDER, "automatic"),
      descriptor("webgpu", WEBGPU_PROVIDER, "automatic"),
    ),
  });

  expect(selection.accelerator).toBe("webgpu");
});

test.each(["experimental", "manual"] as const)(
  "auto never picks a backend below automatic stability (%s)",
  (stability) => {
    const selection = selectBackend("auto", {
      availableProviders: [CUDA_PROVIDER, CPU_PROVIDER],
      registry: registry(descriptor("cuda", CUDA_PROVIDER, stability)),
    });

    expect(selection.accelerator).toBe("cpu");
    expect(selection.honored).toBe(true);
    expect(selection.fallbackReason).not.toBeNull();
    selection.requireHonored();
  },
);

test("an explicit request overrides stability", () => {
  const selection = selectBackend("cuda", {
    availableProviders: [CUDA_PROVIDER, CPU_PROVIDER],
    registry: registry(descriptor("cuda", CUDA_PROVIDER, "manual")),
  });

  expect(selection.accelerator).toBe("cuda");
  expect(selection.honored).toBe(true);
});

test("an explicit request without its provider falls back and says why", () => {
  const selection = selectBackend("cuda", {
    availableProviders: [CPU_PROVIDER],
    registry: registry(descriptor("cuda", CUDA_PROVIDER, "automatic")),
  });

  expect(selection.accelerator).toBe("cpu");
  expect(selection.honored).toBe(false);
  expect(selection.fallbackReason ?? "").toContain(CUDA_PROVIDER);
});

test("an unregistered explicit request falls back rather than raising", () => {
  const selection = selectBackend("cuda", {
    availableProviders: [CPU_PROVIDER],
    registry: [CPU_BACKEND],
  });

  expect(selection.accelerator).toBe("cpu");
  expect(selection.honored).toBe(false);
});

test("strict mode turns a denied request into a backend error", () => {
  const selection = selectBackend("cuda", {
    availableProviders: [CPU_PROVIDER],
    registry: registry(descriptor("cuda", CUDA_PROVIDER, "automatic")),
  });

  let caught: unknown;
  try {
    selection.requireHonored();
  } catch (error) {
    caught = error;
  }

  expect(isCodeIndexingError(caught)).toBe(true);
  if (!isCodeIndexingError(caught)) return;
  expect(caught.code).toBe("BACKEND_UNAVAILABLE");
  expect(caught.details.requested).toBe("cuda");
  expect(caught.details.resolved).toBe("cpu");
});

test("an accelerator keeps cpu behind it but cpu stands alone", () => {
  const cuda = descriptor("cuda", CUDA_PROVIDER, "automatic");

  expect(cuda.providers).toEqual([CUDA_PROVIDER, CPU_PROVIDER]);
  expect(CPU_BACKEND.providers).toEqual([CPU_PROVIDER]);
});

test("falling back records the new backend and the reason", () => {
  const selection = selectBackend("cuda", {
    availableProviders: [CUDA_PROVIDER, CPU_PROVIDER],
    registry: registry(descriptor("cuda", CUDA_PROVIDER, "automatic")),
  });

  const degraded = selection.fellBackTo(CPU_BACKEND, "worker exited");

  expect(degraded.accelerator).toBe("cpu");
  expect(degraded.requested).toBe("cuda");
  expect(degraded.honored).toBe(false);
  expect(degraded.fallbackReason).toBe("worker exited");
  expect(selection.accelerator).toBe("cuda");
});

test("only backends that passed their gates are eligible automatically", () => {
  const automatic = KNOWN_BACKENDS.filter(
    (backend) => !backend.isCpu && backend.stability === "automatic",
  ).map((backend) => backend.accelerator);

  expect(automatic).toEqual(["cuda", "mlx"]);
});

test("auto stays on cpu where no accelerator was prepared", () => {
  const selection = selectBackend("auto", { availableProviders: [CPU_PROVIDER] });

  expect(selection.accelerator).toBe("cpu");
  expect(selection.honored).toBe(true);
  expect(selection.fallbackReason ?? "").toContain("reinstall with --accelerator");
});

test("available providers always include cpu", () => {
  expect(availableExecutionProviders()).toContain(CPU_PROVIDER);
});

test("mlx is registered as a promoted metal backend", () => {
  const mlx = backendFor("mlx");

  expect(mlx).toBeDefined();
  if (mlx === undefined) return;
  expect(mlx.runtime).toBe("mlx");
  expect(mlx.provider).toBe(MLX_PROVIDER);
  expect(mlx.stability).toBe("automatic");
  expect(mlx.usesDirectModel).toBe(true);
  expect(mlx.providerIsPreregistered).toBe(false);
  expect(mlx.runsOnOnnx).toBe(false);
  expect(mlx.publishesExecutionProviders).toBe(false);
});

test("a plugin provider is still an onnx runtime backend", () => {
  const webgpu = backendFor("webgpu");

  expect(webgpu).toBeDefined();
  if (webgpu === undefined) return;
  expect(webgpu.runtime).toBe("onnxruntime-plugin");
  expect(webgpu.providerIsPreregistered).toBe(false);
  expect(webgpu.runsOnOnnx).toBe(true);
  expect(webgpu.publishesExecutionProviders).toBe(true);
});

test("an mlx backend has no onnx cpu provider behind it", () => {
  const mlx = backendFor("mlx");
  expect(mlx).toBeDefined();
  if (mlx === undefined) return;
  expect(mlx.providers).toEqual([MLX_PROVIDER]);
});

test("an explicit mlx request is honoured against a prepared record", () => {
  const selection = selectBackend("mlx", { availableProviders: [CPU_PROVIDER, MLX_PROVIDER] });

  expect(selection.accelerator).toBe("mlx");
  expect(selection.honored).toBe(true);
  expect(selection.usesAccelerator).toBe(true);
});

test("auto selects a prepared mlx environment", () => {
  const selection = selectBackend("auto", { availableProviders: [CPU_PROVIDER, MLX_PROVIDER] });

  expect(selection.accelerator).toBe("mlx");
  expect(selection.honored).toBe(true);
  expect(selection.fallbackReason).toBeNull();
});
