import { describe, expect, test } from "bun:test";
import { planAccelerator, planPreparesEnvironment } from "../src/accelerator.ts";

describe("planAccelerator", () => {
  test("honours an explicit CPU request", () => {
    const plan = planAccelerator("cpu");
    expect(plan.accelerator).toBe("cpu");
    expect(plan.honored).toBe(true);
    expect(planPreparesEnvironment(plan)).toBe(false);
  });

  test("Core ML prepares nothing", () => {
    const plan = planAccelerator("coreml");
    expect(plan.accelerator).toBe("cpu");
    expect(plan.honored).toBe(true);
  });

  test("auto on Apple Silicon nominates MLX", () => {
    const plan = planAccelerator("auto", {
      platformName: "darwin",
      machine: "arm64",
      platformVersion: "15.0",
    });
    expect(plan.accelerator).toBe("mlx");
    expect(plan.honored).toBe(true);
  });

  test("CUDA without a driver falls back to CPU", () => {
    const plan = planAccelerator("cuda", {
      platformName: "linux",
      machine: "x86_64",
      nvidiaReport: () => null,
    });
    expect(plan.accelerator).toBe("cpu");
    expect(plan.honored).toBe(false);
  });

  test("CUDA with a current driver is nominated", () => {
    const plan = planAccelerator("cuda", {
      platformName: "linux",
      machine: "x86_64",
      nvidiaReport: () => "550.54.14, NVIDIA GeForce RTX 4090",
    });
    expect(plan.accelerator).toBe("cuda");
    expect(plan.driverVersion).toBe("550.54.14");
    expect(plan.honored).toBe(true);
  });
});
