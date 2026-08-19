/**
 * CPU-only stub for the installer-written accelerator environment record.
 *
 * Phase 7 owns the real reader. Until then a missing record is a CPU
 * installation, which is also what a machine without a prepared accelerator
 * looks like on the Python side.
 */

import type { BackendDescriptor } from "./backends.ts";

export interface AcceleratorEnvironmentRecord {
  readonly interpreter: string;
  readonly accelerator: string;
  readonly providers: readonly string[];
}

export interface AcceleratorEnvironmentStatus {
  readonly environment: AcceleratorEnvironmentRecord | null;
  readonly reason: string | null;
  readonly providers: readonly string[];
}

export function loadEnvironment(_dataDirectory: string): AcceleratorEnvironmentStatus {
  return { environment: null, reason: null, providers: [] };
}

export function applyEnvironment(
  descriptor: BackendDescriptor,
  _record: AcceleratorEnvironmentRecord | null,
): BackendDescriptor {
  return descriptor;
}
