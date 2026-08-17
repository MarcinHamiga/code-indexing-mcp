/**
 * Embedding backend contract: descriptors, capability probes, and selection.
 *
 * Phase 1 ports only the accelerator name and its parser, which is all
 * `settings.ts` needs to validate `CODE_INDEXING_EMBED_ACCELERATOR`. Backend
 * descriptors, provider probing, and the nomination logic arrive with the
 * embedding stack in Phase 4 -- they depend on what ONNX Runtime reports, and
 * nothing in Phase 1 loads it.
 */

import { CodeIndexingError } from "./errors.ts";

/**
 * The execution targets a passage embedder can be pointed at.
 *
 * Ordered as the Python `StrEnum` is, because `parseAccelerator` lists the
 * members back to the operator when their spelling was wrong and that message
 * should not reorder itself between the two builds.
 */
export const ACCELERATORS = ["auto", "cpu", "cuda", "mlx", "webgpu", "migraphx", "coreml"] as const;

export type Accelerator = (typeof ACCELERATORS)[number];

/** Parse a configured accelerator name into its member. */
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
