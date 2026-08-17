/** The accelerator name, which is all of the backend contract Phase 1 needs. */

import { expect, test } from "bun:test";
import { ACCELERATORS, parseAccelerator } from "../src/backends.ts";
import { isCodeIndexingError } from "../src/errors.ts";

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
  // The operator sees the members in the order the Python StrEnum listed them,
  // so the two builds do not print differently ordered advice.
  expect(caught.message).toContain(ACCELERATORS.join(", "));
});
