/** The stable error surface both adapters render. */

import { expect, test } from "bun:test";
import { CodeIndexingError, isCodeIndexingError } from "../src/errors.ts";

test("the string form leads with the code and omits details", () => {
  const error = new CodeIndexingError("INDEX_BUSY", "Another run holds the lock", {
    project: "abc",
  });

  // Details are carried separately in daemon frames and IndexIssue messages, so
  // repeating them here would duplicate the payload.
  expect(String(error)).toBe("INDEX_BUSY: Another run holds the lock");
  expect(error.message).toBe("Another run holds the lock");
});

test("the client form appends the details", () => {
  const error = new CodeIndexingError("INVALID_CONFIGURATION", "bad value", {
    setting: "CODE_INDEXING_BROKER",
    value: "sometimes",
  });

  expect(error.forClient()).toBe(
    "INVALID_CONFIGURATION: bad value [setting=CODE_INDEXING_BROKER; value=sometimes]",
  );
});

test("a detail-free error renders identically either way", () => {
  const error = new CodeIndexingError("CHUNK_NOT_FOUND", "no such chunk");

  expect(error.forClient()).toBe(String(error));
});

test("a composite detail stays legible instead of collapsing to a comma list", () => {
  const error = new CodeIndexingError("AMBIGUOUS_PROJECT", "two matches", {
    projects: ["one", "two"],
  });

  expect(error.forClient()).toBe('AMBIGUOUS_PROJECT: two matches [projects=["one","two"]]');
});

test("it is a real Error, so it survives a throw and carries a cause", () => {
  const cause = new Error("underlying");
  try {
    throw new CodeIndexingError("PROTOCOL_ERROR", "frame rejected", {}, { cause });
  } catch (caught) {
    expect(isCodeIndexingError(caught)).toBe(true);
    expect(caught).toBeInstanceOf(Error);
    expect((caught as CodeIndexingError).code).toBe("PROTOCOL_ERROR");
    expect((caught as CodeIndexingError).cause).toBe(cause);
    expect((caught as Error).stack).toBeDefined();
  }
});

test("an unrelated value is not mistaken for one", () => {
  expect(isCodeIndexingError(new Error("plain"))).toBe(false);
  expect(isCodeIndexingError({ code: "INDEX_BUSY" })).toBe(false);
});
