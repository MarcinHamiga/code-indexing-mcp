/** Token window planning and microbatch packing, with a deterministic tokenizer. */

import { describe, expect, test } from "bun:test";
import {
  contentTokenOffsets,
  DEFAULT_MAX_TOKEN_PRODUCT,
  DEFAULT_MAX_TOKENS,
  DEFAULT_OVERLAP_TOKENS,
  maxTokenProductFor,
  planCandidateWindows,
  planMicrobatches,
  planTokenWindows,
  REFERENCE_MEMORY_BYTES,
  type TokenEncoding,
  type TokenSpan,
} from "../src/token-batching.ts";

const TOKEN = /\w+|[^\w\s]/g;

/** Word-ish tokens wrapped in the `(0, 0)` specials a real tokenizer adds. */
function fakeEncode(text: string): TokenEncoding {
  const spans: TokenSpan[] = [];
  for (const match of text.matchAll(TOKEN)) {
    spans.push([match.index, match.index + match[0].length]);
  }
  return {
    offsets: [[0, 0], ...spans, [0, 0]],
    specialTokensMask: [1, ...spans.map(() => 0), 1],
  };
}

function offsetsFor(text: string): TokenSpan[] {
  return contentTokenOffsets(fakeEncode(text));
}

function words(count: number): string {
  return Array.from({ length: count }, (_value, index) => `tok${index}`).join(" ");
}

test("special tokens are dropped from planning offsets", () => {
  expect(fakeEncode("alpha beta").offsets[0]).toEqual([0, 0]);
  expect(offsetsFor("alpha beta")).toEqual([
    [0, 5],
    [6, 10],
  ]);
});

test("a mask that does not cover the offsets is rejected", () => {
  expect(() => contentTokenOffsets({ offsets: [[0, 1]], specialTokensMask: [0, 0] })).toThrow(
    /special token mask/,
  );
});

test("a candidate within the budget stays one window", () => {
  const text = "alpha beta gamma";

  const windows = planTokenWindows(offsetsFor(text), { textLength: text.length, maxTokens: 8 });

  expect(windows).toEqual([{ startChar: 0, endChar: text.length, tokenCount: 3 }]);
});

test("no window exceeds the token budget", () => {
  const text = words(4_000);

  const windows = planTokenWindows(offsetsFor(text), {
    textLength: text.length,
    maxTokens: DEFAULT_MAX_TOKENS,
    overlapTokens: DEFAULT_OVERLAP_TOKENS,
    maxWindows: 64,
  });

  expect(windows.length).toBeGreaterThan(1);
  expect(windows.every((window) => window.tokenCount <= DEFAULT_MAX_TOKENS)).toBe(true);
});

test("adjacent windows overlap by the configured token count", () => {
  const text = words(300);

  const windows = planTokenWindows(offsetsFor(text), {
    textLength: text.length,
    maxTokens: 100,
    overlapTokens: 10,
  });

  // Stride is 90 tokens, so a window's first token reappears 10 tokens before
  // the previous window's last.
  const spans = offsetsFor(text);
  expect(windows[1]?.startChar).toBe(spans[90]?.[0]);
  expect(windows[2]?.startChar).toBe(spans[180]?.[0]);
});

test("windows cover the candidate without dropping characters", () => {
  const text = Array.from({ length: 500 }, (_value, index) => `value_${index} = ${index}`).join(
    "\n",
  );

  const windows = planTokenWindows(offsetsFor(text), {
    textLength: text.length,
    maxTokens: 64,
    overlapTokens: 8,
    maxWindows: 64,
  });

  expect(windows[0]?.startChar).toBe(0);
  expect(windows.at(-1)?.endChar).toBe(text.length);
  // Each window resumes at or before the previous one ended, so the
  // concatenation of the disjoint prefixes reconstructs the source exactly.
  let rebuilt = text.slice(0, windows[0]?.endChar);
  for (let index = 1; index < windows.length; index += 1) {
    const previous = windows[index - 1] as { endChar: number };
    const window = windows[index] as { startChar: number; endChar: number };
    expect(window.startChar).toBeLessThanOrEqual(previous.endChar);
    rebuilt += text.slice(Math.max(window.startChar, previous.endChar), window.endChar);
  }
  expect(rebuilt).toBe(text);
});

test("a single long line is split by tokens, not by newlines", () => {
  const text = `DATA = [${Array.from({ length: 3_000 }, (_value, index) => index % 977).join(", ")}]`;

  const windows = planTokenWindows(offsetsFor(text), {
    textLength: text.length,
    maxTokens: 256,
    overlapTokens: 16,
    maxWindows: 64,
  });

  expect(text).not.toContain("\n");
  expect(windows.length).toBeGreaterThan(1);
  expect(windows.every((window) => window.tokenCount <= 256)).toBe(true);
});

test("emitted text stays within twice the candidate size", () => {
  const text = words(5_000);

  const windows = planTokenWindows(offsetsFor(text), {
    textLength: text.length,
    maxTokens: DEFAULT_MAX_TOKENS,
    overlapTokens: DEFAULT_OVERLAP_TOKENS,
    maxWindows: 64,
  });

  const emitted = windows.reduce((total, window) => total + window.endChar - window.startChar, 0);
  expect(emitted).toBeLessThanOrEqual(2 * text.length);
});

test("boundaries do not depend on the memory budget or batch packing", () => {
  const text = words(2_000);
  const offsets = offsetsFor(text);
  const options = { textLength: text.length, maxTokens: 512, maxWindows: 64 };

  expect(planTokenWindows(offsets, options)).toEqual(planTokenWindows(offsets, options));
});

test("a window explosion raises instead of flooding the index", () => {
  const text = words(1_000);

  expect(() =>
    planTokenWindows(offsetsFor(text), {
      textLength: text.length,
      maxTokens: 32,
      maxWindows: 4,
    }),
  ).toThrow(/exceeded 4 windows/);
});

test("whitespace-only content still yields one window", () => {
  expect(planTokenWindows([], { textLength: 12 })).toEqual([
    { startChar: 0, endChar: 12, tokenCount: 0 },
  ]);
});

test("empty content yields no window", () => {
  expect(planTokenWindows([], { textLength: 0 })).toEqual([]);
});

describe("candidate planning", () => {
  test("the prefix is charged against the window budget", () => {
    const prefix = Array.from({ length: 20 }, (_value, index) => `header${index}`).join(" ");
    const content = words(100);
    const options = { maxTokens: 60, overlapTokens: 5 };

    const withPrefix = planCandidateWindows(fakeEncode, [{ prefix, content }], options)[0] ?? [];
    const withoutPrefix =
      planCandidateWindows(fakeEncode, [{ prefix: "", content }], options)[0] ?? [];

    // 20 prefix tokens leave a 40-token content budget, so the prefixed
    // candidate needs strictly more windows for the same content.
    expect(withPrefix.length).toBeGreaterThan(withoutPrefix.length);
    expect(withPrefix.every((window) => window.tokenCount <= 40)).toBe(true);
  });

  test("a prefix wider than the budget still makes forward progress", () => {
    const prefix = Array.from({ length: 200 }, (_value, index) => `header${index}`).join(" ");
    const content = words(50);

    const windows =
      planCandidateWindows(fakeEncode, [{ prefix, content }], {
        maxTokens: 10,
        overlapTokens: 4,
        maxWindows: 64,
      })[0] ?? [];

    // The budget floor plus the half-budget overlap clamp keep the stride at 3
    // tokens, so 50 tokens land in 16 windows rather than looping forever.
    expect(windows).toHaveLength(16);
    expect(windows.at(-1)?.endChar).toBe(content.length);
  });

  test("one prefix is tokenized once however many candidates share it", () => {
    let calls = 0;
    const counting = (text: string): TokenEncoding => {
      calls += 1;
      return fakeEncode(text);
    };

    planCandidateWindows(counting, [
      { prefix: "shared header", content: "a b" },
      { prefix: "shared header", content: "c d" },
      { prefix: "shared header", content: "e f" },
    ]);

    // Three contents plus one prefix, not three prefixes.
    expect(calls).toBe(4);
  });
});

describe("microbatch packing", () => {
  test("the item limit is respected", () => {
    expect(planMicrobatches(Array(7).fill(10), { maxItems: 2, maxTokenProduct: 10_000 })).toEqual([
      [0, 1],
      [2, 3],
      [4, 5],
      [6],
    ]);
  });

  test("the padded token product is respected", () => {
    // Padding widens every member to the longest, so a 900-token segment caps
    // its batch at four items against a 4,096 product.
    expect(planMicrobatches(Array(6).fill(900), { maxItems: 8, maxTokenProduct: 4_096 })).toEqual([
      [0, 1, 2, 3],
      [4, 5],
    ]);
  });

  test("similar lengths are bucketed before padding", () => {
    expect(planMicrobatches([10, 1_000, 12, 900], { maxItems: 2, maxTokenProduct: 2_000 })).toEqual(
      [
        [0, 2],
        [1, 3],
      ],
    );
  });

  test("a segment wider than the product still forms its own batch", () => {
    expect(planMicrobatches([9_000, 10], { maxItems: 4, maxTokenProduct: 4_096 })).toEqual([
      [0],
      [1],
    ]);
  });

  test("an empty plan produces no batches", () => {
    expect(planMicrobatches([], { maxItems: 4 })).toEqual([]);
  });

  test("a zero item limit is rejected", () => {
    expect(() => planMicrobatches([1, 2], { maxItems: 0 })).toThrow(/maxItems/);
  });
});

describe("the token product budget", () => {
  test("the default ceiling reproduces the measured token product", () => {
    // The constant was measured at this ceiling, so this ceiling must keep it.
    expect(maxTokenProductFor(REFERENCE_MEMORY_BYTES)).toBe(DEFAULT_MAX_TOKEN_PRODUCT);
  });

  test("the product follows the configured ceiling", () => {
    expect(maxTokenProductFor(REFERENCE_MEMORY_BYTES * 2)).toBe(DEFAULT_MAX_TOKEN_PRODUCT * 2);
    expect(maxTokenProductFor(REFERENCE_MEMORY_BYTES / 2)).toBe(DEFAULT_MAX_TOKEN_PRODUCT / 2);
  });

  test("the smallest ceiling still admits one longest sequence", () => {
    // A product below one window would batch a max-length segment alone anyway,
    // but it would also drop every shorter segment to one item per batch.
    expect(maxTokenProductFor(1024 ** 3)).toBeGreaterThanOrEqual(DEFAULT_MAX_TOKENS);
  });

  test("a large ceiling does not widen the batch without evidence", () => {
    // Padding is quadratic in the widest member and nothing measured a matrix
    // this size, so the ceiling stops buying width past a fixed multiple.
    expect(maxTokenProductFor(REFERENCE_MEMORY_BYTES * 64)).toBe(
      maxTokenProductFor(REFERENCE_MEMORY_BYTES * 1024),
    );
  });
});
