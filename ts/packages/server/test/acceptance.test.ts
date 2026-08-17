/** Unit tests for the shared retrieval-quality metrics. */

import { describe, expect, test } from "bun:test";
import { cosineRows, topKOverlap, topKRankCorrelation } from "../src/acceptance.ts";

describe("top-k rank correlation", () => {
  test("identical rankings correlate perfectly", () => {
    expect(topKRankCorrelation([["a", "b", "c"]], [["a", "b", "c"]])).toBe(1);
  });

  test("reversed rankings correlate negatively", () => {
    expect(topKRankCorrelation([["a", "b", "c"]], [["c", "b", "a"]])).toBe(-1);
  });

  test("one adjacent swap costs the discordant share", () => {
    // Pairs over (a, b, c): (a, b) discordant; (a, c) and (b, c) concordant.
    // tau = (2 - 1) / 3.
    expect(topKRankCorrelation([["a", "b", "c"]], [["b", "a", "c"]])).toBeCloseTo(1 / 3, 12);
  });

  test("ids missing from one window tie past its end", () => {
    // Union (a, b, c) with c absent from the reference window and a absent from
    // the candidate window: (a, b) and (a, c) discordant, (b, c) concordant.
    // tau = (1 - 2) / 3.
    expect(topKRankCorrelation([["a", "b"]], [["b", "c"]])).toBeCloseTo(-1 / 3, 12);
  });

  test("correlation is averaged over queries", () => {
    const reference = [
      ["a", "b", "c"],
      ["x", "y", "z"],
    ];
    const candidate = [
      ["a", "b", "c"],
      ["z", "y", "x"],
    ];

    expect(topKRankCorrelation(reference, candidate)).toBeCloseTo(0, 12);
  });

  test("single shared id windows are perfectly correlated", () => {
    // One id cannot form a pair, so there is nothing to disagree about.
    expect(topKRankCorrelation([["only"]], [["only"]])).toBe(1);
  });

  test("invalid ranking inputs raise", () => {
    expect(() => topKRankCorrelation([], [])).toThrow();
    expect(() => topKRankCorrelation([["a"]], [["a"], ["b"]])).toThrow();
    expect(() => topKRankCorrelation([[]], [["a"]])).toThrow();
    expect(() => topKRankCorrelation([["a"]], [[]])).toThrow();
  });
});

describe("cosine rows", () => {
  test("a matrix compared with itself scores one everywhere", () => {
    const matrix = [
      [1, 0, 0],
      [0.5, 0.5, 0.5],
    ];

    for (const score of cosineRows(matrix, matrix)) {
      expect(score).toBeCloseTo(1, 12);
    }
  });

  test("orthogonal rows score zero and opposed rows score minus one", () => {
    const scores = cosineRows(
      [
        [1, 0],
        [1, 0],
      ],
      [
        [0, 1],
        [-1, 0],
      ],
    );

    expect(scores[0]).toBeCloseTo(0, 12);
    expect(scores[1]).toBeCloseTo(-1, 12);
  });

  test("magnitude is normalized away", () => {
    expect(cosineRows([[3, 4]], [[300, 400]])[0]).toBeCloseTo(1, 12);
  });

  test("degenerate matrices are rejected rather than scored", () => {
    expect(() => cosineRows([], [])).toThrow(/non-empty/);
    expect(() => cosineRows([[0, 0]], [[1, 0]])).toThrow(/zero-length row/);
    expect(() => cosineRows([[Number.NaN, 1]], [[1, 0]])).toThrow(/non-finite/);
    expect(() =>
      cosineRows(
        [[1, 0]],
        [
          [1, 0],
          [0, 1],
        ],
      ),
    ).toThrow(/shapes differ/);
  });
});

describe("top-k overlap", () => {
  const queries = [[1, 0]];
  const reference = [
    [1, 0],
    [0.9, 0.1],
    [0, 1],
  ];

  test("an identical candidate matrix overlaps completely", () => {
    expect(topKOverlap(queries, reference, reference, { k: 2 })).toBe(1);
  });

  test("a candidate that reorders the window loses the displaced share", () => {
    const candidate = [
      [1, 0],
      [0, 1],
      [0.9, 0.1],
    ];

    // The top-2 window keeps row 0 and swaps row 1 for row 2.
    expect(topKOverlap(queries, reference, candidate, { k: 2 })).toBe(0.5);
  });

  test("k outside the row count is rejected", () => {
    expect(() => topKOverlap(queries, reference, reference, { k: 0 })).toThrow(/k must be/);
    expect(() => topKOverlap(queries, reference, reference, { k: 4 })).toThrow(/k must be/);
  });

  test("a query of the wrong width is rejected", () => {
    expect(() => topKOverlap([[1, 0, 0]], reference, reference, { k: 1 })).toThrow(
      /query dimension/,
    );
  });
});
