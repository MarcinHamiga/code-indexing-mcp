/** Embedding protocol helpers and in-process embedder. */

import { expect, test } from "bun:test";
import {
  composePassage,
  embedPlannedSegments,
  OnnxEmbedder,
  type PassageCandidate,
  passageCandidate,
  planPassages,
  resolveSessionProviders,
  resolveTokenizer,
  segmentPlan,
  setModelFactory,
} from "../src/embedding.ts";
import { removeDirectory, temporaryDirectory } from "./helpers.ts";

test("compose passage matches the extractor layout", () => {
  expect(composePassage("kind: module", "value = 1")).toBe("kind: module\nvalue = 1");
  expect(composePassage("", "value = 1")).toBe("value = 1");
});

test("resolve tokenizer finds the nested fastembed tokenizer", () => {
  class Tokenizer {
    encode(text: string): string {
      return text;
    }
  }
  class Inner {
    tokenizer = new Tokenizer();
  }
  class Model {
    model = new Inner();
  }

  expect(resolveTokenizer(new Model())).toBeInstanceOf(Tokenizer);
});

test("resolve tokenizer returns none when the layout moved", () => {
  expect(resolveTokenizer({})).toBeUndefined();
});

test("direct model reports the provider attached through the plugin api", () => {
  class CpuLookingSession {
    getProviders(): string[] {
      return ["CPUExecutionProvider"];
    }
  }
  class DirectModel {
    resolvedProviders = ["WebGpuExecutionProvider", "CPUExecutionProvider"];
    model = new CpuLookingSession();
  }

  expect(resolveSessionProviders(new DirectModel())).toEqual([
    "WebGpuExecutionProvider",
    "CPUExecutionProvider",
  ]);
});

test("planning without a tokenizer leaves candidates whole", () => {
  const candidates = [passageCandidate("kind: module", "value = 1")];

  const windows = planPassages(undefined, candidates, segmentPlan());

  expect(windows[0]?.map((window) => [window.startChar, window.endChar])).toEqual([[0, 9]]);
});

test("embedding without a tokenizer sends the whole candidate", async () => {
  const seen: string[][] = [];
  const embed = (texts: string[]): string[] => {
    seen.push(texts);
    return texts.map(() => "vector");
  };
  const candidates: PassageCandidate[] = [passageCandidate("kind: module", "value = 1")];
  const result = await embedPlannedSegments(undefined, embed, candidates, segmentPlan());

  expect(seen).toEqual([["kind: module\nvalue = 1"]]);
  expect(result[0]?.length).toBe(1);
});

test("concurrent first use builds a single model", async () => {
  const builds: number[] = [];
  setModelFactory(async () => {
    builds.push(Date.now());
    await Bun.sleep(50);
    return { passageEmbed: (texts: string[]) => texts.map(() => [0]) };
  });
  const directory = temporaryDirectory();
  try {
    const embedder = new OnnxEmbedder(directory);
    await Promise.all(Array.from({ length: 8 }, () => embedder.prepare()));
    expect(builds.length).toBe(1);
  } finally {
    setModelFactory(undefined);
    removeDirectory(directory);
  }
});
