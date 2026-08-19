import { describe, expect, test } from "bun:test";
import fs from "node:fs";
import path from "node:path";
import { Application } from "../src/application.ts";
import type { Embedder } from "../src/embedding.ts";
import { createServer } from "../src/server.ts";
import { temporaryDirectory, removeDirectory } from "./helpers.ts";

class TinyEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  embedPassages(texts: string[]): number[][] {
    return texts.map((text) => [1, 0, 0, text.length]);
  }
  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

/** The fixture is what the Python FastMCP surface emits; see scripts/write_tool_schema_parity.py. */
const FIXTURE = JSON.parse(
  fs.readFileSync(new URL("./fixtures/tool-schemas.json", import.meta.url), "utf8"),
) as {
  tools: Array<{
    name: string;
    title: string;
    description: string;
    annotations: Record<string, boolean> | null;
    inputSchema: Record<string, unknown>;
  }>;
};

/** Drop zod's generator artifacts that pydantic does not emit. */
function stripZodArtifacts(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(stripZodArtifacts);
  if (schema === null || typeof schema !== "object") return schema;
  const record = schema as Record<string, unknown>;
  const stripped: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    // The zod tool schemas validate strictly; Python ignores extra arguments.
    // Both advertise the same fields, so only this keyword differs.
    if (key === "additionalProperties") continue;
    stripped[key] = stripZodArtifacts(value);
  }
  return stripped;
}

/** Inline pydantic's `$ref`/`$defs` packaging, which zod emits flattened. */
function resolveRefs(node: unknown, defs: Record<string, unknown>): unknown {
  if (Array.isArray(node)) return node.map((item) => resolveRefs(item, defs));
  if (node === null || typeof node !== "object") return node;
  const record = node as Record<string, unknown>;
  if (typeof record.$ref === "string" && record.$ref.startsWith("#/$defs/")) {
    const target = defs[record.$ref.slice("#/$defs/".length)];
    if (target === undefined) throw new Error(`unresolved ${record.$ref}`);
    const merged = resolveRefs(target, defs) as Record<string, unknown>;
    for (const [key, extra] of Object.entries(record)) {
      if (key !== "$ref") merged[key] = resolveRefs(extra, defs);
    }
    return merged;
  }
  const resolved: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(record)) {
    // pydantic annotates discriminated unions with OpenAPI's advisory
    // `discriminator` keyword whose mapping points at `$defs` entries; zod
    // flattens both away. The branches themselves are compared in full.
    if (key === "discriminator") continue;
    resolved[key] = resolveRefs(value, defs);
  }
  return resolved;
}

function inlinedPydanticSchema(tool: (typeof FIXTURE)["tools"][number]): unknown {
  const schema = { ...tool.inputSchema } as Record<string, unknown>;
  const defs = (schema.$defs ?? {}) as Record<string, unknown>;
  delete schema.$defs;
  return resolveRefs(schema, defs);
}

describe("MCP tool schema parity with the Python server", () => {
  test("every tool reproduces the Python schema exactly", async () => {
    const temporary = temporaryDirectory();
    try {
      const app = new Application(
        { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") },
        { embedder: new TinyEmbedder(), cwd: temporary },
      );
      const tools = await createServer(app, { autoIndex: false }).listTools();
      const actual = Object.fromEntries(
        tools.map((tool) => [
          tool.name,
          {
            name: tool.name,
            title: tool.title,
            description: tool.description,
            annotations: tool.annotations ?? null,
            inputSchema: stripZodArtifacts(tool.inputSchema),
          },
        ]),
      );
      const expected = Object.fromEntries(
        FIXTURE.tools.map((tool) => [
          tool.name,
          {
            name: tool.name,
            title: tool.title,
            description: tool.description,
            annotations: tool.annotations,
            inputSchema: inlinedPydanticSchema(tool),
          },
        ]),
      );
      expect(Object.keys(actual).sort()).toEqual(Object.keys(expected).sort());
      for (const [name, tool] of Object.entries(expected)) {
        expect(actual[name], `tool ${name}`).toEqual(tool);
      }
    } finally {
      removeDirectory(temporary);
    }
  });
});
