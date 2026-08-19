/**
 * Hybrid retrieval and structural lookup services.
 */

import {
  type ChunkPreview,
  type CodeChunk,
  OutlineItem,
  OutlineResponse,
  SearchHit,
  SearchResponse,
  SymbolResponse,
} from "./models.ts";
import { pathCondition, pathMatches } from "./path-filter.ts";
import type { Embedder } from "./embedding.ts";
import { CodeIndexingError } from "./errors.ts";
import { quote, type LanceStore } from "./storage.ts";

const FALLBACK_FETCH_ROWS = 500;

export class SearchService {
  readonly store: LanceStore;
  readonly embedder: Embedder;

  constructor(store: LanceStore, embedder: Embedder) {
    this.store = store;
    this.embedder = embedder;
  }

  async searchCode(
    query: string,
    projectIds: readonly string[],
    {
      languages,
      paths,
      kinds,
      limit = 8,
    }: {
      languages?: readonly string[];
      paths?: readonly string[];
      kinds?: readonly string[];
      limit?: number;
    } = {},
  ): Promise<SearchResponse> {
    query = query.trim();
    if (query === "" || projectIds.length === 0) {
      throw new CodeIndexingError(
        "INVALID_FILTER",
        "Search requires a query and at least one project",
      );
    }
    const capped = Math.max(1, Math.min(limit, 50));
    const conditions: string[] = [];
    if (languages !== undefined && languages.length > 0) {
      conditions.push(inCondition("language", languages));
    }
    if (kinds !== undefined && kinds.length > 0) {
      conditions.push(inCondition("kind", kinds));
    }
    const pushedPaths = paths !== undefined ? pathCondition(paths) : null;
    if (pushedPaths !== null) conditions.push(pushedPaths);
    const fetch =
      paths !== undefined && pushedPaths === null ? FALLBACK_FETCH_ROWS : Math.max(50, capped * 5);
    const vector = await this.embedder.embedQuery(query);
    const rows = await this.store.withPartitionsAccess([...projectIds], () =>
      this.store.hybridSearch(query, vector, projectIds, {
        ...(conditions.length > 0 ? { condition: conditions.join(" AND ") } : {}),
        limit: fetch,
      }),
    );
    const names = Object.fromEntries(
      (await this.store.listProjects()).map((project) => [project.id, project.name]),
    );
    const hits: SearchHit[] = [];
    const seen = new Set<string>();
    for (const row of rows) {
      const chunk = parsePreview(row);
      if (paths !== undefined && !paths.some((pattern) => pathMatches(chunk.path, pattern))) {
        continue;
      }
      const key = `${chunk.project_id}\0${chunk.path}\0${chunk.start_line}\0${chunk.end_line}`;
      if (seen.has(key)) continue;
      seen.add(key);
      hits.push(hit(chunk, names, Number(row._relevance_score ?? 0)));
      if (hits.length === capped) break;
    }
    hits.sort((left, right) => {
      if (left.score !== right.score) return right.score - left.score;
      if (left.path !== right.path) return left.path < right.path ? -1 : 1;
      return left.start_line - right.start_line;
    });
    return SearchResponse.parse({ query, hits });
  }

  async findSymbol(
    name: string,
    projectId: string,
    {
      match = "exact",
      kinds,
      limit = 20,
    }: {
      match?: "exact" | "prefix" | "contains";
      kinds?: readonly string[];
      limit?: number;
    } = {},
  ): Promise<SymbolResponse> {
    if (match !== "exact" && match !== "prefix" && match !== "contains") {
      throw new CodeIndexingError("INVALID_FILTER", `Invalid symbol match mode: ${match}`);
    }
    const capped = Math.max(1, Math.min(limit, 50));
    const candidates = await this.store.withPartitionAccess(projectId, () =>
      this.store.findSymbolChunks(name, projectId, {
        match,
        ...(kinds === undefined ? {} : { kinds }),
        limit: capped,
      }),
    );
    const names = Object.fromEntries(
      (await this.store.listProjects()).map((project) => [project.id, project.name]),
    );
    const selected = [...candidates]
      .sort((left, right) => {
        if (left.path !== right.path) return left.path < right.path ? -1 : 1;
        if (left.start_line !== right.start_line) return left.start_line - right.start_line;
        return left.kind < right.kind ? -1 : left.kind > right.kind ? 1 : 0;
      })
      .slice(0, capped);
    return SymbolResponse.parse({
      name,
      hits: selected.map((chunk) => hit(chunk, names, 1)),
    });
  }

  async fileOutline(sourcePath: string, projectId: string): Promise<OutlineResponse> {
    const items: OutlineItem[] = [];
    const seen = new Set<string>();
    const chunks = await this.store.withPartitionAccess(projectId, () =>
      this.store.outlineChunks(projectId, sourcePath),
    );
    const ordered = [...chunks].sort((left, right) => {
      if (left.path !== right.path) return left.path < right.path ? -1 : 1;
      return left.start_line - right.start_line;
    });
    for (const chunk of ordered) {
      if (chunk.path !== sourcePath || chunk.symbol === null || chunk.qualified_symbol === null) {
        continue;
      }
      const kind = chunk.kind.endsWith("_part") ? chunk.kind.slice(0, -5) : chunk.kind;
      const key = `${kind}\0${chunk.qualified_symbol}`;
      if (seen.has(key)) continue;
      seen.add(key);
      items.push(
        OutlineItem.parse({
          kind,
          symbol: chunk.symbol,
          qualified_symbol: chunk.qualified_symbol,
          parent_symbol: chunk.parent_symbol,
          start_line: chunk.start_line,
          end_line: chunk.end_line,
        }),
      );
    }
    return OutlineResponse.parse({ project_id: projectId, path: sourcePath, items });
  }

  async getChunk(chunkId: string): Promise<CodeChunk> {
    const chunk = await this.store.getChunk(chunkId);
    if (chunk === null) {
      throw new CodeIndexingError(
        "CHUNK_NOT_FOUND",
        `Unknown chunk: ${chunkId}; chunk ids come from search_code or find_symbol ` +
          "results and change when the file is re-indexed",
        { chunk_id: chunkId },
      );
    }
    return chunk;
  }
}

function inCondition(column: string, values: readonly string[]): string {
  return `${column} IN (${values.map(quote).join(", ")})`;
}

function hit(chunk: ChunkPreview, names: Record<string, string>, score: number): SearchHit {
  const snippet = chunk.content.slice(0, 4000);
  return SearchHit.parse({
    chunk_id: chunk.chunk_id,
    project_id: chunk.project_id,
    project_name: names[chunk.project_id] ?? chunk.project_id,
    path: chunk.path,
    language: chunk.language,
    kind: chunk.kind,
    symbol: chunk.symbol,
    qualified_symbol: chunk.qualified_symbol,
    start_line: chunk.start_line,
    end_line: chunk.end_line,
    score,
    snippet,
    truncated: chunk.content.length > snippet.length,
  });
}

function parsePreview(row: Record<string, unknown>): ChunkPreview {
  return {
    chunk_id: String(row.chunk_id ?? ""),
    project_id: String(row.project_id ?? ""),
    path: String(row.path ?? ""),
    language: String(row.language ?? ""),
    kind: String(row.kind ?? ""),
    symbol: row.symbol == null ? null : String(row.symbol),
    qualified_symbol: row.qualified_symbol == null ? null : String(row.qualified_symbol),
    parent_symbol: row.parent_symbol == null ? null : String(row.parent_symbol),
    start_line: Number(row.start_line ?? 0),
    end_line: Number(row.end_line ?? 0),
    content: String(row.content ?? ""),
  };
}
