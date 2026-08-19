import { parse as parseToml } from "smol-toml";
import { jsoncAsJson, SERVER_NAME } from "./config-files.ts";

export const ENV_KEYS: Readonly<Record<string, string>> = {
  codex: "env",
  "claude-code": "env",
  "kimi-code": "env",
  "claude-desktop": "env",
  opencode: "environment",
  kilocode: "environment",
};

export const OBJECT_KEYS: Readonly<Record<string, string>> = {
  "claude-code": "mcpServers",
  "kimi-code": "mcpServers",
  "claude-desktop": "mcpServers",
  opencode: "mcp",
  kilocode: "mcp",
};

export function entryFromText(slug: string, text: string): Record<string, unknown> | null {
  let servers: unknown;
  try {
    if (slug === "codex") {
      const parsed = parseToml(text) as Record<string, unknown>;
      servers = parsed.mcp_servers;
    } else {
      const objectKey = OBJECT_KEYS[slug];
      if (objectKey === undefined) return null;
      const parsed = JSON.parse(jsoncAsJson(text)) as Record<string, unknown>;
      servers = parsed[objectKey];
    }
  } catch {
    return null;
  }
  if (servers === null || typeof servers !== "object" || Array.isArray(servers)) return null;
  const entry = (servers as Record<string, unknown>)[SERVER_NAME];
  return entry !== null && typeof entry === "object" && !Array.isArray(entry) ? { ...entry } : null;
}

export function envFromEntry(
  slug: string,
  entry: Readonly<Record<string, unknown>>,
): Record<string, string> {
  const key = ENV_KEYS[slug];
  if (key === undefined) return {};
  const raw = entry[key];
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return {};
  const result: Record<string, string> = {};
  for (const [name, value] of Object.entries(raw)) {
    result[name] = String(value);
  }
  return result;
}

export function commandFromEntry(
  _slug: string,
  entry: Readonly<Record<string, unknown>>,
): string | null {
  const raw = entry.command;
  if (typeof raw === "string") return raw || null;
  if (Array.isArray(raw) && typeof raw[0] === "string") return raw[0] || null;
  return null;
}

export function mergeEnv(
  existing: Readonly<Record<string, string>>,
  updates: Readonly<Record<string, string | null>>,
): Record<string, string> {
  const merged = { ...existing };
  for (const [key, value] of Object.entries(updates)) {
    if (value === null) {
      delete merged[key];
    } else {
      merged[key] = value;
    }
  }
  return merged;
}
