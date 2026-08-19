import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { parse as parseToml } from "smol-toml";

export const SERVER_NAME = "code-indexing-mcp";

export class InstallerError extends Error {
  override readonly name = "InstallerError";
}

interface JsonMember {
  readonly key: string;
  readonly keyStart: number;
  readonly valueStart: number;
  readonly valueEnd: number;
}

function skipJsoncTrivia(text: string, position: number): number {
  while (position < text.length) {
    const current = text[position] ?? "";
    if (/\s/.test(current)) {
      position += 1;
      continue;
    }
    if (text.startsWith("//", position)) {
      const newline = text.indexOf("\n", position + 2);
      return newline === -1 ? text.length : skipJsoncTrivia(text, newline + 1);
    }
    if (text.startsWith("/*", position)) {
      const end = text.indexOf("*/", position + 2);
      if (end === -1) throw new Error("unterminated block comment");
      position = end + 2;
      continue;
    }
    break;
  }
  return position;
}

function parseJsonString(text: string, position: number): [string, number] {
  if (position >= text.length || text[position] !== '"') {
    throw new Error("object keys must be double-quoted strings");
  }
  let end = position + 1;
  while (end < text.length) {
    if (text[end] === "\\") {
      end += 2;
      continue;
    }
    if (text[end] === '"') {
      end += 1;
      try {
        return [JSON.parse(text.slice(position, end)) as string, end];
      } catch {
        throw new Error("invalid JSON string");
      }
    }
    end += 1;
  }
  throw new Error("unterminated JSON string");
}

function scanJsoncValue(text: string, position: number): number {
  position = skipJsoncTrivia(text, position);
  if (position >= text.length) throw new Error("missing value");
  const current = text[position] ?? "";
  if (current === '"') {
    const parsed = parseJsonString(text, position);
    return parsed[1];
  }
  if (current === "[" || current === "{") {
    const stack = [current === "[" ? "]" : "}"];
    let cursor = position + 1;
    while (cursor < text.length) {
      if (text[cursor] === '"') {
        cursor = parseJsonString(text, cursor)[1];
        continue;
      }
      if (text.startsWith("//", cursor) || text.startsWith("/*", cursor)) {
        cursor = skipJsoncTrivia(text, cursor);
        continue;
      }
      const token = text[cursor] ?? "";
      if (token === "[" || token === "{") {
        stack.push(token === "[" ? "]" : "}");
      } else if (token === "]" || token === "}") {
        if (token !== stack[stack.length - 1])
          throw new Error(`unexpected ${JSON.stringify(token)}`);
        stack.pop();
        if (stack.length === 0) return cursor + 1;
      }
      cursor += 1;
    }
    throw new Error("unterminated object or array");
  }

  let end = position;
  while (end < text.length && text[end] !== "," && text[end] !== "}") {
    if (text.startsWith("//", end) || text.startsWith("/*", end)) break;
    end += 1;
  }
  const token = text.slice(position, end).trim();
  if (token === "") throw new Error("missing value");
  try {
    JSON.parse(token);
  } catch {
    throw new Error(`invalid value ${JSON.stringify(token)}`);
  }
  return position + text.slice(position, end).trimEnd().length;
}

function jsoncObjectMembers(text: string, objectStart: number): [JsonMember[], number] {
  if (objectStart >= text.length || text[objectStart] !== "{") {
    throw new Error("expected an object");
  }
  const members: JsonMember[] = [];
  let position = objectStart + 1;
  for (;;) {
    position = skipJsoncTrivia(text, position);
    if (position >= text.length) throw new Error("unterminated object");
    if (text[position] === "}") return [members, position];
    const keyStart = position;
    const parsed = parseJsonString(text, position);
    const key = parsed[0];
    position = skipJsoncTrivia(text, parsed[1]);
    if (position >= text.length || text[position] !== ":") {
      throw new Error(`missing colon after ${JSON.stringify(key)}`);
    }
    const valueStart = skipJsoncTrivia(text, position + 1);
    const valueEnd = scanJsoncValue(text, valueStart);
    members.push({ key, keyStart, valueStart, valueEnd });
    position = skipJsoncTrivia(text, valueEnd);
    if (position >= text.length) throw new Error("unterminated object");
    if (text[position] === ",") {
      position += 1;
      continue;
    }
    if (text[position] === "}") return [members, position];
    throw new Error(`expected a comma or closing brace after ${JSON.stringify(key)}`);
  }
}

function lineIndent(text: string, position: number): string {
  const lineStart = text.lastIndexOf("\n", position - 1) + 1;
  const prefix = text.slice(lineStart, position);
  return prefix.slice(0, prefix.length - prefix.trimStart().length);
}

function formatJsonValue(value: unknown, baseIndent: string): string {
  const lines = JSON.stringify(value, null, 2).split("\n");
  const first = lines[0] ?? "";
  return (
    first +
    lines
      .slice(1)
      .map((line) => `\n${baseIndent}${line}`)
      .join("")
  );
}

export function jsoncAsJson(text: string): string {
  const withoutComments: string[] = [];
  let position = 0;
  while (position < text.length) {
    if (text[position] === '"') {
      const end = parseJsonString(text, position)[1];
      withoutComments.push(text.slice(position, end));
      position = end;
      continue;
    }
    if (text.startsWith("//", position)) {
      const end = text.indexOf("\n", position + 2);
      if (end === -1) {
        withoutComments.push(" ".repeat(text.length - position));
        break;
      }
      withoutComments.push(" ".repeat(end - position));
      position = end;
      continue;
    }
    if (text.startsWith("/*", position)) {
      const end = text.indexOf("*/", position + 2);
      if (end === -1) throw new Error("unterminated block comment");
      const comment = text.slice(position, end + 2);
      withoutComments.push(
        [...comment].map((character) => (character === "\n" ? "\n" : " ")).join(""),
      );
      position = end + 2;
      continue;
    }
    withoutComments.push(text[position] ?? "");
    position += 1;
  }

  const cleaned = withoutComments.join("");
  const withoutTrailingCommas: string[] = [];
  position = 0;
  while (position < cleaned.length) {
    if (cleaned[position] === '"') {
      const end = parseJsonString(cleaned, position)[1];
      withoutTrailingCommas.push(cleaned.slice(position, end));
      position = end;
      continue;
    }
    if (cleaned[position] === ",") {
      let nextToken = position + 1;
      while (nextToken < cleaned.length && /\s/.test(cleaned[nextToken] ?? "")) {
        nextToken += 1;
      }
      if (
        nextToken < cleaned.length &&
        (cleaned[nextToken] === "]" || cleaned[nextToken] === "}")
      ) {
        position += 1;
        continue;
      }
    }
    withoutTrailingCommas.push(cleaned[position] ?? "");
    position += 1;
  }
  return withoutTrailingCommas.join("");
}

function validateJsonc(text: string): void {
  try {
    JSON.parse(jsoncAsJson(text));
  } catch (error) {
    throw new Error(error instanceof Error ? error.message : String(error));
  }
}

function insertJsoncMember(
  text: string,
  objectStart: number,
  objectEnd: number,
  members: readonly JsonMember[],
  key: string,
  value: unknown,
): string {
  let updated = text;
  let closing = objectEnd;
  const last = members[members.length - 1];
  if (last !== undefined) {
    const afterValue = skipJsoncTrivia(updated, last.valueEnd);
    if (afterValue >= updated.length || updated[afterValue] !== ",") {
      updated = `${updated.slice(0, last.valueEnd)},${updated.slice(last.valueEnd)}`;
      closing += 1;
    }
  }

  const closingLineStart = updated.lastIndexOf("\n", closing - 1) + 1;
  let insertionPoint: number;
  let closingIndent: string;
  let closingSuffix: string;
  if (updated.slice(closingLineStart, closing).trim() !== "") {
    insertionPoint = closing;
    closingIndent = lineIndent(updated, objectStart);
    closingSuffix = `\n${closingIndent}`;
  } else {
    insertionPoint = closingLineStart;
    closingIndent = updated.slice(closingLineStart, closing);
    closingSuffix = "";
  }

  const memberIndent = `${closingIndent}  `;
  const encodedKey = JSON.stringify(key);
  const encodedValue = formatJsonValue(value, memberIndent);
  const leadingNewline = updated.slice(0, insertionPoint).endsWith("\n") ? "" : "\n";
  const addition = `${leadingNewline}${memberIndent}${encodedKey}: ${encodedValue}\n${closingSuffix}`;
  return updated.slice(0, insertionPoint) + addition + updated.slice(insertionPoint);
}

function mergeJsoncText(
  text: string,
  objectKey: string,
  entryKey: string,
  entryValue: unknown,
): string {
  const rootStart = skipJsoncTrivia(text, 0);
  if (rootStart >= text.length || text[rootStart] !== "{") {
    throw new Error("configuration root must be an object");
  }
  const [rootMembers, rootEnd] = jsoncObjectMembers(text, rootStart);
  if (skipJsoncTrivia(text, rootEnd + 1) !== text.length) {
    throw new Error("unexpected content after the root object");
  }

  const rootMember = rootMembers.find((member) => member.key === objectKey);
  if (rootMember === undefined) {
    return insertJsoncMember(text, rootStart, rootEnd, rootMembers, objectKey, {
      [entryKey]: entryValue,
    });
  }

  const objectStart = rootMember.valueStart;
  if (text[objectStart] !== "{") {
    throw new Error(`${JSON.stringify(objectKey)} must contain an object`);
  }
  const [entries, objectEnd] = jsoncObjectMembers(text, objectStart);
  if (objectEnd + 1 !== rootMember.valueEnd) {
    throw new Error(`${JSON.stringify(objectKey)} has an invalid object value`);
  }
  const entry = entries.find((member) => member.key === entryKey);
  if (entry === undefined) {
    return insertJsoncMember(text, objectStart, objectEnd, entries, entryKey, entryValue);
  }

  const replacement = formatJsonValue(entryValue, lineIndent(text, entry.valueStart));
  return text.slice(0, entry.valueStart) + replacement + text.slice(entry.valueEnd);
}

function removeJsoncMember(text: string, members: readonly JsonMember[], index: number): string {
  const member = members[index];
  if (member === undefined) return text;
  let start = member.keyStart;
  let end = member.valueEnd;
  const after = skipJsoncTrivia(text, end);
  if (after < text.length && text[after] === ",") {
    end = after + 1;
  } else if (index > 0) {
    const previous = members[index - 1];
    if (previous !== undefined) {
      const comma = skipJsoncTrivia(text, previous.valueEnd);
      if (comma < text.length && text[comma] === ",") start = comma;
    }
  }

  const lineStart = text.lastIndexOf("\n", start - 1) + 1;
  if (text.slice(lineStart, start).trim() === "") {
    const newline = text.indexOf("\n", end);
    const lineEnd = newline === -1 ? text.length : newline + 1;
    if (text.slice(end, lineEnd).trim() === "") {
      start = lineStart;
      end = lineEnd;
    }
  }
  return text.slice(0, start) + text.slice(end);
}

function removeJsoncEntry(text: string, objectKey: string, entryKey: string): string | null {
  const rootStart = skipJsoncTrivia(text, 0);
  if (rootStart >= text.length || text[rootStart] !== "{") {
    throw new Error("configuration root must be an object");
  }
  const [rootMembers] = jsoncObjectMembers(text, rootStart);
  const rootMember = rootMembers.find((member) => member.key === objectKey);
  if (rootMember === undefined) return null;
  if (text[rootMember.valueStart] !== "{") {
    throw new Error(`${JSON.stringify(objectKey)} must contain an object`);
  }
  const [entries] = jsoncObjectMembers(text, rootMember.valueStart);
  const index = entries.findIndex((member) => member.key === entryKey);
  if (index === -1) return null;
  const updated = removeJsoncMember(text, entries, index);
  if (entries.length > 1) return updated;
  const nextRootStart = skipJsoncTrivia(updated, 0);
  const [nextRootMembers] = jsoncObjectMembers(updated, nextRootStart);
  const container = nextRootMembers.findIndex((member) => member.key === objectKey);
  if (container === -1) return updated;
  const containerMember = nextRootMembers[container];
  if (containerMember === undefined) return updated;
  const [remaining] = jsoncObjectMembers(updated, containerMember.valueStart);
  if (remaining.length > 0) return updated;
  return removeJsoncMember(updated, nextRootMembers, container);
}

export function atomicWrite(filePath: string, content: string): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true, mode: 0o700 });
  const temporary = path.join(
    path.dirname(filePath),
    `.${path.basename(filePath)}.${process.pid}.${Math.random().toString(16).slice(2)}.tmp`,
  );
  try {
    fs.writeFileSync(temporary, content, { encoding: "utf8" });
    if (fs.existsSync(filePath)) {
      fs.chmodSync(temporary, fs.statSync(filePath).mode);
    }
    fs.renameSync(temporary, filePath);
  } finally {
    fs.rmSync(temporary, { force: true });
  }
}

function backupConfiguration(filePath: string): void {
  const directory = path.dirname(filePath);
  const name = path.basename(filePath);
  const pristine = path.join(directory, `${name}.bak`);
  try {
    if (fs.lstatSync(pristine).isSymbolicLink() || fs.existsSync(pristine)) {
      fs.copyFileSync(filePath, path.join(directory, `${name}.bak.prev`));
      return;
    }
  } catch {
    // No previous backup.
  }
  fs.copyFileSync(filePath, pristine);
}

export function writeChangedConfiguration(
  filePath: string,
  original: string | null,
  updated: string,
): boolean {
  if (original === updated) return false;
  if (original !== null) backupConfiguration(filePath);
  atomicWrite(filePath, updated);
  return true;
}

function readConfiguration(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null;
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return null;
    if (error instanceof TypeError) {
      throw new InstallerError(`Configuration must be UTF-8: ${filePath}`);
    }
    throw new InstallerError(`Configuration must be UTF-8: ${filePath}`);
  }
}

export function mergeJsonObjectEntry(
  filePath: string,
  objectKey: string,
  entryKey: string,
  entryValue: unknown,
): boolean {
  const original = readConfiguration(filePath);
  const source = original !== null && original.trim() !== "" ? original : "{}\n";
  let updated: string;
  try {
    validateJsonc(source);
    updated = mergeJsoncText(source, objectKey, entryKey, entryValue);
    validateJsonc(updated);
  } catch (error) {
    throw new InstallerError(
      `Invalid JSON/JSONC configuration in ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return writeChangedConfiguration(filePath, original, updated);
}

export function removeJsonObjectEntry(
  filePath: string,
  objectKey: string,
  entryKey: string,
): boolean {
  const original = readConfiguration(filePath);
  if (original === null || original.trim() === "") return false;
  let updated: string | null;
  try {
    validateJsonc(original);
    updated = removeJsoncEntry(original, objectKey, entryKey);
    if (updated === null) return false;
    validateJsonc(updated);
  } catch (error) {
    throw new InstallerError(
      `Invalid JSON/JSONC configuration in ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  return writeChangedConfiguration(filePath, original, updated);
}

const TOML_TABLE =
  /^[ \t]*(?:\[\[\s*(?<array_name>[^\]\n]+?)\s*\]\]|\[\s*(?<table_name>[^\]\n]+?)\s*\])[ \t]*(?:#.*)?$/gm;

function splitTomlDottedKey(value: string): string[] {
  const components: string[] = [];
  let position = 0;
  while (position < value.length) {
    while (position < value.length && /\s/.test(value[position] ?? "")) position += 1;
    if (position >= value.length) throw new Error("empty table component");
    let component: string;
    if (value[position] === '"') {
      const parsed = parseJsonString(value, position);
      component = parsed[0];
      position = parsed[1];
    } else if (value[position] === "'") {
      const end = value.indexOf("'", position + 1);
      if (end === -1) throw new Error("unterminated literal table key");
      component = value.slice(position + 1, end);
      position = end + 1;
    } else {
      const match = /^[A-Za-z0-9_-]+/.exec(value.slice(position));
      if (match === null) throw new Error("invalid bare table key");
      component = match[0];
      position += component.length;
    }
    components.push(component);
    while (position < value.length && /\s/.test(value[position] ?? "")) position += 1;
    if (position === value.length) return components;
    if (value[position] !== ".") throw new Error("expected a dot between table keys");
    position += 1;
  }
  throw new Error("empty table component");
}

function codexServerBlock(command: string, env?: Readonly<Record<string, string>>): string {
  const encodedCommand = JSON.stringify(command);
  const lines = [`[mcp_servers.${SERVER_NAME}]`, `command = ${encodedCommand}`, 'args = ["serve"]'];
  if (env !== undefined && Object.keys(env).length > 0) {
    const pairs = Object.keys(env)
      .sort()
      .map((key) => `${key} = ${JSON.stringify(env[key])}`)
      .join(", ");
    lines.push(`env = { ${pairs} }`);
  }
  return `${lines.join("\n")}\n`;
}

function trailingTomlTrivia(text: string): string {
  const lines = text.split(/(?<=\n)/);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const stripped = (lines[index] ?? "").trim();
    if (stripped !== "" && !stripped.startsWith("#")) {
      return lines.slice(index + 1).join("");
    }
  }
  return text;
}

interface TomlHeading {
  readonly start: number;
  readonly components: readonly string[];
  readonly isArray: boolean;
}

function loadToml(source: string, filePath: string): void {
  try {
    parseToml(source);
  } catch (error) {
    throw new InstallerError(
      `Invalid TOML configuration in ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function refuseInvalidToml(source: string, filePath: string): void {
  try {
    parseToml(source);
  } catch (error) {
    throw new InstallerError(
      `Refusing to write invalid TOML configuration to ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function codexHeadings(source: string, filePath: string): TomlHeading[] {
  const headings: TomlHeading[] = [];
  const matcher = new RegExp(TOML_TABLE.source, TOML_TABLE.flags);
  for (const match of source.matchAll(matcher)) {
    try {
      const arrayName = match.groups?.array_name;
      const name = arrayName ?? match.groups?.table_name;
      if (name === undefined) continue;
      headings.push({
        start: match.index ?? 0,
        components: splitTomlDottedKey(name),
        isArray: arrayName !== undefined,
      });
    } catch (error) {
      throw new InstallerError(
        `Invalid TOML table in ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  return headings;
}

function prefixEquals(left: readonly string[], right: readonly string[]): boolean {
  return right.every((part, index) => left[index] === part);
}

function codexTargetSpan(
  source: string,
  headings: readonly TomlHeading[],
  index: number,
): [number, number] {
  const target = ["mcp_servers", SERVER_NAME];
  const heading = headings[index];
  if (heading === undefined) return [0, source.length];
  const start = heading.start;
  let end = source.length;
  for (const next of headings.slice(index + 1)) {
    if (!prefixEquals(next.components, target)) {
      end = next.start;
      break;
    }
  }
  return [start, end];
}

function codexTargetIndex(headings: readonly TomlHeading[]): number | null {
  const target = ["mcp_servers", SERVER_NAME];
  const index = headings.findIndex(
    (heading) =>
      !heading.isArray &&
      heading.components.length === target.length &&
      heading.components.every((part, position) => part === target[position]),
  );
  return index === -1 ? null : index;
}

export function removeCodexServer(filePath: string): boolean {
  const original = readConfiguration(filePath);
  if (original === null || original.trim() === "") return false;
  loadToml(original, filePath);
  const headings = codexHeadings(original, filePath);
  const index = codexTargetIndex(headings);
  if (index === null) return false;
  const [start, end] = codexTargetSpan(original, headings, index);
  const trivia = trailingTomlTrivia(original.slice(start, end));
  let prefix = original.slice(0, start);
  if (prefix.trim() === "") {
    prefix = "";
  } else if (prefix.endsWith("\n\n")) {
    prefix = prefix.slice(0, -1);
  }
  const updated = prefix + trivia + original.slice(end);
  refuseInvalidToml(updated, filePath);
  return writeChangedConfiguration(filePath, original, updated);
}

export function mergeCodexServer(
  filePath: string,
  command: string,
  env?: Readonly<Record<string, string>>,
): boolean {
  const original = readConfiguration(filePath);
  const source = original ?? "";
  let parsed: Record<string, unknown> = {};
  if (source.trim() !== "") {
    try {
      parsed = parseToml(source) as Record<string, unknown>;
    } catch (error) {
      throw new InstallerError(
        `Invalid TOML configuration in ${filePath}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  const headings = codexHeadings(source, filePath);
  const targetIndex = codexTargetIndex(headings);
  const block = codexServerBlock(command, env);
  let updated: string;
  if (targetIndex === null) {
    const mcpServers = parsed.mcp_servers;
    if (
      mcpServers !== null &&
      typeof mcpServers === "object" &&
      !Array.isArray(mcpServers) &&
      SERVER_NAME in mcpServers
    ) {
      throw new InstallerError(
        `Codex server ${JSON.stringify(SERVER_NAME)} uses an inline or dotted TOML definition in ` +
          `${filePath}; convert it to [mcp_servers.${SERVER_NAME}] before rerunning`,
      );
    }
    const prefix = source.trim() !== "" ? `${source.trimEnd()}\n\n` : "";
    updated = prefix + block;
  } else {
    const [start, end] = codexTargetSpan(source, headings, targetIndex);
    const trivia = trailingTomlTrivia(source.slice(start, end));
    updated = source.slice(0, start) + block + trivia + source.slice(end);
  }

  refuseInvalidToml(updated, filePath);
  return writeChangedConfiguration(filePath, original, updated);
}

export function expandUser(value: string): string {
  if (value === "~") return os.homedir();
  if (value.startsWith("~/") || value.startsWith(`~${path.sep}`)) {
    return path.join(os.homedir(), value.slice(2));
  }
  return value;
}

export function resolveExisting(value: string): string {
  return path.resolve(expandUser(value));
}
