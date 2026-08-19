import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { serverExecutable } from "./accelerator.ts";
import { expandUser, InstallerError, writeChangedConfiguration } from "./config-files.ts";
import { backupPath, isUnder, linkDestination, replaceLink } from "./links.ts";

export const LAUNCHER_NAME = "code-indexing-mcp";
export const BLOCK_START = "# >>> code-indexing-mcp >>>";
export const BLOCK_END = "# <<< code-indexing-mcp <<<";

export interface LauncherResult {
  readonly path: string;
  readonly status: "created" | "current" | "replaced" | "skipped" | "failed";
  readonly detail: string;
}

export function launcherOk(result: LauncherResult): boolean {
  return result.status === "created" || result.status === "current" || result.status === "replaced";
}

export interface PathState {
  readonly binDirectory: string;
  readonly launcher: string;
  readonly onPath: boolean;
  readonly shadowedBy: string | null;
  readonly profiles: readonly string[];
  readonly profilesCurrent: boolean;
}

export function defaultBinDirectory(
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): string {
  const home = options.home ?? os.homedir();
  const environment = options.environment ?? process.env;
  for (const variable of ["CODE_INDEXING_MCP_BIN_DIR", "XDG_BIN_HOME"]) {
    const configured = environment[variable];
    if (configured !== undefined && configured !== "") return expandUser(configured);
  }
  return path.join(home, ".local", "bin");
}

export function launcherPath(
  binDirectory: string,
  platformName: string = process.platform,
): string {
  if (platformName.startsWith("win")) return path.join(binDirectory, `${LAUNCHER_NAME}.cmd`);
  return path.join(binDirectory, LAUNCHER_NAME);
}

function pathEntries(environment: NodeJS.ProcessEnv): string[] {
  const raw = environment.PATH ?? environment.Path ?? "";
  const entries: string[] = [];
  for (const part of raw.split(path.delimiter)) {
    if (part === "") continue;
    try {
      entries.push(expandUser(part));
    } catch {}
  }
  return entries;
}

function sameDirectory(left: string, right: string): boolean {
  try {
    return path.resolve(left) === path.resolve(right);
  } catch {
    return left === right;
  }
}

export function isOnPath(
  binDirectory: string,
  options: { environment?: NodeJS.ProcessEnv } = {},
): boolean {
  const environment = options.environment ?? process.env;
  return pathEntries(environment).some((entry) => sameDirectory(entry, binDirectory));
}

function whichIn(name: string, directory: string): string | null {
  const candidates =
    process.platform === "win32" ? [name, `${name}.cmd`, `${name}.exe`, `${name}.bat`] : [name];
  for (const candidate of candidates) {
    const filePath = path.join(directory, candidate);
    try {
      if (fs.statSync(filePath).isFile()) return filePath;
    } catch {}
  }
  return null;
}

export function shadowingExecutable(
  binDirectory: string,
  options: { environment?: NodeJS.ProcessEnv; platformName?: string } = {},
): string | null {
  const environment = options.environment ?? process.env;
  const ours = launcherPath(binDirectory, options.platformName);
  for (const entry of pathEntries(environment)) {
    if (sameDirectory(entry, binDirectory)) return null;
    const found = whichIn(LAUNCHER_NAME, entry);
    if (found !== null && !sameDirectory(path.dirname(found), path.dirname(ours))) return found;
  }
  return null;
}

function isOurLauncher(target: string, executable: string): boolean {
  try {
    if (!fs.lstatSync(target).isSymbolicLink()) return false;
  } catch {
    return false;
  }
  const destination = linkDestination(target);
  const parts = destination.split(path.sep);
  return (
    path.basename(destination) === path.basename(executable) &&
    (parts.includes(".venv") || parts.includes("bin"))
  );
}

function installSymlink(executable: string, target: string): LauncherResult {
  const stale = isOurLauncher(target, executable);
  let backup: string | null = null;
  try {
    if ((fs.lstatSync(target).isSymbolicLink() || fs.existsSync(target)) && !stale) {
      backup = backupPath(target);
    }
  } catch {
    // Target does not exist.
  }
  const created = replaceLink(executable, target, { isDirectory: false, stale });
  if (!created) {
    return { path: target, status: "current", detail: `already points at ${executable}` };
  }
  if (backup !== null) {
    return {
      path: target,
      status: "replaced",
      detail: `the previous entry was kept as ${path.basename(backup)}`,
    };
  }
  return { path: target, status: "created", detail: `points at ${executable}` };
}

const SHIM_TEMPLATE = '@echo off\r\n"{executable}" %*\r\n';

function installShim(executable: string, target: string): LauncherResult {
  const content = SHIM_TEMPLATE.replace("{executable}", executable);
  let existing: string | null = null;
  try {
    if (fs.statSync(target).isFile()) existing = fs.readFileSync(target).toString("utf8");
  } catch {
    existing = null;
  }
  if (existing === content) {
    return { path: target, status: "current", detail: `already runs ${executable}` };
  }
  fs.mkdirSync(path.dirname(target), { recursive: true });
  const changed = writeChangedConfiguration(target, existing, content);
  if (!changed) {
    return { path: target, status: "current", detail: `already runs ${executable}` };
  }
  return {
    path: target,
    status: existing !== null ? "replaced" : "created",
    detail: `runs ${executable}`,
  };
}

export function installLauncher(
  installDirectory: string,
  binDirectory: string,
  options: { platformName?: string } = {},
): LauncherResult {
  const platformName = options.platformName ?? process.platform;
  const target = launcherPath(binDirectory, platformName);
  const executable = serverExecutable(installDirectory, platformName);
  try {
    if (!fs.statSync(executable).isFile()) {
      return { path: target, status: "failed", detail: `no server executable at ${executable}` };
    }
  } catch {
    return { path: target, status: "failed", detail: `no server executable at ${executable}` };
  }
  try {
    return platformName.startsWith("win")
      ? installShim(executable, target)
      : installSymlink(executable, target);
  } catch (error) {
    return {
      path: target,
      status: "failed",
      detail: error instanceof Error ? error.message : String(error),
    };
  }
}

function shimCommandCandidates(content: string): string[] {
  const candidates: string[] = [];
  for (const line of content.split(/\r?\n/)) {
    const opening = line.indexOf('"');
    const closing = opening === -1 ? -1 : line.indexOf('"', opening + 1);
    if (closing !== -1) candidates.push(line.slice(opening + 1, closing));
  }
  return candidates;
}

function removableLauncher(
  target: string,
  installDirectory: string | null,
  platformName: string,
): boolean {
  if (platformName.startsWith("win")) {
    try {
      if (!fs.statSync(target).isFile()) return false;
      const content = fs.readFileSync(target, "utf8");
      if (!content.includes("bin") || !content.includes(LAUNCHER_NAME)) return false;
      if (installDirectory === null) return true;
      return shimCommandCandidates(content).some((candidate) =>
        isUnder(candidate, installDirectory),
      );
    } catch {
      return false;
    }
  }
  try {
    if (!fs.lstatSync(target).isSymbolicLink()) return false;
  } catch {
    return false;
  }
  const destination = linkDestination(target);
  const parts = destination.split(path.sep);
  if (!parts.includes(".venv") && !parts.includes("bin")) return false;
  if (installDirectory === null) return true;
  return isUnder(destination, installDirectory);
}

export function removeLauncher(
  binDirectory: string,
  installDirectory: string | null = null,
  options: { platformName?: string } = {},
): string | null {
  const platformName = options.platformName ?? process.platform;
  const target = launcherPath(binDirectory, platformName);
  if (!removableLauncher(target, installDirectory, platformName)) return null;
  fs.unlinkSync(target);
  return target;
}

function fishProfile(home: string, environment: NodeJS.ProcessEnv): string {
  const configured = environment.XDG_CONFIG_HOME;
  const base =
    configured !== undefined && configured !== ""
      ? expandUser(configured)
      : path.join(home, ".config");
  return path.join(base, "fish", "config.fish");
}

export function shellProfiles(
  options: { home?: string; environment?: NodeJS.ProcessEnv; platformName?: string } = {},
): string[] {
  const home = options.home ?? os.homedir();
  const environment = options.environment ?? process.env;
  const platformName = options.platformName ?? process.platform;
  if (platformName.startsWith("win")) return [];

  const zdotdir = environment.ZDOTDIR;
  const zshrc = path.join(
    zdotdir !== undefined && zdotdir !== "" ? expandUser(zdotdir) : home,
    ".zshrc",
  );
  const bashrc = path.join(home, ".bashrc");
  const bashProfile = path.join(home, ".bash_profile");
  const fish = fishProfile(home, environment);
  const profile = path.join(home, ".profile");

  const shell = path.basename(environment.SHELL ?? "");
  let primary: string[] = [];
  if (shell === "zsh") primary = [zshrc];
  else if (shell === "bash") primary = platformName === "darwin" ? [bashrc, bashProfile] : [bashrc];
  else if (shell === "fish") primary = [fish];
  else if (shell !== "") primary = [profile];

  const selected = [...primary];
  for (const candidate of [zshrc, bashrc, bashProfile, fish, profile]) {
    if (
      !selected.includes(candidate) &&
      fs.existsSync(candidate) &&
      fs.statSync(candidate).isFile()
    ) {
      selected.push(candidate);
    }
  }
  if (selected.length === 0) selected.push(profile);
  return selected;
}

const POSIX_SPECIALS = ["\\", '"', "`", "$"] as const;
const FISH_SPECIALS = ["\\", '"', "$"] as const;

function escaped(text: string, specials: readonly string[]): string {
  let value = text;
  for (const character of specials) {
    value = value.split(character).join(`\\${character}`);
  }
  return value;
}

function quotedLocation(binDirectory: string, home: string, specials: readonly string[]): string {
  const relative = path.relative(home, binDirectory);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    return escaped(binDirectory, specials);
  }
  return `$HOME/${escaped(relative.split(path.sep).join("/"), specials)}`;
}

function block(binDirectory: string, profile: string, home: string): string {
  const line =
    path.basename(profile) === "config.fish"
      ? `fish_add_path "${quotedLocation(binDirectory, home, FISH_SPECIALS)}"`
      : `export PATH="${quotedLocation(binDirectory, home, POSIX_SPECIALS)}:$PATH"`;
  return `${BLOCK_START}\n${line}\n${BLOCK_END}\n`;
}

function homeRelative(filePath: string, home: string): string {
  const relative = path.relative(home, filePath);
  if (relative.startsWith("..") || path.isAbsolute(relative)) return filePath;
  return `$HOME/${relative.split(path.sep).join("/")}`;
}

const PATH_CHARACTERS = new Set(
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./~$".split(""),
);

function mentionsWholePath(line: string, needle: string): boolean {
  let start = line.indexOf(needle);
  while (start !== -1) {
    const end = start + needle.length;
    const before = start === 0 || !PATH_CHARACTERS.has(line[start - 1] ?? "");
    const after = end >= line.length || !PATH_CHARACTERS.has(line[end] ?? "");
    if (before && after) return true;
    start = line.indexOf(needle, start + 1);
  }
  return false;
}

export function profileMentionsDirectory(
  text: string,
  binDirectory: string,
  home: string,
): boolean {
  if (text.includes(BLOCK_START)) return true;
  const needles = new Set([binDirectory, homeRelative(binDirectory, home)]);
  const relative = path.relative(home, binDirectory);
  if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
    needles.add(`~/${relative.split(path.sep).join("/")}`);
  }
  return text
    .split(/\r?\n/)
    .some((line) => [...needles].some((needle) => mentionsWholePath(line, needle)));
}

export function updateProfile(
  profile: string,
  binDirectory: string,
  options: { home?: string } = {},
): boolean {
  const home = options.home ?? os.homedir();
  let original: string | null;
  try {
    original = fs.readFileSync(profile, "utf8");
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") original = null;
    else if (error instanceof Error && error.message.includes("encoding")) {
      throw new InstallerError(`${profile} is not valid UTF-8, so it was left alone`);
    } else {
      throw error;
    }
  }
  if (original !== null && profileMentionsDirectory(original, binDirectory, home)) return false;
  const prefix = original === null || original === "" || original.endsWith("\n") ? "" : "\n";
  const updated = `${original ?? ""}${prefix}${block(binDirectory, profile, home)}`;
  return writeChangedConfiguration(profile, original, updated);
}

export function updateProfiles(
  binDirectory: string,
  profiles: readonly string[],
  options: { home?: string } = {},
): [string[], [string, string][]] {
  const written: string[] = [];
  const failures: [string, string][] = [];
  for (const profile of profiles) {
    try {
      if (updateProfile(profile, binDirectory, options)) written.push(profile);
    } catch (error) {
      failures.push([profile, error instanceof Error ? error.message : String(error)]);
    }
  }
  return [written, failures];
}

export function removePathBlock(profile: string): boolean {
  let original: string;
  try {
    original = fs.readFileSync(profile, "utf8");
  } catch {
    return false;
  }
  let updated = original;
  for (;;) {
    const start = updated.indexOf(BLOCK_START);
    if (start === -1) break;
    const endMarker = updated.indexOf(BLOCK_END, start);
    if (endMarker === -1) break;
    let end = endMarker + BLOCK_END.length;
    if (updated[end] === "\n") end += 1;
    updated = updated.slice(0, start) + updated.slice(end);
  }
  return writeChangedConfiguration(profile, original, updated);
}

export function inspect(
  binDirectory: string,
  options: { home?: string; environment?: NodeJS.ProcessEnv; platformName?: string } = {},
): PathState {
  const home = options.home ?? os.homedir();
  const environment = options.environment ?? process.env;
  const onPath = isOnPath(binDirectory, { environment });
  const profiles = shellProfiles(options);
  let current = true;
  for (const profile of profiles) {
    try {
      const text = fs.readFileSync(profile, "utf8");
      if (!profileMentionsDirectory(text, binDirectory, home)) current = false;
    } catch {
      current = false;
    }
  }
  return {
    binDirectory,
    launcher: launcherPath(binDirectory, options.platformName),
    onPath,
    shadowedBy: shadowingExecutable(binDirectory, options),
    profiles,
    profilesCurrent: current,
  };
}

export function activationHint(
  profiles: readonly string[],
  options: { environment?: NodeJS.ProcessEnv } = {},
): string {
  const environment = options.environment ?? process.env;
  if (profiles.some((profile) => path.basename(profile) === "config.fish")) return "exec fish";
  const shell = environment.SHELL;
  return shell !== undefined && shell !== "" ? `exec ${shell} -l` : "exec $SHELL -l";
}
