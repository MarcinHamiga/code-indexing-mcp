import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  expandUser,
  InstallerError,
  mergeCodexServer,
  mergeJsonObjectEntry,
  removeCodexServer,
  removeJsonObjectEntry,
  SERVER_NAME,
} from "./config-files.ts";
import { entryFromText, envFromEntry, mergeEnv, OBJECT_KEYS } from "./env-blocks.ts";
import { isUnder, linkDestination, replaceLink } from "./links.ts";

export interface HarnessChoice {
  readonly slug: string;
  readonly label: string;
}

export const HARNESS_CHOICES: readonly HarnessChoice[] = [
  { slug: "codex", label: "Codex (CLI + Desktop)" },
  { slug: "claude-code", label: "Claude Code" },
  { slug: "kimi-code", label: "Kimi Code" },
  { slug: "claude-desktop", label: "Claude Desktop" },
  { slug: "opencode", label: "OpenCode" },
  { slug: "kilocode", label: "KiloCode" },
];

export function parseHarnessSelection(selection: string): string[] {
  const value = selection.trim().toLowerCase();
  if (value === "") return [];
  if (value === "all") return HARNESS_CHOICES.map((choice) => choice.slug);

  const bySlug = new Map(HARNESS_CHOICES.map((choice) => [choice.slug, choice.slug]));
  const byNumber = new Map(
    HARNESS_CHOICES.map((choice, index) => [String(index + 1), choice.slug]),
  );
  const selected: string[] = [];
  for (const token of value.split(",").map((part) => part.trim().toLowerCase())) {
    const slug = byNumber.get(token) ?? bySlug.get(token);
    if (slug === undefined) {
      const options = HARNESS_CHOICES.map((choice) => choice.slug).join(", ");
      throw new InstallerError(
        `Unknown harness ${JSON.stringify(token)}; choose 1-6, all, or one of: ${options}`,
      );
    }
    if (!selected.includes(slug)) selected.push(slug);
  }
  return selected;
}

function configuredDirectory(
  environment: NodeJS.ProcessEnv,
  variable: string,
  fallback: string,
): string {
  const configured = environment[variable];
  return configured !== undefined && configured !== "" ? expandUser(configured) : fallback;
}

function preferredJsonConfig(directory: string, stem: string, defaultSuffix: string): string {
  const jsonPath = path.join(directory, `${stem}.json`);
  const jsoncPath = path.join(directory, `${stem}.jsonc`);
  if (fs.existsSync(jsonPath)) return jsonPath;
  if (fs.existsSync(jsoncPath)) return jsoncPath;
  return path.join(directory, `${stem}${defaultSuffix}`);
}

export function configurationPath(
  slug: string,
  options: {
    home?: string;
    environment?: NodeJS.ProcessEnv;
    platformName?: string;
  } = {},
): string {
  const home = options.home ?? os.homedir();
  const environment = options.environment ?? process.env;
  const platformName = options.platformName ?? process.platform;
  const xdgConfig = configuredDirectory(environment, "XDG_CONFIG_HOME", path.join(home, ".config"));

  if (slug === "codex") {
    return path.join(
      configuredDirectory(environment, "CODEX_HOME", path.join(home, ".codex")),
      "config.toml",
    );
  }
  if (slug === "claude-code") {
    return path.join(configuredDirectory(environment, "CLAUDE_CONFIG_DIR", home), ".claude.json");
  }
  if (slug === "kimi-code") {
    return path.join(
      configuredDirectory(environment, "KIMI_CODE_HOME", path.join(home, ".kimi-code")),
      "mcp.json",
    );
  }
  if (slug === "claude-desktop") {
    if (platformName === "darwin") {
      return path.join(
        home,
        "Library",
        "Application Support",
        "Claude",
        "claude_desktop_config.json",
      );
    }
    if (platformName.startsWith("win")) {
      const appData = environment.APPDATA;
      if (appData === undefined || appData === "") {
        throw new InstallerError("APPDATA is required to configure Claude Desktop on Windows");
      }
      return path.join(expandUser(appData), "Claude", "claude_desktop_config.json");
    }
    if (platformName.startsWith("linux")) {
      return path.join(xdgConfig, "Claude", "claude_desktop_config.json");
    }
    throw new InstallerError(`Claude Desktop configuration is not supported on ${platformName}`);
  }
  if (slug === "opencode") {
    const configured = environment.OPENCODE_CONFIG;
    if (configured !== undefined && configured !== "") return expandUser(configured);
    const directory = configuredDirectory(
      environment,
      "OPENCODE_CONFIG_DIR",
      path.join(xdgConfig, "opencode"),
    );
    return preferredJsonConfig(directory, "opencode", ".json");
  }
  if (slug === "kilocode") {
    const configured = environment.KILO_CONFIG;
    if (configured !== undefined && configured !== "") return expandUser(configured);
    const directory = configuredDirectory(
      environment,
      "KILO_CONFIG_DIR",
      path.join(xdgConfig, "kilo"),
    );
    return preferredJsonConfig(directory, "kilo", ".jsonc");
  }
  throw new InstallerError(`Unknown harness ${JSON.stringify(slug)}`);
}

export function readServerEntry(
  slug: string,
  options: {
    home?: string;
    environment?: NodeJS.ProcessEnv;
    platformName?: string;
  } = {},
): Record<string, unknown> | null {
  const filePath = configurationPath(slug, options);
  let text: string;
  try {
    text = fs.readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
  return entryFromText(slug, text);
}

export function configureHarness(
  slug: string,
  command: string,
  options: {
    env?: Readonly<Record<string, string | null>>;
    home?: string;
    environment?: NodeJS.ProcessEnv;
    platformName?: string;
  } = {},
): string {
  const filePath = configurationPath(slug, options);
  let mergedEnv: Record<string, string> = {};
  if (options.env !== undefined) {
    const existing = readServerEntry(slug, options);
    mergedEnv = mergeEnv(existing === null ? {} : envFromEntry(slug, existing), options.env);
  }
  if (slug === "codex") {
    mergeCodexServer(filePath, command, options.env === undefined ? undefined : mergedEnv);
    return filePath;
  }

  let objectKey: string;
  let entry: Record<string, unknown>;
  if (slug === "claude-code") {
    objectKey = "mcpServers";
    entry = { type: "stdio", command, args: ["serve"] };
    if (Object.keys(mergedEnv).length > 0) entry.env = mergedEnv;
  } else if (slug === "kimi-code" || slug === "claude-desktop") {
    objectKey = "mcpServers";
    entry = { command, args: ["serve"] };
    if (Object.keys(mergedEnv).length > 0) entry.env = mergedEnv;
  } else if (slug === "opencode" || slug === "kilocode") {
    objectKey = "mcp";
    entry = { type: "local", command: [command, "serve"], enabled: true };
    if (Object.keys(mergedEnv).length > 0) entry.environment = mergedEnv;
  } else {
    throw new InstallerError(`Unknown harness ${JSON.stringify(slug)}`);
  }

  mergeJsonObjectEntry(filePath, objectKey, SERVER_NAME, entry);
  return filePath;
}

export function deconfigureHarness(
  slug: string,
  options: {
    home?: string;
    environment?: NodeJS.ProcessEnv;
    platformName?: string;
  } = {},
): [string, boolean] {
  const filePath = configurationPath(slug, options);
  if (!fs.existsSync(filePath)) return [filePath, false];
  if (slug === "codex") return [filePath, removeCodexServer(filePath)];
  const objectKey = OBJECT_KEYS[slug];
  if (objectKey === undefined) throw new InstallerError(`Unknown harness ${JSON.stringify(slug)}`);
  return [filePath, removeJsonObjectEntry(filePath, objectKey, SERVER_NAME)];
}

export function deconfigureSelectedHarnesses(
  slugs: readonly string[],
  options: {
    home?: string;
    environment?: NodeJS.ProcessEnv;
    platformName?: string;
  } = {},
): [[string, string, boolean][], [string, string][]] {
  const removed: [string, string, boolean][] = [];
  const failures: [string, string][] = [];
  for (const slug of slugs) {
    try {
      const [filePath, changed] = deconfigureHarness(slug, options);
      removed.push([slug, filePath, changed]);
    } catch (error) {
      failures.push([slug, error instanceof Error ? error.message : String(error)]);
    }
  }
  return [removed, failures];
}

export function skillDirectory(
  slug: string,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): string | null {
  const home = options.home ?? os.homedir();
  const environment = options.environment ?? process.env;
  if (slug === "claude-code") {
    return path.join(
      configuredDirectory(environment, "CLAUDE_CONFIG_DIR", path.join(home, ".claude")),
      "skills",
    );
  }
  if (slug === "codex" || slug === "kimi-code") return path.join(home, ".agents", "skills");
  if (slug === "opencode") {
    const xdgConfig = configuredDirectory(
      environment,
      "XDG_CONFIG_HOME",
      path.join(home, ".config"),
    );
    return path.join(xdgConfig, "opencode", "skills");
  }
  return null;
}

export function isBundledSkillLink(target: string, installDirectory?: string): boolean {
  try {
    if (!fs.lstatSync(target).isSymbolicLink()) return false;
  } catch {
    return false;
  }
  const destination = linkDestination(target);
  const skillsDir = path.dirname(destination);
  if (
    path.basename(skillsDir) !== "skills" ||
    path.basename(path.dirname(skillsDir)) !== "code_indexing_mcp"
  ) {
    return false;
  }
  if (installDirectory === undefined) return true;
  return isUnder(destination, installDirectory);
}

function isStaleBundledLink(target: string): boolean {
  return isBundledSkillLink(target);
}

function linkSkill(source: string, target: string): boolean {
  return replaceLink(source, target, {
    isDirectory: true,
    stale: isStaleBundledLink(target),
  });
}

export function removeSkills(
  slugs: readonly string[],
  installDirectory?: string,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): [string, string][] {
  const results: [string, string][] = [];
  for (const slug of slugs) {
    const directory = skillDirectory(slug, options);
    if (directory === null || !fs.existsSync(directory) || !fs.statSync(directory).isDirectory()) {
      results.push([slug, "skipped: no skill directory"]);
      continue;
    }
    let removed = 0;
    try {
      for (const name of fs.readdirSync(directory).sort()) {
        const entry = path.join(directory, name);
        if (isBundledSkillLink(entry, installDirectory)) {
          fs.unlinkSync(entry);
          removed += 1;
        }
      }
    } catch (error) {
      results.push([slug, `skipped: ${error instanceof Error ? error.message : String(error)}`]);
      continue;
    }
    results.push([slug, `${removed} unlinked from ${directory}`]);
  }
  return results;
}

export function configureSelectedHarnesses(
  slugs: readonly string[],
  command: string,
  options: {
    env?: Readonly<Record<string, string | null>>;
    home?: string;
    environment?: NodeJS.ProcessEnv;
    platformName?: string;
  } = {},
): [[string, string][], [string, string][]] {
  const successes: [string, string][] = [];
  const failures: [string, string][] = [];
  for (const slug of slugs) {
    try {
      successes.push([slug, configureHarness(slug, command, options)]);
    } catch (error) {
      failures.push([slug, error instanceof Error ? error.message : String(error)]);
    }
  }
  return [successes, failures];
}

export function installSkills(
  slugs: readonly string[],
  installDirectory: string,
  options: { home?: string; environment?: NodeJS.ProcessEnv } = {},
): [string, string][] {
  const skillsSource = path.join(installDirectory, "src", "code_indexing_mcp", "skills");
  if (!fs.existsSync(skillsSource) || !fs.statSync(skillsSource).isDirectory()) {
    return slugs.map((slug) => [slug, `skipped: bundled skills not found at ${skillsSource}`]);
  }
  const skills = fs
    .readdirSync(skillsSource)
    .map((name) => path.join(skillsSource, name))
    .filter((entry) => fs.existsSync(path.join(entry, "SKILL.md")))
    .sort();
  const results: [string, string][] = [];
  for (const slug of slugs) {
    const directory = skillDirectory(slug, options);
    if (directory === null) {
      results.push([slug, "skipped: harness has no skill-directory support"]);
      continue;
    }
    try {
      const created = skills.map((skill) =>
        linkSkill(skill, path.join(directory, path.basename(skill))),
      );
      const linked = created.filter(Boolean).length;
      results.push([
        slug,
        `${linked} linked, ${created.length - linked} already installed in ${directory}`,
      ]);
    } catch (error) {
      results.push([slug, `skipped: ${error instanceof Error ? error.message : String(error)}`]);
    }
  }
  return results;
}

export function harnessLabel(slug: string): string {
  return HARNESS_CHOICES.find((choice) => choice.slug === slug)?.label ?? slug;
}
