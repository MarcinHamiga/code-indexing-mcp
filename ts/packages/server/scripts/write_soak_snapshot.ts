/** Write a Phase 9 soak snapshot with the TypeScript build.
 *
 * Indexes every repository in a manifest with this build's `Application` and
 * records the chunk rows and the search rankings for the manifest's queries,
 * in the snapshot shape `src/soak.ts` compares. Run the Python writer first:
 * it writes the `.ci-mcp/project.toml` markers whose project ids both
 * snapshots must share for chunk ids to be comparable.
 *
 * Usage, from the repository root:
 *
 *     bun ts/packages/server/scripts/write_soak_snapshot.ts \
 *       --manifest soak.manifest.json --output soak-ts.json
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Application, runtimePathsFromEnvironment } from "../src/application.ts";
import { dumpJson } from "../src/jsonable.ts";
import { SoakManifest, SoakSnapshot } from "../src/soak.ts";
import { indexSettingsFromEnvironment } from "../src/settings.ts";
import { checkoutHead } from "../src/update-check.ts";

interface Arguments {
  manifest: string;
  output: string;
  dataDir: string | undefined;
}

function parseArguments(argv: string[]): Arguments {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    if (flag === undefined || !flag.startsWith("--") || argv[index + 1] === undefined) {
      throw new Error(`expected --flag value pairs, got: ${argv.join(" ")}`);
    }
    values.set(flag.slice(2), argv[index + 1] as string);
  }
  const manifest = values.get("manifest");
  const output = values.get("output");
  if (manifest === undefined || output === undefined) {
    throw new Error(
      "usage: write_soak_snapshot.ts --manifest <file> --output <file> [--data-dir <dir>]",
    );
  }
  return { manifest, output, dataDir: values.get("data-dir") };
}

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

const { manifest: manifestPath, output, dataDir } = parseArguments(process.argv.slice(2));
const manifest = SoakManifest.parse(JSON.parse(fs.readFileSync(manifestPath, "utf8")));
if (manifest.queries.length === 0) {
  throw new Error("the soak manifest needs at least one query");
}
const limit = manifest.limit ?? 8;
const temporary = dataDir ?? fs.mkdtempSync(path.join(os.tmpdir(), "ci-mcp-ts-soak-"));
const application = new Application(
  { data: temporary, cache: runtimePathsFromEnvironment().cache },
  {
    cwd: process.cwd(),
    settings: {
      ...indexSettingsFromEnvironment(),
      indexExecution: "in-process",
      brokerMode: "off",
    },
  },
);

try {
  const repositories = [];
  for (const repository of manifest.repositories) {
    const root = path.resolve(process.cwd(), repository.path);
    const name = repository.name ?? path.basename(root);
    const project = await application.initProject(root);
    await application.indexProject(project.id);
    const chunks = (await application.store.listChunks([project.id])).sort((left, right) =>
      left.chunk_id.localeCompare(right.chunk_id),
    );
    const queries = [];
    for (const query of manifest.queries) {
      const response = await application.searchCode(query, { projects: [project.id], limit });
      queries.push({
        query,
        hits: response.hits.map((hit) => ({ chunk_id: hit.chunk_id, score: hit.score })),
      });
    }
    process.stderr.write(`${name}: ${chunks.length} chunks, ${queries.length} queries\n`);
    repositories.push({
      name,
      path: root,
      project_id: project.id,
      chunk_count: chunks.length,
      chunks,
      queries,
    });
  }
  const snapshot = SoakSnapshot.parse({
    schema_version: 1,
    build: "typescript",
    revision: checkoutHead(repositoryRoot),
    model_id: application.embedder.modelId,
    repositories,
  });
  fs.mkdirSync(path.dirname(path.resolve(output)), { recursive: true });
  fs.writeFileSync(output, `${dumpJson(snapshot, { indent: 2 })}\n`, "utf8");
  process.stderr.write(`wrote ${output}\n`);
} finally {
  await application.store.close();
  if (dataDir === undefined) fs.rmSync(temporary, { recursive: true, force: true });
}
