import fs from "node:fs";
import path from "node:path";
import { Application } from "../src/application.ts";
import type { Embedder } from "../src/embedding.ts";
import type { ProjectInfo } from "../src/models.ts";
import { type CreatedServer, createServer } from "../src/server.ts";

class SetupEmbedder implements Embedder {
  readonly modelId = "test/tiny";
  readonly dimension = 4;
  embedPassages(texts: string[]): number[][] {
    return texts.map((text) => [1, 0, 0, text.length]);
  }
  embedQuery(text: string): number[] {
    return [1, 0, 0, text.length];
  }
}

/**
 * The shared fixture of the storage/history/scan tool tests: one project with
 * one indexed Python source, served over a fresh default-mode server.
 */
export async function prepare(temporary: string): Promise<{
  app: Application;
  project: ProjectInfo;
  root: string;
  server: CreatedServer;
}> {
  const root = path.join(temporary, "project");
  fs.mkdirSync(root);
  fs.writeFileSync(path.join(root, "pyproject.toml"), "[project]\nname = 'project'\n");
  fs.writeFileSync(path.join(root, "main.py"), "def answer():\n    return 42\n");
  const app = new Application(
    { data: path.join(temporary, "data"), cache: path.join(temporary, "cache") },
    { embedder: new SetupEmbedder(), cwd: temporary },
  );
  const project = await app.initProject(root);
  await app.indexProject(project.id);
  const server = createServer(app);
  server.listRoots = async () => [root];
  server.startCoordinator();
  return { app, project, root, server };
}
