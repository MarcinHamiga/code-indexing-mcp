import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function temporaryDirectory(): string {
  return fs.mkdtempSync(path.join(fs.realpathSync(os.tmpdir()), "ci-mcp-installer-"));
}

export function removeDirectory(directory: string): void {
  fs.rmSync(directory, { recursive: true, force: true });
}

export function writeFile(filePath: string, content: string): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
}
