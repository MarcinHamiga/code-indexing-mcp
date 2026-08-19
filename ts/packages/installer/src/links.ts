import fs from "node:fs";
import path from "node:path";

export function removePath(target: string): void {
  const info = fs.lstatSync(target);
  if (info.isSymbolicLink() || info.isFile()) {
    fs.unlinkSync(target);
    return;
  }
  fs.rmSync(target, { recursive: true, force: true });
}

export function backupPath(target: string): string {
  let candidate = `${target}.bak`;
  let counter = 2;
  while (true) {
    try {
      fs.lstatSync(candidate);
    } catch {
      return candidate;
    }
    candidate = `${target}.bak.${counter}`;
    counter += 1;
  }
}

export function linkDestination(link: string): string {
  return fs.realpathSync(link);
}

export function isUnder(child: string, directory: string): boolean {
  try {
    const relative = path.relative(fs.realpathSync(directory), fs.realpathSync(child));
    return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
  } catch {
    return false;
  }
}

export function replaceLink(
  source: string,
  target: string,
  options: { isDirectory: boolean; stale?: boolean },
): boolean {
  try {
    if (
      fs.lstatSync(target).isSymbolicLink() &&
      linkDestination(target) === fs.realpathSync(source)
    ) {
      return false;
    }
  } catch {
    // Target does not exist yet.
  }
  fs.mkdirSync(path.dirname(target), { recursive: true, mode: 0o700 });
  const staged = `${target}.incoming`;
  try {
    if (fs.lstatSync(staged).isSymbolicLink() || fs.existsSync(staged)) removePath(staged);
  } catch {
    // Staged name is free.
  }
  fs.symlinkSync(source, staged, options.isDirectory ? "dir" : "file");
  try {
    if (options.stale === true) {
      fs.unlinkSync(target);
    } else {
      try {
        if (fs.lstatSync(target).isSymbolicLink() || fs.existsSync(target)) {
          fs.renameSync(target, backupPath(target));
        }
      } catch {
        // Nothing to move aside.
      }
    }
    fs.renameSync(staged, target);
  } catch (error) {
    fs.rmSync(staged, { force: true });
    throw error;
  }
  return true;
}
