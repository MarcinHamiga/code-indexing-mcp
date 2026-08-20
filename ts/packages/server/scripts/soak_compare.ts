/** Compare two Phase 9 soak snapshots and apply the cutover gates.
 *
 * Usage, from the repository root:
 *
 *     bun ts/packages/server/scripts/soak_compare.ts \
 *       --python soak-python.json --typescript soak-ts.json [--output report.json]
 *
 * Exits nonzero when any gate fails; the report JSON is written to --output
 * when given and always printed to stdout.
 */

import fs from "node:fs";
import path from "node:path";
import { dumpJson } from "../src/jsonable.ts";
import {
  DEFAULT_SOAK_RANK_FLOOR,
  DEFAULT_SOAK_RECALL_FLOOR,
  SoakSnapshot,
  compareSoakSnapshots,
} from "../src/soak.ts";

function parseArguments(argv: string[]): {
  python: string;
  typescript: string;
  output: string | undefined;
  recallFloor: number;
  rankFloor: number;
} {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    if (flag === undefined || !flag.startsWith("--") || argv[index + 1] === undefined) {
      throw new Error(`expected --flag value pairs, got: ${argv.join(" ")}`);
    }
    values.set(flag.slice(2), argv[index + 1] as string);
  }
  const python = values.get("python");
  const typescript = values.get("typescript");
  if (python === undefined || typescript === undefined) {
    throw new Error(
      "usage: soak_compare.ts --python <file> --typescript <file> " +
        "[--recall-floor <n>] [--rank-floor <n>] [--output <file>]",
    );
  }
  return {
    python,
    typescript,
    output: values.get("output"),
    recallFloor:
      values.get("recall-floor") === undefined
        ? DEFAULT_SOAK_RECALL_FLOOR
        : Number(values.get("recall-floor")),
    rankFloor:
      values.get("rank-floor") === undefined
        ? DEFAULT_SOAK_RANK_FLOOR
        : Number(values.get("rank-floor")),
  };
}

const arguments_ = parseArguments(process.argv.slice(2));
const python = SoakSnapshot.parse(JSON.parse(fs.readFileSync(arguments_.python, "utf8")));
const typescript = SoakSnapshot.parse(JSON.parse(fs.readFileSync(arguments_.typescript, "utf8")));
const report = compareSoakSnapshots(python, typescript, {
  recallFloor: arguments_.recallFloor,
  rankFloor: arguments_.rankFloor,
});
if (arguments_.output !== undefined) {
  fs.mkdirSync(path.dirname(path.resolve(arguments_.output)), { recursive: true });
  fs.writeFileSync(arguments_.output, `${dumpJson(report, { indent: 2 })}\n`, "utf8");
}
process.stdout.write(`${dumpJson(report, { indent: 2 })}\n`);
if (!report.passed) process.exitCode = 1;
