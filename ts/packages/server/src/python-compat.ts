/** Python-compatible ordering and JSON at persisted or externally visible boundaries. */

import path from "node:path";

/** Python's lexicographic `str` comparison, which compares Unicode code points. */
export function comparePythonStrings(left: string, right: string): number {
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const leftCodePoint = left.codePointAt(leftIndex) as number;
    const rightCodePoint = right.codePointAt(rightIndex) as number;
    if (leftCodePoint !== rightCodePoint) return leftCodePoint < rightCodePoint ? -1 : 1;
    leftIndex += leftCodePoint > 0xffff ? 2 : 1;
    rightIndex += rightCodePoint > 0xffff ? 2 : 1;
  }
  if (leftIndex === left.length && rightIndex === right.length) return 0;
  return leftIndex === left.length ? -1 : 1;
}

/** Python's `Path` ordering: platform-normalized path components, not whole strings. */
export function comparePythonPaths(left: string, right: string): number {
  const normalizePart =
    process.platform === "win32"
      ? (part: string): string => part.toLowerCase()
      : (part: string) => part;
  const leftParts = path.normalize(left).split(path.sep).map(normalizePart);
  const rightParts = path.normalize(right).split(path.sep).map(normalizePart);
  const length = Math.min(leftParts.length, rightParts.length);
  for (let index = 0; index < length; index += 1) {
    const compared = comparePythonStrings(leftParts[index] as string, rightParts[index] as string);
    if (compared !== 0) return compared;
  }
  if (leftParts.length === rightParts.length) return 0;
  return leftParts.length < rightParts.length ? -1 : 1;
}

/**
 * `json.dumps(value, separators=(",", ":"), sort_keys=True)`.
 *
 * Python's default `ensure_ascii=True` is part of the cursor and on-disk row
 * contract, so non-ASCII UTF-16 code units are escaped individually. Escaping
 * surrogate pairs separately also reproduces Python's representation of
 * non-BMP code points.
 */
export function pythonJsonDumps(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return quotePythonJsonString(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (Number.isNaN(value)) return "NaN";
    if (value === Number.POSITIVE_INFINITY) return "Infinity";
    if (value === Number.NEGATIVE_INFINITY) return "-Infinity";
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(pythonJsonDumps).join(",")}]`;
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, item]) => item !== undefined)
      .sort(([left], [right]) => comparePythonStrings(left, right));
    return `{${entries
      .map(([key, item]) => `${quotePythonJsonString(key)}:${pythonJsonDumps(item)}`)
      .join(",")}}`;
  }
  throw new TypeError(`value of type ${typeof value} is not JSON serializable`);
}

function quotePythonJsonString(value: string): string {
  return JSON.stringify(value).replace(/[\u0080-\uffff]/g, (character) => {
    const codeUnit = character.charCodeAt(0).toString(16).padStart(4, "0");
    return `\\u${codeUnit}`;
  });
}
