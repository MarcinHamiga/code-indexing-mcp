/** JSON encoding for daemon frames and CLI output. */

export function jsonable(value: unknown): unknown {
  if (typeof value === "bigint") {
    return Number.isSafeInteger(Number(value)) ? Number(value) : value.toString();
  }
  if (value instanceof Set) {
    return [...value].map(jsonable).sort(compareJson);
  }
  if (Array.isArray(value)) return value.map(jsonable);
  if (value instanceof Map) {
    return Object.fromEntries([...value.entries()].map(([key, item]) => [key, jsonable(item)]));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, jsonable(item)]),
    );
  }
  return value;
}

function compareJson(left: unknown, right: unknown): number {
  const encodedLeft = JSON.stringify(left) ?? "";
  const encodedRight = JSON.stringify(right) ?? "";
  return encodedLeft < encodedRight ? -1 : encodedLeft > encodedRight ? 1 : 0;
}

export function dumpJson(value: unknown, { indent = 0 }: { indent?: number } = {}): string {
  return JSON.stringify(
    sortKeys(jsonable(value)),
    (_key, item: unknown) => (typeof item === "bigint" ? Number(item) : item),
    indent === 0 ? undefined : indent,
  );
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
        .map(([key, item]) => [key, sortKeys(item)]),
    );
  }
  return value;
}
