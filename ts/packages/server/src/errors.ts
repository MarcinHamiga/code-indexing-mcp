/** Stable application errors exposed through CLI and MCP adapters. */

/**
 * The closed set of machine-readable codes.
 *
 * A union of string literals rather than an object of constants: the wire value
 * *is* the member name, so `"INDEX_BUSY"` at a call site is both the shorter
 * spelling and the one a reader can match against a daemon frame or an MCP
 * error without a lookup. `tsc` rejects a typo exactly as `StrEnum` did.
 */
export type ErrorCode =
  | "PROJECT_NOT_FOUND"
  | "AMBIGUOUS_PROJECT"
  | "PROJECT_ID_CONFLICT"
  | "CHUNK_NOT_FOUND"
  | "MODEL_UNAVAILABLE"
  | "INDEX_INCOMPATIBLE"
  | "INDEX_BUSY"
  | "INDEX_RESOURCE_LIMIT"
  | "INDEX_CANCELLED"
  | "EMBEDDING_WORKER_FAILED"
  | "BACKEND_UNAVAILABLE"
  | "DAEMON_UNAVAILABLE"
  | "PROTOCOL_ERROR"
  | "INVALID_CONFIGURATION"
  | "INVALID_FILTER"
  | "STALE_CURSOR"
  | "AMBIGUOUS_SYMBOL"
  | "UNSUPPORTED_LANGUAGE"
  | "REFERENCE_INDEX_UNAVAILABLE"
  | "INVALID_REFACTOR"
  | "INVALID_CURSOR"
  | "UNSUPPORTED_RUNTIME"
  | "OVERLAPPING_PROJECT";

/** Structured context carried alongside the code, never folded into the message. */
export type ErrorDetails = Readonly<Record<string, unknown>>;

/** An error with a stable machine-readable code. */
export class CodeIndexingError extends Error {
  readonly code: ErrorCode;
  readonly details: ErrorDetails;

  constructor(
    code: ErrorCode,
    message: string,
    details: ErrorDetails = {},
    options?: ErrorOptions,
  ) {
    // `cause` carries what Python's `raise ... from exc` carried: the parse or
    // filesystem failure underneath a stable code, kept for a log without
    // leaking into the client-facing rendering below.
    super(message, options);
    this.name = "CodeIndexingError";
    this.code = code;
    this.details = details;
  }

  /**
   * Render as `CODE: message`.
   *
   * Details are deliberately omitted: this string is embedded in `IndexIssue`
   * messages and in daemon frames that already carry `details` as a separate
   * field, where appending them would duplicate the payload.
   */
  override toString(): string {
    return `${this.code}: ${this.message}`;
  }

  /** Render code, message, and details as one line for an MCP tool error. */
  forClient(): string {
    const entries = Object.entries(this.details);
    if (entries.length === 0) {
      return this.toString();
    }
    const rendered = entries.map(([key, value]) => `${key}=${renderDetail(value)}`).join("; ");
    return `${this.toString()} [${rendered}]`;
  }
}

/** Narrow an unknown caught value to this package's error type. */
export function isCodeIndexingError(value: unknown): value is CodeIndexingError {
  return value instanceof CodeIndexingError;
}

/**
 * A detail value as it appears in the one-line client rendering.
 *
 * Scalars print bare, the way an f-string prints them. Anything composite goes
 * through JSON so a list of project ids stays legible as a list instead of
 * collapsing into `a,b` the way `String([...])` would.
 */
function renderDetail(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean" || value == null) {
    return String(value);
  }
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}
