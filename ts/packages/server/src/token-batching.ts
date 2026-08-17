/**
 * Tokenizer-bounded window planning and microbatch packing.
 *
 * Character windows bound characters, not the token count that drives embedding
 * memory. Attention cost is quadratic in sequence length, so the same 4,096
 * characters cost wildly different amounts depending on how densely they
 * tokenize: ordinary source is ~984 tokens, a minified line is ~2,157, and
 * embedding the latter as one sequence adds ~1,172 MiB of resident memory
 * against ~266 MiB for the same characters split into token-bounded windows.
 *
 * Everything here is pure and tokenizer-agnostic so the policy is testable
 * without loading a model. The only tokenizer contact is `contentTokenOffsets`,
 * which reads an already-produced encoding.
 */

export const DEFAULT_MAX_TOKENS = 1024;
export const DEFAULT_OVERLAP_TOKENS = 64;
/**
 * Microbatch packing budget: itemCount * longestPaddedTokens. Padding is to the
 * longest member, so this bounds the padded matrix a batch materializes.
 */
export const DEFAULT_MAX_TOKEN_PRODUCT = 4096;
/**
 * The indexing ceiling `DEFAULT_MAX_TOKEN_PRODUCT` was measured against. The
 * product bounds the padded matrix a microbatch materializes, so it is a memory
 * budget expressed in tokens; leaving it fixed while the configured ceiling
 * moves either wastes the memory the operator granted or overruns the memory
 * they withheld.
 */
export const REFERENCE_MEMORY_BYTES = 2 * 1024 ** 3;
/**
 * Padding cost is quadratic in the widest member, and nothing has measured a
 * matrix beyond this multiple of the reference. A ceiling larger than that buys
 * no further width rather than extrapolating past the evidence.
 */
export const MAX_TOKEN_PRODUCT_MULTIPLE = 8;
/**
 * A candidate never exceeds the extractor's character ceiling, so a window
 * fan-out above this implies a pathological tokenization rather than dense code.
 * It is a tripwire, not a working limit: 4,096 characters cannot tokenize to
 * more than 4,096 tokens, which is five windows at the default budget.
 */
export const MAX_WINDOWS_PER_CANDIDATE = 16;

/** A token-bounded slice of one candidate, in candidate-relative characters. */
export interface TokenWindow {
  readonly startChar: number;
  readonly endChar: number;
  readonly tokenCount: number;
}

/** A character span, `[start, end)`, as a tokenizer reports it. */
export type TokenSpan = readonly [start: number, end: number];

/**
 * The slice of a tokenizer encoding this module reads.
 *
 * Structural rather than nominal, exactly as the Python duck-typing was: the
 * backend adapter in Phase 4 shapes whatever the bindings hand back into this,
 * and the tests supply a deterministic word-splitter instead.
 */
export interface TokenEncoding {
  readonly offsets: readonly TokenSpan[];
  readonly specialTokensMask?: readonly number[] | null | undefined;
}

/**
 * Return the padded-token budget a *memoryBytes* indexing ceiling supports.
 *
 * The floor is one longest window: a product below that would not admit even a
 * single max-length sequence as a batch of one, and would drop every shorter
 * segment to one item per batch on the way there.
 */
export function maxTokenProductFor(
  memoryBytes: number,
  { maxTokens = DEFAULT_MAX_TOKENS }: { maxTokens?: number } = {},
): number {
  const scaled = Math.floor(
    (DEFAULT_MAX_TOKEN_PRODUCT * Math.max(0, memoryBytes)) / REFERENCE_MEMORY_BYTES,
  );
  const ceiling = DEFAULT_MAX_TOKEN_PRODUCT * MAX_TOKEN_PRODUCT_MULTIPLE;
  return Math.max(maxTokens, Math.min(scaled, ceiling));
}

/**
 * Return character spans for real tokens, dropping `[CLS]`/`[SEP]`.
 *
 * Special tokens carry a `(0, 0)` span, so leaving them in would make every
 * window appear to start at the beginning of the text.
 */
export function contentTokenOffsets(encoding: TokenEncoding): TokenSpan[] {
  const offsets = encoding.offsets.map(([start, end]): TokenSpan => [start, end]);
  const mask = encoding.specialTokensMask;
  if (mask === undefined || mask === null) return offsets;
  if (mask.length !== offsets.length) {
    throw new Error(`special token mask covers ${mask.length} of ${offsets.length} offsets`);
  }
  return offsets.filter((_span, index) => !mask[index]);
}

/**
 * Split a candidate into windows of at most *maxTokens* tokens.
 *
 * Windows are contiguous in characters and overlap by *overlapTokens* tokens, so
 * no source is dropped between them. Boundaries depend only on the tokenization,
 * never on the memory budget or on how a batch was packed, so a retry at a
 * smaller microbatch size re-derives the identical windows.
 */
export function planTokenWindows(
  offsets: readonly TokenSpan[],
  {
    textLength,
    maxTokens = DEFAULT_MAX_TOKENS,
    overlapTokens = DEFAULT_OVERLAP_TOKENS,
    maxWindows = MAX_WINDOWS_PER_CANDIDATE,
  }: {
    textLength: number;
    maxTokens?: number;
    overlapTokens?: number;
    maxWindows?: number;
  },
): TokenWindow[] {
  if (maxTokens < 1) throw new Error("maxTokens must be at least 1");
  if (textLength < 0) throw new Error("textLength must not be negative");
  // Half the budget is the widest overlap that keeps the window count within
  // twice the minimum. Without it a budget squeezed by a wide prefix would leave
  // a one-token stride and fan a candidate out into hundreds of windows.
  const overlap = Math.min(Math.max(overlapTokens, 0), maxTokens - 1, Math.floor(maxTokens / 2));
  const stride = maxTokens - overlap;
  const total = offsets.length;
  if (total === 0) {
    // Whitespace-only or untokenizable content still needs one window so the
    // candidate keeps a vector rather than silently vanishing from the index.
    return textLength ? [{ startChar: 0, endChar: textLength, tokenCount: 0 }] : [];
  }
  if (total <= maxTokens) {
    return [{ startChar: 0, endChar: textLength, tokenCount: total }];
  }

  const windows: TokenWindow[] = [];
  let startToken = 0;
  while (startToken < total) {
    const endToken = Math.min(startToken + maxTokens, total);
    // Character bounds run from this token's start to the *next* token's start,
    // so inter-token whitespace stays attached to the earlier window and the
    // concatenated windows cover the candidate exactly.
    const startChar = startToken === 0 ? 0 : (offsets[startToken]?.[0] ?? 0);
    const endChar = endToken === total ? textLength : (offsets[endToken]?.[0] ?? textLength);
    windows.push({ startChar, endChar, tokenCount: endToken - startToken });
    if (windows.length > maxWindows) {
      throw new Error(
        `Token planning exceeded ${maxWindows} windows for a ` +
          `${textLength}-character candidate (${total} tokens)`,
      );
    }
    if (endToken === total) break;
    startToken += stride;
  }
  return windows;
}

/** One `(prefix, content)` candidate awaiting a window plan. */
export interface WindowCandidate {
  readonly prefix: string;
  readonly content: string;
}

/**
 * Plan windows for `(prefix, content)` candidates.
 *
 * The prefix is the context header repeated on every window of a candidate, so
 * it is charged against the budget once and the content windows are sized with
 * what is left.
 */
export function planCandidateWindows(
  encode: (text: string) => TokenEncoding,
  candidates: readonly WindowCandidate[],
  {
    maxTokens = DEFAULT_MAX_TOKENS,
    overlapTokens = DEFAULT_OVERLAP_TOKENS,
    maxWindows = MAX_WINDOWS_PER_CANDIDATE,
  }: { maxTokens?: number; overlapTokens?: number; maxWindows?: number } = {},
): TokenWindow[][] {
  const plans: TokenWindow[][] = [];
  const prefixTokens = new Map<string, number>();
  for (const { prefix, content } of candidates) {
    let charged = prefixTokens.get(prefix);
    if (charged === undefined) {
      charged = prefix ? contentTokenOffsets(encode(prefix)).length : 0;
      prefixTokens.set(prefix, charged);
    }
    // Keep at least one token of forward progress per window even when a
    // pathological prefix would otherwise consume the whole budget.
    const budget = Math.max(overlapTokens + 1, maxTokens - charged);
    plans.push(
      planTokenWindows(contentTokenOffsets(encode(content)), {
        textLength: content.length,
        maxTokens: budget,
        overlapTokens,
        maxWindows,
      }),
    );
  }
  return plans;
}

/**
 * Bucket segment indices by length, then stay within both packing limits.
 *
 * A batch pads to its longest member, so `itemCount * longest` -- not the sum --
 * is what the model materializes. A single segment always forms a batch even
 * when it exceeds the product on its own; there is nothing smaller to fall back
 * to. Power-of-two buckets keep similarly sized segments together without making
 * exact token counts part of the ordering contract.
 */
export function planMicrobatches(
  tokenCounts: readonly number[],
  {
    maxItems = 1,
    maxTokenProduct = DEFAULT_MAX_TOKEN_PRODUCT,
  }: { maxItems?: number; maxTokenProduct?: number } = {},
): number[][] {
  if (maxItems < 1) throw new Error("maxItems must be at least 1");
  const bucket = (count: number): number =>
    count > maxTokenProduct ? -1 : bitLength(Math.max(0, count));
  const ordered = tokenCounts
    .map((count, index) => ({ count, index }))
    .sort((left, right) => bucket(left.count) - bucket(right.count) || left.index - right.index);

  const batches: number[][] = [];
  let current: number[] = [];
  let longest = 0;
  for (const { count, index } of ordered) {
    let widened = Math.max(longest, count);
    if (current.length > 0) {
      if (current.length + 1 > maxItems || (current.length + 1) * widened > maxTokenProduct) {
        batches.push(current);
        current = [];
        widened = count;
      }
    }
    current.push(index);
    longest = widened;
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

/** `int.bit_length()`: how many bits the magnitude of a non-negative int needs. */
function bitLength(value: number): number {
  return value === 0 ? 0 : 32 - Math.clz32(value);
}
