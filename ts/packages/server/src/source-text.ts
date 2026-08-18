/**
 * One parsed file's text in the three coordinate systems the port has to hold
 * at once.
 *
 * Python parses `bytes` and every offset tree-sitter reports is a UTF-8 byte
 * offset. Those offsets are the durable contract: they are stored in the chunk
 * and reference tables, they are what `analyze_refactor` hands a caller to
 * splice a rename into, and they are compared against the file as it sits on
 * disk. The Node binding parses a JavaScript *string* and reports **UTF-16
 * code-unit indices** instead -- for `x = "éé𝄞"` the `def` that follows starts
 * at byte 15 and at code unit 11. Feeding a tree-sitter index straight into a
 * byte field would corrupt every offset in a file with one non-ASCII character
 * in it, silently and only for those files.
 *
 * So the extractor keeps working in bytes exactly as Python does, and every
 * node offset passes through {@link SourceText.byte} first. The common case --
 * a pure-ASCII source -- detects in one length comparison and maps by identity.
 *
 * Python's `str` indexes by *code point*, which is a third coordinate system:
 * `len(content)` and `content[a:b]` in the chunk splitter count code points,
 * while JavaScript's `.length` and `.slice` count UTF-16 units. They agree
 * everywhere except astral characters (emoji, rarer CJK), so the helpers here
 * carry the same fast-path/slow-path shape.
 */

const BOM_CODE_POINT = 0xfeff;

/** Where a decoded chunk of source sits in all three coordinate systems. */
export class SourceText {
  /** The UTF-8 bytes, BOM-stripped -- what Python calls `normalized_source`. */
  readonly bytes: Uint8Array;
  /** The same content as a JavaScript string, which is what tree-sitter parses. */
  readonly text: string;

  /**
   * Code-unit indices at which the running byte/code-unit delta increases, and
   * the delta after each. Empty for ASCII, which is why the common case costs
   * one allocation of nothing and a `<` per lookup.
   */
  readonly #breaks: Int32Array;
  readonly #deltas: Int32Array;

  private constructor(bytes: Uint8Array, text: string) {
    this.bytes = bytes;
    this.text = text;
    if (bytes.length === text.length) {
      // Pure ASCII (or, in principle, any string whose UTF-8 and UTF-16 lengths
      // coincide -- which for real text means ASCII): identity mapping.
      this.#breaks = new Int32Array(0);
      this.#deltas = new Int32Array(0);
      return;
    }
    const breaks: number[] = [];
    const deltas: number[] = [];
    let delta = 0;
    for (let index = 0; index < text.length; ) {
      const code = text.codePointAt(index) as number;
      const units = code > 0xffff ? 2 : 1;
      const width = utf8Width(code);
      index += units;
      if (width !== units) {
        delta += width - units;
        breaks.push(index);
        deltas.push(delta);
      }
    }
    this.#breaks = Int32Array.from(breaks);
    this.#deltas = Int32Array.from(deltas);
  }

  /**
   * Decode a source file the way `extract` does in Python:
   * `source.decode("utf-8-sig").encode("utf-8")` -- a byte-order mark is
   * dropped, and the result is re-encoded so offsets describe BOM-free bytes.
   *
   * Invalid UTF-8 raises here, as it does in Python; the indexer rejects such
   * files before extraction ever sees them.
   */
  static decode(source: Uint8Array): SourceText {
    // Preserve BOM code points during decoding and remove exactly one below,
    // matching `utf-8-sig` even when the source starts with two markers.
    const decoded = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(source);
    const text = decoded.codePointAt(0) === BOM_CODE_POINT ? decoded.slice(1) : decoded;
    return new SourceText(new TextEncoder().encode(text), text);
  }

  /** Build from a string, for callers that already hold decoded text. */
  static fromString(text: string): SourceText {
    return new SourceText(new TextEncoder().encode(text), text);
  }

  /** The UTF-8 byte offset for a tree-sitter UTF-16 code-unit index. */
  byte(codeUnitIndex: number): number {
    if (this.#breaks.length === 0) return codeUnitIndex;
    // The last break at or below the index carries the delta that applies.
    let low = 0;
    let high = this.#breaks.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if ((this.#breaks[middle] as number) <= codeUnitIndex) low = middle + 1;
      else high = middle;
    }
    return codeUnitIndex + (low === 0 ? 0 : (this.#deltas[low - 1] as number));
  }

  /** Decode a byte range back to text, as Python's `source[a:b].decode()` does. */
  slice(startByte: number, endByte: number): string {
    return new TextDecoder("utf-8", { ignoreBOM: true }).decode(
      this.bytes.subarray(startByte, endByte),
    );
  }
}

function utf8Width(codePoint: number): number {
  if (codePoint < 0x80) return 1;
  if (codePoint < 0x800) return 2;
  if (codePoint < 0x10000) return 3;
  return 4;
}

/**
 * Byte offsets of every newline in one file, for O(log n) line lookups.
 *
 * A per-chunk `count("\n")` over the prefix is O(file size), which made line
 * numbering O(chunks x file size); this costs one pass and one push per line.
 * Ported from `_LineIndex`.
 */
export class LineIndex {
  readonly #newlines: Int32Array;

  constructor(source: Uint8Array) {
    const newlines: number[] = [];
    for (let index = 0; index < source.length; index += 1) {
      if (source[index] === 0x0a) newlines.push(index);
    }
    this.#newlines = Int32Array.from(newlines);
  }

  /** The 1-based line number containing `byteOffset`. */
  lineAt(byteOffset: number): number {
    // bisect_left: the count of newlines strictly before the offset.
    let low = 0;
    let high = this.#newlines.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if ((this.#newlines[middle] as number) < byteOffset) low = middle + 1;
      else high = middle;
    }
    return low + 1;
  }
}

/** Python's `len(str)`: a count of code points, not of UTF-16 units. */
export function codePointLength(value: string): number {
  if (!hasSurrogates(value)) return value.length;
  let count = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    // A high surrogate followed by a low surrogate is one code point.
    if (code >= 0xd800 && code <= 0xdbff && index + 1 < value.length) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) index += 1;
    }
    count += 1;
  }
  return count;
}

/** Python's `value[start:end]` on a `str`: slicing by code point. */
export function sliceCodePoints(value: string, start: number, end: number): string {
  if (!hasSurrogates(value)) return value.slice(start, end);
  return Array.from(value).slice(start, end).join("");
}

function hasSurrogates(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdfff) return true;
  }
  return false;
}
