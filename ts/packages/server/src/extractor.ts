/**
 * Tree-sitter based symbol and module chunk extraction.
 *
 * A transliteration of `extractor.py`, held to its output byte for byte by
 * `tests/fixtures/extractor_snapshot.json` -- the same committed fingerprint
 * the Python suite gates on. Chunk identity is a digest of kind, qualified
 * symbol, byte offsets, and part index, so any drift here silently invalidates
 * every stored chunk id and breaks incremental indexing; the snapshot is what
 * makes that a test failure instead of a support ticket.
 *
 * Two things the Python original gets for free need care here. Tree-sitter's
 * Node binding reports UTF-16 code-unit indices where the Python binding
 * reports UTF-8 byte offsets, so every node offset passes through
 * {@link SourceText.byte} (see `source-text.ts`); and Python's `str` indexes by
 * code point where JavaScript's indexes by UTF-16 unit, so the chunk splitter
 * uses {@link codePointLength}/{@link sliceCodePoints} rather than `.length`
 * and `.slice`.
 */

// Imported as a namespace rather than by name so the suite can count how often
// a query file is read, which is how it proves compilation happens once per
// language rather than once per extracted file.
import fs from "node:fs";
import { join } from "node:path";
import Parser from "tree-sitter";
import { type Grammar, grammarFor } from "./grammars.ts";
import type {
  CallShape,
  ExtractedChunk,
  ExtractedDeclarationShape,
  ExtractedReference,
  ExtractionResult,
  ParameterKind,
  ParameterShape,
  ReferenceKind,
} from "./models.ts";
import { codePointLength, LineIndex, SourceText, sliceCodePoints } from "./source-text.ts";

type SyntaxNode = Parser.SyntaxNode;

const CAMEL_BOUNDARY_1 = /([a-z0-9])([A-Z])/g;
const CAMEL_BOUNDARY_2 = /([A-Z]+)([A-Z][a-z])/g;
const NON_WORD = /[^A-Za-z0-9]+/g;

const CONTAINER_KINDS: ReadonlySet<string> = new Set([
  "annotation",
  "array",
  "class",
  "constant",
  "enum",
  "interface",
  "object",
  "record",
  "struct",
]);
const CALLABLE_KINDS: ReadonlySet<string> = new Set(["constructor", "function", "method"]);
const QUOTE_CHARACTERS = ["'", '"'];

/** The languages whose reference queries exist, so structural rows are extracted. */
export const STRUCTURAL_LANGUAGES: ReadonlySet<string> = new Set([
  "python",
  "javascript",
  "typescript",
  "tsx",
]);

const QUERY_DIRECTORY = join(import.meta.dir, "queries");
const REFERENCE_QUERY_DIRECTORY = join(import.meta.dir, "reference-queries");

/**
 * Add one reference row.
 *
 * The Python original passes this as a closure with keyword arguments; an
 * options object is the same shape and keeps the call sites readable.
 */
type ReferenceAdder = (
  kind: ReferenceKind,
  node: SyntaxNode,
  options: {
    writtenName?: string | null;
    targetName: string;
    modulePath?: string | null;
    importedName?: string | null;
    alias?: string | null;
    receiverText?: string | null;
    callShape?: CallShape | null;
  },
) => void;

/**
 * The symbol text of a `@name` capture, without surrounding quotes.
 *
 * Most grammars name a definition with an identifier token and this is the
 * node's own text. A few have no node for the inside of a quoted name --
 * Godot's resource format hands back `"Player"` including the quotes, and a
 * quoted YAML key does the same -- which would otherwise index the quotes as
 * part of the symbol. Only a matched leading/trailing pair is stripped, so an
 * identifier that merely contains a quote is left alone.
 */
function captureName(node: SyntaxNode): string {
  const name = node.text;
  const first = name[0] ?? "";
  if (name.length >= 2 && first === name[name.length - 1] && QUOTE_CHARACTERS.includes(first)) {
    return name.slice(1, -1);
  }
  return name;
}

/**
 * A node's first named child that is not "extra" trivia (a comment).
 *
 * `namedChild(0)` picks whatever sits first in source order, so a comment
 * placed before the real content (`require(/* c *\/ './mod')`,
 * `import * as /* c *\/ ns from 'mod'`) is silently mistaken for it instead of
 * being skipped, the same way a comment was mistaken for a positional call
 * argument (finding 7).
 */
function firstNamedChild(node: SyntaxNode): SyntaxNode | null {
  for (const child of node.namedChildren) {
    if (!child.isExtra) return child;
  }
  return null;
}

export function normalizeIdentifier(value: string): string {
  let result = value.replace(CAMEL_BOUNDARY_2, "$1 $2");
  result = result.replace(CAMEL_BOUNDARY_1, "$1 $2");
  return result.replace(NON_WORD, " ").toLowerCase().split(/\s+/).filter(Boolean).join(" ");
}

interface Definition {
  readonly node: SyntaxNode;
  readonly kind: string;
  readonly name: string;
}

/**
 * Per-file lookups the definition walk needs, built once instead of per definition.
 *
 * `hasDefinitionAncestor`, `symbolContext`, and `contentRange` each rebuilt a
 * whole-file map or set on every call, making extraction quadratic in
 * definition count: a 699 KB generated file with 16,384 definitions spent 31 s
 * here against 8 ms of parsing.
 */
class DefinitionIndex {
  readonly definitions: readonly Definition[];
  readonly byNodeId: Map<number, Definition>;
  /** `definitions` is sorted by (startByte, -endByte), so this is ascending. */
  readonly starts: Int32Array;

  constructor(definitions: readonly Definition[], startsByNode: Map<number, number>) {
    this.definitions = definitions;
    this.byNodeId = new Map(definitions.map((definition) => [definition.node.id, definition]));
    this.starts = Int32Array.from(
      definitions.map((definition) => startsByNode.get(definition.node.id) as number),
    );
  }
}

export interface ExtractorOptions {
  maxChars?: number;
  maxLines?: number;
  overlapLines?: number;
}

export class TreeSitterExtractor {
  readonly maxChars: number;
  readonly maxLines: number;
  readonly overlapLines: number;

  readonly #queries = new Map<string, Parser.Query>();
  readonly #structuralQueries = new Map<string, Parser.Query>();

  constructor(options: ExtractorOptions = {}) {
    this.maxChars = options.maxChars ?? 4_096;
    this.maxLines = options.maxLines ?? 200;
    this.overlapLines = Math.min(options.overlapLines ?? 20, Math.max(0, this.maxLines - 1));
  }

  /**
   * The compiled chunk query for one language, compiled once per process.
   *
   * The `.scm` files are package data and never change at runtime, but reading
   * and recompiling one per extracted file measured at 44% of extraction time
   * across a 35-file pass. Python needs a lock around this because the daemon
   * serves each client on its own thread; a single-threaded runtime does not.
   */
  #query(language: string): Parser.Query {
    const cached = this.#queries.get(language);
    if (cached !== undefined) return cached;
    const compiled = compileQuery(language, join(QUERY_DIRECTORY, `${language}.scm`));
    this.#queries.set(language, compiled);
    return compiled;
  }

  /** The cached structural query for one supported source grammar. */
  #structuralQuery(language: string): Parser.Query {
    const cached = this.#structuralQueries.get(language);
    if (cached !== undefined) return cached;
    const compiled = compileQuery(language, join(REFERENCE_QUERY_DIRECTORY, `${language}.scm`));
    this.#structuralQueries.set(language, compiled);
    return compiled;
  }

  extract(path: string, language: string, source: Uint8Array): ExtractionResult {
    const grammar = requireGrammar(language);
    const text = SourceText.decode(source);
    const parser = new Parser();
    parser.setLanguage(grammar);
    const tree = parser.parse(text.text);
    const root = tree.rootNode;
    const definitions = this.#definitions(language, root);
    const index = new DefinitionIndex(
      definitions,
      new Map(definitions.map((item) => [item.node.id, text.byte(item.node.startIndex)])),
    );
    const lineIndex = new LineIndex(text.bytes);
    let references: ExtractedReference[] = [];
    let declarations: ExtractedDeclarationShape[] = [];
    let referenceExtractionNs = 0;
    if (STRUCTURAL_LANGUAGES.has(language)) {
      const started = process.hrtime.bigint();
      const records = this.#structuralRecords(language, root, text, index, lineIndex);
      references = records.references;
      declarations = records.declarations;
      referenceExtractionNs = Number(process.hrtime.bigint() - started);
    }
    const chunks: ExtractedChunk[] = [];
    const covered: Array<[number, number]> = [];

    for (const definition of definitions) {
      const outer = outerNode(definition.node);
      if (!hasDefinitionAncestor(definition.node, index)) {
        covered.push([text.byte(outer.startIndex), text.byte(outer.endIndex)]);
      }
      const { kind, parent, qualified } = symbolContext(definition, index);
      const [start, end] = contentRange(outer, definition.node, kind, index, text);
      chunks.push(
        ...this.#chunksForRange({
          path,
          language,
          kind,
          symbol: definition.name,
          qualified,
          parent,
          source: text,
          start,
          end,
          lineIndex,
        }),
      );
    }

    chunks.push(...this.#moduleChunks(path, language, text, covered, lineIndex));
    chunks.sort(
      (left, right) =>
        left.start_byte - right.start_byte ||
        left.end_byte - right.end_byte ||
        compareStrings(left.kind, right.kind),
    );
    return {
      chunks,
      references,
      declarations,
      has_errors: root.hasError,
      reference_extraction_ns: referenceExtractionNs,
    };
  }

  /** Extract syntax facts using the already parsed tree and definition index. */
  #structuralRecords(
    language: string,
    root: SyntaxNode,
    source: SourceText,
    index: DefinitionIndex,
    lineIndex: LineIndex,
  ): { references: ExtractedReference[]; declarations: ExtractedDeclarationShape[] } {
    const matches = groupedMatches(this.#structuralQuery(language), root);
    const parameterNodes = new Map<number, SyntaxNode>();
    for (const captures of matches) {
      for (const parameterNode of captures.get("declaration.parameters") ?? []) {
        let owner: SyntaxNode | null = parameterNode.parent;
        while (owner !== null && !index.byNodeId.has(owner.id)) owner = owner.parent;
        if (owner !== null) parameterNodes.set(owner.id, parameterNode);
      }
    }
    const declarations = declarationShapes(language, index, lineIndex, parameterNodes, source);
    const declarationByNode = new Map<number, ExtractedDeclarationShape>();
    index.definitions.forEach((definition, position) => {
      declarationByNode.set(
        definition.node.id,
        declarations[position] as ExtractedDeclarationShape,
      );
    });
    const references: ExtractedReference[] = [];
    const seen = new Set<string>();

    const add: ReferenceAdder = (kind, node, options) => {
      const startByte = source.byte(node.startIndex);
      const endByte = source.byte(node.endIndex);
      const key = `${kind}\0${startByte}\0${endByte}`;
      if (seen.has(key)) return;
      seen.add(key);
      const name = options.writtenName || captureName(node);
      references.push({
        kind,
        written_name: name,
        target_name: options.targetName,
        source_qualified_symbol: enclosingSymbol(node, declarationByNode),
        module_path: options.modulePath ?? null,
        imported_name: options.importedName ?? null,
        alias: options.alias ?? null,
        receiver_text: options.receiverText ?? null,
        start_byte: startByte,
        end_byte: endByte,
        start_line: lineIndex.lineAt(startByte),
        end_line: lineIndex.lineAt(Math.max(startByte, endByte - 1)),
        call_shape: options.callShape ?? null,
      });
    };

    for (const captures of matches) {
      for (const [capture, nodes] of captures) {
        if (!capture.startsWith("reference.")) continue;
        for (const node of nodes) {
          if (capture === "reference.identifier") identifierRecord(language, node, add);
          else if (language === "python") pythonRecords(node, source, add);
          else javascriptRecords(node, source, add);
        }
      }
    }
    references.sort(
      (left, right) =>
        left.start_byte - right.start_byte ||
        left.end_byte - right.end_byte ||
        compareStrings(left.kind, right.kind),
    );
    return { references, declarations };
  }

  #definitions(language: string, root: SyntaxNode): Definition[] {
    const matches = groupedMatches(this.#query(language), root);
    // Keyed by (startIndex, endIndex, kind) as Python keys by the byte range;
    // the mapping between the two is monotonic, so identity is preserved.
    const found = new Map<string, Definition>();
    for (const captures of matches) {
      const nameNodes = captures.get("name");
      if (nameNodes === undefined || nameNodes.length === 0) continue;
      const name = captureName(nameNodes[0] as SyntaxNode);
      for (const [capture, nodes] of captures) {
        if (!capture.startsWith("definition.")) continue;
        const kind = capture.slice("definition.".length);
        const node = nodes[0] as SyntaxNode;
        found.set(`${node.startIndex}\0${node.endIndex}\0${kind}`, { node, kind, name });
      }
    }
    return [...found.values()].sort(
      (left, right) =>
        left.node.startIndex - right.node.startIndex || right.node.endIndex - left.node.endIndex,
    );
  }

  #chunksForRange(options: {
    path: string;
    language: string;
    kind: string;
    symbol: string | null;
    qualified: string | null;
    parent: string | null;
    source: SourceText;
    start: number;
    end: number;
    lineIndex: LineIndex;
  }): ExtractedChunk[] {
    const { path, language, kind, symbol, qualified, parent, source, start, end, lineIndex } =
      options;
    const content = rstrip(source.slice(start, end));
    const startLine = lineIndex.lineAt(start);
    if (codePointLength(content) <= this.maxChars && countNewlines(content) + 1 <= this.maxLines) {
      return [
        makeChunk(
          path,
          language,
          kind,
          symbol,
          qualified,
          parent,
          start,
          end,
          startLine,
          content,
          0,
        ),
      ];
    }

    const lines = splitLinesKeepEnds(content);
    // Cumulative UTF-8 byte offset of each line, so chunk offsets stay linear
    // to compute instead of re-encoding every preceding line per part.
    const lineOffsets = [0];
    const encoder = new TextEncoder();
    for (const line of lines) {
      lineOffsets.push(
        (lineOffsets[lineOffsets.length - 1] as number) + encoder.encode(line).length,
      );
    }
    // Python compares `len(line)` -- a code-point count -- against max_chars.
    const lineLengths = lines.map(codePointLength);
    const partKind = kind === "module" ? "module" : `${kind}_part`;
    const chunks: ExtractedChunk[] = [];
    let cursor = 0;
    let part = 0;
    while (cursor < lines.length) {
      const partLines: string[] = [];
      let charCount = 0;
      let endCursor = cursor;
      while (endCursor < lines.length && partLines.length < this.maxLines) {
        const line = lines[endCursor] as string;
        if (charCount + (lineLengths[endCursor] as number) > this.maxChars) {
          // With no lines accumulated this means the line is oversized on its
          // own, and the fragment path below handles it.
          break;
        }
        partLines.push(line);
        charCount += lineLengths[endCursor] as number;
        endCursor += 1;
      }
      if (partLines.length === 0) {
        // This single line is wider than maxChars on its own. Split just that
        // line into bounded fragments; the surrounding lines keep using the
        // ordinary line windows below.
        for (const [offset, fragment] of this.#lineFragments(lines[cursor] as string)) {
          const fragmentContent = rstrip(fragment);
          if (!fragmentContent) continue;
          const byteStart = start + (lineOffsets[cursor] as number) + offset;
          chunks.push(
            makeChunk(
              path,
              language,
              partKind,
              symbol,
              qualified,
              parent,
              byteStart,
              byteStart + encoder.encode(fragment).length,
              startLine + cursor,
              fragmentContent,
              part,
            ),
          );
          part += 1;
        }
        cursor += 1;
        continue;
      }
      const joined = partLines.join("");
      const partContent = rstrip(joined);
      if (partContent) {
        const byteStart = start + (lineOffsets[cursor] as number);
        const byteEnd = byteStart + encoder.encode(joined).length;
        chunks.push(
          makeChunk(
            path,
            language,
            partKind,
            symbol,
            qualified,
            parent,
            byteStart,
            byteEnd,
            startLine + cursor,
            partContent,
            part,
          ),
        );
        part += 1;
      }
      if (endCursor >= lines.length) break;
      if ((lineLengths[endCursor] as number) > this.maxChars) {
        // The window stopped because the next line is oversized, not because it
        // filled up. Overlapping back into it would emit a near-duplicate
        // window per line until the cursor crawls there.
        cursor = endCursor;
      } else {
        cursor = Math.max(cursor + 1, endCursor - this.overlapLines);
      }
    }
    return chunks;
  }

  /** `(byte offset within line, fragment)` pairs for an oversized line. */
  *#lineFragments(line: string): Generator<[number, string]> {
    const encoder = new TextEncoder();
    const length = codePointLength(line);
    let cursor = 0;
    let byteOffset = 0;
    while (cursor < length) {
      const fragment = sliceCodePoints(line, cursor, cursor + this.maxChars);
      yield [byteOffset, fragment];
      cursor += codePointLength(fragment);
      byteOffset += encoder.encode(fragment).length;
    }
  }

  #moduleChunks(
    path: string,
    language: string,
    source: SourceText,
    covered: Array<[number, number]>,
    lineIndex: LineIndex,
  ): ExtractedChunk[] {
    const chunks: ExtractedChunk[] = [];
    let cursor = 0;
    const ordered = [...covered].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
    for (const [start, end] of ordered) {
      if (cursor < start && hasNonWhitespace(source.bytes, cursor, start)) {
        chunks.push(
          ...this.#chunksForRange({
            path,
            language,
            kind: "module",
            symbol: null,
            qualified: null,
            parent: null,
            source,
            start: cursor,
            end: start,
            lineIndex,
          }),
        );
      }
      cursor = Math.max(cursor, end);
    }
    if (
      cursor < source.bytes.length &&
      hasNonWhitespace(source.bytes, cursor, source.bytes.length)
    ) {
      chunks.push(
        ...this.#chunksForRange({
          path,
          language,
          kind: "module",
          symbol: null,
          qualified: null,
          parent: null,
          source,
          start: cursor,
          end: source.bytes.length,
          lineIndex,
        }),
      );
    }
    return chunks;
  }
}

function requireGrammar(language: string): Grammar {
  const grammar = grammarFor(language);
  if (grammar === undefined) {
    throw new Error(`no tree-sitter grammar for ${language} on ${process.platform}`);
  }
  return grammar;
}

function compileQuery(language: string, file: string): Parser.Query {
  const grammar = requireGrammar(language);
  return new Parser.Query(grammar, fs.readFileSync(file, "utf8"));
}

/** Where the packaged `.scm` query files live, for the suite that checks coverage. */
export const QUERY_DIRECTORIES = {
  chunks: QUERY_DIRECTORY,
  references: REFERENCE_QUERY_DIRECTORY,
} as const;

/**
 * Run a query and hand back one `capture name -> nodes` map per match.
 *
 * The Python binding's `QueryCursor.matches` already returns that shape; the
 * Node binding returns a flat capture list per match, so the grouping happens
 * here. Insertion order is preserved, which is what makes the first-wins
 * dedupe in `add` and in `#definitions` behave identically.
 */
function groupedMatches(query: Parser.Query, root: SyntaxNode): Array<Map<string, SyntaxNode[]>> {
  return query.matches(root).map((match) => {
    const captures = new Map<string, SyntaxNode[]>();
    for (const capture of match.captures) {
      const existing = captures.get(capture.name);
      if (existing === undefined) captures.set(capture.name, [capture.node]);
      else existing.push(capture.node);
    }
    return captures;
  });
}

/** Record identifier values while excluding bindings and richer structural uses. */
function identifierRecord(language: string, node: SyntaxNode, addReference: ReferenceAdder): void {
  const contains = (outer: SyntaxNode | null): boolean =>
    outer !== null && outer.startIndex <= node.startIndex && node.endIndex <= outer.endIndex;

  let current = node;
  let parent = current.parent;
  while (parent !== null) {
    if (
      parent.type === "import_statement" ||
      parent.type === "import_from_statement" ||
      parent.type === "export_clause" ||
      parent.type === "namespace_export" ||
      parent.type === "decorator" ||
      parent.type === "type" ||
      parent.type === "type_annotation" ||
      parent.type === "generic_type" ||
      parent.type === "class_heritage" ||
      parent.type === "extends_type_clause"
    ) {
      return;
    }
    if (
      parent.type === "parameters" ||
      parent.type === "formal_parameters" ||
      parent.type === "lambda_parameters"
    ) {
      let parameter: SyntaxNode = node;
      let reachedDefault = false;
      while (parameter.parent !== null && parameter.parent.id !== parent.id) {
        parameter = parameter.parent;
        // TS's `required_parameter`/`optional_parameter` wrapper exposes a
        // default under a `value` field; bare JS/TS `assignment_pattern`
        // (untyped `a = LIMIT`) exposes it under `right` instead (mirrors the
        // E8 note on `parameterShapes` below). Checking only `value` drops
        // every identifier read inside a plain JS default.
        if (
          contains(parameter.childForFieldName("value")) ||
          contains(parameter.childForFieldName("right"))
        ) {
          reachedDefault = true;
          break;
        }
      }
      if (!reachedDefault) return;
    }
    let excludedFields: readonly string[] = [];
    if (
      parent.type === "function_definition" ||
      parent.type === "function_expression" ||
      parent.type === "generator_function_declaration" ||
      parent.type === "generator_function" ||
      parent.type === "class_definition" ||
      parent.type === "function_declaration" ||
      parent.type === "class_declaration" ||
      parent.type === "method_definition" ||
      parent.type === "variable_declarator" ||
      // TS declaration names surface as type_identifier now that the identifier
      // fallback covers it too (Task 2.2); their own name is a binding, not a
      // reference.
      parent.type === "interface_declaration" ||
      parent.type === "type_alias_declaration" ||
      parent.type === "type_parameter" ||
      // JSX element names get their own `type_use` component-reference row
      // (E14); everything else inside the element (attribute values, children)
      // stays a plain identifier reference.
      parent.type === "jsx_opening_element" ||
      parent.type === "jsx_self_closing_element" ||
      parent.type === "jsx_closing_element"
    ) {
      excludedFields = ["name"];
    } else if (
      parent.type === "assignment" ||
      parent.type === "assignment_expression" ||
      parent.type === "augmented_assignment" ||
      parent.type === "named_expression" ||
      parent.type === "for_statement" ||
      parent.type === "for_in_clause" ||
      // JS `for (const item of items)` -- the loop binding is not a reference
      // to an existing `item` (E11).
      parent.type === "for_in_statement"
    ) {
      excludedFields = ["left", "name"];
    } else if (parent.type === "arrow_function" || parent.type === "lambda") {
      // `parameter` covers a parenless single-identifier arrow param (`x => x`),
      // which has no `formal_parameters` wrapper of its own to be caught by the
      // block above. `parameters` is deliberately NOT excluded here: it names
      // the `formal_parameters`/`lambda_parameters` node wrapping every other
      // case, which the parameter-defaults block above has already walked and
      // decided correctly (including whether an identifier inside a default
      // value is a real read); blanket-excluding the whole field here would
      // undo that decision for every arrow-function/lambda default
      // (`(a = LIMIT) => a`, `lambda a=LIMIT: a`).
      excludedFields = ["parameter"];
    } else if (parent.type === "attribute" || parent.type === "member_expression") {
      excludedFields = ["attribute", "property"];
    } else if (
      parent.type === "call" ||
      parent.type === "call_expression" ||
      parent.type === "new_expression"
    ) {
      excludedFields = ["function", "constructor"];
    } else if (parent.type === "keyword_argument") {
      excludedFields = ["name"];
    } else if (parent.type === "as_pattern" || parent.type === "catch_clause") {
      excludedFields = ["alias", "parameter"];
    } else if (parent.type === "export_statement") {
      excludedFields = ["value"];
    } else if (
      language !== "python" &&
      (parent.type === "pair" || parent.type === "pair_pattern")
    ) {
      excludedFields = ["key"];
    }
    if (excludedFields.some((field) => contains(parent?.childForFieldName(field) ?? null))) return;
    current = parent;
    parent = current.parent;
  }

  const name = captureName(node);
  addReference("read", node, { targetName: name, writtenName: name });
}

function enclosingSymbol(
  node: SyntaxNode,
  declarations: Map<number, ExtractedDeclarationShape>,
): string | null {
  let current: SyntaxNode | null = node;
  while (current !== null) {
    const declaration = declarations.get(current.id);
    if (declaration !== undefined) return declaration.qualified_symbol;
    if (current.type === "decorated_definition") {
      const child = current.childForFieldName("definition");
      const nested = child === null ? undefined : declarations.get(child.id);
      if (nested !== undefined) return nested.qualified_symbol;
    }
    if (current.type === "decorator") {
      // JS/TS: a method/field decorator is a preceding sibling of the
      // `method_definition`/`public_field_definition` it decorates, not its
      // parent (unlike Python's `decorated_definition` wrapper, and unlike a
      // TS/JS *class* decorator, which the grammar does nest inside
      // `class_declaration`) -- attribute it to that sibling.
      const sibling = current.nextSibling;
      const nested = sibling === null ? undefined : declarations.get(sibling.id);
      if (nested !== undefined) return nested.qualified_symbol;
    }
    current = current.parent;
  }
  return null;
}

function declarationShapes(
  language: string,
  index: DefinitionIndex,
  lineIndex: LineIndex,
  parameterNodes: Map<number, SyntaxNode>,
  source: SourceText,
): ExtractedDeclarationShape[] {
  return index.definitions.map((definition) => {
    const { kind, qualified } = symbolContext(definition, index);
    const startByte = source.byte(definition.node.startIndex);
    const endByte = source.byte(definition.node.endIndex);
    return {
      symbol: definition.name,
      qualified_symbol: qualified,
      kind,
      start_byte: startByte,
      end_byte: endByte,
      start_line: lineIndex.lineAt(startByte),
      end_line: lineIndex.lineAt(Math.max(startByte, endByte - 1)),
      parameters: parameterShapes(language, parameterNodes.get(definition.node.id) ?? null),
    };
  });
}

function parameterShapes(language: string, parameters: SyntaxNode | null): ParameterShape[] {
  if (parameters === null) return [];
  let rows: ParameterShape[] = [];
  // Python's own `positional_only` is set to False on the separator branch and
  // never to True, so the `elif positional_only` arm below is unreachable there
  // too: `positional_only` is produced solely by the retroactive rewrite of
  // already-collected rows. Kept in the same shape so the two stay diffable.
  const positionalOnly = false;
  let keywordOnly = false;
  for (const child of parameters.namedChildren) {
    if (child.type === "positional_separator") {
      rows = rows.map((row) =>
        row.kind === "positional" ? { ...row, kind: "positional_only" } : row,
      );
      continue;
    }
    if (child.type === "keyword_separator") {
      keywordOnly = true;
      continue;
    }
    let nameNode: SyntaxNode | null =
      child.type === "identifier" ||
      child.type === "property_identifier" ||
      child.type === "object_pattern" ||
      child.type === "array_pattern"
        ? child
        : null;
    nameNode =
      nameNode ?? child.childForFieldName("name") ?? child.childForFieldName("pattern") ?? null;
    if (nameNode === null) {
      // e.g. a bare `rest_pattern` (`...rest`) -- its identifier is a plain
      // child, not a named field. A leading comment (`.../* c */ rest`) must
      // not be mistaken for it (same class as finding 7/8).
      nameNode = firstNamedChild(child);
    }
    if (nameNode === null) continue;
    // A destructured slot (`{ a, b }` / `[a, b]`) collapses to ONE positional
    // parameter marked `destructured`, never N flat ones -- expanding it would
    // corrupt positional matching for every caller (E7). It can appear bare --
    // JS, or TS without a wrapper, where `child` itself IS the pattern -- or as
    // the `pattern`/`name` field of a `required_parameter`/`optional_parameter`
    // wrapper (TS) -- `nameNode` above already resolves to it in both cases.
    const destructured = nameNode.type === "object_pattern" || nameNode.type === "array_pattern";
    let name: string;
    if (destructured) {
      const bindingNames = [...bindingIdentifiers(nameNode)].map((binding) => binding.text);
      name = bindingNames.length > 0 ? bindingNames.join(",") : "destructured";
    } else {
      name = nameNode.text;
    }
    let kind: ParameterKind;
    if (
      child.type === "list_splat_pattern" ||
      child.type === "rest_pattern" ||
      nameNode.type === "rest_pattern"
    ) {
      kind = "variadic";
      name = removePrefix(removePrefix(name, "*"), "...");
    } else if (child.type === "dictionary_splat_pattern") {
      kind = "keyword_variadic";
      name = removePrefix(name, "**");
    } else if (positionalOnly) {
      kind = "positional_only";
    } else if (keywordOnly) {
      kind = "keyword_only";
    } else {
      kind = "positional";
    }
    // A default value is authoritative via node structure, never text matching
    // (E8): TS's `required_parameter`/`optional_parameter` wrapper exposes it as
    // a `value` field; bare JS/TS `assignment_pattern` (untyped `a = 1`)
    // exposes it as `right`. A callback type's `=>` in the parameter's raw text
    // is not a default and must never be mistaken for one.
    const hasDefault =
      child.childForFieldName("value") !== null || child.childForFieldName("right") !== null;
    let required = !hasDefault && child.type !== "optional_parameter";
    if (language !== "python" && kind === "variadic") required = false;
    rows.push({ name, kind, required, position: rows.length, destructured });
    if (language === "python" && child.type === "list_splat_pattern") keywordOnly = true;
  }
  return rows;
}

// Node types that hold a genuine positional/keyword argument list. Anything
// else reachable through the `arguments` field (a tagged template's
// `template_string`, a `new` with no parens at all) is not a positional arg
// list and must not have its children miscounted as one (E4).
const ARGUMENT_LIST_TYPES: ReadonlySet<string> = new Set(["arguments", "argument_list"]);

function callShapeFor(node: SyntaxNode): CallShape {
  const argumentsNode = node.childForFieldName("arguments");
  let positionalCount = 0;
  const keywords: string[] = [];
  let positionalSpread = false;
  let keywordSpread = false;
  if (argumentsNode !== null && ARGUMENT_LIST_TYPES.has(argumentsNode.type)) {
    for (const argument of argumentsNode.namedChildren) {
      if (argument.isExtra) {
        // A comment is a named "extra" node inside the argument list, not an
        // argument -- `g(1,  # note\n  2)` must count 2 positional args, not 3
        // (E4-adjacent).
        continue;
      }
      if (argument.type === "list_splat" || argument.type === "spread_element") {
        positionalSpread = true;
      } else if (argument.type === "dictionary_splat") {
        keywordSpread = true;
      } else if (argument.type === "keyword_argument") {
        const name = argument.childForFieldName("name");
        if (name !== null) keywords.push(name.text);
      } else {
        positionalCount += 1;
      }
    }
  } else if (argumentsNode !== null && argumentsNode.type === "generator_expression") {
    // Python `summarize(x for x in items)` -- a single argument whose contents
    // are not a positional list. Model it as one positional with spread-like
    // uncertainty so signature analysis routes to `review` instead of
    // fabricating a match (E4).
    positionalCount = 1;
    positionalSpread = true;
  }
  // else: e.g. a tagged template's `template_string` -- no positional args at
  // all; its `string_fragment`/`template_substitution` children are not call
  // arguments and must not be counted (E4).
  const typeArguments = node.childForFieldName("type_arguments");
  return {
    positional_count: positionalCount,
    keywords,
    has_positional_spread: positionalSpread,
    has_keyword_spread: keywordSpread,
    type_argument_count:
      typeArguments === null
        ? null
        : typeArguments.namedChildren.filter((argument) => !argument.isExtra).length,
    constructor: node.type === "new_expression",
  };
}

/**
 * The quote-stripped text of a call's sole string-literal argument.
 *
 * Used for `require('./mod')` and dynamic `import('./mod')` (E9): both keep an
 * ordinary `call` row for signature purposes, but gain a `module_path` so the
 * module edge stays visible to the resolver.
 */
function stringLiteralArgument(node: SyntaxNode): string | null {
  const argumentsNode = node.childForFieldName("arguments");
  if (argumentsNode === null) return null;
  // A leading comment (`require(/* c */ './mod')`) is a named "extra" node in
  // source order before the real argument; `namedChild(0)` would grab it
  // instead and silently drop the module edge.
  const first = firstNamedChild(argumentsNode);
  if (first === null || first.type !== "string") return null;
  return stripQuotes(captureName(first));
}

/** Local binding identifiers, without mistaking object property keys for bindings. */
function* bindingIdentifiers(node: SyntaxNode): Generator<SyntaxNode> {
  if (node.type === "identifier" || node.type === "shorthand_property_identifier_pattern") {
    yield node;
  } else if (node.type === "pair_pattern") {
    const value = node.childForFieldName("value");
    if (value !== null) yield* bindingIdentifiers(value);
  } else if (node.type === "assignment_pattern") {
    const left = node.childForFieldName("left");
    if (left !== null) yield* bindingIdentifiers(left);
  } else if (node.type === "rest_pattern") {
    // A comment right after `...` (`[.../* c */ rest]`) must not be mistaken
    // for the bound identifier -- same class of bug as finding 7/8.
    const child = firstNamedChild(node);
    if (child !== null) yield* bindingIdentifiers(child);
  } else if (node.type === "array_pattern" || node.type === "object_pattern") {
    for (const child of node.namedChildren) yield* bindingIdentifiers(child);
  }
}

const TYPE_WRAPPER_TYPES: ReadonlySet<string> = new Set([
  "union_type",
  "intersection_type",
  "array_type",
  "type_arguments",
]);

/**
 * Descend a TS wrapper node to its identifying name leaf(ves).
 *
 * Unwraps `generic_type` (the head name plus one entry per type argument),
 * `union_type`, `intersection_type`, `array_type`, `function_type` (its
 * `return_type` only -- the parameter list is a binding context, not a type
 * reference), and `type_arguments`, stopping at
 * `type_identifier`/`identifier`/`predefined_type` (`number`, `string`,
 * `void`, ...) leaves. Qualified names such as `ns.Base` stay intact as one
 * resolvable leaf. Anything else (for example an object type literal) yields
 * nothing.
 */
function descendTypeNames(node: SyntaxNode | null): SyntaxNode[] {
  if (node === null) return [];
  if (
    node.type === "type_identifier" ||
    node.type === "identifier" ||
    node.type === "predefined_type" ||
    node.type === "member_expression" ||
    node.type === "nested_type_identifier"
  ) {
    return [node];
  }
  if (node.type === "generic_type") {
    const names = descendTypeNames(node.childForFieldName("name"));
    names.push(...descendTypeNames(node.childForFieldName("type_arguments")));
    return names;
  }
  if (node.type === "function_type") return descendTypeNames(node.childForFieldName("return_type"));
  if (TYPE_WRAPPER_TYPES.has(node.type)) {
    const names: SyntaxNode[] = [];
    for (const child of node.namedChildren) names.push(...descendTypeNames(child));
    return names;
  }
  return [];
}

/** Emit a `type_use` row per identifying name reachable by descending `node`. */
function emitTypeUseNames(node: SyntaxNode | null, addReference: ReferenceAdder): void {
  for (const name of descendTypeNames(node)) {
    addReference("type_use", name, {
      targetName: captureName(name),
      writtenName: captureName(name),
    });
  }
}

/**
 * Emit the head name of a heritage clause as `inheritance`, extras as `type_use`.
 *
 * `extends Base<T>` yields an `inheritance` row for `Base` and a `type_use` row
 * for `T`; `extends Base` (no type arguments) yields just the former.
 */
function emitHeritageName(node: SyntaxNode | null, addReference: ReferenceAdder): void {
  const names = descendTypeNames(node);
  const head = names[0];
  if (head === undefined) return;
  addReference("inheritance", head, {
    targetName: captureName(head),
    writtenName: captureName(head),
  });
  for (const extra of names.slice(1)) {
    addReference("type_use", extra, {
      targetName: captureName(extra),
      writtenName: captureName(extra),
    });
  }
}

/** True if `node` is the LHS of a plain or augmented assignment. */
function isAssignmentTarget(node: SyntaxNode): boolean {
  const parent = node.parent;
  if (parent === null) return false;
  if (
    parent.type === "assignment" ||
    parent.type === "augmented_assignment" ||
    parent.type === "assignment_expression" ||
    parent.type === "augmented_assignment_expression"
  ) {
    return parent.childForFieldName("left")?.id === node.id;
  }
  return false;
}

/**
 * Emit `read`/`write` for a member-access node that is not itself a call.
 *
 * Handles Python `attribute` and JS/TS `member_expression` (E5). Three cases
 * already own this exact span with a different `kind` and must stay singly
 * represented, not duplicated as a `read`/`write` too: a call's
 * `function`/`constructor` (its own `call` row), a Python decorator's target
 * (its own `decorator` row), and a class's superclass entry (its own
 * `inheritance` row).
 */
function emitMemberAccess(node: SyntaxNode, addReference: ReferenceAdder): void {
  const parent = node.parent;
  if (parent !== null) {
    let ancestor: SyntaxNode | null = parent;
    while (ancestor !== null) {
      if (ancestor.type === "class_heritage" || ancestor.type === "extends_type_clause") return;
      ancestor = ancestor.parent;
    }
    if (
      (parent.type === "call" || parent.type === "call_expression") &&
      parent.childForFieldName("function")?.id === node.id
    ) {
      return;
    }
    if (
      parent.type === "new_expression" &&
      parent.childForFieldName("constructor")?.id === node.id
    ) {
      return;
    }
    if (parent.type === "decorator") return;
    if (
      parent.type === "argument_list" &&
      parent.parent !== null &&
      parent.parent.type === "class_definition" &&
      parent.parent.childForFieldName("superclasses")?.id === parent.id
    ) {
      return;
    }
  }
  const propertyField =
    node.childForFieldName("attribute") ?? node.childForFieldName("property") ?? null;
  if (propertyField === null) return;
  const objectField = node.childForFieldName("object");
  const text = captureName(node);
  const kind: ReferenceKind = isAssignmentTarget(node) ? "write" : "read";
  addReference(kind, node, {
    targetName: text,
    writtenName: text,
    receiverText: objectField === null ? null : captureName(objectField),
  });
}

function pythonRecords(node: SyntaxNode, source: SourceText, addReference: ReferenceAdder): void {
  if (node.type === "import_from_statement") {
    const module = node.childForFieldName("module_name");
    const modulePath = module === null ? null : captureName(module);
    for (const child of node.namedChildren) {
      if ((module !== null && child.id === module.id) || child.isExtra) {
        // A comment among the imported names (`from pkg import (a,
        // # note\n b)`) is a named "extra" node too -- without this it would
        // fall through as a bogus import of itself.
        continue;
      }
      const imported = child.type === "aliased_import" ? child.childForFieldName("name") : child;
      const aliasNode = child.type === "aliased_import" ? child.childForFieldName("alias") : null;
      const importedName = imported !== null ? captureName(imported) : captureName(child);
      const alias = aliasNode !== null ? captureName(aliasNode) : null;
      addReference("import", child, {
        targetName: importedName,
        writtenName: alias || importedName,
        modulePath,
        importedName,
        alias,
      });
    }
  } else if (node.type === "import_statement") {
    for (const child of node.namedChildren) {
      if (child.isExtra) continue;
      const imported = child.type === "aliased_import" ? child.childForFieldName("name") : child;
      const aliasNode = child.type === "aliased_import" ? child.childForFieldName("alias") : null;
      const importedName = imported !== null ? captureName(imported) : captureName(child);
      const alias = aliasNode !== null ? captureName(aliasNode) : null;
      addReference("import", child, {
        targetName: importedName,
        writtenName: alias || importedName,
        modulePath: importedName,
        importedName: null,
        alias,
      });
    }
  } else if (node.type === "decorator") {
    let target = node.childForFieldName("function") ?? node.namedChild(0);
    if (target !== null && target.type === "call") target = target.childForFieldName("function");
    if (target !== null) {
      addReference("decorator", target, {
        targetName: captureName(target),
        writtenName: captureName(target),
      });
    }
  } else if (node.type === "class_definition") {
    const superclasses = node.childForFieldName("superclasses");
    if (superclasses !== null) {
      for (const item of superclasses.namedChildren) {
        if (item.isExtra) {
          // A comment among the base classes (`class Child(Base,\n # note\n
          // Other):`) is a named "extra" node too, not a base class.
          continue;
        }
        if (item.type === "keyword_argument") {
          // e.g. `metaclass=Meta` -- the value already surfaces as a `read` via
          // the plain identifier fallback; the clause itself (`metaclass=Meta`)
          // is not a base class (E12).
          continue;
        }
        addReference("inheritance", item, {
          targetName: captureName(item),
          writtenName: captureName(item),
        });
      }
    }
  } else if (node.type === "call") {
    const functionNode = node.childForFieldName("function");
    if (functionNode !== null) {
      const receiver = functionNode.childForFieldName("object");
      addReference("call", functionNode, {
        targetName: captureName(functionNode),
        writtenName: captureName(functionNode),
        receiverText: receiver === null ? null : captureName(receiver),
        callShape: callShapeFor(node),
      });
    }
  } else if (node.type === "type") {
    const target = node.namedChild(0);
    if (target !== null) {
      addReference("type_use", target, {
        targetName: captureName(target),
        writtenName: captureName(target),
      });
    }
  } else if (node.type === "attribute") {
    emitMemberAccess(node, addReference);
  } else if (node.type === "assignment" || node.type === "augmented_assignment") {
    // `__all__ = [...]`/`__all__ += [...]` (E13) -- the query already restricts
    // the match to a literal left-hand `__all__` via #eq?, so no name check is
    // needed here. Each string entry becomes an `export` row naming the symbol
    // it re-publishes; a rename of that symbol must also touch its `__all__`
    // entry.
    const right = node.childForFieldName("right");
    if (right !== null && (right.type === "list" || right.type === "tuple")) {
      for (const entry of right.namedChildren) {
        if (entry.type !== "string") continue;
        const exported = stripQuotes(captureName(entry));
        addReference("export", entry, {
          targetName: exported,
          writtenName: exported,
          importedName: exported,
        });
      }
    }
  }
  void source;
}

function javascriptRecords(
  node: SyntaxNode,
  source: SourceText,
  addReference: ReferenceAdder,
): void {
  if (node.type === "import_statement") {
    const sourceNode = node.childForFieldName("source");
    const modulePath = sourceNode === null ? null : stripQuotes(captureName(sourceNode));
    const clause = node.namedChildren.find((child) => child.type === "import_clause") ?? null;
    if (clause !== null) {
      for (const item of clause.namedChildren) {
        if (item.isExtra) {
          // A comment between clause items (`import Default, /* c */ { a } from
          // 'mod'`) is a named "extra" node too -- the `else` branch below would
          // otherwise mistake it for a bare default-import identifier.
          continue;
        }
        if (item.type === "named_imports") {
          for (const specifier of item.namedChildren) {
            const name = specifier.childForFieldName("name");
            const alias = specifier.childForFieldName("alias");
            const imported = name !== null ? captureName(name) : captureName(specifier);
            const aliasText = alias !== null ? captureName(alias) : null;
            addReference("import", specifier, {
              targetName: imported,
              writtenName: aliasText || imported,
              modulePath,
              importedName: imported,
              alias: aliasText,
            });
          }
        } else if (item.type === "namespace_import") {
          const alias = firstNamedChild(item);
          if (alias !== null) {
            addReference("import", alias, {
              targetName: "*",
              writtenName: captureName(alias),
              modulePath,
              importedName: "*",
              alias: captureName(alias),
            });
          }
        } else {
          addReference("import", item, {
            targetName: "default",
            writtenName: captureName(item),
            modulePath,
            importedName: "default",
          });
        }
      }
    } else {
      // `import './polyfill'` -- no import_clause at all: a side-effect import
      // that still opens a module edge (E9).
      addReference("import", node, {
        targetName: modulePath || "",
        writtenName: modulePath || "",
        modulePath,
        importedName: null,
      });
    }
  } else if (node.type === "export_statement") {
    const sourceNode = node.childForFieldName("source");
    const modulePath = sourceNode === null ? null : stripQuotes(captureName(sourceNode));
    for (const clause of node.namedChildren) {
      if (clause.type === "export_clause") {
        for (const specifier of clause.namedChildren) {
          const name = specifier.childForFieldName("name");
          const alias = specifier.childForFieldName("alias");
          const exported = captureName(alias ?? name ?? specifier);
          addReference("export", specifier, {
            targetName: name !== null ? captureName(name) : exported,
            writtenName: exported,
            modulePath,
            importedName: name !== null ? captureName(name) : exported,
            alias: alias !== null ? captureName(alias) : null,
          });
        }
      } else if (clause.type === "namespace_export") {
        // `export * as ns from './x'` -- the namespace alias lives under the
        // wrapper node, not as a direct export_statement child (E3).
        const aliasNode = firstNamedChild(clause);
        const aliasText = aliasNode === null ? null : captureName(aliasNode);
        addReference("export", clause, {
          targetName: "*",
          writtenName: aliasText || "*",
          modulePath,
          importedName: "*",
          alias: aliasText,
        });
      }
    }
    const star = node.children.find((child) => child.type === "*");
    if (modulePath !== null && star !== undefined) {
      // `export * from './x'` -- bare barrel re-export: no clause at all, just a
      // literal `*` token directly under export_statement (E3). `export * as ns
      // ...` is handled above via `namespace_export`, whose own `*` is nested
      // one level deeper so it never reaches this branch.
      addReference("export", star, {
        targetName: "*",
        writtenName: "*",
        modulePath,
        importedName: "*",
      });
    }
    const declaration = node.childForFieldName("declaration");
    if (declaration !== null) {
      if (
        declaration.type === "lexical_declaration" ||
        declaration.type === "variable_declaration"
      ) {
        for (const declarator of declaration.namedChildren) {
          if (declarator.type !== "variable_declarator") continue;
          const name = declarator.childForFieldName("name");
          if (name === null) continue;
          for (const binding of bindingIdentifiers(name)) {
            const exported = captureName(binding);
            addReference("export", binding, { targetName: exported, writtenName: exported });
          }
        }
      } else {
        const name = declaration.childForFieldName("name");
        if (name !== null) {
          const exported = captureName(name);
          const isDefault = node.children.some((child) => child.type === "default");
          addReference("export", name, {
            targetName: exported,
            writtenName: isDefault ? "default" : exported,
          });
        }
      }
    }
    const value = node.childForFieldName("value");
    if (value !== null) {
      addReference("export", value, { targetName: captureName(value), writtenName: "default" });
    }
  } else if (node.type === "decorator") {
    // `@Name`, `@ns.Name`, `@Factory()` -- mirrors the Python decorator handler
    // (E6). The factory-call form keeps its own `call` row from the
    // `call_expression` branch; this row is additional.
    let target = firstNamedChild(node);
    if (target !== null && target.type === "call_expression") {
      target = target.childForFieldName("function");
    }
    if (target !== null) {
      addReference("decorator", target, {
        targetName: captureName(target),
        writtenName: captureName(target),
      });
    }
  } else if (node.type === "class_heritage") {
    for (const clause of node.namedChildren) {
      if (clause.type === "extends_clause") {
        // TS: `extends Base<T>` -- the identifier is under a `value` field, not
        // the clause itself (E1).
        emitHeritageName(clause.childForFieldName("value"), addReference);
      } else if (clause.type === "implements_clause") {
        // TS: `implements Foo, Bar<T>` -- each interface name is a direct named
        // child of the clause (E1).
        for (const interfaceNode of clause.namedChildren) {
          emitHeritageName(interfaceNode, addReference);
        }
      } else {
        // JS: the grammar puts the identifier directly under class_heritage (no
        // extends_clause wrapper); already worked.
        emitHeritageName(clause, addReference);
      }
    }
  } else if (node.type === "extends_type_clause") {
    // TS interface heritage: named children are already type_identifier (or
    // generic_type, handled by the E2 generic_type branch below) -- left
    // untouched, see hardening plan Task 2.1.
    for (const item of node.namedChildren) {
      if (item.isExtra) {
        // A comment (`extends /* c */ Base`) is a named "extra" node too, not
        // an interface name.
        continue;
      }
      addReference("inheritance", item, {
        targetName: captureName(item),
        writtenName: captureName(item),
      });
    }
  } else if (node.type === "call_expression" || node.type === "new_expression") {
    const functionNode =
      node.childForFieldName("function") ?? node.childForFieldName("constructor") ?? null;
    if (functionNode !== null) {
      const receiver = functionNode.childForFieldName("object");
      const isModuleCall =
        functionNode.type === "import" ||
        (functionNode.type === "identifier" && captureName(functionNode) === "require");
      const modulePath = isModuleCall ? stringLiteralArgument(node) : null;
      addReference("call", functionNode, {
        targetName: captureName(functionNode),
        writtenName: captureName(functionNode),
        receiverText: receiver === null ? null : captureName(receiver),
        callShape: callShapeFor(node),
        modulePath,
      });
    }
  } else if (node.type === "generic_type") {
    // `Box<Item>` -- one type_use for `Box`, one per type argument (E2).
    emitTypeUseNames(node, addReference);
  } else if (node.type === "type_annotation") {
    // `: A | B`, `: C & D`, `: Widget[]`, `: (e: Event) => Widget` -- unwrap to
    // the inner type names instead of capturing the whole expression verbatim
    // (E2). A nested generic_type is also matched by its own top-level pattern
    // above; `add` dedupes the identical span. A leading comment
    // (`: /* c */ Widget`) is a named "extra" node too and must not be mistaken
    // for the annotated type -- that silently dropped the type_use row entirely.
    emitTypeUseNames(firstNamedChild(node), addReference);
  } else if (node.type === "member_expression") {
    emitMemberAccess(node, addReference);
  } else if (
    node.type === "jsx_opening_element" ||
    node.type === "jsx_self_closing_element" ||
    node.type === "jsx_closing_element"
  ) {
    // `<Widget />`, `<Widget>...</Widget>` -- a component-reference row per
    // element name, opening/self-closing and closing alike, so a rename finds
    // every JSX use (E14, TSX only). Lower-case names (`<div>`) are intrinsic
    // HTML tags, not project symbols, but resolving that distinction is the
    // resolver's job, not the extractor's -- an unmatched `type_use` is simply
    // never a hit.
    const target = node.childForFieldName("name");
    if (target !== null) {
      addReference("type_use", target, {
        targetName: captureName(target),
        writtenName: captureName(target),
      });
    }
  }
  void source;
}

function outerNode(node: SyntaxNode): SyntaxNode {
  let outer = node;
  if (node.parent !== null && node.parent.type === "decorated_definition") outer = node.parent;
  if (
    outer.parent !== null &&
    (outer.parent.type === "export_statement" || outer.parent.type === "lexical_declaration")
  ) {
    outer = outer.parent;
  }
  if (outer.parent !== null && outer.parent.type === "export_statement") outer = outer.parent;
  return outer;
}

function hasDefinitionAncestor(node: SyntaxNode, index: DefinitionIndex): boolean {
  let parent = node.parent;
  while (parent !== null) {
    if (index.byNodeId.has(parent.id)) return true;
    parent = parent.parent;
  }
  return false;
}

function symbolContext(
  definition: Definition,
  index: DefinitionIndex,
): { kind: string; parent: string | null; qualified: string } {
  const chain: Definition[] = [];
  let parent = definition.node.parent;
  while (parent !== null) {
    const candidate = index.byNodeId.get(parent.id);
    if (candidate !== undefined) chain.push(candidate);
    parent = parent.parent;
  }
  chain.reverse();
  const scope = chain.some((item) => CALLABLE_KINDS.has(item.kind))
    ? chain
    : chain.filter((item) => CONTAINER_KINDS.has(item.kind));
  const parentName = scope.map((item) => item.name).join(".") || null;
  const qualified = parentName ? `${parentName}.${definition.name}` : definition.name;
  let kind = definition.kind;
  const innermost = scope[scope.length - 1];
  if (innermost !== undefined && CONTAINER_KINDS.has(innermost.kind) && kind === "function") {
    kind = "method";
  }
  return { kind, parent: parentName, qualified };
}

function contentRange(
  outer: SyntaxNode,
  node: SyntaxNode,
  kind: string,
  index: DefinitionIndex,
  source: SourceText,
): [number, number] {
  const outerStart = source.byte(outer.startIndex);
  const outerEnd = source.byte(outer.endIndex);
  if (!CONTAINER_KINDS.has(kind)) return [outerStart, outerEnd];
  // The old code scanned every definition to take the minimum qualifying start.
  // Definitions are start-ascending, so the first qualifying one after
  // outer.startByte *is* that minimum. A definition starting at or after
  // outer.endByte cannot end inside outer, which bounds the scan.
  let position = bisectRight(index.starts, outerStart);
  while (position < index.definitions.length) {
    const candidate = (index.definitions[position] as Definition).node;
    const candidateStart = source.byte(candidate.startIndex);
    if (candidateStart >= outerEnd) break;
    if (candidate.id !== node.id && source.byte(candidate.endIndex) <= outerEnd) {
      return [outerStart, candidateStart];
    }
    position += 1;
  }
  return [outerStart, outerEnd];
}

function makeChunk(
  path: string,
  language: string,
  kind: string,
  symbol: string | null,
  qualified: string | null,
  parent: string | null,
  startByte: number,
  endByte: number,
  startLine: number,
  content: string,
  partIndex: number,
): ExtractedChunk {
  const context = [`language: ${language}`, `path: ${path}`, `kind: ${kind}`];
  if (qualified) context.push(`symbol: ${qualified}`);
  const prefix = context.join("\n");
  const embeddingText = `${prefix}\n${content}`;
  const normalized = normalizeIdentifier(
    [path, qualified || "", symbol || ""].filter(Boolean).join(" "),
  );
  return {
    kind,
    symbol,
    qualified_symbol: qualified,
    parent_symbol: parent,
    start_byte: startByte,
    end_byte: endByte,
    start_line: startLine,
    end_line: startLine + countNewlines(content),
    content,
    embedding_text: embeddingText,
    search_text: `${embeddingText}\n${normalized}`,
    part_index: partIndex,
    embedding_prefix: prefix,
    search_suffix: normalized,
  };
}

// --- small helpers matching Python's string and list semantics ---------------

/** `bisect_right` over an ascending Int32Array. */
function bisectRight(values: Int32Array, target: number): number {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (target < (values[middle] as number)) high = middle;
    else low = middle + 1;
  }
  return low;
}

/**
 * Python's `str.isspace()`.
 *
 * Not the same set as JavaScript's `\s`: Python counts the file/group/record
 * separators `\x1c`-`\x1f` and NEL (`\x85`), and does not count `﻿`. The
 * difference decides where a chunk's content ends, and therefore its byte
 * range, so it is not a detail that can be approximated.
 */
function isPythonSpace(code: number): boolean {
  if (code === 0x20) return true;
  if (code >= 0x09 && code <= 0x0d) return true;
  if (code >= 0x1c && code <= 0x1f) return true;
  if (code === 0x85 || code === 0xa0 || code === 0x1680) return true;
  if (code >= 0x2000 && code <= 0x200a) return true;
  return (
    code === 0x2028 || code === 0x2029 || code === 0x202f || code === 0x205f || code === 0x3000
  );
}

/** Python's `str.rstrip()`: strip every trailing whitespace character. */
function rstrip(value: string): string {
  let end = value.length;
  while (end > 0 && isPythonSpace(value.charCodeAt(end - 1))) end -= 1;
  return end === value.length ? value : value.slice(0, end);
}

function countNewlines(value: string): number {
  let count = 0;
  for (let index = value.indexOf("\n"); index !== -1; index = value.indexOf("\n", index + 1)) {
    count += 1;
  }
  return count;
}

/**
 * Python's `str.splitlines(keepends=True)`.
 *
 * Broader than splitting on `\n`: a form feed is a real page separator in C
 * and Python sources, and Python breaks a line on it. Since each line's UTF-8
 * length becomes a chunk's byte offset, splitting differently here would shift
 * every offset in a file that uses one.
 */
function isLineBoundary(code: number): boolean {
  return (
    (code >= 0x0a && code <= 0x0d) || // \n \v \f \r
    (code >= 0x1c && code <= 0x1e) || // file/group/record separators
    code === 0x85 ||
    code === 0x2028 ||
    code === 0x2029
  );
}

function splitLinesKeepEnds(value: string): string[] {
  const lines: string[] = [];
  let start = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (!isLineBoundary(code)) continue;
    // `\r\n` is one boundary; every other pair is two.
    const width = code === 0x0d && value.charCodeAt(index + 1) === 0x0a ? 2 : 1;
    lines.push(value.slice(start, index + width));
    index += width - 1;
    start = index + 1;
  }
  if (start < value.length) lines.push(value.slice(start));
  return lines;
}

/** Python's `str.strip("'\"")`. */
function stripQuotes(value: string): string {
  let start = 0;
  let end = value.length;
  while (start < end && (value[start] === "'" || value[start] === '"')) start += 1;
  while (end > start && (value[end - 1] === "'" || value[end - 1] === '"')) end -= 1;
  return value.slice(start, end);
}

function removePrefix(value: string, prefix: string): string {
  return value.startsWith(prefix) ? value.slice(prefix.length) : value;
}

/** Python's `<` on `str`: a code-unit-wise comparison, which `localeCompare` is not. */
function compareStrings(left: string, right: string): number {
  if (left === right) return 0;
  return left < right ? -1 : 1;
}

/** Whether a byte range holds anything but ASCII/Unicode whitespace, as `bytes.strip()` asks. */
function hasNonWhitespace(bytes: Uint8Array, start: number, end: number): boolean {
  for (let index = start; index < end; index += 1) {
    const byte = bytes[index] as number;
    // Python's `bytes.strip()` strips exactly this set: space, \t \n \v \f \r.
    if (byte !== 0x20 && (byte < 0x09 || byte > 0x0d)) return true;
  }
  return false;
}
