/**
 * Conservative, syntax-only reference classification over structural rows.
 *
 * A transliteration of `reference_service.py`. The one structural change is the
 * dependency direction: it takes a {@link ReferenceStore} rather than a
 * concrete store, so the resolver lands before LanceDB does and so the pushed-
 * down queries it relies on are a written contract rather than an incidental
 * property of one implementation. Every method that touches storage or the
 * filesystem is async, because the JS LanceDB bindings are.
 *
 * The hard rule this module exists to keep: an `exact` resolution must never be
 * wrong. Everything that cannot be proven from syntax degrades to `likely` or
 * `unresolved` with a reason code, and anything the index could not see at all
 * surfaces as a limitation -- so an empty limitation list is real evidence of
 * full coverage rather than the absence of a check.
 */

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { CodeIndexingError } from "./errors.ts";
import type {
  CompletenessReport,
  DeclarationSelector,
  ParameterShape,
  RefactorAnalysis,
  RefactorFinding,
  RefactorOperation,
  ReferenceBackfillReport,
  ReferenceHit,
  ReferenceKind,
  ReferenceLimitation,
  ReferenceResponse,
  RenameOperation,
  ResolutionLevel,
  SelectedDeclaration,
  SignatureChangeOperation,
} from "./models.ts";
import {
  REFERENCE_SCHEMA_VERSION,
  type ReferenceRecord,
  type ReferenceStore,
} from "./reference-store.ts";
import { STRUCTURAL_LANGUAGES } from "./extractor.ts";

/**
 * Reason codes that describe something the syntax-only index could not see.
 *
 * They surface as limitations whatever resolution level they carry, so a caller
 * never reads an empty limitation list as proof of full coverage.
 * `unproven_reexport` is included deliberately (unlike `name_only_candidate`):
 * a barrel import proves a module edge exists even when the chain cannot be
 * resolved fully, which is stronger evidence than an unqualified bare name.
 */
const LIMITATION_REASONS: ReadonlySet<string> = new Set([
  "wildcard_import",
  "unknown_receiver",
  "ambiguous_symbol",
  "unproven_reexport",
]);

/**
 * Bound re-export traversal so malformed or unusually deep barrel chains are
 * reported as unproven rather than walked indefinitely.
 */
const MAX_REEXPORT_DEPTH = 4;

/**
 * Limitations that mean whole files were never analyzed. Any of them forces the
 * completeness state to "incomplete" rather than merely "dynamic limitations".
 */
const COVERAGE_GAP_CODES: ReadonlySet<string> = new Set([
  "unsupported_language",
  "parse_error",
  "stale_file",
]);

const BOM = new Uint8Array([0xef, 0xbb, 0xbf]);

/** How many individual paths a coverage limitation names before it summarizes. */
const MAX_LIMITATION_PATHS = 10;

/** One file's BOM-stripped bytes and the offset that was removed. */
type FileBytes = readonly [source: Uint8Array, bom: number];

/**
 * Pinned-snapshot data shared by reference and refactor responses.
 *
 * The full hit list and source cache let refactor analysis compute
 * page-independent counts without repeating classification. Records contain
 * reference and coverage rows; declarations are fetched separately through
 * narrowed queries.
 */
interface ReferenceQuery {
  readonly response: ReferenceResponse;
  readonly records: ReferenceRecord[];
  readonly hits: ReferenceHit[];
  readonly root: string | null;
  readonly sources: Map<string, FileBytes>;
}

interface ClassifiedFindings {
  readonly mustChange: RefactorFinding[];
  readonly likelyChange: RefactorFinding[];
  readonly review: RefactorFinding[];
  readonly evidence: RefactorFinding[];
}

interface CursorPayload {
  version: number;
  project_id: string;
  path: string;
  qualified_symbol: string;
  kinds: string[];
  offset: number;
  limit: number;
  operation_digest: string | null;
}

/** Findings whose concrete edit spans have not already been seen. */
function dedupeEditSpans(
  findings: readonly RefactorFinding[],
  seen: Set<string>,
): RefactorFinding[] {
  const deduped: RefactorFinding[] = [];
  for (const finding of findings) {
    if (finding.edit_start_byte === null || finding.edit_end_byte === null) {
      deduped.push(finding);
      continue;
    }
    const key = `${finding.path}\0${finding.edit_start_byte}\0${finding.edit_end_byte}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(finding);
  }
  return deduped;
}

export interface FindReferencesOptions {
  kinds?: ReadonlySet<string> | null;
  limit?: number;
  cursor?: string | null;
  backfill?: ReferenceBackfillReport | null;
  /**
   * Internal to `analyzeRefactor`; binds the operation to the cursor so later
   * pages cannot silently change it.
   */
  operationDigest?: string | null;
}

export interface AnalyzeRefactorOptions {
  limit?: number;
  cursor?: string | null;
  backfill?: ReferenceBackfillReport | null;
}

/** Resolve only syntax facts that select a declaration unambiguously. */
export class ReferenceService {
  readonly store: ReferenceStore;

  constructor(store: ReferenceStore) {
    this.store = store;
  }

  /** Resolve references for `selector`, one page at a time. */
  async findReferences(
    selector: DeclarationSelector,
    options: FindReferencesOptions = {},
  ): Promise<ReferenceResponse> {
    return (await this.#findReferencesWithRecords(selector, options)).response;
  }

  /**
   * `findReferences`'s body, also returning everything it computed along the way.
   *
   * `analyzeRefactor` needs the paginated response, the full pinned-snapshot
   * record set, and the full classified hit list (for the
   * declaration/override/signature work below), not just the page. Fetching
   * through the plain `findReferences` and then re-fetching the records at the
   * same `snapshot_version` -- and separately re-running classification over
   * it -- was a second full-table materialization and a second classification
   * pass over data already in hand (S4/E1).
   */
  async #findReferencesWithRecords(
    selector: DeclarationSelector,
    options: FindReferencesOptions,
  ): Promise<ReferenceQuery> {
    const kinds = options.kinds ?? null;
    const limit = options.limit ?? 100;
    const cursor = options.cursor ?? null;
    const operationDigest = options.operationDigest ?? null;
    if (limit < 1 || limit > 500) {
      throw new CodeIndexingError("INVALID_FILTER", "limit must be between 1 and 500");
    }
    const selected = await this.#select(selector);
    if (!STRUCTURAL_LANGUAGES.has(selected.language)) {
      throw new CodeIndexingError(
        "UNSUPPORTED_LANGUAGE",
        `Structural references are not extracted for ${selected.language}. ` +
          `Supported languages are ${[...STRUCTURAL_LANGUAGES].sort().join(", ")}.`,
        {
          project: selected.project_id,
          path: selected.path,
          language: selected.language,
        },
      );
    }
    if (!(await this.store.hasReferenceTable(selected.project_id))) {
      // A missing table and a legitimately empty one both read as `[]` from the
      // record and version queries (S5). Trusting that silence would report "no
      // references" for a project whose reference index was never built -- e.g.
      // a partition indexed before this feature existed -- instead of surfacing
      // the real, actionable state.
      throw new CodeIndexingError(
        "REFERENCE_INDEX_UNAVAILABLE",
        "The reference index has not been built for this project. " +
          "Run ensure_reference_index (or reindex) before querying references.",
        { project: selected.project_id },
      );
    }
    let version = await this.store.referenceVersion(selected.project_id);
    let offset = 0;
    if (cursor !== null) {
      const payload = decodeCursor(cursor);
      if (payload.project_id !== selected.project_id || payload.path !== selected.path) {
        throw new CodeIndexingError(
          "INVALID_CURSOR",
          "cursor does not match the selected declaration",
        );
      }
      if (payload.qualified_symbol !== selected.qualified_symbol) {
        throw new CodeIndexingError(
          "INVALID_CURSOR",
          "cursor does not match the selected declaration",
        );
      }
      if (!sameStringList(payload.kinds, sortedKinds(kinds))) {
        throw new CodeIndexingError("INVALID_CURSOR", "cursor does not match reference filters");
      }
      if (payload.limit !== limit) {
        throw new CodeIndexingError("INVALID_CURSOR", "cursor does not match the page limit");
      }
      if (payload.operation_digest !== operationDigest) {
        throw new CodeIndexingError(
          "INVALID_CURSOR",
          "cursor does not match the refactor operation",
        );
      }
      offset = payload.offset;
      version = payload.version;
    }

    let records: ReferenceRecord[];
    try {
      // `schemaVersion` is pushed into the storage layer's `WHERE` clause (S4)
      // rather than fetched unfiltered and discarded here: a stale generation
      // written under an earlier `REFERENCE_SCHEMA_VERSION` -- left behind
      // because a reindex could not fully replace it -- would otherwise still
      // be materialized alongside the current generation's rows under their
      // old, since-discarded id scheme, just to be thrown away one line later.
      //
      // `recordKinds` likewise drops `declaration` rows from this fetch
      // (S4/E3): every classification need that reads a declaration is answered
      // below by a narrower, targeted query against the same pinned `version`,
      // so this call no longer has to pull the whole project's declaration
      // table into every page.
      records = await this.store.listReferenceRecords(selected.project_id, {
        version,
        schemaVersion: REFERENCE_SCHEMA_VERSION,
        recordKinds: ["reference", "coverage"],
      });
    } catch (error) {
      throw new CodeIndexingError(
        "STALE_CURSOR",
        "Reference cursor snapshot expired",
        {},
        {
          cause: error,
        },
      );
    }
    const root = await this.#projectRoot(selected.project_id);
    const sources = new Map<string, FileBytes>();
    const { hits, limitations } = await this.#hitsAndLimitations(
      selected,
      kinds,
      records,
      root,
      sources,
      options.backfill ?? null,
      version,
    );
    const page = hits.slice(offset, offset + limit);
    let nextCursor: string | null = null;
    if (offset + page.length < hits.length) {
      nextCursor = encodeCursor({
        version,
        project_id: selected.project_id,
        path: selected.path,
        qualified_symbol: selected.qualified_symbol,
        kinds: sortedKinds(kinds),
        offset: offset + page.length,
        limit,
        operation_digest: operationDigest,
      });
    }
    const unique = new Map<string, ReferenceLimitation>();
    for (const limitation of limitations) {
      unique.set(
        `${limitation.code}\0${limitation.explanation}\0${limitation.path ?? ""}`,
        limitation,
      );
    }
    const response: ReferenceResponse = {
      selected,
      hits: page,
      limitations: [...unique.values()].sort(
        (left, right) =>
          compare(left.code, right.code) || compare(left.path ?? "", right.path ?? ""),
      ),
      cursor: nextCursor,
      snapshot_version: version,
    };
    return { response, records, hits, root, sources };
  }

  /**
   * Classify every reference row into a sorted, unsliced hit list.
   *
   * Shared by `findReferences` (which then slices a page from the result) and
   * `analyzeRefactor` (which needs the full, unsliced list so counts and
   * completeness reflect the whole result set, not just the page currently
   * being returned -- R4).
   *
   * `records` no longer carries `declaration` rows (S4/E3) -- `version` pins the
   * same snapshot `records` was fetched from, so declarations are fetched here
   * from two narrow, targeted queries instead: one for exactly the files that
   * hold a candidate reference (never "every known file" -- that would just add
   * a redundant round trip back to the same data), and one for the single
   * target name the ambiguity check ever looks up.
   */
  async #hitsAndLimitations(
    selected: SelectedDeclaration,
    kinds: ReadonlySet<string> | null,
    records: readonly ReferenceRecord[],
    root: string | null,
    sources: Map<string, FileBytes>,
    backfill: ReferenceBackfillReport | null,
    version: number,
  ): Promise<{ hits: ReferenceHit[]; limitations: ReferenceLimitation[] }> {
    // Lexical/class-scope resolution only ever compares a declaration against a
    // reference row in the *same* file, so a file with no `reference`-kind row
    // of its own can never be looked up below -- narrowing to this set, rather
    // than every known file, is what makes this a real pushdown instead of a
    // redundant fetch of data already in `records`.
    const referenceFileIds = new Set(
      records.filter((row) => row.record_kind === "reference").map((row) => row.file_id),
    );
    const declarations = await this.store.declarationsForFiles(
      selected.project_id,
      referenceFileIds,
      { version, schemaVersion: REFERENCE_SCHEMA_VERSION },
    );
    // Precomputed once per query instead of scanned per row (E2):
    // `lexicalDeclaration` only ever wants declarations sharing a row's own
    // `file_id` and bare target name. Filtering `declarations` fresh for each of
    // the (potentially thousands of) reference rows made classification
    // O(reference rows x declaration rows); grouping once up front makes each
    // row's lookup O(1) plus the size of its own bucket.
    const declarationsByFileTarget = declarationsByFileAndTarget(declarations);
    // The ambiguity fallback only ever wants declarations sharing
    // `selected.symbol` as their own name, project-wide, to tell "exactly one
    // candidate anywhere" from "several" -- fetched directly instead of grouping
    // the whole declaration table by name only to read one bucket out of it.
    const targetCandidates = await this.store.targetNameCandidates(
      selected.project_id,
      selected.symbol,
      { recordKind: "declaration", version, schemaVersion: REFERENCE_SCHEMA_VERSION },
    );
    const imports = importsByFile(records);
    const reexportRows = reexportRowsByPath(records);
    const knownPaths = knownPathsOf(records);
    // A declaration nested directly in a class body is reachable only through a
    // receiver, so it must not shadow a bare name the way a nested function does.
    const classScopes = new Set(
      declarations
        .filter((row) => row.kind === "class" && row.source_qualified_symbol)
        .map((row) => row.source_qualified_symbol as string),
    );
    const hits: ReferenceHit[] = [];
    const limitations = coverageLimitations(records, backfill);
    // Serve-time staleness gate: coverage rows carry the content hash the
    // structural rows were extracted from, and offsets are applied to the file
    // as it sits on disk. A file whose bytes changed since extraction (e.g. a
    // failed replacement that retained its previous generation) would otherwise
    // have those old offsets served against text they never described -- a
    // wrong-edit hazard for callers that trust them.
    const coverageHashes = new Map<string, string>();
    for (const row of records) {
      if (row.record_kind === "coverage" && row.content_hash) {
        coverageHashes.set(row.path, row.content_hash);
      }
    }
    const digests = new Map<string, string>();
    const stalePaths = new Set<string>();
    for (const row of records) {
      if (
        row.record_kind !== "reference" ||
        !mayRefer(row, selected, imports, reexportRows, knownPaths)
      ) {
        continue;
      }
      if (kinds !== null && !kinds.has(row.kind ?? "")) continue;
      const lexical = lexicalDeclaration(row, declarationsByFileTarget, classScopes);
      if (
        lexical !== null &&
        row.file_id === selected.file_id &&
        lexical.source_qualified_symbol !== selected.qualified_symbol
      ) {
        continue;
      }
      const [resolution, reason, explanation] = classify(
        row,
        selected,
        targetCandidates,
        imports,
        reexportRows,
        knownPaths,
      );
      if (LIMITATION_REASONS.has(reason)) {
        limitations.push({ code: reason, explanation, path: row.path });
      }
      const [source, bom] = await readFileBytes(root, row.path, sources);
      if (!matchesCoverage(row.path, coverageHashes, source, bom, digests)) {
        stalePaths.add(row.path);
        continue;
      }
      const startByte = row.start_byte ?? 0;
      const endByte = row.end_byte ?? 0;
      hits.push({
        reference_id: row.reference_id,
        project_id: row.project_id,
        path: row.path,
        language: row.language,
        kind: (row.kind ?? "read") as ReferenceKind,
        start_line: row.start_line ?? 0,
        end_line: row.end_line ?? 0,
        // Offsets are reported against the file as it sits on disk. Extraction
        // works on BOM-stripped bytes, so a byte-order mark has to be added back
        // or every edit lands three bytes early.
        start_byte: startByte + bom,
        end_byte: endByte + bom,
        snippet: decodeLossy(source.subarray(startByte, endByte)),
        written_name: row.written_name,
        resolution: resolution as ResolutionLevel,
        reason_code: reason,
        explanation,
      });
    }
    const [selectedSource, selectedBom] = await readFileBytes(root, selected.path, sources);
    if (!matchesCoverage(selected.path, coverageHashes, selectedSource, selectedBom, digests)) {
      stalePaths.add(selected.path);
    }
    if (stalePaths.size > 0) {
      limitations.push({
        code: "stale_file",
        explanation:
          `${stalePaths.size} file(s) changed on disk after their structural ` +
          "rows were extracted, so offsets stored for them were suppressed as " +
          `stale: ${sample([...stalePaths])}`,
        path: null,
      });
    }
    hits.sort(
      (left, right) =>
        compare(left.path, right.path) ||
        left.start_line - right.start_line ||
        left.start_byte - right.start_byte ||
        compare(left.reference_id, right.reference_id),
    );
    return { hits, limitations };
  }

  async analyzeRefactor(
    selector: DeclarationSelector,
    operation: RefactorOperation,
    options: AnalyzeRefactorOptions = {},
  ): Promise<RefactorAnalysis> {
    const query = await this.#findReferencesWithRecords(selector, {
      limit: options.limit ?? 500,
      cursor: options.cursor ?? null,
      backfill: options.backfill ?? null,
      operationDigest: operationDigest(operation),
    });
    const { response, records, root, sources } = query;
    const isRename = operation.kind === "rename";
    if (isRename) validateRename(response.selected, operation as RenameOperation);

    let declarationFinding: RefactorFinding | null = null;
    let overrideFindings: RefactorFinding[] = [];
    if (isRename) {
      const renamed = await this.#renameFindings(
        response.selected,
        records,
        root,
        sources,
        response.snapshot_version,
      );
      declarationFinding = renamed.declaration;
      overrideFindings = renamed.overrides;
    }

    // Signature shapes are fetched once from the same pinned snapshot and
    // reused for every call-site classification.
    const oldShapes = isRename
      ? []
      : await this.store.declarationShapes(
          response.selected.project_id,
          response.selected.qualified_symbol,
          { version: response.snapshot_version, schemaVersion: REFERENCE_SCHEMA_VERSION },
        );
    const shapesById = new Map(records.map((row) => [row.reference_id, row]));
    const classified = await this.#classifyRefactorHits(
      response.selected,
      operation,
      query.hits,
      shapesById,
      oldShapes,
      root,
      sources,
    );
    let fullMust = classified.mustChange;
    let fullLikely = classified.likelyChange;
    const fullReview = classified.review;
    const fullEvidence = classified.evidence;

    const requiredEditSpans = new Set<string>();
    fullMust = dedupeEditSpans(fullMust, requiredEditSpans);

    // A declaration and export row may point at the same identifier. Keep one
    // concrete edit, but never merge findings whose edit location is unknown.
    if (
      declarationFinding !== null &&
      declarationFinding.edit_start_byte !== null &&
      declarationFinding.edit_end_byte !== null
    ) {
      const key = `${declarationFinding.path}\0${declarationFinding.edit_start_byte}\0${declarationFinding.edit_end_byte}`;
      if (
        fullMust.some(
          (finding) =>
            `${finding.path}\0${finding.edit_start_byte}\0${finding.edit_end_byte}` === key,
        )
      ) {
        declarationFinding = null;
      } else {
        requiredEditSpans.add(key);
      }
    }

    // Exact edits win over likely edits at the same location. Synthetic override
    // findings are considered before hit-derived likely findings because they
    // carry the more specific override reason.
    overrideFindings = dedupeEditSpans(overrideFindings, requiredEditSpans);
    fullLikely = dedupeEditSpans(fullLikely, requiredEditSpans);

    const pageIds = new Set(response.hits.map((hit) => hit.reference_id));
    let mustChange = fullMust.filter((finding) => pageIds.has(finding.reference_id));
    let likelyChange = fullLikely.filter((finding) => pageIds.has(finding.reference_id));
    const review = fullReview.filter((finding) => pageIds.has(finding.reference_id));
    const evidence = fullEvidence.filter((finding) => pageIds.has(finding.reference_id));
    if ((options.cursor ?? null) === null) {
      if (declarationFinding !== null) mustChange = [declarationFinding, ...mustChange];
      likelyChange = [...overrideFindings, ...likelyChange];
    }

    const limitations = response.limitations;
    return {
      selected: response.selected,
      operation,
      must_change: mustChange,
      likely_change: likelyChange,
      review,
      evidence,
      limitations,
      counts: {
        must_change: fullMust.length + (declarationFinding !== null ? 1 : 0),
        likely_change: fullLikely.length + overrideFindings.length,
        review: fullReview.length,
        evidence: fullEvidence.length,
      },
      cursor: response.cursor,
      completeness: completenessReport(limitations, fullReview, fullLikely, overrideFindings),
    };
  }

  async #renameFindings(
    selected: SelectedDeclaration,
    records: readonly ReferenceRecord[],
    root: string | null,
    sources: Map<string, FileBytes>,
    snapshotVersion: number,
  ): Promise<{ declaration: RefactorFinding; overrides: RefactorFinding[] }> {
    const declarations = await this.store.declarationShapes(
      selected.project_id,
      selected.qualified_symbol,
      { version: snapshotVersion, schemaVersion: REFERENCE_SCHEMA_VERSION },
    );
    const declaration = declarations.find((row) => row.file_id === selected.file_id) ?? null;
    let startByte = 0;
    let endByte = 0;
    let editStart: number | null = null;
    let editEnd: number | null = null;
    if (declaration !== null) {
      const [source, bom] = await readFileBytes(root, selected.path, sources);
      startByte = (declaration.start_byte ?? 0) + bom;
      endByte = (declaration.end_byte ?? 0) + bom;
      [editStart, editEnd] = editSpan(
        source,
        declaration.start_byte ?? 0,
        declaration.end_byte ?? 0,
        selected.symbol,
        bom,
      );
    }
    const finding: RefactorFinding = {
      reference_id: `declaration:${selected.file_id}`,
      project_id: selected.project_id,
      path: selected.path,
      language: selected.language,
      kind: "write",
      start_line: selected.start_line,
      end_line: selected.end_line,
      start_byte: startByte,
      end_byte: endByte,
      snippet: selected.symbol,
      resolution: "exact",
      reason_code: "declaration",
      explanation: "The selected declaration must be renamed.",
      written_name: selected.symbol,
      edit_required: true,
      edit_start_byte: editStart,
      edit_end_byte: editEnd,
    };
    const overrides = await this.#overrideFindings(
      selected,
      records,
      root,
      sources,
      snapshotVersion,
    );
    return { declaration: finding, overrides };
  }

  async #classifyRefactorHits(
    selected: SelectedDeclaration,
    operation: RefactorOperation,
    hits: readonly ReferenceHit[],
    shapesById: ReadonlyMap<string, ReferenceRecord>,
    oldShapes: readonly ReferenceRecord[],
    root: string | null,
    sources: Map<string, FileBytes>,
  ): Promise<ClassifiedFindings> {
    const mustChange: RefactorFinding[] = [];
    const likelyChange: RefactorFinding[] = [];
    const review: RefactorFinding[] = [];
    const evidence: RefactorFinding[] = [];
    for (const hit of hits) {
      const finding: RefactorFinding = {
        ...hit,
        edit_required: false,
        edit_start_byte: null,
        edit_end_byte: null,
      };
      if (hit.resolution === "unresolved") {
        review.push(finding);
        continue;
      }
      if (hit.resolution === "likely") {
        likelyChange.push({ ...finding, edit_required: true });
        continue;
      }
      if (operation.kind === "rename") {
        const written = hit.written_name || hit.snippet;
        const tail = written.slice(written.lastIndexOf(".") + 1);
        const needsEdit =
          tail === selected.symbol || hit.kind === "import" || hit.kind === "export";
        if (!needsEdit) {
          evidence.push(finding);
          continue;
        }
        const [source, bom] = await readFileBytes(root, hit.path, sources);
        const [editStart, editEnd] = editSpan(
          source,
          hit.start_byte - bom,
          hit.end_byte - bom,
          selected.symbol,
          bom,
        );
        mustChange.push({
          ...finding,
          edit_required: true,
          edit_start_byte: editStart,
          edit_end_byte: editEnd,
        });
        continue;
      }
      const issue = signatureIssue(
        selected,
        shapesById.get(hit.reference_id) ?? null,
        oldShapes,
        operation,
      );
      if (issue === "spread_uncertainty" || issue === "overload_ambiguity") {
        review.push({ ...finding, reason_code: issue, explanation: issueExplanation(issue) });
      } else if (issue !== null) {
        mustChange.push({
          ...finding,
          edit_required: true,
          reason_code: issue,
          explanation: issueExplanation(issue),
        });
      } else {
        evidence.push(finding);
      }
    }
    return { mustChange, likelyChange, review, evidence };
  }

  /**
   * Walk transitive subclasses of a renamed method's owner class.
   *
   * A rename of `Base.handle` should also flag `Child.handle` for review: the
   * override exists only as a declaration row today, never a reference
   * candidate, so a rename that touched only the base method and its callers
   * would silently leave the override's name stale. Dynamic dispatch means an
   * override can never be proven `exact` from syntax alone, so every finding
   * here is `likely_change`.
   *
   * `records` no longer carries `declaration` rows (S4/E3): both the base and
   * each override are looked up by their own exact `source_qualified_symbol`
   * from the same pinned `version` `records` was fetched from.
   */
  async #overrideFindings(
    selected: SelectedDeclaration,
    records: readonly ReferenceRecord[],
    root: string | null,
    sources: Map<string, FileBytes>,
    version: number,
  ): Promise<RefactorFinding[]> {
    if (!selected.qualified_symbol.includes(".")) return [];
    const separator = selected.qualified_symbol.lastIndexOf(".");
    const ownerSymbol = selected.qualified_symbol.slice(0, separator);
    const methodName = selected.qualified_symbol.slice(separator + 1);
    const ownerDeclarations = await this.store.declarationShapes(selected.project_id, ownerSymbol, {
      version,
      schemaVersion: REFERENCE_SCHEMA_VERSION,
    });
    const baseDeclaration =
      ownerDeclarations.find((row) => row.file_id === selected.file_id && row.kind === "class") ??
      null;
    if (baseDeclaration === null) return [];
    const imports = importsByFile(records);
    const knownPaths = knownPathsOf(records);
    const coverageHashes = new Map<string, string>();
    for (const row of records) {
      if (row.record_kind === "coverage" && row.content_hash) {
        coverageHashes.set(row.path, row.content_hash);
      }
    }
    const digests = new Map<string, string>();
    const inheritanceRows = records.filter(
      (row) => row.record_kind === "reference" && row.kind === "inheritance",
    );
    const findings: RefactorFinding[] = [];
    const visited = new Set([`${selected.file_id}\0${ownerSymbol}`]);
    const queue: Array<[fileId: string, qualified: string, path: string]> = [
      [selected.file_id, ownerSymbol, baseDeclaration.path],
    ];
    while (queue.length > 0) {
      const [baseFileId, baseQualified, basePath] = queue.shift() as [string, string, string];
      const baseTail = baseQualified.slice(baseQualified.lastIndexOf(".") + 1);
      for (const row of inheritanceRows) {
        if (!inheritanceTargets(row, baseFileId, baseTail, basePath, imports, knownPaths)) continue;
        const subclassQualified = row.source_qualified_symbol;
        if (!subclassQualified) continue;
        const key = `${row.file_id}\0${subclassQualified}`;
        if (visited.has(key)) continue;
        visited.add(key);
        queue.push([row.file_id, subclassQualified, row.path]);
        const overrideSymbol = `${subclassQualified}.${methodName}`;
        const overrideCandidates = await this.store.declarationShapes(
          selected.project_id,
          overrideSymbol,
          { version, schemaVersion: REFERENCE_SCHEMA_VERSION },
        );
        const overrideDeclaration =
          overrideCandidates.find((row2) => row2.file_id === row.file_id) ?? null;
        if (overrideDeclaration === null) continue;
        const [source, bom] = await readFileBytes(root, row.path, sources);
        if (!matchesCoverage(row.path, coverageHashes, source, bom, digests)) continue;
        const startByte = (overrideDeclaration.start_byte ?? 0) + bom;
        const endByte = (overrideDeclaration.end_byte ?? 0) + bom;
        const [editStart, editEnd] = editSpan(
          source,
          overrideDeclaration.start_byte ?? 0,
          overrideDeclaration.end_byte ?? 0,
          methodName,
          bom,
        );
        findings.push({
          reference_id: `override:${overrideDeclaration.file_id}:${overrideSymbol}`,
          project_id: row.project_id,
          path: row.path,
          language: row.language,
          kind: "write",
          start_line: overrideDeclaration.start_line ?? 0,
          end_line: overrideDeclaration.end_line ?? 0,
          start_byte: startByte,
          end_byte: endByte,
          snippet: methodName,
          written_name: methodName,
          resolution: "likely",
          reason_code: "override_of_renamed_method",
          explanation:
            `${subclassQualified} overrides the renamed method; dynamic ` +
            "dispatch means this cannot be proven structurally.",
          edit_required: true,
          edit_start_byte: editStart,
          edit_end_byte: editEnd,
        });
      }
    }
    return findings;
  }

  async #select(selector: DeclarationSelector): Promise<SelectedDeclaration> {
    if (selector.chunk_id !== null) {
      const chunk = await this.store.getChunk(selector.chunk_id);
      if (chunk === null || chunk.symbol === null || chunk.qualified_symbol === null) {
        throw new CodeIndexingError(
          "AMBIGUOUS_SYMBOL",
          `chunk_id ${selector.chunk_id} is not a declaration chunk; chunk ids come ` +
            "from find_symbol or search_code results and change when a file is reindexed",
        );
      }
      return {
        project_id: chunk.project_id,
        file_id: chunk.file_id,
        path: chunk.path,
        language: chunk.language,
        symbol: chunk.symbol,
        qualified_symbol: chunk.qualified_symbol,
        kind: chunk.kind,
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        chunk_id: chunk.chunk_id,
      };
    }
    const project = selector.project as string;
    const selectorPath = selector.path as string;
    const qualifiedSymbol = selector.qualified_symbol as string;
    const indexed = await this.store.listChunks([project]);
    const chunks = indexed.filter(
      (chunk) => chunk.path === selectorPath && chunk.qualified_symbol === qualifiedSymbol,
    );
    if (chunks.length > 1) {
      throw new CodeIndexingError(
        "AMBIGUOUS_SYMBOL",
        `${qualifiedSymbol} matches ${chunks.length} declarations in ` +
          `${selectorPath}; select one by chunk_id`,
        {
          project,
          candidates: chunks.map((chunk) => ({
            chunk_id: chunk.chunk_id,
            path: chunk.path,
            qualified_symbol: chunk.qualified_symbol,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
          })),
        },
      );
    }
    if (chunks.length === 0) {
      // Distinguish a typo from a symbol this project genuinely lacks: "no
      // declaration" and "no references" are different answers.
      const tail = qualifiedSymbol.slice(qualifiedSymbol.lastIndexOf(".") + 1);
      const near = [
        ...new Set(
          indexed
            .map((chunk) => chunk.qualified_symbol)
            .filter(
              (symbol): symbol is string =>
                symbol !== null && symbol.slice(symbol.lastIndexOf(".") + 1) === tail,
            ),
        ),
      ].sort(compare);
      throw new CodeIndexingError(
        "AMBIGUOUS_SYMBOL",
        `No declaration ${qualifiedSymbol} in ${selectorPath}`,
        { project, path: selectorPath, candidates: near.slice(0, MAX_LIMITATION_PATHS) },
      );
    }
    const located = chunks[0] as (typeof chunks)[number];
    if (located.symbol === null || located.qualified_symbol === null) {
      throw new CodeIndexingError(
        "AMBIGUOUS_SYMBOL",
        `${qualifiedSymbol} in ${selectorPath} is not a declaration`,
        { project, path: selectorPath },
      );
    }
    // The project id is the selector's own; chunk rows no longer carry it.
    return {
      project_id: project,
      file_id: located.file_id,
      path: located.path,
      language: located.language,
      symbol: located.symbol,
      qualified_symbol: located.qualified_symbol,
      kind: located.kind,
      start_line: located.start_line,
      end_line: located.end_line,
      chunk_id: located.chunk_id,
    };
  }

  async #projectRoot(projectId: string): Promise<string | null> {
    const projects = await this.store.listProjects();
    return projects.find((item) => item.id === projectId)?.root ?? null;
  }
}

// --- classification -------------------------------------------------------

function importsByFile(records: readonly ReferenceRecord[]): Map<string, ReferenceRecord[]> {
  const result = new Map<string, ReferenceRecord[]>();
  for (const row of records) {
    if (row.record_kind !== "reference" || row.kind !== "import") continue;
    const bucket = result.get(row.file_id);
    if (bucket === undefined) result.set(row.file_id, [row]);
    else bucket.push(row);
  }
  return result;
}

/**
 * Declarations grouped by their own `(file_id, target_name)` (E2).
 *
 * `lexicalDeclaration` only ever considers declarations sharing a reference
 * row's `file_id` and bare target name -- filtering the full declaration list
 * against those two fields for every reference row made classification
 * O(reference rows x declaration rows). Grouping once turns each row's lookup
 * into an O(1) map access plus however many declarations actually share that
 * file and name (typically very few).
 */
function declarationsByFileAndTarget(
  declarations: readonly ReferenceRecord[],
): Map<string, ReferenceRecord[]> {
  const result = new Map<string, ReferenceRecord[]>();
  for (const declaration of declarations) {
    const key = `${declaration.file_id}\0${declaration.target_name ?? ""}`;
    const bucket = result.get(key);
    if (bucket === undefined) result.set(key, [declaration]);
    else bucket.push(declaration);
  }
  return result;
}

/**
 * Every file path this query's snapshot has at least one row for.
 *
 * Used to walk a Python absolute import's `__init__.py` chain and find its real
 * package root (a `src/` layout or another package sub-root) without filesystem
 * access. Every indexed file gets a `coverage` row even when it has zero
 * declarations or references, so this is the full universe of paths the
 * resolver can see -- not just the ones with structural content.
 */
function knownPathsOf(records: readonly ReferenceRecord[]): ReadonlySet<string> {
  return new Set(records.map((row) => row.path));
}

/**
 * Import/export rows keyed by their own file's path (R2).
 *
 * A re-export chain is walked file-by-file through `module_path` resolution,
 * which yields a path, not a `file_id` -- unlike `importsByFile`, this also
 * includes `export` rows, since a barrel's re-export (`export { b } from
 * './impl'`, or Python's `from .impl import b` inside `pkg/__init__.py`) is the
 * edge being followed.
 */
function reexportRowsByPath(records: readonly ReferenceRecord[]): Map<string, ReferenceRecord[]> {
  const result = new Map<string, ReferenceRecord[]>();
  for (const row of records) {
    if (row.record_kind !== "reference") continue;
    if (row.kind !== "import" && row.kind !== "export") continue;
    const bucket = result.get(row.path);
    if (bucket === undefined) result.set(row.path, [row]);
    else bucket.push(row);
  }
  return result;
}

function mayRefer(
  row: ReferenceRecord,
  selected: SelectedDeclaration,
  imports: ReadonlyMap<string, ReferenceRecord[]>,
  reexportRows: ReadonlyMap<string, ReferenceRecord[]>,
  knownPaths: ReadonlySet<string>,
): boolean {
  const targetTail = tail(row.target_name ?? "");
  const writtenTail = tail(row.written_name ?? "");
  if (
    selected.symbol === row.target_name ||
    selected.symbol === row.written_name ||
    selected.symbol === targetTail ||
    selected.symbol === writtenTail
  ) {
    return true;
  }
  if (row.kind === "import" && isNamespaceImport(row)) return false;
  const spelling = row.receiver_text || row.written_name;
  return (imports.get(row.file_id) ?? []).some(
    (item) =>
      (item.alias || item.written_name || item.imported_name) === spelling &&
      (importTargets(item, selected, knownPaths) ||
        reexportTargetsSymbol(item, selected.symbol, selected.path, reexportRows, knownPaths)),
  );
}

function lexicalDeclaration(
  row: ReferenceRecord,
  declarationsByFileTarget: ReadonlyMap<string, ReferenceRecord[]>,
  classScopes: ReadonlySet<string>,
): ReferenceRecord | null {
  if (row.receiver_text !== null) return null;
  if (row.kind !== "call" && row.kind !== "read" && row.kind !== "write") return null;
  const source = row.source_qualified_symbol ?? "";
  const target = tail(row.target_name ?? "");
  let best: ReferenceRecord | null = null;
  let bestDepth = -1;
  for (const declaration of declarationsByFileTarget.get(`${row.file_id}\0${target}`) ?? []) {
    const qualified = declaration.source_qualified_symbol ?? "";
    const scope = qualified.includes(".") ? qualified.slice(0, qualified.lastIndexOf(".")) : "";
    // Python and JS/TS both leave the class body out of a method's scope chain:
    // inside `Gate.run`, a bare `helper()` binds to the module-level `helper`,
    // never to the sibling method `Gate.helper`. Treating the class as an
    // enclosing scope silently dropped those call sites from every result.
    if (scope && classScopes.has(scope)) continue;
    if (!scope || source === scope || source.startsWith(`${scope}.`)) {
      const depth = countDots(scope) + (scope ? 1 : 0);
      // `max` in Python keeps the *first* maximum, so ties go to the earliest.
      if (depth > bestDepth) {
        bestDepth = depth;
        best = declaration;
      }
    }
  }
  return best;
}

function classify(
  row: ReferenceRecord,
  selected: SelectedDeclaration,
  targetCandidates: readonly ReferenceRecord[],
  imports: ReadonlyMap<string, ReferenceRecord[]>,
  reexportRows: ReadonlyMap<string, ReferenceRecord[]>,
  knownPaths: ReadonlySet<string>,
): [resolution: string, reason: string, explanation: string] {
  const sourceImports = imports.get(row.file_id) ?? [];
  if (sourceImports.some((item) => item.imported_name === "*" && !isNamespaceImport(item))) {
    return ["unresolved", "wildcard_import", "A wildcard import can bind this name dynamically."];
  }
  if (row.receiver_text !== null) {
    const receiver = row.receiver_text;
    if (
      (receiver === "self" || receiver === "cls" || receiver === "this") &&
      sameOwner(row, selected)
    ) {
      return ["exact", "known_owner_member", "The receiver is the declaration's enclosing owner."];
    }
    for (const item of sourceImports) {
      const binding = item.alias || item.written_name || item.imported_name;
      if (binding === receiver && importTargets(item, selected, knownPaths)) {
        return ["exact", "known_namespace_member", "The receiver is a known imported namespace."];
      }
    }
    return [
      "likely",
      "unknown_receiver",
      "Receiver type inference is outside this syntax-only index.",
    ];
  }
  let unprovenReexport = false;
  for (const item of sourceImports) {
    const alias = item.alias || item.written_name || item.imported_name;
    if (alias !== row.written_name) continue;
    if (importTargets(item, selected, knownPaths)) {
      return ["exact", "direct_import_alias", "The local alias directly imports this declaration."];
    }
    if (reexportTargetsSymbol(item, selected.symbol, selected.path, reexportRows, knownPaths)) {
      return [
        "exact",
        "reexport_chain",
        "The local alias resolves to the declaration through a chain of re-exports.",
      ];
    }
    if (item.module_path !== null) unprovenReexport = true;
  }
  if (unprovenReexport) {
    return [
      "likely",
      "unproven_reexport",
      "The local alias imports from a module, but the chain of re-exports " +
        "to the declaration's file could not be proven.",
    ];
  }
  if (row.file_id === selected.file_id) {
    return ["exact", "same_file_symbol", "The call is in the declaration's source file."];
  }
  // `targetCandidates` is already the project-wide set of declarations sharing
  // `selected.symbol`, so no further filtering by name is needed here.
  if (targetCandidates.length === 1) {
    return [
      "likely",
      "name_only_candidate",
      "The name is unique, but no import or owner proves binding.",
    ];
  }
  return [
    "unresolved",
    "ambiguous_symbol",
    "Multiple declarations or scopes could bind this name.",
  ];
}

function importTargets(
  item: ReferenceRecord,
  selected: SelectedDeclaration,
  knownPaths: ReadonlySet<string>,
): boolean {
  return importTargetsSymbol(item, selected.symbol, selected.path, knownPaths);
}

function importTargetsSymbol(
  item: ReferenceRecord,
  symbol: string,
  targetPath: string,
  knownPaths: ReadonlySet<string>,
): boolean {
  const imported = item.imported_name;
  if (
    !isNamespaceImport(item) &&
    imported !== symbol &&
    imported !== "default" &&
    imported !== null
  ) {
    return false;
  }
  if (item.module_path === null) return false;
  return moduleMatches(item.path, item.language, item.module_path, targetPath, knownPaths);
}

/**
 * Prove a binding through a chain of barrel re-exports/imports (R2).
 *
 * `importTargetsSymbol` proves only a *direct* edge: the import's module
 * resolves straight to the declaration's own file. A barrel (`pkg/__init__.py`
 * doing `from .impl import b`, or `pkg/index.ts` doing `export { b } from
 * './impl'`) sits between the importer and the declaration; this walks such
 * indirections one module-edge at a time. Each hop must be a real, resolvable
 * `import`/`export` row binding the exact name the previous hop asked for --
 * never a guess -- so an unrelated same-named symbol down a different chain can
 * never bind (the corpus hard gate). `visited` (keyed by resolved path plus the
 * name being chased there) and `depth` prevent chasing a cycle or a pathological
 * fan-out forever.
 */
function reexportTargetsSymbol(
  item: ReferenceRecord,
  symbol: string,
  targetPath: string,
  rowsByPath: ReadonlyMap<string, ReferenceRecord[]>,
  knownPaths: ReadonlySet<string>,
  visited: ReadonlySet<string> = new Set(),
  depth = 0,
): boolean {
  if (importTargetsSymbol(item, symbol, targetPath, knownPaths)) return true;
  if (depth >= MAX_REEXPORT_DEPTH || isNamespaceImport(item)) return false;
  if (item.module_path === null) return false;
  const imported = item.imported_name;
  const lookupName = imported === null || imported === "default" ? symbol : imported;
  for (const candidate of moduleCandidates(
    item.path,
    item.language,
    item.module_path,
    knownPaths,
  )) {
    const key = `${candidate}\0${lookupName}`;
    if (visited.has(key)) continue;
    const nextVisited = new Set(visited).add(key);
    for (const hop of rowsByPath.get(candidate) ?? []) {
      const binding = hop.alias || hop.written_name || hop.imported_name;
      if (binding !== lookupName) continue;
      if (
        reexportTargetsSymbol(
          hop,
          symbol,
          targetPath,
          rowsByPath,
          knownPaths,
          nextVisited,
          depth + 1,
        )
      ) {
        return true;
      }
    }
  }
  return false;
}

function isNamespaceImport(item: ReferenceRecord): boolean {
  return (
    (item.imported_name === "*" && item.alias !== null) ||
    (item.language === "python" && item.imported_name === null && item.module_path !== null)
  );
}

function sameOwner(row: ReferenceRecord, selected: SelectedDeclaration): boolean {
  const source = row.source_qualified_symbol ?? "";
  const owner = selected.qualified_symbol.includes(".")
    ? selected.qualified_symbol.slice(0, selected.qualified_symbol.lastIndexOf("."))
    : "";
  return Boolean(owner) && (source === owner || source.startsWith(`${owner}.`));
}

/**
 * True if an `inheritance` row's base name binds to the class at hand.
 *
 * A same-file reference needs no import to bind, but the written name must
 * still be the base class's own name -- otherwise every class in the file would
 * match, not just the ones that actually extend it. A cross-file reference must
 * go through an import in the referring file that resolves to the base class's
 * own file, mirroring `importTargets`; that import's *local* binding (its
 * alias, when aliased) is what has to equal the written name here, since there
 * is no alias to consult for a same-file reference (no import exists).
 */
function inheritanceTargets(
  row: ReferenceRecord,
  baseFileId: string,
  baseTail: string,
  basePath: string,
  imports: ReadonlyMap<string, ReferenceRecord[]>,
  knownPaths: ReadonlySet<string>,
): boolean {
  const writtenTarget = row.target_name ?? "";
  const target = tail(writtenTarget);
  if (!writtenTarget) return false;
  if (row.file_id === baseFileId && !writtenTarget.includes(".")) {
    return writtenTarget === baseTail;
  }
  for (const item of imports.get(row.file_id) ?? []) {
    const binding = item.alias || item.written_name || item.imported_name;
    if (writtenTarget.includes(".")) {
      const separator = writtenTarget.lastIndexOf(".");
      const receiver = writtenTarget.slice(0, separator);
      const member = writtenTarget.slice(separator + 1);
      if (member !== baseTail || binding !== receiver || !isNamespaceImport(item)) continue;
    } else if (
      binding !== target ||
      (item.imported_name !== baseTail && item.imported_name !== "default")
    ) {
      continue;
    }
    if (
      item.module_path !== null &&
      moduleMatches(item.path, item.language, item.module_path, basePath, knownPaths)
    ) {
      return true;
    }
  }
  return false;
}

// --- signature analysis ---------------------------------------------------

function signatureIssue(
  selected: SelectedDeclaration,
  row: ReferenceRecord | null,
  oldShapes: readonly ReferenceRecord[],
  operation: SignatureChangeOperation,
): string | null {
  // Call and declaration shapes come from one pinned snapshot. The caller
  // fetches `oldShapes` once and reuses them for every hit.
  if (row === null || row.shape_json === null) return null;
  const shape = JSON.parse(row.shape_json) as {
    has_positional_spread?: boolean;
    has_keyword_spread?: boolean;
    positional_count?: number;
    keywords?: string[];
  };
  if (shape.has_positional_spread || shape.has_keyword_spread) return "spread_uncertainty";
  const first = oldShapes[0];
  if (oldShapes.length !== 1 || first === undefined || first.shape_json === null) {
    return oldShapes.length > 1 ? "overload_ambiguity" : null;
  }
  const oldParameters = JSON.parse(first.shape_json) as unknown;
  if (!Array.isArray(oldParameters)) return null;
  const oldRecords = oldParameters.filter(
    (parameter): parameter is Record<string, unknown> =>
      typeof parameter === "object" && parameter !== null && !Array.isArray(parameter),
  );
  let newPositional = operation.parameters.filter(
    (parameter) => parameter.kind === "positional" || parameter.kind === "positional_only",
  );
  const positionalCount = Number(shape.positional_count ?? 0);
  const keywords = new Set(shape.keywords ?? []);
  const newByName = new Map(operation.parameters.map((parameter) => [parameter.name, parameter]));
  const boundReceiver =
    selected.language === "python" &&
    selected.kind === "method" &&
    (row.receiver_text === "self" || row.receiver_text === "cls");
  if (boundReceiver) newPositional = newPositional.slice(1);
  if (
    operation.parameters.some(
      (parameter) =>
        parameter.required && parameter.kind === "keyword_only" && !keywords.has(parameter.name),
    )
  ) {
    return "missing_required_parameter";
  }
  for (const keyword of keywords) {
    const parameter = newByName.get(keyword);
    if (parameter === undefined) return "invalid_keyword";
    if (parameter.kind === "positional_only") return "parameter_mode_change";
  }
  if (
    newPositional.some(
      (parameter, position) =>
        parameter.required && position >= positionalCount && !keywords.has(parameter.name),
    )
  ) {
    return "missing_required_parameter";
  }
  const oldPositionalAll = (): Record<string, unknown>[] => {
    const rows = oldRecords.filter(
      (parameter) => parameter.kind === "positional" || parameter.kind === "positional_only",
    );
    return boundReceiver ? rows.slice(1) : rows;
  };
  if (
    positionalCount > newPositional.length &&
    !operation.parameters.some((parameter) => parameter.kind === "variadic")
  ) {
    const oldPositional = oldPositionalAll();
    for (const positionalParameter of oldPositional.slice(newPositional.length, positionalCount)) {
      const oldName = positionalParameter.name;
      const newParameter = typeof oldName === "string" ? newByName.get(oldName) : undefined;
      if (newParameter !== undefined && newParameter.kind === "keyword_only") {
        return "parameter_mode_change";
      }
    }
    return "removed_positional_parameter";
  }
  const oldPositional = oldPositionalAll();
  const oldPositions = new Map<string, number>();
  oldPositional.forEach((parameter, position) => {
    // Last occurrence wins, as Python's dict comprehension does. A signature
    // cannot normally repeat a parameter name, but a malformed stored shape
    // can, and keeping the first instead would compare against a different
    // position than the original does.
    if (typeof parameter.name === "string") oldPositions.set(parameter.name, position);
  });
  const compared: ParameterShape[] = newPositional.slice(0, positionalCount);
  for (const [position, parameter] of compared.entries()) {
    const oldParameter = position < oldPositional.length ? oldPositional[position] : undefined;
    if (oldParameter === undefined) continue;
    const oldName = oldParameter.name;
    const existing = oldPositions.get(parameter.name);
    if (oldName !== parameter.name && existing !== undefined && existing !== position) {
      return "positional_order_change";
    }
  }
  return null;
}

function issueExplanation(issue: string): string {
  const explanations: Record<string, string> = {
    missing_required_parameter: "This call omits a required proposed parameter.",
    invalid_keyword: "This call uses a keyword absent from the proposed signature.",
    parameter_mode_change: "This call is incompatible with a proposed parameter mode.",
    removed_positional_parameter: "This call supplies a removed positional parameter.",
    positional_order_change: "This call depends on a positional parameter order that changes.",
    spread_uncertainty: "A spread argument prevents a deterministic compatibility check.",
    overload_ambiguity: "Multiple declaration shapes prevent a deterministic comparison.",
  };
  return explanations[issue] as string;
}

// --- rename validation ----------------------------------------------------

/** Python's `keyword.kwlist`, which `str.isidentifier()` deliberately excludes. */
const PYTHON_KEYWORDS: ReadonlySet<string> = new Set([
  "False",
  "None",
  "True",
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "try",
  "while",
  "with",
  "yield",
]);

/**
 * Python's `str.isidentifier()`.
 *
 * Unicode-aware, not ASCII: `café` is a legal Python name, and rejecting it
 * would refuse a rename the language allows. The character classes are exactly
 * the ones the language reference names.
 */
const PYTHON_IDENTIFIER = /^[\p{XID_Start}_]\p{XID_Continue}*$/u;
const JS_IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

function validateRename(selected: SelectedDeclaration, operation: RenameOperation): void {
  const valid =
    selected.language === "python"
      ? PYTHON_IDENTIFIER.test(operation.new_name) && !PYTHON_KEYWORDS.has(operation.new_name)
      : JS_IDENTIFIER.test(operation.new_name);
  if (!valid) {
    throw new CodeIndexingError("INVALID_REFACTOR", "Invalid identifier for rename");
  }
}

// --- coverage and completeness --------------------------------------------

function completenessReport(
  limitations: readonly ReferenceLimitation[],
  review: readonly RefactorFinding[],
  likelyChange: readonly RefactorFinding[],
  overrideFindings: readonly RefactorFinding[],
): CompletenessReport {
  if (limitations.some((item) => COVERAGE_GAP_CODES.has(item.code))) {
    return {
      state: "incomplete",
      explanation:
        "Some files could not be analyzed, so this list may omit real uses. See limitations.",
    };
  }
  if (
    limitations.length > 0 ||
    review.length > 0 ||
    likelyChange.length > 0 ||
    overrideFindings.length > 0
  ) {
    return {
      state: "complete_with_dynamic_limitations",
      explanation:
        "Every indexed file was analyzed, but some uses could not be proven " +
        "without type information. See likely_change and review.",
    };
  }
  return {
    state: "complete",
    explanation: "All indexed structural candidates were considered.",
  };
}

/**
 * Report files that hold no structural rows because none could be made.
 *
 * Coverage rows prove a file was parsed under the current schema. A file whose
 * language has no reference query still gets one, so without this the caller
 * cannot tell "searched and found nothing" from "never looked".
 */
function coverageLimitations(
  records: readonly ReferenceRecord[],
  backfill: ReferenceBackfillReport | null,
): ReferenceLimitation[] {
  const limitations: ReferenceLimitation[] = [];
  const unanalyzed = new Map<string, string[]>();
  for (const row of records) {
    if (row.record_kind !== "coverage") continue;
    if (STRUCTURAL_LANGUAGES.has(row.language)) continue;
    const bucket = unanalyzed.get(row.language);
    if (bucket === undefined) unanalyzed.set(row.language, [row.path]);
    else bucket.push(row.path);
  }
  for (const language of [...unanalyzed.keys()].sort(compare)) {
    const paths = unanalyzed.get(language) as string[];
    limitations.push({
      code: "unsupported_language",
      explanation:
        `${paths.length} ${language} file(s) are indexed for search but have no ` +
        "structural reference extraction, so uses of this declaration in them " +
        `are invisible here: ${sample(paths)}`,
      path: null,
    });
  }
  if (backfill !== null) {
    for (const [code, paths] of [
      ["parse_error", backfill.incomplete_paths],
      ["stale_file", backfill.stale_paths],
    ] as const) {
      if (paths.length === 0) continue;
      const reason =
        code === "parse_error"
          ? "could not be parsed, so their references are missing"
          : "changed after they were indexed, so their references may be stale";
      limitations.push({
        code,
        explanation: `${paths.length} file(s) ${reason}: ${sample(paths)}`,
        path: null,
      });
    }
  }
  return limitations;
}

function sample(paths: readonly string[]): string {
  const shown = [...paths].sort(compare).slice(0, MAX_LIMITATION_PATHS);
  const remainder = paths.length - shown.length;
  return shown.join(", ") + (remainder > 0 ? `, and ${remainder} more` : "");
}

// --- file access ----------------------------------------------------------

/**
 * Whether a file's current bytes still match what its rows were extracted from.
 *
 * Reference offsets are applied to the file as it sits on disk, so rows whose
 * extraction hash no longer matches those bytes would be served against text
 * they never described -- a wrong-edit hazard for callers that trust the offsets
 * (a failed replacement deliberately retains its previous generation's rows).
 * Digests are memoized per path because a file can hold many candidate rows. A
 * file with no coverage row has nothing to validate against and is trusted.
 * `source` is BOM-stripped, so the marker is added back before hashing:
 * extraction hashed the raw bytes, marker included.
 */
function matchesCoverage(
  filePath: string,
  coverageHashes: ReadonlyMap<string, string>,
  source: Uint8Array,
  bom: number,
  digests: Map<string, string>,
): boolean {
  const expected = coverageHashes.get(filePath);
  if (expected === undefined) return true;
  let value = digests.get(filePath);
  if (value === undefined) {
    const raw = bom ? concatBytes(BOM, source) : source;
    value = digest(raw);
    digests.set(filePath, value);
  }
  return value === expected;
}

/**
 * One file's BOM-stripped bytes and the offset that was removed.
 *
 * Reads are cached for the life of one query. Resolving a few hundred
 * references used to re-read the same file once per hit.
 */
async function readFileBytes(
  root: string | null,
  filePath: string,
  cache: Map<string, FileBytes>,
): Promise<FileBytes> {
  const cached = cache.get(filePath);
  if (cached !== undefined) return cached;
  let raw = new Uint8Array(0);
  if (root !== null) {
    try {
      const buffer = await fs.promises.readFile(path.join(root, filePath));
      raw = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    } catch {
      raw = new Uint8Array(0);
    }
  }
  const offset = startsWithBom(raw) ? BOM.length : 0;
  const entry: FileBytes = [raw.subarray(offset), offset];
  cache.set(filePath, entry);
  return entry;
}

/**
 * Locate the identifier to rewrite inside a reference's own range.
 *
 * The stored range covers the whole occurrence, which is wider than the name
 * for a qualified call (`auth.authorize`) and for an aliased import
 * (`authorize as check`). Replacing the whole range would drop the module
 * qualifier or the alias, so the exact identifier is located instead and
 * anything ambiguous is left for a human.
 *
 * The search runs on bytes, not text: the offsets it returns are byte offsets
 * into the file, and the `latin1` decoding below is what keeps a string index
 * and a byte index the same number even for a file with non-ASCII content.
 */
function editSpan(
  source: Uint8Array,
  startByte: number,
  endByte: number,
  name: string,
  bom: number,
): [number | null, number | null] {
  const span = Buffer.from(source.subarray(startByte, endByte)).toString("latin1");
  if (span.length === 0 || name.length === 0) return [null, null];
  const needle = Buffer.from(name, "utf8").toString("latin1");
  // No `u` flag: `\w` must mean the ASCII set Python's bytes patterns mean,
  // and the haystack is a byte string rather than text.
  const pattern = new RegExp(`(?<![\\w$])${escapeRegExp(needle)}(?![\\w$])`, "g");
  const matches = [...span.matchAll(pattern)];
  if (matches.length !== 1) return [null, null];
  const match = matches[0] as RegExpMatchArray & { index: number };
  return [startByte + match.index + bom, startByte + match.index + needle.length + bom];
}

// --- module path resolution -----------------------------------------------

/**
 * The directory an absolute import from a file in `directory` resolves against.
 *
 * Python resolves an absolute import from whatever is on `sys.path`, which this
 * syntax-only index never sees directly -- but the parent of the topmost
 * directory in `directory`'s unbroken `__init__.py` chain (its own package, and
 * its package's package, ...) is exactly that directory for any regular
 * (non-namespace) package, `src/`-layout included: `mypkg/a.py` with
 * `mypkg/__init__.py` anchors at the project root; `src/mypkg/a.py` with both
 * `src/mypkg/__init__.py` and no `src/__init__.py` anchors at `src`.
 *
 * A directory with no `__init__.py` at all -- either a flat top-level layout
 * (nothing to walk) or a PEP 420 namespace package (no marker file exists to
 * find) -- falls back to the project root. That is already correct for the flat
 * case, and merely non-exact rather than wrong for a namespace package nested
 * under a further sub-root: it never fabricates a false candidate, which is the
 * property this function exists to protect (S2).
 */
function pythonPackageRoot(
  directoryParts: readonly string[],
  knownPaths: ReadonlySet<string>,
): string[] {
  if (
    directoryParts.length === 0 ||
    !knownPaths.has(joinParts([...directoryParts, "__init__.py"]))
  ) {
    // `directory` itself is not a package (no `__init__.py` of its own) --
    // either there is nothing to walk (a flat top-level layout) or it is a
    // namespace package, which leaves no marker file to find its boundary from.
    // Root-anchoring is exactly right for the flat case and a safe
    // non-fabricating fallback for the namespace one; it must not become
    // `directory` itself, or this collapses back to the sibling-anchor bug this
    // whole function exists to fix.
    return [];
  }
  let boundary = directoryParts.length;
  while (
    boundary > 0 &&
    knownPaths.has(joinParts([...directoryParts.slice(0, boundary), "__init__.py"]))
  ) {
    boundary -= 1;
  }
  return [...directoryParts.slice(0, boundary)];
}

/**
 * Every file path a relative `module_path` from `sourcePath` could mean.
 *
 * Shared by {@link moduleMatches} (a yes/no check against one known target) and
 * {@link reexportTargetsSymbol} (which needs the actual resolved path(s) to keep
 * walking a re-export chain).
 */
function moduleCandidates(
  sourcePath: string,
  language: string,
  modulePath: string,
  knownPaths: ReadonlySet<string>,
): Set<string> {
  const sourceParts = pathParts(sourcePath);
  const parentParts = sourceParts.slice(0, -1);
  if (language === "python") {
    const dots = modulePath.length - modulePath.replace(/^\.+/, "").length;
    const suffix = modulePath.slice(dots);
    const stemParts = suffix ? pathParts(suffix.split(".").join("/")) : [];
    let base: string[];
    if (dots === 0) {
      // An absolute import is resolved from the package root, not from the
      // importing file's own directory -- see `pythonPackageRoot` for how that
      // root is found without filesystem access (a `src/` layout or another
      // package sub-root is common, so it is not always the project root).
      // Blindly anchoring at the parent would let a sibling file bind falsely
      // (e.g. `mypkg/utils.py` for `from utils import f` when the real target is
      // the top-level `utils.py`), which is exactly why this cannot simply walk
      // every ancestor as an equally-plausible candidate.
      base = pythonPackageRoot(parentParts, knownPaths);
    } else {
      base = parentParts.slice(0, Math.max(0, parentParts.length - (dots - 1)));
    }
    return new Set([
      joinParts([...base, ...pathParts(`${displayPath(stemParts)}.py`)]),
      joinParts([...base, ...stemParts, "__init__.py"]),
    ]);
  }
  if (!modulePath.startsWith(".")) return new Set();
  // Walk `modulePath`'s segments against an explicit stack rather than simply
  // appending them to the parent: a `..` segment must pop the last resolved
  // directory, not survive as a literal path component (a pure path never
  // resolves `..` on its own, so `src/app/../utils` never string-equals
  // `src/utils`). A `..` with nothing left to pop would escape the project
  // root, which no in-project target can ever match, so that yields no
  // candidates.
  const parts = [...parentParts];
  for (const part of pathParts(modulePath)) {
    if (part === ".") continue;
    if (part === "..") {
      if (parts.length === 0) return new Set();
      parts.pop();
      continue;
    }
    parts.push(part);
  }
  const normalized = displayPath(parts);
  const extensions = [".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"];
  const candidates = new Set<string>();
  for (const extension of extensions) {
    candidates.add(joinParts(pathParts(`${normalized}${extension}`)));
    candidates.add(joinParts([...parts, `index${extension}`]));
  }
  return candidates;
}

function moduleMatches(
  sourcePath: string,
  language: string,
  modulePath: string,
  targetPath: string,
  knownPaths: ReadonlySet<string>,
): boolean {
  return moduleCandidates(sourcePath, language, modulePath, knownPaths).has(
    joinParts(pathParts(targetPath)),
  );
}

// --- cursors --------------------------------------------------------------

/**
 * A short, stable fingerprint of a refactor operation's full shape.
 *
 * Bound into the cursor so a page-2 `analyzeRefactor` call is rejected if the
 * caller supplies a different `new_name` or a different signature spec than
 * page 1 used -- without this, page 2 would silently classify hits against a
 * different operation than the one whose findings the caller already saw on
 * page 1.
 */
function operationDigest(operation: RefactorOperation): string {
  return digest(canonicalJson(operation)).slice(0, 16);
}

function encodeCursor(payload: CursorPayload): string {
  return Buffer.from(canonicalJson(payload), "utf8").toString("base64url").replace(/=+$/, "");
}

/**
 * The exact key set every cursor payload must carry (T2).
 *
 * Anything else is rejected outright rather than letting a missing key surface
 * later as a bare property access on undefined -- the defect that made a
 * well-formed but foreign cursor leak an internal message straight to the
 * client.
 */
const CURSOR_FIELDS: readonly string[] = [
  "kinds",
  "limit",
  "offset",
  "operation_digest",
  "path",
  "project_id",
  "qualified_symbol",
  "version",
];

function decodeCursor(cursor: string): CursorPayload {
  const invalid = (): CodeIndexingError =>
    new CodeIndexingError("INVALID_CURSOR", "invalid reference cursor");
  let payload: unknown;
  try {
    payload = JSON.parse(Buffer.from(cursor, "base64url").toString("utf8"));
  } catch (error) {
    throw new CodeIndexingError("INVALID_CURSOR", "invalid reference cursor", {}, { cause: error });
  }
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) throw invalid();
  const record = payload as Record<string, unknown>;
  if (!sameStringList(Object.keys(record).sort(compare), CURSOR_FIELDS)) throw invalid();
  for (const field of ["version", "offset", "limit"] as const) {
    const value = record[field];
    if (typeof value !== "number" || !Number.isInteger(value)) throw invalid();
  }
  for (const field of ["project_id", "path", "qualified_symbol"] as const) {
    if (typeof record[field] !== "string") throw invalid();
  }
  const kinds = record.kinds;
  if (!Array.isArray(kinds) || !kinds.every((item) => typeof item === "string")) throw invalid();
  const operation = record.operation_digest;
  if (operation !== null && typeof operation !== "string") throw invalid();
  return {
    version: record.version as number,
    project_id: record.project_id as string,
    path: record.path as string,
    qualified_symbol: record.qualified_symbol as string,
    kinds: kinds as string[],
    offset: record.offset as number,
    limit: record.limit as number,
    operation_digest: operation as string | null,
  };
}

/**
 * `json.dumps(..., separators=(",", ":"), sort_keys=True)`.
 *
 * Cursors and operation digests are compared against themselves across
 * requests, so the serialization has to be canonical rather than merely valid:
 * `JSON.stringify` preserves insertion order, which would make two equal
 * operations digest differently depending on how they were constructed.
 */
function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, item]) => item !== undefined)
    .sort(([left], [right]) => compare(left, right));
  return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`).join(",")}}`;
}

// --- small helpers --------------------------------------------------------

function digest(value: string | Uint8Array): string {
  return createHash("sha256")
    .update(typeof value === "string" ? Buffer.from(value, "utf8") : value)
    .digest("hex");
}

/** Python's `PurePosixPath(value).parts` for a relative path. */
function pathParts(value: string): string[] {
  return value.split("/").filter((part) => part !== "" && part !== ".");
}

/** A canonical joined path; the empty path is the empty string, never ".". */
function joinParts(parts: readonly string[]): string {
  return parts.filter((part) => part !== "" && part !== ".").join("/");
}

/**
 * Python's `str(PurePosixPath(...))`, which renders the empty path as ".".
 *
 * Only used where the original interpolates a path into a string
 * (`f"{stem}.py"`), where that "." is observable: a bare `from . import x`
 * produces a candidate literally named `.py`. Reproduced rather than tidied,
 * because a candidate set that differs from Python's is a resolution that
 * differs from Python's.
 */
function displayPath(parts: readonly string[]): string {
  return parts.length === 0 ? "." : parts.join("/");
}

function tail(value: string): string {
  return value.slice(value.lastIndexOf(".") + 1);
}

function countDots(value: string): number {
  let count = 0;
  for (let index = value.indexOf("."); index !== -1; index = value.indexOf(".", index + 1)) {
    count += 1;
  }
  return count;
}

function sortedKinds(kinds: ReadonlySet<string> | null): string[] {
  return [...(kinds ?? [])].sort(compare);
}

function sameStringList(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}

/** Python's `<` on `str`: a code-unit comparison, which `localeCompare` is not. */
function compare(left: string, right: string): number {
  if (left === right) return 0;
  return left < right ? -1 : 1;
}

function startsWithBom(value: Uint8Array): boolean {
  return value.length >= 3 && value[0] === 0xef && value[1] === 0xbb && value[2] === 0xbf;
}

function concatBytes(left: Uint8Array, right: Uint8Array): Uint8Array {
  const result = new Uint8Array(left.length + right.length);
  result.set(left, 0);
  result.set(right, left.length);
  return result;
}

/** `bytes.decode("utf-8", errors="replace")`, which is TextDecoder's default. */
function decodeLossy(value: Uint8Array): string {
  return new TextDecoder("utf-8").decode(value);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&");
}
