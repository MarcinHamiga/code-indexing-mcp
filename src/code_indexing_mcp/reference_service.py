"""Conservative, syntax-only reference classification over structural rows."""

from __future__ import annotations

import base64
import hashlib
import json
import keyword as keyword_module
import re
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, NamedTuple, cast

from .errors import CodeIndexingError, ErrorCode
from .extractor import STRUCTURAL_LANGUAGES
from .indexing import REFERENCE_SCHEMA_VERSION, _digest
from .models import (
    CompletenessReport,
    DeclarationSelector,
    RefactorAnalysis,
    RefactorCounts,
    RefactorFinding,
    RefactorOperation,
    ReferenceBackfillReport,
    ReferenceHit,
    ReferenceLimitation,
    ReferenceResponse,
    RenameOperation,
    SelectedDeclaration,
    SignatureChangeOperation,
)
from .storage import LanceStore, ReferenceRecord

# Reason codes that describe something the syntax-only index could not see.
# They are surfaced as limitations whatever resolution level they carry, so a
# caller never reads an empty limitation list as proof of full coverage.
# `unproven_reexport` is included deliberately (unlike `name_only_candidate`):
# a barrel import proves a module edge exists even when the chain cannot be
# resolved fully, which is stronger evidence than an unqualified bare name.
_LIMITATION_REASONS: Final = frozenset(
    {"wildcard_import", "unknown_receiver", "ambiguous_symbol", "unproven_reexport"}
)

# Bound re-export traversal so malformed or unusually deep barrel chains are
# reported as unproven rather than walked indefinitely.
_MAX_REEXPORT_DEPTH: Final = 4

# Limitations that mean whole files were never analyzed. Any of them forces the
# completeness state to "incomplete" rather than merely "dynamic limitations".
_COVERAGE_GAP_CODES: Final = frozenset({"unsupported_language", "parse_error", "stale_file"})

_BOM: Final = b"\xef\xbb\xbf"

# How many individual paths a coverage limitation names before it summarizes.
_MAX_LIMITATION_PATHS: Final = 10


class _ReferenceQuery(NamedTuple):
    """Pinned-snapshot data shared by reference and refactor responses.

    The full hit list and source cache let refactor analysis compute page-independent
    counts without repeating classification. Records contain reference and coverage
    rows; declarations are fetched separately through narrowed queries.
    """

    response: ReferenceResponse
    records: list[ReferenceRecord]
    hits: list[ReferenceHit]
    root: Path | None
    sources: dict[str, tuple[bytes, int]]


class _ClassifiedFindings(NamedTuple):
    must_change: list[RefactorFinding]
    likely_change: list[RefactorFinding]
    review: list[RefactorFinding]
    evidence: list[RefactorFinding]


def _dedupe_edit_spans(
    findings: list[RefactorFinding],
    seen: set[tuple[str, int, int]],
) -> list[RefactorFinding]:
    """Return findings whose concrete edit spans have not already been seen."""
    deduped: list[RefactorFinding] = []
    for finding in findings:
        if finding.edit_start_byte is None or finding.edit_end_byte is None:
            deduped.append(finding)
            continue
        key = (finding.path, finding.edit_start_byte, finding.edit_end_byte)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


class ReferenceService:
    """Resolve only syntax facts that select a declaration unambiguously."""

    def __init__(self, store: LanceStore) -> None:
        self.store = store

    def find_references(
        self,
        selector: DeclarationSelector,
        *,
        kinds: set[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
        backfill: ReferenceBackfillReport | None = None,
        operation_digest: str | None = None,
    ) -> ReferenceResponse:
        """Resolve references for `selector`, one page at a time.

        `operation_digest` is internal to `analyze_refactor`; it binds the
        operation to the cursor so later pages cannot silently change it.
        """
        return self._find_references_with_records(
            selector,
            kinds=kinds,
            limit=limit,
            cursor=cursor,
            backfill=backfill,
            operation_digest=operation_digest,
        ).response

    def _find_references_with_records(
        self,
        selector: DeclarationSelector,
        *,
        kinds: set[str] | None = None,
        limit: int = 100,
        cursor: str | None = None,
        backfill: ReferenceBackfillReport | None = None,
        operation_digest: str | None = None,
    ) -> _ReferenceQuery:
        """`find_references`'s body, also returning everything it computed along the way.

        `analyze_refactor` needs the paginated response, the full
        pinned-snapshot record set, and the full classified hit list (for the
        declaration/override/signature work below), not just the page. Fetching
        through the plain `find_references` and then re-fetching
        `list_reference_records` at the same `response.snapshot_version` --
        and separately re-running `_hits_and_limitations` over it -- was a
        second full-table materialization and a second classification pass
        over data already in hand (S4/E1); this lets `analyze_refactor` reuse
        the one fetch and one classification pass made here instead.
        """
        if limit < 1 or limit > 500:
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, "limit must be between 1 and 500")
        cursor_payload: dict[str, object] | None = None
        if cursor is not None:
            cursor_payload = self._decode_cursor(cursor)
            self._require_active_cursor_slot(str(cursor_payload["project_id"]), cursor_payload)
        selected = self._select(selector)
        if selected.language not in STRUCTURAL_LANGUAGES:
            raise CodeIndexingError(
                ErrorCode.UNSUPPORTED_LANGUAGE,
                f"Structural references are not extracted for {selected.language}. "
                f"Supported languages are {', '.join(sorted(STRUCTURAL_LANGUAGES))}.",
                project=selected.project_id,
                path=selected.path,
                language=selected.language,
            )
        if not self.store.has_reference_table(selected.project_id):
            # A missing table and a legitimately empty one both read as `[]`
            # from `list_reference_records`/`reference_version` (S5). Trusting
            # that silence would report "no references" for a project whose
            # reference index was never built -- e.g. a partition indexed
            # before this feature existed, or one where `ensure_reference_index`
            # was skipped -- instead of surfacing the real, actionable state.
            raise CodeIndexingError(
                ErrorCode.REFERENCE_INDEX_UNAVAILABLE,
                "The reference index has not been built for this project. "
                "Run ensure_reference_index (or reindex) before querying references.",
                project=selected.project_id,
            )
        version = self.store.reference_version(selected.project_id)
        offset = 0
        if cursor_payload is not None:
            payload = cursor_payload
            if payload["project_id"] != selected.project_id or payload["path"] != selected.path:
                raise CodeIndexingError(
                    ErrorCode.INVALID_CURSOR, "cursor does not match the selected declaration"
                )
            if payload["qualified_symbol"] != selected.qualified_symbol:
                raise CodeIndexingError(
                    ErrorCode.INVALID_CURSOR, "cursor does not match the selected declaration"
                )
            if payload["kinds"] != sorted(kinds or set()):
                raise CodeIndexingError(
                    ErrorCode.INVALID_CURSOR, "cursor does not match reference filters"
                )
            if payload["limit"] != limit:
                raise CodeIndexingError(
                    ErrorCode.INVALID_CURSOR, "cursor does not match the page limit"
                )
            if payload["operation_digest"] != operation_digest:
                raise CodeIndexingError(
                    ErrorCode.INVALID_CURSOR, "cursor does not match the refactor operation"
                )
            offset = cast(int, payload["offset"])
            version = cast(int, payload["version"])

        try:
            # `schema_version` is pushed into the storage layer's `WHERE`
            # clause (S4) rather than fetched unfiltered and discarded here:
            # a stale generation written under an earlier
            # `REFERENCE_SCHEMA_VERSION` -- left behind because a reindex
            # could not fully replace it -- would otherwise still be
            # materialized (and decoded into Python objects) alongside the
            # current generation's rows under their old, since-discarded id
            # scheme, just to be thrown away one line later. Only rows at the
            # current schema version are ever a real answer.
            #
            # `record_kinds` likewise drops `declaration` rows from this
            # fetch (S4/E3): every classification need that reads a
            # declaration is answered below by a narrower, targeted query
            # (`declarations_for_files`/`target_name_candidates`) against the
            # same pinned `version`, so this call no longer has to pull the
            # whole project's declaration table into every page.
            records = self.store.list_reference_records(
                selected.project_id,
                version=version,
                schema_version=REFERENCE_SCHEMA_VERSION,
                record_kinds=("reference", "coverage"),
            )
        except (FileNotFoundError, ValueError) as error:
            raise CodeIndexingError(
                ErrorCode.STALE_CURSOR, "Reference cursor snapshot expired"
            ) from error
        root = self._project_root(selected.project_id)
        sources: dict[str, tuple[bytes, int]] = {}
        hits, limitations = self._hits_and_limitations(
            selected, kinds, records, root, sources, backfill, version
        )
        page = hits[offset : offset + limit]
        next_cursor = None
        if offset + len(page) < len(hits):
            pointer = self.store.active_pointer(selected.project_id)
            next_cursor = self._encode_cursor(
                {
                    "version": version,
                    "project_id": selected.project_id,
                    "path": selected.path,
                    "qualified_symbol": selected.qualified_symbol,
                    "kinds": sorted(kinds or set()),
                    "offset": offset + len(page),
                    "limit": limit,
                    "operation_digest": operation_digest,
                    "slot_id": "" if pointer is None else pointer[0].slot_id,
                    "activation_epoch": 0 if pointer is None else pointer[1],
                }
            )
        unique_limitations = {
            (limitation.code, limitation.explanation, limitation.path): limitation
            for limitation in limitations
        }
        response = ReferenceResponse(
            selected=selected,
            hits=page,
            limitations=sorted(
                unique_limitations.values(), key=lambda item: (item.code, item.path or "")
            ),
            cursor=next_cursor,
            snapshot_version=version,
        )
        return _ReferenceQuery(response, records, hits, root, sources)

    def _hits_and_limitations(
        self,
        selected: SelectedDeclaration,
        kinds: set[str] | None,
        records: list[ReferenceRecord],
        root: Path | None,
        sources: dict[str, tuple[bytes, int]],
        backfill: ReferenceBackfillReport | None,
        version: int,
    ) -> tuple[list[ReferenceHit], list[ReferenceLimitation]]:
        """Classify every reference row into a sorted, unsliced hit list.

        Shared by `find_references` (which then slices a page from the
        result) and `analyze_refactor` (which needs the full, unsliced list
        so counts/completeness reflect the whole result set, not just the
        page currently being returned — R4).

        `records` no longer carries `declaration` rows (S4/E3) -- `version`
        pins the same snapshot `records` was fetched from, so declarations
        are fetched here from two narrow, targeted queries instead:
        `declarations_for_files` for exactly the files that hold a
        candidate reference (never "every known file" -- that would just
        add a redundant round trip back to the same data), and
        `target_name_candidates` for the one target name `_classify`'s
        ambiguity check ever looks up.
        """

        # `_lexical_declaration`/class-scope resolution only ever compares a
        # declaration against a reference row in the *same* file, so a file
        # with no `reference`-kind row of its own can never be looked up
        # below -- narrowing to this set, rather than every known file,
        # is what makes this a real pushdown instead of a redundant fetch of
        # data already in `records`.
        reference_file_ids = {
            row["file_id"] for row in records if row["record_kind"] == "reference"
        }
        declarations = self.store.declarations_for_files(
            selected.project_id,
            reference_file_ids,
            version=version,
            schema_version=REFERENCE_SCHEMA_VERSION,
        )
        # Precomputed once per query instead of scanned per row (E2):
        # `_lexical_declaration` only ever wants declarations sharing a
        # row's own `file_id` and bare target name. Filtering `declarations`
        # fresh for each of the (potentially thousands of) reference rows
        # made classification O(reference_rows x declaration_rows); grouping
        # once up front makes each row's lookup O(1) plus the size of its
        # own bucket.
        declarations_by_file_target = self._declarations_by_file_target(declarations)
        # `_classify`'s ambiguity fallback only ever wants declarations
        # sharing `selected.symbol` as their own name, project-wide, to tell
        # "exactly one candidate anywhere" from "several" -- fetched
        # directly instead of grouping the whole declaration table by name
        # only to read one bucket out of it.
        target_candidates = self.store.target_name_candidates(
            selected.project_id,
            selected.symbol,
            record_kind="declaration",
            version=version,
            schema_version=REFERENCE_SCHEMA_VERSION,
        )
        imports = self._imports_by_file(records)
        reexport_rows = self._reexport_rows_by_path(records)
        known_paths = self._known_paths(records)
        # A declaration nested directly in a class body is reachable only
        # through a receiver, so it must not shadow a bare name the way a
        # nested function does.
        class_scopes = {
            scope
            for row in declarations
            if row["kind"] == "class" and (scope := row["source_qualified_symbol"])
        }
        hits: list[ReferenceHit] = []
        limitations: list[ReferenceLimitation] = self._coverage_limitations(records, backfill)
        # Serve-time staleness gate: coverage rows carry the content hash the
        # structural rows were extracted from, and offsets are applied to the
        # file as it sits on disk. A file whose bytes changed since extraction
        # (e.g. a failed replacement that retained its previous generation)
        # would otherwise have those old offsets served against text they
        # never described -- a wrong-edit hazard for callers that trust them.
        coverage_hashes = {
            row["path"]: row["content_hash"]
            for row in records
            if row["record_kind"] == "coverage" and row["content_hash"]
        }
        digests: dict[str, str] = {}
        stale_paths: set[str] = set()
        for row in records:
            if row["record_kind"] != "reference" or not self._may_refer(
                row, selected, imports, reexport_rows, known_paths
            ):
                continue
            if kinds is not None and row["kind"] not in kinds:
                continue
            lexical = self._lexical_declaration(row, declarations_by_file_target, class_scopes)
            if (
                lexical is not None
                and row["file_id"] == selected.file_id
                and lexical["source_qualified_symbol"] != selected.qualified_symbol
            ):
                continue
            resolution, reason, explanation = self._classify(
                row, selected, target_candidates, imports, reexport_rows, known_paths
            )
            if reason in _LIMITATION_REASONS:
                limitations.append(
                    ReferenceLimitation(code=reason, explanation=explanation, path=row["path"])
                )
            source, bom = self._file_bytes(root, row["path"], sources)
            if not self._matches_coverage(row["path"], coverage_hashes, source, bom, digests):
                stale_paths.add(row["path"])
                continue
            start_byte = row["start_byte"] or 0
            end_byte = row["end_byte"] or 0
            hits.append(
                ReferenceHit(
                    reference_id=row["reference_id"],
                    project_id=row["project_id"],
                    path=row["path"],
                    language=row["language"],
                    kind=cast(
                        Literal[
                            "import",
                            "export",
                            "call",
                            "type_use",
                            "inheritance",
                            "decorator",
                            "read",
                            "write",
                        ],
                        row["kind"] or "read",
                    ),
                    start_line=row["start_line"] or 0,
                    end_line=row["end_line"] or 0,
                    # Offsets are reported against the file as it sits on disk.
                    # Extraction works on BOM-stripped bytes, so a byte-order
                    # mark has to be added back or every edit lands three bytes
                    # early.
                    start_byte=start_byte + bom,
                    end_byte=end_byte + bom,
                    snippet=source[start_byte:end_byte].decode("utf-8", errors="replace"),
                    written_name=row["written_name"],
                    resolution=cast(Literal["exact", "likely", "unresolved"], resolution),
                    reason_code=reason,
                    explanation=explanation,
                )
            )
        selected_source, selected_bom = self._file_bytes(root, selected.path, sources)
        if not self._matches_coverage(
            selected.path, coverage_hashes, selected_source, selected_bom, digests
        ):
            stale_paths.add(selected.path)
        if stale_paths:
            limitations.append(
                ReferenceLimitation(
                    code="stale_file",
                    explanation=(
                        f"{len(stale_paths)} file(s) changed on disk after their structural "
                        "rows were extracted, so offsets stored for them were suppressed as "
                        f"stale: {self._sample(sorted(stale_paths))}"
                    ),
                )
            )
        hits.sort(key=lambda hit: (hit.path, hit.start_line, hit.start_byte, hit.reference_id))
        return hits, limitations

    def analyze_refactor(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        *,
        limit: int = 500,
        cursor: str | None = None,
        backfill: ReferenceBackfillReport | None = None,
    ) -> RefactorAnalysis:
        query = self._find_references_with_records(
            selector,
            limit=limit,
            cursor=cursor,
            backfill=backfill,
            operation_digest=self._operation_digest(operation),
        )
        response, records, root, sources = (
            query.response,
            query.records,
            query.root,
            query.sources,
        )
        if isinstance(operation, RenameOperation):
            self._validate_rename(response.selected, operation)

        declaration_finding: RefactorFinding | None = None
        override_findings: list[RefactorFinding] = []
        if isinstance(operation, RenameOperation):
            declaration_finding, override_findings = self._rename_findings(
                response.selected,
                records,
                root,
                sources,
                response.snapshot_version,
            )

        # Signature shapes are fetched once from the same pinned snapshot and reused
        # for every call-site classification.
        old_shapes = (
            []
            if isinstance(operation, RenameOperation)
            else self.store.declaration_shapes(
                response.selected.project_id,
                response.selected.qualified_symbol,
                version=response.snapshot_version,
                schema_version=REFERENCE_SCHEMA_VERSION,
            )
        )
        shapes_by_id = {row["reference_id"]: row for row in records}
        classified = self._classify_refactor_hits(
            response.selected,
            operation,
            query.hits,
            shapes_by_id,
            old_shapes,
            root,
            sources,
        )
        full_must = classified.must_change
        full_likely = classified.likely_change
        full_review = classified.review
        full_evidence = classified.evidence

        required_edit_spans: set[tuple[str, int, int]] = set()
        full_must = _dedupe_edit_spans(full_must, required_edit_spans)

        # A declaration and export row may point at the same identifier. Keep one
        # concrete edit, but never merge findings whose edit location is unknown.
        if (
            declaration_finding is not None
            and declaration_finding.edit_start_byte is not None
            and declaration_finding.edit_end_byte is not None
        ):
            dedupe_key = (
                declaration_finding.path,
                declaration_finding.edit_start_byte,
                declaration_finding.edit_end_byte,
            )
            if any(
                (finding.path, finding.edit_start_byte, finding.edit_end_byte) == dedupe_key
                for finding in full_must
            ):
                declaration_finding = None
            else:
                required_edit_spans.add(dedupe_key)

        # Exact edits win over likely edits at the same location. Synthetic
        # override findings are considered before hit-derived likely findings
        # because they carry the more specific override reason.
        override_findings = _dedupe_edit_spans(override_findings, required_edit_spans)
        full_likely = _dedupe_edit_spans(full_likely, required_edit_spans)

        page_ids = {hit.reference_id for hit in response.hits}
        must_change = [finding for finding in full_must if finding.reference_id in page_ids]
        likely_change = [finding for finding in full_likely if finding.reference_id in page_ids]
        review = [finding for finding in full_review if finding.reference_id in page_ids]
        evidence = [finding for finding in full_evidence if finding.reference_id in page_ids]
        if cursor is None:
            if declaration_finding is not None:
                must_change = [declaration_finding, *must_change]
            likely_change = [*override_findings, *likely_change]

        limitations = response.limitations
        counts = RefactorCounts(
            must_change=len(full_must) + (1 if declaration_finding is not None else 0),
            likely_change=len(full_likely) + len(override_findings),
            review=len(full_review),
            evidence=len(full_evidence),
        )
        return RefactorAnalysis(
            selected=response.selected,
            operation=operation,
            must_change=must_change,
            likely_change=likely_change,
            review=review,
            evidence=evidence,
            limitations=limitations,
            counts=counts,
            cursor=response.cursor,
            completeness=self._completeness_report(
                limitations,
                full_review,
                full_likely,
                override_findings,
            ),
        )

    @staticmethod
    def _validate_rename(selected: SelectedDeclaration, operation: RenameOperation) -> None:
        valid_identifier = (
            operation.new_name.isidentifier() and not keyword_module.iskeyword(operation.new_name)
            if selected.language == "python"
            else bool(re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", operation.new_name))
        )
        if not valid_identifier:
            raise CodeIndexingError(ErrorCode.INVALID_REFACTOR, "Invalid identifier for rename")

    def _rename_findings(
        self,
        selected: SelectedDeclaration,
        records: list[ReferenceRecord],
        root: Path | None,
        sources: dict[str, tuple[bytes, int]],
        snapshot_version: int,
    ) -> tuple[RefactorFinding, list[RefactorFinding]]:
        declarations = self.store.declaration_shapes(
            selected.project_id,
            selected.qualified_symbol,
            version=snapshot_version,
            schema_version=REFERENCE_SCHEMA_VERSION,
        )
        declaration = next(
            (row for row in declarations if row["file_id"] == selected.file_id),
            None,
        )
        start_byte, end_byte = 0, 0
        edit_start, edit_end = None, None
        if declaration is not None:
            source, bom = self._file_bytes(root, selected.path, sources)
            start_byte = (declaration["start_byte"] or 0) + bom
            end_byte = (declaration["end_byte"] or 0) + bom
            edit_start, edit_end = self._edit_span(
                source,
                declaration["start_byte"] or 0,
                declaration["end_byte"] or 0,
                selected.symbol,
                bom,
            )
        finding = RefactorFinding(
            reference_id=f"declaration:{selected.file_id}",
            project_id=selected.project_id,
            path=selected.path,
            language=selected.language,
            kind="write",
            start_line=selected.start_line,
            end_line=selected.end_line,
            start_byte=start_byte,
            end_byte=end_byte,
            snippet=selected.symbol,
            resolution="exact",
            reason_code="declaration",
            explanation="The selected declaration must be renamed.",
            written_name=selected.symbol,
            edit_required=True,
            edit_start_byte=edit_start,
            edit_end_byte=edit_end,
        )
        overrides = self._override_findings(selected, records, root, sources, snapshot_version)
        return finding, overrides

    def _classify_refactor_hits(
        self,
        selected: SelectedDeclaration,
        operation: RefactorOperation,
        hits: list[ReferenceHit],
        shapes_by_id: dict[str, ReferenceRecord],
        old_shapes: list[ReferenceRecord],
        root: Path | None,
        sources: dict[str, tuple[bytes, int]],
    ) -> _ClassifiedFindings:
        must_change: list[RefactorFinding] = []
        likely_change: list[RefactorFinding] = []
        review: list[RefactorFinding] = []
        evidence: list[RefactorFinding] = []
        for hit in hits:
            finding = RefactorFinding(**hit.model_dump(), edit_required=False)
            if hit.resolution == "unresolved":
                review.append(finding)
                continue
            if hit.resolution == "likely":
                likely_change.append(finding.model_copy(update={"edit_required": True}))
                continue
            if isinstance(operation, RenameOperation):
                written = hit.written_name or hit.snippet
                needs_edit = written.rsplit(".", 1)[-1] == selected.symbol or hit.kind in {
                    "import",
                    "export",
                }
                if not needs_edit:
                    evidence.append(finding)
                    continue
                source, bom = self._file_bytes(root, hit.path, sources)
                edit_start, edit_end = self._edit_span(
                    source,
                    hit.start_byte - bom,
                    hit.end_byte - bom,
                    selected.symbol,
                    bom,
                )
                must_change.append(
                    finding.model_copy(
                        update={
                            "edit_required": True,
                            "edit_start_byte": edit_start,
                            "edit_end_byte": edit_end,
                        }
                    )
                )
                continue
            issue = self._signature_issue(
                selected,
                shapes_by_id.get(hit.reference_id),
                old_shapes,
                operation,
            )
            if issue in {"spread_uncertainty", "overload_ambiguity"}:
                review.append(
                    finding.model_copy(
                        update={
                            "reason_code": issue,
                            "explanation": self._issue_explanation(issue),
                        }
                    )
                )
            elif issue is not None:
                must_change.append(
                    finding.model_copy(
                        update={
                            "edit_required": True,
                            "reason_code": issue,
                            "explanation": self._issue_explanation(issue),
                        }
                    )
                )
            else:
                evidence.append(finding)
        return _ClassifiedFindings(must_change, likely_change, review, evidence)

    @staticmethod
    def _completeness_report(
        limitations: list[ReferenceLimitation],
        review: list[RefactorFinding],
        likely_change: list[RefactorFinding],
        override_findings: list[RefactorFinding],
    ) -> CompletenessReport:
        if any(item.code in _COVERAGE_GAP_CODES for item in limitations):
            return CompletenessReport(
                state="incomplete",
                explanation=(
                    "Some files could not be analyzed, so this list may omit real uses. "
                    "See limitations."
                ),
            )
        if limitations or review or likely_change or override_findings:
            return CompletenessReport(
                state="complete_with_dynamic_limitations",
                explanation=(
                    "Every indexed file was analyzed, but some uses could not be proven "
                    "without type information. See likely_change and review."
                ),
            )
        return CompletenessReport(
            state="complete",
            explanation="All indexed structural candidates were considered.",
        )

    def _signature_issue(
        self,
        selected: SelectedDeclaration,
        row: ReferenceRecord | None,
        old_shapes: list[ReferenceRecord],
        operation: SignatureChangeOperation,
    ) -> str | None:
        # Call and declaration shapes come from one pinned snapshot. The caller
        # fetches `old_shapes` once and reuses them for every hit.
        if row is None or row["shape_json"] is None:
            return None
        shape = json.loads(row["shape_json"])
        if shape.get("has_positional_spread") or shape.get("has_keyword_spread"):
            return "spread_uncertainty"
        if len(old_shapes) != 1 or old_shapes[0]["shape_json"] is None:
            return "overload_ambiguity" if len(old_shapes) > 1 else None
        old_parameters = json.loads(old_shapes[0]["shape_json"])
        if not isinstance(old_parameters, list):
            return None
        old_records = [
            cast(dict[str, Any], parameter)
            for parameter in old_parameters
            if isinstance(parameter, dict)
        ]
        new_positional = [
            parameter
            for parameter in operation.parameters
            if parameter.kind in {"positional", "positional_only"}
        ]
        positional_count = int(shape.get("positional_count", 0))
        keywords = set(shape.get("keywords", []))
        new_by_name = {parameter.name: parameter for parameter in operation.parameters}
        bound_receiver = (
            selected.language == "python"
            and selected.kind == "method"
            and row["receiver_text"] in {"self", "cls"}
        )
        if bound_receiver:
            new_positional = new_positional[1:]
        if any(
            parameter.required
            and parameter.kind == "keyword_only"
            and parameter.name not in keywords
            for parameter in operation.parameters
        ):
            return "missing_required_parameter"
        for keyword in keywords:
            parameter = new_by_name.get(keyword)
            if parameter is None:
                return "invalid_keyword"
            if parameter.kind == "positional_only":
                return "parameter_mode_change"
        if any(
            parameter.required and position >= positional_count and parameter.name not in keywords
            for position, parameter in enumerate(new_positional)
        ):
            return "missing_required_parameter"
        if positional_count > len(new_positional) and not any(
            parameter.kind == "variadic" for parameter in operation.parameters
        ):
            old_positional = [
                parameter
                for parameter in old_records
                if parameter.get("kind") in {"positional", "positional_only"}
            ]
            if bound_receiver:
                old_positional = old_positional[1:]
            for positional_parameter in old_positional[len(new_positional) : positional_count]:
                old_name = positional_parameter.get("name")
                new_parameter = new_by_name.get(old_name) if isinstance(old_name, str) else None
                if new_parameter is not None and new_parameter.kind == "keyword_only":
                    return "parameter_mode_change"
            return "removed_positional_parameter"
        old_positional = [
            parameter
            for parameter in old_records
            if parameter.get("kind") in {"positional", "positional_only"}
        ]
        if bound_receiver:
            old_positional = old_positional[1:]
        old_positions = {
            name: position
            for position, parameter in enumerate(old_positional)
            if isinstance(name := parameter.get("name"), str)
        }
        for position, parameter in enumerate(new_positional[:positional_count]):
            old_parameter: object = (
                old_positional[position] if position < len(old_positional) else None
            )
            if not isinstance(old_parameter, dict):
                continue
            old_name = old_parameter.get("name")
            if old_name != parameter.name and old_positions.get(parameter.name) not in {
                None,
                position,
            }:
                return "positional_order_change"
        return None

    def _override_findings(
        self,
        selected: SelectedDeclaration,
        records: list[ReferenceRecord],
        root: Path | None,
        sources: dict[str, tuple[bytes, int]],
        version: int,
    ) -> list[RefactorFinding]:
        """Walk transitive subclasses of a renamed method's owner class.

        A rename of `Base.handle` should also flag `Child.handle` for review:
        the override exists only as a declaration row today, never a
        reference candidate, so a rename that touched only the base method
        and its callers would silently leave the override's name stale.
        Dynamic dispatch means an override can never be proven `exact` from
        syntax alone, so every finding here is `likely_change`.

        `records` no longer carries `declaration` rows (S4/E3): both
        `base_decl` and each `override_decl` below are looked up by their own
        exact `source_qualified_symbol` via `declaration_shapes`, from the
        same pinned `version` `records` was fetched from.
        """

        if "." not in selected.qualified_symbol:
            return []
        owner_symbol, method_name = selected.qualified_symbol.rsplit(".", 1)
        owner_declarations = self.store.declaration_shapes(
            selected.project_id,
            owner_symbol,
            version=version,
            schema_version=REFERENCE_SCHEMA_VERSION,
        )
        base_decl = next(
            (
                row
                for row in owner_declarations
                if row["file_id"] == selected.file_id and row["kind"] == "class"
            ),
            None,
        )
        if base_decl is None:
            return []
        imports = self._imports_by_file(records)
        known_paths = self._known_paths(records)
        coverage_hashes = {
            row["path"]: row["content_hash"]
            for row in records
            if row["record_kind"] == "coverage" and row["content_hash"]
        }
        digests: dict[str, str] = {}
        inheritance_rows = [
            row
            for row in records
            if row["record_kind"] == "reference" and row["kind"] == "inheritance"
        ]
        findings: list[RefactorFinding] = []
        visited: set[tuple[str, str]] = {(selected.file_id, owner_symbol)}
        queue: list[tuple[str, str, str]] = [(selected.file_id, owner_symbol, base_decl["path"])]
        while queue:
            base_file_id, base_qualified, base_path = queue.pop(0)
            base_tail = base_qualified.rsplit(".", 1)[-1]
            for row in inheritance_rows:
                if not self._inheritance_targets(
                    row, base_file_id, base_tail, base_path, imports, known_paths
                ):
                    continue
                subclass_qualified = row["source_qualified_symbol"]
                if not subclass_qualified:
                    continue
                key = (row["file_id"], subclass_qualified)
                if key in visited:
                    continue
                visited.add(key)
                queue.append((row["file_id"], subclass_qualified, row["path"]))
                override_symbol = f"{subclass_qualified}.{method_name}"
                override_candidates = self.store.declaration_shapes(
                    selected.project_id,
                    override_symbol,
                    version=version,
                    schema_version=REFERENCE_SCHEMA_VERSION,
                )
                override_decl = next(
                    (decl for decl in override_candidates if decl["file_id"] == row["file_id"]),
                    None,
                )
                if override_decl is None:
                    continue
                source, bom = self._file_bytes(root, row["path"], sources)
                if not self._matches_coverage(row["path"], coverage_hashes, source, bom, digests):
                    continue
                start_byte = (override_decl["start_byte"] or 0) + bom
                end_byte = (override_decl["end_byte"] or 0) + bom
                edit_start, edit_end = self._edit_span(
                    source,
                    override_decl["start_byte"] or 0,
                    override_decl["end_byte"] or 0,
                    method_name,
                    bom,
                )
                findings.append(
                    RefactorFinding(
                        reference_id=f"override:{override_decl['file_id']}:{override_symbol}",
                        project_id=row["project_id"],
                        path=row["path"],
                        language=row["language"],
                        kind="write",
                        start_line=override_decl["start_line"] or 0,
                        end_line=override_decl["end_line"] or 0,
                        start_byte=start_byte,
                        end_byte=end_byte,
                        snippet=method_name,
                        written_name=method_name,
                        resolution="likely",
                        reason_code="override_of_renamed_method",
                        explanation=(
                            f"{subclass_qualified} overrides the renamed method; dynamic "
                            "dispatch means this cannot be proven structurally."
                        ),
                        edit_required=True,
                        edit_start_byte=edit_start,
                        edit_end_byte=edit_end,
                    )
                )
        return findings

    def _inheritance_targets(
        self,
        row: ReferenceRecord,
        base_file_id: str,
        base_tail: str,
        base_path: str,
        imports: dict[str, list[ReferenceRecord]],
        known_paths: frozenset[str],
    ) -> bool:
        """True if an `inheritance` row's base name binds to the class at hand.

        A same-file reference needs no import to bind, but the written name
        must still be the base class's own name -- otherwise every class in
        the file would match, not just the ones that actually extend it. A
        cross-file reference must go through an import in the referring file
        that resolves to the base class's own file, mirroring
        `_import_targets`; that import's *local* binding (its alias, when
        aliased) is what has to equal the written name here, since there is
        no alias to consult for a same-file reference (no import exists).
        """

        written_target = row["target_name"] or ""
        target = written_target.rsplit(".", 1)[-1]
        if not written_target:
            return False
        if row["file_id"] == base_file_id and "." not in written_target:
            return written_target == base_tail
        for item in imports.get(row["file_id"], []):
            binding = item["alias"] or item["written_name"] or item["imported_name"]
            if "." in written_target:
                receiver, member = written_target.rsplit(".", 1)
                if (
                    member != base_tail
                    or binding != receiver
                    or not self._is_namespace_import(item)
                ):
                    continue
            elif binding != target or item["imported_name"] not in {base_tail, "default"}:
                continue
            module_path = item["module_path"]
            if module_path is not None and self._module_matches(
                item["path"], item["language"], module_path, base_path, known_paths
            ):
                return True
        return False

    @staticmethod
    def _issue_explanation(issue: str) -> str:
        explanations = {
            "missing_required_parameter": "This call omits a required proposed parameter.",
            "invalid_keyword": "This call uses a keyword absent from the proposed signature.",
            "parameter_mode_change": "This call is incompatible with a proposed parameter mode.",
            "removed_positional_parameter": "This call supplies a removed positional parameter.",
            "positional_order_change": (
                "This call depends on a positional parameter order that changes."
            ),
            "spread_uncertainty": "A spread argument prevents a deterministic compatibility check.",
            "overload_ambiguity": "Multiple declaration shapes prevent a deterministic comparison.",
        }
        return explanations[issue]

    def _select(self, selector: DeclarationSelector) -> SelectedDeclaration:
        if selector.chunk_id is not None:
            selected_chunk = self.store.get_chunk(selector.chunk_id)
            if (
                selected_chunk is None
                or selected_chunk.symbol is None
                or selected_chunk.qualified_symbol is None
            ):
                raise CodeIndexingError(
                    ErrorCode.AMBIGUOUS_SYMBOL,
                    f"chunk_id {selector.chunk_id} is not a declaration chunk; chunk ids come "
                    "from find_symbol or search_code results and change when a file is reindexed",
                )
            return SelectedDeclaration(
                project_id=selected_chunk.project_id,
                file_id=selected_chunk.file_id,
                path=selected_chunk.path,
                language=selected_chunk.language,
                symbol=selected_chunk.symbol,
                qualified_symbol=selected_chunk.qualified_symbol,
                kind=selected_chunk.kind,
                start_line=selected_chunk.start_line,
                end_line=selected_chunk.end_line,
                chunk_id=selected_chunk.chunk_id,
            )
        assert selector.project is not None and selector.path is not None
        assert selector.qualified_symbol is not None
        indexed = self.store.list_chunks([selector.project])
        chunks = [
            chunk
            for chunk in indexed
            if chunk.path == selector.path and chunk.qualified_symbol == selector.qualified_symbol
        ]
        if len(chunks) > 1:
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_SYMBOL,
                f"{selector.qualified_symbol} matches {len(chunks)} declarations in "
                f"{selector.path}; select one by chunk_id",
                project=selector.project,
                candidates=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "path": chunk.path,
                        "qualified_symbol": chunk.qualified_symbol,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                    }
                    for chunk in chunks
                ],
            )
        if not chunks:
            # Distinguish a typo from a symbol this project genuinely lacks:
            # "no declaration" and "no references" are different answers.
            near = sorted(
                {
                    chunk.qualified_symbol
                    for chunk in indexed
                    if chunk.qualified_symbol is not None
                    and chunk.qualified_symbol.rsplit(".", 1)[-1]
                    == selector.qualified_symbol.rsplit(".", 1)[-1]
                }
            )
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_SYMBOL,
                f"No declaration {selector.qualified_symbol} in {selector.path}",
                project=selector.project,
                path=selector.path,
                candidates=near[:_MAX_LIMITATION_PATHS],
            )
        located_chunk = chunks[0]
        if located_chunk.symbol is None or located_chunk.qualified_symbol is None:
            raise CodeIndexingError(
                ErrorCode.AMBIGUOUS_SYMBOL,
                f"{selector.qualified_symbol} in {selector.path} is not a declaration",
                project=selector.project,
                path=selector.path,
            )
        # The project id is the selector's own; chunk rows no longer carry it.
        assert selector.project is not None
        return SelectedDeclaration(
            project_id=selector.project,
            file_id=located_chunk.file_id,
            path=located_chunk.path,
            language=located_chunk.language,
            symbol=located_chunk.symbol,
            qualified_symbol=located_chunk.qualified_symbol,
            kind=located_chunk.kind,
            start_line=located_chunk.start_line,
            end_line=located_chunk.end_line,
            chunk_id=located_chunk.chunk_id,
        )

    @staticmethod
    def _imports_by_file(records: list[ReferenceRecord]) -> dict[str, list[ReferenceRecord]]:
        result: dict[str, list[ReferenceRecord]] = {}
        for row in records:
            if row["record_kind"] == "reference" and row["kind"] == "import":
                result.setdefault(row["file_id"], []).append(row)
        return result

    @staticmethod
    def _declarations_by_file_target(
        declarations: list[ReferenceRecord],
    ) -> dict[tuple[str, str | None], list[ReferenceRecord]]:
        """Declarations grouped by their own `(file_id, target_name)` (E2).

        `_lexical_declaration` only ever considers declarations sharing a
        reference row's `file_id` and bare target name -- filtering the full
        declaration list against those two fields for every reference row
        made classification O(reference_rows x declaration_rows). Grouping
        once turns each row's lookup into an O(1) dict access plus however
        many declarations actually share that file and name (typically very
        few).
        """
        result: dict[tuple[str, str | None], list[ReferenceRecord]] = {}
        for declaration in declarations:
            key = (declaration["file_id"], declaration["target_name"])
            result.setdefault(key, []).append(declaration)
        return result

    @staticmethod
    def _known_paths(records: list[ReferenceRecord]) -> frozenset[str]:
        """Every file path this query's snapshot has at least one row for.

        Used by `_python_package_root` to walk a Python absolute import's
        `__init__.py` chain and find its real package root (a `src/` layout
        or another package sub-root) without filesystem access. Every
        indexed file gets a `coverage` row even when it has zero
        declarations or references (`_coverage_limitations`), so this is the
        full universe of paths the resolver can see -- not just the ones
        with structural content.
        """
        return frozenset(row["path"] for row in records)

    @staticmethod
    def _reexport_rows_by_path(records: list[ReferenceRecord]) -> dict[str, list[ReferenceRecord]]:
        """Import/export rows keyed by their own file's path (R2).

        A re-export chain is walked file-by-file through `module_path`
        resolution, which yields a path, not a `file_id` -- unlike
        `_imports_by_file`, this also includes `export` rows, since a
        barrel's re-export (`export { b } from './impl'`, or Python's
        `from .impl import b` inside `pkg/__init__.py`) is the edge being
        followed.
        """
        result: dict[str, list[ReferenceRecord]] = {}
        for row in records:
            if row["record_kind"] == "reference" and row["kind"] in {"import", "export"}:
                result.setdefault(row["path"], []).append(row)
        return result

    def _may_refer(
        self,
        row: ReferenceRecord,
        selected: SelectedDeclaration,
        imports: dict[str, list[ReferenceRecord]],
        reexport_rows: dict[str, list[ReferenceRecord]],
        known_paths: frozenset[str],
    ) -> bool:
        target_tail = (row["target_name"] or "").rsplit(".", 1)[-1]
        written_tail = (row["written_name"] or "").rsplit(".", 1)[-1]
        if selected.symbol in {row["target_name"], row["written_name"], target_tail, written_tail}:
            return True
        if row["kind"] == "import" and self._is_namespace_import(row):
            return False
        spelling = row["receiver_text"] or row["written_name"]
        return any(
            (item["alias"] or item["written_name"] or item["imported_name"]) == spelling
            and (
                self._import_targets(item, selected, known_paths)
                or self._reexport_targets_symbol(
                    item, selected.symbol, selected.path, reexport_rows, known_paths
                )
            )
            for item in imports.get(row["file_id"], [])
        )

    @staticmethod
    def _lexical_declaration(
        row: ReferenceRecord,
        declarations_by_file_target: dict[tuple[str, str | None], list[ReferenceRecord]],
        class_scopes: set[str],
    ) -> ReferenceRecord | None:
        if row["receiver_text"] is not None or row["kind"] not in {"call", "read", "write"}:
            return None
        source = row["source_qualified_symbol"] or ""
        target = (row["target_name"] or "").rsplit(".", 1)[-1]
        visible: list[tuple[int, ReferenceRecord]] = []
        for declaration in declarations_by_file_target.get((row["file_id"], target), []):
            qualified = declaration["source_qualified_symbol"] or ""
            scope = qualified.rsplit(".", 1)[0] if "." in qualified else ""
            # Python and JS/TS both leave the class body out of a method's
            # scope chain: inside `Gate.run`, a bare `helper()` binds to the
            # module-level `helper`, never to the sibling method `Gate.helper`.
            # Treating the class as an enclosing scope silently dropped those
            # call sites from every result.
            if scope and scope in class_scopes:
                continue
            if not scope or source == scope or source.startswith(scope + "."):
                visible.append((scope.count(".") + bool(scope), declaration))
        return max(visible, key=lambda item: item[0])[1] if visible else None

    def _classify(
        self,
        row: ReferenceRecord,
        selected: SelectedDeclaration,
        target_candidates: list[ReferenceRecord],
        imports: dict[str, list[ReferenceRecord]],
        reexport_rows: dict[str, list[ReferenceRecord]],
        known_paths: frozenset[str],
    ) -> tuple[str, str, str]:
        source_imports = imports.get(row["file_id"], [])
        if any(
            item["imported_name"] == "*" and not self._is_namespace_import(item)
            for item in source_imports
        ):
            return (
                "unresolved",
                "wildcard_import",
                "A wildcard import can bind this name dynamically.",
            )
        if row["receiver_text"] is not None:
            receiver = row["receiver_text"]
            if receiver in {"self", "cls", "this"} and self._same_owner(row, selected):
                return (
                    "exact",
                    "known_owner_member",
                    "The receiver is the declaration's enclosing owner.",
                )
            for item in source_imports:
                binding = item["alias"] or item["written_name"] or item["imported_name"]
                if binding == receiver and self._import_targets(item, selected, known_paths):
                    return (
                        "exact",
                        "known_namespace_member",
                        "The receiver is a known imported namespace.",
                    )
            return (
                "likely",
                "unknown_receiver",
                "Receiver type inference is outside this syntax-only index.",
            )
        unproven_reexport = False
        for item in source_imports:
            alias = item["alias"] or item["written_name"] or item["imported_name"]
            if alias != row["written_name"]:
                continue
            if self._import_targets(item, selected, known_paths):
                return (
                    "exact",
                    "direct_import_alias",
                    "The local alias directly imports this declaration.",
                )
            if self._reexport_targets_symbol(
                item, selected.symbol, selected.path, reexport_rows, known_paths
            ):
                return (
                    "exact",
                    "reexport_chain",
                    "The local alias resolves to the declaration through a chain of re-exports.",
                )
            if item["module_path"] is not None:
                unproven_reexport = True
        if unproven_reexport:
            return (
                "likely",
                "unproven_reexport",
                "The local alias imports from a module, but the chain of re-exports "
                "to the declaration's file could not be proven.",
            )
        if row["file_id"] == selected.file_id:
            return "exact", "same_file_symbol", "The call is in the declaration's source file."
        # `target_candidates` is already the project-wide set of
        # declarations sharing `selected.symbol` (fetched once, above, via
        # `target_name_candidates`), so no further filtering by name is
        # needed here.
        if len(target_candidates) == 1:
            return (
                "likely",
                "name_only_candidate",
                "The name is unique, but no import or owner proves binding.",
            )
        return (
            "unresolved",
            "ambiguous_symbol",
            "Multiple declarations or scopes could bind this name.",
        )

    def _import_targets(
        self, item: ReferenceRecord, selected: SelectedDeclaration, known_paths: frozenset[str]
    ) -> bool:
        return self._import_targets_symbol(item, selected.symbol, selected.path, known_paths)

    def _import_targets_symbol(
        self, item: ReferenceRecord, symbol: str, path: str, known_paths: frozenset[str]
    ) -> bool:
        imported = item["imported_name"]
        if not self._is_namespace_import(item) and imported not in {symbol, "default", None}:
            return False
        module_path = item["module_path"]
        if module_path is None:
            return False
        return self._module_matches(item["path"], item["language"], module_path, path, known_paths)

    def _reexport_targets_symbol(
        self,
        item: ReferenceRecord,
        symbol: str,
        path: str,
        rows_by_path: dict[str, list[ReferenceRecord]],
        known_paths: frozenset[str],
        visited: frozenset[tuple[str, str]] = frozenset(),
        depth: int = 0,
    ) -> bool:
        """Prove a binding through a chain of barrel re-exports/imports (R2).

        `_import_targets_symbol` proves only a *direct* edge: the import's
        module resolves straight to the declaration's own file. A barrel
        (`pkg/__init__.py` doing `from .impl import b`, or `pkg/index.ts`
        doing `export { b } from './impl'`) sits between the importer and
        the declaration; this walks such indirections one module-edge at a
        time. Each hop must be a real, resolvable `import`/`export` row
        binding the exact name the previous hop asked for -- never a guess
        -- so an unrelated same-named symbol down a different chain can
        never bind (the corpus hard gate). `visited` (keyed by resolved
        path + the name being chased there) and `depth` prevent chasing a
        cycle or a pathological fan-out forever.
        """
        if self._import_targets_symbol(item, symbol, path, known_paths):
            return True
        if depth >= _MAX_REEXPORT_DEPTH or self._is_namespace_import(item):
            return False
        module_path = item["module_path"]
        if module_path is None:
            return False
        imported = item["imported_name"]
        lookup_name = symbol if imported is None or imported == "default" else imported
        for candidate in self._module_candidates(
            item["path"], item["language"], module_path, known_paths
        ):
            key = (str(candidate), lookup_name)
            if key in visited:
                continue
            next_visited = visited | {key}
            for hop in rows_by_path.get(str(candidate), []):
                binding = hop["alias"] or hop["written_name"] or hop["imported_name"]
                if binding != lookup_name:
                    continue
                if self._reexport_targets_symbol(
                    hop, symbol, path, rows_by_path, known_paths, next_visited, depth + 1
                ):
                    return True
        return False

    @staticmethod
    def _is_namespace_import(item: ReferenceRecord) -> bool:
        imported = item["imported_name"]
        return (imported == "*" and item["alias"] is not None) or (
            item["language"] == "python" and imported is None and item["module_path"] is not None
        )

    @staticmethod
    def _same_owner(row: ReferenceRecord, selected: SelectedDeclaration) -> bool:
        source = row["source_qualified_symbol"] or ""
        owner = (
            selected.qualified_symbol.rsplit(".", 1)[0] if "." in selected.qualified_symbol else ""
        )
        return bool(owner and (source == owner or source.startswith(owner + ".")))

    @staticmethod
    def _python_package_root(
        directory: PurePosixPath, known_paths: frozenset[str]
    ) -> PurePosixPath:
        """The directory an absolute import from a file in `directory` resolves against.

        Python resolves an absolute import from whatever is on `sys.path`,
        which this syntax-only index never sees directly -- but the parent of
        the topmost directory in `directory`'s unbroken `__init__.py` chain
        (its own package, and its package's package, ...) is exactly that
        directory for any regular (non-namespace) package, `src/`-layout
        included: `mypkg/a.py` with `mypkg/__init__.py` anchors at the
        project root; `src/mypkg/a.py` with both `src/mypkg/__init__.py` and
        no `src/__init__.py` anchors at `src`.

        A directory with no `__init__.py` at all -- either a flat top-level
        layout (nothing to walk) or a PEP 420 namespace package (no marker
        file exists to find) -- falls back to the project root. That is
        already correct for the flat case, and merely non-exact rather than
        wrong for a namespace package nested under a further sub-root: it
        never fabricates a false candidate, which is the property this
        function exists to protect (S2).
        """
        parts = list(directory.parts)
        if not parts or str(PurePosixPath(*parts, "__init__.py")) not in known_paths:
            # `directory` itself is not a package (no `__init__.py` of its
            # own) -- either there is nothing to walk (a flat top-level
            # layout) or it is a namespace package, which leaves no marker
            # file to find its boundary from. Root-anchoring is exactly
            # right for the flat case and a safe non-fabricating fallback
            # for the namespace one; it must not become `directory` itself,
            # or this collapses back to the sibling-anchor bug this whole
            # function exists to fix.
            return PurePosixPath()
        boundary = len(parts)
        while boundary > 0 and str(PurePosixPath(*parts[:boundary], "__init__.py")) in known_paths:
            boundary -= 1
        return PurePosixPath(*parts[:boundary])

    @staticmethod
    def _module_candidates(
        source_path: str, language: str, module_path: str, known_paths: frozenset[str]
    ) -> set[PurePosixPath]:
        """Every file path a relative `module_path` from `source_path` could mean.

        Shared by `_module_matches` (a yes/no check against one known target)
        and `_reexport_targets_symbol` (which needs the actual resolved
        path(s) to keep walking a re-export chain).
        """
        source = PurePosixPath(source_path)
        if language == "python":
            dots = len(module_path) - len(module_path.lstrip("."))
            suffix = module_path[dots:]
            stem = PurePosixPath(*suffix.split(".")) if suffix else PurePosixPath()
            if dots == 0:
                # An absolute import is resolved from the package root, not
                # from the importing file's own directory -- see
                # `_python_package_root` for how that root is found without
                # filesystem access (a `src/` layout or another package
                # sub-root is common, so it is not always the project root).
                # Blindly anchoring at `source.parent` would let a sibling
                # file bind falsely (e.g. `mypkg/utils.py` for `from utils
                # import f` when the real target is the top-level
                # `utils.py`), which is exactly why this cannot simply walk
                # every ancestor as an equally-plausible candidate.
                base = ReferenceService._python_package_root(source.parent, known_paths)
            else:
                base = source.parent
                for _ in range(dots - 1):
                    base = base.parent
            return {base / f"{stem}.py", base / stem / "__init__.py"}
        if not module_path.startswith("."):
            return set()
        # Walk `module_path`'s segments against an explicit stack instead of
        # simply appending them to `source.parent`: a `..` segment must pop
        # the last resolved directory, not survive as a literal path
        # component (`PurePosixPath` never resolves `..` on its own, so
        # `src/app/../utils` never string-equals `src/utils`). A `..` with
        # nothing left to pop would escape the project root, which no
        # in-project target can ever match, so that yields no candidates.
        parts = list(source.parent.parts)
        for part in PurePosixPath(module_path).parts:
            if part == ".":
                continue
            if part == "..":
                if not parts:
                    return set()
                parts.pop()
                continue
            parts.append(part)
        normalized = PurePosixPath(*parts)
        extensions = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
        candidates = {PurePosixPath(f"{normalized}{extension}") for extension in extensions}
        candidates.update(normalized / f"index{extension}" for extension in extensions)
        return candidates

    @staticmethod
    def _module_matches(
        source_path: str,
        language: str,
        module_path: str,
        target_path: str,
        known_paths: frozenset[str],
    ) -> bool:
        target = PurePosixPath(target_path)
        return target in ReferenceService._module_candidates(
            source_path, language, module_path, known_paths
        )

    @staticmethod
    def _edit_span(
        source: bytes, start_byte: int, end_byte: int, name: str, bom: int
    ) -> tuple[int | None, int | None]:
        """Locate the identifier to rewrite inside a reference's own range.

        The stored range covers the whole occurrence, which is wider than the
        name for a qualified call (`auth.authorize`) and for an aliased import
        (`authorize as check`). Replacing the whole range would drop the module
        qualifier or the alias, so the exact identifier is located instead and
        anything ambiguous is left for a human.
        """

        span = source[start_byte:end_byte]
        if not span or not name:
            return None, None
        matches = list(re.finditer(rf"(?<![\w$]){re.escape(name)}(?![\w$])".encode(), span))
        if len(matches) != 1:
            return None, None
        return start_byte + matches[0].start() + bom, start_byte + matches[0].end() + bom

    def _project_root(self, project_id: str) -> Path | None:
        project = next((item for item in self.store.list_projects() if item.id == project_id), None)
        return project.root if project is not None else None

    @staticmethod
    def _matches_coverage(
        path: str,
        coverage_hashes: dict[str, str],
        source: bytes,
        bom: int,
        digests: dict[str, str],
    ) -> bool:
        """Whether a file's current bytes still match what its rows were extracted from.

        Reference offsets are applied to the file as it sits on disk, so rows
        whose extraction hash no longer matches those bytes would be served
        against text they never described -- a wrong-edit hazard for callers
        that trust the offsets (a failed replacement deliberately retains its
        previous generation's rows). Digests are memoized per path because a
        file can hold many candidate rows. A file with no coverage row has
        nothing to validate against and is trusted. `source` is BOM-stripped,
        so the marker is added back before hashing: extraction hashed the raw
        bytes, marker included.
        """
        expected = coverage_hashes.get(path)
        if expected is None:
            return True
        digest = digests.get(path)
        if digest is None:
            raw = (_BOM + source) if bom else source
            digest = _digest(raw)
            digests[path] = digest
        return digest == expected

    @staticmethod
    def _file_bytes(
        root: Path | None, path: str, cache: dict[str, tuple[bytes, int]]
    ) -> tuple[bytes, int]:
        """Return one file's BOM-stripped bytes and the offset that was removed.

        Reads are cached for the life of one query. Resolving a few hundred
        references used to re-read the same file once per hit.
        """

        entry = cache.get(path)
        if entry is None:
            try:
                raw = (root / path).read_bytes() if root is not None else b""
            except OSError:
                raw = b""
            offset = len(_BOM) if raw.startswith(_BOM) else 0
            entry = (raw[offset:], offset)
            cache[path] = entry
        return entry

    def _coverage_limitations(
        self, records: list[ReferenceRecord], backfill: ReferenceBackfillReport | None
    ) -> list[ReferenceLimitation]:
        """Report files that hold no structural rows because none could be made.

        Coverage rows prove a file was parsed under the current schema. A file
        whose language has no reference query still gets one, so without this
        the caller cannot tell "searched and found nothing" from "never looked".
        """

        limitations: list[ReferenceLimitation] = []
        unanalyzed: dict[str, list[str]] = {}
        for row in records:
            if row["record_kind"] != "coverage":
                continue
            if row["language"] not in STRUCTURAL_LANGUAGES:
                unanalyzed.setdefault(row["language"], []).append(row["path"])
        for language, paths in sorted(unanalyzed.items()):
            limitations.append(
                ReferenceLimitation(
                    code="unsupported_language",
                    explanation=(
                        f"{len(paths)} {language} file(s) are indexed for search but have no "
                        "structural reference extraction, so uses of this declaration in them "
                        f"are invisible here: {self._sample(paths)}"
                    ),
                )
            )
        if backfill is not None:
            for code, paths in (
                ("parse_error", backfill.incomplete_paths),
                ("stale_file", backfill.stale_paths),
            ):
                if not paths:
                    continue
                reason = (
                    "could not be parsed, so their references are missing"
                    if code == "parse_error"
                    else "changed after they were indexed, so their references may be stale"
                )
                limitations.append(
                    ReferenceLimitation(
                        code=code,
                        explanation=(f"{len(paths)} file(s) {reason}: {self._sample(paths)}"),
                    )
                )
        return limitations

    @staticmethod
    def _sample(paths: list[str]) -> str:
        shown = sorted(paths)[:_MAX_LIMITATION_PATHS]
        remainder = len(paths) - len(shown)
        return ", ".join(shown) + (f", and {remainder} more" if remainder else "")

    @staticmethod
    def _operation_digest(operation: RefactorOperation) -> str:
        """A short, stable fingerprint of a refactor operation's full shape.

        Bound into the cursor so a page-2 `analyze_refactor` call is
        rejected if the caller supplies a different `new_name` or a
        different signature spec than page 1 used (T2 new gap) -- without
        this, page 2 would silently classify hits against a different
        operation than the one whose findings the caller already saw on
        page 1.
        """
        raw = json.dumps(
            operation.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _require_active_cursor_slot(self, project_id: str, payload: dict[str, object]) -> None:
        pointer = self.store.active_pointer(project_id)
        if (
            pointer is None
            or pointer[0].slot_id != payload["slot_id"]
            or pointer[1] != payload["activation_epoch"]
        ):
            raise CodeIndexingError(
                ErrorCode.STALE_CURSOR, "Reference cursor slot is no longer active"
            )

    @staticmethod
    def _encode_cursor(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    # The exact key set every cursor payload must carry (T2). `_decode_cursor`
    # rejects anything else outright rather than letting a missing key surface
    # later as a bare `KeyError` from `payload["..."]` -- the defect that made
    # a well-formed but foreign cursor leak `Error executing tool
    # find_references: 'version'` straight to the client.
    _CURSOR_FIELDS: Final = frozenset(
        {
            "version",
            "project_id",
            "path",
            "qualified_symbol",
            "kinds",
            "offset",
            "limit",
            "operation_digest",
            "slot_id",
            "activation_epoch",
        }
    )

    @staticmethod
    def _decode_cursor(cursor: str) -> dict[str, object]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CodeIndexingError(ErrorCode.INVALID_CURSOR, "invalid reference cursor") from exc

        def invalid() -> CodeIndexingError:
            return CodeIndexingError(ErrorCode.INVALID_CURSOR, "invalid reference cursor")

        if not isinstance(payload, dict) or set(payload) != ReferenceService._CURSOR_FIELDS:
            raise invalid()
        for int_field in ("version", "offset", "limit", "activation_epoch"):
            value = payload[int_field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise invalid()
        for str_field in ("project_id", "path", "qualified_symbol", "slot_id"):
            if not isinstance(payload[str_field], str):
                raise invalid()
        kinds = payload["kinds"]
        if not isinstance(kinds, list) or not all(isinstance(item, str) for item in kinds):
            raise invalid()
        operation_digest = payload["operation_digest"]
        if operation_digest is not None and not isinstance(operation_digest, str):
            raise invalid()
        return payload
