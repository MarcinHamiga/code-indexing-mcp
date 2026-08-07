"""Conservative, syntax-only reference classification over structural rows."""

from __future__ import annotations

import base64
import json
import keyword as keyword_module
import re
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, cast

from .errors import CodeIndexingError, ErrorCode
from .extractor import STRUCTURAL_LANGUAGES
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
_LIMITATION_REASONS: Final = frozenset({"wildcard_import", "unknown_receiver", "ambiguous_symbol"})

# Limitations that mean whole files were never analyzed. Any of them forces the
# completeness state to "incomplete" rather than merely "dynamic limitations".
_COVERAGE_GAP_CODES: Final = frozenset({"unsupported_language", "parse_error", "stale_file"})

_BOM: Final = b"\xef\xbb\xbf"

# How many individual paths a coverage limitation names before it summarizes.
_MAX_LIMITATION_PATHS: Final = 10


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
    ) -> ReferenceResponse:
        if limit < 1 or limit > 500:
            raise CodeIndexingError(ErrorCode.INVALID_FILTER, "limit must be between 1 and 500")
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
        version = self.store.reference_version(selected.project_id)
        offset = 0
        if cursor is not None:
            payload = self._decode_cursor(cursor)
            cursor_version = payload["version"]
            if isinstance(cursor_version, bool) or not isinstance(cursor_version, int):
                raise ValueError("invalid reference cursor")
            if payload["project_id"] != selected.project_id or payload["path"] != selected.path:
                raise ValueError("cursor does not match the selected declaration")
            if payload["qualified_symbol"] != selected.qualified_symbol:
                raise ValueError("cursor does not match the selected declaration")
            if payload["kinds"] != sorted(kinds or set()):
                raise ValueError("cursor does not match reference filters")
            offset_value = payload["offset"]
            if isinstance(offset_value, bool) or not isinstance(offset_value, int):
                raise ValueError("invalid reference cursor")
            offset = offset_value
            version = cursor_version

        try:
            records = self.store.list_reference_records(selected.project_id, version=version)
        except (FileNotFoundError, ValueError) as error:
            raise CodeIndexingError(
                ErrorCode.STALE_CURSOR, "Reference cursor snapshot expired"
            ) from error
        declarations = [row for row in records if row["record_kind"] == "declaration"]
        imports = self._imports_by_file(records)
        # A declaration nested directly in a class body is reachable only
        # through a receiver, so it must not shadow a bare name the way a
        # nested function does.
        class_scopes = {
            scope
            for row in declarations
            if row["kind"] == "class" and (scope := row["source_qualified_symbol"])
        }
        root = self._project_root(selected.project_id)
        sources: dict[str, tuple[bytes, int]] = {}
        hits: list[ReferenceHit] = []
        limitations: list[ReferenceLimitation] = self._coverage_limitations(records, backfill)
        for row in records:
            if row["record_kind"] != "reference" or not self._may_refer(row, selected, imports):
                continue
            if kinds is not None and row["kind"] not in kinds:
                continue
            lexical = self._lexical_declaration(row, declarations, class_scopes)
            if (
                lexical is not None
                and row["file_id"] == selected.file_id
                and lexical["source_qualified_symbol"] != selected.qualified_symbol
            ):
                continue
            resolution, reason, explanation = self._classify(row, selected, declarations, imports)
            if reason in _LIMITATION_REASONS:
                limitations.append(
                    ReferenceLimitation(code=reason, explanation=explanation, path=row["path"])
                )
            source, bom = self._file_bytes(root, row["path"], sources)
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
        hits.sort(key=lambda hit: (hit.path, hit.start_line, hit.start_byte, hit.reference_id))
        page = hits[offset : offset + limit]
        next_cursor = None
        if offset + len(page) < len(hits):
            next_cursor = self._encode_cursor(
                {
                    "version": version,
                    "project_id": selected.project_id,
                    "path": selected.path,
                    "qualified_symbol": selected.qualified_symbol,
                    "kinds": sorted(kinds or set()),
                    "offset": offset + len(page),
                }
            )
        unique_limitations = {
            (limitation.code, limitation.explanation, limitation.path): limitation
            for limitation in limitations
        }
        return ReferenceResponse(
            selected=selected,
            hits=page,
            limitations=sorted(
                unique_limitations.values(), key=lambda item: (item.code, item.path or "")
            ),
            cursor=next_cursor,
            snapshot_version=version,
        )

    def analyze_refactor(
        self,
        selector: DeclarationSelector,
        operation: RefactorOperation,
        *,
        limit: int = 500,
        cursor: str | None = None,
        backfill: ReferenceBackfillReport | None = None,
    ) -> RefactorAnalysis:
        response = self.find_references(selector, limit=limit, cursor=cursor, backfill=backfill)
        records = self.store.list_reference_records(
            response.selected.project_id, version=response.snapshot_version
        )
        shapes_by_id = {row["reference_id"]: row for row in records}
        root = self._project_root(response.selected.project_id)
        sources: dict[str, tuple[bytes, int]] = {}
        if isinstance(operation, RenameOperation):
            valid_identifier = (
                operation.new_name.isidentifier()
                and not keyword_module.iskeyword(operation.new_name)
                if response.selected.language == "python"
                else bool(re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", operation.new_name))
            )
            if not valid_identifier:
                raise CodeIndexingError(ErrorCode.INVALID_REFACTOR, "Invalid identifier for rename")
        must_change: list[RefactorFinding] = []
        likely_change: list[RefactorFinding] = []
        review: list[RefactorFinding] = []
        evidence: list[RefactorFinding] = []
        if isinstance(operation, RenameOperation) and cursor is None:
            declaration = next(
                (
                    row
                    for row in records
                    if row["record_kind"] == "declaration"
                    and row["file_id"] == response.selected.file_id
                    and row["source_qualified_symbol"] == response.selected.qualified_symbol
                ),
                None,
            )
            start_byte, end_byte = 0, 0
            edit_start, edit_end = None, None
            if declaration is not None:
                source, bom = self._file_bytes(root, response.selected.path, sources)
                start_byte = (declaration["start_byte"] or 0) + bom
                end_byte = (declaration["end_byte"] or 0) + bom
                edit_start, edit_end = self._edit_span(
                    source,
                    declaration["start_byte"] or 0,
                    declaration["end_byte"] or 0,
                    response.selected.symbol,
                    bom,
                )
            must_change.append(
                RefactorFinding(
                    reference_id=f"declaration:{response.selected.file_id}",
                    project_id=response.selected.project_id,
                    path=response.selected.path,
                    language=response.selected.language,
                    kind="write",
                    start_line=response.selected.start_line,
                    end_line=response.selected.end_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    snippet=response.selected.symbol,
                    resolution="exact",
                    reason_code="declaration",
                    explanation="The selected declaration must be renamed.",
                    written_name=response.selected.symbol,
                    edit_required=True,
                    edit_start_byte=edit_start,
                    edit_end_byte=edit_end,
                )
            )
        for hit in response.hits:
            finding = RefactorFinding(**hit.model_dump(), edit_required=False)
            if hit.resolution == "unresolved":
                review.append(finding)
                continue
            if hit.resolution == "likely":
                likely_change.append(finding.model_copy(update={"edit_required": True}))
                continue
            if isinstance(operation, RenameOperation):
                # The indexed spelling, not the snippet: re-reading the file to
                # decide whether an edit is needed turns an unreadable file or
                # a byte-order mark into a silently skipped call site.
                written = hit.written_name or hit.snippet
                needs_edit = written.rsplit(".", 1)[-1] == response.selected.symbol or hit.kind in {
                    "import",
                    "export",
                }
                if needs_edit:
                    source, bom = self._file_bytes(root, hit.path, sources)
                    edit_start, edit_end = self._edit_span(
                        source,
                        hit.start_byte - bom,
                        hit.end_byte - bom,
                        response.selected.symbol,
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
                else:
                    evidence.append(finding)
                continue
            issue = self._signature_issue(
                response.selected, shapes_by_id.get(hit.reference_id), records, operation
            )
            if issue in {"spread_uncertainty", "overload_ambiguity"}:
                review.append(
                    finding.model_copy(
                        update={"reason_code": issue, "explanation": self._issue_explanation(issue)}
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
        limitations = response.limitations
        coverage_gaps = [item for item in limitations if item.code in _COVERAGE_GAP_CODES]
        if response.cursor is not None:
            state = "incomplete"
            explanation = "More structural candidates remain available through the cursor."
        elif coverage_gaps:
            state = "incomplete"
            explanation = (
                "Some files could not be analyzed, so this list may omit real uses. "
                "See limitations."
            )
        elif limitations or review or likely_change:
            # "likely" is unproven by definition, so a result carrying any is
            # not the same as one the resolver could fully account for.
            state = "complete_with_dynamic_limitations"
            explanation = (
                "Every indexed file was analyzed, but some uses could not be proven "
                "without type information. See likely_change and review."
            )
        else:
            state = "complete"
            explanation = "All indexed structural candidates were considered."
        return RefactorAnalysis(
            selected=response.selected,
            operation=operation,
            must_change=must_change,
            likely_change=likely_change,
            review=review,
            evidence=evidence,
            limitations=limitations,
            counts=RefactorCounts(
                must_change=len(must_change),
                likely_change=len(likely_change),
                review=len(review),
                evidence=len(evidence),
            ),
            cursor=response.cursor,
            completeness=CompletenessReport(
                state=cast(
                    Literal["complete", "complete_with_dynamic_limitations", "incomplete"], state
                ),
                explanation=explanation,
            ),
        )

    def _signature_issue(
        self,
        selected: SelectedDeclaration,
        row: ReferenceRecord | None,
        records: list[ReferenceRecord],
        operation: SignatureChangeOperation,
    ) -> str | None:
        # Both the call shape and the declaration shapes come from the caller's
        # pinned snapshot. Re-querying the live table per hit rescanned the
        # whole reference table once per call site, and a refresh mid-analysis
        # could report an incompatible call as compatible.
        if row is None or row["shape_json"] is None:
            return None
        shape = json.loads(row["shape_json"])
        if shape.get("has_positional_spread") or shape.get("has_keyword_spread"):
            return "spread_uncertainty"
        old_shapes = [
            record
            for record in records
            if record["record_kind"] == "declaration"
            and record["source_qualified_symbol"] == selected.qualified_symbol
        ]
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
        return SelectedDeclaration(
            project_id=located_chunk.project_id,
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

    def _may_refer(
        self,
        row: ReferenceRecord,
        selected: SelectedDeclaration,
        imports: dict[str, list[ReferenceRecord]],
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
            and self._import_targets(item, selected)
            for item in imports.get(row["file_id"], [])
        )

    @staticmethod
    def _lexical_declaration(
        row: ReferenceRecord,
        declarations: list[ReferenceRecord],
        class_scopes: set[str],
    ) -> ReferenceRecord | None:
        if row["receiver_text"] is not None or row["kind"] not in {"call", "read", "write"}:
            return None
        source = row["source_qualified_symbol"] or ""
        target = (row["target_name"] or "").rsplit(".", 1)[-1]
        visible: list[tuple[int, ReferenceRecord]] = []
        for declaration in declarations:
            if declaration["file_id"] != row["file_id"] or declaration["target_name"] != target:
                continue
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
        declarations: list[ReferenceRecord],
        imports: dict[str, list[ReferenceRecord]],
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
                if binding == receiver and self._import_targets(item, selected):
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
        for item in source_imports:
            alias = item["alias"] or item["written_name"] or item["imported_name"]
            if alias == row["written_name"] and self._import_targets(item, selected):
                return (
                    "exact",
                    "direct_import_alias",
                    "The local alias directly imports this declaration.",
                )
        if row["file_id"] == selected.file_id:
            return "exact", "same_file_symbol", "The call is in the declaration's source file."
        candidates = [
            declaration
            for declaration in declarations
            if declaration["target_name"] == selected.symbol
        ]
        if len(candidates) == 1:
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

    def _import_targets(self, item: ReferenceRecord, selected: SelectedDeclaration) -> bool:
        imported = item["imported_name"]
        if not self._is_namespace_import(item) and imported not in {
            selected.symbol,
            "default",
            None,
        }:
            return False
        module_path = item["module_path"]
        if module_path is None:
            return False
        return self._module_matches(item["path"], item["language"], module_path, selected.path)

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
    def _module_matches(
        source_path: str, language: str, module_path: str, target_path: str
    ) -> bool:
        source = PurePosixPath(source_path)
        target = PurePosixPath(target_path)
        if language == "python":
            dots = len(module_path) - len(module_path.lstrip("."))
            suffix = module_path[dots:]
            base = source.parent
            for _ in range(max(0, dots - 1)):
                base = base.parent
            stem = PurePosixPath(*suffix.split(".")) if suffix else PurePosixPath()
            candidates = {base / f"{stem}.py", base / stem / "__init__.py"}
            return target in candidates
        if not module_path.startswith("."):
            return False
        stem = source.parent / module_path
        normalized = PurePosixPath(*(part for part in stem.parts if part != "."))
        extensions = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
        candidates = {PurePosixPath(f"{normalized}{extension}") for extension in extensions}
        candidates.update(normalized / f"index{extension}" for extension in extensions)
        return target in candidates

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
    def _encode_cursor(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> dict[str, object]:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid reference cursor") from exc
        if not isinstance(payload, dict):
            raise ValueError("invalid reference cursor")
        return payload
