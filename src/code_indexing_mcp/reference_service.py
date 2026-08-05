"""Conservative, syntax-only reference classification over structural rows."""

from __future__ import annotations

import base64
import json
from pathlib import PurePosixPath
from typing import Literal, cast

from .errors import CodeIndexingError, ErrorCode
from .models import (
    DeclarationSelector,
    ReferenceHit,
    ReferenceLimitation,
    ReferenceResponse,
    SelectedDeclaration,
)
from .storage import LanceStore, ReferenceRecord


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
    ) -> ReferenceResponse:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        selected = self._select(selector)
        version = self.store.reference_version(selected.project_id)
        offset = 0
        if cursor is not None:
            payload = self._decode_cursor(cursor)
            if payload["version"] != version:
                raise CodeIndexingError(ErrorCode.STALE_CURSOR, "Reference cursor snapshot expired")
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

        records = self.store.list_reference_records(selected.project_id)
        declarations = [row for row in records if row["record_kind"] == "declaration"]
        imports = self._imports_by_file(records)
        hits: list[ReferenceHit] = []
        limitations: list[ReferenceLimitation] = []
        for row in records:
            if row["record_kind"] != "reference" or not self._may_refer(row, selected, imports):
                continue
            if kinds is not None and row["kind"] not in kinds:
                continue
            resolution, reason, explanation = self._classify(row, selected, declarations, imports)
            if resolution == "unresolved" and reason in {"wildcard_import", "unknown_receiver"}:
                limitations.append(
                    ReferenceLimitation(code=reason, explanation=explanation, path=row["path"])
                )
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
                    start_byte=row["start_byte"] or 0,
                    end_byte=row["end_byte"] or 0,
                    snippet=self._snippet(row),
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

    def _select(self, selector: DeclarationSelector) -> SelectedDeclaration:
        if selector.chunk_id is not None:
            selected_chunk = self.store.get_chunk(selector.chunk_id)
            if (
                selected_chunk is None
                or selected_chunk.symbol is None
                or selected_chunk.qualified_symbol is None
            ):
                raise ValueError("chunk_id does not identify a declaration")
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
        chunks = [
            chunk
            for chunk in self.store.list_chunks([selector.project])
            if chunk.path == selector.path and chunk.qualified_symbol == selector.qualified_symbol
        ]
        if len(chunks) != 1:
            raise ValueError("selector does not identify exactly one declaration")
        located_chunk = chunks[0]
        if located_chunk.symbol is None or located_chunk.qualified_symbol is None:
            raise ValueError("selector does not identify a declaration")
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
        spelling = row["receiver_text"] or row["written_name"]
        return any(
            (item["alias"] or item["imported_name"]) == spelling
            and self._import_targets(item, selected)
            for item in imports.get(row["file_id"], [])
        )

    def _classify(
        self,
        row: ReferenceRecord,
        selected: SelectedDeclaration,
        declarations: list[ReferenceRecord],
        imports: dict[str, list[ReferenceRecord]],
    ) -> tuple[str, str, str]:
        source_imports = imports.get(row["file_id"], [])
        if any(item["imported_name"] == "*" for item in source_imports):
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
                if item["alias"] == receiver and self._import_targets(item, selected):
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
            alias = item["alias"] or item["imported_name"]
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
        if imported not in {selected.symbol, "default", None}:
            return False
        module_path = item["module_path"]
        if module_path is None:
            return False
        return self._module_matches(item["path"], item["language"], module_path, selected.path)

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

    def _snippet(self, row: ReferenceRecord) -> str:
        project = next(
            (item for item in self.store.list_projects() if item.id == row["project_id"]), None
        )
        if project is None or row["start_byte"] is None or row["end_byte"] is None:
            return ""
        try:
            source = (project.root / row["path"]).read_bytes()
        except OSError:
            return ""
        return source[row["start_byte"] : row["end_byte"]].decode("utf-8", errors="replace")

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
