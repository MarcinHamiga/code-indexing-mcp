"""Snapshot-cached symbol graph, Specificity-Adjusted PPR, and subgraph localization."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import threading
import zlib
from array import array
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from .models import (
    ChunkGraphContext,
    ChunkPreview,
    CompletenessReport,
    GraphNeighborRef,
    SelectedDeclaration,
    SubgraphEdge,
    SubgraphNode,
    SubgraphPath,
    SubgraphResponse,
)

if TYPE_CHECKING:
    from .reference_service import CompiledReferenceGraph, ReferenceService
    from .storage import LanceStore

logger = logging.getLogger(__name__)

_GRAPH_CACHE_MAGIC: Final = b"SYNDEX_GRAPH_V1\n"

# Edge kind -> integer code for compact CSR arrays
EDGE_KINDS: Final[tuple[str, ...]] = (
    "inheritance",
    "type_use",
    "call",
    "decorator",
    "write",
    "read",
    "export",
    "import",
    "contains",
)
_KIND_TO_CODE: Final[dict[str, int]] = {name: idx for idx, name in enumerate(EDGE_KINDS)}

# Forward weight (source -> target: caller -> callee, subtype -> supertype)
_FORWARD_KIND_WEIGHT: Final[dict[int, float]] = {
    _KIND_TO_CODE["export"]: 0.92,
    _KIND_TO_CODE["call"]: 0.90,
    _KIND_TO_CODE["inheritance"]: 0.85,
    _KIND_TO_CODE["decorator"]: 0.70,
    _KIND_TO_CODE["write"]: 0.45,
    _KIND_TO_CODE["read"]: 0.40,
    _KIND_TO_CODE["type_use"]: 0.35,
    _KIND_TO_CODE["contains"]: 0.25,
    _KIND_TO_CODE["import"]: 0.20,
}

# Reverse weight multiplier (target -> source: interface -> implementer, callee -> caller)
_REVERSE_KIND_MULTIPLIER: Final[dict[int, float]] = {
    _KIND_TO_CODE["inheritance"]: 0.95,
    _KIND_TO_CODE["call"]: 0.70,
    _KIND_TO_CODE["decorator"]: 0.65,
    _KIND_TO_CODE["write"]: 0.55,
    _KIND_TO_CODE["read"]: 0.45,
    _KIND_TO_CODE["export"]: 0.40,
    _KIND_TO_CODE["type_use"]: 0.35,
    _KIND_TO_CODE["contains"]: 0.30,
    _KIND_TO_CODE["import"]: 0.20,
}

_RESOLUTION_EXACT: Final = 1
_RESOLUTION_LIKELY: Final = 0
_LIKELY_WEIGHT_MULTIPLIER: Final = 0.55

_PPR_ALPHA: Final = 0.45
_PPR_ITERATIONS: Final = 8
_HUB_DEGREE_EXPONENT: Final = 0.85
_PRIOR_DAMPING_EXPONENT: Final = 0.25
_MIN_GRAPH_ACTIVATION: Final = 1e-5
_MAX_HYBRID_SEED_RANK: Final = 8
_RELATIVE_RRF_GRAPH_FLOOR: Final = 0.08

_IDENTIFIER_SPLIT_RE: Final = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_BOUNDARY_RE: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _query_terms(text: str) -> frozenset[str]:
    tokens: set[str] = set()
    for raw in _IDENTIFIER_SPLIT_RE.split(text):
        if not raw:
            continue
        lower = raw.lower()
        if len(lower) >= 2:
            tokens.add(lower)
        for part in _CAMEL_BOUNDARY_RE.split(raw):
            plower = part.lower()
            if len(plower) >= 2:
                tokens.add(plower)
    return frozenset(tokens)


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One canonical symbol or barrel module node in a PartitionGraph."""

    node_id: int
    chunk_id: str
    file_id: str
    path: str
    language: str
    kind: str
    symbol: str
    qualified_symbol: str
    parent_symbol: str | None
    start_line: int
    end_line: int
    is_test: bool
    is_barrel: bool


@dataclass(frozen=True, slots=True)
class GraphActivation:
    """Per-node activation score and provenance from Seeded Personalized PageRank."""

    node_id: int
    score: float
    rank: int
    via: str | None


@dataclass(frozen=True)
class PartitionGraph:
    """Immutable CSR symbol graph for one pinned ``(partition_id, nonce, cver, rver)``."""

    project_id: str
    partition_id: str
    nonce: str
    chunks_version: int
    references_version: int
    nodes: tuple[GraphNode, ...]
    # CSR outbound: source_id -> target_id
    out_indptr: array[int]
    out_indices: array[int]
    out_kinds: array[int]
    out_resolutions: array[int]
    out_weights: array[float]
    # CSR inbound: target_id -> source_id
    in_indptr: array[int]
    in_indices: array[int]
    in_kinds: array[int]
    in_resolutions: array[int]
    in_weights: array[float]
    # Global stationary hub prior (length == len(nodes))
    global_pagerank: array[float]
    # Lookups
    any_chunk_to_node: Mapping[str, int]
    file_qual_to_node: Mapping[tuple[str, str], int]
    path_qual_to_node: Mapping[tuple[str, str], int]
    symbol_to_nodes: Mapping[str, tuple[int, ...]]
    barrel_file_ids: frozenset[str]
    # barrel_file_id -> tuple of (exported_name_lower, target_node_id)
    barrel_targets_by_file: Mapping[str, tuple[tuple[str, int], ...]]


def _via_label(kind_code: int, is_reverse: bool, seed_symbol: str) -> str:
    kind = EDGE_KINDS[kind_code]
    if kind == "call":
        prefix = "calls" if is_reverse else "called_by"
    elif kind == "inheritance":
        prefix = "inherits" if is_reverse else "implemented_by"
    elif kind == "type_use":
        prefix = "uses_type" if is_reverse else "used_by"
    elif kind == "decorator":
        prefix = "decorates" if is_reverse else "decorated_by"
    elif kind == "export":
        prefix = "reexported_by" if is_reverse else "reexport_of"
    elif kind == "contains":
        prefix = "member_of" if not is_reverse else "contains"
    else:
        prefix = f"{kind}_from" if not is_reverse else f"{kind}_to"
    # Note on direction naming: when seed u propagates forward along u -> v (e.g. u calls v),
    # v was reached *because seed u calls v*, i.e., v is "called_by <seed u>".
    # When seed v propagates backward along u -> v to u, u *calls <seed v>*!
    return f"{prefix}:{seed_symbol}"


def _build_csr(
    num_nodes: int,
    raw_edges: list[tuple[int, int, int, int, float]],
) -> tuple[array[int], array[int], array[int], array[int], array[float]]:
    buckets: list[list[tuple[int, int, int, float]]] = [[] for _ in range(num_nodes)]
    for src, dst, kcode, res, weight in raw_edges:
        buckets[src].append((dst, kcode, res, weight))
    indptr = array("I", [0])
    indices = array("I")
    kinds = array("B")
    resolutions = array("B")
    weights = array("f")
    for bucket in buckets:
        bucket.sort(key=lambda item: (item[0], item[1]))
        for dst, kcode, res, weight in bucket:
            indices.append(dst)
            kinds.append(kcode)
            resolutions.append(res)
            weights.append(weight)
        indptr.append(len(indices))
    return indptr, indices, kinds, resolutions, weights


def _compute_global_pagerank(
    num_nodes: int,
    out_indptr: array[int],
    out_indices: array[int],
    out_weights: array[float],
    in_indptr: array[int],
    in_indices: array[int],
    in_weights: array[float],
    iterations: int = 12,
    alpha: float = 0.85,
) -> array[float]:
    if num_nodes == 0:
        return array("f")
    uniform = 1.0 / num_nodes
    scores = [uniform] * num_nodes
    # Precompute total coupled outbound weight per node
    norm = [0.0] * num_nodes
    for u in range(num_nodes):
        total = 0.0
        for idx in range(out_indptr[u], out_indptr[u + 1]):
            total += out_weights[idx]
        for idx in range(in_indptr[u], in_indptr[u + 1]):
            total += 0.5 * in_weights[idx]
        norm[u] = total

    base_reset = (1.0 - alpha) * uniform
    for _ in range(iterations):
        nxt = [base_reset] * num_nodes
        dangling_mass = 0.0
        for u in range(num_nodes):
            total = norm[u]
            su = scores[u]
            if total <= 1e-12:
                dangling_mass += su
                continue
            scale = alpha * su / total
            for idx in range(out_indptr[u], out_indptr[u + 1]):
                nxt[out_indices[idx]] += scale * out_weights[idx]
            for idx in range(in_indptr[u], in_indptr[u + 1]):
                nxt[in_indices[idx]] += scale * 0.5 * in_weights[idx]
        if dangling_mass > 0.0:
            share = alpha * dangling_mass * uniform
            for i in range(num_nodes):
                nxt[i] += share
        scores = nxt
    return array("f", scores)


def _serialize_graph(graph: PartitionGraph) -> bytes:
    payload = {
        "project_id": graph.project_id,
        "partition_id": graph.partition_id,
        "nonce": graph.nonce,
        "chunks_version": graph.chunks_version,
        "references_version": graph.references_version,
        "nodes": [
            [
                n.node_id,
                n.chunk_id,
                n.file_id,
                n.path,
                n.language,
                n.kind,
                n.symbol,
                n.qualified_symbol,
                n.parent_symbol,
                n.start_line,
                n.end_line,
                n.is_test,
                n.is_barrel,
            ]
            for n in graph.nodes
        ],
        "out_indptr": list(graph.out_indptr),
        "out_indices": list(graph.out_indices),
        "out_kinds": list(graph.out_kinds),
        "out_resolutions": list(graph.out_resolutions),
        "out_weights": [round(float(x), 6) for x in graph.out_weights],
        "in_indptr": list(graph.in_indptr),
        "in_indices": list(graph.in_indices),
        "in_kinds": list(graph.in_kinds),
        "in_resolutions": list(graph.in_resolutions),
        "in_weights": [round(float(x), 6) for x in graph.in_weights],
        "global_pagerank": [round(float(x), 8) for x in graph.global_pagerank],
        "any_chunk_to_node": dict(graph.any_chunk_to_node),
        "barrel_file_ids": sorted(graph.barrel_file_ids),
        "barrel_targets_by_file": {
            fid: [[name, nid] for name, nid in targets]
            for fid, targets in graph.barrel_targets_by_file.items()
        },
    }
    raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _GRAPH_CACHE_MAGIC + zlib.compress(raw_json, level=3)


def _deserialize_graph(blob: bytes) -> PartitionGraph | None:
    if not blob.startswith(_GRAPH_CACHE_MAGIC):
        return None
    try:
        data = json.loads(zlib.decompress(blob[len(_GRAPH_CACHE_MAGIC) :]).decode("utf-8"))
        nodes = tuple(
            GraphNode(
                node_id=int(r[0]),
                chunk_id=str(r[1]),
                file_id=str(r[2]),
                path=str(r[3]),
                language=str(r[4]),
                kind=str(r[5]),
                symbol=str(r[6]),
                qualified_symbol=str(r[7]),
                parent_symbol=None if r[8] is None else str(r[8]),
                start_line=int(r[9]),
                end_line=int(r[10]),
                is_test=bool(r[11]),
                is_barrel=bool(r[12]),
            )
            for r in data["nodes"]
        )
        file_qual_to_node: dict[tuple[str, str], int] = {}
        path_qual_to_node: dict[tuple[str, str], int] = {}
        sym_buckets: dict[str, list[int]] = {}
        for node in nodes:
            if not node.is_barrel and node.qualified_symbol:
                file_qual_to_node[(node.file_id, node.qualified_symbol)] = node.node_id
                path_qual_to_node[(node.path, node.qualified_symbol)] = node.node_id
                if node.symbol:
                    sym_buckets.setdefault(node.symbol, []).append(node.node_id)
        barrel_targets = {
            str(fid): tuple((str(pair[0]), int(pair[1])) for pair in pairs)
            for fid, pairs in data["barrel_targets_by_file"].items()
        }
        return PartitionGraph(
            project_id=str(data["project_id"]),
            partition_id=str(data["partition_id"]),
            nonce=str(data["nonce"]),
            chunks_version=int(data["chunks_version"]),
            references_version=int(data["references_version"]),
            nodes=nodes,
            out_indptr=array("I", data["out_indptr"]),
            out_indices=array("I", data["out_indices"]),
            out_kinds=array("B", data["out_kinds"]),
            out_resolutions=array("B", data["out_resolutions"]),
            out_weights=array("f", data["out_weights"]),
            in_indptr=array("I", data["in_indptr"]),
            in_indices=array("I", data["in_indices"]),
            in_kinds=array("B", data["in_kinds"]),
            in_resolutions=array("B", data["in_resolutions"]),
            in_weights=array("f", data["in_weights"]),
            global_pagerank=array("f", data["global_pagerank"]),
            any_chunk_to_node={str(k): int(v) for k, v in data["any_chunk_to_node"].items()},
            file_qual_to_node=file_qual_to_node,
            path_qual_to_node=path_qual_to_node,
            symbol_to_nodes={sym: tuple(ids) for sym, ids in sym_buckets.items()},
            barrel_file_ids=frozenset(str(x) for x in data["barrel_file_ids"]),
            barrel_targets_by_file=barrel_targets,
        )
    except Exception:
        logger.debug("Ignoring unreadable sidecar graph cache", exc_info=True)
        return None


class GraphIndexService:
    """Compiles, caches, and queries snapshot-pinned symbol graphs for Hybrid GraphRAG."""

    def __init__(
        self,
        store: LanceStore,
        reference_service: ReferenceService,
        *,
        enabled: bool = True,
        cache_limit: int = 8,
        persist_sidecar: bool = True,
    ) -> None:
        self.store = store
        self.reference_service = reference_service
        self.enabled = enabled
        self.cache_limit = max(1, cache_limit)
        self.persist_sidecar = persist_sidecar
        self._cache: OrderedDict[tuple[str, str, int, int], PartitionGraph] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._compile_locks: dict[str, threading.Lock] = {}
        self._compile_locks_guard = threading.Lock()

    def _partition_lock(self, partition_id: str) -> threading.Lock:
        with self._compile_locks_guard:
            lock = self._compile_locks.get(partition_id)
            if lock is None:
                lock = threading.Lock()
                self._compile_locks[partition_id] = lock
            return lock

    def _sidecar_path(
        self, partition_id: str, nonce: str, chunks_version: int, references_version: int
    ) -> Path:
        return (
            self.store.directory
            / "graph-cache"
            / partition_id
            / f"{nonce}-{chunks_version}-{references_version}.bin"
        )

    def clear_memory_cache(self) -> None:
        """Evict all in-memory compiled partition graphs without deleting sidecars."""
        with self._cache_lock:
            self._cache.clear()

    def get_partition_graph(
        self, project_id: str, *, partition_id: str | None = None
    ) -> PartitionGraph | None:
        """Return the compiled PartitionGraph for the current partition snapshot, or None."""
        identity = self.store.partition_graph_identity(project_id, partition_id=partition_id)
        if identity is None:
            return None
        physical, nonce, cver, rver = identity
        cache_key = (physical, nonce, cver, rver)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        with self._partition_lock(physical):
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
                    return cached

            sidecar = self._sidecar_path(physical, nonce, cver, rver)
            if self.persist_sidecar and sidecar.is_file():
                with contextlib.suppress(OSError):
                    loaded = _deserialize_graph(sidecar.read_bytes())
                    if loaded is not None and (
                        loaded.partition_id == physical
                        and loaded.nonce == nonce
                        and loaded.chunks_version == cver
                        and loaded.references_version == rver
                    ):
                        self._remember(cache_key, loaded)
                        return loaded

            compiled_refs = self.reference_service.compile_partition_graph(
                project_id, partition_id=physical, snapshot_version=rver
            )
            if compiled_refs is None:
                return None
            previews, window_to_canonical = self.store.list_partition_chunk_metadata(
                project_id, partition_id=physical, version=cver
            )
            graph = self._assemble_graph(
                project_id=project_id,
                partition_id=physical,
                nonce=nonce,
                chunks_version=cver,
                references_version=rver,
                compiled=compiled_refs,
                previews=previews,
                window_to_canonical=window_to_canonical,
            )
            self._remember(cache_key, graph)
            if self.persist_sidecar:
                self._write_sidecar(sidecar, graph)
            return graph

    def _remember(self, key: tuple[str, str, int, int], graph: PartitionGraph) -> None:
        with self._cache_lock:
            self._cache[key] = graph
            self._cache.move_to_end(key)
            # Evict older versions of the same physical partition immediately
            stale_same_partition = [
                existing_key
                for existing_key in self._cache
                if existing_key[0] == key[0] and existing_key != key
            ]
            for stale_key in stale_same_partition:
                self._cache.pop(stale_key, None)
            while len(self._cache) > self.cache_limit:
                self._cache.popitem(last=False)

    def _write_sidecar(self, sidecar: Path, graph: PartitionGraph) -> None:
        with contextlib.suppress(OSError):
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            tmp = sidecar.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(_serialize_graph(graph))
            os.replace(tmp, sidecar)
            # Best-effort cleanup of superseded sidecars in this partition folder
            for entry in sidecar.parent.iterdir():
                if entry.name != sidecar.name and entry.suffix == ".bin":
                    with contextlib.suppress(OSError):
                        entry.unlink()

    @staticmethod
    def _assemble_graph(
        *,
        project_id: str,
        partition_id: str,
        nonce: str,
        chunks_version: int,
        references_version: int,
        compiled: CompiledReferenceGraph,
        previews: list[ChunkPreview],
        window_to_canonical: dict[str, str],
    ) -> PartitionGraph:
        # Group chunk previews by (file_id, qualified_symbol) and track barrel module chunks
        decl_chunks: dict[tuple[str, str], list[ChunkPreview]] = {}
        module_chunks_by_file: dict[str, list[ChunkPreview]] = {}
        for preview in previews:
            fid = preview.file_id or ""
            if preview.qualified_symbol and preview.symbol:
                decl_chunks.setdefault((fid, preview.qualified_symbol), []).append(preview)
            elif preview.kind == "module" and fid:
                module_chunks_by_file.setdefault(fid, []).append(preview)

        nodes: list[GraphNode] = []
        any_chunk_to_node: dict[str, int] = {}
        file_qual_to_node: dict[tuple[str, str], int] = {}
        path_qual_to_node: dict[tuple[str, str], int] = {}
        sym_buckets: dict[str, list[int]] = {}
        barrel_file_ids: set[str] = set()

        # 1. Create canonical declaration nodes
        for decl in compiled.declarations:
            parts = decl_chunks.get((decl.file_id, decl.qualified_symbol))
            if not parts:
                continue
            parts_sorted = sorted(parts, key=lambda c: (c.part_index, c.start_line, c.chunk_id))
            first = parts_sorted[0]
            end_line = max(decl.end_line, max(p.end_line for p in parts_sorted))
            node_id = len(nodes)
            is_barrel = decl.path in compiled.barrel_files
            node = GraphNode(
                node_id=node_id,
                chunk_id=first.chunk_id,
                file_id=decl.file_id,
                path=decl.path,
                language=decl.language,
                kind=decl.kind,
                symbol=decl.symbol,
                qualified_symbol=decl.qualified_symbol,
                parent_symbol=first.parent_symbol,
                start_line=min(decl.start_line, first.start_line),
                end_line=end_line,
                is_test=any(p.is_test for p in parts_sorted),
                is_barrel=is_barrel,
            )
            nodes.append(node)
            file_qual_to_node[(decl.file_id, decl.qualified_symbol)] = node_id
            path_qual_to_node[(decl.path, decl.qualified_symbol)] = node_id
            sym_buckets.setdefault(decl.symbol, []).append(node_id)
            for p in parts_sorted:
                any_chunk_to_node[p.chunk_id] = node_id

        # Also include chunk declarations that had no `declaration` row (e.g. non-structural
        # languages or synthetic container skeletons) so search chunk ids map cleanly.
        for (fid, qual), parts in sorted(decl_chunks.items()):
            if (fid, qual) in file_qual_to_node:
                continue
            parts_sorted = sorted(parts, key=lambda c: (c.part_index, c.start_line, c.chunk_id))
            first = parts_sorted[0]
            if not first.symbol:
                continue
            node_id = len(nodes)
            node = GraphNode(
                node_id=node_id,
                chunk_id=first.chunk_id,
                file_id=fid,
                path=first.path,
                language=first.language,
                kind=first.kind.removesuffix("_part"),
                symbol=first.symbol,
                qualified_symbol=qual,
                parent_symbol=first.parent_symbol,
                start_line=first.start_line,
                end_line=max(p.end_line for p in parts_sorted),
                is_test=any(p.is_test for p in parts_sorted),
                is_barrel=first.path in compiled.barrel_files,
            )
            nodes.append(node)
            file_qual_to_node[(fid, qual)] = node_id
            path_qual_to_node[(first.path, qual)] = node_id
            sym_buckets.setdefault(first.symbol, []).append(node_id)
            for p in parts_sorted:
                any_chunk_to_node[p.chunk_id] = node_id

        # 2. Create barrel module nodes so pure re-export files (`index.ts`, `__init__.py`)
        # map directly from their `module` chunk_id into the graph.
        barrel_node_by_file: dict[str, int] = {}
        reexport_file_ids = {edge.barrel_file_id for edge in compiled.barrel_reexports}
        for fid, mod_chunks in sorted(module_chunks_by_file.items()):
            first_mod = mod_chunks[0]
            if first_mod.path in compiled.barrel_files or fid in reexport_file_ids:
                barrel_file_ids.add(fid)
                node_id = len(nodes)
                mod_node = GraphNode(
                    node_id=node_id,
                    chunk_id=first_mod.chunk_id,
                    file_id=fid,
                    path=first_mod.path,
                    language=first_mod.language,
                    kind="module",
                    symbol="",
                    qualified_symbol=first_mod.path,
                    parent_symbol=None,
                    start_line=first_mod.start_line,
                    end_line=max(m.end_line for m in mod_chunks),
                    is_test=any(m.is_test for m in mod_chunks),
                    is_barrel=True,
                )
                nodes.append(mod_node)
                barrel_node_by_file[fid] = node_id
                for m in mod_chunks:
                    any_chunk_to_node[m.chunk_id] = node_id

        # Map every split embedding window chunk_id to its canonical chunk's node_id
        for win_cid, canon_cid in window_to_canonical.items():
            mapped = any_chunk_to_node.get(canon_cid)
            if mapped is not None:
                any_chunk_to_node[win_cid] = mapped

        # 3. Assemble directed CSR edges (deduplicated per (src, dst, kind_code))
        best_edges: dict[tuple[int, int, int], tuple[int, float]] = {}

        def record_edge(src_id: int, dst_id: int, kcode: int, res_code: int) -> None:
            if src_id == dst_id:
                return
            base_w = _FORWARD_KIND_WEIGHT.get(kcode, 0.5)
            w = base_w if res_code == _RESOLUTION_EXACT else base_w * _LIKELY_WEIGHT_MULTIPLIER
            key = (src_id, dst_id, kcode)
            prev = best_edges.get(key)
            if prev is None or w > prev[1]:
                best_edges[key] = (res_code, w)

        for edge in compiled.edges:
            dst_id = file_qual_to_node.get((edge.target_file_id, edge.target_qualified_symbol))
            if dst_id is None:
                continue
            src_id: int | None = None
            if edge.source_qualified_symbol is not None:
                src_id = file_qual_to_node.get((edge.source_file_id, edge.source_qualified_symbol))
                if src_id is None and "." in edge.source_qualified_symbol:
                    outer = edge.source_qualified_symbol.rpartition(".")[0]
                    src_id = file_qual_to_node.get((edge.source_file_id, outer))
            else:
                src_id = barrel_node_by_file.get(edge.source_file_id)
            if src_id is None:
                continue
            kcode = _KIND_TO_CODE.get(edge.kind)
            if kcode is None:
                continue
            res_code = _RESOLUTION_EXACT if edge.resolution == "exact" else _RESOLUTION_LIKELY
            record_edge(src_id, dst_id, kcode, res_code)

        # Structural containment edges (enclosing container -> member)
        contains_code = _KIND_TO_CODE["contains"]
        for node in nodes:
            if node.is_barrel or not node.parent_symbol:
                continue
            parent_id = file_qual_to_node.get((node.file_id, node.parent_symbol))
            if parent_id is not None:
                record_edge(parent_id, node.node_id, contains_code, _RESOLUTION_EXACT)

        # Barrel re-export edges
        export_code = _KIND_TO_CODE["export"]
        barrel_targets_map: dict[str, list[tuple[str, int]]] = {}
        for brex in compiled.barrel_reexports:
            target_id = file_qual_to_node.get((brex.target_file_id, brex.target_qualified_symbol))
            if target_id is None:
                continue
            barrel_targets_map.setdefault(brex.barrel_file_id, []).append(
                (brex.exported_name.lower(), target_id)
            )
            barrel_src = barrel_node_by_file.get(brex.barrel_file_id)
            if barrel_src is not None:
                record_edge(barrel_src, target_id, export_code, _RESOLUTION_EXACT)

        out_raw: list[tuple[int, int, int, int, float]] = []
        in_raw: list[tuple[int, int, int, int, float]] = []
        for (src_id, dst_id, kcode), (res_code, w) in best_edges.items():
            out_raw.append((src_id, dst_id, kcode, res_code, w))
            rev_w = w * _REVERSE_KIND_MULTIPLIER.get(kcode, 0.6)
            in_raw.append((dst_id, src_id, kcode, res_code, rev_w))

        num_nodes = len(nodes)
        out_indptr, out_indices, out_kinds, out_res, out_weights = _build_csr(num_nodes, out_raw)
        in_indptr, in_indices, in_kinds, in_res, in_weights = _build_csr(num_nodes, in_raw)
        global_pr = _compute_global_pagerank(
            num_nodes,
            out_indptr,
            out_indices,
            out_weights,
            in_indptr,
            in_indices,
            in_weights,
        )

        return PartitionGraph(
            project_id=project_id,
            partition_id=partition_id,
            nonce=nonce,
            chunks_version=chunks_version,
            references_version=references_version,
            nodes=tuple(nodes),
            out_indptr=out_indptr,
            out_indices=out_indices,
            out_kinds=out_kinds,
            out_resolutions=out_res,
            out_weights=out_weights,
            in_indptr=in_indptr,
            in_indices=in_indices,
            in_kinds=in_kinds,
            in_resolutions=in_res,
            in_weights=in_weights,
            global_pagerank=global_pr,
            any_chunk_to_node=any_chunk_to_node,
            file_qual_to_node=file_qual_to_node,
            path_qual_to_node=path_qual_to_node,
            symbol_to_nodes={sym: tuple(ids) for sym, ids in sym_buckets.items()},
            barrel_file_ids=frozenset(barrel_file_ids),
            barrel_targets_by_file={
                fid: tuple(dict.fromkeys(items)) for fid, items in barrel_targets_map.items()
            },
        )

    def seeded_ppr(
        self,
        graph: PartitionGraph,
        seed_weights: Mapping[int, float],
        *,
        alpha: float = _PPR_ALPHA,
        iterations: int = _PPR_ITERATIONS,
    ) -> dict[int, GraphActivation]:
        """Run Specificity-Adjusted Personalized PageRank from *seed_weights*."""
        num_nodes = len(graph.nodes)
        if num_nodes == 0 or not seed_weights:
            return {}
        total_seed = sum(w for w in seed_weights.values() if w > 0.0)
        if total_seed <= 1e-12:
            return {}

        seed_dist: dict[int, float] = {
            nid: w / total_seed for nid, w in seed_weights.items() if 0 <= nid < num_nodes and w > 0
        }
        if not seed_dist:
            return {}

        out_indptr = graph.out_indptr
        out_indices = graph.out_indices
        out_kinds = graph.out_kinds
        out_weights = graph.out_weights
        in_indptr = graph.in_indptr
        in_indices = graph.in_indices
        in_kinds = graph.in_kinds
        in_weights = graph.in_weights

        # Inverse-degree specificity damping so ubiquitous utility hubs do not absorb mass
        hub_damping = [
            1.0
            / (
                (
                    1.0
                    + (in_indptr[i + 1] - in_indptr[i])
                    + 0.25 * (out_indptr[i + 1] - out_indptr[i])
                )
                ** _HUB_DEGREE_EXPONENT
            )
            for i in range(num_nodes)
        ]

        p = [0.0] * num_nodes
        best_contrib = [0.0] * num_nodes
        # provenance: (origin_seed_node_id, edge_kind_code, is_reverse_step)
        provenance: dict[int, tuple[int, int, bool]] = {}
        origin_seed: dict[int, int] = {}

        for nid, sw in seed_dist.items():
            p[nid] = sw
            origin_seed[nid] = nid

        reset_scale = 1.0 - alpha
        for _ in range(iterations):
            nxt = [0.0] * num_nodes
            for nid, sw in seed_dist.items():
                nxt[nid] += reset_scale * sw

            for u in range(num_nodes):
                pu = p[u]
                if pu <= 1e-9:
                    continue
                o_start, o_end = out_indptr[u], out_indptr[u + 1]
                i_start, i_end = in_indptr[u], in_indptr[u + 1]
                if o_start == o_end and i_start == i_end:
                    nxt[u] += alpha * pu
                    continue

                total_w = 0.0
                for idx in range(o_start, o_end):
                    v = out_indices[idx]
                    total_w += out_weights[idx] * hub_damping[v]
                for idx in range(i_start, i_end):
                    v = in_indices[idx]
                    total_w += in_weights[idx] * hub_damping[v]
                if total_w <= 1e-12:
                    nxt[u] += alpha * pu
                    continue

                scale = alpha * pu / total_w
                src_origin = origin_seed.get(u, u)
                for idx in range(o_start, o_end):
                    v = out_indices[idx]
                    flow = scale * out_weights[idx] * hub_damping[v]
                    nxt[v] += flow
                    if src_origin != v and flow > best_contrib[v]:
                        best_contrib[v] = flow
                        provenance[v] = (u, out_kinds[idx], False)
                        if v not in seed_dist:
                            origin_seed[v] = src_origin
                for idx in range(i_start, i_end):
                    v = in_indices[idx]
                    flow = scale * in_weights[idx] * hub_damping[v]
                    nxt[v] += flow
                    if src_origin != v and flow > best_contrib[v]:
                        best_contrib[v] = flow
                        provenance[v] = (u, in_kinds[idx], True)
                        if v not in seed_dist:
                            origin_seed[v] = src_origin
            p = nxt

        # Subtract self-reset baseline so isolated nodes with zero neighbor flow score 0.0
        raw_activated: list[tuple[int, float]] = []
        global_pr = graph.global_pagerank
        for v in range(num_nodes):
            propagated = p[v] - reset_scale * seed_dist.get(v, 0.0)
            if propagated <= _MIN_GRAPH_ACTIVATION:
                continue
            # Require genuine cross-node evidence (either non-seed or corroborated by another node)
            if v not in provenance:
                continue
            prior = (float(global_pr[v]) ** _PRIOR_DAMPING_EXPONENT) + 1e-6
            adj_score = propagated / prior
            if adj_score > _MIN_GRAPH_ACTIVATION:
                raw_activated.append((v, adj_score))

        raw_activated.sort(key=lambda item: (-item[1], graph.nodes[item[0]].chunk_id))
        activations: dict[int, GraphActivation] = {}
        rank = 0
        prev_score: float | None = None
        for pos, (nid, score) in enumerate(raw_activated, start=1):
            rounded = round(score, 8)
            if rounded != prev_score:
                rank = pos
                prev_score = rounded
            prov = provenance.get(nid)
            via_str: str | None = None
            if prov is not None:
                seed_node = graph.nodes[prov[0]]
                seed_label = seed_node.qualified_symbol or seed_node.path
                via_str = _via_label(prov[1], prov[2], seed_label)
            activations[nid] = GraphActivation(
                node_id=nid,
                score=score,
                rank=rank,
                via=via_str,
            )
        return activations

    def expand_hybrid_rows(
        self,
        project_id: str,
        partition_id: str | None,
        query_text: str,
        rows: list[dict[str, Any]],
        condition: str | None,
        limit: int,
        *,
        query_vector: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Enrich partition candidates with PPR graph ranks, barrel transfer, and discovery."""
        if not self.enabled or not rows:
            return rows
        try:
            graph = self.get_partition_graph(project_id, partition_id=partition_id)
        except Exception:
            logger.debug("Graph compilation skipped for %s", project_id, exc_info=True)
            return rows
        if graph is None or not graph.nodes:
            return rows

        # Compute initial per-row seed strength from top partition-local vector & FTS anchors
        vec_sorted = sorted(
            (r for r in rows if "_distance" in r),
            key=lambda r: (float(r["_distance"]), str(r["chunk_id"])),
        )
        vec_ranks: dict[str, int] = {}
        for pos, r in enumerate(vec_sorted, start=1):
            vec_ranks.setdefault(str(r["chunk_id"]), pos)

        fts_ranks: dict[str, int] = {}
        row_by_cid: dict[str, dict[str, Any]] = {}
        for r in rows:
            cid = str(r["chunk_id"])
            row_by_cid.setdefault(cid, r)
            if "_fts_local_rank" in r:
                f_rank = int(r["_fts_local_rank"])
                prev_f = fts_ranks.get(cid)
                if prev_f is None or f_rank < prev_f:
                    fts_ranks[cid] = f_rank

        seed_weights: dict[int, float] = {}
        q_terms = _query_terms(query_text)
        barrel_candidate_file_ids: set[str] = set()

        for cid, r in row_by_cid.items():
            v_rank = vec_ranks.get(cid)
            fts_rank = fts_ranks.get(cid)
            v_seed = (
                0.85 / (v_rank**1.5)
                if v_rank is not None and v_rank <= _MAX_HYBRID_SEED_RANK
                else 0.0
            )
            f_seed = (
                1.35 / (fts_rank**1.5)
                if fts_rank is not None and fts_rank <= _MAX_HYBRID_SEED_RANK
                else 0.0
            )
            row_seed = v_seed + f_seed
            if v_seed > 0.0 and f_seed > 0.0:
                row_seed *= 1.5
            if row_seed <= 0.0:
                continue

            fid = str(r.get("file_id") or "")
            barrel_targets = graph.barrel_targets_by_file.get(fid)
            nid = graph.any_chunk_to_node.get(cid)
            is_barrel_row = (nid is not None and graph.nodes[nid].is_barrel) or (
                barrel_targets is not None and r.get("kind") == "module"
            )

            if is_barrel_row and barrel_targets:
                barrel_candidate_file_ids.add(fid)
                matched_targets = [
                    target_nid
                    for exp_name, target_nid in barrel_targets
                    if exp_name in q_terms
                    or any(term in exp_name for term in q_terms if len(term) >= 3)
                ]
                chosen_targets = (
                    matched_targets if matched_targets else [t_nid for _, t_nid in barrel_targets]
                )
                if chosen_targets:
                    share = (0.95 * row_seed) / len(chosen_targets)
                    for target_nid in chosen_targets:
                        seed_weights[target_nid] = seed_weights.get(target_nid, 0.0) + share
            if nid is not None:
                seed_weights[nid] = seed_weights.get(nid, 0.0) + row_seed

        if not seed_weights:
            return rows

        activations = self.seeded_ppr(graph, seed_weights)
        if not activations and not barrel_candidate_file_ids:
            return rows

        existing_cids = {str(r["chunk_id"]) for r in rows}
        existing_node_ids = {
            graph.any_chunk_to_node[cid] for cid in existing_cids if cid in graph.any_chunk_to_node
        }

        # Discover top multi-hop / barrel-forwarded candidate chunks not yet in `rows`
        max_discover = min(max(limit, 8), 16)
        discover_chunk_ids: list[str] = []
        discover_meta: dict[str, GraphActivation] = {}

        # Ensure direct barrel targets matching query terms are hydrated even on 1-edge barrels
        for fid in barrel_candidate_file_ids:
            for exp_name, target_nid in graph.barrel_targets_by_file.get(fid, ()):
                if target_nid in existing_node_ids:
                    continue
                if exp_name in q_terms or len(graph.barrel_targets_by_file.get(fid, ())) <= 8:
                    target_node = graph.nodes[target_nid]
                    act = activations.get(target_nid)
                    if act is None:
                        act = GraphActivation(
                            node_id=target_nid,
                            score=seed_weights.get(target_nid, 0.01),
                            rank=1,
                            via=f"reexport_of:{target_node.qualified_symbol}",
                        )
                        activations[target_nid] = act

        ranked_activations = sorted(
            activations.values(), key=lambda a: (-a.score, graph.nodes[a.node_id].chunk_id)
        )
        peak_score = ranked_activations[0].score if ranked_activations else 0.0
        min_rrf_score = peak_score * _RELATIVE_RRF_GRAPH_FLOOR
        ranked_activations = [act for act in ranked_activations if act.score >= min_rrf_score][
            :max_discover
        ]
        # Re-normalize sequential competition ranks if synthetic barrel activations were added
        reindexed: dict[int, GraphActivation] = {}
        rank_counter = 0
        prev_s: float | None = None
        for idx, act in enumerate(ranked_activations, start=1):
            rounded = round(act.score, 8)
            if rounded != prev_s:
                rank_counter = idx
                prev_s = rounded
            reindexed[act.node_id] = GraphActivation(
                node_id=act.node_id,
                score=act.score,
                rank=rank_counter,
                via=act.via,
            )

        for nid, act in reindexed.items():
            if nid in existing_node_ids:
                continue
            target_cid = graph.nodes[nid].chunk_id
            if target_cid not in existing_cids:
                discover_chunk_ids.append(target_cid)
                discover_meta[target_cid] = act
                if len(discover_chunk_ids) >= max_discover:
                    break

        hydrated: list[dict[str, Any]] = []
        if discover_chunk_ids:
            hydrated = self.store.fetch_search_rows_by_ids(
                project_id,
                discover_chunk_ids,
                condition=condition,
                partition_id=partition_id,
                probe_vectors=[query_vector] if query_vector is not None else None,
            )

        combined_rows = [dict(r) for r in rows] + hydrated
        present_node_ids = {
            graph.any_chunk_to_node[str(r["chunk_id"])]
            for r in combined_rows
            if str(r["chunk_id"]) in graph.any_chunk_to_node
        }

        # Choose at most one representative chunk (plus any embedding windows of it)
        # per symbol node so multi-part declarations do not crowd out distinct neighbors.
        best_rep_by_node: dict[int, tuple[tuple[int, int, int, str], str, tuple[Any, ...]]] = {}
        for r in combined_rows:
            cid = str(r["chunk_id"])
            nid = graph.any_chunk_to_node.get(cid)
            if nid is None or graph.nodes[nid].is_barrel:
                continue
            rank_key = (
                fts_ranks.get(cid, 999_999),
                vec_ranks.get(cid, 999_999),
                int(r.get("part_index") or 0),
                cid,
            )
            win_sig = (
                r.get("file_id"),
                r.get("source_start_byte"),
                r.get("source_end_byte"),
                r.get("source_content_hash"),
            )
            prev_rep = best_rep_by_node.get(nid)
            if prev_rep is None or rank_key < prev_rep[0]:
                best_rep_by_node[nid] = (rank_key, cid, win_sig)

        for r in combined_rows:
            cid = str(r["chunk_id"])
            nid = graph.any_chunk_to_node.get(cid)
            if nid is not None and not graph.nodes[nid].is_barrel:
                rep = best_rep_by_node.get(nid)
                win_sig = (
                    r.get("file_id"),
                    r.get("source_start_byte"),
                    r.get("source_end_byte"),
                    r.get("source_content_hash"),
                )
                is_rep = rep is not None and (
                    cid == rep[1] or (win_sig[1] is not None and win_sig == rep[2])
                )
                act = reindexed.get(nid) if is_rep else None
                if act is not None:
                    r["_graph_local_rank"] = act.rank
                    r["_graph_score"] = act.score
                    if act.via:
                        r["_graph_via"] = act.via
            fid = str(r.get("file_id") or "")
            if (
                fid in barrel_candidate_file_ids
                and (r.get("kind") == "module" or (nid is not None and graph.nodes[nid].is_barrel))
                and any(
                    t_nid in present_node_ids and not graph.nodes[t_nid].is_barrel
                    for _, t_nid in graph.barrel_targets_by_file.get(fid, ())
                )
            ):
                r["_barrel_damped"] = True

        return combined_rows

    def expand_example_rows(
        self,
        project_id: str,
        partition_id: str | None,
        outbound_symbols: Mapping[str, float],
        rows: list[dict[str, Any]],
        condition: str | None,
        limit: int,
        *,
        probe_vectors: Sequence[list[float]] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse vector similarity with AST structural co-citation for ``search_by_example``."""
        if not self.enabled or (not rows and not outbound_symbols):
            return rows
        try:
            graph = self.get_partition_graph(project_id, partition_id=partition_id)
        except Exception:
            logger.debug("Example graph expansion skipped for %s", project_id, exc_info=True)
            return rows
        if graph is None or not graph.nodes:
            return rows

        # 1. Structural co-citation: find declarations in the repo that reference the SAME
        # callees / base types / decorators that the pasted example snippet references!
        cocitation_scores: dict[int, float] = {}
        cocitation_via: dict[int, str] = {}
        in_indptr = graph.in_indptr
        in_indices = graph.in_indices
        in_weights = graph.in_weights

        for sym, sym_weight in outbound_symbols.items():
            target_nids = graph.symbol_to_nodes.get(sym, ())
            if not target_nids or len(target_nids) > 16:
                continue
            per_target = sym_weight / len(target_nids)
            for t_nid in target_nids:
                deg = in_indptr[t_nid + 1] - in_indptr[t_nid]
                if deg == 0:
                    continue
                idf_scale = 1.0 / (1.0 + math.log1p(deg))
                for idx in range(in_indptr[t_nid], in_indptr[t_nid + 1]):
                    caller_nid = in_indices[idx]
                    if graph.nodes[caller_nid].is_barrel:
                        continue
                    gain = per_target * in_weights[idx] * idf_scale
                    cocitation_scores[caller_nid] = cocitation_scores.get(caller_nid, 0.0) + gain
                    cocitation_via.setdefault(
                        caller_nid, f"shares_ref:{graph.nodes[t_nid].qualified_symbol}"
                    )

        # 2. Also run light PPR from top vector seeds
        seed_weights: dict[int, float] = {}
        for pos, r in enumerate(rows[: min(len(rows), 16)], start=1):
            nid = graph.any_chunk_to_node.get(str(r["chunk_id"]))
            if nid is not None and not graph.nodes[nid].is_barrel:
                seed_weights[nid] = seed_weights.get(nid, 0.0) + (1.0 / (60 + pos))

        ppr_acts = self.seeded_ppr(graph, seed_weights) if seed_weights else {}

        combined_graph_score: dict[int, tuple[float, str | None]] = {}
        max_cocit = max(cocitation_scores.values(), default=0.0)
        max_ppr = max((a.score for a in ppr_acts.values()), default=0.0)
        for nid in set(cocitation_scores) | set(ppr_acts):
            c_norm = (
                (cocitation_scores[nid] / max_cocit)
                if (nid in cocitation_scores and max_cocit > 0)
                else 0.0
            )
            p_norm = (ppr_acts[nid].score / max_ppr) if (nid in ppr_acts and max_ppr > 0) else 0.0
            score = 0.70 * c_norm + 0.30 * p_norm
            if score > 0.0:
                via = cocitation_via.get(nid) or (ppr_acts[nid].via if nid in ppr_acts else None)
                combined_graph_score[nid] = (score, via)

        if not combined_graph_score:
            return rows

        ranked_nids = sorted(
            combined_graph_score,
            key=lambda n: (-combined_graph_score[n][0], graph.nodes[n].chunk_id),
        )
        graph_ranks: dict[int, int] = {nid: pos for pos, nid in enumerate(ranked_nids, start=1)}

        existing_cids = {str(r["chunk_id"]) for r in rows}
        existing_nids = {
            graph.any_chunk_to_node[cid] for cid in existing_cids if cid in graph.any_chunk_to_node
        }
        discover_cids = [
            graph.nodes[nid].chunk_id
            for nid in ranked_nids
            if nid not in existing_nids and graph.nodes[nid].chunk_id not in existing_cids
        ][: min(limit, 12)]

        hydrated = (
            self.store.fetch_search_rows_by_ids(
                project_id,
                discover_cids,
                condition=condition,
                partition_id=partition_id,
                probe_vectors=probe_vectors,
            )
            if discover_cids
            else []
        )

        # Fuse cosine similarity with bounded structural co-citation/PPR boost in [0, 1]
        all_rows = [dict(r) for r in rows] + hydrated
        for r in all_rows:
            cid = str(r["chunk_id"])
            nid = graph.any_chunk_to_node.get(cid)
            base_cos = max(0.0, min(1.0, float(r.get("score", 0.0))))
            if nid is not None and nid in graph_ranks:
                g_score, g_via = combined_graph_score[nid]
                g_rank = graph_ranks[nid]
                r["_graph_local_rank"] = g_rank
                r["_graph_score"] = round(g_score, 6)
                if g_via:
                    r["_graph_via"] = g_via
                boost = 0.25 * (61.0 / (60.0 + g_rank)) * min(1.0, g_score)
                fused = min(1.0, base_cos + boost)
            else:
                fused = base_cos
            r["_relevance_score"] = fused
            r["score"] = fused

        all_rows.sort(
            key=lambda r: (
                -float(r.get("_relevance_score", 0.0)),
                float(r.get("_distance", float("inf"))),
                str(r["chunk_id"]),
            )
        )
        return all_rows[:limit]

    def ego_context(
        self,
        project_id: str,
        chunk_id: str,
        *,
        partition_id: str | None = None,
        max_per_group: int = 5,
    ) -> ChunkGraphContext | None:
        """Return compact 1-hop structural neighborhood around *chunk_id*, or None."""
        if not self.enabled:
            return None
        graph = self.get_partition_graph(project_id, partition_id=partition_id)
        if graph is None:
            return None
        nid = graph.any_chunk_to_node.get(chunk_id)
        if nid is None:
            return None

        def to_ref(other: GraphNode, relation: str, res_code: int) -> GraphNeighborRef:
            return GraphNeighborRef(
                chunk_id=other.chunk_id,
                path=other.path,
                language=other.language,
                kind=other.kind,
                symbol=other.symbol or other.path,
                qualified_symbol=other.qualified_symbol,
                start_line=other.start_line,
                end_line=other.end_line,
                relation=relation,
                resolution="exact" if res_code == _RESOLUTION_EXACT else "likely",
            )

        callers: list[tuple[int, float, GraphNeighborRef]] = []
        callees: list[tuple[int, float, GraphNeighborRef]] = []
        supertypes: list[tuple[int, float, GraphNeighborRef]] = []
        subtypes: list[tuple[int, float, GraphNeighborRef]] = []
        related: list[tuple[int, float, GraphNeighborRef]] = []

        # Outbound edges from `nid`: nid -> other
        for idx in range(graph.out_indptr[nid], graph.out_indptr[nid + 1]):
            other_id = graph.out_indices[idx]
            other = graph.nodes[other_id]
            if other.is_barrel:
                continue
            kname = EDGE_KINDS[graph.out_kinds[idx]]
            res = graph.out_resolutions[idx]
            w = float(graph.out_weights[idx])
            if kname == "call":
                callees.append((res, w, to_ref(other, "calls", res)))
            elif kname == "inheritance":
                supertypes.append((res, w, to_ref(other, "inherits", res)))
            elif kname != "import":
                rel = (
                    "contains"
                    if kname == "contains"
                    else f"uses_{kname}"
                    if kname == "type_use"
                    else kname
                )
                related.append((res, w, to_ref(other, rel, res)))

        # Inbound edges to `nid`: other -> nid
        for idx in range(graph.in_indptr[nid], graph.in_indptr[nid + 1]):
            other_id = graph.in_indices[idx]
            other = graph.nodes[other_id]
            if other.is_barrel:
                continue
            kname = EDGE_KINDS[graph.in_kinds[idx]]
            res = graph.in_resolutions[idx]
            w = float(graph.in_weights[idx])
            if kname == "call":
                callers.append((res, w, to_ref(other, "called_by", res)))
            elif kname == "inheritance":
                subtypes.append((res, w, to_ref(other, "implemented_by", res)))
            elif kname != "import":
                rel = (
                    "member_of"
                    if kname == "contains"
                    else "used_as_type_by"
                    if kname == "type_use"
                    else f"{kname}_by"
                )
                related.append((res, w, to_ref(other, rel, res)))

        def top_refs(
            items: list[tuple[int, float, GraphNeighborRef]],
        ) -> list[GraphNeighborRef]:
            items.sort(key=lambda x: (-x[0], -x[1], x[2].path, x[2].start_line, x[2].chunk_id))
            seen: set[str] = set()
            out: list[GraphNeighborRef] = []
            for _, _, ref in items:
                if ref.chunk_id in seen:
                    continue
                seen.add(ref.chunk_id)
                out.append(ref)
                if len(out) >= max_per_group:
                    break
            return out

        ctx = ChunkGraphContext(
            callers=top_refs(callers),
            callees=top_refs(callees),
            supertypes=top_refs(supertypes),
            subtypes=top_refs(subtypes),
            related=top_refs(related),
        )
        if not (ctx.callers or ctx.callees or ctx.supertypes or ctx.subtypes or ctx.related):
            return None
        return ctx

    def extract_subgraph(
        self,
        project_id: str,
        *,
        partition_id: str | None = None,
        seed_declarations: Sequence[SelectedDeclaration] = (),
        query_seed_chunk_scores: Mapping[str, float] | None = None,
        query: str | None = None,
        max_nodes: int = 15,
        max_hops: int = 3,
        include_likely: bool = True,
        scope: Literal["code", "tests", "all"] = "code",
    ) -> SubgraphResponse:
        """Extract a connected, high-signal multi-hop subgraph and typed paths around seeds."""
        graph = self.get_partition_graph(project_id, partition_id=partition_id)
        if graph is None or not graph.nodes:
            return SubgraphResponse(
                project_id=project_id,
                query=query,
                completeness=CompletenessReport(
                    state="incomplete",
                    explanation="Reference graph is unavailable for this project.",
                ),
            )

        def scope_allows(node: GraphNode) -> bool:
            if node.is_barrel:
                return False
            if scope == "code" and node.is_test:
                return False
            return not (scope == "tests" and not node.is_test)

        explicit_seed_nids: list[int] = []
        seed_weights: dict[int, float] = {}
        for decl in seed_declarations:
            nid = graph.file_qual_to_node.get((decl.file_id, decl.qualified_symbol))
            if nid is None:
                nid = graph.path_qual_to_node.get((decl.path, decl.qualified_symbol))
            if nid is None and decl.chunk_id:
                nid = graph.any_chunk_to_node.get(decl.chunk_id)
            if nid is not None and not graph.nodes[nid].is_barrel:
                if nid not in explicit_seed_nids:
                    explicit_seed_nids.append(nid)
                seed_weights[nid] = seed_weights.get(nid, 0.0) + 1.0

        if query_seed_chunk_scores:
            for cid, cscore in query_seed_chunk_scores.items():
                nid = graph.any_chunk_to_node.get(cid)
                if nid is not None and scope_allows(graph.nodes[nid]):
                    seed_weights[nid] = seed_weights.get(nid, 0.0) + max(0.01, float(cscore))
                    if len(explicit_seed_nids) < 5 and nid not in explicit_seed_nids:
                        explicit_seed_nids.append(nid)

        if not seed_weights:
            return SubgraphResponse(
                project_id=project_id,
                query=query,
                snapshot_version=graph.references_version,
            )

        # Helper to iterate neighbors (undirected view over directed edges)
        def neighbors_of(u: int) -> Iterable[tuple[int, int, int, float, bool]]:
            # yields (v, kind_code, res_code, weight, is_forward_u_to_v)
            for idx in range(graph.out_indptr[u], graph.out_indptr[u + 1]):
                res = graph.out_resolutions[idx]
                if not include_likely and res != _RESOLUTION_EXACT:
                    continue
                v = graph.out_indices[idx]
                if scope_allows(graph.nodes[v]) or v in explicit_seed_nids:
                    yield v, graph.out_kinds[idx], res, float(graph.out_weights[idx]), True
            for idx in range(graph.in_indptr[u], graph.in_indptr[u + 1]):
                res = graph.in_resolutions[idx]
                if not include_likely and res != _RESOLUTION_EXACT:
                    continue
                v = graph.in_indices[idx]
                if scope_allows(graph.nodes[v]) or v in explicit_seed_nids:
                    yield v, graph.in_kinds[idx], res, float(graph.in_weights[idx]), False

        # Find shortest highest-weight structural paths between pairs of explicit seeds
        paths: list[SubgraphPath] = []
        path_nodes: set[int] = set()
        if len(explicit_seed_nids) >= 2:
            anchor_seeds = explicit_seed_nids[:6]
            for i, src_nid in enumerate(anchor_seeds):
                for dst_nid in anchor_seeds[i + 1 :]:
                    found = self._best_bounded_path(
                        graph, src_nid, dst_nid, neighbors_of, max_hops=max_hops
                    )
                    if found is not None:
                        p_nids, p_kinds, p_score = found
                        path_nodes.update(p_nids)
                        paths.append(
                            SubgraphPath(
                                chunk_ids=[graph.nodes[n].chunk_id for n in p_nids],
                                symbols=[graph.nodes[n].qualified_symbol for n in p_nids],
                                edge_kinds=p_kinds,
                                score=round(p_score, 4),
                            )
                        )
            paths.sort(key=lambda item: (-item.score, len(item.chunk_ids)))

        # Score neighborhood via PPR and grow a connected prize-collecting subgraph
        ppr_acts = self.seeded_ppr(graph, seed_weights)
        node_scores: dict[int, float] = {}
        max_seed = max(seed_weights.values(), default=1.0)
        max_act = max((a.score for a in ppr_acts.values()), default=0.0)
        for nid, sw in seed_weights.items():
            node_scores[nid] = 0.65 * (sw / max_seed)
        for nid, act in ppr_acts.items():
            if scope_allows(graph.nodes[nid]) or nid in explicit_seed_nids:
                add = 0.55 * (act.score / max_act) if max_act > 0 else 0.0
                node_scores[nid] = node_scores.get(nid, 0.0) + add

        selected_nids: list[int] = []
        selected_set: set[int] = set()

        for nid in explicit_seed_nids:
            if len(selected_nids) < max_nodes and nid not in selected_set:
                selected_nids.append(nid)
                selected_set.add(nid)
        for nid in sorted(path_nodes, key=lambda n: -node_scores.get(n, 0.0)):
            if len(selected_nids) < max_nodes and nid not in selected_set:
                selected_nids.append(nid)
                selected_set.add(nid)

        # Bounded BFS distance from seeds so subgraph stays within `max_hops`
        hop_dist: dict[int, int] = {nid: 0 for nid in selected_set}
        queue: deque[int] = deque(selected_nids)
        while queue:
            u = queue.popleft()
            d = hop_dist[u]
            if d >= max_hops:
                continue
            for v, _, _, _, _ in neighbors_of(u):
                if v not in hop_dist:
                    hop_dist[v] = d + 1
                    queue.append(v)

        # Greedily add highest-scoring reachable nodes adjacent to `selected_set`
        while len(selected_nids) < max_nodes:
            best_next: int | None = None
            best_val = -1.0
            for u in list(selected_nids):
                if hop_dist.get(u, 0) >= max_hops:
                    continue
                for v, _, _, ew, _ in neighbors_of(u):
                    if v in selected_set or hop_dist.get(v, max_hops + 1) > max_hops:
                        continue
                    val = node_scores.get(v, 0.0) + 0.15 * ew
                    if val > best_val:
                        best_val = val
                        best_next = v
            if best_next is None or best_val <= 0.0:
                break
            selected_nids.append(best_next)
            selected_set.add(best_next)

        # Hydrate snippets for selected nodes
        chunk_rows = self.store.fetch_search_rows_by_ids(
            project_id,
            [graph.nodes[n].chunk_id for n in selected_nids],
            partition_id=partition_id,
        )
        content_by_cid = {str(r["chunk_id"]): str(r.get("content") or "") for r in chunk_rows}
        explicit_seed_set = set(explicit_seed_nids)

        sub_nodes: list[SubgraphNode] = []
        for nid in selected_nids:
            gnode = graph.nodes[nid]
            raw_content = content_by_cid.get(gnode.chunk_id, "")
            truncated = len(raw_content) > 600
            snippet = raw_content[:600] if truncated else raw_content
            sub_nodes.append(
                SubgraphNode(
                    chunk_id=gnode.chunk_id,
                    project_id=project_id,
                    path=gnode.path,
                    language=gnode.language,
                    kind=gnode.kind,
                    symbol=gnode.symbol,
                    qualified_symbol=gnode.qualified_symbol,
                    start_line=gnode.start_line,
                    end_line=gnode.end_line,
                    score=round(node_scores.get(nid, 0.0), 4),
                    is_seed=nid in explicit_seed_set,
                    snippet=snippet,
                    truncated=truncated,
                )
            )
        sub_nodes.sort(key=lambda n: (not n.is_seed, -n.score, n.path, n.start_line))

        # Induced directed edges among `selected_set`
        edge_acc: dict[tuple[int, int], tuple[set[str], int, float]] = {}
        for u in selected_nids:
            for idx in range(graph.out_indptr[u], graph.out_indptr[u + 1]):
                v = graph.out_indices[idx]
                if v not in selected_set:
                    continue
                res = int(graph.out_resolutions[idx])
                if not include_likely and res != _RESOLUTION_EXACT:
                    continue
                kname = EDGE_KINDS[graph.out_kinds[idx]]
                w = float(graph.out_weights[idx])
                prev = edge_acc.get((u, v))
                if prev is None:
                    edge_acc[(u, v)] = ({kname}, res, w)
                else:
                    prev[0].add(kname)
                    edge_acc[(u, v)] = (prev[0], max(prev[1], res), max(prev[2], w))

        sub_edges = [
            SubgraphEdge(
                source_chunk_id=graph.nodes[u].chunk_id,
                target_chunk_id=graph.nodes[v].chunk_id,
                source_symbol=graph.nodes[u].qualified_symbol,
                target_symbol=graph.nodes[v].qualified_symbol,
                kinds=sorted(kinds),
                resolution="exact" if res == _RESOLUTION_EXACT else "likely",
                weight=round(w, 4),
            )
            for (u, v), (kinds, res, w) in sorted(
                edge_acc.items(),
                key=lambda item: (
                    -item[1][2],
                    graph.nodes[item[0][0]].path,
                    graph.nodes[item[0][1]].path,
                ),
            )
        ]

        # If single-seed / query mode had <2 explicit seeds, synthesize top ego paths from seed #1
        if not paths and explicit_seed_nids:
            root_nid = explicit_seed_nids[0]
            for other_nid in selected_nids[1:6]:
                found = self._best_bounded_path(
                    graph, root_nid, other_nid, neighbors_of, max_hops=max_hops
                )
                if found is not None and len(found[0]) >= 2:
                    p_nids, p_kinds, p_score = found
                    paths.append(
                        SubgraphPath(
                            chunk_ids=[graph.nodes[n].chunk_id for n in p_nids],
                            symbols=[graph.nodes[n].qualified_symbol for n in p_nids],
                            edge_kinds=p_kinds,
                            score=round(p_score, 4),
                        )
                    )
            paths.sort(key=lambda item: (-item.score, len(item.chunk_ids)))

        has_likely = any(e.resolution == "likely" for e in sub_edges)
        completeness = CompletenessReport(
            state="complete_with_dynamic_limitations" if has_likely else "complete",
            explanation=(
                "Subgraph includes heuristic likely edges alongside exact bindings."
                if has_likely
                else "All subgraph edges resolved from exact syntax bindings."
            ),
        )
        return SubgraphResponse(
            project_id=project_id,
            query=query,
            nodes=sub_nodes,
            edges=sub_edges,
            paths=paths[:8],
            snapshot_version=graph.references_version,
            completeness=completeness,
        )

    @staticmethod
    def _best_bounded_path(
        graph: PartitionGraph,
        src_nid: int,
        dst_nid: int,
        neighbors_of: Any,
        *,
        max_hops: int,
    ) -> tuple[list[int], list[str], float] | None:
        if src_nid == dst_nid:
            return [src_nid], [], 1.0
        # BFS for shortest hop distance, breaking ties by highest cumulative edge weight
        best_dist: dict[int, int] = {src_nid: 0}
        best_score: dict[int, float] = {src_nid: 1.0}
        parent: dict[int, tuple[int, str]] = {}
        queue: deque[int] = deque([src_nid])

        while queue:
            u = queue.popleft()
            d = best_dist[u]
            if d >= max_hops:
                continue
            if dst_nid in best_dist and d >= best_dist[dst_nid]:
                continue
            for v, kcode, _, ew, is_fwd in neighbors_of(u):
                if kcode == _KIND_TO_CODE["import"]:
                    continue
                nd = d + 1
                cand_score = best_score[u] * ew * 0.85
                prev_d = best_dist.get(v)
                if prev_d is None or nd < prev_d or (nd == prev_d and cand_score > best_score[v]):
                    best_dist[v] = nd
                    best_score[v] = cand_score
                    arrow = EDGE_KINDS[kcode] if is_fwd else f"<-{EDGE_KINDS[kcode]}"
                    parent[v] = (u, arrow)
                    if prev_d is None:
                        queue.append(v)

        if dst_nid not in parent:
            return None
        cur = dst_nid
        rev_nodes = [cur]
        rev_kinds: list[str] = []
        while cur != src_nid:
            p_nid, arrow = parent[cur]
            rev_nodes.append(p_nid)
            rev_kinds.append(arrow)
            cur = p_nid
        rev_nodes.reverse()
        rev_kinds.reverse()
        return rev_nodes, rev_kinds, best_score[dst_nid]
