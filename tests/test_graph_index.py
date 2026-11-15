from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.graph_index import (
    _GRAPH_CACHE_MAGIC,
    GraphActivation,
    GraphIndexService,
    _deserialize_graph,
    _serialize_graph,
)
from code_indexing_mcp.indexing import Indexer
from code_indexing_mcp.models import DeclarationSelector
from code_indexing_mcp.projects import initialize_project
from code_indexing_mcp.reference_service import ReferenceService
from code_indexing_mcp.scanner import SourceScanner
from code_indexing_mcp.storage import LanceStore


class _TinyEmbedder:
    model_id = "test/graph-embedder"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[float((len(t) % 5) + 1), 0.5, 0.25, 0.1] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float((len(text) % 5) + 1), 0.5, 0.25, 0.1]


def _index_repo(
    tmp_path: Path, files: dict[str, str]
) -> tuple[LanceStore, ReferenceService, str, Indexer]:
    embedder = _TinyEmbedder()
    store = LanceStore(tmp_path / "data", vector_dimension=embedder.dimension)
    indexer = Indexer(
        store=store,
        scanner=SourceScanner(),
        extractor=TreeSitterExtractor(),
        embedder=embedder,
        lock_directory=tmp_path / "locks",
    )
    root = tmp_path / "repo"
    root.mkdir()
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    project = initialize_project(root)
    indexer.index(project)
    return store, ReferenceService(store), project.id, indexer


def test_compile_partition_graph_matches_find_references_across_languages(tmp_path: Path) -> None:
    _store, ref_service, project_id, _ = _index_repo(
        tmp_path,
        {
            "pkg/__init__.py": "from .core import Engine\n",
            "pkg/core.py": (
                "class BaseEngine:\n"
                "    def validate(self, payload: str) -> bool:\n"
                "        return bool(payload)\n\n"
                "class Engine(BaseEngine):\n"
                "    def run(self, item: str) -> bool:\n"
                "        return self.validate(item)\n"
            ),
            "pkg/runner.py": (
                "from pkg import Engine\n\n"
                "def execute_job(task: str) -> bool:\n"
                "    eng = Engine()\n"
                "    return eng.run(task)\n"
            ),
            "web/impl.ts": (
                "export interface Renderer {\n"
                "  render(view: string): string;\n"
                "}\n\n"
                "export class HtmlRenderer implements Renderer {\n"
                "  render(view: string): string {\n"
                "    return formatHtml(view);\n"
                "  }\n"
                "}\n\n"
                "export function formatHtml(raw: string): string {\n"
                "  return `<p>${raw}</p>`;\n"
                "}\n"
            ),
            "web/index.ts": "export { HtmlRenderer, formatHtml } from './impl';\n",
            "web/app.ts": (
                "import { HtmlRenderer, formatHtml } from './index';\n\n"
                "export function mountApp(name: string): string {\n"
                "  const r = new HtmlRenderer();\n"
                "  return r.render(formatHtml(name));\n"
                "}\n"
            ),
        },
    )

    compiled = ref_service.compile_partition_graph(project_id)
    assert compiled is not None
    assert "pkg/__init__.py" in compiled.barrel_files
    assert "web/index.ts" in compiled.barrel_files

    barrel_targets = {
        (b.barrel_path, b.exported_name, b.target_path, b.target_qualified_symbol)
        for b in compiled.barrel_reexports
    }
    assert ("pkg/__init__.py", "Engine", "pkg/core.py", "Engine") in barrel_targets
    assert ("web/index.ts", "HtmlRenderer", "web/impl.ts", "HtmlRenderer") in barrel_targets
    assert ("web/index.ts", "formatHtml", "web/impl.ts", "formatHtml") in barrel_targets

    # Verify exact edges match find_references for key declarations
    for path, qual in [
        ("pkg/core.py", "BaseEngine"),
        ("pkg/core.py", "Engine"),
        ("web/impl.ts", "Renderer"),
        ("web/impl.ts", "formatHtml"),
    ]:
        refs = ref_service.find_references(
            DeclarationSelector(project=project_id, path=path, qualified_symbol=qual),
            root=tmp_path / "repo",
        )
        exact_hit_paths = {h.path for h in refs.hits if h.resolution == "exact"}
        compiled_inbound_paths = {
            e.source_path
            for e in compiled.edges
            if e.target_path == path
            and e.target_qualified_symbol == qual
            and e.resolution == "exact"
        }
        assert exact_hit_paths == compiled_inbound_paths


def test_graph_sidecar_cache_invalidation_and_maintenance(tmp_path: Path) -> None:
    store, ref_service, project_id, indexer = _index_repo(
        tmp_path,
        {
            "a.py": "def helper() -> int:\n    return 1\n",
            "b.py": "from a import helper\n\ndef caller() -> int:\n    return helper()\n",
        },
    )
    graph_service = GraphIndexService(store, ref_service)
    g1 = graph_service.get_partition_graph(project_id)
    assert g1 is not None

    # Cold service instance loads directly from the persisted sidecar cache
    cold_service = GraphIndexService(store, ref_service)
    g1_from_disk = cold_service.get_partition_graph(project_id)
    assert g1_from_disk is not None
    assert g1_from_disk.chunks_version == g1.chunks_version
    assert g1_from_disk.references_version == g1.references_version
    assert len(g1_from_disk.nodes) == len(g1.nodes)

    # Incremental edit adds a new symbol & call edge -> invalidates previous snapshot automatically
    (tmp_path / "repo" / "a.py").write_text(
        "def helper() -> int:\n    return 1\n\ndef extra() -> int:\n    return 2\n",
        encoding="utf-8",
    )
    (tmp_path / "repo" / "b.py").write_text(
        "from a import extra\n\ndef caller() -> int:\n    return extra()\n",
        encoding="utf-8",
    )
    project = next(p for p in store.list_projects() if p.id == project_id)
    indexer.index(project)

    g2 = graph_service.get_partition_graph(project_id)
    assert g2 is not None
    assert (g2.chunks_version, g2.references_version) != (
        g1.chunks_version,
        g1.references_version,
    )
    assert ("a.py", "extra") in g2.path_qual_to_node

    # Writing g2 supersedes g1's sidecar; only the active snapshot's sidecar remains
    cache_dir = store.directory / "graph-cache" / g2.partition_id
    bins = list(cache_dir.glob("*.bin"))
    assert len(bins) == 1
    assert bins[0].name == f"{g2.nonce}-{g2.chunks_version}-{g2.references_version}.bin"

    # Maintenance prunes stale sidecars after fragment compaction, and remove_project cleans the dir
    stale_bin = cache_dir / "deadbeef-0-0.bin"
    stale_bin.write_bytes(b"stale")
    store.maintain_project(project_id, cleanup_older_than=timedelta(hours=1))
    assert not stale_bin.exists()
    g3 = graph_service.get_partition_graph(project_id)
    assert g3 is not None
    store.remove_project(project_id)
    assert not cache_dir.exists()


def test_specificity_ppr_damps_ubiquitous_hub_and_promotes_specific_callee(tmp_path: Path) -> None:
    files: dict[str, str] = {
        "common.py": "def log_event(msg: str) -> None:\n    pass\n",
        "secret_impl.py": "def compute_merkle_digest(blob: bytes) -> str:\n    return blob.hex()\n",
        "entry.py": (
            "from common import log_event\n"
            "from secret_impl import compute_merkle_digest\n\n"
            "def verify_archive_integrity(data: bytes) -> str:\n"
            "    log_event('start')\n"
            "    return compute_merkle_digest(data)\n"
        ),
    }
    for i in range(12):
        files[f"mod_{i}.py"] = (
            f"from common import log_event\n\ndef handler_{i}() -> None:\n    log_event('h{i}')\n"
        )

    store, ref_service, project_id, _ = _index_repo(tmp_path, files)
    graph_service = GraphIndexService(store, ref_service)
    graph = graph_service.get_partition_graph(project_id)
    assert graph is not None

    entry_nid = graph.path_qual_to_node[("entry.py", "verify_archive_integrity")]
    specific_nid = graph.path_qual_to_node[("secret_impl.py", "compute_merkle_digest")]
    hub_nid = graph.path_qual_to_node[("common.py", "log_event")]

    activations = graph_service.seeded_ppr(graph, {entry_nid: 1.0})
    assert specific_nid in activations
    assert hub_nid in activations
    # Specific 1-caller callee must outrank the 13-caller ubiquitous logger hub!
    assert activations[specific_nid].score > activations[hub_nid].score
    assert activations[specific_nid].rank < activations[hub_nid].rank
    assert activations[specific_nid].via == "called_by:verify_archive_integrity"


def test_ego_context_and_subgraph_path_extraction(tmp_path: Path) -> None:
    store, ref_service, project_id, _ = _index_repo(
        tmp_path,
        {
            "api.py": (
                "from service import process_order\n\n"
                "def post_order_endpoint(payload: dict) -> str:\n"
                "    return process_order(payload)\n"
            ),
            "service.py": (
                "from repo import persist_order\n\n"
                "def process_order(payload: dict) -> str:\n"
                "    return persist_order(payload)\n"
            ),
            "repo.py": ("def persist_order(payload: dict) -> str:\n    return 'ok'\n"),
        },
    )
    graph_service = GraphIndexService(store, ref_service)
    graph = graph_service.get_partition_graph(project_id)
    assert graph is not None

    svc_node = graph.nodes[graph.path_qual_to_node[("service.py", "process_order")]]
    ego = graph_service.ego_context(project_id, svc_node.chunk_id)
    assert ego is not None
    assert [c.qualified_symbol for c in ego.callers] == ["post_order_endpoint"]
    assert [c.qualified_symbol for c in ego.callees] == ["persist_order"]

    decls = ref_service.compile_partition_graph(project_id)
    assert decls is not None
    by_qual = {d.qualified_symbol: d for d in decls.declarations}
    sub = graph_service.extract_subgraph(
        project_id,
        seed_declarations=[by_qual["post_order_endpoint"], by_qual["persist_order"]],
        max_nodes=10,
        max_hops=3,
    )
    assert len(sub.paths) >= 1
    assert sub.paths[0].symbols == ["post_order_endpoint", "process_order", "persist_order"]
    assert {n.qualified_symbol for n in sub.nodes} >= {
        "post_order_endpoint",
        "process_order",
        "persist_order",
    }


def test_entry_scripts_are_not_barrels(tmp_path: Path) -> None:
    store, ref_service, project_id, _ = _index_repo(
        tmp_path,
        {
            "app/__init__.py": "",
            "app/service.py": "def process_order(oid: str) -> str:\n    return oid\n",
            "main.py": "from app.service import process_order\n\nprocess_order('a1')\n",
            "web/util.ts": "export function fmt(x: string): string { return x; }\n",
            "web/run.ts": "import { fmt } from './util';\nconsole.log(fmt('x'));\n",
        },
    )
    compiled = ref_service.compile_partition_graph(project_id)
    assert compiled is not None
    assert "main.py" not in compiled.barrel_files
    assert "web/run.ts" not in compiled.barrel_files
    assert not {e.barrel_path for e in compiled.barrel_reexports} & {"main.py", "web/run.ts"}

    graph_service = GraphIndexService(store, ref_service)
    graph = graph_service.get_partition_graph(project_id)
    assert graph is not None
    assert not [n for n in graph.nodes if n.is_barrel and n.path in {"main.py", "web/run.ts"}]

    query = "process_order a1"
    rows = store._hybrid_search_rows(project_id, query, [1.0, 0.5, 0.25, 0.1], None, 20)
    assert rows
    expanded = graph_service.expand_hybrid_rows(
        project_id, None, query, rows, None, 20, query_vector=[1.0, 0.5, 0.25, 0.1]
    )
    assert expanded
    assert not [r for r in expanded if r.get("_barrel_damped")]


def test_csharp_using_with_class_is_not_a_barrel(tmp_path: Path) -> None:
    store, ref_service, project_id, _ = _index_repo(
        tmp_path,
        {
            "Demo/Shop/Order.cs": (
                "using Demo.Stock;\n\nnamespace Demo.Shop;\n\n"
                "public class Order\n{\n    public int Id;\n}\n"
            ),
            "Demo/Stock/Item.cs": "namespace Demo.Stock;\n\npublic class Item\n{\n}\n",
        },
    )
    compiled = ref_service.compile_partition_graph(project_id)
    assert compiled is not None
    assert "Demo/Shop/Order.cs" not in compiled.barrel_files
    assert compiled.barrel_reexports == []
    graph = GraphIndexService(store, ref_service).get_partition_graph(project_id)
    assert graph is not None
    assert not [n for n in graph.nodes if n.is_barrel]


def test_genuine_barrels_still_detected(tmp_path: Path) -> None:
    _store, ref_service, project_id, _ = _index_repo(
        tmp_path,
        {
            "pkg/__init__.py": "from .core import Engine\n",
            "pkg/core.py": "class Engine:\n    pass\n",
            "web/impl.ts": "export function f(): number { return 1; }\n",
            "web/index.ts": "export { f } from './impl';\n",
        },
    )
    compiled = ref_service.compile_partition_graph(project_id)
    assert compiled is not None
    assert {"pkg/__init__.py", "web/index.ts"} <= compiled.barrel_files
    edges = {(e.barrel_path, e.exported_name, e.target_path) for e in compiled.barrel_reexports}
    assert ("pkg/__init__.py", "Engine", "pkg/core.py") in edges
    assert ("web/index.ts", "f", "web/impl.ts") in edges


_CHAIN = {
    "chain.py": (
        "def a() -> int:\n    return b()\n\n"
        "def b() -> int:\n    return c()\n\n"
        "def c() -> int:\n    return d()\n\n"
        "def d() -> int:\n    return e()\n\n"
        "def e() -> int:\n    return 1\n"
    )
}


def _chain_subgraph(tmp_path: Path, files: dict[str, str], **kwargs: int):  # type: ignore[no-untyped-def]
    store, ref_service, project_id, _ = _index_repo(tmp_path, files)
    graph_service = GraphIndexService(store, ref_service)
    compiled = ref_service.compile_partition_graph(project_id)
    assert compiled is not None
    by_qual = {d.qualified_symbol: d for d in compiled.declarations}
    sub = graph_service.extract_subgraph(project_id, seed_declarations=[by_qual["a"]], **kwargs)
    return sub, graph_service, project_id


def test_subgraph_complete_only_when_untruncated(tmp_path: Path) -> None:
    sub, _, _ = _chain_subgraph(tmp_path, _CHAIN, max_nodes=15, max_hops=10)
    assert {n.qualified_symbol for n in sub.nodes} == {"a", "b", "c", "d", "e"}
    assert sub.completeness.state == "complete"


def test_subgraph_truncated_by_max_nodes_is_not_complete(tmp_path: Path) -> None:
    sub, _, _ = _chain_subgraph(tmp_path, _CHAIN, max_nodes=2, max_hops=10)
    assert len(sub.nodes) == 2
    assert sub.completeness.state == "incomplete"
    assert "max_nodes=2" in sub.completeness.explanation


def test_subgraph_truncated_by_max_hops_is_not_complete(tmp_path: Path) -> None:
    sub, _, _ = _chain_subgraph(tmp_path, _CHAIN, max_nodes=15, max_hops=1)
    assert len(sub.nodes) < 5
    assert sub.completeness.state == "incomplete"
    assert "max_hops=1" in sub.completeness.explanation


def test_subgraph_reports_coverage_limitation_as_incomplete(tmp_path: Path) -> None:
    # Real indexing: a YAML file is indexed for search but has no structural
    # reference extraction, which compile_partition_graph reports as a limitation.
    files = {**_CHAIN, "config.yaml": "service:\n  port: 8080\n"}
    sub, graph_service, project_id = _chain_subgraph(tmp_path, files, max_nodes=15, max_hops=10)
    graph = graph_service.get_partition_graph(project_id)
    assert graph is not None
    assert graph.coverage_limitation_count >= 1
    assert graph.coverage_limitations[0][0] == "unsupported_language"
    assert sub.completeness.state == "incomplete"
    assert "coverage" in sub.completeness.explanation.lower()
    assert "config.yaml" in sub.completeness.explanation

    # Sidecar round trip preserves the limitation info.
    restored = _deserialize_graph(_serialize_graph(graph))
    assert restored is not None
    assert restored.coverage_limitation_count == graph.coverage_limitation_count
    assert restored.coverage_limitations == graph.coverage_limitations

    # Cold service loads the persisted V2 sidecar with the limitation intact.
    cold = GraphIndexService(graph_service.store, graph_service.reference_service)
    loaded = cold.get_partition_graph(project_id)
    assert loaded is not None
    assert loaded.coverage_limitations == graph.coverage_limitations


def test_v1_sidecar_blob_is_ignored(tmp_path: Path) -> None:
    _, graph_service, project_id = _chain_subgraph(tmp_path, _CHAIN)
    graph = graph_service.get_partition_graph(project_id)
    assert graph is not None
    blob = _serialize_graph(graph)
    assert _GRAPH_CACHE_MAGIC == b"SYNDEX_GRAPH_V2\n"
    v1 = b"SYNDEX_GRAPH_V1\n" + blob[len(_GRAPH_CACHE_MAGIC) :]
    assert _deserialize_graph(v1) is None
    assert _deserialize_graph(blob) is not None


def test_synthetic_barrel_activation_never_raises_floor_or_outranks_genuine(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    store, ref_service, project_id, _ = _index_repo(
        tmp_path,
        {
            "pkg/__init__.py": "from .core import Engine\n",
            "pkg/core.py": "class Engine:\n    pass\n",
            "other.py": "def unrelated() -> int:\n    return 1\n",
        },
    )
    graph_service = GraphIndexService(store, ref_service)
    graph = graph_service.get_partition_graph(project_id)
    assert graph is not None
    barrel_nid = next(
        n.node_id for n in graph.nodes if n.is_barrel and n.path.endswith("__init__.py")
    )
    engine_nid = graph.path_qual_to_node[("pkg/core.py", "Engine")]
    other_nid = graph.path_qual_to_node[("other.py", "unrelated")]
    barrel_rows = store.fetch_search_rows_by_ids(project_id, [graph.nodes[barrel_nid].chunk_id])
    assert barrel_rows
    row = {**barrel_rows[0], "_distance": 0.1, "_fts_local_rank": 1}

    # Genuine activation is far below the seed scale (~1+) that the synthetic
    # barrel target would previously have used as the peak.
    genuine = GraphActivation(node_id=other_nid, score=0.01, rank=1, via="called_by:x")
    monkeypatch.setattr(graph_service, "seeded_ppr", lambda *_a, **_k: {other_nid: genuine})

    out = graph_service.expand_hybrid_rows(
        project_id, None, "engine", [row], None, 8, query_vector=[1.0, 0.5, 0.25, 0.1]
    )
    by_nid = {
        graph.any_chunk_to_node[str(r["chunk_id"])]: r
        for r in out
        if "_graph_local_rank" in r and str(r["chunk_id"]) in graph.any_chunk_to_node
    }
    assert other_nid in by_nid, "genuine activation must survive the floor"
    assert engine_nid in by_nid, "synthetic barrel target is still hydrated"
    assert by_nid[other_nid]["_graph_score"] == 0.01
    assert by_nid[engine_nid]["_graph_score"] <= by_nid[other_nid]["_graph_score"]
    assert by_nid[engine_nid]["_graph_local_rank"] >= by_nid[other_nid]["_graph_local_rank"]
