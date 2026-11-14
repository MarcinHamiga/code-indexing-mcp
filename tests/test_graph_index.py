from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from code_indexing_mcp.extractor import TreeSitterExtractor
from code_indexing_mcp.graph_index import GraphIndexService
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
