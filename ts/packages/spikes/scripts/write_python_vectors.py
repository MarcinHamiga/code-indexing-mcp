"""Produce the reference vectors S3 scores the TypeScript embedder against.

Runs under the project's own environment, through the same code path the index
is built with, so the vectors carry the model revision, tokenizer, pooling, and
normalization that the stored index actually depends on.

Usage:
    .venv/bin/python write_python_vectors.py <output.json> [--offline]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The probe corpus: short code passages plus natural-language queries over
# them. Small enough to embed in seconds, varied enough that a pooling or
# normalization mistake moves the ranking rather than hiding in the average.
#
# Written as block strings so the passages stay readable and stay inside the
# line limit; the embedder sees exactly these bytes, so editing one changes the
# vectors and any reference file produced from an older revision is stale.
DOCUMENTS = [
    """def load_tokenizer(directory):
    return Tokenizer.from_file(directory / 'tokenizer.json')
""",
    """def mean_pool(output, mask):
    return (output * mask[..., None]).sum(1) / mask.sum(1)[:, None]
""",
    """class LanceStore:
    def __init__(self, directory):
        self._db = lancedb.connect(directory)
""",
    """func openPartition(root string) (*Table, error) {
\treturn connect(filepath.Join(root, "projects"))
}
""",
    """export function pathCondition(patterns: string[]): string | null {
  return patterns.map(globToRegex).join(' OR ')
}
""",
    """SELECT chunk_id, path FROM chunks WHERE language = 'python' ORDER BY start_line;
""",
    """resource "aws_s3_bucket" "index" {
  bucket = var.bucket_name
  acl    = "private"
}
""",
    """impl Parser {
    pub fn parse(&mut self, source: &str) -> Tree {
        self.inner.parse(source)
    }
}
""",
    """async def index_project(root: Path) -> IndexResult:
    files = await scan(root)
    return await embed_all(files)
""",
    """public class Catalog {
    private final Map<String, Entry> entries = new HashMap<>();
}
""",
]

QUERIES = [
    "how do I load a tokenizer from disk",
    "attention mask mean pooling over model output",
    "open a lancedb table for a project partition",
    "translate a glob pattern into a SQL predicate",
    "recursively scan a directory and embed the files",
]


def main(destination: Path, *, offline: bool) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
    from code_indexing_mcp.application import RuntimePaths
    from code_indexing_mcp.embedding import DEFAULT_DIMENSION, DEFAULT_MODEL, FastEmbedder

    # The same cache the serving environment uses, so this reuses an already
    # downloaded model rather than pulling a second ~640 MB copy.
    cache_directory = RuntimePaths.from_environment().cache / "models"
    embedder = FastEmbedder(cache_directory, offline=offline)
    documents = embedder.embed_passages(list(DOCUMENTS))
    queries = embedder.embed_passages(list(QUERIES))

    payload = {
        "modelId": DEFAULT_MODEL,
        "dimension": DEFAULT_DIMENSION,
        "documents": DOCUMENTS,
        "queries": QUERIES,
        "documentVectors": documents,
        "queryVectors": queries,
    }
    destination.write_text(json.dumps(payload), encoding="utf8")
    print(f"wrote {len(documents)} document and {len(queries)} query vectors to {destination}")


if __name__ == "__main__":
    arguments = [argument for argument in sys.argv[1:] if argument != "--offline"]
    if len(arguments) != 1:
        raise SystemExit("usage: write_python_vectors.py <output.json> [--offline]")
    main(Path(arguments[0]).resolve(), offline="--offline" in sys.argv[1:])
