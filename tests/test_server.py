from pathlib import Path

import pytest

from incode_mcp.application import Application, RuntimePaths
from incode_mcp.server import create_server


class TinyEmbedder:
    model_id = "test/tiny"
    dimension = 4

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, float(len(text))]


@pytest.mark.asyncio
async def test_server_registers_the_focused_tool_suite(tmp_path: Path) -> None:
    app = Application(
        RuntimePaths(data=tmp_path / "data", cache=tmp_path / "cache"),
        embedder=TinyEmbedder(),
        cwd=tmp_path,
    )
    server = create_server(app)

    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {
        "init_project",
        "index_project",
        "project_status",
        "list_projects",
        "remove_project",
        "search_code",
        "find_symbol",
        "file_outline",
        "get_chunk",
    }
    assert all("ctx" not in tool.inputSchema.get("properties", {}) for tool in tools)
