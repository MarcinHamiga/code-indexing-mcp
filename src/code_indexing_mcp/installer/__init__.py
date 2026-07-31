"""Install-time logic for Code Indexing MCP: configuration, accelerators, harnesses.

Everything in this package runs inside the synced installation environment.
The curl-pipeable bootstrap at the repository root (`install.py`) stays
stdlib-only and delegates here after `uv sync`.
"""
