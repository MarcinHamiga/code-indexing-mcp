"""``python -m incode_mcp.installer`` — the bootstrap's delegation target."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
