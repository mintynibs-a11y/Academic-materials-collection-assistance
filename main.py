"""
Entry point for the Academic Assistant.

Run the CLI assistant:
    python main.py "your research query"

Run the MCP server:
    python main.py --mcp-server
"""

from __future__ import annotations

import sys


def main() -> None:
    if "--mcp-server" in sys.argv:
        sys.argv.remove("--mcp-server")
        import asyncio
        from academic_assistant.mcp_server.server import main as server_main
        asyncio.run(server_main())
    else:
        from academic_assistant.assistant import main as assistant_main
        assistant_main()


if __name__ == "__main__":
    main()
