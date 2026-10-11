"""MCP server exposing the same tools to any MCP client (Claude Desktop, an IDE, another agent).

    python -m cadence.mcp_server        # stdio transport

The server keeps its state in memory for the life of the process.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .domain import State
from .tools import Toolbox


def build_server(state: State | None = None) -> FastMCP:
    toolbox = Toolbox(state or State())
    server = FastMCP("cadence")
    for name in toolbox.tool_names():
        fn = getattr(toolbox, name)
        server.add_tool(fn, name=name, description=fn.__doc__)
    return server


if __name__ == "__main__":
    build_server().run()
