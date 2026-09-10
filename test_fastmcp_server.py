from fastmcp import FastMCP


mcp = FastMCP(
    name="FastMCP Test Server",
)


@mcp.tool()
def ping() -> str:
    """Simple MCP connectivity test."""
    return "pong"


if __name__ == "__main__":
    mcp.run()