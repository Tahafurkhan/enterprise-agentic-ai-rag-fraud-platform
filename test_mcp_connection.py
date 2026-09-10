import asyncio
import os
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def main():
    project_root = Path(__file__).resolve().parent

    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "backend.mcp.fraud_mcp_server",
        ],
        cwd=str(project_root),
        env=os.environ.copy(),
    )

    client = Client(transport)

    print("CONNECTING_TO_MCP")

    try:
        async with client:

            print("CONNECTED")

            tools = await asyncio.wait_for(
                client.list_tools(),
                timeout=15,
            )

            print("LIST_TOOLS_SUCCESS")

            for tool in tools:
                print(
                    f"- {tool.name}"
                )

    except asyncio.TimeoutError:
        print("MCP_CONNECTION_TIMEOUT")

    except Exception as exc:
        print("MCP_CONNECTION_ERROR")
        print(type(exc).__name__)
        print(str(exc))


if __name__ == "__main__":
    asyncio.run(main())