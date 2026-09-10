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
            str(project_root / "test_fastmcp_server.py"),
        ],
        cwd=str(project_root),
        env=os.environ.copy(),
    )

    client = Client(transport)

    print("CONNECTING")

    try:
        async with client:
            print("CONNECTED")

            tools = await asyncio.wait_for(
                client.list_tools(),
                timeout=15,
            )

            print("TOOLS:", [tool.name for tool in tools])

            print("CALLING_PING")

            result = await asyncio.wait_for(
                client.call_tool(
                    "ping",
                    {},
                ),
                timeout=15,
            )

            print("PING_SUCCESS")
            print("RESULT:", result)

    except asyncio.TimeoutError:
        print("PING_TIMEOUT")

    except Exception as exc:
        print("PING_ERROR")
        print(type(exc).__name__)
        print(str(exc))


if __name__ == "__main__":
    asyncio.run(main())