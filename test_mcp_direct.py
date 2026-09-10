import asyncio

from backend.mcp.fraud_mcp_client import FraudMCPClient


async def main():
    client = FraudMCPClient()

    print("CALLING_MCP")

    try:
        result = await asyncio.wait_for(
            client.call_tool(
                "get_high_value_transactions",
                {
                    "minimum_amount": 100000,
                    "limit": 5,
                },
            ),
            timeout=30,
        )

        print("MCP_SUCCESS")
        print(result)

    except asyncio.TimeoutError:
        print("MCP_TIMEOUT")

    except Exception as exc:
        print("MCP_ERROR")
        print(type(exc).__name__)
        print(str(exc))


if __name__ == "__main__":
    asyncio.run(main())