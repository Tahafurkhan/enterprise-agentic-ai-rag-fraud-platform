import asyncio

from dotenv import load_dotenv

from backend.mcp.tavily_mcp_client import TavilyMCPClient


load_dotenv()


async def main() -> None:
    client = TavilyMCPClient()

    results = await client.search(
        "latest fraud detection trends",
        max_results=5,
    )

    print("\nTAVILY RESULTS")
    print("=" * 80)

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("-" * 80)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())