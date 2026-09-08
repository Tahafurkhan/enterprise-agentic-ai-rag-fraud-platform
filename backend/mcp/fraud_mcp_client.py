
"""
MCP client for governed fraud analytics execution.

The Fraud Agent communicates with the Fraud MCP Server through
the MCP protocol. The client does not access Databricks directly.
"""

import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


class FraudMCPClient:
    """
    Client used by the Fraud Agent to call the governed Fraud MCP Server.
    """

    def __init__(self) -> None:
        # Project root:
        # enterprise-agentic-ai-rag-fraud-platform/
        project_root = Path(__file__).resolve().parents[2]

        # Forward the current backend environment to the MCP subprocess.
        environment = os.environ.copy()

        transport = StdioTransport(
            command=sys.executable,
            args=[
                "-m",
                "backend.mcp.fraud_mcp_server",
            ],
            cwd=str(project_root),
            env=environment,
        )

        self.client = Client(transport)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """
        Call an approved tool exposed by the Fraud MCP Server.
        """

        arguments = arguments or {}

        async with self.client:
            result = await self.client.call_tool(
                tool_name,
                arguments,
            )

        # FastMCP structured response
        if hasattr(result, "structured_content"):
            structured = result.structured_content

            if structured is not None:
                # FastMCP may return:
                # {"result": [...]}
                if (
                    isinstance(structured, dict)
                    and set(structured.keys()) == {"result"}
                ):
                    return structured["result"]

                return structured

        # Fallback for content-based responses
        if hasattr(result, "content"):
            return result.content

        # Defensive normalization if the client returns a plain dict
        if isinstance(result, dict):
            if set(result.keys()) == {"result"}:
                return result["result"]

        return result

