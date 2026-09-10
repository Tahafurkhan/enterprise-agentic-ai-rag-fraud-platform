"""
MCP client for Tavily external web research.

The External Research Agent communicates with the official
Tavily MCP Server through the MCP protocol.

This client does not call the Tavily API directly.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


class TavilyMCPClient:
    """
    Client used by the External Research Agent to call
    the official Tavily MCP Server.
    """

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]

        tavily_api_key = os.getenv("TAVILY_API_KEY")

        if not tavily_api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable is required."
            )

        environment = os.environ.copy()
        environment["TAVILY_API_KEY"] = tavily_api_key

        transport = StdioTransport(
            command="npx",
            args=[
                "-y",
                "tavily-mcp@latest",
            ],
            cwd=str(project_root),
            env=environment,
        )

        self.client = Client(transport)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Execute Tavily search through MCP.
        """

        query = query.strip()

        if not query:
            raise ValueError(
                "Tavily search query cannot be empty."
            )

        if max_results < 1:
            raise ValueError(
                "max_results must be at least 1."
            )

        arguments = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }

        async with self.client:
            result = await self.client.call_tool(
                "tavily_search",
                arguments,
            )

        normalized = self._normalize_result(result)

        for item in normalized:
            item.setdefault("provider", "tavily")
            item.setdefault("query", query)

        return normalized

    @classmethod
    def _normalize_result(
        cls,
        result: Any,
    ) -> list[dict[str, Any]]:
        """
        Normalize FastMCP/Tavily output.
        """

        if hasattr(result, "structured_content"):
            structured = result.structured_content

            if structured is not None:
                normalized = cls._extract_structured(
                    structured
                )

                if normalized:
                    return normalized

        if hasattr(result, "content"):
            for item in result.content:
                text = getattr(item, "text", None)

                if not text:
                    continue

                normalized = cls._parse_text_results(text)

                if normalized:
                    return normalized

        if isinstance(result, list):
            return [
                item
                for item in result
                if isinstance(item, dict)
            ]

        if isinstance(result, dict):
            return cls._extract_structured(result)

        return []

    @classmethod
    def _extract_structured(
        cls,
        data: Any,
    ) -> list[dict[str, Any]]:
        if isinstance(data, dict):

            if "result" in data:
                return cls._extract_structured(
                    data["result"]
                )

            if "results" in data:
                return [
                    item
                    for item in data["results"]
                    if isinstance(item, dict)
                ]

            if any(
                key in data
                for key in (
                    "title",
                    "url",
                    "content",
                )
            ):
                return [data]

        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
            ]

        return []

    @staticmethod
    def _parse_text_results(
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Parse the text format returned by Tavily MCP.
        """

        title_matches = list(
            re.finditer(
                r"(?:^|\n)Title:\s*(.+?)(?=\nID:)",
                text,
            )
        )

        results: list[dict[str, Any]] = []

        for index, match in enumerate(title_matches):

            start = match.start()

            end = (
                title_matches[index + 1].start()
                if index + 1 < len(title_matches)
                else len(text)
            )

            block = text[start:end].strip()

            title_match = re.search(
                r"Title:\s*(.+?)(?=\nID:)",
                block,
                flags=re.DOTALL,
            )

            id_match = re.search(
                r"ID:\s*(.+?)(?=\nURL:)",
                block,
                flags=re.DOTALL,
            )

            url_match = re.search(
                r"URL:\s*(.+?)(?=\nContent:)",
                block,
                flags=re.DOTALL,
            )

            content_match = re.search(
                r"Content:\s*(.*)",
                block,
                flags=re.DOTALL,
            )

            results.append(
                {
                    "title": (
                        title_match.group(1).strip()
                        if title_match
                        else ""
                    ),
                    "id": (
                        id_match.group(1).strip()
                        if id_match
                        else ""
                    ),
                    "url": (
                        url_match.group(1).strip()
                        if url_match
                        else ""
                    ),
                    "content": (
                        content_match.group(1).strip()
                        if content_match
                        else ""
                    ),
                    "provider": "tavily",
                }
            )

        return results