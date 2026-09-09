"""
FastMCP server for governed fraud analytics tools.

The MCP layer exposes only approved fraud capabilities.
Business logic and data governance remain in fraud_tools.py
and FraudDataAccess.
"""

from typing import Any, Dict

from fastmcp import FastMCP

from ..data_access.fraud_data import FraudDataAccess
from ..agents.tools.fraud_tools import build_fraud_tools


mcp = FastMCP(
    name="Fraud Analytics MCP Server",
    instructions=(
        "Provides governed, read-only fraud analytics capabilities. "
        "Only approved fraud tools are exposed. Arbitrary SQL, arbitrary "
        "table access, and sensitive-field retrieval are not supported."
    ),
)


_data_access = FraudDataAccess()
_tools = build_fraud_tools(data_access=_data_access)
_tool_registry = {tool.name: tool for tool in _tools}




@mcp.tool()
def get_high_value_transactions(
    minimum_amount: float = 100000,
    limit: int = 50,
) -> list[Dict[str, Any]]:
    """Get approved high-value fraud transactions."""
    tool = _tool_registry["get_high_value_transactions"]

    return tool.invoke(
        {
            "minimum_amount": minimum_amount,
            "limit": limit,
        }
    )


@mcp.tool()
def get_fraud_card_alerts(
    limit: int = 50,
) -> list[Dict[str, Any]]:
    """Get approved fraud card alerts."""
    tool = _tool_registry["get_fraud_card_alerts"]

    return tool.invoke(
        {
            "limit": limit,
        }
    )


@mcp.tool()
def get_transaction_velocity(
    limit: int = 50,
) -> list[Dict[str, Any]]:
    """Get approved transaction velocity analytics."""
    tool = _tool_registry["get_transaction_velocity"]

    return tool.invoke(
        {
            "limit": limit,
        }
    )


@mcp.tool()
def list_approved_datasets() -> list[Dict[str, str]]:
    """List datasets approved for governed fraud analytics."""
    tool = _tool_registry["list_approved_datasets"]

    return tool.invoke({})


@mcp.tool()
def get_safe_dataset_schema(
    dataset_name: str,
) -> list[Dict[str, Any]]:
    """Get the non-sensitive schema for an approved dataset."""
    tool = _tool_registry["get_safe_dataset_schema"]

    return tool.invoke(
        {
            "dataset_name": dataset_name,
        }
    )


@mcp.tool()
def get_dataset_rows(
    dataset_name: str,
    limit: int = 50,
) -> list[Dict[str, Any]]:
    """Read non-sensitive rows from an approved dataset."""
    tool = _tool_registry["get_dataset_rows"]

    return tool.invoke(
        {
            "dataset_name": dataset_name,
            "limit": limit,
        }
    )


if __name__ == "__main__":
    mcp.run()