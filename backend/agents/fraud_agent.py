from langgraph.graph import END, START, StateGraph

from .graph_state import AgentState
from ..mcp.fraud_mcp_client import FraudMCPClient


async def fraud_scope_node(state: AgentState):
    """Validate that this request is allowed into the Fraud Agent."""

    if state.get("supervisor_domain") != "FRAUD_ANALYTICS":
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Fraud Agent requires FRAUD_ANALYTICS domain."
            ),
        }

    if not state.get("supervisor_tools"):
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "No approved fraud capability selected."
            ),
        }

    return {
        "agent_name": "fraud_agent",
        "fraud_execution_allowed": True,
    }


def fraud_tool_selection_node(state: AgentState):
    selected_tools = {
        str(tool).lower()
        for tool in state.get("supervisor_tools", [])
    }

    query = state.get("query", "").lower()

    # --------------------------------------------------------
    # Fraud metrics / velocity
    # --------------------------------------------------------
    if (
        "velocity" in query
        or "transaction velocity" in query
        or "fraud metrics" in query
        or "get_fraud_metrics" in selected_tools
    ):
        return {
            "fraud_tool_name": "get_transaction_velocity",
            "fraud_tool_arguments": {
                "limit": 50,
            },
        }

    # --------------------------------------------------------
    # Fraud transaction data
    # --------------------------------------------------------
    if "query_fraud_data" in selected_tools:
        if "high value" in query or "high-value" in query:
            return {
                "fraud_tool_name": "get_high_value_transactions",
                "fraud_tool_arguments": {
                    "minimum_amount": 100000,
                    "limit": 50,
                },
            }

        return {
            "fraud_tool_name": "get_fraud_card_alerts",
            "fraud_tool_arguments": {
                "limit": 50,
            },
        }

    return {
        "fraud_execution_allowed": False,
        "fraud_execution_reason": (
            "No supported fraud MCP capability was selected."
        ),
    }


async def fraud_tool_execution_node(state: AgentState):
    """
    Execute fraud analytics exclusively through MCP.

    Fraud Agent
        ↓
    MCP Client
        ↓
    Fraud MCP Server
        ↓
    Fraud Tools
        ↓
    FraudDataAccess
        ↓
    Gold Tables
    """

    if not state.get("fraud_execution_allowed"):
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Fraud execution was not authorized."
            ),
        }

    tool_name = state.get("fraud_tool_name")
    arguments = state.get("fraud_tool_arguments", {})

    if not tool_name:
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "No fraud MCP tool selected."
            ),
        }

    client = FraudMCPClient()

    try:
        result = await client.call_tool(
            tool_name,
            arguments,
        )

        return {
            "tool_results": [result],
            "evidence": [
                {
                    "source": "fraud_gold",
                    "tool": tool_name,
                    "data": result,
                }
            ],
            "fraud_execution_allowed": True,
            "fraud_execution_reason": (
                "Approved fraud tool executed through MCP."
            ),
        }

    except Exception as exc:
        return {
            "tool_results": [],
            "evidence": [],
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                f"Fraud MCP execution failed: {exc}"
            ),
        }


def build_fraud_agent():
    """Build the LangGraph Fraud Agent."""

    graph = StateGraph(AgentState)

    graph.add_node(
        "fraud_scope",
        fraud_scope_node,
    )

    graph.add_node(
        "fraud_tool_selection",
        fraud_tool_selection_node,
    )

    graph.add_node(
        "fraud_tool_execution",
        fraud_tool_execution_node,
    )

    graph.add_edge(
        START,
        "fraud_scope",
    )

    graph.add_edge(
        "fraud_scope",
        "fraud_tool_selection",
    )

    graph.add_edge(
        "fraud_tool_selection",
        "fraud_tool_execution",
    )

    graph.add_edge(
        "fraud_tool_execution",
        END,
    )

    return graph.compile()