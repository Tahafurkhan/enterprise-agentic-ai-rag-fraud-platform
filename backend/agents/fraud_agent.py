
"""
LangGraph Fraud Agent subgraph.

The Fraud Agent is responsible for executing approved fraud-analysis
capabilities after the Supervisor has routed the request to the
FRAUD_ANALYTICS domain.
"""

from typing import Dict

from langgraph.graph import END, START, StateGraph

from .graph_state import AgentState
from .supervisor import SupervisorTool
from .tools.fraud_tools import build_fraud_tools
from ..data_access.fraud_data import FraudDataAccess


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def build_fraud_tool_registry():
    """
    Build the bounded Fraud Agent tool registry.

    The registry maps Supervisor-approved tool names to LangChain tools.
    """

    data_access = FraudDataAccess()

    tools = build_fraud_tools(
        data_access=data_access,
    )

    return {
        tool.name: tool
        for tool in tools
    }


# ---------------------------------------------------------------------------
# Fraud scope validation
# ---------------------------------------------------------------------------

def fraud_scope_node(state: AgentState) -> AgentState:
    """
    Verify that the Supervisor actually routed this request to Fraud
    Analytics and selected an approved fraud tool.
    """

    domain = state.get(
        "supervisor_domain",
        "",
    )

    tools = state.get(
        "supervisor_tools",
        [],
    )

    if domain != "FRAUD_ANALYTICS":
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Fraud Agent received a request outside "
                "the FRAUD_ANALYTICS domain."
            ),
            "current_stage": "fraud_agent",
            "allowed": False,
        }

    if not tools:
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Supervisor did not select a fraud tool."
            ),
            "current_stage": "fraud_agent",
            "allowed": False,
        }

    approved_tools = {
        SupervisorTool.QUERY_FRAUD_DATA.value,
        SupervisorTool.GET_FRAUD_METRICS.value,
        SupervisorTool.GENERATE_VISUALIZATION.value,
    }

    if not any(
        tool in approved_tools
        for tool in tools
    ):
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Supervisor selected no approved fraud capability."
            ),
            "current_stage": "fraud_agent",
            "allowed": False,
        }

    return {
        "fraud_execution_allowed": True,
        "fraud_execution_reason": (
            "Fraud Agent execution scope validated."
        ),
        "agent_name": "fraud_agent",
        "current_stage": "fraud_agent",
    }


# ---------------------------------------------------------------------------
# Tool selection
# ---------------------------------------------------------------------------

def fraud_tool_selection_node(
    state: AgentState,
) -> AgentState:
    """
    Map the Supervisor's abstract tool decision to a concrete governed
    LangChain tool.
    """

    supervisor_tools = state.get(
        "supervisor_tools",
        [],
    )

    query = state.get(
        "query",
        "",
    ).lower()

    selected_tool = None

    if (
        SupervisorTool.QUERY_FRAUD_DATA.value
        in supervisor_tools
    ):
        selected_tool = "get_fraud_card_alerts"

    if (
        SupervisorTool.GET_FRAUD_METRICS.value
        in supervisor_tools
    ):
        selected_tool = "get_transaction_velocity"

    if (
        "high value" in query
        or "high-value" in query
    ):
        selected_tool = "get_high_value_transactions"

    if selected_tool is None:
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "No concrete governed fraud tool could be selected."
            ),
            "current_stage": "fraud_agent",
            "allowed": False,
        }

    return {
        "fraud_tool_name": selected_tool,
        "fraud_tool_arguments": {},
        "current_stage": "fraud_agent",
    }


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def fraud_tool_execution_node(
    state: AgentState,
) -> AgentState:
    """
    Execute exactly one governed Fraud Agent tool.
    """

    tool_name = state.get(
        "fraud_tool_name"
    )

    if not tool_name:
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "No fraud tool was selected."
            ),
            "allowed": False,
            "current_stage": "fraud_agent",
        }

    registry = build_fraud_tool_registry()

    tool = registry.get(tool_name)

    if tool is None:
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Selected fraud tool is not registered."
            ),
            "allowed": False,
            "current_stage": "fraud_agent",
        }

    arguments: Dict = state.get(
        "fraud_tool_arguments",
        {},
    )

    try:
        result = tool.invoke(arguments)

    except Exception as error:
        return {
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Fraud tool execution failed."
            ),
            "error": str(error),
            "allowed": False,
            "current_stage": "fraud_agent",
        }

    return {
        "tool_results": [
            {
                "tool": tool_name,
                "result": result,
            }
        ],
        "evidence": [
            {
                "source": "fraud_gold",
                "tool": tool_name,
                "records": result,
            }
        ],
        "fraud_execution_allowed": True,
        "fraud_execution_reason": (
            "Governed fraud tool executed successfully."
        ),
        "current_stage": "fraud_agent",
        "allowed": True,
    }


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------

def fraud_scope_route(
    state: AgentState,
) -> str:
    if state.get(
        "fraud_execution_allowed",
        False,
    ):
        return "allowed"

    return "blocked"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_fraud_agent_graph():
    """
    Compile the Fraud Agent LangGraph subgraph.
    """

    builder = StateGraph(AgentState)

    builder.add_node(
        "fraud_scope",
        fraud_scope_node,
    )

    builder.add_node(
        "fraud_tool_selection",
        fraud_tool_selection_node,
    )

    builder.add_node(
        "fraud_tool_execution",
        fraud_tool_execution_node,
    )

    builder.add_edge(
        START,
        "fraud_scope",
    )

    builder.add_conditional_edges(
        "fraud_scope",
        fraud_scope_route,
        {
            "allowed": "fraud_tool_selection",
            "blocked": END,
        },
    )

    builder.add_edge(
        "fraud_tool_selection",
        "fraud_tool_execution",
    )

    builder.add_edge(
        "fraud_tool_execution",
        END,
    )

    return builder.compile()

