
"""
Integration tests for:

Security Graph
    ↓
Supervisor
    ↓
Fraud Agent
    ↓
Fraud MCP Client
    ↓
Fraud MCP Server
    ↓
Fraud Tools
    ↓
Databricks Gold
"""

import pytest

from backend.agents.security_graph import (
    build_security_graph,
)


@pytest.mark.asyncio
async def test_security_graph_routes_fraud_request_to_fraud_agent():
    """
    Verify an authenticated and authorized fraud request passes
    through the complete Security Graph and reaches the Fraud Agent.
    """

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "Show me high value fraud transactions",
    }

    result = await graph.ainvoke(state)

    # --------------------------------------------------------
    # Security boundaries
    # --------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # --------------------------------------------------------
    # Supervisor
    # --------------------------------------------------------

    assert result["supervisor_domain"] == "FRAUD_ANALYTICS"

    assert "query_fraud_data" in result["supervisor_tools"]

    # --------------------------------------------------------
    # Fraud Agent
    # --------------------------------------------------------

    assert result["agent_name"] == "fraud_agent"

    assert result["fraud_execution_allowed"] is True

    assert result["fraud_tool_name"] == (
        "get_high_value_transactions"
    )

    # --------------------------------------------------------
    # MCP → Gold evidence
    # --------------------------------------------------------

    assert result["tool_results"]
    assert result["evidence"]

    assert result["evidence"][0]["source"] == "fraud_gold"

    assert (
        result["evidence"][0]["tool"]
        == "get_high_value_transactions"
    )


@pytest.mark.asyncio
async def test_security_graph_routes_fraud_metrics_to_fraud_agent():
    """
    Verify fraud metrics requests reach the Fraud Agent and
    execute the approved transaction velocity MCP capability.
    """

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "Show me fraud transaction velocity",
    }

    result = await graph.ainvoke(state)

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    assert result["supervisor_domain"] == "FRAUD_ANALYTICS"

    assert result["agent_name"] == "fraud_agent"

    assert result["fraud_execution_allowed"] is True

    assert result["fraud_tool_name"] == (
        "get_transaction_velocity"
    )

    assert result["tool_results"]
    assert result["evidence"]


@pytest.mark.asyncio
async def test_security_graph_blocks_unknown_user_before_fraud_agent():
    """
    Verify authentication stops an unknown user before Fraud Agent
    execution.
    """

    graph = build_security_graph()

    state = {
        "user_id": "unknown-user",
        "query": "Show me high value fraud transactions",
    }

    result = await graph.ainvoke(state)

    assert result["authenticated"] is False

    assert result["current_stage"] == "authentication"

    assert result.get("fraud_execution_allowed") is not True

    assert "agent_name" not in result


@pytest.mark.asyncio
async def test_security_graph_preserves_non_fraud_routing():
    """
    Verify enterprise knowledge requests are still routed to END
    because the Knowledge/RAG agent has not yet been connected.

    This confirms the Fraud Agent integration does not change the
    existing non-fraud routing behavior.
    """

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "What is the policy threshold for a high-value transaction?",
    }

    result = await graph.ainvoke(state)

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    assert (
        result["supervisor_domain"]
        == "ENTERPRISE_KNOWLEDGE"
    )

    assert "agent_name" not in result

    assert result.get("fraud_execution_allowed") is not True

