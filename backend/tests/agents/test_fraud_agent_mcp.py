
import pytest

from backend.agents.fraud_agent import build_fraud_agent


@pytest.mark.asyncio
async def test_fraud_agent_executes_through_mcp():
    """
    Verify the LangGraph Fraud Agent executes an approved
    fraud capability through the MCP client/server boundary.
    """

    graph = build_fraud_agent()

    state = {
        "query": "Show me high value fraud transactions",
        "supervisor_domain": "FRAUD_ANALYTICS",
        "supervisor_tools": ["QUERY_FRAUD_DATA"],
    }

    result = await graph.ainvoke(state)

    assert result["agent_name"] == "fraud_agent"
    assert result["fraud_execution_allowed"] is True
    assert result["fraud_tool_name"] == "get_high_value_transactions"
    assert result["tool_results"]
    assert result["evidence"]
    assert result["evidence"][0]["source"] == "fraud_gold"
    assert (
        result["evidence"][0]["tool"]
        == "get_high_value_transactions"
    )


@pytest.mark.asyncio
async def test_fraud_agent_executes_metrics_through_mcp():
    """
    Verify the Fraud Agent can execute an approved fraud
    metrics capability through MCP.
    """

    graph = build_fraud_agent()

    state = {
        "query": "Show fraud transaction velocity",
        "supervisor_domain": "FRAUD_ANALYTICS",
        "supervisor_tools": ["GET_FRAUD_METRICS"],
    }

    result = await graph.ainvoke(state)

    assert result["agent_name"] == "fraud_agent"
    assert result["fraud_execution_allowed"] is True
    assert result["fraud_tool_name"] == "get_transaction_velocity"
    assert result["tool_results"]
    assert result["evidence"]


@pytest.mark.asyncio
async def test_fraud_agent_rejects_wrong_domain():
    """
    Verify the Fraud Agent does not execute when the supervisor
    routes the request outside the fraud analytics domain.
    """

    graph = build_fraud_agent()

    state = {
        "query": "Search company policy",
        "supervisor_domain": "ENTERPRISE_KNOWLEDGE",
        "supervisor_tools": ["SEARCH_DOCUMENTS"],
    }

    result = await graph.ainvoke(state)

    assert result["fraud_execution_allowed"] is False
    assert result["fraud_execution_reason"]



