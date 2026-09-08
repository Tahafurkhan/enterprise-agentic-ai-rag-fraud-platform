import pytest

from backend.agents.security_graph import build_security_graph


@pytest.mark.asyncio
async def test_fraud_response_passes_output_guardrails():
    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "Show me high value fraud transactions",
    }

    result = await graph.ainvoke(state)

    # Security
    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # Supervisor
    assert result["supervisor_domain"] == "FRAUD_ANALYTICS"

    # Fraud Agent
    assert result["agent_name"] == "fraud_agent"
    assert result["fraud_execution_allowed"] is True
    assert result["fraud_tool_name"] == (
        "get_high_value_transactions"
    )

    # Evidence
    assert result["evidence"]

    # Response Generator
    assert result["generated_response"]
    assert result["generated_response"]["answer"]
    assert result["generated_response"]["claims"]

    # Output Guardrails
    assert result["output_guardrails_allowed"] is True
    assert result["safe_response"]
    assert (
        result["output_guardrails_reason"]
        == "Output passed all guardrail checks."
    )


@pytest.mark.asyncio
async def test_fraud_metrics_response_passes_output_guardrails():
    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "Show me fraud transaction velocity",
    }

    result = await graph.ainvoke(state)

    assert result["supervisor_domain"] == "FRAUD_ANALYTICS"

    assert result["agent_name"] == "fraud_agent"

    assert result["fraud_execution_allowed"] is True

    assert result["fraud_tool_name"] == (
        "get_transaction_velocity"
    )

    assert result["evidence"]

    assert result["generated_response"]

    assert result["output_guardrails_allowed"] is True

    assert result["safe_response"]


@pytest.mark.asyncio
async def test_unknown_user_never_reaches_response_generator():
    graph = build_security_graph()

    state = {
        "user_id": "unknown-user",
        "query": "Show me high value fraud transactions",
    }

    result = await graph.ainvoke(state)

    assert result["authenticated"] is False

    assert "generated_response" not in result

    assert "safe_response" not in result