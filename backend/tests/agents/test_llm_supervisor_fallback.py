import json
from unittest.mock import Mock
from backend.agents.llm_supervisor import LLMSupervisor
from backend.agents.supervisor import SupervisorDomain, SupervisorTool


def test_fallback_is_used_when_llm_returns_invalid_json():
    supervisor = LLMSupervisor(api_key="test-api-key")

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "THIS IS NOT JSON"

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.return_value = response

    decision = supervisor.route(
        "Show me the fraud count by day."
    )

    assert decision.domain == SupervisorDomain.FRAUD_ANALYTICS
    assert decision.requires_structured_data is True
    assert SupervisorTool.QUERY_FRAUD_DATA in decision.tools
    assert "deterministic fallback used" in decision.reason
    assert decision.confidence <= 0.80


def test_fallback_is_used_when_llm_returns_invalid_tool():
    supervisor = LLMSupervisor(api_key="test-api-key")

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = json.dumps(
        {
            "domain": "FRAUD_ANALYTICS",
            "tools": ["search_web"],
            "reason": "Invalid tool selection.",
            "confidence": 0.95,
            "requires_policy": False,
            "requires_structured_data": True,
            "requires_external_research": False,
        }
    )

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.return_value = response

    decision = supervisor.route(
        "Show me the fraud count by day."
    )

    assert decision.domain == SupervisorDomain.FRAUD_ANALYTICS
    assert decision.requires_structured_data is True
    assert "deterministic fallback used" in decision.reason


def test_fallback_is_used_when_llm_call_fails():
    supervisor = LLMSupervisor(api_key="test-api-key")

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.side_effect = RuntimeError(
        "Groq service unavailable"
    )

    decision = supervisor.route(
        "What is the policy for high-value transactions?"
    )

    assert decision.domain == SupervisorDomain.ENTERPRISE_KNOWLEDGE
    assert decision.requires_policy is True
    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert "deterministic fallback used" in decision.reason
    assert decision.confidence <= 0.80


def test_fallback_handles_external_research():
    supervisor = LLMSupervisor(api_key="test-api-key")

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.side_effect = RuntimeError(
        "Network failure"
    )

    decision = supervisor.route(
        "Search the web for the latest fraud regulations."
    )

    assert decision.domain == SupervisorDomain.EXTERNAL_RESEARCH
    assert decision.requires_external_research is True
    assert SupervisorTool.SEARCH_WEB in decision.tools
    assert "deterministic fallback used" in decision.reason


def test_fallback_handles_multi_domain_request():
    supervisor = LLMSupervisor(api_key="test-api-key")

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.side_effect = RuntimeError(
        "Temporary LLM failure"
    )

    decision = supervisor.route(
        "Which transactions violated our fraud monitoring policy?"
    )

    assert decision.domain == SupervisorDomain.MULTI_DOMAIN
    assert decision.requires_policy is True
    assert decision.requires_structured_data is True
    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert SupervisorTool.QUERY_FRAUD_DATA in decision.tools


def test_successful_llm_response_does_not_use_fallback():
    supervisor = LLMSupervisor(api_key="test-api-key")

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = json.dumps(
        {
            "domain": "FRAUD_ANALYTICS",
            "tools": ["query_fraud_data"],
            "reason": "The request requires structured fraud data.",
            "confidence": 0.95,
            "requires_policy": False,
            "requires_structured_data": True,
            "requires_external_research": False,
        }
    )

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.return_value = response

    supervisor.fallback_supervisor = Mock()

    decision = supervisor.route(
        "Show me fraud alerts."
    )

    assert decision.domain == SupervisorDomain.FRAUD_ANALYTICS
    assert decision.confidence == 0.95

    supervisor.fallback_supervisor.route.assert_not_called()


def test_empty_query_does_not_use_fallback():
    supervisor = LLMSupervisor(api_key="test-api-key")

    supervisor.client = Mock()
    supervisor.fallback_supervisor = Mock()

    decision = supervisor.route("")

    assert decision.domain == SupervisorDomain.UNKNOWN
    assert decision.confidence == 1.0

    supervisor.client.chat.completions.create.assert_not_called()
    supervisor.fallback_supervisor.route.assert_not_called()