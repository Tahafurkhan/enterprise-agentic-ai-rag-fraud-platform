import json
from unittest.mock import Mock

import pytest

from backend.agents.llm_supervisor import LLMSupervisor
from backend.agents.supervisor import SupervisorDomain, SupervisorTool


def build_supervisor_with_response(payload):
    """
    Create an LLMSupervisor with a mocked Groq response.
    No real API request is made.
    """
    supervisor = LLMSupervisor(api_key="test-api-key")

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = json.dumps(payload)

    supervisor.client = Mock()
    supervisor.client.chat.completions.create.return_value = response

    return supervisor


def test_llm_routes_fraud_analytics():
    supervisor = build_supervisor_with_response(
        {
            "domain": "FRAUD_ANALYTICS",
            "tools": [
                "query_fraud_data",
                "get_fraud_metrics",
            ],
            "reason": "The request requires structured fraud data.",
            "confidence": 0.95,
            "requires_policy": False,
            "requires_structured_data": True,
            "requires_external_research": False,
        }
    )

    decision = supervisor.route(
        "Show me the number of fraud alerts by merchant."
    )

    assert decision.domain == SupervisorDomain.FRAUD_ANALYTICS
    assert SupervisorTool.QUERY_FRAUD_DATA in decision.tools
    assert SupervisorTool.GET_FRAUD_METRICS in decision.tools
    assert decision.requires_structured_data is True
    assert decision.requires_policy is False
    assert decision.requires_external_research is False


def test_llm_routes_enterprise_knowledge():
    supervisor = build_supervisor_with_response(
        {
            "domain": "ENTERPRISE_KNOWLEDGE",
            "tools": [
                "search_policies",
                "search_documents",
            ],
            "reason": "The request requires internal policy knowledge.",
            "confidence": 0.94,
            "requires_policy": True,
            "requires_structured_data": False,
            "requires_external_research": False,
        }
    )

    decision = supervisor.route(
        "What is the policy for high-value transactions?"
    )

    assert decision.domain == SupervisorDomain.ENTERPRISE_KNOWLEDGE
    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert SupervisorTool.SEARCH_DOCUMENTS in decision.tools
    assert decision.requires_policy is True
    assert decision.requires_structured_data is False


def test_llm_routes_external_research():
    supervisor = build_supervisor_with_response(
        {
            "domain": "EXTERNAL_RESEARCH",
            "tools": ["search_web"],
            "reason": "The request requires current external information.",
            "confidence": 0.91,
            "requires_policy": False,
            "requires_structured_data": False,
            "requires_external_research": True,
        }
    )

    decision = supervisor.route(
        "Search the web for the latest fraud regulations."
    )

    assert decision.domain == SupervisorDomain.EXTERNAL_RESEARCH
    assert decision.tools == frozenset({SupervisorTool.SEARCH_WEB})
    assert decision.requires_external_research is True


def test_llm_routes_multi_domain():
    supervisor = build_supervisor_with_response(
        {
            "domain": "MULTI_DOMAIN",
            "tools": [
                "search_policies",
                "query_fraud_data",
            ],
            "reason": "The request requires both policy and structured fraud data.",
            "confidence": 0.93,
            "requires_policy": True,
            "requires_structured_data": True,
            "requires_external_research": False,
        }
    )

    decision = supervisor.route(
        "Which transactions violated our fraud monitoring policy?"
    )

    assert decision.domain == SupervisorDomain.MULTI_DOMAIN
    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert SupervisorTool.QUERY_FRAUD_DATA in decision.tools
    assert decision.requires_policy is True
    assert decision.requires_structured_data is True


def test_llm_routes_direct_request():
    supervisor = build_supervisor_with_response(
        {
            "domain": "DIRECT",
            "tools": [],
            "reason": "The request is a simple conversational greeting.",
            "confidence": 0.99,
            "requires_policy": False,
            "requires_structured_data": False,
            "requires_external_research": False,
        }
    )

    decision = supervisor.route("Hello")

    assert decision.domain == SupervisorDomain.DIRECT
    assert decision.tools == frozenset()
    assert decision.requires_policy is False
    assert decision.requires_structured_data is False
    assert decision.requires_external_research is False


def test_empty_query_does_not_call_llm():
    supervisor = LLMSupervisor(api_key="test-api-key")
    supervisor.client = Mock()

    decision = supervisor.route("")

    assert decision.domain == SupervisorDomain.UNKNOWN
    assert decision.confidence == 1.0
    assert decision.reason == "The request is empty."

    supervisor.client.chat.completions.create.assert_not_called()


def test_non_string_query_raises_type_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    with pytest.raises(TypeError, match="query must be a string"):
        supervisor.route(None)


def test_markdown_fenced_json_is_parsed():
    supervisor = LLMSupervisor(api_key="test-api-key")

    content = """```json
{
    "domain": "DIRECT",
    "tools": [],
    "reason": "The request is conversational.",
    "confidence": 0.99,
    "requires_policy": false,
    "requires_structured_data": false,
    "requires_external_research": false
}
```"""

    parsed = supervisor._parse_json(content)

    assert parsed["domain"] == "DIRECT"
    assert parsed["tools"] == []


def test_invalid_json_raises_value_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    with pytest.raises(
        ValueError,
        match="Supervisor model returned invalid JSON",
    ):
        supervisor._parse_json("this is not valid json")


def test_missing_required_field_raises_value_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "DIRECT",
        "tools": [],
        "reason": "Conversational request.",
        "confidence": 0.9,
        "requires_policy": False,
        "requires_structured_data": False,
    }

    with pytest.raises(
        ValueError,
        match="Supervisor response is missing fields",
    ):
        supervisor._parse_route_decision(payload)


def test_invalid_domain_raises_value_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "NOT_A_REAL_DOMAIN",
        "tools": [],
        "reason": "Invalid domain test.",
        "confidence": 0.9,
        "requires_policy": False,
        "requires_structured_data": False,
        "requires_external_research": False,
    }

    with pytest.raises(
        ValueError,
        match="Invalid Supervisor domain",
    ):
        supervisor._parse_route_decision(payload)


def test_invalid_tool_raises_value_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "FRAUD_ANALYTICS",
        "tools": ["not_a_real_tool"],
        "reason": "Invalid tool test.",
        "confidence": 0.9,
        "requires_policy": False,
        "requires_structured_data": True,
        "requires_external_research": False,
    }

    with pytest.raises(
        ValueError,
        match="Invalid Supervisor tool",
    ):
        supervisor._parse_route_decision(payload)


def test_confidence_out_of_range_raises_value_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "DIRECT",
        "tools": [],
        "reason": "Invalid confidence test.",
        "confidence": 1.5,
        "requires_policy": False,
        "requires_structured_data": False,
        "requires_external_research": False,
    }

    decision = supervisor._parse_route_decision(payload)

    with pytest.raises(
        ValueError,
        match="confidence must be between 0.0 and 1.0",
    ):
        supervisor._validate_and_convert(decision)


def test_invalid_tool_for_domain_raises_value_error():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "FRAUD_ANALYTICS",
        "tools": ["search_web"],
        "reason": "Invalid tool-domain combination.",
        "confidence": 0.9,
        "requires_policy": False,
        "requires_structured_data": True,
        "requires_external_research": False,
    }

    decision = supervisor._parse_route_decision(payload)

    with pytest.raises(
        ValueError,
        match="tools that are not allowed",
    ):
        supervisor._validate_and_convert(decision)


def test_fraud_domain_without_structured_data_is_rejected():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "FRAUD_ANALYTICS",
        "tools": ["query_fraud_data"],
        "reason": "Fraud analysis request.",
        "confidence": 0.9,
        "requires_policy": False,
        "requires_structured_data": False,
        "requires_external_research": False,
    }

    decision = supervisor._parse_route_decision(payload)

    with pytest.raises(
        ValueError,
        match="FRAUD_ANALYTICS requires structured data",
    ):
        supervisor._validate_and_convert(decision)


def test_external_domain_without_external_research_is_rejected():
    supervisor = LLMSupervisor(api_key="test-api-key")

    payload = {
        "domain": "EXTERNAL_RESEARCH",
        "tools": ["search_web"],
        "reason": "External research request.",
        "confidence": 0.9,
        "requires_policy": False,
        "requires_structured_data": False,
        "requires_external_research": False,
    }

    decision = supervisor._parse_route_decision(payload)

    with pytest.raises(
        ValueError,
        match="EXTERNAL_RESEARCH requires external research",
    ):
        supervisor._validate_and_convert(decision)


def test_model_is_called_with_expected_configuration():
    supervisor = build_supervisor_with_response(
        {
            "domain": "DIRECT",
            "tools": [],
            "reason": "Simple conversational request.",
            "confidence": 0.99,
            "requires_policy": False,
            "requires_structured_data": False,
            "requires_external_research": False,
        }
    )

    supervisor.route("Hello")

    supervisor.client.chat.completions.create.assert_called_once()

    call_kwargs = supervisor.client.chat.completions.create.call_args.kwargs

    assert call_kwargs["temperature"] == 0
    assert call_kwargs["max_tokens"] == 500
    assert call_kwargs["model"] == supervisor.model
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"



def test_llm_routes_company_knowledge_without_policy_requirement():
    supervisor = build_supervisor_with_response(
        {
            "domain": "ENTERPRISE_KNOWLEDGE",
            "tools": [
                "search_documents",
            ],
            "reason": (
                "The request requires internal company knowledge."
            ),
            "confidence": 0.94,
            "requires_policy": False,
            "requires_structured_data": False,
            "requires_external_research": False,
        }
    )

    decision = supervisor.route(
        "What does our internal onboarding documentation say?"
    )

    assert (
        decision.domain
        == SupervisorDomain.ENTERPRISE_KNOWLEDGE
    )
    assert (
        SupervisorTool.SEARCH_DOCUMENTS
        in decision.tools
    )
    assert decision.requires_policy is False
    assert decision.requires_structured_data is False
    assert decision.requires_external_research is False
    assert decision.is_valid()

