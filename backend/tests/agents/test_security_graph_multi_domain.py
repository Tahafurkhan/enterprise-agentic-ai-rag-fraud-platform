"""
Tests for the Enterprise Security Graph.

Coverage:
    - Input guardrails
    - Authentication
    - Authorization
    - Supervisor routing
    - Fraud Analytics execution
    - Policy RAG execution
    - Company Knowledge execution
    - External Research execution
    - Multi-domain orchestration
    - Shared Evidence Layer
    - Output Guardrails
    - Unknown-user blocking
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import backend.agents.security_graph as security_graph_module
from backend.agents.security_graph import build_security_graph
from backend.agents.supervisor import SupervisorDomain, SupervisorTool


# =====================================================================
# Deterministic supervisor mock
# =====================================================================


def _set_deterministic_supervisor(
    monkeypatch,
    *,
    domain: SupervisorDomain = SupervisorDomain.MULTI_DOMAIN,
    tools: list[SupervisorTool] | None = None,
    requires_policy: bool = False,
    requires_structured_data: bool = False,
    requires_external_research: bool = False,
) -> None:
    """
    Replace the LLM supervisor with a deterministic decision.

    This keeps orchestration tests independent of the LLM.
    """

    tools = tools or []

    decision = SimpleNamespace(
        domain=domain,
        tools=tools,
        requires_policy=requires_policy,
        requires_structured_data=requires_structured_data,
        requires_external_research=requires_external_research,
        reasoning="Deterministic test supervisor decision.",
    )

    async def mock_supervisor_node(state):
        return {
            **state,
            "supervisor_domain": decision.domain.value,
            "supervisor_tools": [
                tool.value
                for tool in decision.tools
            ],
            "requires_policy": decision.requires_policy,
            "requires_structured_data": (
                decision.requires_structured_data
            ),
            "requires_external_research": (
                decision.requires_external_research
            ),
            "supervisor_reasoning": decision.reasoning,
            "current_stage": "supervisor",
        }

    monkeypatch.setattr(
        security_graph_module,
        "supervisor_node",
        mock_supervisor_node,
    )


# =====================================================================
# Deterministic domain-agent mocks
# =====================================================================


async def _mock_fraud_agent_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic Fraud Analytics result.

    Used only by orchestration tests.
    """

    return {
        **state,
        "fraud_results": [
            {
                "transaction_id": "txn-001",
                "risk_score": 0.97,
                "amount": 125000,
                "status": "high_risk",
            }
        ],
        "evidence": [
            {
                "evidence_id": "fraud-evidence-1",
                "source": "fraud_analytics",
                "tool": "get_fraud_metrics",
                "retrieval_type": "structured_query",
                "data": {
                    "transaction_id": "txn-001",
                    "risk_score": 0.97,
                    "amount": 125000,
                    "status": "high_risk",
                },
                "metadata": {
                    "provider": "fraud_mcp",
                },
            }
        ],
        "fraud_execution_allowed": True,
        "fraud_execution_reason": (
            "Fraud Analytics execution completed successfully."
        ),
        "fraud_execution_error": "",
        "allowed": True,
        "execution_allowed": True,
        "execution_reason": (
            "Fraud Analytics execution completed successfully."
        ),
        "current_stage": "fraud_analytics",
    }


async def _mock_policy_rag_harness_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic Policy RAG result.
    """

    return {
        **state,
        "policy_results": [
            {
                "document_id": "policy-001",
                "title": "Fraud Monitoring Policy",
                "content": (
                    "High-value transactions require enhanced review."
                ),
            }
        ],
        "evidence": [
            {
                "evidence_id": "policy-evidence-1",
                "source": "policy_rag",
                "tool": "search_policies",
                "retrieval_type": "vector_search",
                "data": {
                    "document_id": "policy-001",
                    "title": "Fraud Monitoring Policy",
                    "content": (
                        "High-value transactions require enhanced review."
                    ),
                },
                "metadata": {
                    "provider": "databricks_vector_search",
                },
            }
        ],
        "policy_execution_allowed": True,
        "policy_execution_reason": (
            "Policy RAG execution completed successfully."
        ),
        "policy_execution_error": "",
        "allowed": True,
        "execution_allowed": True,
        "execution_reason": (
            "Policy RAG execution completed successfully."
        ),
        "current_stage": "policy_rag",
    }


async def _mock_company_knowledge_harness_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic Company Knowledge result.
    """

    return {
        **state,
        "company_results": [
            {
                "document_id": "company-001",
                "title": "Internal Fraud Operations",
                "content": (
                    "The fraud operations team monitors high-value "
                    "transaction alerts."
                ),
            }
        ],
        "evidence": [
            {
                "evidence_id": "company-evidence-1",
                "source": "company_knowledge",
                "tool": "search_documents",
                "retrieval_type": "vector_search",
                "data": {
                    "document_id": "company-001",
                    "title": "Internal Fraud Operations",
                    "content": (
                        "The fraud operations team monitors high-value "
                        "transaction alerts."
                    ),
                },
                "metadata": {
                    "provider": "databricks_vector_search",
                },
            }
        ],
        "company_execution_allowed": True,
        "company_execution_reason": (
            "Company Knowledge execution completed successfully."
        ),
        "company_execution_error": "",
        "allowed": True,
        "execution_allowed": True,
        "execution_reason": (
            "Company Knowledge execution completed successfully."
        ),
        "current_stage": "company_knowledge",
    }


async def _mock_external_research_harness_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Deterministic External Research result.

    This mock intentionally mirrors the fields returned by the real
    external_research_harness_node so that orchestration tests do not
    depend on a live Tavily MCP server.
    """

    query = (
        state.get("current_query")
        or state.get("query")
        or ""
    )

    return {
        **state,

        # ------------------------------------------------------------
        # External search results
        # ------------------------------------------------------------

        "web_results": [
            {
                "title": "Current Fraud Intelligence",
                "url": (
                    "https://example.com/fraud-intelligence"
                ),
                "snippet": (
                    "Recent fraud trends affecting financial "
                    "institutions."
                ),
                "content": (
                    "Recent fraud trends affecting financial "
                    "institutions."
                ),
                "provider": "tavily",
                "query": query,
            }
        ],

        # ------------------------------------------------------------
        # Normalized evidence
        # ------------------------------------------------------------

        "evidence": [
            {
                "evidence_id": "web-evidence-1",
                "source": "external_web",
                "tool": "tavily_search",
                "retrieval_type": "web_search",
                "data": {
                    "title": "Current Fraud Intelligence",
                    "url": (
                        "https://example.com/fraud-intelligence"
                    ),
                    "snippet": (
                        "Recent fraud trends affecting financial "
                        "institutions."
                    ),
                    "content": (
                        "Recent fraud trends affecting financial "
                        "institutions."
                    ),
                },
                "metadata": {
                    "provider": "tavily",
                    "query": query,
                },
            }
        ],

        # ------------------------------------------------------------
        # External Research status
        # ------------------------------------------------------------

        "external_research_allowed": True,

        "external_research_execution_allowed": True,

        "external_research_reason": (
            "External web search completed successfully."
        ),

        "external_research_error": "",

        # ------------------------------------------------------------
        # Shared execution status
        # ------------------------------------------------------------

        "allowed": True,

        "execution_allowed": True,

        "execution_reason": (
            "External web search completed successfully."
        ),

        "current_stage": "external_research",
    }


# =====================================================================
# Test: Policy + Fraud
# =====================================================================


@pytest.mark.asyncio
async def test_security_graph_multi_domain_policy_and_fraud(
    monkeypatch,
):
    """
    Verify MULTI_DOMAIN orchestration can combine:

        Fraud Analytics
        Policy RAG

    before reaching the shared Evidence Layer.
    """

    _set_deterministic_supervisor(
        monkeypatch,
        tools=[
            SupervisorTool.SEARCH_POLICIES,
            SupervisorTool.GET_FRAUD_METRICS,
        ],
        requires_policy=True,
        requires_structured_data=True,
        requires_external_research=False,
    )

    monkeypatch.setattr(
        security_graph_module,
        "fraud_agent_node",
        _mock_fraud_agent_node,
    )

    monkeypatch.setattr(
        security_graph_module,
        "policy_rag_harness_node",
        _mock_policy_rag_harness_node,
    )

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": (
            "Compare high-value fraud alerts "
            "with our internal fraud policy."
        ),
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Security boundaries
    # ------------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # ------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------

    assert result["supervisor_domain"] == "MULTI_DOMAIN"

    assert result["requires_policy"] is True
    assert result["requires_structured_data"] is True
    assert result["requires_external_research"] is False

    # ------------------------------------------------------------
    # Multi-domain execution
    # ------------------------------------------------------------

    assert result["allowed"] is True
    assert result["current_stage"] == "output_guardrails"

    # ------------------------------------------------------------
    # Fraud
    # ------------------------------------------------------------

    assert result["fraud_execution_allowed"] is True

    # ------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------

    assert result["policy_execution_allowed"] is True

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    assert result.get("evidence")
    assert len(result["evidence"]) >= 2


# =====================================================================
# Test: Company + Fraud
# =====================================================================


@pytest.mark.asyncio
async def test_security_graph_multi_domain_company_and_fraud(
    monkeypatch,
):
    """
    Verify MULTI_DOMAIN orchestration can combine:

        Fraud Analytics
        Company Knowledge

    before reaching the shared Evidence Layer.
    """

    _set_deterministic_supervisor(
        monkeypatch,
        tools=[
            SupervisorTool.GET_FRAUD_METRICS,
            SupervisorTool.SEARCH_DOCUMENTS,
        ],
        requires_policy=False,
        requires_structured_data=True,
        requires_external_research=False,
    )

    monkeypatch.setattr(
        security_graph_module,
        "fraud_agent_node",
        _mock_fraud_agent_node,
    )

    monkeypatch.setattr(
        security_graph_module,
        "company_knowledge_harness_node",
        _mock_company_knowledge_harness_node,
    )

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": (
            "Compare fraud transactions with "
            "our internal fraud operations."
        ),
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Security boundaries
    # ------------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # ------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------

    assert result["supervisor_domain"] == "MULTI_DOMAIN"

    assert result["requires_policy"] is False
    assert result["requires_structured_data"] is True
    assert result["requires_external_research"] is False

    # ------------------------------------------------------------
    # Multi-domain execution
    # ------------------------------------------------------------

    assert result["allowed"] is True
    assert result["current_stage"] == "output_guardrails"

    # ------------------------------------------------------------
    # Fraud
    # ------------------------------------------------------------

    assert result["fraud_execution_allowed"] is True

    # ------------------------------------------------------------
    # Company Knowledge
    # ------------------------------------------------------------

    assert result["company_execution_allowed"] is True

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    assert result.get("evidence")
    assert len(result["evidence"]) >= 2


# =====================================================================
# Test: Fraud + Policy + Company + External Research
# =====================================================================


@pytest.mark.asyncio
async def test_security_graph_multi_domain_with_external_research(
    monkeypatch,
):
    """
    Verify MULTI_DOMAIN orchestration can combine:

        Fraud
        Policy
        Company Knowledge
        External Research

    before reaching the shared Evidence Layer.
    """

    _set_deterministic_supervisor(
        monkeypatch,
        tools=[
            SupervisorTool.SEARCH_POLICIES,
            SupervisorTool.GET_FRAUD_METRICS,
            SupervisorTool.SEARCH_DOCUMENTS,
        ],
        requires_policy=True,
        requires_structured_data=True,
        requires_external_research=True,
    )

    monkeypatch.setattr(
        security_graph_module,
        "fraud_agent_node",
        _mock_fraud_agent_node,
    )

    monkeypatch.setattr(
        security_graph_module,
        "policy_rag_harness_node",
        _mock_policy_rag_harness_node,
    )

    monkeypatch.setattr(
        security_graph_module,
        "company_knowledge_harness_node",
        _mock_company_knowledge_harness_node,
    )

    monkeypatch.setattr(
        security_graph_module,
        "external_research_harness_node",
        _mock_external_research_harness_node,
    )

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": (
            "Compare our fraud alerts and internal policy "
            "with current external fraud trends."
        ),
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Security boundaries
    # ------------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # ------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------

    assert result["supervisor_domain"] == "MULTI_DOMAIN"

    assert result["requires_policy"] is True
    assert result["requires_structured_data"] is True
    assert result["requires_external_research"] is True

    # ------------------------------------------------------------
    # Multi-domain execution
    # ------------------------------------------------------------

    assert result["allowed"] is True
    assert result["current_stage"] == "output_guardrails"

    # ------------------------------------------------------------
    # Fraud
    # ------------------------------------------------------------

    assert result["fraud_execution_allowed"] is True

    # ------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------

    assert result["policy_execution_allowed"] is True

    # ------------------------------------------------------------
    # Company Knowledge
    # ------------------------------------------------------------

    assert result["company_execution_allowed"] is True

    # ------------------------------------------------------------
    # External Research
    # ------------------------------------------------------------

    assert result["external_research_allowed"] is True

    assert (
        result["external_research_execution_allowed"]
        is True
    )

    assert (
        result["external_research_error"]
        == ""
    )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    assert result.get("evidence")

    assert len(result["evidence"]) >= 4

    evidence_sources = {
        item.get("source")
        for item in result["evidence"]
    }

    assert "fraud_analytics" in evidence_sources
    assert "policy_rag" in evidence_sources
    assert "company_knowledge" in evidence_sources
    assert "external_web" in evidence_sources


# =====================================================================
# Test: Fraud response passes output guardrails
# =====================================================================


@pytest.mark.asyncio
async def test_fraud_response_passes_output_guardrails():
    """
    Verify a normal Fraud Analytics request reaches the response
    generator and passes output guardrails.

    This test intentionally uses the real Fraud MCP integration.
    """

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "Show me high value fraud transactions",
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Security boundaries
    # ------------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # ------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------

    assert result["supervisor_domain"] == "FRAUD_ANALYTICS"

    assert result["agent_name"] == "fraud_agent"

    # ------------------------------------------------------------
    # Fraud execution
    # ------------------------------------------------------------

    assert result["fraud_execution_allowed"] is True

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    assert result.get("evidence")

    # ------------------------------------------------------------
    # Output guardrails
    # ------------------------------------------------------------

    assert result["allowed"] is True
    assert result["current_stage"] == "output_guardrails"

    assert result.get("response") is not None


# =====================================================================
# Test: Fraud metrics response passes output guardrails
# =====================================================================


@pytest.mark.asyncio
async def test_fraud_metrics_response_passes_output_guardrails():
    """
    Verify a Fraud Metrics request reaches output guardrails.

    This test intentionally uses the real Fraud MCP integration.
    """

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": "Show me fraud transaction velocity",
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Security boundaries
    # ------------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # ------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------

    assert result["supervisor_domain"] == "FRAUD_ANALYTICS"

    assert result["agent_name"] == "fraud_agent"

    # ------------------------------------------------------------
    # Fraud execution
    # ------------------------------------------------------------

    assert result["fraud_execution_allowed"] is True

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    assert result.get("evidence")

    # ------------------------------------------------------------
    # Output guardrails
    # ------------------------------------------------------------

    assert result["allowed"] is True
    assert result["current_stage"] == "output_guardrails"

    assert result.get("response") is not None


# =====================================================================
# Test: Unknown user blocked
# =====================================================================


@pytest.mark.asyncio
async def test_unknown_user_never_reaches_response_generator(
    monkeypatch,
):
    """
    Verify an unknown user is blocked before agent execution and
    cannot reach the response generator.
    """

    response_called = False

    async def mock_response_generator(state):
        nonlocal response_called
        response_called = True

        return {
            **state,
            "response": "This should never be generated.",
        }

    monkeypatch.setattr(
        security_graph_module,
        "response_generator_node",
        mock_response_generator,
    )

    graph = build_security_graph()

    state = {
        "user_id": "unknown-user",
        "query": "Show me fraud transactions",
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Authentication / authorization
    # ------------------------------------------------------------

    assert result["authenticated"] is False

    assert result["authorization_allowed"] is False

    # ------------------------------------------------------------
    # Security decision
    # ------------------------------------------------------------

    assert result["allowed"] is False

    # ------------------------------------------------------------
    # Response generator must not execute
    # ------------------------------------------------------------

    assert response_called is False


# =====================================================================
# Test: External Research routing
# =====================================================================


@pytest.mark.asyncio
async def test_security_graph_routes_external_research(
    monkeypatch,
):
    """
    Verify that the Supervisor's EXTERNAL_RESEARCH routing decision
    reaches the External Research node.
    """

    _set_deterministic_supervisor(
        monkeypatch,
        domain=SupervisorDomain.EXTERNAL_RESEARCH,
        tools=[],
        requires_policy=False,
        requires_structured_data=False,
        requires_external_research=True,
    )

    monkeypatch.setattr(
        security_graph_module,
        "external_research_harness_node",
        _mock_external_research_harness_node,
    )

    graph = build_security_graph()

    state = {
        "user_id": "user-001",
        "query": (
            "Find the latest external fraud intelligence "
            "for financial institutions."
        ),
    }

    result = await graph.ainvoke(state)

    # ------------------------------------------------------------
    # Security boundaries
    # ------------------------------------------------------------

    assert result["input_guardrail_allowed"] is True
    assert result["safety_allowed"] is True
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True

    # ------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------

    assert (
        result["supervisor_domain"]
        == SupervisorDomain.EXTERNAL_RESEARCH.value
    )

    assert result["requires_external_research"] is True

    # ------------------------------------------------------------
    # External Research
    # ------------------------------------------------------------

    assert result["external_research_allowed"] is True

    assert (
        result["external_research_execution_allowed"]
        is True
    )

    assert (
        result["external_research_error"]
        == ""
    )

    # ------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------

    assert result.get("evidence")

    evidence_sources = {
        item.get("source")
        for item in result["evidence"]
    }

    assert "external_web" in evidence_sources

    # ------------------------------------------------------------
    # Shared execution
    # ------------------------------------------------------------

    assert result["allowed"] is True
    assert result["current_stage"] == "output_guardrails"