from backend.agents.supervisor import (
    ENTERPRISE_KNOWLEDGE_TOOLS,
    EXTERNAL_RESEARCH_TOOLS,
    FRAUD_ANALYTICS_TOOLS,
    Supervisor,
    SupervisorDomain,
    SupervisorTool,
)


def test_direct_query_routes_to_direct():
    decision = Supervisor().route("Hello")

    assert decision.domain == SupervisorDomain.DIRECT
    assert decision.tools == frozenset()
    assert decision.is_valid()


def test_high_value_transaction_count_routes_to_fraud_analytics():
    decision = Supervisor().route(
        "How many high-value transactions occurred on August 11?"
    )

    assert decision.domain == SupervisorDomain.FRAUD_ANALYTICS
    assert decision.requires_structured_data
    assert SupervisorTool.QUERY_FRAUD_DATA in decision.tools
    assert decision.is_valid()


def test_policy_question_routes_to_enterprise_knowledge():
    decision = Supervisor().route(
        "What is the policy threshold for a high-value transaction?"
    )

    assert decision.domain == SupervisorDomain.ENTERPRISE_KNOWLEDGE
    assert decision.requires_policy
    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert decision.is_valid()


def test_policy_and_data_question_routes_to_multi_domain():
    decision = Supervisor().route(
        "Which high-value transactions violated our fraud policy on August 11?"
    )

    assert decision.domain == SupervisorDomain.MULTI_DOMAIN
    assert decision.requires_policy
    assert decision.requires_structured_data

    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert SupervisorTool.QUERY_FRAUD_DATA in decision.tools

    assert decision.is_valid()


def test_external_research_routes_to_external_research():
    decision = Supervisor().route(
        "Search the web for the latest fraud regulations."
    )

    assert decision.domain == SupervisorDomain.EXTERNAL_RESEARCH
    assert decision.requires_external_research
    assert SupervisorTool.SEARCH_WEB in decision.tools
    assert decision.is_valid()


def test_policy_plus_external_research_routes_to_multi_domain():
    decision = Supervisor().route(
        "Compare our fraud policy with the latest external regulations."
    )

    assert decision.domain == SupervisorDomain.MULTI_DOMAIN
    assert decision.requires_policy
    assert decision.requires_external_research

    assert SupervisorTool.SEARCH_POLICIES in decision.tools
    assert SupervisorTool.SEARCH_WEB in decision.tools

    assert decision.is_valid()


def test_unknown_query_is_not_executed():
    decision = Supervisor().route(
        "Tell me something completely unrelated to the enterprise platform."
    )

    assert decision.domain == SupervisorDomain.UNKNOWN
    assert decision.tools == frozenset()
    assert decision.is_valid()


def test_empty_query_routes_to_unknown():
    decision = Supervisor().route("")

    assert decision.domain == SupervisorDomain.UNKNOWN
    assert decision.tools == frozenset()
    assert decision.is_valid()


def test_fraud_tools_are_explicitly_defined():
    assert SupervisorTool.QUERY_FRAUD_DATA in FRAUD_ANALYTICS_TOOLS
    assert SupervisorTool.GET_FRAUD_METRICS in FRAUD_ANALYTICS_TOOLS
    assert SupervisorTool.GENERATE_VISUALIZATION in FRAUD_ANALYTICS_TOOLS


def test_knowledge_tools_are_explicitly_defined():
    assert SupervisorTool.SEARCH_DOCUMENTS in ENTERPRISE_KNOWLEDGE_TOOLS
    assert SupervisorTool.SEARCH_POLICIES in ENTERPRISE_KNOWLEDGE_TOOLS
    assert SupervisorTool.GET_DOCUMENT_METADATA in ENTERPRISE_KNOWLEDGE_TOOLS


def test_external_tools_are_explicitly_defined():
    assert SupervisorTool.SEARCH_WEB in EXTERNAL_RESEARCH_TOOLS