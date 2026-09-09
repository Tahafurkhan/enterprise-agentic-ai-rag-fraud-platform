from backend.response.response_generator import (
    ResponseGenerator,
    build_response_generator,
)


def test_response_generator_uses_approved_evidence():
    generator = ResponseGenerator()

    evidence = [
        {
            "source": "fraud_gold",
            "tool": "get_high_value_transactions",
            "data": [
                {
                    "transaction_id": "TX001",
                    "amount": 150000,
                },
                {
                    "transaction_id": "TX002",
                    "amount": 200000,
                },
            ],
        }
    ]

    result = generator.generate(
        query="Show me high value fraud transactions",
        evidence=evidence,
    )

    assert result["answer"]
    assert result["evidence_used"] == evidence
    assert result["claims"]

    assert (
        "2 high-value transaction records"
        in result["claims"][0]["claim"]
    )


def test_response_generator_preserves_evidence_source():
    generator = ResponseGenerator()

    evidence = [
        {
            "source": "fraud_gold",
            "tool": "get_transaction_velocity",
            "data": [
                {
                    "velocity_count": 10,
                }
            ],
        }
    ]

    result = generator.generate(
        query="Show transaction velocity",
        evidence=evidence,
    )

    assert result["claims"][0]["source"] == "fraud_gold"
    assert (
        result["claims"][0]["tool"]
        == "get_transaction_velocity"
    )


def test_response_generator_handles_no_evidence():
    generator = ResponseGenerator()

    result = generator.generate(
        query="Show me fraud transactions",
        evidence=[],
    )

    assert result["evidence_used"] == []
    assert result["claims"] == []

    assert (
        "could not find approved evidence"
        in result["answer"]
    )


def test_response_generator_factory():
    generator = build_response_generator()

    assert isinstance(generator, ResponseGenerator)


def test_response_generator_handles_company_evidence():
    generator = ResponseGenerator()

    evidence = [
        {
            "evidence_id": "company-evidence-1",
            "source_domain": "company",
            "source": "neo4j_graph_rag",
            "tool": "company_knowledge_rag",
            "retrieval_type": "graph",
            "data": [
                {
                    "chunk_id": "COMP001",
                    "chunk_text": "Company process information.",
                },
            ],
        }
    ]

    result = generator.generate(
        query="Show company process information",
        evidence=evidence,
    )

    assert result["answer"]
    assert "approved company knowledge evidence" in result["answer"]
    assert "fraud evidence" not in result["answer"]

    assert result["evidence_used"] == evidence
    assert result["claims"][0]["source_domain"] == "company"


def test_response_generator_handles_policy_evidence():
    generator = ResponseGenerator()

    evidence = [
        {
            "evidence_id": "policy-evidence-1",
            "source_domain": "policy",
            "source": "databricks_vector_search",
            "tool": "policy_rag",
            "retrieval_type": "vector",
            "data": [
                {
                    "chunk_id": "POL001",
                    "document_type": "policy",
                    "chunk_text": "Fraud investigation policy.",
                },
                {
                    "chunk_id": "POL002",
                    "document_type": "procedure",
                    "chunk_text": "Investigation procedure.",
                },
            ],
        }
    ]

    result = generator.generate(
        query="What does the fraud investigation policy say?",
        evidence=evidence,
    )

    assert result["answer"]
    assert "approved policy evidence" in result["answer"]
    assert "fraud evidence" not in result["answer"]

    assert result["evidence_used"] == evidence
    assert result["claims"][0]["source_domain"] == "policy"


def test_response_generator_handles_unknown_evidence_domain():
    generator = ResponseGenerator()

    evidence = [
        {
            "source": "approved_source",
            "tool": "approved_tool",
            "data": [
                {
                    "id": "001",
                }
            ],
        }
    ]

    result = generator.generate(
        query="Show approved evidence",
        evidence=evidence,
    )

    assert result["answer"]
    assert "Based on the approved evidence:" in result["answer"]
    assert "fraud evidence" not in result["answer"]
    assert result["claims"][0]["source_domain"] == "unknown"

