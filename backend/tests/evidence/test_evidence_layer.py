
from __future__ import annotations

from backend.evidence.evidence_layer import (
    evidence_layer_node,
)


def test_evidence_layer_normalizes_policy_evidence():
    state = {
        "query": (
            "What procedure must fraud investigations follow?"
        ),
        "current_stage": "policy_rag",
        "evidence": [
            {
                "evidence_id": "policy-evidence-1",
                "source": "databricks_vector_search",
                "tool": "policy_rag",
                "retrieval_type": "vector",
                "retrieval_source": "databricks_vector_search",
                "data": {
                    "chunk_id": "policy-chunk-001",
                    "document_id": "policy-doc-001",
                    "file_name": "Fraud Investigation Policy.pdf",
                    "page_number": 5,
                    "section": "Investigation Procedure",
                    "document_type": "policy",
                    "chunk_text": (
                        "Fraud investigations must follow "
                        "approved investigation procedures."
                    ),
                },
                "metadata": {
                    "source_domain": "policy",
                    "retrieval_type": "vector",
                    "retrieval_source": (
                        "databricks_vector_search"
                    ),
                },
            }
        ],
    }

    result = evidence_layer_node(state)

    assert result["current_stage"] == "evidence_layer"
    assert "allowed" not in result

    assert "evidence" in result
    assert len(result["evidence"]) == 1

    evidence = result["evidence"][0]

    assert evidence["evidence_id"] == "policy-evidence-1"
    assert evidence["source_domain"] == "policy"
    assert evidence["source"] == "databricks_vector_search"
    assert evidence["retrieval_type"] == "vector"
    assert (
        evidence["retrieval_source"]
        == "databricks_vector_search"
    )
    assert evidence["tool"] == "policy_rag"

    assert (
        evidence["data"]["chunk_id"]
        == "policy-chunk-001"
    )
    assert (
        evidence["data"]["document_id"]
        == "policy-doc-001"
    )
    assert (
        evidence["data"]["file_name"]
        == "Fraud Investigation Policy.pdf"
    )


def test_evidence_layer_normalizes_company_evidence():
    state = {
        "query": "What is the fraud investigation process?",
        "current_stage": "company_knowledge",
        "evidence": [
            {
                "evidence_id": "company-evidence-1",
                "source": "neo4j_graph_rag",
                "tool": "company_knowledge_rag",
                "retrieval_type": "graph",
                "retrieval_source": "neo4j_graph_rag",
                "data": {
                    "chunk_id": "company-chunk-001",
                    "document_id": "company-doc-001",
                    "chunk_text": (
                        "The fraud investigation process "
                        "contains approved investigation steps."
                    ),
                },
                "metadata": {
                    "source_domain": "company",
                    "retrieval_type": "graph",
                    "retrieval_source": "neo4j_graph_rag",
                },
            }
        ],
    }

    result = evidence_layer_node(state)

    evidence = result["evidence"][0]

    assert evidence["source_domain"] == "company"
    assert evidence["source"] == "neo4j_graph_rag"
    assert evidence["retrieval_type"] == "graph"
    assert (
        evidence["retrieval_source"]
        == "neo4j_graph_rag"
    )
    assert evidence["tool"] == "company_knowledge_rag"


def test_evidence_layer_normalizes_fraud_evidence():
    state = {
        "query": "Show high value fraud transactions.",
        "current_stage": "fraud_agent",
        "evidence": [
            {
                "source": "fraud_gold",
                "tool": "get_high_value_transactions",
                "data": [
                    {
                        "transaction_id": "txn-001",
                        "amount": 150000,
                    }
                ],
            }
        ],
    }

    result = evidence_layer_node(state)

    evidence = result["evidence"][0]

    assert evidence["evidence_id"] == "evidence-1"
    assert evidence["source_domain"] == "fraud"
    assert evidence["source"] == "fraud_gold"
    assert evidence["retrieval_type"] == "structured_data"
    assert evidence["retrieval_source"] == "databricks_gold"
    assert evidence["tool"] == "get_high_value_transactions"

    assert len(evidence["data"]) == 1
    assert (
        evidence["data"][0]["transaction_id"]
        == "txn-001"
    )


def test_evidence_layer_preserves_state():
    state = {
        "user_id": "user-001",
        "query": "What is the policy?",
        "authenticated": True,
        "authorization_allowed": True,
        "current_stage": "policy_rag",
        "evidence": [],
    }

    result = evidence_layer_node(state)

    assert result["user_id"] == "user-001"
    assert result["query"] == state["query"]
    assert result["authenticated"] is True
    assert result["authorization_allowed"] is True
    assert result["evidence"] == []
    assert result["current_stage"] == "evidence_layer"


def test_evidence_layer_handles_missing_evidence():
    state = {
        "query": "What is the policy?",
        "current_stage": "policy_rag",
    }

    result = evidence_layer_node(state)

    assert result["evidence"] == []
    assert result["current_stage"] == "evidence_layer"
    assert "allowed" not in result


def test_evidence_layer_fails_closed_on_invalid_normalizer_input(
    monkeypatch,
):
    def failing_normalizer(evidence):
        raise RuntimeError("normalization failure")

    monkeypatch.setattr(
        "backend.evidence.evidence_layer.normalize_evidence",
        failing_normalizer,
    )

    state = {
        "query": "What is the policy?",
        "current_stage": "policy_rag",
        "evidence": [
            {
                "source": "policy_documents",
            }
        ],
    }

    result = evidence_layer_node(state)

    assert result["evidence"] == []
    assert result["allowed"] is False
    assert result["current_stage"] == "evidence_layer"
    assert result["error"] == "RuntimeError"

