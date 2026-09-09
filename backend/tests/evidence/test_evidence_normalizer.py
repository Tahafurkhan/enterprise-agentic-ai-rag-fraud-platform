
from __future__ import annotations

from backend.evidence.evidence_normalizer import normalize_evidence


def test_normalize_company_vector_evidence():
    evidence = {
        "evidence_id": "company-evidence-1",
        "source": "company_knowledge",
        "tool": "company_knowledge_rag",
        "retrieval_type": "vector",
        "data": {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "file_name": "Quality Manual V 3.0.pdf",
            "page_number": 12,
            "section": "Quality Controls",
            "chunk_text": "Quality control procedures...",
        },
        "metadata": {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "file_name": "Quality Manual V 3.0.pdf",
            "page_number": 12,
            "section": "Quality Controls",
            "retrieval_type": "vector",
            "retrieval_source": "databricks_vector_search",
        },
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["evidence_id"] == "company-evidence-1"
    assert normalized["source_domain"] == "company"
    assert normalized["source"] == "company_knowledge"
    assert normalized["retrieval_type"] == "vector"
    assert normalized["retrieval_source"] == "databricks_vector_search"
    assert normalized["tool"] == "company_knowledge_rag"

    assert normalized["data"]["chunk_id"] == "chunk-001"
    assert normalized["data"]["file_name"] == "Quality Manual V 3.0.pdf"

    assert normalized["metadata"]["document_id"] == "doc-001"


def test_normalize_company_graph_evidence():
    evidence = {
        "evidence_id": "company-evidence-2",
        "source": "neo4j_graph_rag",
        "tool": "company_knowledge_rag",
        "retrieval_type": "graph",
        "data": {
            "entity_name": "Quality Department",
            "chunk_text": "Quality Department manages quality controls.",
        },
        "metadata": {
            "retrieval_type": "graph",
            "retrieval_source": "neo4j_graph_rag",
            "graph_labels": ["Department"],
            "name": "Quality Department",
        },
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["evidence_id"] == "company-evidence-2"
    assert normalized["source_domain"] == "company"
    assert normalized["source"] == "neo4j_graph_rag"
    assert normalized["retrieval_type"] == "graph"
    assert normalized["retrieval_source"] == "neo4j_graph_rag"
    assert normalized["tool"] == "company_knowledge_rag"

    assert normalized["data"]["entity_name"] == "Quality Department"
    assert normalized["metadata"]["graph_labels"] == ["Department"]


def test_normalize_policy_vector_evidence():
    evidence = {
        "evidence_id": "policy-evidence-1",
        "source": "policy_documents",
        "tool": "policy_rag",
        "retrieval_type": "vector",
        "data": {
            "chunk_id": "policy-chunk-001",
            "document_id": "policy-doc-001",
            "file_name": "Fraud Investigation Policy.pdf",
            "file_path": (
                "/Volumes/enterprise_rag/source/rag_docs/incoming/"
                "Fraud_policy_documents/Fraud Investigation Policy.pdf"
            ),
            "page_number": 5,
            "section": "Investigation Procedure",
            "document_type": "policy",
            "chunk_text": (
                "Fraud investigations must follow approved procedures."
            ),
        },
        "metadata": {
            "retrieval_type": "vector",
            "retrieval_source": "databricks_vector_search",
            "knowledge_domain": "policy",
            "namespace": "policy_knowledge",
        },
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["evidence_id"] == "policy-evidence-1"
    assert normalized["source_domain"] == "policy"
    assert normalized["source"] == "policy_documents"
    assert normalized["retrieval_type"] == "vector"
    assert normalized["retrieval_source"] == "databricks_vector_search"
    assert normalized["tool"] == "policy_rag"

    assert normalized["data"]["chunk_id"] == "policy-chunk-001"
    assert normalized["data"]["document_type"] == "policy"

    assert normalized["metadata"]["knowledge_domain"] == "policy"
    assert normalized["metadata"]["namespace"] == "policy_knowledge"


def test_normalize_fraud_mcp_structured_evidence():
    evidence = {
        "evidence_id": "fraud-evidence-1",
        "source": "fraud_gold",
        "tool": "get_high_value_transactions",
        "data": {
            "transactions": [
                {
                    "transaction_id": "txn-001",
                    "amount": 150000,
                    "fraud_flag": True,
                }
            ]
        },
        "metadata": {
            "dataset": "enterprise_rag.gold.fraud_transactions",
        },
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["evidence_id"] == "fraud-evidence-1"
    assert normalized["source_domain"] == "fraud"
    assert normalized["source"] == "fraud_gold"
    assert normalized["retrieval_type"] == "structured_data"
    assert normalized["retrieval_source"] == "databricks_gold"
    assert normalized["tool"] == "get_high_value_transactions"

    assert (
        normalized["data"]["transactions"][0]["transaction_id"]
        == "txn-001"
    )
    assert normalized["data"]["transactions"][0]["amount"] == 150000

    assert (
        normalized["metadata"]["dataset"]
        == "enterprise_rag.gold.fraud_transactions"
    )


def test_normalize_multiple_evidence_records():
    evidence = [
        {
            "evidence_id": "company-evidence-1",
            "source": "company_knowledge",
            "tool": "company_knowledge_rag",
            "retrieval_type": "vector",
            "retrieval_source": "databricks_vector_search",
            "data": {"chunk_id": "chunk-001"},
            "metadata": {},
        },
        {
            "evidence_id": "policy-evidence-1",
            "source": "policy_documents",
            "tool": "policy_rag",
            "retrieval_type": "vector",
            "retrieval_source": "databricks_vector_search",
            "data": {"chunk_id": "policy-001"},
            "metadata": {},
        },
    ]

    result = normalize_evidence(evidence)

    assert len(result) == 2

    assert result[0]["source_domain"] == "company"
    assert result[0]["evidence_id"] == "company-evidence-1"

    assert result[1]["source_domain"] == "policy"
    assert result[1]["evidence_id"] == "policy-evidence-1"


def test_normalize_empty_evidence():
    assert normalize_evidence(None) == []
    assert normalize_evidence([]) == []


def test_normalize_preserves_existing_fields():
    evidence = {
        "evidence_id": "company-evidence-1",
        "source": "company_knowledge",
        "tool": "company_knowledge_rag",
        "retrieval_type": "vector",
        "retrieval_source": "databricks_vector_search",
        "relevance_score": 0.94,
        "custom_field": "must-be-preserved",
        "data": {
            "chunk_id": "chunk-001",
        },
        "metadata": {
            "custom_metadata": "must-be-preserved",
        },
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["relevance_score"] == 0.94
    assert normalized["custom_field"] == "must-be-preserved"
    assert normalized["metadata"]["custom_metadata"] == "must-be-preserved"


def test_normalize_infers_company_source_from_tool():
    evidence = {
        "evidence_id": "company-evidence-1",
        "tool": "company_knowledge_rag",
        "data": {
            "chunk_id": "chunk-001",
        },
        "metadata": {},
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["source_domain"] == "company"
    assert normalized["source"] == "company_knowledge"
    assert normalized["tool"] == "company_knowledge_rag"


def test_normalize_infers_policy_source_from_tool():
    evidence = {
        "evidence_id": "policy-evidence-1",
        "tool": "policy_rag",
        "data": {
            "chunk_id": "policy-001",
        },
        "metadata": {},
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["source_domain"] == "policy"
    assert normalized["source"] == "policy_documents"
    assert normalized["tool"] == "policy_rag"


def test_normalize_infers_fraud_source_from_tool():
    evidence = {
        "evidence_id": "fraud-evidence-1",
        "tool": "fraud_mcp",
        "data": {
            "transactions": [],
        },
        "metadata": {},
    }

    result = normalize_evidence(evidence)

    assert len(result) == 1

    normalized = result[0]

    assert normalized["source_domain"] == "fraud"
    assert normalized["retrieval_type"] == "structured_data"
    assert normalized["retrieval_source"] == "fraud_mcp"
    assert normalized["tool"] == "fraud_mcp"

