from langchain_core.documents import Document
import pytest

from backend.rag.hybrid_rag_agent import (
    HybridRAGState,
    build_hybrid_rag_agent,
    build_evidence_node,
    retrieve_node,
)


class FakeHybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return self.documents


def test_retrieve_node_returns_hybrid_documents():
    documents = [
        Document(
            page_content="Vector evidence",
            metadata={
                "retrieval_type": "vector",
                "retrieval_source": "databricks_vector_search",
                "chunk_id": "v1",
            },
        ),
        Document(
            page_content="Graph evidence",
            metadata={
                "retrieval_type": "graph",
                "retrieval_source": "neo4j_graph_rag",
                "node_id": "g1",
            },
        ),
    ]

    retriever = FakeHybridRetriever(documents)

    state: HybridRAGState = {
        "question": "quality management",
    }

    result = retrieve_node(
        state,
        retriever,
    )

    assert result["current_query"] == "quality management"
    assert result["retrieved_docs"] == documents

    assert result["vector_documents"] == 1
    assert result["graph_documents"] == 1
    assert result["total_documents"] == 2

    assert result["source_used"] == "hybrid"
    assert result["retrieval_type"] == "hybrid"

    assert retriever.calls == ["quality management"]


def test_retrieve_node_uses_current_query():
    documents = [
        Document(
            page_content="Evidence",
            metadata={
                "retrieval_type": "vector",
            },
        )
    ]

    retriever = FakeHybridRetriever(documents)

    state: HybridRAGState = {
        "question": "original question",
        "current_query": "rewritten question",
    }

    result = retrieve_node(
        state,
        retriever,
    )

    assert result["current_query"] == "rewritten question"
    assert retriever.calls == ["rewritten question"]


def test_retrieve_node_rejects_empty_query():
    retriever = FakeHybridRetriever([])

    state: HybridRAGState = {
        "question": "",
    }

    with pytest.raises(
        ValueError,
        match="Query must not be empty",
    ):
        retrieve_node(
            state,
            retriever,
        )


def test_retrieve_node_rejects_invalid_retriever_result():
    class InvalidRetriever:
        def invoke(self, query):
            return "invalid result"

    state: HybridRAGState = {
        "question": "test",
    }

    with pytest.raises(
        TypeError,
        match="must return a list",
    ):
        retrieve_node(
            state,
            InvalidRetriever(),
        )


def test_retrieve_node_filters_invalid_documents():
    documents = [
        Document(
            page_content="Valid evidence",
            metadata={
                "retrieval_type": "vector",
            },
        ),
        "invalid document",
        None,
    ]

    retriever = FakeHybridRetriever(documents)

    state: HybridRAGState = {
        "question": "test",
    }

    result = retrieve_node(
        state,
        retriever,
    )

    assert len(result["retrieved_docs"]) == 1
    assert result["total_documents"] == 1
    assert result["vector_documents"] == 1
    assert result["graph_documents"] == 0


def test_build_evidence_creates_normalized_evidence():
    documents = [
        Document(
            page_content="Vector evidence",
            metadata={
                "retrieval_type": "vector",
                "retrieval_source": "databricks_vector_search",
                "chunk_id": "v1",
            },
        ),
        Document(
            page_content="Graph evidence",
            metadata={
                "retrieval_type": "graph",
                "retrieval_source": "neo4j_graph_rag",
                "node_id": "g1",
            },
        ),
    ]

    state: HybridRAGState = {
        "retrieved_docs": documents,
    }

    result = build_evidence_node(state)

    evidence = result["evidence"]

    assert len(evidence) == 2

    assert evidence[0]["evidence_id"] == "hybrid-evidence-1"
    assert evidence[0]["source"] == "databricks_vector_search"
    assert evidence[0]["retrieval_type"] == "vector"
    assert evidence[0]["content"] == "Vector evidence"

    assert evidence[1]["evidence_id"] == "hybrid-evidence-2"
    assert evidence[1]["source"] == "neo4j_graph_rag"
    assert evidence[1]["retrieval_type"] == "graph"
    assert evidence[1]["content"] == "Graph evidence"


def test_build_evidence_preserves_metadata():
    document = Document(
        page_content="Evidence",
        metadata={
            "chunk_id": "chunk-123",
            "file_name": "quality_manual.pdf",
            "page_number": 15,
            "retrieval_type": "vector",
            "retrieval_source": "databricks_vector_search",
        },
    )

    state: HybridRAGState = {
        "retrieved_docs": [document],
    }

    result = build_evidence_node(state)

    evidence_metadata = result["evidence"][0]["metadata"]

    assert evidence_metadata["chunk_id"] == "chunk-123"
    assert evidence_metadata["file_name"] == "quality_manual.pdf"
    assert evidence_metadata["page_number"] == 15
    assert (
        evidence_metadata["retrieval_type"]
        == "vector"
    )


def test_build_evidence_handles_empty_documents():
    state: HybridRAGState = {
        "retrieved_docs": [],
    }

    result = build_evidence_node(state)

    assert result["evidence"] == []


def test_build_evidence_handles_missing_metadata():
    document = Document(
        page_content="Evidence without metadata",
    )

    state: HybridRAGState = {
        "retrieved_docs": [document],
    }

    result = build_evidence_node(state)

    evidence = result["evidence"][0]

    assert evidence["evidence_id"] == "hybrid-evidence-1"
    assert evidence["source"] == "unknown"
    assert evidence["retrieval_type"] == "unknown"
    assert evidence["content"] == "Evidence without metadata"


@pytest.mark.asyncio
async def test_hybrid_rag_agent_executes_complete_graph():
    documents = [
        Document(
            page_content="Vector evidence",
            metadata={
                "retrieval_type": "vector",
                "retrieval_source": "databricks_vector_search",
                "chunk_id": "v1",
            },
        ),
        Document(
            page_content="Graph evidence",
            metadata={
                "retrieval_type": "graph",
                "retrieval_source": "neo4j_graph_rag",
                "node_id": "g1",
            },
        ),
    ]

    retriever = FakeHybridRetriever(documents)

    agent = build_hybrid_rag_agent(
        retriever
    )

    result = await agent.ainvoke(
        {
            "question": "quality management",
        }
    )

    assert result["current_query"] == "quality management"

    assert len(result["retrieved_docs"]) == 2
    assert len(result["evidence"]) == 2

    assert result["vector_documents"] == 1
    assert result["graph_documents"] == 1
    assert result["total_documents"] == 2

    assert result["source_used"] == "hybrid"
    assert result["retrieval_type"] == "hybrid"

    assert retriever.calls == ["quality management"]


@pytest.mark.asyncio
async def test_hybrid_rag_agent_handles_empty_results():
    retriever = FakeHybridRetriever([])

    agent = build_hybrid_rag_agent(
        retriever
    )

    result = await agent.ainvoke(
        {
            "question": "unknown topic",
        }
    )

    assert result["retrieved_docs"] == []
    assert result["evidence"] == []

    assert result["vector_documents"] == 0
    assert result["graph_documents"] == 0
    assert result["total_documents"] == 0

    assert result["source_used"] == "hybrid"
    assert result["retrieval_type"] == "hybrid"


def test_build_hybrid_rag_agent_requires_retriever():
    with pytest.raises(
        ValueError,
        match="Hybrid retriever is required",
    ):
        build_hybrid_rag_agent(None)


@pytest.mark.asyncio
async def test_hybrid_rag_agent_preserves_existing_state():
    documents = [
        Document(
            page_content="Evidence",
            metadata={
                "retrieval_type": "vector",
                "retrieval_source": "databricks_vector_search",
            },
        )
    ]

    retriever = FakeHybridRetriever(documents)

    agent = build_hybrid_rag_agent(
        retriever
    )

    result = await agent.ainvoke(
        {
            "question": "test question",
            "current_query": "current query",
            "source_used": "previous_source",
        }
    )

    assert result["question"] == "test question"
    assert result["current_query"] == "current query"

    # Hybrid retrieval intentionally replaces the retrieval source.
    assert result["source_used"] == "hybrid"

    assert len(result["retrieved_docs"]) == 1
    assert len(result["evidence"]) == 1