from langchain_core.documents import Document
import pytest

from backend.rag.hybrid_rag_retriever import HybridRAGRetriever


class FakeVectorRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return self.documents


class FakeGraphRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return self.documents


def test_hybrid_retriever_merges_vector_and_graph_results():
    vector = FakeVectorRetriever(
        [
            Document(
                page_content="Vector evidence one",
                metadata={"chunk_id": "v1"},
            ),
            Document(
                page_content="Vector evidence two",
                metadata={"chunk_id": "v2"},
            ),
        ]
    )

    graph = FakeGraphRetriever(
        [
            Document(
                page_content="Graph evidence one",
                metadata={"node_id": "g1"},
            ),
            Document(
                page_content="Graph evidence two",
                metadata={"node_id": "g2"},
            ),
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.invoke("quality management")

    assert len(documents) == 4

    assert documents[0].page_content == "Vector evidence one"
    assert documents[1].page_content == "Vector evidence two"
    assert documents[2].page_content == "Graph evidence one"
    assert documents[3].page_content == "Graph evidence two"

    assert vector.calls == ["quality management"]
    assert graph.calls == ["quality management"]


def test_vector_documents_receive_source_metadata():
    vector = FakeVectorRetriever(
        [
            Document(
                page_content="Vector evidence",
                metadata={"chunk_id": "v1"},
            )
        ]
    )

    graph = FakeGraphRetriever([])

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.retrieve("test query")

    assert len(documents) == 1

    metadata = documents[0].metadata

    assert metadata["retrieval_type"] == "vector"
    assert metadata["retrieval_source"] == "databricks_vector_search"
    assert metadata["retrieval_rank"] == 1


def test_graph_documents_receive_source_metadata():
    vector = FakeVectorRetriever([])

    graph = FakeGraphRetriever(
        [
            Document(
                page_content="Graph evidence",
                metadata={"node_id": "g1"},
            )
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.retrieve("test query")

    assert len(documents) == 1

    metadata = documents[0].metadata

    assert metadata["retrieval_type"] == "graph"
    assert metadata["retrieval_source"] == "neo4j_graph_rag"
    assert metadata["retrieval_rank"] == 1


def test_duplicate_chunk_ids_are_removed():
    vector = FakeVectorRetriever(
        [
            Document(
                page_content="Same evidence",
                metadata={"chunk_id": "shared-1"},
            )
        ]
    )

    graph = FakeGraphRetriever(
        [
            Document(
                page_content="Same evidence from graph",
                metadata={"chunk_id": "shared-1"},
            ),
            Document(
                page_content="Unique graph evidence",
                metadata={"chunk_id": "graph-2"},
            ),
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.retrieve("test query")

    assert len(documents) == 2

    chunk_ids = [
        document.metadata.get("chunk_id")
        for document in documents
    ]

    assert chunk_ids == [
        "shared-1",
        "graph-2",
    ]


def test_duplicate_document_ids_are_removed():
    vector = FakeVectorRetriever(
        [
            Document(
                page_content="Company document",
                metadata={"document_id": "doc-1"},
            )
        ]
    )

    graph = FakeGraphRetriever(
        [
            Document(
                page_content="Same company document",
                metadata={"document_id": "doc-1"},
            ),
            Document(
                page_content="Another document",
                metadata={"document_id": "doc-2"},
            ),
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.retrieve("company")

    assert len(documents) == 2


def test_duplicate_content_is_removed_when_no_ids_exist():
    vector = FakeVectorRetriever(
        [
            Document(
                page_content="Quality management procedure",
                metadata={},
            )
        ]
    )

    graph = FakeGraphRetriever(
        [
            Document(
                page_content="  QUALITY   MANAGEMENT procedure ",
                metadata={},
            ),
            Document(
                page_content="Different graph evidence",
                metadata={},
            ),
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.retrieve("quality")

    assert len(documents) == 2

    assert documents[0].page_content == "Quality management procedure"
    assert documents[1].page_content == "Different graph evidence"


def test_vector_top_k_limit_is_respected():
    vector = FakeVectorRetriever(
        [
            Document(page_content="V1"),
            Document(page_content="V2"),
            Document(page_content="V3"),
        ]
    )

    graph = FakeGraphRetriever([])

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
        vector_top_k=2,
    )

    documents = retriever.retrieve("test")

    assert len(documents) == 2
    assert [doc.page_content for doc in documents] == [
        "V1",
        "V2",
    ]


def test_graph_top_k_limit_is_respected():
    vector = FakeVectorRetriever([])

    graph = FakeGraphRetriever(
        [
            Document(page_content="G1"),
            Document(page_content="G2"),
            Document(page_content="G3"),
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
        graph_top_k=2,
    )

    documents = retriever.retrieve("test")

    assert len(documents) == 2
    assert [doc.page_content for doc in documents] == [
        "G1",
        "G2",
    ]


def test_final_top_k_limit_is_respected():
    vector = FakeVectorRetriever(
        [
            Document(page_content="V1"),
            Document(page_content="V2"),
            Document(page_content="V3"),
        ]
    )

    graph = FakeGraphRetriever(
        [
            Document(page_content="G1"),
            Document(page_content="G2"),
            Document(page_content="G3"),
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
        final_top_k=4,
    )

    documents = retriever.retrieve("test")

    assert len(documents) == 4


def test_empty_results_are_supported():
    vector = FakeVectorRetriever([])
    graph = FakeGraphRetriever([])

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    documents = retriever.retrieve("test")

    assert documents == []


def test_empty_query_is_rejected():
    vector = FakeVectorRetriever([])
    graph = FakeGraphRetriever([])

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("")


def test_whitespace_query_is_rejected():
    vector = FakeVectorRetriever([])
    graph = FakeGraphRetriever([])

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        retriever.retrieve("   ")


def test_missing_vector_retriever_is_rejected():
    graph = FakeGraphRetriever([])

    with pytest.raises(
        ValueError,
        match="vector_retriever is required",
    ):
        HybridRAGRetriever(
            vector_retriever=None,
            graph_retriever=graph,
        )


def test_missing_graph_retriever_is_rejected():
    vector = FakeVectorRetriever([])

    with pytest.raises(
        ValueError,
        match="graph_retriever is required",
    ):
        HybridRAGRetriever(
            vector_retriever=vector,
            graph_retriever=None,
        )


def test_invalid_retriever_interface_is_rejected():
    class InvalidRetriever:
        pass

    graph = FakeGraphRetriever([])

    retriever = HybridRAGRetriever(
        vector_retriever=InvalidRetriever(),
        graph_retriever=graph,
    )

    with pytest.raises(
        TypeError,
        match="invoke\\(\\) or retrieve\\(\\)",
    ):
        retriever.retrieve("test")


def test_retrieve_with_metadata_returns_statistics():
    vector = FakeVectorRetriever(
        [
            Document(
                page_content="Vector evidence",
                metadata={"chunk_id": "v1"},
            )
        ]
    )

    graph = FakeGraphRetriever(
        [
            Document(
                page_content="Graph evidence",
                metadata={"node_id": "g1"},
            )
        ]
    )

    retriever = HybridRAGRetriever(
        vector_retriever=vector,
        graph_retriever=graph,
    )

    result = retriever.retrieve_with_metadata(
        "quality management"
    )

    assert result["query"] == "quality management"
    assert result["document_count"] == 2
    assert result["vector_documents"] == 1
    assert result["graph_documents"] == 1
    assert len(result["documents"]) == 2