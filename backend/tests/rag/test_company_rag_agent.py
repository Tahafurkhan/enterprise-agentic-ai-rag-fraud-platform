
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

import backend.rag.company_rag_agent as company_rag_agent
from backend.cache.retrieval_cache import build_retrieval_cache
from backend.rag.company_rag_agent import (
    build_company_knowledge_agent,
)


@pytest.fixture(autouse=True)
def isolate_retrieval_cache(monkeypatch):
    monkeypatch.setattr(
        company_rag_agent,
        "build_configured_retrieval_cache",
        lambda namespace: build_retrieval_cache(
            ttl_seconds=300,
            namespace=namespace,
        ),
    )





class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        return self.documents


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append(query)
        return [
            (document, 1.0)
            for document in documents
        ]


class FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class FakeEvidenceResult:
    def __init__(
        self,
        grade="good",
        score=0.95,
        reason="Relevant evidence.",
    ):
        self.grade = grade
        self.score = score
        self.reason = reason


class FakeAnswerResult:
    def __init__(
        self,
        groundedness_score=0.95,
        completeness_score=0.95,
        citation_score=0.95,
        decision="pass",
        feedback="Answer is grounded.",
    ):
        self.groundedness_score = groundedness_score
        self.completeness_score = completeness_score
        self.citation_score = citation_score
        self.decision = decision
        self.feedback = feedback


class FakeStructuredLLM:
    def __init__(
        self,
        evidence_result=None,
        answer_result=None,
        answer_results=None,
    ):
        self.evidence_result = evidence_result
        self.answer_result = answer_result
        self.answer_results = answer_results
        self.answer_index = 0

    def invoke(self, prompt):
        if self.evidence_result is not None:
            return self.evidence_result

        if self.answer_results is not None:
            result = self.answer_results[
                min(
                    self.answer_index,
                    len(self.answer_results) - 1,
                )
            ]
            self.answer_index += 1
            return result

        return self.answer_result


class FakeLLM:
    def __init__(
        self,
        answer="Approved answer.",
        evidence_result=None,
        answer_result=None,
        rewritten_query="improved enterprise retrieval query",
    ):
        self.answer = answer
        self.evidence_result = (
            evidence_result
            or FakeEvidenceResult()
        )
        self.answer_result = (
            answer_result
            or FakeAnswerResult()
        )
        self.rewritten_query = rewritten_query

    def with_structured_output(
        self,
        schema,
        method=None,
    ):
        name = getattr(
            schema,
            "__name__",
            "",
        )

        if name == "EvidenceGrade":
            return FakeStructuredLLM(
                evidence_result=self.evidence_result
            )

        return FakeStructuredLLM(
            answer_result=self.answer_result
        )

    def invoke(self, prompt):
        if (
            "improved retrieval query"
            in prompt.lower()
        ):
            return FakeLLMResponse(
                self.rewritten_query
            )

        if (
            "better retrieval query"
            in prompt.lower()
        ):
            return FakeLLMResponse(
                self.rewritten_query
            )

        return FakeLLMResponse(
            self.answer
        )


def document():
    return Document(
        page_content=(
            "The quality management system "
            "requires documented corrective action."
        ),
        metadata={
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "file_name": "Quality Manual V 3.0.pdf",
            "page_number": 35,
            "section": "Corrective Action",
        },
    )


def test_company_rag_passes_with_good_evidence():

    retriever = FakeRetriever(
        [document()]
    )

    reranker = FakeReranker()

    llm = FakeLLM()

    graph = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = graph.invoke(
        {
            "question": (
                "What does the quality manual "
                "say about corrective action?"
            ),
            "current_query": (
                "What does the quality manual "
                "say about corrective action?"
            ),
            "retry_count": 0,
        }
    )

    assert (
        result["answer_grade"]
        == "pass"
    )

    assert (
        result["source_used"]
        == "company_knowledge"
    )

    assert len(
        result["evidence"]
    ) == 1


def test_company_rag_performs_corrective_query_rewrite():

    retriever = FakeRetriever(
        [document()]
    )

    reranker = FakeReranker()

    llm = FakeLLM(
        evidence_result=FakeEvidenceResult(
            grade="weak",
            score=0.30,
            reason="Evidence is insufficient.",
        )
    )

    graph = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = graph.invoke(
        {
            "question": (
                "Explain the quality procedure."
            ),
            "current_query": (
                "quality procedure"
            ),
            "retry_count": 0,
        }
    )

    assert result["retry_count"] >= 1

    assert (
        retriever.calls[0]
        == "quality procedure"
    )

    assert (
        "improved enterprise retrieval query"
        in retriever.calls
    )


def test_company_rag_self_rag_retries_failed_answer():

    retriever = FakeRetriever(
        [document()]
    )

    reranker = FakeReranker()

    answer_results = [
        FakeAnswerResult(
            groundedness_score=0.40,
            completeness_score=0.50,
            citation_score=0.40,
            decision="retry",
            feedback=(
                "The answer needs stronger "
                "supporting evidence."
            ),
        ),
        FakeAnswerResult(
            groundedness_score=0.95,
            completeness_score=0.95,
            citation_score=0.95,
            decision="pass",
            feedback="Answer is grounded.",
        ),
    ]

    class SequentialLLM(FakeLLM):

        def with_structured_output(
            self,
            schema,
            method=None,
        ):
            name = getattr(
                schema,
                "__name__",
                "",
            )

            if name == "EvidenceGrade":
                return FakeStructuredLLM(
                    evidence_result=FakeEvidenceResult()
                )

            return FakeStructuredLLM(
                answer_results=answer_results
            )

    llm = SequentialLLM()

    graph = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = graph.invoke(
        {
            "question": (
                "What is corrective action?"
            ),
            "current_query": (
                "What is corrective action?"
            ),
            "retry_count": 0,
        }
    )

    assert (
        result["answer_grade"]
        == "pass"
    )

    assert (
        result["retry_count"]
        >= 1
    )


def test_company_rag_stops_after_max_retries():

    retriever = FakeRetriever(
        [document()]
    )

    reranker = FakeReranker()

    llm = FakeLLM(
        evidence_result=FakeEvidenceResult(
            grade="weak",
            score=0.10,
            reason="Insufficient evidence.",
        )
    )

    graph = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = graph.invoke(
        {
            "question": (
                "Find an unsupported procedure."
            ),
            "current_query": (
                "unsupported procedure"
            ),
            "retry_count": 2,
        }
    )

    assert (
        "could not find sufficient"
        in result["answer"].lower()
    )

    assert (
        result["evidence"]
        == []
    )


def test_company_rag_builds_provenance_evidence():

    retriever = FakeRetriever(
        [document()]
    )

    reranker = FakeReranker()

    llm = FakeLLM()

    graph = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = graph.invoke(
        {
            "question": (
                "What is corrective action?"
            ),
            "current_query": (
                "What is corrective action?"
            ),
            "retry_count": 0,
        }
    )

    evidence = result["evidence"][0]

    assert (
        evidence["source"]
        == "company_knowledge"
    )

    assert (
        evidence["tool"]
        == "company_knowledge_rag"
    )

    assert (
        evidence["data"]["chunk_id"]
        == "chunk-001"
    )

    assert (
        evidence["data"]["file_name"]
        == "Quality Manual V 3.0.pdf"
    )


def test_company_rag_rejects_llm_pass_below_threshold():

    retriever = FakeRetriever(
        [document()]
    )

    reranker = FakeReranker()

    llm = FakeLLM(
        answer_result=FakeAnswerResult(
            groundedness_score=0.95,
            completeness_score=0.60,
            citation_score=0.95,
            decision="pass",
            feedback=(
                "The answer is not complete."
            ),
        )
    )

    graph = build_company_knowledge_agent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
    )

    result = graph.invoke(
        {
            "question": (
                "What is corrective action?"
            ),
            "current_query": (
                "What is corrective action?"
            ),
            "retry_count": 2,
        }
    )

    assert (
        result["answer_grade"]
        == "retry"
    )

 