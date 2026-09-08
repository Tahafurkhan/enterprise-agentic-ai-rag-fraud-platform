"""
Company Knowledge RAG Agent.

This agent handles enterprise/company knowledge questions by retrieving
approved documents from Databricks Vector Search and generating an
evidence-grounded answer with the shared application LLM.

Architecture:

    Supervisor
        |
        v
    Company RAG Agent
        |
        +--> Databricks Vector Search
        |
        +--> Retrieved Documents
        |
        +--> Evidence
        |
        v
    Shared Application LLM
        |
        v
    Grounded Answer
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from backend.agents.graph_state import AgentState
from backend.core.llm import llm
from backend.rag.databricks_retriever import DatabricksVectorSearchRetriever


COMPANY_RAG_SYSTEM_PROMPT = """
You are the Company Knowledge Assistant for an enterprise environment.

Answer the user's question using ONLY the provided company knowledge
evidence.

Rules:
1. Do not invent facts.
2. Do not use knowledge outside the provided evidence.
3. If the evidence is insufficient, clearly say that the approved company
   knowledge does not contain enough information to answer.
4. Keep the answer concise and factual.
5. Preserve important document terminology.
6. Do not expose sensitive information that is not present in the evidence.
7. Do not mention internal implementation details such as Vector Search,
   embeddings, LangGraph, or retrieval infrastructure.
"""


class CompanyRAGAgent:
    """
    Company Knowledge RAG agent.

    Responsibilities:
    - Validate that the Supervisor routed the request to enterprise knowledge.
    - Retrieve approved company documents.
    - Convert retrieved documents into governed evidence.
    - Generate an evidence-grounded answer.
    """

    def __init__(
        self,
        retriever: DatabricksVectorSearchRetriever | None = None,
    ) -> None:
        self.retriever = retriever or DatabricksVectorSearchRetriever(
            num_results=5
        )

    # ============================================================
    # SCOPE
    # ============================================================

    @staticmethod
    def scope_node(state: AgentState) -> Dict[str, Any]:
        """
        Confirm that this agent is allowed to handle the request.
        """

        domain = state.get("supervisor_domain")
        tools = state.get("supervisor_tools", [])

        normalized_tools = {
            str(tool).lower()
            for tool in tools
        }

        allowed_domain = domain == "ENTERPRISE_KNOWLEDGE"
        allowed_tool = (
            "search_documents" in normalized_tools
            or "SupervisorTool.SEARCH_DOCUMENTS".lower() in normalized_tools
        )

        if not allowed_domain:
            return {
                "agent_name": "company_rag_agent",
                "allowed": False,
                "current_stage": "company_rag_scope",
                "error": (
                    "Company RAG Agent received a request outside the "
                    "ENTERPRISE_KNOWLEDGE domain."
                ),
            }

        if not allowed_tool:
            return {
                "agent_name": "company_rag_agent",
                "allowed": False,
                "current_stage": "company_rag_scope",
                "error": (
                    "Company RAG Agent was not granted the "
                    "SEARCH_DOCUMENTS capability."
                ),
            }

        return {
            "agent_name": "company_rag_agent",
            "allowed": True,
            "current_stage": "company_rag_scope",
        }

    # ============================================================
    # RETRIEVAL
    # ============================================================

    def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve approved company knowledge from Databricks Vector Search.
        """

        if not state.get("allowed", False):
            return {
                "current_stage": "company_rag_retrieval",
            }

        query = state.get("query", "").strip()

        if not query:
            return {
                "current_stage": "company_rag_retrieval",
                "error": "Company RAG query is empty.",
            }

        documents = self.retriever.retrieve(query)

        existing_evidence = list(state.get("evidence", []))
        existing_tool_results = list(state.get("tool_results", []))

        company_evidence = [
            self._document_to_evidence(document)
            for document in documents
        ]

        existing_evidence.extend(company_evidence)

        existing_tool_results.append(
            {
                "source": "company_knowledge",
                "tool": "search_documents",
                "query": query,
                "result_count": len(documents),
            }
        )

        return {
            "current_stage": "company_rag_retrieval",
            "evidence": existing_evidence,
            "tool_results": existing_tool_results,
        }

    # ============================================================
    # ANSWER
    # ============================================================

    def answer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate an answer grounded only in retrieved company evidence.
        """

        if not state.get("allowed", False):
            return {
                "current_stage": "company_rag_answer",
            }

        evidence = state.get("evidence", [])
        query = state.get("query", "").strip()

        company_evidence = [
            item
            for item in evidence
            if item.get("source") == "company_knowledge"
        ]

        if not company_evidence:
            answer = (
                "I could not find approved company knowledge evidence "
                "to answer this request."
            )

            return {
                "current_stage": "company_rag_answer",
                "response": answer,
                "generated_response": {
                    "answer": answer,
                    "evidence_used": [],
                    "claims": [],
                },
            }

        context = self._format_evidence(company_evidence)

        messages = [
            SystemMessage(content=COMPANY_RAG_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question:\n{query}\n\n"
                    f"Approved company knowledge evidence:\n{context}\n\n"
                    "Answer the user's question using only this evidence."
                )
            ),
        ]

        result = llm.invoke(messages)

        answer = self._extract_text(result)

        claims = [
            {
                "claim": answer,
                "source": "company_knowledge",
                "tool": "search_documents",
            }
        ]

        generated_response = {
            "answer": answer,
            "evidence_used": company_evidence,
            "claims": claims,
        }

        return {
            "current_stage": "company_rag_answer",
            "response": answer,
            "generated_response": generated_response,
        }

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _document_to_evidence(document: Document) -> Dict[str, Any]:
        """
        Convert a LangChain Document into the platform evidence format.
        """

        metadata = dict(document.metadata)

        return {
            "source": "company_knowledge",
            "tool": "search_documents",
            "content": document.page_content,
            "file_name": metadata.get("file_name"),
            "file_path": metadata.get("file_path"),
            "document_id": metadata.get("document_id"),
            "page_number": metadata.get("page_number"),
            "section": metadata.get("section"),
            "chunk_id": metadata.get("chunk_id"),
            "score": metadata.get("score"),
        }

    @staticmethod
    def _format_evidence(
        evidence: List[Dict[str, Any]],
    ) -> str:
        """
        Format retrieved evidence for the LLM.
        """

        formatted: List[str] = []

        for index, item in enumerate(evidence, start=1):
            file_name = item.get("file_name") or "Unknown document"
            page_number = item.get("page_number")
            section = item.get("section")
            content = item.get("content", "")

            source_line = f"Source {index}: {file_name}"

            if page_number is not None:
                source_line += f", page {page_number}"

            if section:
                source_line += f", section {section}"

            formatted.append(
                f"{source_line}\n{content}"
            )

        return "\n\n".join(formatted)

    @staticmethod
    def _extract_text(result: Any) -> str:
        """
        Normalize the response returned by the application LLM.
        """

        content = getattr(result, "content", result)

        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "\n".join(
                str(item)
                for item in content
                if item is not None
            ).strip()
        else:
            text = str(content).strip()

        if not text:
            return (
                "I could not generate an answer from the approved "
                "company knowledge evidence."
            )

        return text


def build_company_rag_agent(
    retriever: DatabricksVectorSearchRetriever | None = None,
):
    """
    Build the Company Knowledge RAG LangGraph.
    """

    agent = CompanyRAGAgent(retriever=retriever)

    graph = StateGraph(AgentState)

    graph.add_node(
        "company_rag_scope",
        agent.scope_node,
    )

    graph.add_node(
        "company_rag_retrieval",
        agent.retrieval_node,
    )

    graph.add_node(
        "company_rag_answer",
        agent.answer_node,
    )

    graph.add_edge(
        START,
        "company_rag_scope",
    )

    graph.add_edge(
        "company_rag_scope",
        "company_rag_retrieval",
    )

    graph.add_edge(
        "company_rag_retrieval",
        "company_rag_answer",
    )

    graph.add_edge(
        "company_rag_answer",
        END,
    )

    return graph.compile()