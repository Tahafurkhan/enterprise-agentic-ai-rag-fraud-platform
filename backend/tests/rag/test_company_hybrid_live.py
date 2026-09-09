import asyncio
import sys
from pathlib import Path


# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from backend.core.llm import llm
from backend.rag.company_rag_adapter import (
    build_company_knowledge_node,
)


async def main():
    print("\n=== COMPANY HYBRID RAG LIVE TEST ===\n")

    company_node = build_company_knowledge_node(
        llm=llm,
    )

    query = (
        "Which department owns the Expense Reimbursement process?"
    )

    print(f"Query: {query}\n")

    print(
        "Running Company Knowledge RAG with Hybrid Retrieval..."
    )
    print("  - Databricks Vector Search")
    print("  - Neo4j Graph RAG")
    print("  - Hybrid RAG Retriever")
    print("  - Cross-Encoder Reranker")
    print()

    result = await company_node(
        {
            "query": query,
        }
    )

    print("=== RESULT ===")

    print("\nAnswer:")
    print(result.get("answer"))

    print("\nSource:")
    print(result.get("source_used"))

    print("\nGroundedness:")
    print(result.get("groundedness_score"))

    print("\nCompleteness:")
    print(result.get("completeness_score"))

    print("\nCitation score:")
    print(result.get("citation_score"))

    print("\nAnswer grade:")
    print(result.get("answer_grade"))

    print("\n=== EVIDENCE ===")

    evidence = result.get("evidence", [])

    if not evidence:
        print("No evidence returned.")

    else:
        for index, item in enumerate(
            evidence,
            start=1,
        ):
            print(
                f"\n--- Evidence {index} ---"
            )

            print(
                "Evidence ID:",
                item.get("evidence_id"),
            )

            print(
                "Source:",
                item.get("source"),
            )

            print(
                "Tool:",
                item.get("tool"),
            )

            print(
                "Retrieval type:",
                item.get("retrieval_type"),
            )

            print(
                "Retrieval source:",
                item.get("retrieval_source"),
            )

            data = item.get("data", {})

            print(
                "File:",
                data.get("file_name"),
            )

            print(
                "Page:",
                data.get("page_number"),
            )

            print("Content:")

            content = data.get("chunk_text")

            if content:
                print(content)

            else:
                print(
                    "No chunk text available."
                )

    print("\n=== EVIDENCE SOURCE SUMMARY ===")

    vector_evidence = [
        item
        for item in evidence
        if item.get("retrieval_type") == "vector"
    ]

    graph_evidence = [
        item
        for item in evidence
        if item.get("retrieval_type") == "graph"
    ]

    print(
        "Vector evidence:",
        len(vector_evidence),
    )

    print(
        "Graph evidence:",
        len(graph_evidence),
    )

    print("\n=== RAW RESULT KEYS ===")

    print(
        list(result.keys())
    )


if __name__ == "__main__":
    asyncio.run(main())