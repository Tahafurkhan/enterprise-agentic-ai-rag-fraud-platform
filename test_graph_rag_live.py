import asyncio

from backend.rag.graph_rag_agent import build_graph_rag_agent
from backend.rag.graph_rag_retriever import Neo4jGraphRAGRetriever


async def main():
    retriever = Neo4jGraphRAGRetriever()

    try:
        agent = build_graph_rag_agent(
            retriever=retriever,
        )

        result = await agent.ainvoke(
            {
                "question": "Which department owns the Expense Reimbursement process?",
                "current_query": "Which department owns the Expense Reimbursement process?",
                "retrieved_docs": [],
                "evidence": [],
            }
        )

        print("\n=== GRAPH RAG RESULT ===")
        print(result)

    finally:
        retriever.close()


if __name__ == "__main__":
    asyncio.run(main())
