"""
Neo4j Graph RAG Retriever.

Graph RAG is used as a complementary retrieval mechanism
for Company Knowledge.

Architecture:

    User Query
        |
        v
    Graph RAG Retriever
        |
        v
    Neo4j AuraDB
        |
        v
    Graph Evidence
        |
        v
    LangChain Documents

This module performs retrieval only.
It does not generate answers.
It does not use MCP.
It does not perform external web search.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from neo4j import GraphDatabase


load_dotenv()


class Neo4jGraphRAGRetriever:
    """
    LangChain-compatible Graph RAG retriever.

    The initial graph schema is intentionally generic so that
    we can connect AuraDB first and then build the enterprise
    knowledge graph schema from the existing company documents.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        top_k: int = 5,
        driver: Any | None = None,
    ) -> None:

        self.uri = uri or os.getenv(
            "NEO4J_URI",
            "",
        )

        self.username = username or os.getenv(
            "NEO4J_USERNAME",
            "neo4j",
        )

        self.password = password or os.getenv(
            "NEO4J_PASSWORD",
            "",
        )

        self.database = database or os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        )

        self.top_k = top_k

        if driver is not None:
            self.driver = driver
        else:
            if not self.uri:
                raise RuntimeError(
                    "NEO4J_URI is not configured."
                )

            if not self.password:
                raise RuntimeError(
                    "NEO4J_PASSWORD is not configured."
                )

            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(
                    self.username,
                    self.password,
                ),
            )

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    def verify_connectivity(self) -> None:
        """
        Verify that the Neo4j database is reachable.
        """

        self.driver.verify_connectivity()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Return a lightweight Neo4j health result.
        """

        try:
            self.verify_connectivity()

            with self.driver.session(
                database=self.database
            ) as session:

                result = session.run(
                    "RETURN 1 AS healthy"
                )

                record = result.single()

            return {
                "connected": True,
                "healthy": (
                    record is not None
                    and record["healthy"] == 1
                ),
                "database": self.database,
            }

        except Exception as exc:

            return {
                "connected": False,
                "healthy": False,
                "database": self.database,
                "error": type(exc).__name__,
            }

    # ------------------------------------------------------------------
    # Graph Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve graph evidence for a user query.

        The initial implementation performs conservative
        property matching.

        Once the enterprise graph schema is created, this
        query will be replaced with explicit graph traversal
        queries.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        normalized_query = query.strip()

        cypher = """
        MATCH (n)
        WHERE any(
            key IN keys(n)
            WHERE toString(n[key]) CONTAINS $query
        )
        RETURN
            labels(n) AS labels,
            properties(n) AS properties
        LIMIT $top_k
        """

        documents: List[Document] = []

        with self.driver.session(
            database=self.database
        ) as session:

            result = session.run(
                cypher,
                query=normalized_query,
                top_k=self.top_k,
            )

            for record in result:

                labels = record.get(
                    "labels",
                    [],
                )

                properties = record.get(
                    "properties",
                    {},
                )

                if not isinstance(properties, dict):
                    continue

                text_parts = []

                for key, value in properties.items():

                    text_parts.append(
                        f"{key}: {value}"
                    )

                page_content = "\n".join(
                    text_parts
                )

                if not page_content:
                    continue

                metadata = {
                    "source": "company_knowledge_graph",
                    "retrieval_type": "graph",
                    "graph_labels": labels,
                    **properties,
                }

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=metadata,
                    )
                )

        return documents

    # ------------------------------------------------------------------
    # LangChain-compatible interface
    # ------------------------------------------------------------------

    def invoke(
        self,
        query: str,
    ) -> List[Document]:

        return self.retrieve(query)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:

        self.driver.close()