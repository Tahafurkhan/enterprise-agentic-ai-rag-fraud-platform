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
    Graph Traversal
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

    The retriever performs two stages:

    1. Find relevant graph entities using keyword matching.
    2. Traverse relationships around those entities to produce
       relationship-aware graph evidence.

    This allows questions such as:

        "Which department owns the Expense Reimbursement process?"

    to retrieve the actual graph relationship:

        Finance --OWNS_PROCESS--> Expense Reimbursement
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
    # Query Processing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_search_terms(query: str) -> List[str]:
        """
        Extract meaningful terms from a natural-language query.
        """

        stop_words = {
            "which",
            "what",
            "where",
            "when",
            "who",
            "whom",
            "why",
            "how",
            "does",
            "do",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "to",
            "for",
            "in",
            "on",
            "by",
            "and",
            "or",
            "with",
            "from",
            "owns",
            "own",
            "owned",
            "department",
            "process",
            "procedure",
            "policy",
            "tell",
            "me",
            "about",
            "please",
        }

        terms = [
            term.lower().strip("?,.!:;()[]{}")
            for term in query.split()
        ]

        terms = [
            term
            for term in terms
            if len(term) >= 3
            and term not in stop_words
        ]

        return terms

    # ------------------------------------------------------------------
    # Graph Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve relationship-aware graph evidence.

        The process is:

            Natural-language query
                    |
                    v
            Search terms
                    |
                    v
            Matching graph nodes
                    |
                    v
            One-hop graph traversal
                    |
                    v
            Graph evidence
        """

        if not query or not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        normalized_query = query.strip()

        search_terms = self._extract_search_terms(
            normalized_query
        )

        if not search_terms:
            search_terms = [
                normalized_query.lower()
            ]

        cypher = """
        MATCH (matched)
        WHERE any(
            term IN $search_terms
            WHERE any(
                key IN keys(matched)
                WHERE toLower(
                    toString(matched[key])
                ) CONTAINS term
            )
        )

        OPTIONAL MATCH (matched)-[out_rel]->(out_node)

        OPTIONAL MATCH (in_node)-[in_rel]->(matched)

        WITH
            matched,
            collect(
                DISTINCT {
                    direction: "outgoing",
                    relationship: type(out_rel),
                    node_labels: labels(out_node),
                    node_properties: properties(out_node)
                }
            ) AS outgoing,
            collect(
                DISTINCT {
                    direction: "incoming",
                    relationship: type(in_rel),
                    node_labels: labels(in_node),
                    node_properties: properties(in_node)
                }
            ) AS incoming

        RETURN
            labels(matched) AS matched_labels,
            properties(matched) AS matched_properties,
            outgoing,
            incoming

        LIMIT $top_k
        """

        documents: List[Document] = []

        with self.driver.session(
            database=self.database
        ) as session:

            result = session.run(
                cypher,
                search_terms=search_terms,
                top_k=self.top_k,
            )

            for record in result:

                matched_labels = record.get(
                    "matched_labels",
                    record.get("labels", []),
                )

                matched_properties = record.get(
                    "matched_properties",
                    record.get("properties", {}),
                )

                outgoing = record.get(
                    "outgoing",
                    [],
                )

                incoming = record.get(
                    "incoming",
                    [],
                )

                if not isinstance(
                    matched_properties,
                    dict,
                ):
                    continue

                text_parts: List[str] = []

                # Matched node
                text_parts.append(
                    "MATCHED NODE"
                )

                text_parts.append(
                    f"Labels: {matched_labels}"
                )

                for key, value in matched_properties.items():
                    text_parts.append(
                        f"{key}: {value}"
                    )

                # Outgoing relationships
                for relationship in outgoing:

                    if not relationship:
                        continue

                    relationship_type = relationship.get(
                        "relationship"
                    )

                    node_labels = relationship.get(
                        "node_labels",
                        [],
                    )

                    node_properties = relationship.get(
                        "node_properties",
                        {},
                    )

                    if not relationship_type:
                        continue

                    text_parts.append(
                        ""
                    )

                    text_parts.append(
                        "OUTGOING RELATIONSHIP"
                    )

                    text_parts.append(
                        f"Relationship: "
                        f"{relationship_type}"
                    )

                    text_parts.append(
                        f"Target labels: "
                        f"{node_labels}"
                    )

                    if isinstance(
                        node_properties,
                        dict,
                    ):
                        for key, value in node_properties.items():
                            text_parts.append(
                                f"Target {key}: {value}"
                            )

                # Incoming relationships
                for relationship in incoming:

                    if not relationship:
                        continue

                    relationship_type = relationship.get(
                        "relationship"
                    )

                    node_labels = relationship.get(
                        "node_labels",
                        [],
                    )

                    node_properties = relationship.get(
                        "node_properties",
                        {},
                    )

                    if not relationship_type:
                        continue

                    text_parts.append(
                        ""
                    )

                    text_parts.append(
                        "INCOMING RELATIONSHIP"
                    )

                    text_parts.append(
                        f"Relationship: "
                        f"{relationship_type}"
                    )

                    text_parts.append(
                        f"Source labels: "
                        f"{node_labels}"
                    )

                    if isinstance(
                        node_properties,
                        dict,
                    ):
                        for key, value in node_properties.items():
                            text_parts.append(
                                f"Source {key}: {value}"
                            )

                page_content = "\n".join(
                    text_parts
                )

                if not page_content:
                    continue

                metadata = {
                    "source": (
                        "company_knowledge_graph"
                    ),
                    "retrieval_type": "graph",
                    "graph_labels": matched_labels,
                    "search_terms": search_terms,
                    "outgoing_relationship_count": (
                        len(outgoing)
                        if isinstance(outgoing, list)
                        else 0
                    ),
                    "incoming_relationship_count": (
                        len(incoming)
                        if isinstance(incoming, list)
                        else 0
                    ),
                    **matched_properties,
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