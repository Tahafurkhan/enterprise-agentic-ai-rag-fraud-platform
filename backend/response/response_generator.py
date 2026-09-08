"""
Response Generator for the enterprise fraud analytics platform.

The Response Generator:
- receives evidence produced by approved agents
- does not access Databricks directly
- does not call MCP directly
- does not invent evidence
- creates a structured response for downstream output guardrails
"""

from __future__ import annotations

from typing import Any, Dict, List


class ResponseGenerator:
    """
    Converts approved evidence into a structured response.

    This first implementation is deterministic so that the
    evidence-to-response boundary can be tested safely before
    introducing an LLM-based response generator.
    """

    def generate(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate a response using only supplied evidence.

        Args:
            query: Original user query.
            evidence: Evidence returned by approved agent tools.

        Returns:
            Structured response containing:
            - answer
            - evidence_used
            - claims
        """

        if not evidence:
            return {
                "answer": (
                    "I could not find approved evidence to answer "
                    "this request."
                ),
                "evidence_used": [],
                "claims": [],
            }

        claims: List[Dict[str, Any]] = []

        for item in evidence:
            tool = item.get("tool", "unknown")
            source = item.get("source", "unknown")
            data = item.get("data")

            claims.append(
                {
                    "claim": self._build_claim(
                        tool=tool,
                        data=data,
                    ),
                    "source": source,
                    "tool": tool,
                }
            )

        answer = self._build_answer(
            query=query,
            claims=claims,
        )

        return {
            "answer": answer,
            "evidence_used": evidence,
            "claims": claims,
        }

    @staticmethod
    def _build_claim(
        tool: str,
        data: Any,
    ) -> str:
        """
        Build a factual claim from approved tool evidence.

        No external information is introduced here.
        """

        if isinstance(data, list):
            count = len(data)

            if tool == "get_high_value_transactions":
                return (
                    f"Approved fraud evidence contains "
                    f"{count} high-value transaction records."
                )

            if tool == "get_fraud_card_alerts":
                return (
                    f"Approved fraud evidence contains "
                    f"{count} fraud card alert records."
                )

            if tool == "get_transaction_velocity":
                return (
                    f"Approved fraud evidence contains "
                    f"{count} transaction velocity records."
                )

            return (
                f"Approved fraud evidence contains "
                f"{count} records from {tool}."
            )

        return (
            f"Approved fraud evidence was returned by {tool}."
        )

    @staticmethod
    def _build_answer(
        query: str,
        claims: List[Dict[str, Any]],
    ) -> str:
        """
        Build the user-facing answer strictly from generated claims.
        """

        if not claims:
            return (
                "I could not find approved evidence to answer "
                "this request."
            )

        claim_text = " ".join(
            claim["claim"]
            for claim in claims
        )

        return (
            f"Based on the approved fraud evidence: "
            f"{claim_text}"
        )


def build_response_generator() -> ResponseGenerator:
    """
    Factory for the Response Generator.
    """

    return ResponseGenerator()