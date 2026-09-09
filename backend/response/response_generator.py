
"""
Response Generator for the enterprise fraud intelligence platform.

The Response Generator:
- receives evidence produced by approved agents
- does not access Databricks directly
- does not call MCP directly
- does not retrieve evidence
- does not invent evidence
- creates a structured response for downstream output guardrails
"""

from __future__ import annotations

from typing import Any, Dict, List


class ResponseGenerator:
    """
    Converts approved evidence into a structured response.

    This implementation is deterministic so that the
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
            evidence: Evidence returned by approved agents/tools.

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
            source_domain = self._resolve_source_domain(item)

            claims.append( { "claim": self._build_claim( source_domain=source_domain, tool=tool, data=data, ), "source": source, "tool": tool, "source_domain": source_domain, } )

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
    def _resolve_source_domain(
        evidence: Dict[str, Any],
    ) -> str:
        """
        Resolve the evidence domain from the canonical Evidence contract.

        The Evidence Layer normally provides source_domain.
        The fallbacks support existing evidence while keeping this
        component backward-compatible.
        """

        source_domain = evidence.get("source_domain")
        if source_domain:
            return str(source_domain).lower()

        source = str(evidence.get("source", "")).lower()
        tool = str(evidence.get("tool", "")).lower()

        combined = f"{source} {tool}"

        if "policy" in combined:
            return "policy"

        if "company" in combined:
            return "company"

        if "fraud" in combined or "gold" in combined:
            return "fraud"

        return "unknown"

    @staticmethod
    def _build_claim(
        source_domain: str,
        tool: str,
        data: Any,
    ) -> str:
        """
        Build a factual claim from approved evidence.

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

            if source_domain == "company":
                return (
                    f"Approved company knowledge evidence contains "
                    f"{count} records from {tool}."
                )

            if source_domain == "policy":
                return (
                    f"Approved policy evidence contains "
                    f"{count} records from {tool}."
                )

            if source_domain == "fraud":
                return (
                    f"Approved fraud evidence contains "
                    f"{count} records from {tool}."
                )

            return (
                f"Approved evidence contains "
                f"{count} records from {tool}."
            )

        if source_domain == "company":
            return (
                f"Approved company knowledge evidence "
                f"was returned by {tool}."
            )

        if source_domain == "policy":
            return (
                f"Approved policy evidence was returned by {tool}."
            )

        if source_domain == "fraud":
            return (
                f"Approved fraud evidence was returned by {tool}."
            )

        return (
            f"Approved evidence was returned by {tool}."
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

        domains = {
            claim["source_domain"]
            for claim in claims
            if claim.get("source_domain")
        }

        if domains == {"fraud"}:
            prefix = "Based on the approved fraud evidence:"

        elif domains == {"company"}:
            prefix = "Based on the approved company knowledge evidence:"

        elif domains == {"policy"}:
            prefix = "Based on the approved policy evidence:"

        else:
            prefix = "Based on the approved evidence:"

        return f"{prefix} {claim_text}"


def build_response_generator() -> ResponseGenerator:
    """
    Factory for the Response Generator.
    """

    return ResponseGenerator()

