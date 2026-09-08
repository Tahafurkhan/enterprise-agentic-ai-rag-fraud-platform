"""
Output Guardrails for the enterprise fraud analytics platform.

All generated responses must pass through this layer before they
are allowed to reach the user-facing chat interface.

The guardrails check:
- sensitive data leakage
- unsupported claims
- evidence grounding
- unauthorized/internal system information
- unsafe output
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class OutputGuardrails:
    """
    Validates generated responses before they reach the user.
    """

    # Basic patterns for sensitive information.
    # These are intentionally conservative first-stage checks.
    PAN_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    UNSAFE_PATTERNS = (
        "ignore previous instructions",
        "system prompt",
        "developer prompt",
        "secret key",
        "api key",
        "password",
    )

    def validate(
        self,
        response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate a generated response.

        Returns:
            {
                "allowed": bool,
                "reason": str,
                "safe_response": str,
            }
        """

        if not isinstance(response, dict):
            return {
                "allowed": False,
                "reason": "Response must be a dictionary.",
                "safe_response": "",
            }

        answer = response.get("answer", "")
        evidence = response.get("evidence_used", [])
        claims = response.get("claims", [])

        if not isinstance(answer, str):
            return {
                "allowed": False,
                "reason": "Response answer must be text.",
                "safe_response": "",
            }

        # --------------------------------------------------------
        # 1. Sensitive-data leakage
        # --------------------------------------------------------

        if self._contains_sensitive_data(answer):
            return {
                "allowed": False,
                "reason": "Sensitive information detected in response.",
                "safe_response": "",
            }

        # --------------------------------------------------------
        # 2. Unsafe/system information
        # --------------------------------------------------------

        if self._contains_unsafe_content(answer):
            return {
                "allowed": False,
                "reason": "Unsafe or restricted information detected.",
                "safe_response": "",
            }

        # --------------------------------------------------------
        # 3. Evidence must exist for factual claims
        # --------------------------------------------------------

        if claims and not evidence:
            return {
                "allowed": False,
                "reason": (
                    "Response contains claims without supporting evidence."
                ),
                "safe_response": "",
            }

        # --------------------------------------------------------
        # 4. Every claim must have a source and tool
        # --------------------------------------------------------

        for claim in claims:
            if not isinstance(claim, dict):
                return {
                    "allowed": False,
                    "reason": "Invalid claim structure.",
                    "safe_response": "",
                }

            if not claim.get("claim"):
                return {
                    "allowed": False,
                    "reason": "Claim text is missing.",
                    "safe_response": "",
                }

            if not claim.get("source"):
                return {
                    "allowed": False,
                    "reason": "Claim source is missing.",
                    "safe_response": "",
                }

            if not claim.get("tool"):
                return {
                    "allowed": False,
                    "reason": "Claim tool provenance is missing.",
                    "safe_response": "",
                }

        # --------------------------------------------------------
        # 5. Evidence provenance validation
        # --------------------------------------------------------

        for item in evidence:
            if not isinstance(item, dict):
                return {
                    "allowed": False,
                    "reason": "Invalid evidence structure.",
                    "safe_response": "",
                }

            if not item.get("source"):
                return {
                    "allowed": False,
                    "reason": "Evidence source is missing.",
                    "safe_response": "",
                }

            if not item.get("tool"):
                return {
                    "allowed": False,
                    "reason": "Evidence tool provenance is missing.",
                    "safe_response": "",
                }

        # --------------------------------------------------------
        # 6. Final response
        # --------------------------------------------------------

        return {
            "allowed": True,
            "reason": "Output passed all guardrail checks.",
            "safe_response": answer,
        }

    def _contains_sensitive_data(self, text: str) -> bool:
        """
        Detect obvious sensitive customer information.
        """

        if self.PAN_PATTERN.search(text):
            return True

        if self.EMAIL_PATTERN.search(text):
            return True

        return False

    def _contains_unsafe_content(self, text: str) -> bool:
        """
        Detect obvious attempts to expose internal instructions,
        secrets, or restricted system information.
        """

        normalized = text.lower()

        return any(
            pattern in normalized
            for pattern in self.UNSAFE_PATTERNS
        )


def build_output_guardrails() -> OutputGuardrails:
    """
    Factory for OutputGuardrails.
    """

    return OutputGuardrails()