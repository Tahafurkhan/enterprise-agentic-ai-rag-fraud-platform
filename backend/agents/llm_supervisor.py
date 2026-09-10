import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

from .supervisor import (
    ENTERPRISE_KNOWLEDGE_TOOLS,
    EXTERNAL_RESEARCH_TOOLS,
    FRAUD_ANALYTICS_TOOLS,
    Supervisor,
    SupervisorDecision,
    SupervisorDomain,
    SupervisorTool,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-oss-20b"


@dataclass
class LLMRouteDecision:
    domain: SupervisorDomain
    tools: List[SupervisorTool]
    reason: str
    confidence: float
    requires_policy: bool
    requires_structured_data: bool
    requires_external_research: bool


SUPERVISOR_SYSTEM_PROMPT = """
You are the Supervisor Agent for an enterprise fraud intelligence platform.

Your job is ONLY to classify a user's request and determine which internal
evidence sources or tools are required.

You do NOT execute tools.
You do NOT query databases.
You do NOT search documents.
You do NOT search the web.
You do NOT answer the user's question.

Supported domains:

1. DIRECT
   - Greetings
   - Thanks
   - Simple conversational requests
   - Requests that do not require enterprise evidence

2. FRAUD_ANALYTICS
   - Questions requiring structured fraud data
   - Counts
   - Aggregations
   - Transaction analysis
   - Alert analysis
   - Merchant analysis
   - Fraud metrics
   - Data-driven fraud analysis

3. ENTERPRISE_KNOWLEDGE
   - Internal policies
   - Procedures
   - Rules
   - Requirements
   - Enterprise knowledge documents
   - Fraud policy interpretation

4. EXTERNAL_RESEARCH
   - Current external information
   - Latest regulations
   - Recent regulatory updates
   - Explicit web/internet research

5. MULTI_DOMAIN
   - Requests requiring more than one evidence source
   - Examples:
     * Internal policy + structured fraud data
     * Internal policy + external regulations
     * Structured fraud data + external research
     * Policy + structured data + external research

6. UNKNOWN
   - Requests that cannot confidently be mapped to the supported domains

Available tools:

Fraud analytics:
- query_fraud_data
- get_fraud_metrics
- generate_visualization

Enterprise knowledge:
- search_documents
- search_policies
- get_document_metadata

External research:
- search_web

Routing principles:

- Policy questions require enterprise knowledge.
- Questions asking what actually happened in the data require structured
  fraud analytics.
- Questions asking whether observed data violates a policy usually require
  BOTH policy knowledge and structured fraud data.
- Requests comparing internal policy with current external regulations require
  BOTH enterprise knowledge and external research.
- Explicit current/latest web research requires external research.
- Do not select tools that are not required.
- Do not invent tools.
- Do not execute tools.
- Return ONLY valid JSON.

Required JSON structure:

{
  "domain": "DIRECT | FRAUD_ANALYTICS | ENTERPRISE_KNOWLEDGE | EXTERNAL_RESEARCH | MULTI_DOMAIN | UNKNOWN",
  "tools": ["tool_name"],
  "reason": "short explanation",
  "confidence": 0.0,
  "requires_policy": true,
  "requires_structured_data": true,
  "requires_external_research": false
}

Confidence must be between 0.0 and 1.0.
"""


class LLMSupervisor:
    """
    LLM-powered semantic Supervisor.

    Responsibilities:
    - Send the user query to the routing model.
    - Parse structured JSON.
    - Convert the result into an application-level SupervisorDecision.
    - Validate domain/tool consistency.
    - Fall back to deterministic routing if the LLM fails.

    The Supervisor does NOT execute any tools.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv(
            "GROQ_SUPERVISOR_MODEL",
            DEFAULT_MODEL,
        )

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        self.client = Groq(api_key=self.api_key)
        self.fallback_supervisor = Supervisor()

    def route(self, query: str) -> SupervisorDecision:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            return SupervisorDecision(
                domain=SupervisorDomain.UNKNOWN,
                reason="The request is empty.",
                confidence=1.0,
            )

        try:
            llm_decision = self._classify(query)

            return self._validate_and_convert(llm_decision)

        except (ValueError, TypeError, KeyError, IndexError):
            return self._fallback(query)

        except Exception:
            return self._fallback(query)

    def _fallback(self, query: str) -> SupervisorDecision:
        """
        Safely fall back to the deterministic Supervisor.

        The fallback Supervisor only produces a routing decision.
        It does not execute any tools.
        """
        decision = self.fallback_supervisor.route(query)

        return SupervisorDecision(
            domain=decision.domain,
            tools=decision.tools,
            reason=(
                "LLM Supervisor unavailable; "
                f"deterministic fallback used. {decision.reason}"
            ),
            confidence=min(decision.confidence, 0.80),
            requires_policy=decision.requires_policy,
            requires_structured_data=decision.requires_structured_data,
            requires_external_research=decision.requires_external_research,
        )

    def _classify(self, query: str) -> LLMRouteDecision:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": SUPERVISOR_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Supervisor model returned an empty response.")

        payload = self._parse_json(content)

        return self._parse_route_decision(payload)

    @staticmethod
    def _parse_json(content: str) -> dict:
        cleaned = content.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Supervisor model returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Supervisor model response must be a JSON object."
            )

        return payload

    @staticmethod
    def _parse_route_decision(payload: dict) -> LLMRouteDecision:
        required_fields = {
            "domain",
            "tools",
            "reason",
            "confidence",
            "requires_policy",
            "requires_structured_data",
            "requires_external_research",
        }

        missing_fields = required_fields - payload.keys()

        if missing_fields:
            raise ValueError(
                "Supervisor response is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

        try:
            domain = SupervisorDomain(payload["domain"])
        except ValueError as exc:
            raise ValueError(
                f"Invalid Supervisor domain: {payload['domain']}"
            ) from exc

        raw_tools = payload["tools"]

        if not isinstance(raw_tools, list):
            raise ValueError("Supervisor tools must be a list.")

        tools: List[SupervisorTool] = []

        for raw_tool in raw_tools:
            try:
                tools.append(SupervisorTool(raw_tool))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Supervisor tool: {raw_tool}"
                ) from exc

        reason = payload["reason"]

        if not isinstance(reason, str):
            raise ValueError("Supervisor reason must be a string.")

        confidence = payload["confidence"]

        if not isinstance(confidence, (int, float)):
            raise ValueError("Supervisor confidence must be numeric.")

        requires_policy = payload["requires_policy"]
        requires_structured_data = payload["requires_structured_data"]
        requires_external_research = payload["requires_external_research"]

        if not isinstance(requires_policy, bool):
            raise ValueError("requires_policy must be boolean.")

        if not isinstance(requires_structured_data, bool):
            raise ValueError(
                "requires_structured_data must be boolean."
            )

        if not isinstance(requires_external_research, bool):
            raise ValueError(
                "requires_external_research must be boolean."
            )

        return LLMRouteDecision(
            domain=domain,
            tools=tools,
            reason=reason,
            confidence=float(confidence),
            requires_policy=requires_policy,
            requires_structured_data=requires_structured_data,
            requires_external_research=requires_external_research,
        )

    @staticmethod
    def _validate_and_convert(
        decision: LLMRouteDecision,
    ) -> SupervisorDecision:
        if not 0.0 <= decision.confidence <= 1.0:
            raise ValueError(
                "Supervisor confidence must be between 0.0 and 1.0."
            )

        allowed_tools = {
            SupervisorDomain.DIRECT: frozenset(),
            SupervisorDomain.UNKNOWN: frozenset(),
            SupervisorDomain.FRAUD_ANALYTICS: FRAUD_ANALYTICS_TOOLS,
            SupervisorDomain.ENTERPRISE_KNOWLEDGE: ENTERPRISE_KNOWLEDGE_TOOLS,
            SupervisorDomain.EXTERNAL_RESEARCH: EXTERNAL_RESEARCH_TOOLS,
            SupervisorDomain.MULTI_DOMAIN: (
                FRAUD_ANALYTICS_TOOLS
                | ENTERPRISE_KNOWLEDGE_TOOLS
                | EXTERNAL_RESEARCH_TOOLS
            ),
        }

        requested_tools = frozenset(decision.tools)

        if not requested_tools.issubset(allowed_tools[decision.domain]):
            raise ValueError(
                "Supervisor selected tools that are not allowed for "
                f"domain {decision.domain.value}."
            )

        if (
            decision.domain == SupervisorDomain.FRAUD_ANALYTICS
            and not decision.requires_structured_data
        ):
            raise ValueError(
                "FRAUD_ANALYTICS requires structured data."
            )

        # if (
        #     decision.domain == SupervisorDomain.ENTERPRISE_KNOWLEDGE
        #     and not decision.requires_policy
        # ):
        #     raise ValueError(
        #         "ENTERPRISE_KNOWLEDGE requires policy or knowledge evidence."
        #     )

        if (
            decision.domain == SupervisorDomain.EXTERNAL_RESEARCH
            and not decision.requires_external_research
        ):
            raise ValueError(
                "EXTERNAL_RESEARCH requires external research."
            )

        if decision.domain == SupervisorDomain.MULTI_DOMAIN:
            evidence_required = (
                decision.requires_policy
                or decision.requires_structured_data
                or decision.requires_external_research
            )

            if not evidence_required:
                raise ValueError(
                    "MULTI_DOMAIN requires at least one evidence requirement."
                )

        application_decision = SupervisorDecision(
            domain=decision.domain,
            tools=requested_tools,
            reason=decision.reason,
            confidence=decision.confidence,
            requires_policy=decision.requires_policy,
            requires_structured_data=decision.requires_structured_data,
            requires_external_research=decision.requires_external_research,
        )

        if not application_decision.is_valid():
            raise ValueError(
                "Supervisor decision failed application validation."
            )

        return application_decision