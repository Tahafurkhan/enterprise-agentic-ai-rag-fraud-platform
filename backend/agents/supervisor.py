from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


class SupervisorDomain(str, Enum):
    """Supported high-level routing domains."""

    DIRECT = "DIRECT"
    FRAUD_ANALYTICS = "FRAUD_ANALYTICS"
    ENTERPRISE_KNOWLEDGE = "ENTERPRISE_KNOWLEDGE"
    EXTERNAL_RESEARCH = "EXTERNAL_RESEARCH"
    MULTI_DOMAIN = "MULTI_DOMAIN"
    UNKNOWN = "UNKNOWN"


class SupervisorTool(str, Enum):
    """Tools that the Supervisor may request."""

    QUERY_FRAUD_DATA = "query_fraud_data"
    GET_FRAUD_METRICS = "get_fraud_metrics"
    SEARCH_DOCUMENTS = "search_documents"
    SEARCH_POLICIES = "search_policies"
    GET_DOCUMENT_METADATA = "get_document_metadata"
    SEARCH_WEB = "search_web"
    GENERATE_VISUALIZATION = "generate_visualization"


@dataclass(frozen=True)
class SupervisorDecision:
    """Structured routing decision produced by the Supervisor."""

    domain: SupervisorDomain
    tools: FrozenSet[SupervisorTool] = field(default_factory=frozenset)
    reason: str = ""
    confidence: float = 0.0
    requires_policy: bool = False
    requires_structured_data: bool = False
    requires_external_research: bool = False

    def is_valid(self) -> bool:
        """Validate the Supervisor decision."""

        if not self.reason.strip():
            return False

        if not 0.0 <= self.confidence <= 1.0:
            return False

        if self.domain == SupervisorDomain.FRAUD_ANALYTICS:
            if not self.requires_structured_data:
                return False

        if self.domain == SupervisorDomain.ENTERPRISE_KNOWLEDGE:
            if self.requires_structured_data:
                return False

        if self.domain == SupervisorDomain.EXTERNAL_RESEARCH:
            if not self.requires_external_research:
                return False

        if self.domain == SupervisorDomain.MULTI_DOMAIN:
            if not (
                self.requires_policy
                or self.requires_structured_data
                or self.requires_external_research
            ):
                return False

        return True


FRAUD_ANALYTICS_TOOLS = frozenset(
    {
        SupervisorTool.QUERY_FRAUD_DATA,
        SupervisorTool.GET_FRAUD_METRICS,
        SupervisorTool.GENERATE_VISUALIZATION,
    }
)

ENTERPRISE_KNOWLEDGE_TOOLS = frozenset(
    {
        SupervisorTool.SEARCH_DOCUMENTS,
        SupervisorTool.SEARCH_POLICIES,
        SupervisorTool.GET_DOCUMENT_METADATA,
    }
)

EXTERNAL_RESEARCH_TOOLS = frozenset(
    {
        SupervisorTool.SEARCH_WEB,
    }
)


class Supervisor:
    """
    Domain Supervisor for the Enterprise Fraud Intelligence Platform.

    The Supervisor decides which domain and evidence sources are required.
    It does not execute tools, query Databricks, search AI Search, or call
    external services.
    """

    def route(self, query: str) -> SupervisorDecision:
        """
        Return a deterministic development routing decision.

        This method is intentionally conservative. LLM-based semantic
        classification will be introduced after the routing contract and
        tests are established.
        """

        normalized_query = query.strip().lower()

        if not normalized_query:
            return SupervisorDecision(
                domain=SupervisorDomain.UNKNOWN,
                reason="The request is empty.",
                confidence=1.0,
            )

        if self._is_direct_query(normalized_query):
            return SupervisorDecision(
                domain=SupervisorDomain.DIRECT,
                reason=(
                    "The request is conversational and does not require "
                    "enterprise evidence."
                ),
                confidence=0.95,
            )

        requires_policy = self._requires_policy(normalized_query)
        requires_structured_data = self._requires_structured_data(
            normalized_query
        )
        requires_external = self._requires_external_research(
            normalized_query
        )

        if requires_external and (requires_policy or requires_structured_data):
            return SupervisorDecision(
                domain=SupervisorDomain.MULTI_DOMAIN,
                tools=EXTERNAL_RESEARCH_TOOLS
                | (
                    ENTERPRISE_KNOWLEDGE_TOOLS
                    if requires_policy
                    else frozenset()
                )
                | (
                    FRAUD_ANALYTICS_TOOLS
                    if requires_structured_data
                    else frozenset()
                ),
                reason=(
                    "The request requires external research together with "
                    "internal enterprise evidence."
                ),
                confidence=0.85,
                requires_policy=requires_policy,
                requires_structured_data=requires_structured_data,
                requires_external_research=True,
            )

        if requires_external:
            return SupervisorDecision(
                domain=SupervisorDomain.EXTERNAL_RESEARCH,
                tools=EXTERNAL_RESEARCH_TOOLS,
                reason="The request requires external research.",
                confidence=0.90,
                requires_external_research=True,
            )

        if requires_policy and requires_structured_data:
            return SupervisorDecision(
                domain=SupervisorDomain.MULTI_DOMAIN,
                tools=ENTERPRISE_KNOWLEDGE_TOOLS | FRAUD_ANALYTICS_TOOLS,
                reason=(
                    "The request requires both fraud policy knowledge and "
                    "structured fraud data."
                ),
                confidence=0.95,
                requires_policy=True,
                requires_structured_data=True,
            )

        if requires_structured_data:
            return SupervisorDecision(
                domain=SupervisorDomain.FRAUD_ANALYTICS,
                tools=FRAUD_ANALYTICS_TOOLS,
                reason="The request requires structured fraud analytics.",
                confidence=0.90,
                requires_structured_data=True,
            )

        if requires_policy:
            return SupervisorDecision(
                domain=SupervisorDomain.ENTERPRISE_KNOWLEDGE,
                tools=ENTERPRISE_KNOWLEDGE_TOOLS,
                reason=(
                    "The request requires enterprise policy or knowledge "
                    "retrieval."
                ),
                confidence=0.90,
                requires_policy=True,
            )

        return SupervisorDecision(
            domain=SupervisorDomain.UNKNOWN,
            reason=(
                "The request could not be confidently mapped to a supported "
                "enterprise domain."
            ),
            confidence=0.55,
        )

    @staticmethod
    def _is_direct_query(query: str) -> bool:
        direct_phrases = {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
        }

        return query in direct_phrases

    @staticmethod
    def _requires_policy(query: str) -> bool:
        policy_terms = {
            "policy",
            "policies",
            "threshold",
            "rule",
            "rules",
            "requirement",
            "requirements",
            "procedure",
            "procedures",
            "escalation",
            "monitoring requirement",
            "high-value transaction policy",
            "fraud policy",
            "watchlist policy",
        }

        return any(term in query for term in policy_terms)

    @staticmethod
    def _requires_structured_data(query: str) -> bool:
        structured_terms = {
            "how many",
            "count",
            "number of",
            "total number",
            "transaction count",
            "alert count",
            "merchant count",
            "watchlist records",
            "fraud records",
            "risk distribution",
            "amount",
            "volume",
            "occurred",
            "detected",
            "violated",
            "violation",
            "on august",
            "on september",
        }

        return any(term in query for term in structured_terms)

    @staticmethod
    def _requires_external_research(query: str) -> bool:
        external_phrases = (
            "search the web",
            "web search",
            "search the internet",
            "internet search",
            "latest regulation",
            "latest regulations",
            "latest fraud regulation",
            "latest fraud regulations",
            "current regulation",
            "current regulations",
            "current fraud regulation",
            "current fraud regulations",
            "external regulation",
            "external regulations",
            "external research",
            "recent regulatory update",
            "recent regulations",
            "recent fraud regulations",
        )

        return any(phrase in query for phrase in external_phrases)