"""
LangGraph security and routing graph.

Current graph:

    START
      ↓
    Input Guardrails
      ↓
    AI Safety
      ↓
    Authentication
      ↓
    Authorization
      ↓
    Supervisor
      ↓
    Routing
      ├── Fraud Agent
      │      ↓
      │   Fraud MCP Client
      │      ↓
      │   Fraud MCP Server
      │      ↓
      │   Governed Fraud Tools
      │      ↓
      │   Databricks Gold
      │
      └── Company Knowledge RAG
             ↓
          Databricks Vector Search
             ↓
          Reranking
             ↓
          Corrective RAG
             ↓
          Self-RAG
             ↓
          Evidence
      ↓
    Response Generator
      ↓
    Output Guardrails
      ↓
    Safe Response

The graph stops immediately when a security boundary fails.

External research, policy-specific RAG, multi-domain orchestration,
direct conversational responses, and unknown routes remain unimplemented
at this stage.
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from backend.core.llm import llm

from .fraud_agent import build_fraud_agent
from .graph_state import AgentState
from .llm_supervisor import LLMSupervisor
from .supervisor import SupervisorDomain

from ..authorization.policy import (
    AuthorizationPolicy,
    Resource,
    UserIdentity,
)

from ..guardrails.input_guardrails import (
    validate_user_query,
)

from ..guardrails.safety_classifier import (
    ModelSafetyClassifier,
)

from ..guardrails.output_guardrails import (
    build_output_guardrails,
)

from ..response.response_generator import (
    build_response_generator,
)

from ..rag.company_rag_adapter import (
    build_company_knowledge_node,
)

from ..rag.policy_rag_adapter import build_policy_rag_node


logger = logging.getLogger(__name__)


# ============================================================
# LOCAL DEVELOPMENT USERS
# ============================================================

TEST_USERS = {
    "user-001": UserIdentity(
        user_id="user-001",
        roles={"fraud_investigator"},
        departments={"fraud"},
    ),
    "user-002": UserIdentity(
        user_id="user-002",
        roles={"fraud_manager"},
        departments={"fraud"},
    ),
    "user-003": UserIdentity(
        user_id="user-003",
        roles={"employee"},
        departments={"finance"},
    ),
    "admin-001": UserIdentity(
        user_id="admin-001",
        roles={"admin"},
        departments={"fraud"},
    ),
}


# ============================================================
# GOVERNED RESOURCE
# ============================================================

FRAUD_KNOWLEDGE_RESOURCE = Resource(
    resource_id="fraud-knowledge-base",
    resource_type="knowledge_base",
    classification="confidential",
    departments={"fraud"},
)


# ============================================================
# SECURITY SERVICES
# ============================================================

authorization_policy = AuthorizationPolicy()


try:
    safety_classifier = ModelSafetyClassifier()

    logger.info(
        "LangGraph AI Safety classifier initialized."
    )

except Exception as exc:
    logger.error(
        "Failed to initialize LangGraph AI Safety classifier: %s",
        exc,
    )

    safety_classifier = None


try:
    llm_supervisor = LLMSupervisor()

    logger.info(
        "LangGraph LLM Supervisor initialized."
    )

except Exception as exc:
    logger.error(
        "Failed to initialize LangGraph LLM Supervisor: %s",
        exc,
    )

    llm_supervisor = None


# ============================================================
# FRAUD AGENT
# ============================================================

# Build the Fraud Agent once for this backend process.
#
# The Fraud Agent itself owns:
#
#     Fraud Scope
#          ↓
#     Fraud Tool Selection
#          ↓
#     Fraud MCP Client
#
# The MCP client then communicates with the MCP server.
#
# This keeps the security graph responsible for security/routing
# while the Fraud Agent remains responsible for fraud execution.

fraud_agent = build_fraud_agent()


# ============================================================
# COMPANY KNOWLEDGE RAG
# ============================================================

# Build the Company Knowledge RAG node through the adapter.
#
# The adapter translates:
#
#     Enterprise AgentState
#             ↓
#     CompanyKnowledgeState
#
# and then executes the complete Company Knowledge RAG:
#
#     Retrieve
#       ↓
#     Rerank
#       ↓
#     Evidence Grading
#       ↓
#     Corrective RAG
#       ↓
#     Answer Generation
#       ↓
#     Self-RAG
#       ↓
#     Self-Correction
#
# The adapter converts the final RAG state back into AgentState.

company_knowledge_node = build_company_knowledge_node(
    llm=llm,
)

policy_rag_node = build_policy_rag_node(
    llm=llm,
)

# ============================================================
# FRAUD AGENT NODE
# ============================================================

async def fraud_agent_node(
    state: AgentState,
) -> AgentState:
    """
    Execute the governed LangGraph Fraud Agent.

    The Fraud Agent communicates with Databricks only through:

        Fraud Agent
            ↓
        Fraud MCP Client
            ↓
        Fraud MCP Server
            ↓
        Fraud Tools
            ↓
        FraudDataAccess
            ↓
        Approved Databricks Gold tables

    No direct Databricks access occurs here.
    """

    try:
        result = await fraud_agent.ainvoke(state)

        return {
            **state,
            **result,
            "current_stage": "fraud_agent",
        }

    except Exception as exc:
        logger.exception(
            "Fraud Agent execution failed."
        )

        return {
            **state,
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Fraud Agent execution failed."
            ),
            "allowed": False,
            "current_stage": "fraud_agent",
            "error": type(exc).__name__,
        }


# ============================================================
# RESPONSE SERVICES
# ============================================================

# Build these once for the backend process.
#
# Response Generator:
#
#     Evidence → Structured Response
#
# Output Guardrails:
#
#     Structured Response → Safe Response
#
# Neither component accesses Databricks directly.

response_generator = build_response_generator()

output_guardrails = build_output_guardrails()


# ============================================================
# NODE: RESPONSE GENERATOR
# ============================================================

def response_generator_node(
    state: AgentState,
) -> AgentState:
    """
    Generate a structured response from approved evidence.

    The Response Generator does not access Databricks or MCP.
    It only consumes evidence already produced by an agent/RAG path.
    """

    try:
        result = response_generator.generate(
            query=state.get("query", ""),
            evidence=state.get("evidence", []),
        )

        return {
            **state,
            "generated_response": result,
            "current_stage": "response_generator",
        }

    except Exception as exc:
        logger.exception(
            "Response Generator execution failed."
        )

        return {
            **state,
            "generated_response": {},
            "allowed": False,
            "current_stage": "response_generator",
            "error": type(exc).__name__,
        }


# ============================================================
# NODE: OUTPUT GUARDRAILS
# ============================================================

def output_guardrails_node(
    state: AgentState,
) -> AgentState:
    """
    Validate the generated response before it reaches the user.

    No generated response is allowed to reach the final user-facing
    layer unless it passes these guardrails.
    """

    try:
        generated_response = state.get(
            "generated_response",
            {},
        )

        result = output_guardrails.validate(
            generated_response,
        )

        return {
            **state,
            "output_guardrails_allowed": result["allowed"],
            "output_guardrails_reason": result["reason"],
            "safe_response": result["safe_response"],
            "allowed": result["allowed"],
            "current_stage": "output_guardrails",
        }

    except Exception as exc:
        logger.exception(
            "Output Guardrails execution failed."
        )

        return {
            **state,
            "output_guardrails_allowed": False,
            "output_guardrails_reason": (
                "Output Guardrails execution failed."
            ),
            "safe_response": "",
            "allowed": False,
            "current_stage": "output_guardrails",
            "error": type(exc).__name__,
        }


# ============================================================
# NODE: INPUT GUARDRAIL
# ============================================================

def input_guardrail_node(
    state: AgentState,
) -> AgentState:

    query = state["query"]

    try:
        result = validate_user_query(query)

    except Exception as exc:
        logger.exception(
            "Input guardrail failed."
        )

        return {
            **state,
            "input_guardrail_allowed": False,
            "input_guardrail_reason": (
                "Input validation could not be completed."
            ),
            "allowed": False,
            "current_stage": "input_guardrails",
            "error": type(exc).__name__,
        }

    if not result.allowed:
        logger.warning(
            "Input guardrail blocked request."
        )

        return {
            **state,
            "input_guardrail_allowed": False,
            "input_guardrail_reason": result.reason,
            "allowed": False,
            "current_stage": "input_guardrails",
        }

    return {
        **state,
        "input_guardrail_allowed": True,
        "input_guardrail_reason": "",
        "current_stage": "input_guardrails",
        "allowed": True,
    }


# ============================================================
# NODE: AI SAFETY
# ============================================================

def ai_safety_node(
    state: AgentState,
) -> AgentState:

    if safety_classifier is None:

        return {
            **state,
            "safety_allowed": False,
            "safety_category": "unavailable",
            "safety_risk_level": "unknown",
            "safety_reason": (
                "AI safety service is temporarily unavailable."
            ),
            "allowed": False,
            "current_stage": "model_safety",
        }

    try:
        result = safety_classifier.classify(
            state["query"]
        )

    except Exception as exc:
        logger.exception(
            "AI Safety classification failed."
        )

        return {
            **state,
            "safety_allowed": False,
            "safety_category": "error",
            "safety_risk_level": "unknown",
            "safety_reason": (
                "AI safety validation could not be completed."
            ),
            "allowed": False,
            "current_stage": "model_safety",
            "error": type(exc).__name__,
        }

    if not result.allowed:

        return {
            **state,
            "safety_allowed": False,
            "safety_category": result.category,
            "safety_risk_level": result.risk_level,
            "safety_reason": (
                "Request blocked by AI safety policy."
            ),
            "allowed": False,
            "current_stage": "model_safety",
        }

    return {
        **state,
        "safety_allowed": True,
        "safety_category": result.category,
        "safety_risk_level": result.risk_level,
        "safety_reason": result.reason,
        "allowed": True,
        "current_stage": "model_safety",
    }


# ============================================================
# NODE: AUTHENTICATION
# ============================================================

def authentication_node(
    state: AgentState,
) -> AgentState:

    user = TEST_USERS.get(
        state["user_id"]
    )

    if user is None:

        return {
            **state,
            "authenticated": False,
            "user": None,
            "allowed": False,
            "current_stage": "authentication",
            "error": "Authentication failed.",
        }

    return {
        **state,
        "authenticated": True,
        "user": user,
        "allowed": True,
        "current_stage": "authentication",
    }


# ============================================================
# NODE: AUTHORIZATION
# ============================================================

def authorization_node(
    state: AgentState,
) -> AgentState:

    user = state.get("user")

    if user is None:

        return {
            **state,
            "authorization_allowed": False,
            "authorization_reason": (
                "Authorization requires authentication."
            ),
            "allowed": False,
            "current_stage": "authorization",
        }

    try:

        result = authorization_policy.authorize(
            user,
            FRAUD_KNOWLEDGE_RESOURCE,
        )

    except Exception as exc:

        logger.exception(
            "Authorization failed."
        )

        return {
            **state,
            "authorization_allowed": False,
            "authorization_reason": (
                "Authorization could not be completed."
            ),
            "allowed": False,
            "current_stage": "authorization",
            "error": type(exc).__name__,
        }

    if not result.allowed:

        return {
            **state,
            "authorization_allowed": False,
            "authorization_reason": result.reason,
            "allowed": False,
            "current_stage": "authorization",
        }

    return {
        **state,
        "authorization_allowed": True,
        "authorization_reason": "",
        "allowed": True,
        "current_stage": "authorization",
    }


# ============================================================
# NODE: SUPERVISOR
# ============================================================

def supervisor_node(
    state: AgentState,
) -> AgentState:

    if llm_supervisor is None:

        return {
            **state,
            "allowed": False,
            "current_stage": "supervisor",
            "error": (
                "Supervisor service is temporarily unavailable."
            ),
        }

    try:

        decision = llm_supervisor.route(
            state["query"]
        )

    except Exception as exc:

        logger.exception(
            "Supervisor routing failed."
        )

        return {
            **state,
            "allowed": False,
            "current_stage": "supervisor",
            "error": (
                "Request routing could not be completed."
            ),
        }

    return {
        **state,
        "supervisor_decision": decision,
        "supervisor_domain": decision.domain.value,
        "supervisor_tools": [
            tool.value
            for tool in decision.tools
        ],
        "supervisor_reason": decision.reason,
        "supervisor_confidence": decision.confidence,
        "requires_policy": decision.requires_policy,
        "requires_structured_data": (
            decision.requires_structured_data
        ),
        "requires_external_research": (
            decision.requires_external_research
        ),
        "allowed": True,
        "current_stage": "supervisor",
    }


# ============================================================
# CONDITIONAL SECURITY ROUTING
# ============================================================

def security_route(
    state: AgentState,
) -> Literal[
    "ai_safety",
    "authentication",
    "authorization",
    "supervisor",
    "blocked",
]:

    stage = state.get("current_stage")

    if not state.get(
        "input_guardrail_allowed",
        False,
    ):
        return "blocked"

    if stage == "input_guardrails":
        return "ai_safety"

    if not state.get(
        "safety_allowed",
        False,
    ):
        return "blocked"

    if stage == "model_safety":
        return "authentication"

    if not state.get(
        "authenticated",
        False,
    ):
        return "blocked"

    if stage == "authentication":
        return "authorization"

    if not state.get(
        "authorization_allowed",
        False,
    ):
        return "blocked"

    if stage == "authorization":
        return "supervisor"

    if stage == "supervisor":
        return "supervisor"

    return "blocked"


# ============================================================
# SUPERVISOR ROUTING
# ============================================================

def supervisor_route(
    state: AgentState,
) -> Literal[
    "fraud",
    "knowledge",
    "policy",
    "external",
    "multi_domain",
    "direct",
    "unknown",
]:

    domain = state.get(
        "supervisor_domain"
    )

    if domain == SupervisorDomain.FRAUD_ANALYTICS.value:
        return "fraud"

    if domain == SupervisorDomain.ENTERPRISE_KNOWLEDGE.value:
        if state.get("requires_policy", False):
            return "policy"
        return "knowledge"

    if domain == SupervisorDomain.EXTERNAL_RESEARCH.value:
        return "external"

    if domain == SupervisorDomain.MULTI_DOMAIN.value:
        return "multi_domain"

    if domain == SupervisorDomain.DIRECT.value:
        return "direct"

    return "unknown"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_security_graph():

    builder = StateGraph(AgentState)

    # --------------------------------------------------------
    # Security nodes
    # --------------------------------------------------------

    builder.add_node(
        "input_guardrails",
        input_guardrail_node,
    )

    builder.add_node(
        "ai_safety",
        ai_safety_node,
    )

    builder.add_node(
        "authentication",
        authentication_node,
    )

    builder.add_node(
        "authorization",
        authorization_node,
    )

    builder.add_node(
        "supervisor",
        supervisor_node,
    )

    # --------------------------------------------------------
    # Fraud Agent
    # --------------------------------------------------------

    builder.add_node(
        "fraud_agent",
        fraud_agent_node,
    )

    # --------------------------------------------------------
    # Company Knowledge RAG
    # --------------------------------------------------------

    builder.add_node(
        "company_knowledge",
        company_knowledge_node,
    )
    
    builder.add_node(
    "policy_rag",
    policy_rag_node,
)

    # --------------------------------------------------------
    # Response Generator
    # --------------------------------------------------------

    builder.add_node(
        "response_generator",
        response_generator_node,
    )

    # --------------------------------------------------------
    # Output Guardrails
    # --------------------------------------------------------

    builder.add_node(
        "output_guardrails",
        output_guardrails_node,
    )

    # ========================================================
    # START
    # ========================================================

    builder.add_edge(
        START,
        "input_guardrails",
    )

    # ========================================================
    # SECURITY FLOW
    # ========================================================

    builder.add_conditional_edges(
        "input_guardrails",
        security_route,
        {
            "ai_safety": "ai_safety",
            "blocked": END,
        },
    )

    builder.add_conditional_edges(
        "ai_safety",
        security_route,
        {
            "authentication": "authentication",
            "blocked": END,
        },
    )

    builder.add_conditional_edges(
        "authentication",
        security_route,
        {
            "authorization": "authorization",
            "blocked": END,
        },
    )

    builder.add_conditional_edges(
        "authorization",
        security_route,
        {
            "supervisor": "supervisor",
            "blocked": END,
        },
    )

    # ========================================================
    # SUPERVISOR ROUTING
    # ========================================================

    builder.add_conditional_edges(
    "supervisor",
    supervisor_route,
    {
        "fraud": "fraud_agent",
        "knowledge": "company_knowledge",
        "policy": "policy_rag",
        "external": END,
        "multi_domain": END,
        "direct": END,
        "unknown": END,
    },
)

    # ========================================================
    # AGENT/RAG → RESPONSE GENERATOR
    # ========================================================

    builder.add_edge(
        "fraud_agent",
        "response_generator",
    )

    builder.add_edge(
        "company_knowledge",
        "response_generator",
    )
    
    builder.add_edge(
    "policy_rag",
    "response_generator",
)

    # ========================================================
    # RESPONSE GENERATOR → OUTPUT GUARDRAILS
    # ========================================================

    builder.add_edge(
        "response_generator",
        "output_guardrails",
    )

    # ========================================================
    # OUTPUT GUARDRAILS → END
    # ========================================================

    builder.add_edge(
        "output_guardrails",
        END,
    )

    return builder.compile()