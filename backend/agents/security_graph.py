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
      │   Agent Harness
      │      ↓
      │   Fraud MCP Client
      │      ↓
      │   Fraud MCP Server
      │      ↓
      │   Governed Fraud Tools
      │      ↓
      │   Databricks Gold
      │
      ├── Company Knowledge RAG
      │      ↓
      │   Agent Harness
      │      ↓
      │   Company RAG
      │      ├── Databricks Vector Search
      │      └── Neo4j Knowledge Graph
      │
      ├── Policy RAG
      │      ↓
      │   Agent Harness
      │      ↓
      │   Policy RAG
      │      ↓
      │   Databricks Vector Search
      │
      └── External Research
             ↓
          Agent Harness
             ↓
       External Research Agent
             ↓
        Tavily MCP Client
             ↓
        Tavily MCP Server
             ↓
             Tavily
             ↓
          Web Results

      ↓
    Shared Evidence Layer
      ↓
    Response Generator
      ↓
    Output Guardrails
      ↓
    Safe Response

Multi-domain orchestration can combine:

    Fraud Agent
    Company Knowledge RAG
    Policy RAG
    External Research

All routed domain executions remain behind the shared
Agent Harness.

The graph stops immediately when a security boundary fails.
"""

from __future__ import annotations

import inspect
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from backend.core.llm import llm

from .agent_harness import (
    AgentExecutionResult,
    AgentHarness,
    AgentHarnessConfig,
)

from .fraud_agent import build_fraud_agent

from .graph_state import AgentState

from .llm_supervisor import LLMSupervisor

from .supervisor import SupervisorDomain

from .external_research_agent import (
    build_external_research_agent,
)

from ..mcp.tavily_mcp_client import (
    TavilyMCPClient,
)

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

from ..evidence.evidence_layer import (
    build_evidence_layer_node,
)

from ..rag.company_rag_adapter import (
    build_company_knowledge_node,
)

from ..rag.policy_rag_adapter import (
    build_policy_rag_node,
)


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
# AGENT HARNESS
# ============================================================

agent_harness = AgentHarness(
    AgentHarnessConfig(
        max_iterations=5,
        max_tool_calls=10,
        max_retries=2,
        timeout_seconds=60.0,
        token_budget=8000,
        min_confidence=0.80,
        require_evidence=True,
    )
)


# ============================================================
# FRAUD AGENT
# ============================================================

fraud_agent = build_fraud_agent()


# ============================================================
# COMPANY KNOWLEDGE RAG
# ============================================================

company_knowledge_node = build_company_knowledge_node(
    llm=llm,
)


# ============================================================
# POLICY RAG
# ============================================================

policy_rag_node = build_policy_rag_node(
    llm=llm,
)


# ============================================================
# EXTERNAL RESEARCH
# ============================================================

"""
The External Research Agent uses the official Tavily MCP server.

The flow is:

    Security Graph
        ↓
    External Research Agent
        ↓
    TavilyMCPClient
        ↓
    Tavily MCP Server
        ↓
    Tavily
        ↓
    Web
"""

try:
    tavily_mcp_client = TavilyMCPClient()

    external_research_agent = build_external_research_agent(
        tavily_mcp_client,
        max_results=5,
    )

    logger.info(
        "Tavily External Research Agent initialized."
    )

except Exception as exc:
    logger.error(
        "Failed to initialize Tavily External Research: %s",
        exc,
    )

    tavily_mcp_client = None
    external_research_agent = None


# ============================================================
# SHARED EVIDENCE LAYER
# ============================================================

evidence_layer_node = build_evidence_layer_node()


# ============================================================
# HARNESS HELPERS
# ============================================================

def _build_harness_execution_result(
    result: AgentState,
) -> AgentExecutionResult:
    """
    Convert a domain-agent AgentState into the common harness result.

    Company and Policy RAG expose groundedness_score directly.

    Fraud and External Research may not expose an explicit confidence
    score. Successful governed execution with evidence is therefore
    treated as confidence 1.0 until an explicit confidence metric exists.
    """

    if not isinstance(result, dict):
        raise TypeError(
            "Domain agent must return an AgentState dictionary."
        )

    evidence = result.get(
        "evidence",
        [],
    )

    explicit_confidence = result.get(
        "confidence"
    )

    if explicit_confidence is not None:

        confidence = float(
            explicit_confidence
        )

    else:

        groundedness_score = result.get(
            "groundedness_score"
        )

        if groundedness_score is not None:

            confidence = float(
                groundedness_score
            )

        elif (
            evidence
            and result.get(
                "allowed",
                True,
            )
        ):

            confidence = 1.0

        else:

            confidence = 0.0

    tool_results = result.get(
        "tool_results",
        [],
    )

    tokens_used = result.get(
        "tokens_used",
        0,
    )

    return AgentExecutionResult(
        response=(
            result.get("response")
            or result.get("answer")
        ),
        confidence=confidence,
        evidence_found=bool(
            evidence
        ),
        tool_calls=len(
            tool_results
        ),
        tokens_used=int(
            tokens_used or 0
        ),
        completed=bool(
            result.get(
                "allowed",
                True,
            )
        ),
        metadata={
            "agent_state": result,
        },
    )


async def _run_agent_through_harness(
    agent_callable,
    state: AgentState,
):
    """
    Execute a routed agent through the shared Agent Harness.
    """

    async def execute(
        execution_state: AgentState,
    ) -> AgentExecutionResult:

        result = agent_callable(
            execution_state
        )

        if inspect.isawaitable(
            result
        ):
            result = await result

        return _build_harness_execution_result(
            result
        )

    return await agent_harness.run(
        execute,
        state=state,
    )


def _restore_harness_agent_state(
    state: AgentState,
    harness_result,
    current_stage: str,
) -> AgentState:
    """
    Restore the underlying domain-agent state after harness execution.

    Preserve the domain agent's actual `allowed` value.

    This prevents a failed external search, failed RAG execution,
    or other domain-level denial from accidentally becoming:

        allowed = True
    """

    if not harness_result.success:

        failure_reason = (
            harness_result.error
            or harness_result.termination_reason
            or "Agent Harness execution failed."
        )

        return {
            **state,
            "allowed": False,
            "current_stage": current_stage,
            "error": failure_reason,
        }

    execution = harness_result.metadata.get(
        "last_execution"
    )

    if not isinstance(
        execution,
        AgentExecutionResult,
    ):

        return {
            **state,
            "allowed": False,
            "current_stage": current_stage,
            "error": (
                "Agent Harness returned no final execution result."
            ),
        }

    domain_state = execution.metadata.get(
        "agent_state"
    )

    if not isinstance(
        domain_state,
        dict,
    ):

        return {
            **state,
            "allowed": False,
            "current_stage": current_stage,
            "error": (
                "Agent Harness returned no domain agent state."
            ),
        }

    domain_allowed = bool(
        domain_state.get(
            "allowed",
            True,
        )
    )

    return {
        **state,
        **domain_state,
        "allowed": domain_allowed,
        "current_stage": current_stage,
    }


# ============================================================
# FRAUD AGENT NODE
# ============================================================

async def fraud_agent_node(
    state: AgentState,
) -> AgentState:
    """
    Execute the governed Fraud Agent through the Agent Harness.
    """

    try:

        harness_result = await _run_agent_through_harness(
            fraud_agent.ainvoke,
            state,
        )

        result = _restore_harness_agent_state(
            state=state,
            harness_result=harness_result,
            current_stage="fraud_agent",
        )

        return {
            **result,
            "agent_name": "fraud_agent",
            "fraud_execution_allowed": bool(
                result.get(
                    "allowed",
                    False,
                )
            ),
            "fraud_execution_reason": (
                "Approved fraud tool executed through MCP."
                if result.get(
                    "allowed",
                    False,
                )
                else result.get(
                    "error",
                    "Fraud execution failed.",
                )
            ),
        }

    except Exception as exc:

        logger.exception(
            "Fraud Agent execution through harness failed."
        )

        return {
            **state,
            "fraud_execution_allowed": False,
            "fraud_execution_reason": (
                "Fraud Agent execution failed."
            ),
            "fraud_execution_error": str(exc),
            "allowed": False,
            "current_stage": "fraud_agent",
            "error": type(exc).__name__,
        }


# ============================================================
# COMPANY KNOWLEDGE RAG NODE
# ============================================================

async def company_knowledge_harness_node(
    state: AgentState,
) -> AgentState:
    """
    Execute Company Knowledge RAG through the Agent Harness.
    """

    try:

        harness_result = await _run_agent_through_harness(
            company_knowledge_node,
            state,
        )

        return _restore_harness_agent_state(
            state=state,
            harness_result=harness_result,
            current_stage="company_knowledge",
        )

    except Exception as exc:

        logger.exception(
            "Company Knowledge RAG execution through harness failed."
        )

        return {
            **state,
            "allowed": False,
            "current_stage": "company_knowledge",
            "error": type(exc).__name__,
        }


# ============================================================
# POLICY RAG NODE
# ============================================================

async def policy_rag_harness_node(
    state: AgentState,
) -> AgentState:
    """
    Execute Policy RAG through the Agent Harness.
    """
    try:
        harness_result = await _run_agent_through_harness(
            policy_rag_node,
            state,
        )

        result = _restore_harness_agent_state(
            state=state,
            harness_result=harness_result,
            current_stage="policy_rag",
        )

        policy_allowed = bool(
            result.get("allowed", False)
        )

        return {
            **result,
            "policy_execution_allowed": policy_allowed,
            "policy_execution_reason": (
                "Policy RAG execution completed successfully."
                if policy_allowed
                else result.get(
                    "error",
                    "Policy RAG execution failed.",
                )
            ),
            "policy_execution_error": (
                ""
                if policy_allowed
                else result.get(
                    "error",
                    "",
                )
            ),
            "current_stage": "policy_rag",
        }

    except Exception as exc:
        logger.exception(
            "Policy RAG execution through harness failed."
        )

        return {
            **state,
            "policy_execution_allowed": False,
            "policy_execution_reason": (
                "Policy RAG execution failed."
            ),
            "policy_execution_error": str(exc),
            "allowed": False,
            "current_stage": "policy_rag",
            "error": type(exc).__name__,
        }

# ============================================================
# EXTERNAL RESEARCH NODE
# ============================================================

async def external_research_harness_node(
    state: AgentState,
) -> AgentState:
    """
    Execute External Research through the Agent Harness.

    Flow:

        Security Graph
            ↓
        Agent Harness
            ↓
        External Research Agent
            ↓
        Tavily MCP Client
            ↓
        Tavily MCP Server
            ↓
        Tavily
            ↓
        Web
    """

    if external_research_agent is None:

        logger.error(
            "External Research Agent is unavailable."
        )

        return {
            **state,
            "web_results": [],
            "evidence": [],
            "external_research_allowed": False,
            "external_research_execution_allowed": False,
            "external_research_reason": (
                "External Research Agent is unavailable."
            ),
            "allowed": False,
            "current_stage": "external_research",
            "error": (
                "External Research Agent is unavailable."
            ),
        }

    try:

        # ----------------------------------------------------
        # Build external-agent state
        # ----------------------------------------------------

        authorization_context = (
            state.get(
                "authorization_reason",
                "",
            )
            or "Authorization approved."
        )

        external_state = {
            "question": state.get(
                "query",
                "",
            ),

            "current_query": (
                state.get(
                    "current_query",
                    "",
                )
                or state.get(
                    "query",
                    "",
                )
            ),

            "authorization_context": (
                authorization_context
            ),

            "search_web_allowed": bool(
                state.get(
                    "authorization_allowed",
                    False,
                )
            ),

            "max_results": 5,
        }

        # ----------------------------------------------------
        # Adapter callable
        # ----------------------------------------------------

        async def invoke_external_agent(
            _: AgentState,
        ) -> AgentState:

            result = await external_research_agent.ainvoke(
                external_state
            )

            if not isinstance(
                result,
                dict,
            ):
                raise TypeError(
                    "External Research Agent returned "
                    "an invalid state."
                )

            external_allowed = bool(
                result.get(
                    "allowed",
                    result.get(
                        "execution_allowed",
                        False,
                    ),
                )
            )

            search_results = result.get(
                "search_results",
                [],
            )

            evidence = result.get(
                "evidence",
                [],
            )

            return {
    **state,

    "web_results": search_results,

    "evidence": evidence,

    "allowed": external_allowed,

    "execution_allowed": result.get(
        "execution_allowed",
        False,
    ),

    "execution_reason": result.get(
        "execution_reason",
        "",
    ),

    "external_research_allowed": external_allowed,

    "external_research_execution_allowed": bool(
        result.get(
            "execution_allowed",
            False,
        )
    ),

    "external_research_reason": result.get(
        "execution_reason",
        "",
    ),

    "external_research_error": result.get(
        "error",
        "",
    ),

    "current_query": result.get(
        "current_query",
        external_state["current_query"],
    ),

    "current_stage": "external_research",
}

        # ----------------------------------------------------
        # Execute through common harness
        # ----------------------------------------------------

        harness_result = await _run_agent_through_harness(
            invoke_external_agent,
            state,
        )

        result = _restore_harness_agent_state(
            state=state,
            harness_result=harness_result,
            current_stage="external_research",
        )

        external_allowed = bool(
            result.get(
                "allowed",
                False,
            )
        )

        external_reason = (
            result.get(
                "execution_reason",
                "",
            )
            or result.get(
                "external_research_error",
                "",
            )
            or result.get(
                "error",
                "",
            )
        )

        return {
            **result,

            # Canonical external-research fields.
            "external_research_allowed": (
                external_allowed
            ),

            "external_research_execution_allowed": (
                external_allowed
            ),

            "external_research_reason": (
                external_reason
            ),

            "web_results": result.get(
                "web_results",
                [],
            ),

            "evidence": result.get(
                "evidence",
                [],
            ),

            "current_stage": (
                "external_research"
            ),
        }

    except Exception as exc:

        logger.exception(
            "External Research execution through harness failed."
        )

        return {
            **state,
            "web_results": [],
            "evidence": [],
            "external_research_allowed": False,
            "external_research_execution_allowed": False,
            "external_research_reason": (
                "External Research execution failed."
            ),
            "external_research_error": str(exc),
            "allowed": False,
            "current_stage": "external_research",
            "error": type(exc).__name__,
        }


# ============================================================
# MULTI-DOMAIN ORCHESTRATION
# ============================================================

async def multi_domain_node(
    state: AgentState,
) -> AgentState:
    """
    Execute the evidence sources requested by the Supervisor.

    Supported domains:

        Fraud Agent
        Company Knowledge RAG
        Policy RAG
        External Research

    All results converge into one shared evidence collection.
    """

    current_state = dict(
        state
    )

    merged_evidence = list(
        state.get(
            "evidence",
            [],
        )
        or []
    )

    supervisor_tools = {
        str(tool).lower()
        for tool in state.get(
            "supervisor_tools",
            [],
        )
    }

    requires_structured_data = bool(
        state.get(
            "requires_structured_data",
            False,
        )
    )

    requires_policy = bool(
        state.get(
            "requires_policy",
            False,
        )
    )

    requires_external_research = bool(
        state.get(
            "requires_external_research",
            False,
        )
    )

    requires_company_knowledge = (
        "search_documents" in supervisor_tools
        or "get_document_metadata" in supervisor_tools
    )

    try:

        # ====================================================
        # FRAUD ANALYTICS
        # ====================================================

        if requires_structured_data:

            fraud_result = await fraud_agent_node(
                current_state
            )

            if not fraud_result.get(
                "allowed",
                False,
            ):

                return {
                    **current_state,
                    **fraud_result,
                    "allowed": False,
                    "current_stage": "multi_domain",
                    "error": (
                        "Fraud Agent failed during "
                        "multi-domain execution."
                    ),
                }

            merged_evidence.extend(
                fraud_result.get(
                    "evidence",
                    [],
                )
                or []
            )

            current_state = {
                **current_state,
                **fraud_result,
                "evidence": merged_evidence,
            }

        # ====================================================
        # POLICY RAG
        # ====================================================

        if requires_policy:

            policy_result = await policy_rag_harness_node(
                current_state
            )
            
            print(
    "DEBUG POLICY RESULT:",
    policy_result,
)
            if not policy_result.get(
                "allowed",
                False,
            ):

                return {
                    **current_state,
                    **policy_result,
                    "allowed": False,
                    "current_stage": "multi_domain",
                    "evidence": merged_evidence,
                    "error": (
                        "Policy RAG failed during "
                        "multi-domain execution."
                    ),
                }

            merged_evidence.extend(
                policy_result.get(
                    "evidence",
                    [],
                )
                or []
            )

            current_state = {
                **current_state,
                **policy_result,
                "evidence": merged_evidence,
            }

        # ====================================================
        # COMPANY KNOWLEDGE
        # ====================================================

        if requires_company_knowledge:

            company_result = (
                await company_knowledge_harness_node(
                    current_state
                )
            )

            if not company_result.get(
                "allowed",
                False,
            ):

                return {
                    **current_state,
                    **company_result,
                    "allowed": False,
                    "current_stage": "multi_domain",
                    "evidence": merged_evidence,
                    "error": (
                        "Company Knowledge RAG failed during "
                        "multi-domain execution."
                    ),
                }

            merged_evidence.extend(
                company_result.get(
                    "evidence",
                    [],
                )
                or []
            )

            current_state = {
                **current_state,
                **company_result,
                "evidence": merged_evidence,
            }

        # ====================================================
        # EXTERNAL RESEARCH
        # ====================================================

        if requires_external_research:

            external_result = (
                await external_research_harness_node(
                    current_state
                )
            )
            
            print(
    "DEBUG EXTERNAL RESULT:",
    external_result,
)

            if not external_result.get(
                "allowed",
                False,
            ):

                return {
                    **current_state,
                    **external_result,
                    "allowed": False,
                    "current_stage": "multi_domain",
                    "evidence": merged_evidence,
                    "error": (
                        "External Research failed during "
                        "multi-domain execution."
                    ),
                }

            # -----------------------------------------------
            # Preserve external state
            # -----------------------------------------------

            merged_evidence.extend(
                external_result.get(
                    "evidence",
                    [],
                )
                or []
            )

            current_state = {
                **current_state,
                **external_result,

                # Explicitly preserve the merged evidence.
                "evidence": merged_evidence,

                # Explicitly preserve external state.
                "external_research_allowed": bool(
                    external_result.get(
                        "external_research_allowed",
                        external_result.get(
                            "allowed",
                            False,
                        ),
                    )
                ),

                "external_research_execution_allowed": bool(
                    external_result.get(
                        "external_research_execution_allowed",
                        external_result.get(
                            "allowed",
                            False,
                        ),
                    )
                ),

                "external_research_reason": (
                    external_result.get(
                        "external_research_reason",
                        "",
                    )
                    or external_result.get(
                        "execution_reason",
                        "",
                    )
                ),

                "web_results": external_result.get(
                    "web_results",
                    [],
                ),
            }

        # ====================================================
        # VALIDATE EXECUTION
        # ====================================================

        if not (
            requires_structured_data
            or requires_policy
            or requires_company_knowledge
            or requires_external_research
        ):

            return {
                **current_state,
                "allowed": False,
                "current_stage": "multi_domain",
                "error": (
                    "Multi-domain request declared no "
                    "supported evidence sources."
                ),
            }

        # ====================================================
        # DEDUPLICATE EVIDENCE
        # ====================================================

        deduplicated_evidence = []

        seen_evidence_ids = set()

        for item in merged_evidence:

            if not isinstance(
                item,
                dict,
            ):
                continue

            evidence_id = item.get(
                "evidence_id"
            )

            if evidence_id:

                if evidence_id in seen_evidence_ids:
                    continue

                seen_evidence_ids.add(
                    evidence_id
                )

            deduplicated_evidence.append(
                item
            )

        # ====================================================
        # SUCCESS
        # ====================================================

        return {
            **current_state,
            "evidence": deduplicated_evidence,
            "allowed": True,
            "current_stage": "multi_domain",
        }

    except Exception as exc:

        logger.exception(
            "Multi-domain orchestration failed."
        )

        return {
            **current_state,
            "evidence": merged_evidence,
            "allowed": False,
            "current_stage": "multi_domain",
            "error": type(exc).__name__,
        }


# ============================================================
# AGENT EXECUTION ROUTING
# ============================================================

def agent_execution_route(
    state: AgentState,
) -> Literal[
    "evidence",
    "blocked",
]:
    """
    Route successful agent execution to the shared Evidence Layer.

    Failed executions stop the graph.
    """

    if not state.get(
        "allowed",
        False,
    ):
        return "blocked"

    if not state.get(
        "evidence",
        [],
    ):
        return "blocked"

    return "evidence"


# ============================================================
# RESPONSE SERVICES
# ============================================================

response_generator = build_response_generator()

output_guardrails = build_output_guardrails()


# ============================================================
# NODE: RESPONSE GENERATOR
# ============================================================

def response_generator_node(
    state: AgentState,
) -> AgentState:
    """
    Generate the final response from approved evidence.
    """

    try:

        result = response_generator.generate(
            query=state.get(
                "query",
                "",
            ),
            evidence=state.get(
                "evidence",
                [],
            ),
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
    Validate the generated response before returning it.
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
            "output_guardrails_allowed": result[
                "allowed"
            ],
            "output_guardrails_reason": result[
                "reason"
            ],
            "safe_response": result[
                "safe_response"
            ],
            "allowed": result[
                "allowed"
            ],
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

        result = validate_user_query(
            query
        )

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
            "authorization_allowed": False,
            "authorization_reason": (
                "Authorization requires authentication."
            ),
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

    user = state.get(
        "user"
    )

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

    stage = state.get(
        "current_stage"
    )

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

        if state.get(
            "requires_policy",
            False,
        ):
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

    builder = StateGraph(
        AgentState
    )

    # ========================================================
    # SECURITY NODES
    # ========================================================

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

    # ========================================================
    # FRAUD AGENT
    # ========================================================

    builder.add_node(
        "fraud_agent",
        fraud_agent_node,
    )

    # ========================================================
    # COMPANY KNOWLEDGE RAG
    # ========================================================

    builder.add_node(
        "company_knowledge",
        company_knowledge_harness_node,
    )

    # ========================================================
    # POLICY RAG
    # ========================================================

    builder.add_node(
        "policy_rag",
        policy_rag_harness_node,
    )

    # ========================================================
    # EXTERNAL RESEARCH
    # ========================================================

    builder.add_node(
        "external_research",
        external_research_harness_node,
    )

    # ========================================================
    # MULTI-DOMAIN
    # ========================================================

    builder.add_node(
        "multi_domain",
        multi_domain_node,
    )

    # ========================================================
    # SHARED EVIDENCE LAYER
    # ========================================================

    builder.add_node(
        "evidence_layer",
        evidence_layer_node,
    )

    # ========================================================
    # RESPONSE GENERATOR
    # ========================================================

    builder.add_node(
        "response_generator",
        response_generator_node,
    )

    # ========================================================
    # OUTPUT GUARDRAILS
    # ========================================================

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

            "external": "external_research",

            "multi_domain": "multi_domain",

            "direct": END,

            "unknown": END,
        },
    )

    # ========================================================
    # FRAUD → EVIDENCE
    # ========================================================

    builder.add_conditional_edges(
        "fraud_agent",
        agent_execution_route,
        {
            "evidence": "evidence_layer",
            "blocked": END,
        },
    )

    # ========================================================
    # COMPANY KNOWLEDGE → EVIDENCE
    # ========================================================

    builder.add_conditional_edges(
        "company_knowledge",
        agent_execution_route,
        {
            "evidence": "evidence_layer",
            "blocked": END,
        },
    )

    # ========================================================
    # POLICY → EVIDENCE
    # ========================================================

    builder.add_conditional_edges(
        "policy_rag",
        agent_execution_route,
        {
            "evidence": "evidence_layer",
            "blocked": END,
        },
    )

    # ========================================================
    # EXTERNAL RESEARCH → EVIDENCE
    # ========================================================

    builder.add_conditional_edges(
        "external_research",
        agent_execution_route,
        {
            "evidence": "evidence_layer",
            "blocked": END,
        },
    )

    # ========================================================
    # MULTI-DOMAIN → EVIDENCE
    # ========================================================

    builder.add_conditional_edges(
        "multi_domain",
        agent_execution_route,
        {
            "evidence": "evidence_layer",
            "blocked": END,
        },
    )

    # ========================================================
    # EVIDENCE → RESPONSE
    # ========================================================

    builder.add_edge(
        "evidence_layer",
        "response_generator",
    )

    # ========================================================
    # RESPONSE → OUTPUT GUARDRAILS
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