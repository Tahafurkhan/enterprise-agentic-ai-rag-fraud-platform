"""
Shared LangGraph state for the Enterprise Fraud Intelligence platform.

The state travels through the graph and accumulates:

    request
        ↓
    security results
        ↓
    supervisor decision
        ↓
    agent/tool evidence
        ↓
    final response
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    `total=False` allows nodes to populate fields progressively.
    """

    # ============================================================
    # REQUEST
    # ============================================================

    user_id: str
    query: str

    # ============================================================
    # AUTHENTICATION
    # ============================================================

    authenticated: bool
    user: Any

    # ============================================================
    # INPUT GUARDRAILS
    # ============================================================

    input_guardrail_allowed: bool
    input_guardrail_reason: str

    # ============================================================
    # AI SAFETY
    # ============================================================

    safety_allowed: bool
    safety_category: str
    safety_risk_level: str
    safety_reason: str

    # ============================================================
    # AUTHORIZATION
    # ============================================================

    authorization_allowed: bool
    authorization_reason: str

    # ============================================================
    # SUPERVISOR
    # ============================================================

    supervisor_decision: Any
    supervisor_domain: Optional[str]
    supervisor_tools: List[str]
    supervisor_reason: str
    supervisor_confidence: float

    requires_policy: bool
    requires_structured_data: bool
    requires_external_research: bool

    # ============================================================
    # AGENT EXECUTION
    # ============================================================

    agent_name: Optional[str]
    tool_results: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]

    # ============================================================
    # RESPONSE
    # ============================================================

    response: Optional[str]

    # ============================================================
    # GRAPH CONTROL
    # ============================================================

    current_stage: str
    allowed: bool
    error: Optional[str]
    
    
    # FRAUD AGENT
    fraud_tool_name: Optional[str]
    fraud_tool_arguments: Dict[str, Any]
    fraud_execution_allowed: bool
    fraud_execution_reason: str