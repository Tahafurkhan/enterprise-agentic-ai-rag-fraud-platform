
"""
Shared Evidence Layer for the Enterprise Fraud Intelligence platform.

The Evidence Layer provides a common runtime boundary between
agent/RAG-specific evidence and the downstream Response Generator.

Flow:

    Agent / RAG
        ↓
    Evidence Layer
        ↓
    Canonical Evidence
        ↓
    Response Generator

Responsibilities:
- normalize evidence into the shared Evidence contract
- preserve existing evidence fields
- provide a single AgentState-compatible node
- fail closed if evidence processing cannot be completed

The Evidence Layer does not:
- access Databricks directly
- call MCP
- perform retrieval
- generate answers
- modify authorization decisions
"""

from __future__ import annotations

import logging
from typing import Any

from .evidence_normalizer import normalize_evidence


logger = logging.getLogger(__name__)


def evidence_layer_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize and validate evidence accumulated in AgentState.

    The node preserves the existing state and replaces only the
    evidence field with the canonical normalized evidence list.

    Args:
        state:
            Current enterprise AgentState.

    Returns:
        Updated state containing normalized evidence.

    Fail-closed behavior:
        If evidence normalization fails, the node marks the request
        as not allowed and records the failure stage/error.
    """

    try:
        evidence = state.get("evidence", [])

        normalized_evidence = normalize_evidence(
            evidence,
        )

        return {
            **state,
            "evidence": normalized_evidence,
            "current_stage": "evidence_layer",
        }

    except Exception as exc:
        logger.exception(
            "Evidence Layer processing failed."
        )

        return {
            **state,
            "evidence": [],
            "allowed": False,
            "current_stage": "evidence_layer",
            "error": type(exc).__name__,
        }


def build_evidence_layer_node():
    """
    Factory for the shared Evidence Layer node.

    Returns:
        Evidence Layer callable suitable for LangGraph.
    """

    return evidence_layer_node

