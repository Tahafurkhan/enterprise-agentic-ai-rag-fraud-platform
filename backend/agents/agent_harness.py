
"""
Agent execution harness for the Enterprise Fraud Intelligence platform.

The Agent Harness provides a common execution boundary around domain agents
and RAG agents.

Responsibilities:

    Agent
        ↓
    Agent Harness
        ├── iteration limit
        ├── retry limit
        ├── tool-call limit
        ├── timeout
        ├── token-budget accounting
        ├── confidence threshold
        ├── evidence requirement
        └── termination decision

The harness does not perform routing, authentication, authorization,
retrieval, MCP execution, or response generation.

Domain-specific agents remain responsible for their own internal logic.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


# ============================================================
# HARNESS CONFIGURATION
# ============================================================


@dataclass
class AgentHarnessConfig:
    """
    Configuration controlling agent execution.

    These limits apply to the outer agent execution boundary.
    Domain agents may have their own internal retry/retrieval logic.
    """

    max_iterations: int = 5
    max_tool_calls: int = 10
    max_retries: int = 2
    timeout_seconds: float = 60.0
    token_budget: int = 8000
    min_confidence: float = 0.80
    require_evidence: bool = True

    def __post_init__(self) -> None:
        """Validate harness configuration."""

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        if self.max_tool_calls < 0:
            raise ValueError(
                "max_tool_calls must be non-negative."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries must be non-negative."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        if self.token_budget <= 0:
            raise ValueError(
                "token_budget must be greater than 0."
            )

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                "min_confidence must be between 0.0 and 1.0."
            )


# ============================================================
# HARNESS RESULT
# ============================================================


@dataclass
class HarnessResult:
    """
    Final result produced by the Agent Harness.
    """

    success: bool = False
    response: Optional[str] = None

    iterations: int = 0
    tool_calls: int = 0
    retries: int = 0
    tokens_used: int = 0

    confidence: float = 0.0
    evidence_found: bool = False

    termination_reason: str = ""
    error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# AGENT EXECUTION RESULT
# ============================================================


@dataclass
class AgentExecutionResult:
    """
    Normalized execution result expected by the harness.

    Domain agents do not have to return this exact class.
    The harness normalizes dictionaries into this contract.
    """

    response: Optional[str] = None
    confidence: float = 0.0
    evidence_found: bool = False

    tool_calls: int = 0
    tokens_used: int = 0

    completed: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# AGENT HARNESS
# ============================================================


class AgentHarness:
    """
    Common execution boundary for enterprise agents.

    The harness governs execution but does not own domain-specific
    agent behavior.
    """

    def __init__(
        self,
        config: AgentHarnessConfig | None = None,
    ) -> None:
        self.config = config or AgentHarnessConfig()

    # ========================================================
    # PUBLIC EXECUTION
    # ========================================================

    async def run(
        self,
        agent: Callable[..., Any],
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> HarnessResult:
        """
        Execute an agent under the configured harness limits.

        The agent may be synchronous or asynchronous.

        Retry state is passed back into the agent when execution
        does not satisfy the harness termination conditions.
        """

        state = dict(state or {})

        start_time = time.monotonic()

        total_tool_calls = 0
        total_tokens = 0
        retries = 0

        last_execution: Optional[
            AgentExecutionResult
        ] = None

        last_error: Optional[Exception] = None

        for iteration in range(
            1,
            self.config.max_iterations + 1,
        ):
            # ------------------------------------------------
            # TIMEOUT CHECK
            # ------------------------------------------------

            elapsed = time.monotonic() - start_time

            if elapsed >= self.config.timeout_seconds:
                return self._build_result(
                    success=False,
                    execution=last_execution,
                    iterations=iteration - 1,
                    tool_calls=total_tool_calls,
                    retries=retries,
                    tokens_used=total_tokens,
                    termination_reason="timeout",
                    error=(
                        str(last_error)
                        if last_error is not None
                        else None
                    ),
                )

            # ------------------------------------------------
            # TOOL-CALL LIMIT CHECK
            # ------------------------------------------------

            if (
                total_tool_calls
                >= self.config.max_tool_calls
            ):
                return self._build_result(
                    success=False,
                    execution=last_execution,
                    iterations=iteration - 1,
                    tool_calls=total_tool_calls,
                    retries=retries,
                    tokens_used=total_tokens,
                    termination_reason="tool_call_limit",
                )

            remaining_timeout = max(
                0.01,
                self.config.timeout_seconds - elapsed,
            )

            try:
                # --------------------------------------------
                # AGENT EXECUTION
                # --------------------------------------------

                raw_result = await asyncio.wait_for(
                    self._execute_agent(
                        agent,
                        state,
                    ),
                    timeout=remaining_timeout,
                )

                execution = self._normalize_result(
                    raw_result
                )

                last_execution = execution

                total_tool_calls += (
                    execution.tool_calls
                )

                total_tokens += (
                    execution.tokens_used
                )

                # --------------------------------------------
                # TOKEN BUDGET
                # --------------------------------------------

                if (
                    total_tokens
                    > self.config.token_budget
                ):
                    return self._build_result(
                        success=False,
                        execution=last_execution,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        tokens_used=total_tokens,
                        termination_reason=(
                            "token_budget_exceeded"
                        ),
                    )

                # --------------------------------------------
                # TOOL-CALL BUDGET
                # --------------------------------------------

                if (
                    total_tool_calls
                    > self.config.max_tool_calls
                ):
                    return self._build_result(
                        success=False,
                        execution=last_execution,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        tokens_used=total_tokens,
                        termination_reason=(
                            "tool_call_limit"
                        ),
                    )

                # --------------------------------------------
                # SUCCESSFUL TERMINATION
                # --------------------------------------------

                if self._termination_successful(
                    execution
                ):
                    return self._build_result(
                        success=True,
                        execution=last_execution,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        tokens_used=total_tokens,
                        termination_reason="success",
                    )

                # --------------------------------------------
                # ITERATION LIMIT
                # --------------------------------------------

                if (
                    iteration
                    >= self.config.max_iterations
                ):
                    return self._build_result(
                        success=False,
                        execution=last_execution,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        tokens_used=total_tokens,
                        termination_reason=(
                            "iteration_limit"
                        ),
                    )

                # --------------------------------------------
                # RETRY LIMIT
                # --------------------------------------------

                if retries >= self.config.max_retries:
                    return self._build_result(
                        success=False,
                        execution=last_execution,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        tokens_used=total_tokens,
                        termination_reason=(
                            "retry_exhausted"
                        ),
                    )

                # --------------------------------------------
                # PREPARE NEXT RETRY
                # --------------------------------------------

                retries += 1

                state = self._prepare_retry_state(
                    state,
                    execution,
                )

            except asyncio.TimeoutError:
                return self._build_result(
                    success=False,
                    execution=last_execution,
                    iterations=iteration,
                    tool_calls=total_tool_calls,
                    retries=retries,
                    tokens_used=total_tokens,
                    termination_reason="timeout",
                    error="Agent execution timed out.",
                )

            except Exception as exc:
                last_error = exc

                if retries >= self.config.max_retries:
                    return self._build_result(
                        success=False,
                        execution=last_execution,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        tokens_used=total_tokens,
                        termination_reason="error",
                        error=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )

                retries += 1

                state = {
                    **state,
                    "harness_retry": True,
                    "harness_retry_count": retries,
                    "harness_last_error": str(exc),
                }

        return self._build_result(
            success=False,
            execution=last_execution,
            iterations=self.config.max_iterations,
            tool_calls=total_tool_calls,
            retries=retries,
            tokens_used=total_tokens,
            termination_reason="iteration_limit",
            error=(
                str(last_error)
                if last_error is not None
                else None
            ),
        )

    # ========================================================
    # AGENT EXECUTION
    # ========================================================

    async def _execute_agent(
        self,
        agent: Callable[..., Any],
        state: Dict[str, Any],
    ) -> Any:
        """
        Execute either a synchronous or asynchronous agent.
        """

        result = agent(state)

        if inspect.isawaitable(result):
            return await result

        return result

    # ========================================================
    # RESULT NORMALIZATION
    # ========================================================

    def _normalize_result(
        self,
        result: Any,
    ) -> AgentExecutionResult:
        """
        Normalize supported agent result formats.

        Supported:

            AgentExecutionResult
            dict

        A missing result is treated as incomplete execution.
        """

        if isinstance(
            result,
            AgentExecutionResult,
        ):
            self._validate_execution_metrics(
                result
            )

            return result

        if result is None:
            return AgentExecutionResult(
                response=None,
                confidence=0.0,
                evidence_found=False,
                tool_calls=0,
                tokens_used=0,
                completed=False,
            )

        if isinstance(result, dict):
            response = result.get(
                "response",
                result.get("answer"),
            )

            # IMPORTANT:
            #
            # Do not fall back to supervisor_confidence.
            #
            # Supervisor confidence describes routing confidence,
            # not execution or answer confidence.

            confidence = result.get(
                "confidence",
                result.get(
                    "groundedness_score",
                    0.0,
                ),
            )

            evidence = result.get(
                "evidence",
                [],
            )

            evidence_found = result.get(
                "evidence_found",
                bool(evidence),
            )

            tool_calls = result.get(
                "tool_calls",
                result.get(
                    "tool_call_count",
                    0,
                ),
            )

            tokens_used = result.get(
                "tokens_used",
                result.get(
                    "token_usage",
                    0,
                ),
            )

            completed = result.get(
                "completed",
                True,
            )

            metadata = {
                key: value
                for key, value in result.items()
                if key not in {
                    "response",
                    "answer",
                    "confidence",
                    "groundedness_score",
                    "evidence",
                    "evidence_found",
                    "tool_calls",
                    "tool_call_count",
                    "tokens_used",
                    "token_usage",
                    "completed",
                }
            }

            normalized = AgentExecutionResult(
                response=response,
                confidence=float(
                    confidence or 0.0
                ),
                evidence_found=bool(
                    evidence_found
                ),
                tool_calls=int(
                    tool_calls or 0
                ),
                tokens_used=int(
                    tokens_used or 0
                ),
                completed=bool(completed),
                metadata=metadata,
            )

            self._validate_execution_metrics(
                normalized
            )

            return normalized

        raise TypeError(
            "Agent result must be "
            "AgentExecutionResult, dict, or None."
        )

    # ========================================================
    # EXECUTION METRIC VALIDATION
    # ========================================================

    @staticmethod
    def _validate_execution_metrics(
        execution: AgentExecutionResult,
    ) -> None:
        """
        Validate execution metrics reported by an agent.
        """

        if execution.tool_calls < 0:
            raise ValueError(
                "tool_calls must be non-negative."
            )

        if execution.tokens_used < 0:
            raise ValueError(
                "tokens_used must be non-negative."
            )

        if not 0.0 <= execution.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0."
            )

    # ========================================================
    # TERMINATION
    # ========================================================

    def _termination_successful(
        self,
        execution: AgentExecutionResult,
    ) -> bool:
        """
        Determine whether execution satisfies the harness
        success criteria.
        """

        if not execution.completed:
            return False

        if (
            execution.confidence
            < self.config.min_confidence
        ):
            return False

        if (
            self.config.require_evidence
            and not execution.evidence_found
        ):
            return False

        return True

    # ========================================================
    # RETRY STATE
    # ========================================================

    def _prepare_retry_state(
        self,
        state: Dict[str, Any],
        execution: AgentExecutionResult,
    ) -> Dict[str, Any]:
        """
        Prepare state for the next harness retry.
        """

        retry_reasons = []

        if (
            execution.confidence
            < self.config.min_confidence
        ):
            retry_reasons.append(
                "low_confidence"
            )

        if (
            self.config.require_evidence
            and not execution.evidence_found
        ):
            retry_reasons.append(
                "missing_evidence"
            )

        return {
            **state,
            "harness_retry": True,
            "harness_retry_reasons": retry_reasons,
            "harness_previous_response": (
                execution.response
            ),
            "harness_previous_confidence": (
                execution.confidence
            ),
            "harness_previous_evidence_found": (
                execution.evidence_found
            ),
        }

    # ========================================================
    # RESULT BUILDER
    # ========================================================

    @staticmethod
    def _build_result(
        *,
        success: bool,
        execution: Optional[
            AgentExecutionResult
        ],
        iterations: int,
        tool_calls: int,
        retries: int,
        tokens_used: int,
        termination_reason: str,
        error: Optional[str] = None,
    ) -> HarnessResult:
        """
        Build a consistent HarnessResult.

        The final execution is preserved under:

            metadata["last_execution"]

        This allows the LangGraph orchestration layer to recover
        the complete domain-agent state after successful execution.
        """

        if execution is None:
            response = None
            confidence = 0.0
            evidence_found = False
        else:
            response = execution.response
            confidence = execution.confidence
            evidence_found = execution.evidence_found

        return HarnessResult(
            success=success,
            response=response,
            iterations=iterations,
            tool_calls=tool_calls,
            retries=retries,
            tokens_used=tokens_used,
            confidence=confidence,
            evidence_found=evidence_found,
            termination_reason=termination_reason,
            error=error,
            metadata={
                "last_execution": execution,
            },
        )

