"""
Agent Harness

Provides controlled execution for autonomous agents.

Responsibilities:
- Maximum iterations
- Maximum tool calls
- Maximum retries
- Execution timeout
- Token budget tracking
- Minimum confidence threshold
- Evidence requirement
- Retry handling
- Termination conditions

The harness is intentionally independent from specific agents.

Architecture:

    Supervisor
        |
        v
    Agent Harness
        |
        +---- Fraud Agent
        |
        +---- Company Knowledge RAG
        |
        +---- Policy RAG
        |
        +---- External Research
        |
        v
    HarnessResult
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional


@dataclass
class AgentHarnessConfig:
    """
    Configuration controlling autonomous agent execution.
    """

    max_iterations: int = 5
    max_tool_calls: int = 10
    max_retries: int = 2
    timeout_seconds: float = 60.0
    token_budget: int = 8000

    min_confidence: float = 0.80
    require_evidence: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative.")

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")

        if self.token_budget <= 0:
            raise ValueError("token_budget must be greater than 0.")

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0.")


@dataclass
class HarnessResult:
    """
    Structured result returned by the Agent Harness.
    """

    success: bool = False
    response: Optional[str] = None

    iterations: int = 0
    tool_calls: int = 0
    retries: int = 0

    confidence: float = 0.0
    evidence_found: bool = False

    termination_reason: str = ""

    error: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecutionResult:
    """
    Normalized result returned by an agent execution.

    Agents can return a dictionary or this dataclass.
    """

    response: Optional[str] = None
    confidence: float = 0.0
    evidence_found: bool = False

    tool_calls: int = 0
    tokens_used: int = 0

    completed: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentHarness:
    """
    Controlled execution layer for autonomous agents.

    The harness does not know how an agent reasons.

    It only controls:
    - how many times the agent may execute
    - how many tool calls are allowed
    - how many retries are allowed
    - how long execution may continue
    - how much token budget may be consumed
    - whether confidence/evidence requirements are satisfied
    """

    def __init__(
        self,
        config: AgentHarnessConfig | None = None,
    ) -> None:
        self.config = config or AgentHarnessConfig()

    async def run(
        self,
        agent: Callable[..., Any],
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> HarnessResult:
        """
        Execute an agent under harness controls.

        Parameters
        ----------
        agent:
            Callable agent. It may be synchronous or asynchronous.

        state:
            Optional state dictionary passed to the agent.

        Returns
        -------
        HarnessResult
            Structured execution result.
        """

        state = dict(state or {})

        start_time = time.monotonic()

        total_tool_calls = 0
        total_tokens = 0
        retries = 0

        last_execution: Optional[AgentExecutionResult] = None
        last_error: Optional[Exception] = None

        for iteration in range(1, self.config.max_iterations + 1):

            elapsed = time.monotonic() - start_time

            if elapsed >= self.config.timeout_seconds:
                return HarnessResult(
                    success=False,
                    iterations=iteration - 1,
                    tool_calls=total_tool_calls,
                    retries=retries,
                    confidence=(
                        last_execution.confidence
                        if last_execution
                        else 0.0
                    ),
                    evidence_found=(
                        last_execution.evidence_found
                        if last_execution
                        else False
                    ),
                    termination_reason="timeout",
                    error="Agent execution exceeded timeout.",
                )

            if total_tool_calls >= self.config.max_tool_calls:
                return HarnessResult(
                    success=False,
                    iterations=iteration - 1,
                    tool_calls=total_tool_calls,
                    retries=retries,
                    confidence=(
                        last_execution.confidence
                        if last_execution
                        else 0.0
                    ),
                    evidence_found=(
                        last_execution.evidence_found
                        if last_execution
                        else False
                    ),
                    termination_reason="tool_call_limit",
                    error="Maximum tool-call limit reached.",
                )

            remaining_timeout = max(
                0.01,
                self.config.timeout_seconds - elapsed,
            )

            try:
                raw_result = await asyncio.wait_for(
                    self._execute_agent(
                        agent,
                        state,
                    ),
                    timeout=remaining_timeout,
                )

                execution = self._normalize_result(raw_result)

                last_execution = execution

                total_tool_calls += execution.tool_calls
                total_tokens += execution.tokens_used

                if total_tokens > self.config.token_budget:
                    return HarnessResult(
                        success=False,
                        response=execution.response,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        confidence=execution.confidence,
                        evidence_found=execution.evidence_found,
                        termination_reason="token_budget_exceeded",
                        error="Agent token budget exceeded.",
                        metadata=execution.metadata,
                    )

                if total_tool_calls > self.config.max_tool_calls:
                    return HarnessResult(
                        success=False,
                        response=execution.response,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        confidence=execution.confidence,
                        evidence_found=execution.evidence_found,
                        termination_reason="tool_call_limit",
                        error="Maximum tool-call limit exceeded.",
                        metadata=execution.metadata,
                    )

                if self._termination_successful(execution):
                    return HarnessResult(
                        success=True,
                        response=execution.response,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        confidence=execution.confidence,
                        evidence_found=execution.evidence_found,
                        termination_reason="success",
                        metadata=execution.metadata,
                    )

                if iteration >= self.config.max_iterations:
                    return HarnessResult(
                        success=False,
                        response=execution.response,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        confidence=execution.confidence,
                        evidence_found=execution.evidence_found,
                        termination_reason="iteration_limit",
                        error="Maximum iteration limit reached.",
                        metadata=execution.metadata,
                    )

                if retries >= self.config.max_retries:
                    return HarnessResult(
                        success=False,
                        response=execution.response,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        confidence=execution.confidence,
                        evidence_found=execution.evidence_found,
                        termination_reason="retry_exhausted",
                        error="Maximum retry limit reached.",
                        metadata=execution.metadata,
                    )

                retries += 1

                state = self._prepare_retry_state(
                    state,
                    execution,
                )

            except asyncio.TimeoutError:
                return HarnessResult(
                    success=False,
                    iterations=iteration,
                    tool_calls=total_tool_calls,
                    retries=retries,
                    confidence=(
                        last_execution.confidence
                        if last_execution
                        else 0.0
                    ),
                    evidence_found=(
                        last_execution.evidence_found
                        if last_execution
                        else False
                    ),
                    termination_reason="timeout",
                    error="Agent execution timed out.",
                )

            except Exception as exc:
                last_error = exc

                if retries >= self.config.max_retries:
                    return HarnessResult(
                        success=False,
                        iterations=iteration,
                        tool_calls=total_tool_calls,
                        retries=retries,
                        confidence=(
                            last_execution.confidence
                            if last_execution
                            else 0.0
                        ),
                        evidence_found=(
                            last_execution.evidence_found
                            if last_execution
                            else False
                        ),
                        termination_reason="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )

                retries += 1

                state = {
                    **state,
                    "harness_retry": True,
                    "harness_retry_count": retries,
                    "harness_last_error": str(exc),
                }

        return HarnessResult(
            success=False,
            iterations=self.config.max_iterations,
            tool_calls=total_tool_calls,
            retries=retries,
            confidence=(
                last_execution.confidence
                if last_execution
                else 0.0
            ),
            evidence_found=(
                last_execution.evidence_found
                if last_execution
                else False
            ),
            termination_reason="iteration_limit",
            error=(
                str(last_error)
                if last_error
                else "Agent execution did not complete."
            ),
        )

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

    def _normalize_result(
        self,
        result: Any,
    ) -> AgentExecutionResult:
        """
        Normalize different agent result formats.
        """

        if isinstance(result, AgentExecutionResult):
            return result

        if result is None:
            return AgentExecutionResult(
                response=None,
                confidence=0.0,
                evidence_found=False,
                completed=False,
            )

        if isinstance(result, dict):
            response = result.get(
                "response",
                result.get("answer"),
            )

            confidence = result.get(
                "confidence",
                result.get(
                    "supervisor_confidence",
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

            return AgentExecutionResult(
                response=response,
                confidence=float(confidence or 0.0),
                evidence_found=bool(evidence_found),
                tool_calls=int(tool_calls or 0),
                tokens_used=int(tokens_used or 0),
                completed=bool(completed),
                metadata={
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "response",
                        "answer",
                        "confidence",
                        "supervisor_confidence",
                        "evidence",
                        "evidence_found",
                        "tool_calls",
                        "tool_call_count",
                        "tokens_used",
                        "token_usage",
                        "completed",
                    }
                },
            )

        raise TypeError(
            "Agent result must be AgentExecutionResult or dict."
        )

    def _termination_successful(
        self,
        execution: AgentExecutionResult,
    ) -> bool:
        """
        Determine whether agent execution satisfies success criteria.
        """

        if not execution.completed:
            return False

        if execution.confidence < self.config.min_confidence:
            return False

        if (
            self.config.require_evidence
            and not execution.evidence_found
        ):
            return False

        return True

    def _prepare_retry_state(
        self,
        state: Dict[str, Any],
        execution: AgentExecutionResult,
    ) -> Dict[str, Any]:
        """
        Add harness feedback to the next execution attempt.
        """

        retry_reasons = []

        if execution.confidence < self.config.min_confidence:
            retry_reasons.append("low_confidence")

        if (
            self.config.require_evidence
            and not execution.evidence_found
        ):
            retry_reasons.append("missing_evidence")

        return {
            **state,
            "harness_retry": True,
            "harness_retry_reasons": retry_reasons,
            "harness_previous_response": execution.response,
            "harness_previous_confidence": execution.confidence,
            "harness_previous_evidence_found": (
                execution.evidence_found
            ),
        }