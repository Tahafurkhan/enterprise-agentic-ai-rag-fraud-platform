
from __future__ import annotations

import asyncio

import pytest

from backend.agents.agent_harness import (
    AgentExecutionResult,
    AgentHarness,
    AgentHarnessConfig,
)


# ============================================================
# HELPERS
# ============================================================

def build_harness(**overrides) -> AgentHarness:
    config = AgentHarnessConfig(
        max_iterations=5,
        max_tool_calls=10,
        max_retries=2,
        timeout_seconds=5.0,
        token_budget=8000,
        min_confidence=0.80,
        require_evidence=True,
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    return AgentHarness(config)


# ============================================================
# CONFIGURATION
# ============================================================

def test_default_configuration_is_valid():
    config = AgentHarnessConfig()

    assert config.max_iterations == 5
    assert config.max_tool_calls == 10
    assert config.max_retries == 2
    assert config.timeout_seconds == 60.0
    assert config.token_budget == 8000
    assert config.min_confidence == 0.80
    assert config.require_evidence is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_iterations", 0),
        ("max_tool_calls", -1),
        ("max_retries", -1),
        ("timeout_seconds", 0),
        ("token_budget", 0),
    ],
)
def test_invalid_positive_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        AgentHarnessConfig(**{field: value})


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1],
)
def test_invalid_confidence_is_rejected(confidence):
    with pytest.raises(ValueError):
        AgentHarnessConfig(min_confidence=confidence)


# ============================================================
# BASIC EXECUTION
# ============================================================

@pytest.mark.asyncio
async def test_successful_agent_execution():
    harness = build_harness()

    async def agent(state):
        return AgentExecutionResult(
            response="Fraud evidence found.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(
        agent,
        state={
            "query": "Show suspicious transactions",
        },
    )

    assert result.success is True
    assert result.response == "Fraud evidence found."
    assert result.iterations == 1
    assert result.tool_calls == 1
    assert result.retries == 0
    assert result.confidence == 0.95
    assert result.evidence_found is True
    assert result.tokens_used == 100
    assert result.termination_reason == "success"
    assert result.error is None


@pytest.mark.asyncio
async def test_synchronous_agent_is_supported():
    harness = build_harness()

    def agent(state):
        return AgentExecutionResult(
            response="Synchronous response.",
            confidence=0.90,
            evidence_found=True,
            tool_calls=0,
            tokens_used=50,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Synchronous response."
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_dictionary_agent_result_is_normalized():
    harness = build_harness()

    async def agent(state):
        return {
            "response": "Grounded answer.",
            "confidence": 0.92,
            "evidence": [
                {
                    "source": "test",
                }
            ],
            "tool_calls": 2,
            "tokens_used": 150,
            "completed": True,
        }

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Grounded answer."
    assert result.confidence == 0.92
    assert result.evidence_found is True
    assert result.tool_calls == 2
    assert result.tokens_used == 150


# ============================================================
# CONFIDENCE
# ============================================================

@pytest.mark.asyncio
async def test_low_confidence_causes_retry():
    harness = build_harness(
        max_iterations=2,
        max_retries=1,
        min_confidence=0.80,
    )

    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        if calls == 1:
            return AgentExecutionResult(
                response="Weak answer.",
                confidence=0.50,
                evidence_found=True,
                tool_calls=1,
                tokens_used=100,
                completed=True,
            )

        return AgentExecutionResult(
            response="Strong answer.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Strong answer."
    assert result.iterations == 2
    assert result.retries == 1


@pytest.mark.asyncio
async def test_low_confidence_stops_when_retry_is_exhausted():
    harness = build_harness(
        max_iterations=3,
        max_retries=1,
        min_confidence=0.80,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Weak answer.",
            confidence=0.50,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.retries == 1
    assert result.termination_reason == "retry_exhausted"


# ============================================================
# EVIDENCE
# ============================================================

@pytest.mark.asyncio
async def test_missing_evidence_causes_retry():
    harness = build_harness(
        max_iterations=2,
        max_retries=1,
        require_evidence=True,
    )

    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        if calls == 1:
            return AgentExecutionResult(
                response="No evidence.",
                confidence=0.95,
                evidence_found=False,
                tool_calls=0,
                tokens_used=100,
                completed=True,
            )

        return AgentExecutionResult(
            response="Evidence-backed answer.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is True
    assert result.iterations == 2
    assert result.retries == 1
    assert result.evidence_found is True


@pytest.mark.asyncio
async def test_missing_evidence_reaches_iteration_limit():
    harness = build_harness(
        max_iterations=2,
        max_retries=1,
        require_evidence=True,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Unsupported answer.",
            confidence=0.95,
            evidence_found=False,
            tool_calls=0,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.evidence_found is False
    assert result.termination_reason == "iteration_limit"


@pytest.mark.asyncio
async def test_evidence_requirement_can_be_disabled():
    harness = build_harness(
        require_evidence=False,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Answer without evidence.",
            confidence=0.95,
            evidence_found=False,
            tool_calls=0,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is True


# ============================================================
# TOOL CALL LIMIT
# ============================================================

@pytest.mark.asyncio
async def test_tool_call_limit_is_enforced():
    harness = build_harness(
        max_tool_calls=2,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Too many tool calls.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=3,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.tool_calls == 3
    assert result.termination_reason == "tool_call_limit"


@pytest.mark.asyncio
async def test_tool_calls_are_accumulated_across_iterations():
    harness = build_harness(
        max_iterations=2,
        max_tool_calls=10,
        max_retries=1,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Retry.",
            confidence=0.50,
            evidence_found=True,
            tool_calls=2,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.iterations == 2
    assert result.tool_calls == 4


# ============================================================
# TOKEN BUDGET
# ============================================================

@pytest.mark.asyncio
async def test_token_budget_is_enforced():
    harness = build_harness(
        token_budget=500,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Large execution.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=600,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.tokens_used == 600
    assert result.termination_reason == "token_budget_exceeded"


@pytest.mark.asyncio
async def test_tokens_are_accumulated_across_iterations():
    harness = build_harness(
        max_iterations=2,
        max_retries=1,
        token_budget=500,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Retry.",
            confidence=0.50,
            evidence_found=True,
            tool_calls=1,
            tokens_used=300,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.tokens_used == 600
    assert result.termination_reason == "token_budget_exceeded"


# ============================================================
# ITERATION LIMIT
# ============================================================

@pytest.mark.asyncio
async def test_iteration_limit_is_enforced():
    harness = build_harness(
        max_iterations=2,
        max_retries=10,
        min_confidence=0.80,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Still weak.",
            confidence=0.50,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.iterations == 2
    assert result.termination_reason == "iteration_limit"


# ============================================================
# TIMEOUT
# ============================================================

@pytest.mark.asyncio
async def test_timeout_is_enforced():
    harness = build_harness(
        timeout_seconds=0.05,
    )

    async def agent(state):
        await asyncio.sleep(0.20)

        return AgentExecutionResult(
            response="Too late.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.termination_reason == "timeout"


# ============================================================
# RETRIES
# ============================================================

@pytest.mark.asyncio
async def test_agent_exception_can_retry():
    harness = build_harness(
        max_iterations=2,
        max_retries=1,
    )

    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise RuntimeError("temporary failure")

        return AgentExecutionResult(
            response="Recovered.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Recovered."
    assert result.iterations == 2
    assert result.retries == 1


@pytest.mark.asyncio
async def test_agent_exception_fails_after_retry_exhaustion():
    harness = build_harness(
        max_iterations=3,
        max_retries=1,
    )

    async def agent(state):
        raise RuntimeError("permanent failure")

    result = await harness.run(agent)

    assert result.success is False
    assert result.retries == 1
    assert result.termination_reason == "error"
    assert result.error is not None


# ============================================================
# RETRY STATE
# ============================================================

@pytest.mark.asyncio
async def test_retry_state_contains_previous_execution_context():
    harness = build_harness(
        max_iterations=2,
        max_retries=1,
    )

    captured_states = []

    async def agent(state):
        captured_states.append(dict(state))

        if len(captured_states) == 1:
            return AgentExecutionResult(
                response="Weak response.",
                confidence=0.50,
                evidence_found=True,
                tool_calls=1,
                tokens_used=100,
                completed=True,
            )

        return AgentExecutionResult(
            response="Recovered response.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(
        agent,
        state={
            "query": "test query",
        },
    )

    assert result.success is True
    assert len(captured_states) == 2

    retry_state = captured_states[1]

    assert retry_state["query"] == "test query"
    assert retry_state["harness_retry"] is True
    assert retry_state["harness_previous_confidence"] == 0.50


# ============================================================
# RESULT NORMALIZATION
# ============================================================

@pytest.mark.asyncio
async def test_answer_field_is_supported():
    harness = build_harness()

    async def agent(state):
        return {
            "answer": "Answer from RAG.",
            "groundedness_score": 0.90,
            "evidence": [
                {
                    "source": "policy",
                }
            ],
            "tool_results": [
                {
                    "tool": "policy_rag",
                }
            ],
            "tokens_used": 200,
            "completed": True,
        }

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Answer from RAG."
    assert result.confidence == 0.90
    assert result.evidence_found is True

    # The current harness counts the explicit
    # "tool_calls" field, not the length of tool_results.
    assert result.tool_calls == 0

    assert result.tokens_used == 200


@pytest.mark.asyncio
async def test_none_result_reaches_iteration_limit():
    harness = build_harness(
        max_iterations=1,
        max_retries=0,
    )

    async def agent(state):
        return None

    result = await harness.run(agent)

    assert result.success is False
    assert result.termination_reason == "iteration_limit"


@pytest.mark.asyncio
async def test_invalid_result_type_fails():
    harness = build_harness(
        max_iterations=1,
        max_retries=0,
    )

    async def agent(state):
        return "invalid result"

    result = await harness.run(agent)

    assert result.success is False
    assert result.error is not None


# ============================================================
# INVALID EXECUTION METRICS
# ============================================================

@pytest.mark.asyncio
async def test_negative_tool_calls_are_rejected():
    harness = build_harness(
        max_iterations=1,
        max_retries=0,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Invalid.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=-1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_negative_tokens_are_rejected():
    harness = build_harness(
        max_iterations=1,
        max_retries=0,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Invalid.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=-1,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_invalid_execution_confidence_is_rejected():
    harness = build_harness(
        max_iterations=1,
        max_retries=0,
    )

    async def agent(state):
        return AgentExecutionResult(
            response="Invalid.",
            confidence=1.5,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
        )

    result = await harness.run(agent)

    assert result.success is False
    assert result.error is not None


# ============================================================
# DOMAIN AGENT STATE PRESERVATION
# ============================================================

@pytest.mark.asyncio
async def test_domain_agent_state_can_be_preserved_in_metadata():
    harness = build_harness()

    domain_state = {
        "response": "Fraud result.",
        "evidence": [
            {
                "source": "fraud_gold",
                "tool": "get_fraud_card_alerts",
            }
        ],
        "tool_results": [
            {
                "tool": "get_fraud_card_alerts",
            }
        ],
        "allowed": True,
    }

    async def agent(state):
        return AgentExecutionResult(
            response="Fraud result.",
            confidence=1.0,
            evidence_found=True,
            tool_calls=1,
            tokens_used=100,
            completed=True,
            metadata={
                "agent_state": domain_state,
            },
        )

    result = await harness.run(agent)

    assert result.success is True

    last_execution = result.metadata["last_execution"]

    assert isinstance(
        last_execution,
        AgentExecutionResult,
    )

    assert (
        last_execution.metadata["agent_state"]
        == domain_state
    )


# ============================================================
# SUPERVISOR CONFIDENCE IS NOT USED AS AGENT CONFIDENCE
# ============================================================

@pytest.mark.asyncio
async def test_supervisor_confidence_is_not_used_as_agent_confidence():
    harness = build_harness(
        max_iterations=1,
        max_retries=0,
        min_confidence=0.80,
    )

    async def agent(state):
        return {
            "response": "Agent response.",
            "supervisor_confidence": 0.99,
            "evidence": [
                {
                    "source": "test",
                }
            ],
            "completed": True,
        }

    result = await harness.run(agent)

    assert result.success is False
    assert result.confidence == 0.0
    assert result.termination_reason == "iteration_limit"

