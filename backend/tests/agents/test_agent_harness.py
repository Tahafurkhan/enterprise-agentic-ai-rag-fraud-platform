import asyncio

import pytest

from backend.agents.agent_harness import (
    AgentExecutionResult,
    AgentHarness,
    AgentHarnessConfig,
)


@pytest.mark.asyncio
async def test_successful_execution():
    async def agent(state):
        return AgentExecutionResult(
            response="Fraud analysis completed.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=1,
            tokens_used=500,
            completed=True,
        )

    harness = AgentHarness()

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Fraud analysis completed."
    assert result.iterations == 1
    assert result.tool_calls == 1
    assert result.retries == 0
    assert result.confidence == 0.95
    assert result.evidence_found is True
    assert result.termination_reason == "success"


@pytest.mark.asyncio
async def test_iteration_limit():
    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        return AgentExecutionResult(
            response="Still processing.",
            confidence=0.50,
            evidence_found=True,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=3,
        max_retries=10,
        min_confidence=0.80,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert result.iterations == 3
    assert calls == 3
    assert result.termination_reason == "iteration_limit"


@pytest.mark.asyncio
async def test_tool_call_limit():
    async def agent(state):
        return AgentExecutionResult(
            response="Tool-heavy execution.",
            confidence=0.95,
            evidence_found=True,
            tool_calls=4,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=5,
        max_tool_calls=3,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert result.tool_calls == 4
    assert result.termination_reason == "tool_call_limit"


@pytest.mark.asyncio
async def test_retry_then_success():
    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        if calls == 1:
            return AgentExecutionResult(
                response="Initial weak answer.",
                confidence=0.50,
                evidence_found=True,
                completed=True,
            )

        return AgentExecutionResult(
            response="Corrected answer.",
            confidence=0.95,
            evidence_found=True,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=5,
        max_retries=2,
        min_confidence=0.80,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Corrected answer."
    assert result.iterations == 2
    assert result.retries == 1
    assert calls == 2
    assert result.termination_reason == "success"


@pytest.mark.asyncio
async def test_retry_exhaustion():
    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        return AgentExecutionResult(
            response="Weak answer.",
            confidence=0.40,
            evidence_found=True,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=5,
        max_retries=2,
        min_confidence=0.80,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert calls == 3
    assert result.retries == 2
    assert result.termination_reason == "retry_exhausted"


@pytest.mark.asyncio
async def test_confidence_threshold():
    async def agent(state):
        return AgentExecutionResult(
            response="Low confidence answer.",
            confidence=0.79,
            evidence_found=True,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=1,
        min_confidence=0.80,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert result.confidence == 0.79
    assert result.termination_reason == "iteration_limit"


@pytest.mark.asyncio
async def test_evidence_required():
    async def agent(state):
        return AgentExecutionResult(
            response="Answer without evidence.",
            confidence=0.95,
            evidence_found=False,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=1,
        require_evidence=True,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert result.evidence_found is False
    assert result.termination_reason == "iteration_limit"


@pytest.mark.asyncio
async def test_evidence_not_required():
    async def agent(state):
        return AgentExecutionResult(
            response="Answer without evidence.",
            confidence=0.95,
            evidence_found=False,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_iterations=1,
        require_evidence=False,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Answer without evidence."
    assert result.termination_reason == "success"


@pytest.mark.asyncio
async def test_timeout():
    async def agent(state):
        await asyncio.sleep(1)

        return AgentExecutionResult(
            response="Too late.",
            confidence=0.95,
            evidence_found=True,
            completed=True,
        )

    config = AgentHarnessConfig(
        timeout_seconds=0.05,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert result.termination_reason == "timeout"
    assert result.error is not None


@pytest.mark.asyncio
async def test_token_budget():
    async def agent(state):
        return AgentExecutionResult(
            response="Large token response.",
            confidence=0.95,
            evidence_found=True,
            tokens_used=9000,
            completed=True,
        )

    config = AgentHarnessConfig(
        token_budget=8000,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is False
    assert result.termination_reason == "token_budget_exceeded"


@pytest.mark.asyncio
async def test_sync_agent_is_supported():
    def agent(state):
        return {
            "response": "Synchronous response.",
            "confidence": 0.95,
            "evidence_found": True,
            "tool_calls": 1,
            "tokens_used": 100,
            "completed": True,
        }

    harness = AgentHarness()

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Synchronous response."
    assert result.tool_calls == 1
    assert result.termination_reason == "success"


@pytest.mark.asyncio
async def test_agent_exception_retries():
    calls = 0

    async def agent(state):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise RuntimeError("Temporary agent failure.")

        return AgentExecutionResult(
            response="Recovered response.",
            confidence=0.95,
            evidence_found=True,
            completed=True,
        )

    config = AgentHarnessConfig(
        max_retries=1,
    )

    harness = AgentHarness(config)

    result = await harness.run(agent)

    assert result.success is True
    assert result.response == "Recovered response."
    assert result.retries == 1
    assert calls == 2


def test_invalid_configuration():
    with pytest.raises(ValueError):
        AgentHarnessConfig(max_iterations=0)

    with pytest.raises(ValueError):
        AgentHarnessConfig(max_tool_calls=-1)

    with pytest.raises(ValueError):
        AgentHarnessConfig(max_retries=-1)

    with pytest.raises(ValueError):
        AgentHarnessConfig(timeout_seconds=0)

    with pytest.raises(ValueError):
        AgentHarnessConfig(token_budget=0)

    with pytest.raises(ValueError):
        AgentHarnessConfig(min_confidence=1.5)