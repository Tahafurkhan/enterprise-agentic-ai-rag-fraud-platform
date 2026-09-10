import pytest

from backend.agents.external_research_agent import (
    build_external_research_agent,
)


class FakeWebSearchProvider:
    """Deterministic web-search provider for unit tests."""

    def __init__(self):
        self.calls = []

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ):
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
            }
        )

        return [
            {
                "title": "Example Result",
                "url": "https://example.com/result",
                "snippet": "Example search result.",
                "content": "Example result content.",
                "provider": "fake",
                "published_at": "2026-09-01",
            }
        ]


class FailingWebSearchProvider:
    """Provider used to verify failure handling."""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ):
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_external_research_executes_authorized_web_search():
    provider = FakeWebSearchProvider()

    agent = build_external_research_agent(
        search_provider=provider,
    )

    result = await agent.ainvoke(
        {
            "question": "Latest fraud detection trends",
            "authorization_context": "test-user",
            "search_web_allowed": True,
        }
    )

    assert result["execution_allowed"] is True
    assert result["search_results"]
    assert result["evidence"]

    assert provider.calls == [
        {
            "query": "Latest fraud detection trends",
            "max_results": 5,
        }
    ]

    evidence = result["evidence"][0]

    assert evidence["evidence_id"] == "web-evidence-1"
    assert evidence["source"] == "external_web"
    assert evidence["tool"] == "web_search"
    assert evidence["retrieval_type"] == "web_search"

    assert evidence["data"]["title"] == "Example Result"
    assert evidence["data"]["url"] == (
        "https://example.com/result"
    )
    assert evidence["data"]["snippet"] == (
        "Example search result."
    )


@pytest.mark.asyncio
async def test_external_research_uses_current_query_when_present():
    provider = FakeWebSearchProvider()

    agent = build_external_research_agent(
        search_provider=provider,
    )

    result = await agent.ainvoke(
        {
            "question": "Original question",
            "current_query": "Improved external search query",
            "authorization_context": "test-user",
            "search_web_allowed": True,
            "max_results": 3,
        }
    )

    assert result["execution_allowed"] is True

    assert provider.calls == [
        {
            "query": "Improved external search query",
            "max_results": 3,
        }
    ]


@pytest.mark.asyncio
async def test_external_research_rejects_unauthorized_search():
    provider = FakeWebSearchProvider()

    agent = build_external_research_agent(
        search_provider=provider,
    )

    result = await agent.ainvoke(
        {
            "question": "Search the public web",
            "authorization_context": "test-user",
            "search_web_allowed": False,
        }
    )

    assert result["execution_allowed"] is False
    assert result["search_results"] == []
    assert result["evidence"] == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_external_research_rejects_missing_authorization_context():
    provider = FakeWebSearchProvider()

    agent = build_external_research_agent(
        search_provider=provider,
    )

    result = await agent.ainvoke(
        {
            "question": "Search the public web",
            "search_web_allowed": True,
        }
    )

    assert result["execution_allowed"] is False
    assert result["search_results"] == []
    assert result["evidence"] == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_external_research_rejects_empty_query():
    provider = FakeWebSearchProvider()

    agent = build_external_research_agent(
        search_provider=provider,
    )

    result = await agent.ainvoke(
        {
            "question": "",
            "authorization_context": "test-user",
            "search_web_allowed": True,
        }
    )

    assert result["execution_allowed"] is False
    assert result["search_results"] == []
    assert result["evidence"] == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_external_research_handles_provider_failure():
    provider = FailingWebSearchProvider()

    agent = build_external_research_agent(
        search_provider=provider,
    )

    result = await agent.ainvoke(
        {
            "question": "Latest fraud trends",
            "authorization_context": "test-user",
            "search_web_allowed": True,
        }
    )

    assert result["execution_allowed"] is False
    assert result["search_results"] == []
    assert result["evidence"] == []
    assert "provider unavailable" in result["execution_reason"]


def test_external_research_rejects_invalid_max_results():
    provider = FakeWebSearchProvider()

    with pytest.raises(ValueError):
        build_external_research_agent(
            search_provider=provider,
            max_results=0,
        )