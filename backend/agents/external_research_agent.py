"""
External Research Agent.

The External Research Agent performs governed external web research
through an injected WebSearchProvider.

In this project:

    External Research Agent
            ↓
       TavilyMCPClient
            ↓
       Tavily MCP Server
            ↓
           Tavily
            ↓
        Web Results

The agent itself does not call Tavily directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph


# ============================================================
# WEB SEARCH PROVIDER
# ============================================================

class WebSearchProvider(Protocol):
    """
    Provider interface for external web search.

    The External Research Agent remains provider-neutral.
    """

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        ...


# ============================================================
# STATE
# ============================================================

class ExternalResearchState(TypedDict, total=False):
    """
    State for the External Research Agent.
    """

    question: str
    current_query: str

    authorization_context: str

    search_results: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]

    search_web_allowed: bool

    execution_allowed: bool
    execution_reason: str

    max_results: int

    # Shared security-graph execution fields.
    allowed: bool
    error: str


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def _normalize_search_result(
    result: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Convert a provider-specific web result into the platform's
    common Evidence representation.
    """

    content = result.get(
        "content",
        "",
    )

    snippet = result.get(
        "snippet",
        content,
    )

    return {
        "evidence_id": (
            f"web-evidence-{index}"
        ),

        "source": "external_web",

        "tool": "web_search",

        "retrieval_type": "web_search",

        "data": {
            "title": result.get(
                "title",
                "",
            ),
            "url": result.get(
                "url",
                "",
            ),
            "snippet": snippet,
            "content": content,
        },

        "metadata": {
            "provider": result.get(
                "provider",
            ),
            "published_at": result.get(
                "published_at",
            ),
            "author": result.get(
                "author",
            ),
            "query": result.get(
                "query",
            ),
            "result_id": result.get(
                "id",
            ),
        },
    }


# ============================================================
# AGENT BUILDER
# ============================================================

def build_external_research_agent(
    search_provider: WebSearchProvider,
    *,
    max_results: int = 5,
):
    """
    Build the External Research Agent.

    Parameters
    ----------
    search_provider:
        Injected web-search provider.

    max_results:
        Maximum number of web results returned by default.
    """

    if search_provider is None:
        raise ValueError(
            "search_provider is required."
        )

    if max_results < 1:
        raise ValueError(
            "max_results must be at least 1."
        )

    # ========================================================
    # SCOPE NODE
    # ========================================================

    async def scope_node(
        state: ExternalResearchState,
    ) -> Dict[str, Any]:
        """
        Validate that external research is authorized.
        """

        authorization_context = (
            state.get(
                "authorization_context",
                "",
            )
            or ""
        ).strip()

        search_web_allowed = bool(
            state.get(
                "search_web_allowed",
                False,
            )
        )

        # ----------------------------------------------------
        # Authorization context
        # ----------------------------------------------------

        if not authorization_context:
            return {
                "allowed": False,
                "execution_allowed": False,
                "execution_reason": (
                    "External research requires "
                    "authorization context."
                ),
                "error": (
                    "Missing authorization context."
                ),
            }

        # ----------------------------------------------------
        # External web permission
        # ----------------------------------------------------

        if not search_web_allowed:
            return {
                "allowed": False,
                "execution_allowed": False,
                "execution_reason": (
                    "External web search is not "
                    "authorized for this request."
                ),
                "error": (
                    "External web search is not authorized."
                ),
            }

        # ----------------------------------------------------
        # Authorized
        # ----------------------------------------------------

        return {
            "allowed": True,
            "execution_allowed": True,
            "execution_reason": (
                "External web research authorized."
            ),
            "error": "",
        }

    # ========================================================
    # SEARCH NODE
    # ========================================================

    async def search_node(
        state: ExternalResearchState,
    ) -> Dict[str, Any]:
        """
        Execute the approved external web search.
        """

        # ----------------------------------------------------
        # Verify execution authorization
        # ----------------------------------------------------

        if not state.get(
            "execution_allowed",
            False,
        ):
            return {
                "allowed": False,
                "execution_allowed": False,
                "execution_reason": state.get(
                    "execution_reason",
                    "External search is not allowed.",
                ),
                "error": state.get(
                    "error",
                    "External search is not allowed.",
                ),
                "search_results": [],
                "evidence": [],
            }

        # ----------------------------------------------------
        # Determine query
        # ----------------------------------------------------

        query = (
            state.get(
                "current_query",
                "",
            )
            or state.get(
                "question",
                "",
            )
            or ""
        ).strip()

        if not query:
            return {
                "allowed": False,
                "execution_allowed": False,
                "execution_reason": (
                    "External search requires a query."
                ),
                "error": (
                    "External search query is empty."
                ),
                "search_results": [],
                "evidence": [],
            }

        # ----------------------------------------------------
        # Determine result count
        # ----------------------------------------------------

        requested_max_results = state.get(
            "max_results",
            max_results,
        )

        try:
            requested_max_results = int(
                requested_max_results
            )

        except (
            TypeError,
            ValueError,
        ):
            requested_max_results = max_results

        # Defensive limit.

        requested_max_results = max(
            1,
            min(
                requested_max_results,
                20,
            ),
        )

        # ----------------------------------------------------
        # Execute provider
        # ----------------------------------------------------

        try:
            results = await search_provider.search(
                query,
                max_results=requested_max_results,
            )

            if not isinstance(
                results,
                list,
            ):
                raise TypeError(
                    "Web search provider returned "
                    "an invalid result type."
                )

            # Only accept dictionaries as normalized provider
            # records.

            valid_results = [
                result
                for result in results
                if isinstance(
                    result,
                    dict,
                )
            ]

            # ------------------------------------------------
            # Convert to shared Evidence
            # ------------------------------------------------

            evidence = [
                _normalize_search_result(
                    result,
                    index,
                )
                for index, result in enumerate(
                    valid_results,
                    start=1,
                )
            ]

            return {
                "allowed": True,
                "execution_allowed": True,
                "execution_reason": (
                    "External web search completed."
                ),
                "error": "",
                "search_results": valid_results,
                "evidence": evidence,
                "current_query": query,
            }

        except Exception as exc:

            return {
                "allowed": False,
                "execution_allowed": False,
                "execution_reason": (
                    "External web search failed."
                ),
                "error": str(exc),
                "search_results": [],
                "evidence": [],
                "current_query": query,
            }

    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    builder = StateGraph(
        ExternalResearchState
    )

    builder.add_node(
        "scope",
        scope_node,
    )

    builder.add_node(
        "search",
        search_node,
    )

    builder.add_edge(
        START,
        "scope",
    )

    builder.add_edge(
        "scope",
        "search",
    )

    builder.add_edge(
        "search",
        END,
    )

    return builder.compile()