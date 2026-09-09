from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List


def normalize_evidence(
    evidence: Dict[str, Any] | Iterable[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """
    Normalize evidence from Company RAG, Policy RAG, or Fraud MCP
    into a shared evidence contract.

    Existing evidence fields are preserved. The normalizer only adds
    shared fields required by the cross-domain Evidence Layer.

    Shared contract:

        evidence_id
        source_domain
        source
        retrieval_type
        retrieval_source
        tool
        relevance_score
        data
        metadata
    """

    if evidence is None:
        return []

    if isinstance(evidence, dict):
        items = [evidence]
    else:
        items = list(evidence)

    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        normalized.append(
            _normalize_single_evidence(
                evidence=item,
                index=index,
            )
        )

    return normalized


def _normalize_single_evidence(
    evidence: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Normalize one evidence record while preserving its existing structure.
    """

    result = deepcopy(evidence)

    metadata = result.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    data = result.get("data")

    if data is None:
        data = {}

    # Preserve the existing evidence_id when available.
    evidence_id = result.get("evidence_id")

    if not evidence_id:
        evidence_id = f"evidence-{index}"

    # Determine the domain.
    source_domain = _resolve_source_domain(
        evidence=result,
        metadata=metadata,
    )

    # Preserve existing source semantics.
    source = result.get("source")

    if not source:
        source = _resolve_source(
            evidence=result,
            metadata=metadata,
            source_domain=source_domain,
        )

    # Retrieval type.
    retrieval_type = result.get("retrieval_type")

    if not retrieval_type:
        retrieval_type = metadata.get("retrieval_type")

    if not retrieval_type:
        retrieval_type = _infer_retrieval_type(
            evidence=result,
            source_domain=source_domain,
        )

    # Retrieval source.
    retrieval_source = result.get("retrieval_source")

    if not retrieval_source:
        retrieval_source = metadata.get("retrieval_source")

    if not retrieval_source:
        retrieval_source = _infer_retrieval_source(
            evidence=result,
            source_domain=source_domain,
            retrieval_type=retrieval_type,
        )

    # Tool.
    tool = result.get("tool")

    if not tool:
        tool = _infer_tool(
            evidence=result,
            source_domain=source_domain,
        )

    # Optional relevance score.
    relevance_score = result.get("relevance_score")

    if relevance_score is None:
        relevance_score = _first_present(
            metadata,
            "relevance_score",
            "score",
            "rerank_score",
        )

    # Build the shared fields.
    result["evidence_id"] = evidence_id
    result["source_domain"] = source_domain
    result["source"] = source
    result["retrieval_type"] = retrieval_type
    result["retrieval_source"] = retrieval_source
    result["tool"] = tool
    result["relevance_score"] = relevance_score
    result["data"] = data
    result["metadata"] = metadata

    return result


def _resolve_source_domain(
    evidence: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    """
    Resolve the high-level knowledge/data domain.

    Supported domains:
        company
        policy
        fraud
        unknown
    """

    explicit_domain = _first_present(
        evidence,
        "source_domain",
        "knowledge_domain",
        "domain",
    )

    if explicit_domain:
        return str(explicit_domain)

    metadata_domain = _first_present(
        metadata,
        "source_domain",
        "knowledge_domain",
        "domain",
    )

    if metadata_domain:
        return str(metadata_domain)

    source = str(evidence.get("source", "")).lower()
    tool = str(evidence.get("tool", "")).lower()

    combined = f"{source} {tool}"

    if "policy" in combined:
        return "policy"

    if "company" in combined:
        return "company"

    if "fraud" in combined:
        return "fraud"

    # Fraud MCP evidence may use fraud_gold as its source.
    if "gold" in combined:
        return "fraud"

    return "unknown"


def _resolve_source(
    evidence: Dict[str, Any],
    metadata: Dict[str, Any],
    source_domain: str,
) -> str:
    """
    Resolve a source while preserving the existing source conventions.
    """

    explicit_source = _first_present(
        evidence,
        "source",
    )

    if explicit_source:
        return str(explicit_source)

    metadata_source = _first_present(
        metadata,
        "source",
    )

    if metadata_source:
        return str(metadata_source)

    if source_domain == "company":
        return "company_knowledge"

    if source_domain == "policy":
        return "policy_documents"

    if source_domain == "fraud":
        return "fraud_gold"

    return "unknown"


def _infer_retrieval_type(
    evidence: Dict[str, Any],
    source_domain: str,
) -> str:
    """
    Infer retrieval type when older evidence does not explicitly provide it.
    """

    source = str(evidence.get("source", "")).lower()
    tool = str(evidence.get("tool", "")).lower()

    combined = f"{source} {tool}"

    if source_domain == "fraud":
        return "structured_data"

    if "graph" in combined or "neo4j" in combined:
        return "graph"

    if "vector" in combined or "databricks" in combined:
        return "vector"

    if "mcp" in combined or "tool" in combined:
        return "structured_data"

    return "unknown"


def _infer_retrieval_source(
    evidence: Dict[str, Any],
    source_domain: str,
    retrieval_type: str,
) -> str:
    """
    Infer retrieval source from the existing evidence contract.
    """

    source = str(evidence.get("source", "")).lower()
    tool = str(evidence.get("tool", "")).lower()

    if source_domain == "company":
        if retrieval_type == "graph":
            return "neo4j_graph_rag"

        if retrieval_type == "vector":
            return "databricks_vector_search"

    if source_domain == "policy":
        if retrieval_type == "vector":
            return "databricks_vector_search"

    if source_domain == "fraud":
        if "mcp" in tool:
            return "fraud_mcp"

        if "fraud_gold" in source:
            return "databricks_gold"

        return "databricks_gold"

    if "neo4j" in source:
        return "neo4j_graph_rag"

    if "databricks" in source:
        return "databricks_vector_search"

    return "unknown"


def _infer_tool(
    evidence: Dict[str, Any],
    source_domain: str,
) -> str:
    """
    Infer a tool name only when the source evidence does not already contain one.
    """

    if source_domain == "company":
        return "company_knowledge_rag"

    if source_domain == "policy":
        return "policy_rag"

    if source_domain == "fraud":
        return "fraud_mcp"

    return "unknown"


def _first_present(
    mapping: Dict[str, Any],
    *keys: str,
) -> Any:
    """
    Return the first non-empty value for the supplied keys.
    """

    for key in keys:
        value = mapping.get(key)

        if value is not None and value != "":
            return value

    return None

