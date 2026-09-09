
from __future__ import annotations

import time

import pytest

from backend.rag.retrieval_cache import (
    RetrievalCache,
    build_retrieval_cache,
)


# ============================================================
# HELPERS
# ============================================================


def sample_evidence():
    return [
        {
            "evidence_id": "company-evidence-001",
            "source_domain": "company",
            "source": "neo4j_graph_rag",
            "tool": "company_knowledge_rag",
            "data": {
                "chunk_id": "COMP001",
                "chunk_text": "Company process information.",
            },
        }
    ]


# ============================================================
# CONSTRUCTION
# ============================================================


def test_retrieval_cache_requires_positive_ttl():

    with pytest.raises(ValueError):

        RetrievalCache(
            ttl_seconds=0
        )


def test_retrieval_cache_requires_namespace():

    with pytest.raises(ValueError):

        RetrievalCache(
            ttl_seconds=300,
            namespace="   ",
        )


def test_retrieval_cache_factory():

    cache = build_retrieval_cache(
        ttl_seconds=300,
        namespace="test",
    )

    assert isinstance(
        cache,
        RetrievalCache,
    )

    assert cache.ttl_seconds == 300
    assert cache.namespace == "test"


# ============================================================
# QUERY NORMALIZATION
# ============================================================


def test_normalize_query():

    result = RetrievalCache.normalize_query(
        "  What   is   corrective   action?  "
    )

    assert (
        result
        == "what is corrective action?"
    )


def test_normalize_query_rejects_non_string():

    with pytest.raises(TypeError):

        RetrievalCache.normalize_query(
            123
        )


# ============================================================
# CACHE MISS
# ============================================================


def test_cache_miss_returns_none():

    cache = RetrievalCache()

    result = cache.get(
        domain="company",
        query="What is corrective action?",
        authorization_context="fraud_investigator",
    )

    assert result is None


# ============================================================
# CACHE HIT
# ============================================================


def test_cache_hit_returns_cached_value():

    cache = RetrievalCache()

    evidence = sample_evidence()

    cached = cache.set(
        domain="company",
        query="What is corrective action?",
        authorization_context="fraud_investigator",
        value=evidence,
    )

    assert cached is True

    result = cache.get(
        domain="company",
        query="What is corrective action?",
        authorization_context="fraud_investigator",
    )

    assert result == evidence


# ============================================================
# QUERY NORMALIZATION CACHE HIT
# ============================================================


def test_normalized_queries_share_cache_entry():

    cache = RetrievalCache()

    evidence = sample_evidence()

    cache.set(
        domain="company",
        query="What   is   corrective action?",
        authorization_context="fraud_investigator",
        value=evidence,
    )

    result = cache.get(
        domain="company",
        query="  what is corrective action?  ",
        authorization_context="fraud_investigator",
    )

    assert result == evidence


# ============================================================
# DOMAIN ISOLATION
# ============================================================


def test_company_and_policy_domains_are_isolated():

    cache = RetrievalCache()

    company_evidence = [
        {
            "source_domain": "company",
            "data": "company evidence",
        }
    ]

    policy_evidence = [
        {
            "source_domain": "policy",
            "data": "policy evidence",
        }
    ]

    cache.set(
        domain="company",
        query="investigation procedure",
        authorization_context="fraud_investigator",
        value=company_evidence,
    )

    cache.set(
        domain="policy",
        query="investigation procedure",
        authorization_context="fraud_investigator",
        value=policy_evidence,
    )

    company_result = cache.get(
        domain="company",
        query="investigation procedure",
        authorization_context="fraud_investigator",
    )

    policy_result = cache.get(
        domain="policy",
        query="investigation procedure",
        authorization_context="fraud_investigator",
    )

    assert company_result == company_evidence
    assert policy_result == policy_evidence
    assert company_result != policy_result


# ============================================================
# AUTHORIZATION ISOLATION
# ============================================================


def test_authorization_context_isolation():

    cache = RetrievalCache()

    investigator_evidence = [
        {
            "data": "investigator-visible evidence",
        }
    ]

    manager_evidence = [
        {
            "data": "manager-visible evidence",
        }
    ]

    cache.set(
        domain="fraud",
        query="high value transactions",
        authorization_context="fraud_investigator",
        value=investigator_evidence,
    )

    cache.set(
        domain="fraud",
        query="high value transactions",
        authorization_context="fraud_manager",
        value=manager_evidence,
    )

    investigator_result = cache.get(
        domain="fraud",
        query="high value transactions",
        authorization_context="fraud_investigator",
    )

    manager_result = cache.get(
        domain="fraud",
        query="high value transactions",
        authorization_context="fraud_manager",
    )

    assert (
        investigator_result
        == investigator_evidence
    )

    assert (
        manager_result
        == manager_evidence
    )

    assert (
        investigator_result
        != manager_result
    )


def test_unknown_authorization_context_does_not_share_with_named_context():

    cache = RetrievalCache()

    evidence = sample_evidence()

    cache.set(
        domain="company",
        query="company procedure",
        authorization_context="fraud_investigator",
        value=evidence,
    )

    result = cache.get(
        domain="company",
        query="company procedure",
        authorization_context=None,
    )

    assert result is None


# ============================================================
# RETRIEVAL VERSION ISOLATION
# ============================================================


def test_retrieval_version_isolates_cache_entries():

    cache = RetrievalCache()

    old_evidence = [
        {
            "data": "old retrieval",
        }
    ]

    new_evidence = [
        {
            "data": "new retrieval",
        }
    ]

    cache.set(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        retrieval_version="v1",
        value=old_evidence,
    )

    cache.set(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        retrieval_version="v2",
        value=new_evidence,
    )

    assert cache.get(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        retrieval_version="v1",
    ) == old_evidence

    assert cache.get(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        retrieval_version="v2",
    ) == new_evidence


# ============================================================
# EMPTY RESULTS
# ============================================================


def test_empty_list_is_not_cached():

    cache = RetrievalCache()

    result = cache.set(
        domain="company",
        query="unknown procedure",
        authorization_context="employee",
        value=[],
    )

    assert result is False

    assert cache.get(
        domain="company",
        query="unknown procedure",
        authorization_context="employee",
    ) is None


def test_empty_tuple_is_not_cached():

    cache = RetrievalCache()

    result = cache.set(
        domain="company",
        query="unknown procedure",
        authorization_context="employee",
        value=(),
    )

    assert result is False


def test_none_is_not_cached():

    cache = RetrievalCache()

    result = cache.set(
        domain="company",
        query="unknown procedure",
        authorization_context="employee",
        value=None,
    )

    assert result is False


# ============================================================
# TTL
# ============================================================


def test_cache_entry_expires_after_ttl():

    cache = RetrievalCache(
        ttl_seconds=1
    )

    evidence = sample_evidence()

    cache.set(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        value=evidence,
    )

    assert cache.get(
        domain="company",
        query="corrective action",
        authorization_context="employee",
    ) == evidence

    time.sleep(1.1)

    assert cache.get(
        domain="company",
        query="corrective action",
        authorization_context="employee",
    ) is None


def test_custom_ttl_can_be_used():

    cache = RetrievalCache(
        ttl_seconds=300
    )

    evidence = sample_evidence()

    cache.set(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        value=evidence,
        ttl_seconds=1,
    )

    time.sleep(1.1)

    assert cache.get(
        domain="company",
        query="corrective action",
        authorization_context="employee",
    ) is None


# ============================================================
# INVALID TTL
# ============================================================


def test_set_rejects_invalid_ttl():

    cache = RetrievalCache()

    with pytest.raises(ValueError):

        cache.set(
            domain="company",
            query="corrective action",
            authorization_context="employee",
            value=sample_evidence(),
            ttl_seconds=0,
        )


# ============================================================
# INVALIDATE
# ============================================================


def test_invalidate_removes_entry():

    cache = RetrievalCache()

    evidence = sample_evidence()

    cache.set(
        domain="company",
        query="corrective action",
        authorization_context="employee",
        value=evidence,
    )

    removed = cache.invalidate(
        domain="company",
        query="corrective action",
        authorization_context="employee",
    )

    assert removed is True

    assert cache.get(
        domain="company",
        query="corrective action",
        authorization_context="employee",
    ) is None


def test_invalidate_returns_false_for_missing_entry():

    cache = RetrievalCache()

    result = cache.invalidate(
        domain="company",
        query="missing",
        authorization_context="employee",
    )

    assert result is False


# ============================================================
# CLEAR
# ============================================================


def test_clear_removes_all_entries():

    cache = RetrievalCache()

    cache.set(
        domain="company",
        query="company query",
        authorization_context="employee",
        value=sample_evidence(),
    )

    cache.set(
        domain="policy",
        query="policy query",
        authorization_context="employee",
        value=sample_evidence(),
    )

    assert cache.size() == 2

    cache.clear()

    assert cache.size() == 0


# ============================================================
# SIZE
# ============================================================


def test_size_reports_cached_entries():

    cache = RetrievalCache()

    assert cache.size() == 0

    cache.set(
        domain="company",
        query="query one",
        authorization_context="employee",
        value=sample_evidence(),
    )

    assert cache.size() == 1

    cache.set(
        domain="company",
        query="query two",
        authorization_context="employee",
        value=sample_evidence(),
    )

    assert cache.size() == 2


# ============================================================
# EXPIRED CLEANUP
# ============================================================


def test_cleanup_expired_removes_expired_entries():

    cache = RetrievalCache(
        ttl_seconds=1
    )

    cache.set(
        domain="company",
        query="query one",
        authorization_context="employee",
        value=sample_evidence(),
    )

    cache.set(
        domain="policy",
        query="query two",
        authorization_context="employee",
        value=sample_evidence(),
    )

    assert cache.size() == 2

    time.sleep(1.1)

    removed = cache.cleanup_expired()

    assert removed == 2
    assert cache.size() == 0


# ============================================================
# KEY GENERATION
# ============================================================


def test_same_inputs_generate_same_key():

    cache = RetrievalCache()

    key_one = cache.build_key(
        domain="company",
        query="Corrective Action",
        authorization_context="employee",
        retrieval_version="v1",
    )

    key_two = cache.build_key(
        domain="company",
        query="  corrective   action  ",
        authorization_context="employee",
        retrieval_version="v1",
    )

    assert key_one == key_two


def test_different_domains_generate_different_keys():

    cache = RetrievalCache()

    company_key = cache.build_key(
        domain="company",
        query="procedure",
        authorization_context="employee",
    )

    policy_key = cache.build_key(
        domain="policy",
        query="procedure",
        authorization_context="employee",
    )

    assert company_key != policy_key


def test_different_authorization_contexts_generate_different_keys():

    cache = RetrievalCache()

    investigator_key = cache.build_key(
        domain="fraud",
        query="transactions",
        authorization_context="fraud_investigator",
    )

    manager_key = cache.build_key(
        domain="fraud",
        query="transactions",
        authorization_context="fraud_manager",
    )

    assert investigator_key != manager_key


# ============================================================
# INVALID INPUT
# ============================================================


def test_build_key_rejects_empty_domain():

    cache = RetrievalCache()

    with pytest.raises(ValueError):

        cache.build_key(
            domain="   ",
            query="procedure",
        )


def test_build_key_rejects_empty_query():

    cache = RetrievalCache()

    with pytest.raises(ValueError):

        cache.build_key(
            domain="company",
            query="   ",
        )


def test_build_key_rejects_non_string_domain():

    cache = RetrievalCache()

    with pytest.raises(TypeError):

        cache.build_key(
            domain=123,
            query="procedure",
        )


def test_build_key_rejects_empty_retrieval_version():

    cache = RetrievalCache()

    with pytest.raises(ValueError):

        cache.build_key(
            domain="company",
            query="procedure",
            retrieval_version="   ",
        )


# ============================================================
# AUTHORIZATION CONTEXT COLLECTIONS
# ============================================================


def test_authorization_context_collection_is_deterministic():

    cache = RetrievalCache()

    key_one = cache.build_key(
        domain="company",
        query="procedure",
        authorization_context={
            "fraud",
            "investigator",
        },
    )

    key_two = cache.build_key(
        domain="company",
        query="procedure",
        authorization_context={
            "investigator",
            "fraud",
        },
    )

    assert key_one == key_two

