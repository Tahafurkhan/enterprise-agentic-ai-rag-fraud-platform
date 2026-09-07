import sys
from pathlib import Path

# ============================================================
# Test Import Configuration
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from authorization.policy import (
    AuthorizationPolicy,
    Resource,
    UserIdentity,
)

# ============================================================
# Test Fixtures / Helpers
# ============================================================


def create_policy():
    return AuthorizationPolicy()


def create_investigator():
    return UserIdentity(
        user_id="user-001",
        roles={"fraud_investigator"},
        departments={"fraud"},
    )


def create_fraud_manager():
    return UserIdentity(
        user_id="user-002",
        roles={"fraud_manager"},
        departments={"fraud"},
    )


def create_admin():
    return UserIdentity(
        user_id="admin-001",
        roles={"admin"},
        departments={"fraud"},
    )


def create_fraud_resource(
    classification="confidential",
):
    return Resource(
        resource_id="document-001",
        resource_type="document",
        classification=classification,
        departments={"fraud"},
    )


# ============================================================
# Authorized Access Tests
# ============================================================


def test_fraud_investigator_can_access_confidential_resource():
    policy = create_policy()

    user = create_investigator()
    resource = create_fraud_resource("confidential")

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


def test_fraud_manager_can_access_confidential_resource():
    policy = create_policy()

    user = create_fraud_manager()
    resource = create_fraud_resource("confidential")

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


def test_admin_can_access_restricted_resource():
    policy = create_policy()

    user = create_admin()
    resource = create_fraud_resource("restricted")

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


# ============================================================
# Restricted Resource Tests
# ============================================================


def test_fraud_investigator_can_access_restricted_resource():
    policy = create_policy()

    user = create_investigator()
    resource = create_fraud_resource("restricted")

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


def test_fraud_manager_cannot_access_restricted_resource():
    policy = create_policy()

    user = create_fraud_manager()
    resource = create_fraud_resource("restricted")

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "restricted_resource" in result.violations


# ============================================================
# Role-Based Access Tests
# ============================================================


def test_regular_employee_cannot_access_confidential_resource():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-003",
        roles={"employee"},
        departments={"fraud"},
    )

    resource = create_fraud_resource("confidential")

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "confidential_resource" in result.violations


def test_regular_employee_cannot_access_restricted_resource():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-004",
        roles={"employee"},
        departments={"fraud"},
    )

    resource = create_fraud_resource("restricted")

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "restricted_resource" in result.violations


# ============================================================
# Department Access Tests
# ============================================================


def test_user_from_correct_department_can_access_resource():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-005",
        roles={"fraud_investigator"},
        departments={"fraud"},
    )

    resource = Resource(
        resource_id="fraud-policy-001",
        resource_type="document",
        classification="confidential",
        departments={"fraud"},
    )

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


def test_user_from_wrong_department_is_denied():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-006",
        roles={"fraud_investigator"},
        departments={"finance"},
    )

    resource = Resource(
        resource_id="fraud-policy-002",
        resource_type="document",
        classification="confidential",
        departments={"fraud"},
    )

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "department_access_denied" in result.violations


# ============================================================
# Public / Internal Resource Tests
# ============================================================


def test_employee_can_access_public_resource():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-007",
        roles={"employee"},
        departments={"finance"},
    )

    resource = Resource(
        resource_id="public-001",
        resource_type="document",
        classification="public",
        departments=set(),
    )

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


def test_employee_can_access_internal_resource():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-008",
        roles={"employee"},
        departments={"finance"},
    )

    resource = Resource(
        resource_id="internal-001",
        resource_type="document",
        classification="internal",
        departments=set(),
    )

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert result.violations == []


# ============================================================
# Invalid Input Tests
# ============================================================


def test_missing_user_is_denied():
    policy = create_policy()

    user = UserIdentity(
        user_id="",
        roles={"employee"},
        departments={"fraud"},
    )

    resource = create_fraud_resource()

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "user_missing" in result.violations


def test_missing_resource_is_denied():
    policy = create_policy()

    user = create_investigator()

    resource = Resource(
        resource_id="",
        resource_type="document",
        classification="confidential",
        departments={"fraud"},
    )

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "resource_missing" in result.violations


def test_invalid_classification_is_denied():
    policy = create_policy()

    user = create_investigator()

    resource = Resource(
        resource_id="document-invalid",
        resource_type="document",
        classification="top_secret_unknown",
        departments={"fraud"},
    )

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert "invalid_classification" in result.violations


# ============================================================
# Result Structure Tests
# ============================================================


def test_authorized_result_contains_reason():
    policy = create_policy()

    user = create_investigator()
    resource = create_fraud_resource()

    result = policy.authorize(user, resource)

    assert result.allowed is True
    assert isinstance(result.reason, str)
    assert result.reason != ""


def test_denied_result_contains_reason():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-009",
        roles={"employee"},
        departments={"finance"},
    )

    resource = create_fraud_resource("restricted")

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert isinstance(result.reason, str)
    assert result.reason != ""


def test_denied_result_contains_violations():
    policy = create_policy()

    user = UserIdentity(
        user_id="user-010",
        roles={"employee"},
        departments={"finance"},
    )

    resource = create_fraud_resource("confidential")

    result = policy.authorize(user, resource)

    assert result.allowed is False
    assert isinstance(result.violations, list)
    assert len(result.violations) > 0
