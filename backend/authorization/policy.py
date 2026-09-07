from dataclasses import dataclass

# ============================================================
# Authorization Result
# ============================================================


@dataclass
class AuthorizationResult:
    allowed: bool
    reason: str
    violations: list[str]


# ============================================================
# User Identity
# ============================================================


@dataclass
class UserIdentity:
    user_id: str
    roles: set[str]
    departments: set[str]


# ============================================================
# Resource
# ============================================================


@dataclass
class Resource:
    resource_id: str
    resource_type: str
    classification: str
    departments: set[str]


# ============================================================
# Classification Levels
# ============================================================

CLASSIFICATION_LEVELS = {
    "public": 1,
    "internal": 2,
    "confidential": 3,
    "restricted": 4,
}


# ============================================================
# Authorization Policy
# ============================================================


class AuthorizationPolicy:
    """
    Local authorization policy engine.

    This is a prototype/production-foundation layer.
    Actual enterprise identity and Unity Catalog permissions
    will be integrated later.
    """

    def authorize(
        self,
        user: UserIdentity,
        resource: Resource,
    ) -> AuthorizationResult:

        violations: list[str] = []

        # ----------------------------------------------------
        # Validate user
        # ----------------------------------------------------

        if not user.user_id:
            violations.append("user_missing")

        # ----------------------------------------------------
        # Validate resource
        # ----------------------------------------------------

        if not resource.resource_id:
            violations.append("resource_missing")

        # ----------------------------------------------------
        # Validate classification
        # ----------------------------------------------------

        if resource.classification not in CLASSIFICATION_LEVELS:
            violations.append("invalid_classification")

        if violations:
            return AuthorizationResult(
                allowed=False,
                reason="Authorization request is invalid.",
                violations=violations,
            )

        # ----------------------------------------------------
        # Admin access
        # ----------------------------------------------------

        if "admin" in user.roles:
            return AuthorizationResult(
                allowed=True,
                reason="User has administrator access.",
                violations=[],
            )

        # ----------------------------------------------------
        # Restricted resources
        # ----------------------------------------------------

        if resource.classification == "restricted":
            if "fraud_investigator" not in user.roles:
                violations.append("restricted_resource")

        # ----------------------------------------------------
        # Confidential resources
        # ----------------------------------------------------

        if resource.classification == "confidential":
            if not (
                "fraud_investigator" in user.roles or "fraud_manager" in user.roles
            ):
                violations.append("confidential_resource")

        # ----------------------------------------------------
        # Department-level authorization
        # ----------------------------------------------------

        if resource.departments:
            if not user.departments.intersection(resource.departments):
                violations.append("department_access_denied")

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        if violations:
            return AuthorizationResult(
                allowed=False,
                reason="User is not authorized to access this resource.",
                violations=violations,
            )

        return AuthorizationResult(
            allowed=True,
            reason="User is authorized to access this resource.",
            violations=[],
        )
