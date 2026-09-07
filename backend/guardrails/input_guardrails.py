import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

MAX_QUERY_LENGTH = 4000
MIN_QUERY_LENGTH = 2


# ============================================================
# Guardrail Result
# ============================================================


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    violations: list[str]


# ============================================================
# Prompt Injection Patterns
# ============================================================

PROMPT_INJECTION_PATTERNS = [
    # Previous / prior instruction manipulation
    r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
    r"\bignore\s+(?:all\s+)?prior\s+instructions\b",
    r"\bdisregard\s+(?:all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(?:all\s+)?prior\s+instructions\b",
    r"\bforget\s+(?:all\s+)?previous\s+instructions\b",
    r"\bforget\s+(?:all\s+)?prior\s+instructions\b",
    # System prompt extraction
    r"\breveal\s+(?:the\s+)?system\s+prompt\b",
    r"\bshow\s+(?:me\s+)?(?:the\s+)?system\s+prompt\b",
    r"\bprint\s+(?:the\s+)?system\s+prompt\b",
    r"\bdisplay\s+(?:the\s+)?system\s+prompt\b",
    # Hidden instruction extraction
    r"\breveal\s+(?:your\s+)?hidden\s+instructions\b",
    r"\bshow\s+(?:me\s+)?(?:your\s+)?hidden\s+instructions\b",
    r"\bprint\s+(?:your\s+)?hidden\s+instructions\b",
    r"\bdisplay\s+(?:your\s+)?hidden\s+instructions\b",
    # Guardrail manipulation
    r"\bbypass\s+(?:the\s+)?guardrails\b",
    r"\bdisable\s+(?:the\s+)?guardrails\b",
    r"\boverride\s+(?:the\s+)?guardrails\b",
    r"\bremove\s+(?:the\s+)?guardrails\b",
    # Safety manipulation
    r"\bbypass\s+(?:the\s+)?safety\s+(?:rules|controls|restrictions)\b",
    r"\bdisable\s+(?:the\s+)?safety\s+(?:rules|controls|restrictions)\b",
    # Privilege escalation / role manipulation
    r"\bact\s+as\s+(?:an?\s+)?admin\b",
    r"\bact\s+as\s+(?:an?\s+)?administrator\b",
    r"\bnow\s+you\s+are\s+(?:an?\s+)?admin\b",
    r"\bpretend\s+(?:to\s+be|you\s+are)\s+(?:an?\s+)?admin\b",
]


# ============================================================
# Unsafe Request Patterns
# ============================================================

UNSAFE_PATTERNS = [
    # Shell / operating-system commands
    r"\bexecute\s+(?:a\s+)?shell\s+command\b",
    r"\brun\s+(?:a\s+)?shell\s+command\b",
    r"\bexecute\s+(?:a\s+)?system\s+command\b",
    r"\brun\s+(?:a\s+)?system\s+command\b",
    # Code execution
    r"\bexecute\s+(?:python|code)\b",
    r"\brun\s+(?:python|code)\b",
    # Destructive DELETE operations
    r"\bdelete\s+.+\b(?:data|records|rows|tables)\b",
    r"\bdelete\s+.+\bfrom\s+(?:the\s+)?database\b",
    # Destructive DROP operations
    r"\bdrop\s+(?:the\s+)?(?:table|database|schema)\b",
    r"\bdrop\s+.+\btable\b",
    r"\bdrop\s+.+\bdatabase\b",
    r"\bdrop\s+.+\bschema\b",
    # TRUNCATE operations
    r"\btruncate\s+(?:the\s+)?(?:table|database)\b",
    r"\btruncate\s+.+\btable\b",
    # General destructive removal
    r"\bremove\s+(?:all\s+)?(?:data|records|rows|tables)\b",
]


# ============================================================
# Repeated Character Abuse
# ============================================================

REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{20,}")


# ============================================================
# Query Normalization
# ============================================================


def normalize_query(query: str) -> str:
    """
    Normalize user input before applying guardrail rules.

    - Converts input to string
    - Removes leading/trailing whitespace
    - Collapses repeated whitespace
    - Converts to lowercase
    """

    if query is None:
        return ""

    normalized = str(query).strip()

    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.lower()


# ============================================================
# Pattern Detection
# ============================================================


def contains_pattern(
    query: str,
    patterns: list[str],
) -> bool:
    """
    Returns True if any configured regex pattern
    matches the normalized query.
    """

    for pattern in patterns:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return True

    return False


# ============================================================
# Input Guardrail Validation
# ============================================================


def validate_user_query(query: str) -> GuardrailResult:
    """
    Validate a user query before it reaches the RAG,
    agent, SQL, or LLM layers.

    This is the first security boundary.

    Returns:
        GuardrailResult
    """

    violations: list[str] = []

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if query is None:
        logger.warning("Input guardrail blocked null query.")

        return GuardrailResult(
            allowed=False,
            reason="Query is required.",
            violations=["query_missing"],
        )

    normalized_query = normalize_query(query)

    # --------------------------------------------------------
    # Minimum length
    # --------------------------------------------------------

    if len(normalized_query) < MIN_QUERY_LENGTH:
        violations.append("query_too_short")

    # --------------------------------------------------------
    # Maximum length
    # --------------------------------------------------------

    elif len(normalized_query) > MAX_QUERY_LENGTH:
        violations.append("query_too_long")

    # --------------------------------------------------------
    # Prompt injection detection
    # --------------------------------------------------------

    if contains_pattern(
        normalized_query,
        PROMPT_INJECTION_PATTERNS,
    ):
        violations.append("prompt_injection")

    # --------------------------------------------------------
    # Unsafe request detection
    # --------------------------------------------------------

    if contains_pattern(
        normalized_query,
        UNSAFE_PATTERNS,
    ):
        violations.append("unsafe_request")

    # --------------------------------------------------------
    # Repeated character abuse
    # --------------------------------------------------------

    if REPEATED_CHARACTER_PATTERN.search(query):
        violations.append("repeated_character_abuse")

    # --------------------------------------------------------
    # Return blocked result
    # --------------------------------------------------------

    if violations:
        logger.warning(
            "Input guardrail blocked query. Violations=%s",
            violations,
        )

        return GuardrailResult(
            allowed=False,
            reason="The request was blocked by the input safety guardrails.",
            violations=violations,
        )

    # --------------------------------------------------------
    # Query allowed
    # --------------------------------------------------------

    return GuardrailResult(
        allowed=True,
        reason="Query passed input guardrails.",
        violations=[],
    )
