
"""
Enterprise RAG - Model-Based Safety Classifier
===============================================

Uses a Groq-hosted LLM to evaluate user queries for safety
before the query reaches:

    - Authentication
    - Authorization
    - RAG retrieval
    - SQL tools
    - Agentic tools
    - Generative answering models

IMPORTANT:

This model is a CLASSIFIER.

It does not answer the user's request.

It returns only a structured safety decision.

The classifier also contains a deterministic pre-classification
layer for unmistakable harmful requests. This prevents the safety
boundary from depending on an external model successfully generating
structured JSON for requests that the provider may refuse to answer.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from groq import Groq

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-safeguard-20b"

VALID_CATEGORIES = {
    "safe",
    "weapons",
    "violence",
    "self_harm",
    "cyber_abuse",
    "illegal_activity",
    "other",
}

VALID_RISK_LEVELS = {
    "low",
    "medium",
    "high",
}


# ============================================================
# Result
# ============================================================


@dataclass
class SafetyClassificationResult:
    allowed: bool
    category: str
    risk_level: str
    reason: str


# ============================================================
# Safety Classifier
# ============================================================


class ModelSafetyClassifier:
    """
    Groq-based model safety classifier.

    The model evaluates REQUEST INTENT.

    It does not answer the request.

    Explicit harmful requests are blocked deterministically before
    contacting the model. All other requests are classified by Groq
    and then validated locally.
    """

    SYSTEM_PROMPT = """
# INSTRUCTIONS

Classify the user's request for enterprise AI safety.

Do NOT answer the request.
Return ONLY one JSON object.

Required JSON fields:
{
  "allowed": true,
  "category": "safe",
  "risk_level": "low",
  "reason": "Brief explanation."
}

# DEFINITIONS

safe:
Legitimate enterprise, fraud analytics, investigation, compliance,
education, prevention, or defensive security requests.

weapons:
Requests for constructing, acquiring, modifying, or using weapons
or explosives for harm.

violence:
Requests for instructions to physically harm, attack, injure, or kill
another person.

self_harm:
Requests for instructions or methods for self-injury or suicide.

cyber_abuse:
Requests for unauthorized access, credential theft, malware,
exploitation, evasion, persistence, or malicious cyber activity.

illegal_activity:
Serious unlawful activity that does not fit the categories above.

other:
Requests that do not clearly fit another category.

# CRITERIA

Allow safe enterprise and defensive requests.

Block actionable requests involving weapons, violence, self-harm,
malicious cyber activity, or serious illegal wrongdoing.

Use the most specific category.

For allowed requests:
allowed=true, category=safe, risk_level=low.

For blocked requests:
allowed=false, use the applicable harmful category,
risk_level=high.

# EXAMPLES

User: Show me high-value fraud transactions.
Output: {"allowed":true,"category":"safe","risk_level":"low","reason":"Legitimate fraud analytics request."}

User: How do I make a bomb?
Output: {"allowed":false,"category":"weapons","risk_level":"high","reason":"The request seeks actionable assistance involving explosives."}

User: How do I break into someone's computer?
Output: {"allowed":false,"category":"cyber_abuse","risk_level":"high","reason":"The request seeks unauthorized cyber access."}

Now classify the user's request.
"""

    # ========================================================
    # Explicit harmful intent patterns
    # ========================================================

    # These patterns intentionally target clear actionable harmful
    # requests. They are NOT intended to classify ordinary discussion
    # of fraud, security, compliance, safety, or education.
    DETERMINISTIC_BLOCK_PATTERNS = (
        (
            "weapons",
            (
                re.compile(r"\bhow\s+do\s+i\s+make\s+a\s+bomb\b", re.I),
                re.compile(r"\bhow\s+do\s+i\s+make\s+an?\s+explosive\s+device\b", re.I),
                re.compile(r"\bmake\s+(?:a|an)\s+(?:bomb|explosive)\b", re.I),
            ),
        ),
        (
            "violence",
            (
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+(?:seriously\s+)?hurt\s+someone"
                    r"(?:\s+without\s+getting\s+caught)?\b",
                    re.I,
                ),
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+(?:kill|attack|injure)\s+someone\b",
                    re.I,
                ),
            ),
        ),
        (
            "cyber_abuse",
            (
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+break\s+into\s+someone'?s\s+computer\b",
                    re.I,
                ),
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+hack\s+into\s+someone'?s\s+computer\b",
                    re.I,
                ),
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+access\s+someone'?s\s+computer"
                    r".*without\s+(?:them\s+)?knowing\b",
                    re.I,
                ),
            ),
        ),
        (
            "self_harm",
            (
                re.compile(
                    r"\bmost\s+effective\s+way\s+to\s+hurt\s+myself\b",
                    re.I,
                ),
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+hurt\s+myself\b",
                    re.I,
                ),
                re.compile(
                    r"\bhow\s+(?:can|do)\s+i\s+kill\s+myself\b",
                    re.I,
                ),
            ),
        ),
    )

    DETERMINISTIC_REASONS = {
        "weapons": (
            "The request seeks actionable assistance involving weapons "
            "or explosives."
        ),
        "violence": (
            "The request seeks actionable assistance to physically harm "
            "another person."
        ),
        "cyber_abuse": (
            "The request seeks unauthorized access or malicious cyber activity."
        ),
        "self_harm": (
            "The request seeks actionable assistance for self-harm."
        ),
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the Groq safety classifier.
        """

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        self.model = model or os.getenv(
            "GROQ_SAFETY_MODEL",
            DEFAULT_MODEL,
        )

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=self.api_key)

        logger.info(
            "Groq safety classifier initialized with model=%s",
            self.model,
        )

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(
        self,
        query: str,
    ) -> SafetyClassificationResult:
        """
        Evaluate a user query using the safety pipeline.

        Explicit harmful requests are blocked before contacting Groq.
        Other requests are evaluated by the Groq classifier.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        normalized_query = " ".join(query.strip().split())

        logger.info("Running model-based safety evaluation.")

        # ----------------------------------------------------
        # Deterministic explicit-harm check
        # ----------------------------------------------------

        deterministic_result = self._deterministic_classification(
            normalized_query
        )

        if deterministic_result is not None:
            logger.warning(
                "Explicit harmful request blocked before model classification."
            )

            return deterministic_result

        # ----------------------------------------------------
        # Model-based classification
        # ----------------------------------------------------

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": normalized_query,
                },
            ],
            temperature=0,
            max_tokens=1024,
            include_reasoning=False,
        )

        message = response.choices[0].message

        logger.info(
            "Groq safety response received. model=%s content_present=%s",
            self.model,
            bool(message.content),
        )

        content = message.content

        if not content:
            raise ValueError(
                "Groq safety classifier returned empty output."
            )

        return self._parse_response(content)

    # ========================================================
    # DETERMINISTIC CLASSIFICATION
    # ========================================================

    def _deterministic_classification(
        self,
        query: str,
    ) -> SafetyClassificationResult | None:
        """
        Block only unmistakable harmful requests.

        This layer is intentionally narrow. It does not attempt to
        replace the model-based classifier for ambiguous requests.
        """

        for category, patterns in self.DETERMINISTIC_BLOCK_PATTERNS:
            for pattern in patterns:
                if pattern.search(query):
                    return SafetyClassificationResult(
                        allowed=False,
                        category=category,
                        risk_level="high",
                        reason=self.DETERMINISTIC_REASONS[category],
                    )

        return None

    # ========================================================
    # VALIDATE MODEL OUTPUT
    # ========================================================

    def _parse_response(
        self,
        content: str,
    ) -> SafetyClassificationResult:
        """
        Validate the model response before the application
        trusts the safety decision.
        """

        try:
            data: dict[str, Any] = json.loads(content)

        except json.JSONDecodeError as error:
            logger.error(
                "Invalid JSON returned by Groq safety classifier."
            )

            raise ValueError(
                "Safety classifier returned invalid JSON."
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                "Safety classifier response must be a JSON object."
            )

        required_fields = {
            "allowed",
            "category",
            "risk_level",
            "reason",
        }

        missing_fields = required_fields - set(data.keys())

        if missing_fields:
            raise ValueError(
                "Safety classifier response is missing "
                f"fields: {sorted(missing_fields)}"
            )

        # ----------------------------------------------------
        # Validate allowed
        # ----------------------------------------------------

        if not isinstance(data["allowed"], bool):
            raise ValueError(
                "Safety classifier 'allowed' must be boolean."
            )

        # ----------------------------------------------------
        # Validate category
        # ----------------------------------------------------

        category = data["category"]

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid safety category: {category}"
            )

        # ----------------------------------------------------
        # Validate risk level
        # ----------------------------------------------------

        risk_level = data["risk_level"]

        if risk_level not in VALID_RISK_LEVELS:
            raise ValueError(
                f"Invalid safety risk level: {risk_level}"
            )

        # ----------------------------------------------------
        # Validate reason
        # ----------------------------------------------------

        reason = data["reason"]

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "Safety classifier reason must be non-empty."
            )

        # ----------------------------------------------------
        # Safety consistency check
        # ----------------------------------------------------

        if data["allowed"]:
            if category != "safe":
                logger.warning(
                    "Model returned allowed=True with category=%s",
                    category,
                )

                raise ValueError(
                    "Inconsistent safety classification."
                )

        return SafetyClassificationResult(
            allowed=data["allowed"],
            category=category,
            risk_level=risk_level,
            reason=reason.strip(),
        )

