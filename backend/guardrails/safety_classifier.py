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
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from groq import Groq

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-20b"

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
    """

    SYSTEM_PROMPT = """
You are an enterprise AI safety classification system.

Your job is to classify the user's request for safety BEFORE the request
is sent to retrieval systems, databases, tools, agents, or a generative
answering model.

You must NOT answer the user's request.
You must ONLY classify the request.

Classify the user's intent using these categories:

- safe:
  Benign, legitimate, preventive, educational, compliance, fraud investigation,
  or defensive requests that do not seek harmful execution.

- weapons:
  Requests to construct, acquire, modify, or use weapons or explosives for harm.

- violence:
  Requests seeking instructions to physically harm, attack, injure, or kill
  another person.

- self_harm:
  Requests seeking instructions, methods, or optimization for self-injury
  or suicide.

- cyber_abuse:
  Requests seeking unauthorized access, credential theft, malware deployment,
  exploitation, evasion, persistence, data theft, or other malicious cyber
  activity.

- illegal_activity:
  Other serious unlawful wrongdoing that does not fit the more specific
  categories above.

- other:
  Requests that do not clearly fit the categories above.

Category selection rules:

1. Choose the MOST SPECIFIC category.
2. If a request involves unauthorized access to another person's computer,
   account, or system, classify it as cyber_abuse, not illegal_activity.
3. If a request involves malware, credential theft, exploitation, or malicious
   cyber activity, classify it as cyber_abuse.
4. Use illegal_activity only when the request is unlawful but does not fit
   weapons, violence, self_harm, or cyber_abuse.
5. Benign fraud investigation, fraud analytics, security analysis, compliance,
   prevention, and defensive security requests should normally be classified
   as safe.

Category priority when multiple categories could apply:

1. self_harm
2. weapons
3. cyber_abuse
4. violence
5. illegal_activity
6. safe
7. other

Block requests that seek actionable assistance for:
- constructing or using weapons or explosives for harm
- physically harming or killing people
- self-harm or suicide
- unauthorized or malicious cyber activity
- serious illegal wrongdoing

Allow legitimate enterprise, fraud investigation, compliance, educational,
preventive, and defensive requests when they do not seek harmful execution.

Return ONLY the required structured classification.
"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the Groq safety classifier.
        """

        self.api_key = api_key or os.getenv("GROQ_API_KEY")

        self.model = model or os.getenv("GROQ_SAFETY_MODEL", DEFAULT_MODEL)

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")

        self.client = Groq(api_key=self.api_key)

        logger.info(f"Groq safety classifier initialized with model={self.model}")

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(
        self,
        query: str,
    ) -> SafetyClassificationResult:
        """
        Evaluate a user query using the Groq LLM.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info("Running model-based safety evaluation.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0,
            max_tokens=200,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "safety_classification",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "allowed": {"type": "boolean"},
                            "category": {
                                "type": "string",
                                "enum": [
                                    "safe",
                                    "weapons",
                                    "violence",
                                    "self_harm",
                                    "cyber_abuse",
                                    "illegal_activity",
                                    "other",
                                ],
                            },
                            "risk_level": {
                                "type": "string",
                                "enum": [
                                    "low",
                                    "medium",
                                    "high",
                                ],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "allowed",
                            "category",
                            "risk_level",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Groq safety classifier returned empty output.")

        return self._parse_response(content)

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
            logger.error("Invalid JSON returned by Groq safety classifier.")

            raise ValueError("Safety classifier returned invalid JSON.") from error

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
            raise ValueError("Safety classifier 'allowed' must be boolean.")

        # ----------------------------------------------------
        # Validate category
        # ----------------------------------------------------

        category = data["category"]

        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid safety category: {category}")

        # ----------------------------------------------------
        # Validate risk level
        # ----------------------------------------------------

        risk_level = data["risk_level"]

        if risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"Invalid safety risk level: {risk_level}")

        # ----------------------------------------------------
        # Validate reason
        # ----------------------------------------------------

        reason = data["reason"]

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Safety classifier reason must be non-empty.")

        # ----------------------------------------------------
        # Safety consistency check
        # ----------------------------------------------------

        if data["allowed"]:
            if category != "safe":
                logger.warning(f"Model returned allowed=True with category={category}")

                raise ValueError("Inconsistent safety classification.")

        return SafetyClassificationResult(
            allowed=data["allowed"],
            category=category,
            risk_level=risk_level,
            reason=reason.strip(),
        )
