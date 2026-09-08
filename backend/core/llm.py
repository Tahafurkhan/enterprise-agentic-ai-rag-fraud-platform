
"""
Shared Application LLM

This module provides the application LLM used by:
- Supervisor
- Company Knowledge RAG
- Policy RAG
- Fraud Agent response generation
- Response Generator
- Self-RAG
- LLM-as-Judge

IMPORTANT:
The AI Safety classifier is intentionally NOT configured here.
It remains a separate security component.

Architecture:

    Agents
       |
       v
    core.llm
       |
       v
    Application LLM Gateway
       |
       +---- Groq (development)
       |
       +---- Databricks Unity Gateway (production)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq


# ================================================================
# LOAD ENVIRONMENT
# ================================================================

# Project root:
# enterprise-agentic-ai-rag-fraud-platform/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


# ================================================================
# BUILD APPLICATION LLM
# ================================================================

def build_application_llm() -> BaseChatModel:
    """
    Build the shared application LLM.

    The provider is selected through environment variables.

    Current development provider:
        Groq

    Future production provider:
        Databricks Unity AI Gateway
    """

    provider = os.getenv(
        "LLM_PROVIDER",
        "groq",
    ).strip().lower()

    model_name = os.getenv(
        "APPLICATION_LLM_MODEL",
        "llama-3.1-8b-instant",
    ).strip()

    temperature = float(
        os.getenv(
            "APPLICATION_LLM_TEMPERATURE",
            "0.0",
        )
    )

    # ============================================================
    # GROQ
    # ============================================================

    if provider == "groq":

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured.\n"
                f"Expected .env file at: {ENV_FILE}\n"
                "Add GROQ_API_KEY to the .env file."
            )

        api_key = api_key.strip()

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is empty."
            )

        return ChatGroq(
            model=model_name,
            temperature=temperature,
            groq_api_key=api_key,
        )

    # ============================================================
    # FUTURE: DATABRICKS UNITY AI GATEWAY
    # ============================================================

    if provider == "databricks":

        raise NotImplementedError(
            "Databricks Unity AI Gateway provider is not "
            "configured yet. Keep LLM_PROVIDER=groq "
            "during development."
        )

    # ============================================================
    # UNKNOWN PROVIDER
    # ============================================================

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Supported providers: groq, databricks."
    )


# ================================================================
# SHARED APPLICATION LLM
# ================================================================

llm = build_application_llm()
