import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agents.security_graph import build_security_graph
from .authorization.policy import (
    AuthorizationPolicy,
)
from .config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
)
from .databricks_client import DatabricksVolumeClient


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Enterprise Fraud Intelligence & Governed RAG Platform",
    description=(
        "Enterprise fraud intelligence API with input guardrails, "
        "AI safety, authorization, and governed LangGraph agent routing."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = PROJECT_ROOT / "frontend"

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )


# ---------------------------------------------------------------------------
# Databricks client
# ---------------------------------------------------------------------------

try:
    databricks_client = DatabricksVolumeClient()

    logger.info("Databricks client initialized")

except Exception as error:
    logger.error(
        "Failed to initialize Databricks client: %s",
        error,
    )

    databricks_client = None


# ---------------------------------------------------------------------------
# Authorization Policy
#
# The security graph owns the actual authorization node.
# This instance is retained here for health reporting and future API-level
# authorization concerns.
# ---------------------------------------------------------------------------

authorization_policy = AuthorizationPolicy()

logger.info("Authorization policy initialized")


# ---------------------------------------------------------------------------
# LangGraph Security Workflow
# ---------------------------------------------------------------------------

try:
    security_graph = build_security_graph()

    logger.info(
        "LangGraph security workflow initialized"
    )

except Exception as error:
    logger.error(
        "Failed to initialize LangGraph security workflow: %s",
        error,
    )

    security_graph = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    user_id: str


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """
    Return application and security-component health status.

    LangGraph is now the owner of the security/routing workflow.
    """

    return {
        "status": "ok",
        "service": "enterprise-fraud-intelligence-platform",
        "security": {
            "input_guardrails": True,
            "model_safety": security_graph is not None,
            "authorization": authorization_policy is not None,
            "supervisor": security_graph is not None,
        },
        "databricks": {
            "client": databricks_client is not None,
        },
        "langgraph": {
            "security_graph": security_graph is not None,
        },
        "agents": {
            "status": "not_connected",
        },
        "rag": {
            "status": "not_connected",
        },
    }


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Serve the frontend if available.
    """

    index_file = FRONTEND_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {
        "service": "Enterprise Fraud Intelligence & Governed RAG Platform",
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Execute the LangGraph security and routing workflow.

    Current graph:

        START
          ↓
        Input Guardrails
          ↓
        AI Safety
          ↓
        Authentication
          ↓
        Authorization
          ↓
        Supervisor
          ↓
        Routing decision
          ↓
        END

    The graph currently stops after Supervisor routing.

    Agent execution, governed tools, RAG, MCP, context construction,
    response generation, and output guardrails will be connected in
    subsequent phases.
    """

    logger.info(
        "Received chat request for user_id=%s",
        request.user_id,
    )

    # -----------------------------------------------------------------------
    # LangGraph availability
    # -----------------------------------------------------------------------

    if security_graph is None:
        logger.error(
            "LangGraph security workflow is unavailable."
        )

        return {
            "stage": "security_graph",
            "allowed": False,
            "message": (
                "Security workflow is temporarily unavailable."
            ),
        }

    # -----------------------------------------------------------------------
    # Invoke LangGraph
    # -----------------------------------------------------------------------

    try:
        graph_result = security_graph.invoke(
            {
                "user_id": request.user_id,
                "query": request.query,
            }
        )

    except Exception as error:
        logger.error(
            "LangGraph execution failed: %s",
            error,
        )

        return {
            "stage": "security_graph",
            "allowed": False,
            "message": (
                "Security workflow could not be completed."
            ),
        }

    # -----------------------------------------------------------------------
    # Extract graph state
    # -----------------------------------------------------------------------

    allowed = graph_result.get(
        "allowed",
        False,
    )

    current_stage = graph_result.get(
        "current_stage",
        "security_graph",
    )

    # -----------------------------------------------------------------------
    # Input Guardrail result
    # -----------------------------------------------------------------------

    input_guardrail_allowed = graph_result.get(
        "input_guardrail_allowed",
        False,
    )

    input_guardrail_reason = graph_result.get(
        "input_guardrail_reason",
        "",
    )

    # -----------------------------------------------------------------------
    # AI Safety result
    # -----------------------------------------------------------------------

    safety_category = graph_result.get(
        "safety_category",
        "unknown",
    )

    safety_risk_level = graph_result.get(
        "safety_risk_level",
        "unknown",
    )

    safety_reason = graph_result.get(
        "safety_reason",
        "",
    )

    safety = {
        "category": safety_category,
        "risk_level": safety_risk_level,
        "detection_method": "groq_model_safety_classifier",
    }

    # -----------------------------------------------------------------------
    # Authorization result
    # -----------------------------------------------------------------------

    authorization_allowed = graph_result.get(
        "authorization_allowed",
        False,
    )

    authorization_reason = graph_result.get(
        "authorization_reason",
        "",
    )

    # -----------------------------------------------------------------------
    # Authentication failure
    # -----------------------------------------------------------------------

    if not graph_result.get(
        "authenticated",
        False,
    ):
        logger.warning(
            "Authentication failed for user_id=%s",
            request.user_id,
        )

        return {
            "stage": "authentication",
            "allowed": False,
            "message": "Authentication failed.",
        }

    # -----------------------------------------------------------------------
    # Input guardrail failure
    # -----------------------------------------------------------------------

    if not input_guardrail_allowed:
        logger.warning(
            "Input blocked by guardrails for user_id=%s",
            request.user_id,
        )

        return {
            "stage": "input_guardrails",
            "allowed": False,
            "message": (
                input_guardrail_reason
                or "Input validation failed."
            ),
        }

    # -----------------------------------------------------------------------
    # AI Safety failure
    # -----------------------------------------------------------------------

    if not graph_result.get(
        "safety_allowed",
        False,
    ):
        logger.warning(
            "Request blocked by AI Safety for user_id=%s; "
            "category=%s; risk_level=%s",
            request.user_id,
            safety_category,
            safety_risk_level,
        )

        return {
            "stage": "model_safety",
            "allowed": False,
            "message": (
                safety_reason
                or "Request blocked by AI safety policy."
            ),
            "safety": safety,
        }

    # -----------------------------------------------------------------------
    # Authorization failure
    # -----------------------------------------------------------------------

    if not authorization_allowed:
        logger.warning(
            "Authorization denied for user_id=%s",
            request.user_id,
        )

        return {
            "stage": "authorization",
            "allowed": False,
            "message": (
                authorization_reason
                or "Authorization denied."
            ),
        }

    # -----------------------------------------------------------------------
    # Supervisor result
    # -----------------------------------------------------------------------

    supervisor_decision = graph_result.get(
        "supervisor_decision"
    )

    supervisor_domain = graph_result.get(
        "supervisor_domain",
        "UNKNOWN",
    )

    supervisor_tools = graph_result.get(
        "supervisor_tools",
        [],
    )

    supervisor_reason = graph_result.get(
        "supervisor_reason",
        "",
    )

    supervisor_confidence = graph_result.get(
        "supervisor_confidence",
        0.0,
    )

    requires_policy = graph_result.get(
        "requires_policy",
        False,
    )

    requires_structured_data = graph_result.get(
        "requires_structured_data",
        False,
    )

    requires_external_research = graph_result.get(
        "requires_external_research",
        False,
    )

    # -----------------------------------------------------------------------
    # Supervisor UNKNOWN routing
    # -----------------------------------------------------------------------

    if supervisor_domain == "UNKNOWN":
        logger.warning(
            "Supervisor could not confidently route request "
            "for user_id=%s",
            request.user_id,
        )

        return {
            "stage": "supervisor",
            "allowed": False,
            "message": (
                "The request could not be confidently routed."
            ),
            "supervisor": {
                "domain": supervisor_domain,
                "tools": supervisor_tools,
                "reason": supervisor_reason,
                "confidence": supervisor_confidence,
            },
        }

    # -----------------------------------------------------------------------
    # Successful security + routing boundary
    # -----------------------------------------------------------------------

    logger.info(
        "LangGraph security workflow completed for user_id=%s; "
        "domain=%s; confidence=%.2f; tools=%s",
        request.user_id,
        supervisor_domain,
        supervisor_confidence,
        supervisor_tools,
    )

    return {
        "stage": "supervisor",
        "allowed": allowed,
        "message": (
            "Security checks passed and request was routed."
        ),

        # Next implementation stages.
        "next_stage": "agents",
        "agents_status": "not_connected",
        "rag_status": "not_connected",

        # AI Safety evidence.
        "safety": safety,

        # Supervisor routing decision.
        "supervisor": {
            "domain": supervisor_domain,
            "tools": supervisor_tools,
            "reason": supervisor_reason,
            "confidence": supervisor_confidence,
            "requires_policy": requires_policy,
            "requires_structured_data": (
                requires_structured_data
            ),
            "requires_external_research": (
                requires_external_research
            ),
        },
    }


# ---------------------------------------------------------------------------
# Document upload endpoint
# ---------------------------------------------------------------------------

@app.post("/api/documents/upload")
async def upload_document(file):
    """
    Upload a document to the configured Databricks volume.

    This endpoint remains intentionally simple for the current phase.
    """

    if databricks_client is None:
        raise HTTPException(
            status_code=503,
            detail="Databricks client is unavailable.",
        )

    try:
        contents = await file.read()

        result = databricks_client.upload_file(
            file_name=file.filename,
            file_content=contents,
        )

        return {
            "status": "uploaded",
            "file_name": file.filename,
            "result": result,
        }

    except Exception as error:
        logger.error(
            "Document upload failed: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Document upload failed.",
        ) from error


# ---------------------------------------------------------------------------
# Document listing endpoint
# ---------------------------------------------------------------------------

@app.get("/api/documents")
def list_documents():
    """
    List documents from the configured Databricks volume.
    """

    if databricks_client is None:
        raise HTTPException(
            status_code=503,
            detail="Databricks client is unavailable.",
        )

    try:
        documents = databricks_client.list_files()

        return {
            "documents": documents,
        }

    except Exception as error:
        logger.error(
            "Document listing failed: %s",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail="Document listing failed.",
        ) from error


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Prevent internal exception details from being returned to clients.
    """

    logger.exception(
        "Unhandled application exception: %s",
        exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error.",
            "message": "The request could not be completed.",
        },
    )


# ---------------------------------------------------------------------------
# Startup logging
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info(
        "Enterprise Fraud Intelligence Platform starting..."
    )

    logger.info(
        "Input Guardrails: enabled"
    )

    logger.info(
        "AI Safety: %s",
        "enabled" if security_graph else "unavailable",
    )

    logger.info(
        "Authorization: %s",
        "enabled" if authorization_policy else "unavailable",
    )

    logger.info(
        "LangGraph Security Workflow: %s",
        "enabled" if security_graph else "unavailable",
    )

    logger.info(
        "Agent Execution: not connected"
    )

    logger.info(
        "RAG: not connected"
    )


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

