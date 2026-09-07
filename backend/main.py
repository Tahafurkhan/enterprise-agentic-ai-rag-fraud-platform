"""
Enterprise RAG API - FastAPI Backend
=====================================

Handles:

- Document uploads
- Document listing
- Health checks
- Frontend serving
- Knowledge Copilot security flow
- Deterministic input guardrails
- Model-based LLM safety guardrails
- Authentication
- Authorization
- Databricks integration

Current /api/chat security flow:

    User Query
         ↓
    Existing Input Guardrails
         ↓
    Groq LLM Safety Classifier
         ↓
    Authentication
         ↓
    Authorization
         ↓
    Security Boundary
         ↓
    RAG  <-- next implementation stage

RAG retrieval, AI Search, LLM generation, and agentic tools
are intentionally NOT connected yet.
"""

import logging
from pathlib import Path

from authorization.policy import (
    AuthorizationPolicy,
    Resource,
    UserIdentity,
)
from config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
)
from databricks_client import DatabricksVolumeClient
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
)
from fastapi.staticfiles import StaticFiles
from guardrails.input_guardrails import (
    validate_user_query,
)
from guardrails.safety_classifier import (
    ModelSafetyClassifier,
)
from pydantic import BaseModel

# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

logger.info(
    "Base directory: %s",
    BASE_DIR,
)

logger.info(
    "Frontend directory: %s",
    FRONTEND_DIR,
)


if not FRONTEND_DIR.exists():
    logger.warning(
        "Frontend directory not found: %s",
        FRONTEND_DIR,
    )

else:
    logger.info(
        "Frontend files: %s",
        list(FRONTEND_DIR.glob("*")),
    )


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description=("Enterprise Fraud Intelligence and Governed RAG Platform API"),
)


# =========================================================
# CORS MIDDLEWARE
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured")


# =========================================================
# DATABRICKS CLIENT
# =========================================================

try:
    databricks_client = DatabricksVolumeClient()

    logger.info("Databricks client initialized")

except Exception as error:
    logger.error(
        "Failed to initialize Databricks client: %s",
        error,
    )

    databricks_client = None


# =========================================================
# MODEL-BASED SAFETY CLASSIFIER
# =========================================================

try:
    safety_classifier = ModelSafetyClassifier()

    logger.info("Groq model-based safety classifier initialized")

except Exception as error:
    logger.error(
        "Failed to initialize Groq safety classifier: %s",
        error,
    )

    safety_classifier = None


# =========================================================
# AUTHORIZATION POLICY
# =========================================================

authorization_policy = AuthorizationPolicy()


# =========================================================
# TEST USERS
# =========================================================
#
# IMPORTANT:
# These identities are ONLY for local development/testing.
#
# Production will replace this with:
#
# Enterprise Identity Provider
#        ↓
# Authentication
#        ↓
# Groups / Roles / Attributes
#        ↓
# Authorization
#
# Do NOT use hardcoded users in production.
# =========================================================

TEST_USERS = {
    "user-001": UserIdentity(
        user_id="user-001",
        roles={"fraud_investigator"},
        departments={"fraud"},
    ),
    "user-002": UserIdentity(
        user_id="user-002",
        roles={"fraud_manager"},
        departments={"fraud"},
    ),
    "user-003": UserIdentity(
        user_id="user-003",
        roles={"employee"},
        departments={"finance"},
    ),
    "admin-001": UserIdentity(
        user_id="admin-001",
        roles={"admin"},
        departments={"fraud"},
    ),
}


# =========================================================
# TEMPORARY KNOWLEDGE RESOURCE
# =========================================================

FRAUD_KNOWLEDGE_RESOURCE = Resource(
    resource_id="fraud-knowledge-base",
    resource_type="knowledge_base",
    classification="confidential",
    departments={"fraud"},
)


# =========================================================
# CHAT REQUEST MODEL
# =========================================================


class ChatRequest(BaseModel):
    query: str

    user_id: str


# =========================================================
# STATIC FILES
# =========================================================

if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )

    logger.info("Static files mounted at /static")

else:
    logger.warning("Frontend directory does not exist - static files not mounted")


# =========================================================
# FRONTEND
# =========================================================


@app.get(
    "/",
    include_in_schema=False,
)
async def serve_frontend():
    """
    Serve the main frontend application.
    """

    index_path = FRONTEND_DIR / "index.html"

    if not index_path.exists():
        logger.error(
            "index.html not found at %s",
            index_path,
        )

        raise HTTPException(
            status_code=404,
            detail=("Frontend not found. Check deployment."),
        )

    return FileResponse(str(index_path))


# =========================================================
# HEALTH CHECK
# =========================================================


@app.get("/health")
async def health_check():
    """
    Check API health status.
    """

    return {
        "status": "ok",
        "message": ("Enterprise RAG API is running"),
        "version": "1.0.0",
        "security": {
            "input_guardrails": True,
            "model_safety_classifier": (safety_classifier is not None),
            "authentication": True,
            "authorization": True,
        },
    }


# =========================================================
# CHAT / KNOWLEDGE COPILOT
# =========================================================


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
):
    """
    Knowledge Copilot security boundary.

    Security flow:

        User Query
             ↓
        Existing Input Guardrails
             ↓
        Groq LLM Safety Evaluation
             ↓
        Authentication
             ↓
        Authorization
             ↓
        Security Boundary
             ↓
        RAG

    RAG retrieval is intentionally NOT connected yet.
    """

    logger.info(
        "Knowledge Copilot request received for user_id=%s",
        request.user_id,
    )

    # =====================================================
    # STEP 1 - EXISTING INPUT GUARDRAILS
    # =====================================================

    guardrail_result = validate_user_query(request.query)

    if not guardrail_result.allowed:
        logger.warning(
            "Existing input guardrail blocked request for user_id=%s; violations=%s",
            request.user_id,
            guardrail_result.violations,
        )

        return {
            "stage": "input_guardrails",
            "allowed": False,
            "message": guardrail_result.reason,
            "violations": (guardrail_result.violations),
            "detection_method": ("deterministic_input_guardrails"),
        }

    logger.info(
        "Existing input guardrails passed for user_id=%s",
        request.user_id,
    )

    # =====================================================
    # STEP 2 - MODEL-BASED LLM SAFETY EVALUATION
    # =====================================================
    #
    # IMPORTANT:
    #
    # The Groq model does NOT answer the user's query.
    #
    # It only evaluates:
    #
    #     allowed
    #     category
    #     risk_level
    #     reason
    #
    # If the model cannot evaluate the request,
    # FAIL CLOSED.
    # =====================================================

    if safety_classifier is None:
        logger.error("Model safety classifier unavailable. Failing closed.")

        return {
            "stage": "model_safety_guardrail",
            "allowed": False,
            "message": ("Safety evaluation is temporarily unavailable."),
            "category": "other",
            "risk_level": "high",
            "detection_method": ("groq_model_safety_classifier"),
        }

    try:
        safety_result = safety_classifier.classify(request.query)

    except Exception as error:
        logger.error(
            "Model safety evaluation failed for user_id=%s: %s",
            request.user_id,
            error,
            exc_info=True,
        )

        # -------------------------------------------------
        # FAIL CLOSED
        # -------------------------------------------------

        return {
            "stage": "model_safety_guardrail",
            "allowed": False,
            "message": ("Safety evaluation could not be completed."),
            "category": "other",
            "risk_level": "high",
            "detection_method": ("groq_model_safety_classifier"),
        }

    # =====================================================
    # MODEL SAFETY BLOCK
    # =====================================================

    if not safety_result.allowed:
        logger.warning(
            "Groq model safety classifier "
            "blocked request "
            "for user_id=%s; "
            "category=%s; "
            "risk=%s",
            request.user_id,
            safety_result.category,
            safety_result.risk_level,
        )

        return {
            "stage": ("model_safety_guardrail"),
            "allowed": False,
            "message": (safety_result.reason),
            "category": (safety_result.category),
            "risk_level": (safety_result.risk_level),
            "detection_method": ("groq_model_safety_classifier"),
        }

    logger.info(
        "Groq model safety evaluation passed for user_id=%s; category=%s; risk=%s",
        request.user_id,
        safety_result.category,
        safety_result.risk_level,
    )

    # =====================================================
    # STEP 3 - AUTHENTICATION
    # =====================================================

    user = TEST_USERS.get(request.user_id)

    if user is None:
        logger.warning(
            "Authentication failed for user_id=%s",
            request.user_id,
        )

        return {
            "stage": "authentication",
            "allowed": False,
            "message": ("User authentication failed."),
        }

    logger.info(
        "Authentication passed for user_id=%s",
        request.user_id,
    )

    # =====================================================
    # STEP 4 - AUTHORIZATION
    # =====================================================

    authorization_result = authorization_policy.authorize(
        user=user,
        resource=(FRAUD_KNOWLEDGE_RESOURCE),
    )

    if not authorization_result.allowed:
        logger.warning(
            "Authorization denied for user_id=%s; violations=%s",
            request.user_id,
            authorization_result.violations,
        )

        return {
            "stage": "authorization",
            "allowed": False,
            "message": (authorization_result.reason),
            "violations": (authorization_result.violations),
        }

    logger.info(
        "Authorization passed for user_id=%s",
        request.user_id,
    )

    # =====================================================
    # STEP 5 - SECURITY BOUNDARY
    # =====================================================
    #
    # No RAG retrieval has happened yet.
    #
    # All security checks have passed:
    #
    #   1. Deterministic input guardrails
    #   2. LLM model safety evaluation
    #   3. Authentication
    #   4. Authorization
    #
    # =====================================================

    return {
        "stage": "security_boundary",
        "allowed": True,
        "message": ("Security checks passed."),
        "next_stage": "rag",
        "rag_status": "not_connected",
        "safety": {
            "category": (safety_result.category),
            "risk_level": (safety_result.risk_level),
            "detection_method": ("groq_model_safety_classifier"),
        },
    }


# =========================================================
# UPLOAD DOCUMENTS
# =========================================================


@app.post("/api/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
):
    """
    Upload documents to the knowledge base.
    """

    if not databricks_client:
        raise HTTPException(
            status_code=503,
            detail=("Databricks service unavailable"),
        )

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided",
        )

    uploaded_files = []

    failed_files = []

    # =====================================================
    # PROCESS FILES
    # =====================================================

    for file in files:
        # -------------------------------------------------
        # VALIDATE FILENAME
        # -------------------------------------------------

        if not file.filename:
            failed_files.append(
                {
                    "filename": "unknown",
                    "reason": ("Filename is missing"),
                }
            )

            continue

        safe_filename = Path(file.filename).name

        extension = Path(safe_filename).suffix.lower()

        logger.info(
            "Processing file: %s",
            safe_filename,
        )

        # -------------------------------------------------
        # VALIDATE FILE TYPE
        # -------------------------------------------------

        if extension not in ALLOWED_EXTENSIONS:
            logger.warning(
                "Unsupported file type: %s (%s)",
                safe_filename,
                extension,
            )

            failed_files.append(
                {
                    "filename": safe_filename,
                    "reason": (
                        f"Unsupported file type: "
                        f"{extension}. "
                        f"Allowed types: "
                        f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    ),
                }
            )

            continue

        # -------------------------------------------------
        # READ FILE
        # -------------------------------------------------

        try:
            content = await file.read()

        except Exception as error:
            logger.error(
                "Failed to read file %s: %s",
                safe_filename,
                error,
            )

            failed_files.append(
                {
                    "filename": safe_filename,
                    "reason": (f"Failed to read file: {error!s}"),
                }
            )

            continue

        # -------------------------------------------------
        # FILE SIZE
        # -------------------------------------------------

        file_size = len(content)

        if file_size == 0:
            logger.warning(
                "Empty file: %s",
                safe_filename,
            )

            failed_files.append(
                {
                    "filename": safe_filename,
                    "reason": "File is empty",
                }
            )

            continue

        if file_size > MAX_FILE_SIZE:
            logger.warning(
                "File exceeds size limit: %s (%s bytes)",
                safe_filename,
                file_size,
            )

            failed_files.append(
                {
                    "filename": safe_filename,
                    "reason": (f"File exceeds maximum size of {MAX_FILE_SIZE} bytes"),
                }
            )

            continue

        # =================================================
        # UPLOAD TO DATABRICKS
        # =================================================

        try:
            logger.info(
                "Uploading %s (%s bytes) to Databricks",
                safe_filename,
                file_size,
            )

            target_path = databricks_client.upload_file(
                safe_filename,
                content,
            )

            uploaded_files.append(
                {
                    "filename": safe_filename,
                    "path": target_path,
                    "size": file_size,
                    "extension": extension,
                }
            )

            logger.info(
                "Successfully uploaded: %s → %s",
                safe_filename,
                target_path,
            )

        except Exception as error:
            logger.error(
                "Failed to upload %s: %s",
                safe_filename,
                error,
                exc_info=True,
            )

            failed_files.append(
                {
                    "filename": safe_filename,
                    "reason": str(error),
                }
            )

    # =====================================================
    # ALL FILES FAILED
    # =====================================================

    if not uploaded_files and failed_files:
        logger.error(
            "Upload failed - all %s files failed",
            len(failed_files),
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": ("No files were uploaded successfully"),
                "failed_files": (failed_files),
            },
        )

    # =====================================================
    # SUCCESS
    # =====================================================

    logger.info(
        "Upload completed: %s successful, %s failed",
        len(uploaded_files),
        len(failed_files),
    )

    if failed_files:
        message = (
            f"{len(uploaded_files)} file(s) "
            "uploaded successfully. "
            f"{len(failed_files)} file(s) failed."
        )

    else:
        message = f"{len(uploaded_files)} file(s) uploaded successfully."

    return {
        "message": message,
        "uploaded_files": (uploaded_files),
        "failed_files": (failed_files),
        "summary": {
            "total": (len(uploaded_files) + len(failed_files)),
            "successful": (len(uploaded_files)),
            "failed": (len(failed_files)),
        },
    }


# =========================================================
# LIST DOCUMENTS
# =========================================================


@app.get("/api/documents")
async def list_documents():
    """
    List all documents in the knowledge base.
    """

    if not databricks_client:
        raise HTTPException(
            status_code=503,
            detail=("Databricks service unavailable"),
        )

    try:
        logger.info("Fetching document list")

        files = databricks_client.list_files()

        logger.info(
            "Found %s documents",
            len(files),
        )

        return {
            "files": files,
            "count": len(files),
            "status": "success",
        }

    except Exception as error:
        logger.error(
            "Failed to list documents: %s",
            error,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(f"Failed to retrieve documents: {error!s}"),
        )


# =========================================================
# ERROR HANDLERS
# =========================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request,
    exc,
):

    logger.error(
        "HTTP Exception: %s - %s",
        exc.status_code,
        exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "status_code": (exc.status_code),
            "detail": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request,
    exc,
):

    logger.error(
    "Unhandled exception: %s",
    exc,
)

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "status_code": 500,
            "detail": str(exc),
        },
    )


# =========================================================
# STARTUP
# =========================================================


@app.on_event("startup")
async def startup_event():

    logger.info("=" * 60)

    logger.info("Enterprise RAG API Starting")

    logger.info("=" * 60)

    logger.info(
        "Frontend: %s",
        FRONTEND_DIR,
    )

    logger.info(
        "Databricks Client: %s",
        ("OK" if databricks_client else "UNAVAILABLE"),
    )

    logger.info(
        "Groq Safety Classifier: %s",
        ("OK" if safety_classifier else "UNAVAILABLE"),
    )

    logger.info("Deterministic Input Guardrails: ENABLED")

    logger.info("Model Safety Evaluation: ENABLED")

    logger.info("Authentication: ENABLED")

    logger.info("Authorization: ENABLED")

    logger.info("RAG Retrieval: NOT CONNECTED")

    logger.info("=" * 60)


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
