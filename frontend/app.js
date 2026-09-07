/* =========================================================
   ENTERPRISE RAG FRONTEND - ENHANCED
   ========================================================= */


/* =========================================================
   CONFIGURATION
   ========================================================= */

// Because the frontend is served by FastAPI itself,
// an empty API_BASE_URL means:
//
//     /api/...
//
// will use the same host and port as the frontend.

const API_BASE_URL = "";


/* =========================================================
   STATE
   ========================================================= */

let selectedFiles = [];
let allDocuments = [];


/* =========================================================
   DOM ELEMENTS
   ========================================================= */

const uploadModal =
    document.getElementById("uploadModal");

const openUploadButton =
    document.getElementById("openUploadButton");

const quickUpload =
    document.getElementById("quickUpload");

const documentsUploadButton =
    document.getElementById("documentsUploadButton");

const closeUploadButton =
    document.getElementById("closeUploadButton");

const cancelUploadButton =
    document.getElementById("cancelUploadButton");

const fileInput =
    document.getElementById("fileInput");

const dropZone =
    document.getElementById("dropZone");

const selectedFilesContainer =
    document.getElementById("selectedFiles");

const uploadButton =
    document.getElementById("uploadButton");

const toast =
    document.getElementById("toast");

const documentsBody =
    document.getElementById("documentsBody");

const recentDocumentsBody =
    document.getElementById("recentDocumentsBody");

const searchInput =
    document.getElementById("searchInput");

const refreshDocumentsButton =
    document.getElementById("refreshDocumentsButton");


/* =========================================================
   COPILOT DOM ELEMENTS
   ========================================================= */

const copilotInput =
    document.getElementById("copilotInput");

const copilotSendButton =
    document.getElementById("copilotSendButton");

const testUserSelect =
    document.getElementById("testUserSelect");

const securityResult =
    document.getElementById("securityResult");

const securityResultIcon =
    document.getElementById("securityResultIcon");

const securityResultTitle =
    document.getElementById("securityResultTitle");

const securityResultMessage =
    document.getElementById("securityResultMessage");


/* =========================================================
   SECURITY RESULT DETAILS
   ========================================================= */

const resultGuardrail =
    document.getElementById("resultGuardrail");

const resultModelSafety =
    document.getElementById("resultModelSafety");

const resultAuthorization =
    document.getElementById("resultAuthorization");

const resultNextStage =
    document.getElementById("resultNextStage");


/* =========================================================
   SECURITY PIPELINE ELEMENTS
   ========================================================= */

const guardrailStep =
    document.getElementById("guardrailStep");

const modelSafetyStep =
    document.getElementById("modelSafetyStep");

const authorizationStep =
    document.getElementById("authorizationStep");

const ragStep =
    document.getElementById("ragStep");


const guardrailStatus =
    document.getElementById("guardrailStatus");

const modelSafetyStatus =
    document.getElementById("modelSafetyStatus");

const authorizationStatus =
    document.getElementById("authorizationStatus");

const ragStatus =
    document.getElementById("ragStatus");


/* =========================================================
   PAGE NAVIGATION
   ========================================================= */

const navItems =
    document.querySelectorAll(".nav-item");

const pageButtons =
    document.querySelectorAll("[data-page]");


const pages = {

    overview:
        document.getElementById("overviewPage"),

    documents:
        document.getElementById("documentsPage"),

    copilot:
        document.getElementById("copilotPage"),

    processing:
        document.getElementById("processingPage"),

    vector:
        document.getElementById("vectorPage"),

    settings:
        document.getElementById("settingsPage"),

};


/* =========================================================
   PAGE DISPLAY
   ========================================================= */

function showPage(pageKey) {

    Object.values(pages).forEach(page => {

        if (page) {
            page.classList.remove("active");
        }

    });


    if (pages[pageKey]) {

        pages[pageKey].classList.add("active");

    }


    const pageTitle =
        document.getElementById("pageTitle");


    if (pageTitle) {

        const pageNames = {

            overview: "Overview",

            documents: "Knowledge Base",

            copilot: "Knowledge Copilot",

            processing: "Processing",

            vector: "Vector Index",

            settings: "Settings",

        };


        pageTitle.textContent =
            pageNames[pageKey] || "Overview";

    }


    navItems.forEach(item => {

        item.classList.remove("active");


        if (item.dataset.page === pageKey) {

            item.classList.add("active");

        }

    });

}


/* =========================================================
   NAVIGATION EVENT HANDLERS
   ========================================================= */

navItems.forEach(button => {

    button.addEventListener("click", () => {

        const page =
            button.dataset.page;

        if (page) {

            showPage(page);

        }

    });

});


/*
 * Existing quick-action buttons also use data-page.
 */

pageButtons.forEach(button => {

    button.addEventListener("click", () => {

        const page =
            button.dataset.page;

        if (page) {

            showPage(page);

        }

    });

});


/* =========================================================
   UPLOAD MODAL
   ========================================================= */

function openUploadModal() {

    if (uploadModal) {

        uploadModal.classList.add("active");

    }

}


function closeUploadModal() {

    if (!uploadModal) {
        return;
    }

    uploadModal.classList.remove("active");

    selectedFiles = [];

    renderSelectedFiles();

    if (fileInput) {
        fileInput.value = "";
    }

}


if (openUploadButton) {

    openUploadButton.addEventListener(
        "click",
        openUploadModal
    );

}


quickUpload?.addEventListener(
    "click",
    openUploadModal
);


documentsUploadButton?.addEventListener(
    "click",
    openUploadModal
);


closeUploadButton?.addEventListener(
    "click",
    closeUploadModal
);


cancelUploadButton?.addEventListener(
    "click",
    closeUploadModal
);


uploadModal?.addEventListener(
    "click",
    event => {

        if (event.target === uploadModal) {

            closeUploadModal();

        }

    }
);


/* =========================================================
   FILE PICKER
   ========================================================= */

fileInput?.addEventListener(
    "change",
    event => {

        addFiles(event.target.files);

    }
);


/* =========================================================
   DRAG & DROP
   ========================================================= */

if (dropZone) {

    ["dragenter", "dragover"].forEach(
        eventName => {

            dropZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();

                    event.stopPropagation();

                    dropZone.classList.add(
                        "drag-over"
                    );

                }
            );

        }
    );


    ["dragleave", "drop"].forEach(
        eventName => {

            dropZone.addEventListener(
                eventName,
                event => {

                    event.preventDefault();

                    event.stopPropagation();

                    dropZone.classList.remove(
                        "drag-over"
                    );

                }
            );

        }
    );


    dropZone.addEventListener(
        "drop",
        event => {

            const files =
                event.dataTransfer.files;

            addFiles(files);

        }
    );

}


/* =========================================================
   ADD FILES
   ========================================================= */

function addFiles(files) {

    if (!files || files.length === 0) {

        return;

    }


    const allowedExtensions = [

        ".pdf",
        ".xml",
        ".txt",
        ".md",
        ".doc",
        ".docx",
        ".ppt",
        ".pptx",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"

    ];


    for (const file of files) {

        const extension =
            getFileExtension(file.name);


        if (!allowedExtensions.includes(extension)) {

            showToast(
                `⚠️ Unsupported file type: ${file.name}`
            );

            continue;

        }


        const alreadyExists =
            selectedFiles.some(
                existingFile =>
                    existingFile.name === file.name &&
                    existingFile.size === file.size
            );


        if (!alreadyExists) {

            selectedFiles.push(file);

        } else {

            showToast(
                `📋 File already selected: ${file.name}`
            );

        }

    }


    renderSelectedFiles();

}


/* =========================================================
   GET FILE EXTENSION
   ========================================================= */

function getFileExtension(fileName) {

    const lastDot =
        fileName.lastIndexOf(".");


    if (lastDot === -1) {

        return "";

    }


    return fileName
        .substring(lastDot)
        .toLowerCase();

}


/* =========================================================
   RENDER SELECTED FILES
   ========================================================= */

function renderSelectedFiles() {

    if (!selectedFilesContainer) {

        return;

    }


    selectedFilesContainer.innerHTML = "";


    if (selectedFiles.length === 0) {

        if (uploadButton) {
            uploadButton.disabled = true;
        }

        return;

    }


    if (uploadButton) {
        uploadButton.disabled = false;
    }


    selectedFiles.forEach(
        (file, index) => {

            const item =
                document.createElement("div");

            item.className =
                "file-item";


            const left =
                document.createElement("div");


            const name =
                document.createElement("div");

            name.className =
                "file-name";

            name.textContent =
                file.name;


            const size =
                document.createElement("span");

            size.className =
                "file-size";

            size.textContent =
                formatFileSize(file.size);


            left.appendChild(name);

            left.appendChild(size);


            const removeButton =
                document.createElement("button");

            removeButton.className =
                "secondary-button";

            removeButton.textContent =
                "Remove";


            removeButton.addEventListener(
                "click",
                () => {

                    selectedFiles.splice(
                        index,
                        1
                    );

                    renderSelectedFiles();

                }
            );


            item.appendChild(left);

            item.appendChild(removeButton);


            selectedFilesContainer.appendChild(
                item
            );

        }
    );

}


/* =========================================================
   FORMAT FILE SIZE
   ========================================================= */

function formatFileSize(bytes) {

    if (bytes === 0) {

        return "0 Bytes";

    }


    const units = [

        "Bytes",
        "KB",
        "MB",
        "GB"

    ];


    const index =
        Math.floor(
            Math.log(bytes) /
            Math.log(1024)
        );


    return (
        bytes /
        Math.pow(1024, index)
    ).toFixed(2) +
        " " +
        units[index];

}


/* =========================================================
   UPLOAD DOCUMENTS
   ========================================================= */

uploadButton?.addEventListener(
    "click",
    uploadDocuments
);


async function uploadDocuments() {

    if (selectedFiles.length === 0) {

        showToast(
            "📄 Please select at least one document"
        );

        return;

    }


    uploadButton.disabled = true;

    uploadButton.textContent =
        "Uploading...";


    const formData =
        new FormData();


    selectedFiles.forEach(file => {

        formData.append(
            "files",
            file
        );

    });


    try {

        const response =
            await fetch(
                `${API_BASE_URL}/api/documents/upload`,
                {
                    method: "POST",
                    body: formData,
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            let message =
                "Upload failed.";


            if (data.detail) {

                if (
                    typeof data.detail ===
                    "string"
                ) {

                    message =
                        data.detail;

                } else if (
                    data.detail.message
                ) {

                    message =
                        data.detail.message;

                }

            }


            throw new Error(message);

        }


        showToast(
            "✅ " +
            (
                data.message ||
                "Documents uploaded successfully"
            )
        );


        if (
            data.failed_files &&
            data.failed_files.length > 0
        ) {

            console.warn(
                "Failed files:",
                data.failed_files
            );

        }


        closeUploadModal();

        await loadDocuments();


    } catch (error) {

        console.error(
            "Upload error:",
            error
        );


        showToast(
            "❌ " +
            (
                error.message ||
                "Upload failed"
            )
        );


    } finally {

        uploadButton.disabled =
            selectedFiles.length === 0;

        uploadButton.textContent =
            "Upload documents";

    }

}


/* =========================================================
   LOAD DOCUMENTS
   ========================================================= */

async function loadDocuments() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/api/documents`
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to load documents"
            );

        }


        allDocuments =
            data.files || [];


        updateMetrics(
            allDocuments
        );


        renderDocuments(
            allDocuments
        );


        renderRecentDocuments(
            allDocuments
        );


    } catch (error) {

        console.error(
            "Document loading error:",
            error
        );


        if (documentsBody) {

            documentsBody.innerHTML = `

                <tr>

                    <td
                        colspan="4"
                        style="
                            text-align: center;
                            color: #9ca3af;
                        "
                    >
                        Unable to load documents
                    </td>

                </tr>

            `;

        }

    }

}


/* =========================================================
   RENDER DOCUMENTS
   ========================================================= */

function renderDocuments(documents) {

    if (!documentsBody) {
        return;
    }


    if (
        !documents ||
        documents.length === 0
    ) {

        documentsBody.innerHTML = `

            <tr>

                <td
                    colspan="4"
                    style="
                        text-align: center;
                        color: #9ca3af;
                    "
                >
                    No documents found
                </td>

            </tr>

        `;

        return;

    }


    documentsBody.innerHTML = "";


    documents.forEach(document => {

        const row =
            document.createElement("tr");


        const extension =
            getFileExtension(
                document.name
            )
            .replace(".", "")
            .toUpperCase();


        row.innerHTML = `

            <td>
                ${escapeHtml(document.name)}
            </td>

            <td>
                ${extension}
            </td>

            <td>
                ${formatFileSize(
                    document.size || 0
                )}
            </td>

            <td>
                ${formatDate(
                    document.modified_time
                )}
            </td>

        `;


        documentsBody.appendChild(row);

    });

}


/* =========================================================
   RENDER RECENT DOCUMENTS
   ========================================================= */

function renderRecentDocuments(documents) {

    if (!recentDocumentsBody) {
        return;
    }


    if (
        !documents ||
        documents.length === 0
    ) {

        recentDocumentsBody.innerHTML = `

            <tr>

                <td
                    colspan="4"
                    style="
                        text-align: center;
                        color: #9ca3af;
                    "
                >
                    No documents uploaded yet
                </td>

            </tr>

        `;

        return;

    }


    const recent =
        documents.slice(0, 5);


    recentDocumentsBody.innerHTML =
        "";


    recent.forEach(document => {

        const row =
            document.createElement("tr");


        const extension =
            getFileExtension(
                document.name
            )
            .replace(".", "")
            .toUpperCase();


        row.innerHTML = `

            <td>
                ${escapeHtml(document.name)}
            </td>

            <td>
                ${extension}
            </td>

            <td>
                ${formatFileSize(
                    document.size || 0
                )}
            </td>

            <td>
                <span class="status-badge ready">
                    Uploaded
                </span>
            </td>

        `;


        recentDocumentsBody.appendChild(
            row
        );

    });

}


/* =========================================================
   UPDATE METRICS
   ========================================================= */

function updateMetrics(documents) {

    const documentCount =
        document.getElementById(
            "documentCount"
        );

    const readyCount =
        document.getElementById(
            "readyCount"
        );

    const processingCount =
        document.getElementById(
            "processingCount"
        );

    const vectorCount =
        document.getElementById(
            "vectorCount"
        );


    const count =
        documents.length;


    if (documentCount) {

        documentCount.textContent =
            count;

    }


    if (readyCount) {

        readyCount.textContent =
            count;

    }


    if (processingCount) {

        processingCount.textContent =
            "0";

    }


    if (vectorCount) {

        vectorCount.textContent =
            "0";

    }

}


/* =========================================================
   SEARCH
   ========================================================= */

if (searchInput) {

    searchInput.addEventListener(
        "input",
        () => {

            const searchTerm =
                searchInput.value
                    .trim()
                    .toLowerCase();


            if (searchTerm === "") {

                renderDocuments(
                    allDocuments
                );

                return;

            }


            const filtered =
                allDocuments.filter(
                    document =>
                        document.name
                            .toLowerCase()
                            .includes(searchTerm)
                );


            renderDocuments(
                filtered
            );

        }
    );

}


/* =========================================================
   REFRESH
   ========================================================= */

if (refreshDocumentsButton) {

    refreshDocumentsButton.addEventListener(
        "click",
        async () => {

            refreshDocumentsButton.disabled =
                true;


            refreshDocumentsButton.textContent =
                "Refreshing...";


            await loadDocuments();


            refreshDocumentsButton.disabled =
                false;


            refreshDocumentsButton.textContent =
                "Refresh";

        }
    );

}


/* =========================================================
   HEALTH CHECK
   ========================================================= */

async function checkHealth() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/health`
            );


        if (response.ok) {

            console.log(
                "✅ Enterprise RAG API is healthy"
            );

        }

    } catch (error) {

        console.warn(
            "⚠️ API health check failed:",
            error
        );

    }

}


/* =========================================================
   TOAST NOTIFICATIONS
   ========================================================= */

let toastTimer;


function showToast(message) {

    if (!toast) {
        return;
    }


    toast.textContent =
        message;


    toast.classList.add(
        "show"
    );


    clearTimeout(
        toastTimer
    );


    toastTimer =
        setTimeout(
            () => {

                toast.classList.remove(
                    "show"
                );

            },
            4000
        );

}


/* =========================================================
   HTML ESCAPE
   ========================================================= */

function escapeHtml(value) {

    return String(value)

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );

}


/* =========================================================
   FORMAT DATE
   ========================================================= */

function formatDate(value) {

    if (!value) {

        return "-";

    }


    try {

        const date =
            new Date(value);


        if (
            Number.isNaN(
                date.getTime()
            )
        ) {

            return "-";

        }


        return date.toLocaleString();


    } catch {

        return "-";

    }

}


/* =========================================================
   COPILOT SECURITY FLOW
   ========================================================= */

if (copilotSendButton) {

    copilotSendButton.addEventListener(
        "click",
        submitCopilotQuestion
    );

}


if (copilotInput) {

    copilotInput.addEventListener(
        "keydown",
        event => {

            /*
             * Enter submits.
             *
             * Shift + Enter creates a new line.
             */

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                submitCopilotQuestion();

            }

        }
    );

}


/* =========================================================
   SUBMIT COPILOT QUESTION
   ========================================================= */

async function submitCopilotQuestion() {

    if (!copilotInput) {
        return;
    }


    const query =
        copilotInput.value.trim();


    if (!query) {

        showToast(
            "💬 Please enter a question"
        );

        return;

    }


    resetSecurityPipeline();


    if (copilotSendButton) {

        copilotSendButton.disabled =
            true;

        copilotSendButton.textContent =
            "Checking...";

    }


    try {

        /*
         * Temporary test identity.
         *
         * This is deliberately NOT enterprise
         * authentication.
         *
         * It will later be replaced with the
         * authenticated enterprise identity.
         */

        const userId =
            testUserSelect?.value ||
            "user-001";


        /*
         * ===============================
         * INPUT GUARDRAILS
         * ===============================
         */

        if (guardrailStatus) {

            guardrailStatus.textContent =
                "Checking...";

        }


        const response =
            await fetch(
                `${API_BASE_URL}/api/chat`,
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body: JSON.stringify({

                        query: query,

                        user_id: userId

                    })

                }
            );


        const data =
            await response.json();


        /*
         * ===============================
         * INPUT GUARDRAIL BLOCKED
         * ===============================
         */

        if (
            data.stage ===
                "input_guardrails" &&
            data.allowed === false
        ) {

            markGuardrailBlocked();


            showSecurityResult({

                blocked: true,

                title:
                    "Request blocked",

                message:
                    data.message ||
                    "The request was blocked by the input safety guardrails.",

                guardrail:
                    "Blocked",

                modelSafety:
                    "Not evaluated",

                authorization:
                    "Not evaluated",

                nextStage:
                    "Request stopped"

            });


            return;

        }


        /*
         * ===============================
         * MODEL SAFETY GUARDRAIL
         * ===============================
         */

        if (
            data.stage ===
                "model_safety_guardrail"
        ) {

            /*
             * The backend should normally return
             * this stage only when the model safety
             * classifier blocks the request.
             */

            if (data.allowed === false) {

                markGuardrailPassed();

                markModelSafetyBlocked();


                showSecurityResult({

                    blocked: true,

                    title:
                        "Request blocked",

                    message:
                        data.message ||
                        "The AI safety classifier blocked this request.",

                    guardrail:
                        "Passed",

                    modelSafety:
                        "Blocked",

                    authorization:
                        "Not evaluated",

                    nextStage:
                        "Request stopped"

                });


                return;

            }


            /*
             * Defensive handling if the backend
             * ever explicitly returns a successful
             * model-safety stage.
             *
             * The current backend normally continues
             * directly to authentication/authorization.
             */

            if (data.allowed === true) {

                markGuardrailPassed();

                markModelSafetyPassed();

            }

        }


        /*
         * ===============================
         * AUTHENTICATION FAILED
         * ===============================
         */

        if (
            data.stage ===
                "authentication" &&
            data.allowed === false
        ) {

            markGuardrailPassed();

            markModelSafetyPassed();

            markAuthorizationNotEvaluated();


            showSecurityResult({

                blocked: true,

                title:
                    "Authentication failed",

                message:
                    data.message ||
                    "User identity could not be verified.",

                guardrail:
                    "Passed",

                modelSafety:
                    "Passed",

                authorization:
                    "Not evaluated",

                nextStage:
                    "Request stopped"

            });


            return;

        }


        /*
         * ===============================
         * AUTHORIZATION FAILED
         * ===============================
         */

        if (
            data.stage ===
                "authorization" &&
            data.allowed === false
        ) {

            markGuardrailPassed();

            markModelSafetyPassed();

            markAuthorizationBlocked();


            showSecurityResult({

                blocked: true,

                title:
                    "Access denied",

                message:
                    data.message ||
                    "You are not authorized to access this resource.",

                guardrail:
                    "Passed",

                modelSafety:
                    "Passed",

                authorization:
                    "Denied",

                nextStage:
                    "Request stopped"

            });


            return;

        }


        /*
         * ===============================
         * SECURITY CHECKS PASSED
         * ===============================
         */

        if (
            data.stage ===
                "security_boundary" &&
            data.allowed === true
        ) {

            /*
             * The current backend does not return
             * a separate "model_safety passed" response.
             *
             * Therefore reaching security_boundary
             * means the request successfully passed:
             *
             * 1. Input Guardrails
             * 2. AI Safety
             * 3. Authentication
             * 4. Authorization
             */

            markGuardrailPassed();

            markModelSafetyPassed();

            markAuthorizationPassed();


            /*
             * RAG is intentionally NOT connected yet.
             */

            if (ragStep) {

                ragStep.classList.remove(
                    "passed",
                    "blocked"
                );

            }


            if (ragStatus) {

                ragStatus.textContent =
                    "Not connected";

            }


            showSecurityResult({

                blocked: false,

                title:
                    "Security checks passed",

                message:
                    "Your request passed input safety, AI safety, authentication, and authorization checks.",

                guardrail:
                    "Passed",

                modelSafety:
                    "Passed",

                authorization:
                    "Authorized",

                nextStage:
                    "RAG"

            });


            return;

        }
        
        /*
 * ===============================
 * SUPERVISOR ROUTING PASSED
 * ===============================
 */

if (
    data.stage === "supervisor" &&
    data.allowed === true
) {

    /*
     * Reaching the Supervisor means the request
     * successfully passed:
     *
     * 1. Input Guardrails
     * 2. AI Safety
     * 3. Authentication
     * 4. Authorization
     * 5. Supervisor routing
     *
     * Agent execution is intentionally not connected yet.
     */

    markGuardrailPassed();

    markModelSafetyPassed();

    markAuthorizationPassed();


    /*
     * RAG / agent execution are intentionally
     * not connected at this phase.
     */

    if (ragStep) {

        ragStep.classList.remove(
            "passed",
            "blocked"
        );

    }


    if (ragStatus) {

        ragStatus.textContent =
            "Not connected";

    }


    showSecurityResult({

        blocked: false,

        title:
            "Security checks passed",

        message:
            data.message ||
            "Your request passed security, authorization, and Supervisor routing.",

        guardrail:
            "Passed",

        modelSafety:
            "Passed",

        authorization:
            "Authorized",

        nextStage:
            data.next_stage ||
            "Agent Execution"

    });


    return;

}

        /*
         * ===============================
         * UNEXPECTED RESPONSE
         * ===============================
         */

        showSecurityResult({

            blocked: true,

            title:
                "Unexpected response",

            message:
                "The platform returned an unexpected security response.",

            guardrail:
                "-",

            modelSafety:
                "-",

            authorization:
                "-",

            nextStage:
                "Request stopped"

        });


    } catch (error) {

        console.error(
            "Copilot security check error:",
            error
        );


        showSecurityResult({

            blocked: true,

            title:
                "Unable to process request",

            message:
                "The security service could not be reached.",

            guardrail:
                "-",

            modelSafety:
                "-",

            authorization:
                "-",

            nextStage:
                "Request stopped"

        });


    } finally {

        if (copilotSendButton) {

            copilotSendButton.disabled =
                false;

            copilotSendButton.textContent =
                "Send question";

        }

    }

}


/* =========================================================
   SECURITY PIPELINE UI
   ========================================================= */

function resetSecurityPipeline() {

    guardrailStep?.classList.remove(
        "passed",
        "blocked"
    );


    modelSafetyStep?.classList.remove(
        "passed",
        "blocked"
    );


    authorizationStep?.classList.remove(
        "passed",
        "blocked"
    );


    ragStep?.classList.remove(
        "passed",
        "blocked"
    );


    if (guardrailStatus) {

        guardrailStatus.textContent =
            "Checking...";

    }


    if (modelSafetyStatus) {

        modelSafetyStatus.textContent =
            "Waiting";

    }


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Waiting";

    }


    if (ragStatus) {

        ragStatus.textContent =
            "Not connected";

    }


    if (securityResult) {

        securityResult.style.display =
            "none";

    }

}


/* =========================================================
   INPUT GUARDRAIL - PASSED
   ========================================================= */

function markGuardrailPassed() {

    guardrailStep?.classList.remove(
        "blocked"
    );


    guardrailStep?.classList.add(
        "passed"
    );


    if (guardrailStatus) {

        guardrailStatus.textContent =
            "Passed";

    }


    if (modelSafetyStatus) {

        modelSafetyStatus.textContent =
            "Checking...";

    }

}


/* =========================================================
   INPUT GUARDRAIL - BLOCKED
   ========================================================= */

function markGuardrailBlocked() {

    guardrailStep?.classList.remove(
        "passed"
    );


    guardrailStep?.classList.add(
        "blocked"
    );


    if (guardrailStatus) {

        guardrailStatus.textContent =
            "Blocked";

    }


    modelSafetyStep?.classList.remove(
        "passed"
    );


    if (modelSafetyStatus) {

        modelSafetyStatus.textContent =
            "Not evaluated";

    }


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Not evaluated";

    }


    if (ragStep) {

        ragStep.classList.remove(
            "passed"
        );

        ragStep.classList.add(
            "blocked"
        );

    }


    if (ragStatus) {

        ragStatus.textContent =
            "Not evaluated";

    }

}


/* =========================================================
   MODEL SAFETY - PASSED
   ========================================================= */

function markModelSafetyPassed() {

    modelSafetyStep?.classList.remove(
        "blocked"
    );


    modelSafetyStep?.classList.add(
        "passed"
    );


    if (modelSafetyStatus) {

        modelSafetyStatus.textContent =
            "Passed";

    }


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Checking...";

    }

}


/* =========================================================
   MODEL SAFETY - BLOCKED
   ========================================================= */

function markModelSafetyBlocked() {

    modelSafetyStep?.classList.remove(
        "passed"
    );


    modelSafetyStep?.classList.add(
        "blocked"
    );


    if (modelSafetyStatus) {

        modelSafetyStatus.textContent =
            "Blocked";

    }


    if (authorizationStep) {

        authorizationStep.classList.remove(
            "passed"
        );

        authorizationStep.classList.add(
            "blocked"
        );

    }


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Not evaluated";

    }


    if (ragStep) {

        ragStep.classList.remove(
            "passed"
        );

        ragStep.classList.add(
            "blocked"
        );

    }


    if (ragStatus) {

        ragStatus.textContent =
            "Not evaluated";

    }

}


/* =========================================================
   AUTHORIZATION - NOT EVALUATED
   ========================================================= */

function markAuthorizationNotEvaluated() {

    authorizationStep?.classList.remove(
        "passed",
        "blocked"
    );


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Not evaluated";

    }


    if (ragStep) {

        ragStep.classList.remove(
            "passed"
        );

        ragStep.classList.add(
            "blocked"
        );

    }


    if (ragStatus) {

        ragStatus.textContent =
            "Not evaluated";

    }

}


/* =========================================================
   AUTHORIZATION - PASSED
   ========================================================= */

function markAuthorizationPassed() {

    authorizationStep?.classList.remove(
        "blocked"
    );


    authorizationStep?.classList.add(
        "passed"
    );


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Authorized";

    }

}


/* =========================================================
   AUTHORIZATION - BLOCKED
   ========================================================= */

function markAuthorizationBlocked() {

    authorizationStep?.classList.remove(
        "passed"
    );


    authorizationStep?.classList.add(
        "blocked"
    );


    if (authorizationStatus) {

        authorizationStatus.textContent =
            "Denied";

    }


    if (ragStep) {

        ragStep.classList.remove(
            "passed"
        );

        ragStep.classList.add(
            "blocked"
        );

    }


    if (ragStatus) {

        ragStatus.textContent =
            "Not evaluated";

    }

}


/* =========================================================
   SECURITY RESULT
   ========================================================= */

function showSecurityResult({

    blocked,

    title,

    message,

    guardrail,

    modelSafety,

    authorization,

    nextStage

}) {

    if (!securityResult) {
        return;
    }


    securityResult.style.display =
        "block";


    securityResult.classList.toggle(
        "blocked",
        blocked
    );


    if (securityResultIcon) {

        securityResultIcon.textContent =
            blocked ? "!" : "✓";

    }


    if (securityResultTitle) {

        securityResultTitle.textContent =
            title;

    }


    if (securityResultMessage) {

        securityResultMessage.textContent =
            message;

    }


    if (resultGuardrail) {

        resultGuardrail.textContent =
            guardrail || "-";

    }


    if (resultModelSafety) {

        resultModelSafety.textContent =
            modelSafety || "-";

    }


    if (resultAuthorization) {

        resultAuthorization.textContent =
            authorization || "-";

    }


    if (resultNextStage) {

        resultNextStage.textContent =
            nextStage || "-";

    }

}


/* =========================================================
   INITIALIZATION
   ========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    async () => {

        showPage("overview");


        /*
         * Because the frontend is served by
         * FastAPI itself, an empty API_BASE_URL
         * is valid.
         */

        await checkHealth();

        await loadDocuments();

    }
);