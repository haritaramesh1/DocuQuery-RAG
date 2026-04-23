"""FastAPI application: PDF upload indexing and grounded Q&A."""

# Postponed annotations for typing.
from __future__ import annotations

# Standard logger so handled exceptions still surface in uvicorn output.
import logging
# asynccontextmanager lets us define startup/shutdown hooks around the app instance.
from contextlib import asynccontextmanager

# Loads key=value pairs from a .env file into os.environ before other imports use them.
from dotenv import load_dotenv
# FastAPI core: routing, dependency injection, file uploads, HTTP errors.
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
# CORS middleware so the browser SPA on another port may call this API.
from fastapi.middleware.cors import CORSMiddleware
# Local Ollama chat model class used as a FastAPI dependency type hint.
from langchain_community.chat_models import ChatOllama

# Helpers to read which Ollama models are configured (for error messages).
from app.config import ollama_embed_model, ollama_llm_model
# Pydantic models for request bodies and JSON responses.
from app.schemas import QueryRequest, QueryResponse, UploadResponse
# Service modules: PDF→FAISS and RAG query path.
from app.services import ingestion, rag
# Shared singleton state and per-document index record type.
from app.state import AppState, DocumentIndexEntry, app_state

# Populate os.environ from .env at import time (safe if file missing).
load_dotenv()

# Module logger used to print full tracebacks for upstream Ollama failures.
logger = logging.getLogger(__name__)


# FastAPI lifespan hook: runs around the serving loop (placeholder for future init).
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Yield control to the server; code before yield runs on startup, after on shutdown.
    yield


# Construct the ASGI application with metadata and lifespan wiring.
app = FastAPI(title="Document Q&A", lifespan=lifespan)

# Register Starlette CORS middleware with permissive dev defaults.
app.add_middleware(
    # Middleware class that injects Access-Control-* headers on responses.
    CORSMiddleware,
    # Explicit allowlist of browser origins that may use cookies or fetch from JS.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    # Permit Authorization/cookie headers from those origins.
    allow_credentials=True,
    # Allow all HTTP verbs used by fetch (GET/POST/etc.).
    allow_methods=["*"],
    # Allow arbitrary request headers from the browser preflight.
    allow_headers=["*"],
)


# FastAPI dependency that returns the process-wide AppState singleton.
def get_state() -> AppState:
    # Return the module-level shared cache object.
    return app_state


# FastAPI dependency that builds a fresh ChatOllama client per request (lightweight).
def get_chat_llm() -> ChatOllama:
    # Delegate construction to rag module (reads env for model/base URL).
    return rag.make_chat_ollama()


# POST /upload: multipart PDF → hash → optional cache hit → else embed + FAISS.
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    # Incoming multipart part named “file” (required).
    file: UploadFile = File(...),
    # Injected global state for cache lookup and storage.
    state: AppState = Depends(get_state),
):
    # Use client-provided filename or a sensible default string.
    filename = file.filename or "document.pdf"
    # Only accept PDFs by extension (simple, predictable rule).
    if not filename.lower().endswith(".pdf"):
        # Reject non-PDF uploads early with HTTP 400.
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read entire upload body into memory (PDFs expected small enough for this demo).
    file_bytes = await file.read()
    # Reject empty uploads to avoid useless hashing and errors downstream.
    if not file_bytes:
        # HTTP 400 with explicit message for the UI.
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Compute stable digest for deduplication and client-side document identity.
    doc_hash = ingestion.sha256_bytes(file_bytes)

    # Look for an existing in-memory index for this exact byte content.
    cached = state.get(doc_hash)
    # If found, skip all embedding work and report cache hit metadata.
    if cached is not None:
        # Return JSON matching UploadResponse schema immediately.
        return UploadResponse(
            # Echo the hash so the client can query this document explicitly.
            document_hash=doc_hash,
            # Signal that embeddings were not recomputed.
            cached=True,
            # Number of chunks already stored for this hash.
            chunk_count=len(cached.chunk_texts),
            # Filename for UI (may differ from prior upload with same bytes).
            filename=filename,
        )

    # Build Ollama-backed embedding client for this indexing operation.
    embeddings = rag.make_embeddings()
    # Outer try maps unexpected failures to HTTP 502 with helpful context.
    try:
        # Inner try separates user-fixable PDF issues from infra failures.
        try:
            # Split PDF to docs, embed, and build FAISS asynchronously.
            vectorstore, chunk_texts = await ingestion.pdf_bytes_to_faiss(file_bytes, embeddings)
        # Missing text or splitter anomalies → client error.
        except ValueError as exc:
            # Convert ValueError message to HTTP 400 body.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Any other exception (Ollama down, network, etc.) becomes 502 with hint text.
    except Exception as exc:
        # Surface embedding model name in error to speed up debugging.
        raise HTTPException(
            # Bad gateway: upstream dependency (Ollama) likely failed.
            status_code=502,
            # Include configured embed model name in the detail string.
            detail=f"Failed to process PDF or build embeddings (is Ollama running with `{ollama_embed_model()}` pulled?). {exc}",
        ) from exc

    # Persist the new index and mark it as the default query target.
    state.put(
        # Key rows in the state dict by file hash.
        doc_hash,
        # Bundle FAISS store, chunk strings, and filename together.
        DocumentIndexEntry(faiss=vectorstore, chunk_texts=chunk_texts, filename=filename),
    )

    # Successful fresh indexing response payload.
    return UploadResponse(
        # Same hash the client should reuse on /query.
        document_hash=doc_hash,
        # Indicate work was performed this request.
        cached=False,
        # Count chunks embedded for observability.
        chunk_count=len(chunk_texts),
        # Filename echo.
        filename=filename,
    )


# POST /query: JSON question → retrieve chunks → Ollama JSON answer → validate.
@app.post("/query", response_model=QueryResponse)
async def query_documents(
    # Parsed and validated JSON body (question, optional hash, top_k).
    body: QueryRequest,
    # Shared cache of indices.
    state: AppState = Depends(get_state),
    # Injected chat model instance for this request.
    llm: ChatOllama = Depends(get_chat_llm),
):
    # Resolve which document to query: explicit hash or most recent upload.
    doc_hash = body.document_hash or state.last_hash
    # If no uploads happened yet, fail with a clear 400 message.
    if not doc_hash:
        # Tell the user to upload first or pass document_hash explicitly.
        raise HTTPException(
            # Client error: missing prerequisite state.
            status_code=400,
            # Actionable error text for the SPA.
            detail="No document is indexed. Upload a PDF first or pass document_hash.",
        )

    # Fetch the in-memory index entry for the chosen hash.
    entry = state.get(doc_hash)
    # Unknown hash means stale client state or server restart loss.
    if entry is None:
        # 404 distinguishes “not found” from generic bad request.
        raise HTTPException(status_code=404, detail="Unknown document_hash. Upload the document again.")

    # Vector search can still fail if Ollama/embeddings break mid-request.
    try:
        # Async nearest-neighbor search in FAISS using the question text.
        retrieved = await rag.retrieve_chunks(entry.faiss, body.question, body.top_k)
    # Wrap any retrieval stack trace as a 502 with short message.
    except Exception as exc:
        # Retrieval depends on embeddings + FAISS internal state.
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc

    # Defensive guard if the store returns an empty list unexpectedly.
    if not retrieved:
        # Without chunks there is no grounded context to send to the LLM.
        raise HTTPException(status_code=502, detail="Retrieval returned no chunks.")

    # Call Ollama and parse/validate structured JSON output.
    try:
        # Run chat model and coerce response into LLMStructuredAnswer.
        structured = await rag.answer_with_context(llm, body.question, retrieved)
    # JSON parsing / schema issues from rag layer bubble as ValueError messages.
    except ValueError as exc:
        # Typically “non-JSON content” or malformed JSON object.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    # Other failures: Ollama offline, model missing, etc.
    except Exception as exc:
        # Log the full traceback so the real cause (timeout, 404, etc.) appears in server logs.
        logger.exception("Chat completion failed")
        # Mention configured LLM model name for quick operator checks.
        raise HTTPException(
            # Upstream generation failure.
            status_code=502,
            # Actionable hint referencing the chat model tag and exception class.
            detail=f"Chat completion failed (model `{ollama_llm_model()}`): {type(exc).__name__}: {exc}",
        ) from exc

    # Build the outward API response including echo metadata fields.
    return QueryResponse(
        # Natural language answer for the chat bubble.
        answer=structured.answer,
        # Prefer model-provided sources; if empty, use retrieved chunk texts.
        source_chunks=structured.source_chunks if structured.source_chunks else retrieved,
        # Float confidence for UI meter/badge.
        confidence=structured.confidence,
        # Echo which document was queried.
        document_hash=doc_hash,
        # Echo k used for retrieval transparency.
        top_k=body.top_k,
    )


# GET /health: simple liveness probe for load balancers and dev sanity checks.
@app.get("/health")
async def health():
    # Minimal JSON payload indicating the server process is responsive.
    return {"status": "ok"}
