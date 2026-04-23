"""Pydantic request/response models."""

# Enable postponed annotation evaluation (PEP 563 style for this module).
from __future__ import annotations

# Pydantic: data validation; Field sets constraints; field_validator customizes parsing.
from pydantic import BaseModel, Field, field_validator


# JSON shape returned to the client after POST /upload succeeds or cache hits.
class UploadResponse(BaseModel):
    """Response after processing an upload (new or cache hit)."""

    # SHA-256 hex digest of raw PDF bytes; client can send this back on /query.
    document_hash: str
    # True if this hash was already indexed in memory (skipped re-embedding).
    cached: bool
    # Number of text chunks stored in FAISS for this document.
    chunk_count: int
    # Original uploaded filename (for UI display).
    filename: str


# JSON body accepted by POST /query.
class QueryRequest(BaseModel):
    """User question against an indexed document."""

    # User question; min_length=1 rejects empty string at validation time.
    question: str = Field(..., min_length=1)
    # Optional: target a specific upload by hash; None means “use last uploaded doc”.
    document_hash: str | None = None
    # How many nearest chunks to retrieve from FAISS.
    top_k: int = Field(6, ge=1, le=20)

    # Runs after initial parsing to normalize and reject whitespace-only questions.
    @field_validator("question")
    @classmethod
    def strip_not_empty(cls, v: str) -> str:
        # Remove leading/trailing whitespace from the question.
        s = v.strip()
        # Reject if nothing meaningful remains after stripping.
        if not s:
            raise ValueError("Question cannot be empty or whitespace only")
        # Return the cleaned string for downstream use.
        return s


# Expected keys inside the LLM’s JSON reply before building QueryResponse.
class LLMStructuredAnswer(BaseModel):
    """Validated structured output from the chat model."""

    # Natural-language answer grounded in context (or “I don’t know”).
    answer: str
    # Verbatim or cited passages the model used; default empty list if omitted in JSON.
    source_chunks: list[str] = Field(default_factory=list)
    # Model self-rated confidence strictly between 0 and 1 inclusive.
    confidence: float = Field(..., ge=0.0, le=1.0)


# JSON shape returned to the client after POST /query.
class QueryResponse(BaseModel):
    """API response including retrieval-backed sources."""

    # Final answer string shown in the UI.
    answer: str
    # Chunks shown in the “sources” panel (may fall back to retrieved text server-side).
    source_chunks: list[str]
    # Same confidence field as validated from the LLM JSON.
    confidence: float
    # Echoes which document index was queried.
    document_hash: str
    # Echoes how many neighbors were requested for retrieval.
    top_k: int
