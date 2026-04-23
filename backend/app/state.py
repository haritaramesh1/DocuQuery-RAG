"""In-memory application state for FAISS indices keyed by document hash."""

# Postponed annotations for forward-compatible typing.
from __future__ import annotations

# dataclass reduces boilerplate for simple data containers; field() customizes defaults.
from dataclasses import dataclass, field
# Lock serializes reads/writes to the in-memory dict from concurrent async workers.
from threading import Lock
# Any used only for the dependency helper return type flexibility.
from typing import Any

# FAISS vector store type from LangChain (holds vectors + doc texts).
from langchain_community.vectorstores import FAISS


# One row in the cache: vector index plus human-readable chunks and display name.
@dataclass
class DocumentIndexEntry:
    """Cached vector index and chunk texts for one uploaded document."""

    # LangChain FAISS store bound to the embedding model used at index time.
    faiss: FAISS
    # Parallel list of chunk strings (same order as embedded docs) for debugging/UI.
    chunk_texts: list[str]
    # Original filename for responses and UI.
    filename: str


# Global singleton state; FastAPI Depends(get_state) returns this instance.
@dataclass
class AppState:
    """Global state shared across requests (thread-safe for concurrent access)."""

    # Mutex protecting _indices and _last_hash mutations from race conditions.
    _lock: Lock = field(default_factory=Lock)
    # Map document_hash -> DocumentIndexEntry for O(1) lookup after upload.
    _indices: dict[str, DocumentIndexEntry] = field(default_factory=dict)
    # Most recently stored document hash when document_hash is omitted on /query.
    _last_hash: str | None = None

    # Thread-safe read of a cached index by hash; returns None if unknown.
    def get(self, doc_hash: str) -> DocumentIndexEntry | None:
        # Acquire lock for duration of dict read.
        with self._lock:
            # Return cached entry or None.
            return self._indices.get(doc_hash)

    # Thread-safe insert/replace of an index and update of “last” pointer.
    def put(self, doc_hash: str, entry: DocumentIndexEntry) -> None:
        # Serialize writers so two uploads don’t corrupt the dict.
        with self._lock:
            # Store or overwrite the entry for this file hash.
            self._indices[doc_hash] = entry
            # Remember as default target for subsequent queries without explicit hash.
            self._last_hash = doc_hash

    # Property accessor for last_hash with lock (read-only from outside).
    @property
    def last_hash(self) -> str | None:
        # Locked read so callers never see a torn write.
        with self._lock:
            # Return last successful upload hash or None if never indexed.
            return self._last_hash

    # Optional pattern: wrap self in a zero-arg callable for FastAPI Depends.
    def as_dependency(self) -> Any:
        """FastAPI dependency: returns this singleton."""

        # Inner closure captures self and returns it when FastAPI calls the dependency.
        def _get_state() -> AppState:
            # Return the shared AppState instance.
            return self

        # Expose the callable to FastAPI’s dependency injection system.
        return _get_state


# Module-level singleton used across all requests in this process.
app_state = AppState()
