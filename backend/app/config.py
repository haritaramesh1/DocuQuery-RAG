"""Ollama endpoints and model names (read from environment)."""

# Postpone evaluation of type hints until runtime (allows forward references if needed).
from __future__ import annotations

# Standard library: read process environment variables.
import os


# Returns the Ollama HTTP base URL, e.g. http://127.0.0.1:11434 (no trailing slash).
def ollama_base_url() -> str:
    # Read OLLAMA_BASE_URL or default to local Ollama; strip trailing slash for consistent joins.
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


# Returns the Ollama model tag used to embed text for FAISS (must be pulled in Ollama).
def ollama_embed_model() -> str:
    # Read OLLAMA_EMBED_MODEL or default to a common small embedding model.
    return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# Returns the Ollama model tag used for JSON chat answers.
def ollama_llm_model() -> str:
    # Default to a faster text model; llava is much slower and not needed for plain PDF QA.
    return os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")


# Returns max generated tokens per answer; lower values reduce latency.
def ollama_num_predict() -> int:
    # Balanced default: room for a 3-5 sentence JSON answer without overlong generation.
    return int(os.getenv("OLLAMA_NUM_PREDICT", "384"))
