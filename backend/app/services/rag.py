"""Embedding queries, FAISS retrieval, and structured chat via local Ollama."""

# Postponed annotations for typing convenience.
from __future__ import annotations

# json parses the model’s JSON string output into Python dicts.
import json
# re strips optional ```json fences if the model wraps JSON in markdown.
import re
# logging surfaces real Ollama errors instead of opaque generic 502s.
import logging

# LangChain wrapper for Ollama chat models (local LLM).
from langchain_community.chat_models import ChatOllama
# LangChain wrapper for Ollama embedding HTTP API.
from langchain_community.embeddings import OllamaEmbeddings
# FAISS vector store for similarity search over chunk embeddings.
from langchain_community.vectorstores import FAISS
# Chat message types for system vs user turns.
from langchain_core.messages import HumanMessage, SystemMessage
# Pydantic validation error type for structured LLM JSON recovery.
from pydantic import ValidationError

# Centralized env-backed Ollama URL and model names.
from app.config import ollama_base_url, ollama_embed_model, ollama_llm_model, ollama_num_predict
# Schema for validating parsed LLM JSON into Python fields.
from app.schemas import LLMStructuredAnswer

# Module logger so real errors show up in uvicorn output.
logger = logging.getLogger(__name__)

# System prompt for non-streaming JSON mode (structured answer).
SYSTEM_INSTRUCTION = """You explain passages from a user's uploaded document.

Rules:
- Use the numbered context passages below to answer.
- Do NOT copy a passage verbatim as the answer. Write your own explanation.
- If the user asks to "explain ...", give 3-5 sentences that say what the topic is, define the key terms, and describe what is being asked or done.
- If the user asks a direct factual question, answer in 1-3 sentences.
- You may combine information from multiple passages.
- Only answer "I don't know" if the context truly does not contain the answer.

Reply with ONE JSON object and nothing else:
{"answer": "<your explanation>", "source_chunks": ["<short quote 1>", "<short quote 2>"], "confidence": <0-1>}
"""

# System prompt for streaming mode (plain prose — no JSON wrapping).
STREAM_SYSTEM_INSTRUCTION = """You explain passages from a user's uploaded document.

Rules:
- Use the numbered context passages below to answer.
- Do NOT copy a passage verbatim. Write your own explanation in natural language.
- If the user asks to "explain ...", give 3-5 sentences covering what the topic is, key terms, and what is being asked or done.
- For direct factual questions, answer in 1-3 sentences.
- You may combine information from multiple passages.
- Only say "I don't know" if the context truly does not contain the answer.

Answer in plain prose. Do NOT output JSON, bullet markup, or markdown code fences."""


# Formats retrieved chunk strings as numbered blocks for the prompt.
def _format_context(chunks: list[str]) -> str:
    # Collect each numbered passage as its own line block.
    lines: list[str] = []
    # Enumerate from 1 so the model can refer to “[1]” style indices.
    for i, c in enumerate(chunks, start=1):
        # Prefix chunk text with bracketed index for clarity.
        lines.append(f"[{i}] {c}")
    # Separate passages with blank lines for readability in the prompt.
    return "\n\n".join(lines)


# Normalizes LangChain message content to a plain string (handles list segments).
def _message_content_to_text(content: object) -> str:
    # Fast path: already a string completion body.
    if isinstance(content, str):
        # Return text as-is.
        return content
    # Some models return a list of content parts (e.g. multimodal wrappers).
    if isinstance(content, list):
        # Buffer concatenated text fragments.
        parts: list[str] = []
        # Walk each segment in order.
        for block in content:
            # Plain string fragment.
            if isinstance(block, str):
                # Append raw string fragment.
                parts.append(block)
            # Structured text block dict from some providers.
            elif isinstance(block, dict) and block.get("type") == "text":
                # Extract nested text field safely as string.
                parts.append(str(block.get("text", "")))
        # Join all fragments into one JSON-bearing string.
        return "".join(parts)
    # Fallback: coerce unknown content types to string.
    return str(content)


# Parses model output into a dict, stripping markdown fences if present.
def _extract_json_object(raw: str) -> dict:
    # Trim outer whitespace from the raw model output.
    s = raw.strip()
    # Detect common ```json ... ``` wrapping even when format=json is set.
    if s.startswith("```"):
        # Remove opening fence with optional “json” language tag.
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        # Remove closing triple backtick fence.
        s = re.sub(r"\s*```$", "", s)
        # Re-strip after fence removal.
        s = s.strip()
    # Try strict JSON parsing of the cleaned string.
    try:
        # Parse JSON text into Python values.
        data = json.loads(s)
    # Map JSON syntax errors to a single application error type upstream.
    except json.JSONDecodeError as exc:
        # Signal invalid JSON to the FastAPI handler.
        raise ValueError("Model returned non-JSON content") from exc
    # Ensure top-level JSON is an object with keys (not array/primitive).
    if not isinstance(data, dict):
        # Reject arrays/strings/numbers at root (can’t map to Pydantic model).
        raise ValueError("Model JSON was not an object")
    # Return the dict for Pydantic validation.
    return data


# Runs async MMR search over FAISS for diverse, relevant chunks.
async def retrieve_chunks(
    store: FAISS,
    question: str,
    top_k: int,
) -> list[str]:
    # Cap k so small CPU-bound LLMs don't get overloaded with context.
    effective_k = min(top_k, 4)
    # MMR balances semantic similarity with diversity, which improves answer coverage.
    try:
        docs = await store.amax_marginal_relevance_search(
            question,
            k=effective_k,
            fetch_k=max(effective_k * 4, 16),
            lambda_mult=0.6,
        )
    except Exception:
        # Fallback to plain similarity search if MMR is unavailable.
        pairs = await store.asimilarity_search_with_score(question, k=effective_k)
        docs = [d for d, _ in pairs]
    # Return raw chunk text for prompt construction.
    return [d.page_content for d in docs]


# Calls Ollama chat with system+user messages; validates JSON into LLMStructuredAnswer.
async def answer_with_context(
    llm: ChatOllama,
    question: str,
    retrieved_chunks: list[str],
) -> LLMStructuredAnswer:
    # Turn retrieved chunk list into numbered context for the user message.
    context_block = _format_context(retrieved_chunks)
    # Build the user turn: static headings + context + the actual question.
    user_msg = f"""Context passages:

{context_block}

Question: {question}
"""

    # Send system instructions plus the constructed user content to the local LLM.
    try:
        response = await llm.ainvoke(
            [
                # High-priority system rules (grounding + JSON shape).
                SystemMessage(content=SYSTEM_INSTRUCTION),
                # Actual task input containing passages and question.
                HumanMessage(content=user_msg),
            ],
        )
    except Exception:
        # Log the full traceback so the root cause shows up in uvicorn logs.
        logger.exception("ChatOllama call failed")
        raise
    # Normalize AIMessage.content to a single string for JSON parsing.
    raw_text = _message_content_to_text(response.content)
    # Parse JSON object from model text, tolerating markdown fences.
    data = _extract_json_object(raw_text)

    # First try strict Pydantic validation of all fields together.
    try:
        # Construct validated LLMStructuredAnswer from dict keys.
        parsed = LLMStructuredAnswer.model_validate(data)
    # If strict validation fails, attempt a looser manual salvage path.
    except ValidationError:
        # Only salvage when at least an “answer” key exists.
        if isinstance(data, dict) and "answer" in data:
            # Coerce answer to trimmed string.
            answer = str(data.get("answer", "")).strip()
            # Parse confidence with float(); default 0 if missing/invalid later clamped.
            conf = float(data.get("confidence", 0.0))
            # Read source_chunks key or substitute empty list.
            chunks = data.get("source_chunks") or []
            # Ensure chunks is a list before iterating.
            if not isinstance(chunks, list):
                # Reset to empty if the model returned a wrong type.
                chunks = []
            # Manually build Pydantic model with clamped confidence range.
            parsed = LLMStructuredAnswer(
                # Salvaged natural-language answer.
                answer=answer,
                # Coerce each chunk element to string for schema compliance.
                source_chunks=[str(x) for x in chunks],
                # Clamp confidence into [0,1] even if model overshoots.
                confidence=max(0.0, min(1.0, conf)),
            )
        else:
            # No recoverable fields; re-raise to surface a 502 upstream.
            raise

    # If model omitted sources but did answer, fall back to retrieved chunk texts.
    if not parsed.source_chunks and parsed.answer.strip().lower() != "i don't know":
        # Immutable update via model_copy to inject retrieval fallback list.
        parsed = parsed.model_copy(update={"source_chunks": list(retrieved_chunks)})

    # Return validated structured answer for API response mapping.
    return parsed


# Builds a configured OllamaEmbeddings client for indexing and query embedding.
def make_embeddings() -> OllamaEmbeddings:
    # Instantiate embeddings with model tag and base URL from environment.
    return OllamaEmbeddings(
        # Embedding model pulled in Ollama (e.g. nomic-embed-text).
        model=ollama_embed_model(),
        # HTTP endpoint for Ollama (embeddings API).
        base_url=ollama_base_url(),
    )


# Builds a ChatOllama client with JSON format for structured answers.
def make_chat_ollama() -> ChatOllama:
    # Return chat model configured for local inference and JSON-only completions.
    return ChatOllama(
        # Chat model tag pulled in Ollama (default is a faster text model).
        model=ollama_llm_model(),
        # Same Ollama HTTP server as embeddings.
        base_url=ollama_base_url(),
        # Deterministic decoding improves consistency and often speeds up generation.
        temperature=0.0,
        # Hard cap answer length to reduce latency on local models.
        num_predict=ollama_num_predict(),
        # Larger context window so top_k chunks + prompt fit without truncation.
        num_ctx=4096,
        # Keep model warm in memory between requests for faster follow-up answers.
        keep_alive="30m",
        # Allow enough time for first-token latency after a cold load on CPU.
        timeout=300.0,
        # Ask Ollama to emit JSON object text for easier parsing.
        format="json",
    )


# Builds a streaming ChatOllama client (plain text, no JSON format constraint).
def make_streaming_chat_ollama() -> ChatOllama:
    # Same base settings as the JSON variant but without format="json" so tokens
    # can be emitted incrementally as natural prose.
    return ChatOllama(
        model=ollama_llm_model(),
        base_url=ollama_base_url(),
        temperature=0.1,
        num_predict=ollama_num_predict(),
        num_ctx=4096,
        keep_alive="30m",
        timeout=300.0,
        streaming=True,
    )


# Async generator: yields individual text tokens/chunks from the LLM as they stream.
async def stream_answer(
    llm: ChatOllama,
    question: str,
    retrieved_chunks: list[str],
):
    # Numbered context block used as grounding input.
    context_block = _format_context(retrieved_chunks)
    # User turn mirrors the non-streaming endpoint for prompt parity.
    user_msg = f"""Context passages:

{context_block}

Question: {question}
"""
    # astream yields AIMessageChunk objects one token (or small group) at a time.
    try:
        async for chunk in llm.astream(
            [
                SystemMessage(content=STREAM_SYSTEM_INSTRUCTION),
                HumanMessage(content=user_msg),
            ]
        ):
            # Convert any possible multi-part content to a flat string.
            text = _message_content_to_text(chunk.content)
            # Skip empty keepalive chunks so the UI only sees real text.
            if text:
                yield text
    except Exception:
        # Surface the full traceback in uvicorn logs for debugging.
        logger.exception("ChatOllama stream failed")
        raise
