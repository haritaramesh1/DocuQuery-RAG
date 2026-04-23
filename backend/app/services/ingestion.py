"""PDF loading, hashing, splitting, and FAISS index construction."""

# Postponed evaluation of annotations in this module.
from __future__ import annotations

# hashlib provides SHA-256 for stable document fingerprints / cache keys.
import hashlib
# BytesIO lets pypdf read PDF bytes from memory without a temp file.
from io import BytesIO

# LangChain community FAISS integration for vector search.
from langchain_community.vectorstores import FAISS
# Semantic chunking based on embedding-distance breakpoints.
from langchain_experimental.text_splitter import SemanticChunker
# Document is the standard LangChain wrapper for page_content metadata.
from langchain_core.documents import Document
# Protocol/type for any embedding implementation (Ollama, OpenAI, etc.).
from langchain_core.embeddings import Embeddings
# Token-aware recursive splitting of long text into overlapping chunks.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Pure-Python PDF parser used to extract plain text from bytes.
from pypdf import PdfReader


# Target chunk size measured in tokens (via tiktoken encoder below).
CHUNK_SIZE_TOKENS = 180
# Overlap between consecutive chunks to preserve boundary context.
CHUNK_OVERLAP_TOKENS = 40


# Computes hex SHA-256 over raw bytes for deduplication keys.
def sha256_bytes(data: bytes) -> str:
    # Hash the bytes with SHA-256 and format as lowercase hex string.
    return hashlib.sha256(data).hexdigest()


# Reads every page of a PDF in memory and returns per-page text blocks.
def load_pdf_pages(file_bytes: bytes) -> list[str]:
    # Wrap bytes as a seekable file-like object for PdfReader.
    reader = PdfReader(BytesIO(file_bytes))
    # Accumulate non-empty page texts in document order.
    parts: list[str] = []
    # Iterate pages in order.
    for page in reader.pages:
        # extract_text may return None for scanned pages; coerce to "" and strip noise.
        text = (page.extract_text() or "").strip()
        # Keep only pages with real text so blank pages do not create empty chunks.
        if text:
            parts.append(text)
    # Return list to preserve natural page boundaries before token splitting.
    return parts


# Factory for recursive fallback splitter (fine-grained chunks with overlap).
def build_text_splitter() -> RecursiveCharacterTextSplitter:
    # Use tiktoken’s cl100k_base via model_name for counting tokens consistently.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # Encoder family compatible with many OpenAI-style models (counting only).
        model_name="gpt-3.5-turbo",
        # Maximum tokens per chunk before splitting on separators.
        chunk_size=CHUNK_SIZE_TOKENS,
        # Tokens repeated between adjacent chunks for smoother retrieval.
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )


# Runs semantic chunking on page documents.
def semantic_split_documents(page_docs: list[Document], embeddings: Embeddings) -> list[Document]:
    # SemanticChunker groups adjacent sentences by embedding similarity.
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=82,
    )
    # Returns variable-length chunks aligned to semantic topic shifts.
    return splitter.split_documents(page_docs)


# End-to-end: PDF bytes → semantic/fallback chunks → async FAISS index + chunk list.
async def pdf_bytes_to_faiss(
    file_bytes: bytes,
    embeddings: Embeddings,
) -> tuple[FAISS, list[str]]:
    """Split PDF text into token-sized chunks and build an async-compatible FAISS index."""
    # Extract one text block per PDF page for better chunk granularity.
    pages = load_pdf_pages(file_bytes)
    # Fail fast if the PDF has no selectable text (e.g. blank or image-only).
    if not pages:
        raise ValueError("No extractable text found in PDF")

    # Create one source document per page first to avoid over-merging short pages.
    page_docs = [Document(page_content=page_text, metadata={"page": i + 1}) for i, page_text in enumerate(pages)]
    # Try semantic chunking first for topic-coherent chunks.
    docs = semantic_split_documents(page_docs, embeddings)
    # Semantic mode can occasionally under-split short docs; enforce useful minimum granularity.
    min_expected_chunks = max(6, len(page_docs) * 2)
    if len(docs) < min_expected_chunks:
        # Fall back to deterministic recursive chunking when semantic output is too coarse.
        splitter = build_text_splitter()
        docs = splitter.split_documents(page_docs)
    # Guard against pathological splitter output.
    if not docs:
        raise ValueError("Document produced zero chunks after splitting")

    # Async embed all chunk documents and construct the in-memory FAISS index.
    store = await FAISS.afrom_documents(docs, embeddings)
    # Preserve plain strings for UI / response parity with vector rows.
    chunk_texts = [d.page_content for d in docs]
    # Return both the searchable store and the parallel text list.
    return store, chunk_texts
