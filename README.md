# Document Q&A Platform

Full-stack application for uploading PDFs, embedding them into a FAISS vector index with a **local Ollama** embedding model, and answering questions with a **local Ollama LLM** (default **LLaVA**). The FastAPI backend caches indexes in memory by SHA-256 of the raw file bytes so repeat uploads skip re-embedding.

## Layout

- `backend/` — FastAPI app (`app/main.py`), in-memory FAISS cache (`app/state.py`), Ollama settings (`app/config.py`), ingestion and RAG helpers (`app/services/`).
- `frontend/` — Vite + React SPA with PDF upload, chat, and a sources panel.
- `requirements.txt` — Python dependencies (install into a virtual environment at the repo root).

## Prerequisites

- Python 3.11+ recommended (3.13 works with the pinned wheels used here).
- Node.js 18+ and npm.
- [Ollama](https://ollama.com/) installed and running (`ollama serve`, usually on `http://127.0.0.1:11434`).

### Pull models

LLaVA answers questions but does **not** supply text embeddings. Pull an embedding model and the chat model (defaults match `.env.example`):

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

You can change names via `OLLAMA_EMBED_MODEL` and `OLLAMA_LLM_MODEL` in `.env`.

## Backend setup

1. Create and activate a virtual environment from the project root (examples shown for Windows PowerShell):

   ```powershell
   cd "d:\rag project"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Configure Ollama (optional). Copy the example env file to `.env` in the **project root** (same folder as `requirements.txt`):

   ```powershell
   copy .env.example .env
   ```

   Defaults work for a local Ollama on port 11434. Edit `.env` if your base URL or model tags differ.

   `python-dotenv` loads this automatically when the app starts.

3. Run the API (must use the `backend` directory on `PYTHONPATH` so `app` resolves):

   ```powershell
   cd backend
   python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

   Health check: `http://127.0.0.1:8000/health`.

## Frontend setup

1. Install dependencies:

   ```powershell
   cd "d:\rag project\frontend"
   npm install
   ```

2. Start the dev server (proxies `/upload`, `/query`, and `/health` to `http://127.0.0.1:8000`):

   ```powershell
   npm run dev
   ```

   Open the printed local URL (default `http://127.0.0.1:5173`).

3. Optional: if the API is on another origin, create `frontend/.env`:

   ```
   VITE_API_BASE=http://127.0.0.1:8000
   ```

   Then adjust `vite.config.js` or call the backend directly without the dev proxy.

## API overview

| Method | Path      | Description |
|--------|-----------|-------------|
| POST   | `/upload` | Multipart form field `file`: PDF. Returns `document_hash`, `cached`, `chunk_count`, `filename`. |
| POST   | `/query`  | JSON body: `question` (required), `document_hash` (optional; defaults to last successful upload), `top_k` (default 3). Returns `answer`, `source_chunks`, `confidence`, `document_hash`, `top_k`. |

Both route handlers are `async`. Unsupported file types, empty bodies, and missing indices return structured HTTP errors.

## CORS

`CORSMiddleware` allows local Vite and Create React App default ports. Add your production frontend origin in `backend/app/main.py` as needed.

## Notes

- Indexes and chunk metadata live **in memory** only; restarting the server clears them (hashes still dedupe re-uploads within a session).
- Chunking uses LangChain `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with roughly **220** tokens per chunk and **40** tokens overlap, and starts from per-page text for better granularity.
- Embeddings and chat both call **Ollama** over HTTP; chat uses Ollama `format="json"` with **LLaVA** by default, and responses are validated with Pydantic (`answer`, `source_chunks`, `confidence`).
- If you change the embedding model, rebuild or restart before relying on old `document_hash` entries (indexes are tied to the embedding space you used when uploading).
