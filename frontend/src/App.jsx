import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

// Resolves the backend base URL; empty string uses the Vite dev proxy.
const apiBase = () => import.meta.env.VITE_API_BASE?.replace(/\/$/, "") || "";

async function postJson(path, body) {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text || "Invalid JSON from server" };
  }
  if (!res.ok) {
    const detail = data?.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
        ? detail.map((d) => d.msg).join("; ")
        : JSON.stringify(data);
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return data;
}

async function postUpload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${apiBase()}/upload`, { method: "POST", body: fd });
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = data?.detail;
    const msg = typeof detail === "string" ? detail : JSON.stringify(data);
    throw new Error(msg || `Upload failed (${res.status})`);
  }
  return data;
}

// ──────────────────────────────────────────────────────────────
// Icons (inline SVG so there are no extra deps).
// ──────────────────────────────────────────────────────────────
const IconUpload = (p) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
);

const IconSend = (p) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const IconDoc = (p) => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);

const IconChat = (p) => (
  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
  </svg>
);

// ──────────────────────────────────────────────────────────────
// Main component
// ──────────────────────────────────────────────────────────────
export default function App() {
  // Document / upload state.
  const [documentHash, setDocumentHash] = useState(null);
  const [filename, setFilename] = useState(null);
  const [cached, setCached] = useState(null);
  const [chunkCount, setChunkCount] = useState(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Chat state.
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [queryBusy, setQueryBusy] = useState(false);
  const [queryError, setQueryError] = useState(null);
  const [selectedSources, setSelectedSources] = useState(null);

  const fileInputRef = useRef(null);
  const chatBodyRef = useRef(null);
  const textareaRef = useRef(null);

  const hasDoc = Boolean(documentHash);

  // Auto-scroll chat to bottom on new messages / typing indicator.
  useEffect(() => {
    const el = chatBodyRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, queryBusy]);

  // Auto-grow the composer textarea with content.
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [input]);

  const onFiles = useCallback(async (files) => {
    const file = files?.[0];
    if (!file) return;
    setUploadError(null);
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }
    setUploadBusy(true);
    try {
      const data = await postUpload(file);
      setDocumentHash(data.document_hash);
      setFilename(data.filename);
      setCached(data.cached);
      setChunkCount(data.chunk_count);
      setMessages((m) => [
        ...m,
        {
          role: "system",
          content: data.cached
            ? `Loaded cached index for "${data.filename}" · ${data.chunk_count} chunks`
            : `Indexed "${data.filename}" into ${data.chunk_count} chunks`,
        },
      ]);
    } catch (e) {
      setUploadError(e.message || String(e));
    } finally {
      setUploadBusy(false);
    }
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      onFiles(e.dataTransfer.files);
    },
    [onFiles],
  );

  const sendQuestion = useCallback(async () => {
    const q = input.trim();
    setQueryError(null);
    if (!q) {
      setQueryError("Please enter a question.");
      return;
    }
    if (!documentHash) {
      setQueryError("Upload a PDF first.");
      return;
    }
    setQueryBusy(true);
    setInput("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    try {
      const data = await postJson("/query", {
        question: q,
        document_hash: documentHash,
        top_k: 6,
      });
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data.answer,
          confidence: data.confidence,
          sources: data.source_chunks,
        },
      ]);
      setSelectedSources(data.source_chunks);
    } catch (e) {
      setQueryError(e.message || String(e));
      setMessages((m) => [...m, { role: "error", content: e.message || String(e) }]);
    } finally {
      setQueryBusy(false);
    }
  }, [documentHash, input]);

  // The most recent assistant's sources drive the right-sidebar panel.
  const lastAssistantSources = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === "assistant" && messages[i].sources) return messages[i].sources;
    }
    return null;
  }, [messages]);

  const sourcesToShow = selectedSources ?? lastAssistantSources;

  return (
    <div className="app">
      {/* ─── Header ──────────────────────────────────────────── */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">DQ</div>
          <div className="brand-text">
            <h1>DocuQuery</h1>
            <p className="tagline">Ask grounded questions about your PDFs</p>
          </div>
        </div>
        <div className="header-status">
          <span className={`status-dot ${hasDoc ? "ok" : ""}`} />
          {hasDoc ? (
            <>
              <span className="filename" title={filename}>{filename}</span>
              <span className="badge blue">{chunkCount} chunks</span>
              {cached != null ? (
                <span className="badge ok">{cached ? "cached" : "indexed"}</span>
              ) : null}
            </>
          ) : (
            <span>No document loaded</span>
          )}
        </div>
      </header>

      <div className="workspace">
        {/* ─── Sidebar ──────────────────────────────────────── */}
        <aside className="sidebar">
          {/* Upload card */}
          <section className="card">
            <h3>Document</h3>
            <div
              className={`dropzone ${dragActive ? "drag" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
            >
              <div className="drop-icon">
                <IconUpload />
              </div>
              <p className="primary-text">
                {uploadBusy ? "Processing…" : "Drop PDF or click to browse"}
              </p>
              <p className="hint">Only .pdf files, stored in memory</p>
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf,.pdf"
                className="hidden-input"
                onChange={(e) => onFiles(e.target.files)}
              />
            </div>
            {uploadError ? <p className="inline-error" style={{ marginTop: 10 }}>{uploadError}</p> : null}
          </section>

          {/* Document metadata */}
          {hasDoc ? (
            <section className="card">
              <h3>Loaded file</h3>
              <div className="doc-meta">
                <div className="doc-row">
                  <span className="label">
                    <IconDoc style={{ verticalAlign: "-4px", marginRight: 6 }} />
                    File
                  </span>
                  <span className="value" title={filename}>{filename}</span>
                </div>
                <div className="doc-row">
                  <span className="label">Chunks</span>
                  <span className="value">{chunkCount}</span>
                </div>
                <div className="doc-row">
                  <span className="label">Status</span>
                  <span className="value">
                    {cached ? <span className="badge ok">cache hit</span> : <span className="badge blue">fresh embed</span>}
                  </span>
                </div>
              </div>
            </section>
          ) : null}

          {/* Sources */}
          <section className="card sources-card">
            <div className="sources-head">
              <h3 style={{ margin: 0 }}>Sources</h3>
              {selectedSources && lastAssistantSources && selectedSources !== lastAssistantSources ? (
                <button type="button" className="btn ghost" style={{ padding: "4px 10px", fontSize: "0.75rem" }} onClick={() => setSelectedSources(null)}>
                  Show latest
                </button>
              ) : null}
            </div>
            <div className="sources-list">
              {sourcesToShow?.length ? (
                sourcesToShow.map((c, i) => (
                  <article key={i} className="source-chunk">
                    <div className="source-label">Chunk {i + 1}</div>
                    <p>{c}</p>
                  </article>
                ))
              ) : (
                <p className="empty">Ask a question to see which passages the answer was grounded in.</p>
              )}
            </div>
          </section>
        </aside>

        {/* ─── Chat panel ──────────────────────────────────── */}
        <main className="chat-panel">
          <div className="chat-header">
            <div>
              <h2>Conversation</h2>
              <div className="subtext">
                {hasDoc ? "Ask anything about the loaded document" : "Upload a document to start chatting"}
              </div>
            </div>
          </div>

          {queryError ? <div className="chat-error">{queryError}</div> : null}

          <div className="chat-body" ref={chatBodyRef}>
            {messages.length === 0 ? (
              <div className="chat-empty">
                <div className="illus"><IconChat /></div>
                <h4>Start a conversation</h4>
                <p>Upload a PDF on the left, then ask questions. Answers will cite the passages they used.</p>
              </div>
            ) : (
              <ul className="messages">
                {messages.map((m, idx) => {
                  if (m.role === "system") {
                    return (
                      <li key={idx} className="msg system">
                        <div className="bubble">{m.content}</div>
                      </li>
                    );
                  }
                  if (m.role === "error") {
                    return (
                      <li key={idx} className="msg error">
                        <div className="avatar assistant">!</div>
                        <div className="bubble">{m.content}</div>
                      </li>
                    );
                  }
                  if (m.role === "user") {
                    return (
                      <li key={idx} className="msg user">
                        <div className="bubble">{m.content}</div>
                        <div className="avatar user">You</div>
                      </li>
                    );
                  }
                  // assistant
                  return (
                    <li key={idx} className="msg assistant">
                      <div className="avatar assistant">AI</div>
                      <div>
                        <div className="bubble">{m.content}</div>
                        {(m.confidence != null || m.sources?.length) ? (
                          <div className="msg-footer">
                            {m.confidence != null ? (
                              <span className="confidence-bar" title="Answer confidence">
                                <span className="bar-track">
                                  <span className="bar-fill" style={{ width: `${Math.round((m.confidence || 0) * 100)}%` }} />
                                </span>
                                {(m.confidence * 100).toFixed(0)}%
                              </span>
                            ) : null}
                            {m.sources?.length ? (
                              <button type="button" className="source-button" onClick={() => setSelectedSources(m.sources)}>
                                View {m.sources.length} source{m.sources.length === 1 ? "" : "s"}
                              </button>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    </li>
                  );
                })}
                {queryBusy ? (
                  <li className="msg assistant">
                    <div className="avatar assistant">AI</div>
                    <div className="bubble">
                      <div className="typing"><span /><span /><span /></div>
                    </div>
                  </li>
                ) : null}
              </ul>
            )}
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void sendQuestion();
            }}
          >
            <div className="composer-inner">
              <textarea
                ref={textareaRef}
                rows={1}
                placeholder={hasDoc ? "Ask a question about the document…" : "Upload a PDF to begin"}
                value={input}
                disabled={!hasDoc || queryBusy}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void sendQuestion();
                  }
                }}
              />
              <button type="submit" className="btn primary send-btn" disabled={!hasDoc || queryBusy || !input.trim()} aria-label="Send">
                <IconSend />
              </button>
            </div>
          </form>
        </main>
      </div>
    </div>
  );
}
