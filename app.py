import os

import httpx
import streamlit as st

from src.config import CHAT_MODEL, DATA_DIR, EMBEDDING_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL
from src.loader import SUPPORTED_EXTENSIONS
from src.rag_pipeline import RAGPipeline
from src import history

OLLAMA_HELP = (
    f"Can't reach Ollama at {OLLAMA_BASE_URL}. Install it from https://ollama.com, "
    f"run `ollama pull {CHAT_MODEL}`, then make sure the Ollama app/service is running."
)
GEMINI_HELP = (
    "Gemini request failed. Check that GEMINI_API_KEY in your .env file is set and valid "
    "(get one free at https://aistudio.google.com/apikey)."
)

FILE_ICONS = {".pdf": "📕", ".txt": "📄", ".md": "📝"}

st.set_page_config(page_title="RAG Chatbot", page_icon="💠", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --bg-0: #0a0c11;
        --bg-1: #0f1219;
        --surface: rgba(255,255,255,0.045);
        --surface-hi: rgba(255,255,255,0.08);
        --border: rgba(255,255,255,0.09);
        --text: #eef1f7;
        --text-dim: #93a0b8;
        --violet: #8b6bff;
        --cyan: #2fd9c7;
        --grad: linear-gradient(120deg, #8b6bff 0%, #4d8bff 55%, #2fd9c7 100%);
        --ok: #34e0a1;
        --bad: #ff6b6b;
    }

    html, body, .stApp {
        background: var(--bg-0) !important;
        color: var(--text);
        font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif;
    }
    /* Streamlit renders markdown with its own font stack — override it */
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    .stButton button, [data-testid="stChatInput"] textarea {
        font-family: 'IBM Plex Sans', ui-sans-serif, sans-serif !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        line-height: 1.65; font-size: 14.8px;
    }

    /* Dividers: hairline, not glaring white */
    hr, [data-testid="stSidebar"] hr, [data-testid="stDivider"] hr {
        border: none !important; border-top: 1px solid var(--border) !important;
        background: transparent !important; margin: 14px 0 !important; opacity: 1 !important;
    }

    /* Captions (source lines) — dimmed for hierarchy */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, small {
        color: var(--text-dim) !important;
        font-family: 'IBM Plex Mono', monospace !important; font-size: 11.5px !important;
    }

    .stApp {
        background-image:
            radial-gradient(60% 45% at 12% -8%, rgba(139,107,255,0.28) 0%, transparent 60%),
            radial-gradient(50% 40% at 105% 10%, rgba(47,217,199,0.20) 0%, transparent 60%),
            radial-gradient(40% 35% at 50% 100%, rgba(77,139,255,0.14) 0%, transparent 65%) !important;
        background-attachment: fixed !important;
    }

    h1, h2, h3, .app-title, .section-label { font-family: 'Sora', ui-sans-serif, sans-serif; }

    /* Streamlit chrome — keep it transparent so the gradient shows through */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }
    [data-testid="stHeader"] { border-bottom: none !important; }
    [data-testid="stBottom"] > div {
        background: transparent !important;
        backdrop-filter: blur(10px);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(19,16,32,0.92) 0%, rgba(11,13,19,0.92) 55%, rgba(10,17,22,0.92) 100%) !important;
        border-right: 1px solid var(--border);
        backdrop-filter: blur(16px);
    }
    [data-testid="stSidebar"] * { color: var(--text); }

    .side-label {
        display: flex; align-items: center; gap: 9px;
        font-family: 'Sora', sans-serif; font-size: 11.5px; font-weight: 600;
        letter-spacing: 0.11em; text-transform: uppercase; color: var(--text-dim);
        margin: 2px 0 10px;
    }
    .side-label::before {
        content: ""; width: 3px; height: 13px; border-radius: 2px;
        background: var(--grad); flex-shrink: 0;
    }
    .side-meta {
        font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
        color: var(--text-dim); opacity: 0.75; line-height: 1.7; margin-bottom: 2px;
    }
    .side-meta b { color: var(--cyan); font-weight: 500; }

    /* Hero */
    .hero {
        position: relative; overflow: hidden;
        padding: 30px 28px; margin-bottom: 22px; border-radius: 20px;
        background: linear-gradient(160deg, rgba(139,107,255,0.16), rgba(47,217,199,0.10));
        border: 1px solid var(--border);
        box-shadow: 0 20px 60px -20px rgba(80,60,200,0.35);
    }
    .hero::after {
        content: ""; position: absolute; inset: -40%;
        background: conic-gradient(from 0deg, transparent 0%, rgba(139,107,255,0.10) 20%, transparent 40%);
        animation: spin 14s linear infinite; pointer-events: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) { .hero::after { animation: none; } }

    .hero-row { position: relative; z-index: 1; display: flex; align-items: center; gap: 16px; }
    .hero-badge {
        width: 52px; height: 52px; border-radius: 14px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center; font-size: 26px;
        background: var(--grad); box-shadow: 0 10px 24px -8px rgba(139,107,255,0.6);
    }
    .hero h1 {
        font-family: 'Sora', ui-sans-serif, sans-serif !important;
        font-size: 26px !important; font-weight: 800 !important; margin: 0 !important;
        letter-spacing: -0.01em; padding: 0 !important;
        background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero p { margin: 4px 0 0; color: var(--text-dim); font-size: 13.5px; }

    .pill-row { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
    .cap-pill {
        font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; font-weight: 500;
        padding: 5px 11px; border-radius: 999px; color: var(--text-dim);
        background: var(--surface); border: 1px solid var(--border);
    }

    .status-pill {
        display: inline-flex; align-items: center; gap: 7px;
        padding: 5px 11px; border-radius: 999px; font-size: 12px; font-weight: 600;
        margin-bottom: 6px; border: 1px solid var(--border); background: var(--surface);
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .dot-ok { background: var(--ok); box-shadow: 0 0 0 0 rgba(52,224,161,0.6); animation: pulse 2s infinite; }
    .dot-bad { background: var(--bad); }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52,224,161,0.55); }
        70% { box-shadow: 0 0 0 7px rgba(52,224,161,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,224,161,0); }
    }
    .pill-ok { color: var(--ok); }
    .pill-bad { color: var(--bad); }

    .doc-card {
        display: flex; align-items: center; gap: 10px;
        padding: 9px 11px; border-radius: 10px; background: var(--surface);
        border: 1px solid var(--border);
        margin-bottom: 6px; font-size: 13px; color: var(--text);
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .doc-card:hover { background: var(--surface-hi); border-color: rgba(139,107,255,0.35); }
    .doc-card .name { flex: 1; word-break: break-word; }

    /* Sidebar chat/doc row buttons: keep labels from overflowing */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
        padding: 6px 8px !important; min-height: 36px;
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    /* Buttons */
    [data-testid="stSidebar"] button, .stButton button {
        border-radius: 10px !important; border: 1px solid var(--border) !important;
        background: var(--surface) !important; color: var(--text) !important;
        transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease !important;
    }
    [data-testid="stSidebar"] button:hover, .stButton button:hover {
        border-color: rgba(139,107,255,0.5) !important; background: var(--surface-hi) !important;
        transform: translateY(-1px);
    }
    button[kind="primary"], [data-testid="baseButton-primary"] {
        background: var(--grad) !important; border: none !important; color: #0a0c11 !important; font-weight: 600 !important;
    }
    button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover {
        filter: brightness(1.08); transform: translateY(-1px);
    }

    /* Chat area */
    [data-testid="stChatMessage"] {
        background: var(--surface) !important; border: 1px solid var(--border) !important;
        border-radius: 16px !important; padding: 4px 6px !important;
        animation: fadeInUp 0.35s ease both;
    }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) { [data-testid="stChatMessage"] { animation: none; } }

    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        background: rgba(20,24,34,0.92) !important;
        border: 1px solid var(--border) !important;
        box-shadow: 0 12px 36px -14px rgba(0,0,0,0.8);
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(139,107,255,0.65) !important;
        box-shadow: 0 0 0 3px rgba(139,107,255,0.14), 0 12px 36px -14px rgba(0,0,0,0.8);
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important; border: none !important;
        color: var(--text) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: var(--text-dim) !important; opacity: 0.75; }

    [data-testid="stExpander"] {
        background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
    }

    .empty-state { text-align: center; padding: 54px 20px; color: var(--text-dim); }
    .empty-state .big {
        font-size: 44px; margin-bottom: 10px; display: inline-block;
        filter: drop-shadow(0 6px 18px rgba(139,107,255,0.45));
    }
    .empty-state .title {
        font-family: 'Sora', sans-serif; font-weight: 700; font-size: 17px; color: var(--text); margin-bottom: 4px;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px !important;
        border: 1.5px dashed rgba(139,107,255,0.42) !important;
        background: rgba(139,107,255,0.06) !important;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(139,107,255,0.75) !important;
        background: rgba(139,107,255,0.11) !important;
    }
    [data-testid="stFileUploaderDropzone"] * { color: var(--text-dim) !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--surface-hi) !important; color: var(--text) !important;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.14); border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-row">
            <div class="hero-badge">💠</div>
            <div>
                <h1>RAG Chatbot</h1>
                <p>Ask questions about your own documents — grounded, cited, private.</p>
            </div>
        </div>
        <div class="pill-row">
            <span class="cap-pill">🧠 Local Embeddings</span>
            <span class="cap-pill">⚡ FAISS Vector Search</span>
            <span class="cap-pill">✨ Gemini / Ollama</span>
            <span class="cap-pill">🔌 MCP Ready</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return RAGPipeline()


def provider_is_reachable() -> bool:
    if LLM_PROVIDER == "gemini":
        from src.config import GEMINI_API_KEY
        return bool(GEMINI_API_KEY and GEMINI_API_KEY != "your-gemini-key-here")
    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except httpx.HTTPError:
        return False


if "session_id" not in st.session_state:
    sessions = history.list_sessions()
    st.session_state.session_id = sessions[0]["id"] if sessions else history.create_session()
    st.session_state.messages = history.get_messages(st.session_state.session_id)


with st.sidebar:
    provider_label = "Gemini" if LLM_PROVIDER == "gemini" else "Ollama"
    st.markdown('<div class="side-label">Status</div>', unsafe_allow_html=True)
    if provider_is_reachable():
        st.markdown(
            f'<span class="status-pill pill-ok"><span class="status-dot dot-ok"></span>{provider_label} ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-pill pill-bad"><span class="status-dot dot-bad"></span>{provider_label} not configured</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="side-meta">model <b>{CHAT_MODEL}</b><br>embed <b>{EMBEDDING_MODEL}</b></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="side-label">Chats</div>', unsafe_allow_html=True)
    if st.button("➕ New chat", use_container_width=True, type="primary"):
        st.session_state.session_id = history.create_session()
        st.session_state.messages = []
        st.rerun()

    if "renaming_id" not in st.session_state:
        st.session_state.renaming_id = None

    for session in history.list_sessions():
        is_active = session["id"] == st.session_state.session_id

        if st.session_state.renaming_id == session["id"]:
            new_title = st.text_input(
                "Rename chat",
                value=session["title"],
                key=f"rename_input_{session['id']}",
                label_visibility="collapsed",
            )
            save_col, cancel_col = st.columns(2)
            with save_col:
                if st.button("✅ Save", key=f"save_{session['id']}", use_container_width=True):
                    history.rename_session(session["id"], new_title)
                    st.session_state.renaming_id = None
                    st.rerun()
            with cancel_col:
                if st.button("✖️ Cancel", key=f"cancel_{session['id']}", use_container_width=True):
                    st.session_state.renaming_id = None
                    st.rerun()
            continue

        row_col, rename_col, delete_col = st.columns([5, 1.15, 1.15], gap="small")
        with row_col:
            if st.button(
                session["title"],
                key=f"chat_{session['id']}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state.session_id = session["id"]
                st.session_state.messages = history.get_messages(session["id"])
                st.rerun()
        with rename_col:
            if st.button("✏️", key=f"rename_{session['id']}", help="Rename this chat"):
                st.session_state.renaming_id = session["id"]
                st.rerun()
        with delete_col:
            if st.button("🗑️", key=f"delete_chat_{session['id']}", help="Delete this chat"):
                history.delete_session(session["id"])
                if is_active:
                    remaining = history.list_sessions()
                    if remaining:
                        st.session_state.session_id = remaining[0]["id"]
                        st.session_state.messages = history.get_messages(remaining[0]["id"])
                    else:
                        st.session_state.session_id = history.create_session()
                        st.session_state.messages = []
                st.rerun()

    st.divider()

    st.markdown('<div class="side-label">Knowledge base</div>', unsafe_allow_html=True)
    existing_files = [
        f for f in sorted(os.listdir(DATA_DIR))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ] if os.path.isdir(DATA_DIR) else []

    if existing_files:
        for fname in existing_files:
            icon = FILE_ICONS.get(os.path.splitext(fname)[1].lower(), "📄")
            name_col, delete_col = st.columns([5, 1])
            with name_col:
                st.markdown(
                    f'<div class="doc-card"><span>{icon}</span><span class="name">{fname}</span></div>',
                    unsafe_allow_html=True,
                )
            with delete_col:
                if st.button("🗑️", key=f"delete_{fname}", help=f"Remove {fname}"):
                    os.remove(os.path.join(DATA_DIR, fname))
                    with st.spinner("Removing from index..."):
                        try:
                            get_pipeline().rebuild()
                        except ValueError:
                            pass
                    st.rerun()
    else:
        st.caption("No documents yet — upload one below to get started.")

    st.divider()

    st.markdown('<div class="side-label">Add documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF, TXT, or MD files here",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        if st.button("➕ Add to knowledge base", use_container_width=True, type="primary"):
            for uploaded_file in uploaded_files:
                dest_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            with st.spinner("Indexing uploaded file(s)..."):
                try:
                    count = get_pipeline().rebuild()
                    st.success(f"Added {len(uploaded_files)} file(s). Index now has {count} chunks.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Rebuilding index from documents..."):
            try:
                count = get_pipeline().rebuild()
                st.success(f"Index rebuilt with {count} chunks.")
            except ValueError as exc:
                st.error(str(exc))

try:
    pipeline = get_pipeline()
except ValueError as exc:
    st.info(str(exc))
    st.stop()

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="big">✨</div>
            <div class="title">Ask me anything</div>
            <div>I'll answer using only your indexed documents, with sources cited.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.caption(f"{FILE_ICONS.get(os.path.splitext(src['source'])[1].lower(), '📄')} {src['source']} · relevance {src['score']:.2f}")

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = pipeline.answer(question)
            except (httpx.ConnectError, ConnectionError):
                st.error(OLLAMA_HELP if LLM_PROVIDER == "ollama" else GEMINI_HELP)
                st.stop()
            except Exception as exc:
                if LLM_PROVIDER == "gemini":
                    st.error(f"{GEMINI_HELP}\n\nDetails: {exc}")
                    st.stop()
                raise
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📎 Sources"):
                    for src in result["sources"]:
                        st.caption(f"{FILE_ICONS.get(os.path.splitext(src['source'])[1].lower(), '📄')} {src['source']} · relevance {src['score']:.2f}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    history.save_messages(st.session_state.session_id, st.session_state.messages)
