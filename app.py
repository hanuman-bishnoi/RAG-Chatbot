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

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7f8fc 0%, #eef1f8 100%); }
    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e6e8f0; }
    .app-header {
        display: flex; align-items: center; gap: 14px;
        padding: 18px 22px; margin-bottom: 18px; border-radius: 16px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 8px 24px rgba(99,102,241,0.25);
    }
    .app-header .icon {
        font-size: 30px; width: 48px; height: 48px; display: flex;
        align-items: center; justify-content: center; border-radius: 12px;
        background: rgba(255,255,255,0.18);
    }
    .app-header h1 { color: #fff; font-size: 22px; margin: 0; }
    .app-header p { color: rgba(255,255,255,0.85); margin: 2px 0 0 0; font-size: 13px; }

    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
        margin-bottom: 4px;
    }
    .pill-ok { background: #dcfce7; color: #15803d; }
    .pill-bad { background: #fee2e2; color: #b91c1c; }

    .doc-card {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 10px; border-radius: 10px; background: #f4f5fb;
        margin-bottom: 6px; font-size: 13px; color: #374151;
    }
    .doc-card .name { flex: 1; word-break: break-word; }

    .chat-card {
        padding: 7px 10px; border-radius: 10px; font-size: 13px; color: #374151;
        text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .chat-card.active { background: #eef0ff; color: #4338ca; font-weight: 600; }

    .stChatMessage { border-radius: 14px !important; }

    .empty-state {
        text-align: center; padding: 40px 20px; color: #6b7280;
    }
    .empty-state .big { font-size: 40px; margin-bottom: 8px; }

    div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px; border: 2px dashed #c7c9f5; background: #f8f8ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="icon">💬</div>
        <div>
            <h1>RAG Chatbot</h1>
            <p>Ask questions about your own documents</p>
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
    st.subheader("⚙️ Status")
    if provider_is_reachable():
        st.markdown(f'<span class="status-pill pill-ok">🟢 {provider_label} ready</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-pill pill-bad">🔴 {provider_label} not configured</span>', unsafe_allow_html=True)
    st.caption(f"Chat model: `{CHAT_MODEL}`  ·  Embeddings: `{EMBEDDING_MODEL}`")

    st.divider()

    st.subheader("🗂️ Chats")
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

        row_col, rename_col, delete_col = st.columns([4, 1, 1])
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

    st.subheader("📚 Knowledge base")
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

    st.subheader("⬆️ Add documents")
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
            <div class="big">🤔</div>
            <div>Ask me anything about the documents in your knowledge base.</div>
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
