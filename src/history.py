import json
import os
import uuid
from datetime import datetime, timezone

from src.config import HISTORY_PATH


def _load_all() -> dict:
    if not os.path.exists(HISTORY_PATH):
        return {}
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_all(sessions: dict) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)


def list_sessions() -> list[dict]:
    """Return session summaries (id, title, created_at), newest first."""
    sessions = _load_all()
    summaries = [
        {"id": sid, "title": s["title"], "created_at": s["created_at"]}
        for sid, s in sessions.items()
    ]
    summaries.sort(key=lambda s: s["created_at"], reverse=True)
    return summaries


def get_messages(session_id: str) -> list[dict]:
    sessions = _load_all()
    session = sessions.get(session_id)
    return session["messages"] if session else []


def create_session() -> str:
    sessions = _load_all()
    session_id = uuid.uuid4().hex[:12]
    sessions[session_id] = {
        "title": "New chat",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    _save_all(sessions)
    return session_id


def save_messages(session_id: str, messages: list[dict]) -> None:
    sessions = _load_all()
    if session_id not in sessions:
        sessions[session_id] = {
            "title": "New chat",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

    sessions[session_id]["messages"] = messages

    if sessions[session_id]["title"] == "New chat":
        first_question = next((m["content"] for m in messages if m["role"] == "user"), None)
        if first_question:
            title = first_question.strip().splitlines()[0]
            sessions[session_id]["title"] = title[:50] + ("…" if len(title) > 50 else "")

    _save_all(sessions)


def rename_session(session_id: str, new_title: str) -> None:
    sessions = _load_all()
    if session_id in sessions and new_title.strip():
        sessions[session_id]["title"] = new_title.strip()
        _save_all(sessions)


def delete_session(session_id: str) -> None:
    sessions = _load_all()
    sessions.pop(session_id, None)
    _save_all(sessions)
