import ollama

from src.config import CHAT_MODEL, OLLAMA_BASE_URL
from src.prompts import SYSTEM_PROMPT, build_user_prompt

_client = ollama.Client(host=OLLAMA_BASE_URL)


def generate_answer(question: str, contexts: list[dict]) -> str:
    user_prompt = build_user_prompt(question, contexts)

    response = _client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]
