import ollama
from google import genai
from google.genai import types as genai_types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.prompts import SYSTEM_PROMPT, build_user_prompt

_ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
_gemini_client = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file to use LLM_PROVIDER=gemini."
            )
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _generate_ollama(user_prompt: str) -> str:
    response = _ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]


def _generate_gemini(user_prompt: str) -> str:
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text


def generate_answer(question: str, contexts: list[dict]) -> str:
    user_prompt = build_user_prompt(question, contexts)

    if LLM_PROVIDER == "gemini":
        return _generate_gemini(user_prompt)
    return _generate_ollama(user_prompt)
