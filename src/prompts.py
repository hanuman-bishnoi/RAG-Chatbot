SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context. If the answer is not contained in the context, say you don't know "
    "instead of guessing. Cite the source filename(s) you used at the end of your answer."
)


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    context_blocks = []
    for i, ctx in enumerate(contexts, start=1):
        context_blocks.append(f"[{i}] (source: {ctx['source']})\n{ctx['text']}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."

    return (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )
