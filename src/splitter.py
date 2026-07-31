from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - chunk_overlap

    return chunks


def split_documents(documents: list[dict]) -> list[dict]:
    """Split loaded documents into chunks, preserving source metadata."""
    chunks = []
    for doc in documents:
        for i, chunk in enumerate(split_text(doc["text"])):
            chunks.append(
                {
                    "text": chunk,
                    "source": doc["source"],
                    "chunk_id": i,
                }
            )
    return chunks
