import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return an (n, dim) float32 array of embeddings for the given texts."""
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = _get_model()
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
