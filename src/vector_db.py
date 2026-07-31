import os
import pickle

import faiss
import numpy as np

from src.config import INDEX_PATH, METADATA_PATH
from src.utils import get_logger

logger = get_logger(__name__)


class VectorDB:
    def __init__(self):
        self.index = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if embeddings.shape[0] == 0:
            raise ValueError("No embeddings to index.")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized)
        self.metadata = metadata
        logger.info("Built FAISS index with %d vectors (dim=%d)", self.index.ntotal, dim)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No index to save. Call build() first.")
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Saved index to %s", INDEX_PATH)

    def clear(self) -> None:
        self.index = None
        self.metadata = []
        for path in (INDEX_PATH, METADATA_PATH):
            if os.path.exists(path):
                os.remove(path)
        logger.info("Cleared index.")

    def load(self) -> bool:
        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("Loaded index with %d vectors", self.index.ntotal)
            return True
        except (FileNotFoundError, RuntimeError):
            return False

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1)
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results
