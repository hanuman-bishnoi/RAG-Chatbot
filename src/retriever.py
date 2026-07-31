from src.config import DATA_DIR, TOP_K
from src.embeddings import embed_query, embed_texts
from src.loader import load_documents
from src.splitter import split_documents
from src.utils import get_logger
from src.vector_db import VectorDB

logger = get_logger(__name__)


class Retriever:
    def __init__(self):
        self.db = VectorDB()

    def index_exists(self) -> bool:
        return self.db.load()

    def build_index(self) -> int:
        """(Re)build the vector index from documents in DATA_DIR. Returns chunk count."""
        documents = load_documents(DATA_DIR)
        if not documents:
            self.db.clear()
            raise ValueError(f"No documents found in {DATA_DIR}. Add PDF/TXT/MD files and try again.")

        chunks = split_documents(documents)
        if not chunks:
            raise ValueError("Documents were loaded but produced no text chunks.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts)

        self.db.build(embeddings, chunks)
        self.db.save()
        return len(chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        query_embedding = embed_query(query)
        return self.db.search(query_embedding, top_k)
