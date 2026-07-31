from src.llm import generate_answer
from src.retriever import Retriever
from src.utils import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        if not self.retriever.index_exists():
            logger.info("No existing index found; building a new one.")
            self.retriever.build_index()

    def rebuild(self) -> int:
        return self.retriever.build_index()

    def answer(self, question: str) -> dict:
        contexts = self.retriever.retrieve(question)
        answer = generate_answer(question, contexts)
        return {"answer": answer, "sources": contexts}
