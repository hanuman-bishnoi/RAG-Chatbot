import os

from mcp.server.mcpserver import MCPServer

from src.config import DATA_DIR
from src.loader import SUPPORTED_EXTENSIONS
from src.rag_pipeline import RAGPipeline

mcp = MCPServer(
    name="rag-chatbot",
    instructions=(
        "Query a local, private knowledge base built from the user's own documents "
        "(PDF/TXT/MD). Use ask_knowledge_base to get an answer with citations, "
        "list_documents to see what's indexed, and rebuild_index after documents "
        "are added or removed from the data folder."
    ),
)

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    """Answer a question using the local RAG knowledge base, citing sources."""
    result = _get_pipeline().answer(question)
    sources = sorted({src["source"] for src in result["sources"]})
    if not sources:
        return result["answer"]
    return f"{result['answer']}\n\nSources: {', '.join(sources)}"


@mcp.tool()
def list_documents() -> list[str]:
    """List the document filenames currently in the knowledge base."""
    if not os.path.isdir(DATA_DIR):
        return []
    return [
        f
        for f in sorted(os.listdir(DATA_DIR))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]


@mcp.tool()
def rebuild_index() -> str:
    """Rebuild the vector index from the documents currently in the knowledge base folder."""
    count = _get_pipeline().rebuild()
    return f"Index rebuilt with {count} chunks."


if __name__ == "__main__":
    mcp.run()
