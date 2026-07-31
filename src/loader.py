import os

from pypdf import PdfReader

from src.utils import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(data_dir: str) -> list[dict]:
    """Load all supported documents from data_dir.

    Returns a list of {"source": filename, "text": content} dicts.
    """
    documents = []

    if not os.path.isdir(data_dir):
        return documents

    for filename in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        path = os.path.join(data_dir, filename)
        try:
            if ext == ".pdf":
                text = load_pdf(path)
            else:
                text = load_text(path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", filename, exc)
            continue

        if text.strip():
            documents.append({"source": filename, "text": text})

    logger.info("Loaded %d documents from %s", len(documents), data_dir)
    return documents
