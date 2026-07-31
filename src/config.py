import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))

INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
METADATA_PATH = os.path.join(VECTORSTORE_DIR, "metadata.pkl")

for _dir in (DATA_DIR, VECTORSTORE_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)
