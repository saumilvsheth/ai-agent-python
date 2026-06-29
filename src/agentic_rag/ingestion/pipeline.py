from pathlib import Path

from langchain_core.documents import Document

from agentic_rag.ingestion.loader import chunk_documents, load_documents
from agentic_rag.retrieval.vector_store import VectorStore


def ingest_path(path: Path, *, reset: bool = False) -> dict[str, int]:
    """Load, chunk, and index documents from a file or directory."""
    documents = load_documents(path)
    if not documents:
        return {"documents": 0, "chunks": 0}

    chunks = chunk_documents(documents)
    store = VectorStore()
    if reset:
        store.reset()
    store.add_documents(chunks)
    return {"documents": len(documents), "chunks": len(chunks)}
