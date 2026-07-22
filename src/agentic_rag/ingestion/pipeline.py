"""
End-to-end ingestion orchestration: load → chunk → index.

This is the function called by both the CLI and REST API when users ingest
documents. It returns simple counts for progress reporting.
"""

from pathlib import Path

from langchain_core.documents import Document

from agentic_rag.ingestion.loader import chunk_documents, load_documents
from agentic_rag.retrieval.vector_store import VectorStore


def ingest_path(path: Path, *, reset: bool = False) -> dict[str, int]:
    """
    Load, chunk, and index documents from a file or directory.

    Args:
        path: File or folder to ingest.
        reset: If True, wipe the existing FAISS index before adding new chunks.

    Returns:
        Dict with document and chunk counts (both zero if nothing was found).
    """
    documents = load_documents(path)
    if not documents:
        return {"documents": 0, "chunks": 0}

    chunks = chunk_documents(documents)
    store = VectorStore()

    # Optional full re-index: delete old vectors then write fresh ones.
    if reset:
        store.reset()

    store.add_documents(chunks)
    return {"documents": len(documents), "chunks": len(chunks)}
