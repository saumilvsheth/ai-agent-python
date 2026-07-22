"""
FAISS-backed vector store for semantic document search.

Embeddings are computed via OpenAI and persisted to disk so ingestion and
querying can happen in separate processes without rebuilding the index.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import settings


class VectorStore:
    """
    Thin wrapper around LangChain's FAISS integration.

    Handles lazy embedding client creation, disk persistence, and safe
    errors when the index is empty.
    """

    def __init__(self) -> None:
        # Ensure ./data/index exists; defer OpenAI client until first embed/search.
        settings.ensure_dirs()
        self._index_path = settings.vector_index_dir
        self._embeddings: OpenAIEmbeddings | None = None
        self._store = self._load_or_create()

    def _get_embeddings(self) -> OpenAIEmbeddings:
        """
        Create the OpenAI embeddings client on first use.

        Lazy initialization lets stats/health checks run without an API key.
        """
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.require_api_key(),
            )
        return self._embeddings

    def _load_or_create(self) -> FAISS | None:
        """
        Load an existing index from disk, or return None if none exists yet.

        FAISS stores index.faiss plus a pickle docstore alongside it.
        """
        index_file = self._index_path / "index.faiss"
        if index_file.exists():
            return FAISS.load_local(
                str(self._index_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return None

    def as_retriever(self) -> VectorStoreRetriever:
        """
        Expose similarity search for the LangGraph retrieve node.

        top_k controls how many chunks are returned per query.
        """
        store = self._require_store()
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_top_k},
        )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed and store new chunks, creating the index on first write.

        Always persists to disk after changes so data survives restarts.
        """
        if self._store is None:
            self._store = FAISS.from_documents(documents, self._get_embeddings())
        else:
            self._store.add_documents(documents)
        self._persist()

    def count(self) -> int:
        """Return the number of indexed chunks (0 if the index is empty)."""
        if self._store is None:
            return 0
        return len(self._store.docstore._dict)

    def reset(self) -> None:
        """Delete all persisted index files and clear the in-memory store."""
        for path in self._index_path.glob("*"):
            path.unlink()
        self._store = None

    def _persist(self) -> None:
        """Write the current FAISS index and docstore to vector_index_dir."""
        if self._store is not None:
            self._store.save_local(str(self._index_path))

    def _require_store(self) -> FAISS:
        """
        Return the loaded index or raise if the user has not ingested yet.

        Gives a actionable error pointing to the ingest CLI command.
        """
        if self._store is None:
            raise RuntimeError(
                "Vector index is empty. Run `agentic-rag ingest <path>` first."
            )
        return self._store
