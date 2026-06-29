from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from agentic_rag.config import settings


class VectorStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self._index_path = settings.vector_index_dir
        self._embeddings: OpenAIEmbeddings | None = None
        self._store = self._load_or_create()

    def _get_embeddings(self) -> OpenAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.require_api_key(),
            )
        return self._embeddings

    def _load_or_create(self) -> FAISS | None:
        index_file = self._index_path / "index.faiss"
        if index_file.exists():
            return FAISS.load_local(
                str(self._index_path),
                self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        return None

    def as_retriever(self) -> VectorStoreRetriever:
        store = self._require_store()
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_top_k},
        )

    def add_documents(self, documents: list[Document]) -> None:
        if self._store is None:
            self._store = FAISS.from_documents(documents, self._get_embeddings())
        else:
            self._store.add_documents(documents)
        self._persist()

    def count(self) -> int:
        if self._store is None:
            return 0
        return len(self._store.docstore._dict)

    def reset(self) -> None:
        for path in self._index_path.glob("*"):
            path.unlink()
        self._store = None

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save_local(str(self._index_path))

    def _require_store(self) -> FAISS:
        if self._store is None:
            raise RuntimeError(
                "Vector index is empty. Run `agentic-rag ingest <path>` first."
            )
        return self._store
