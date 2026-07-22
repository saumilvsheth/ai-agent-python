"""
Load raw files and split them into retrieval-sized chunks.

Supports single files or entire directories of .txt, .md, and .pdf content.
Each loader returns LangChain Document objects with page_content and metadata.
"""

from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_rag.config import settings


def load_documents(path: Path) -> list[Document]:
    """
    Load documents from a single file or recursively from a directory.

    Raises FileNotFoundError if the path does not exist.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    # Single-file ingestion uses a dedicated loader based on extension.
    if path.is_file():
        return _load_file(path)

    # Directory ingestion runs one loader per supported file type and merges results.
    loaders = [
        DirectoryLoader(
            str(path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        ),
        DirectoryLoader(
            str(path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        ),
        DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
    ]

    documents: list[Document] = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def _load_file(path: Path) -> list[Document]:
    """
    Pick the correct LangChain loader for one file based on its suffix.

    Returns a list (usually one item) to match the directory-loader API.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix == ".md":
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix == ".txt":
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split large documents into overlapping chunks suitable for embedding.

    Chunk size and overlap come from settings. add_start_index=True records
    where each chunk began in the original document for citation/debugging.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
