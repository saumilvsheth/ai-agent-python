"""
Shared business logic for the REST API.

Defines Pydantic request/response models and thin wrapper functions that
connect HTTP handlers to ingestion, querying, and index stats. Keeps FastAPI
route files free of domain logic.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from agentic_rag.agent.service import RAGAgent
from agentic_rag.ingestion.pipeline import ingest_path
from agentic_rag.retrieval.vector_store import VectorStore


# --- Request/response schemas (also used for OpenAPI docs) ---


class IngestRequest(BaseModel):
    """POST /ingest body: path to ingest and optional index reset flag."""

    path: str = Field(description="File or directory path to ingest")
    reset: bool = Field(default=False, description="Clear the index before ingesting")


class IngestResponse(BaseModel):
    """Counts returned after a successful ingestion run."""

    documents: int
    chunks: int


class QueryRequest(BaseModel):
    """POST /query body: the user's natural-language question."""

    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    """Agent answer plus metadata about routing, sources, and rewrites."""

    answer: str
    route: str | None
    sources: list[str]
    rewrite_count: int


class StatsResponse(BaseModel):
    """GET /stats: how many chunks are currently in the FAISS index."""

    chunk_count: int


# --- Service functions called by FastAPI route handlers ---


def ingest_documents(path: str, *, reset: bool = False) -> IngestResponse:
    """Run the ingestion pipeline and wrap counts in IngestResponse."""
    stats = ingest_path(Path(path), reset=reset)
    return IngestResponse(**stats)


def query_documents(question: str) -> QueryResponse:
    """Run the agent and map QueryResult fields into QueryResponse."""
    result = RAGAgent().query(question)
    return QueryResponse(
        answer=result.answer,
        route=result.route,
        sources=result.sources,
        rewrite_count=result.rewrite_count,
    )


def index_stats() -> StatsResponse:
    """Return current vector index size without requiring an OpenAI call."""
    return StatsResponse(chunk_count=VectorStore().count())
