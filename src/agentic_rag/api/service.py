from pathlib import Path

from pydantic import BaseModel, Field

from agentic_rag.agent.service import RAGAgent
from agentic_rag.ingestion.pipeline import ingest_path
from agentic_rag.retrieval.vector_store import VectorStore


class IngestRequest(BaseModel):
    path: str = Field(description="File or directory path to ingest")
    reset: bool = Field(default=False, description="Clear the index before ingesting")


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    route: str | None
    sources: list[str]
    rewrite_count: int


class StatsResponse(BaseModel):
    chunk_count: int


def ingest_documents(path: str, *, reset: bool = False) -> IngestResponse:
    stats = ingest_path(Path(path), reset=reset)
    return IngestResponse(**stats)


def query_documents(question: str) -> QueryResponse:
    result = RAGAgent().query(question)
    return QueryResponse(
        answer=result.answer,
        route=result.route,
        sources=result.sources,
        rewrite_count=result.rewrite_count,
    )


def index_stats() -> StatsResponse:
    return StatsResponse(chunk_count=VectorStore().count())
