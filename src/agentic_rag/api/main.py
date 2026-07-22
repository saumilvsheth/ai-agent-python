"""
FastAPI application and HTTP route definitions.

Maps URLs to service functions from api.service. This file is the entry
point uvicorn loads when you run `agentic-rag serve`.
"""

from fastapi import FastAPI, HTTPException

from agentic_rag import __version__
from agentic_rag.api.service import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    index_stats,
    ingest_documents,
    query_documents,
)

# FastAPI app instance; metadata appears in auto-generated /docs UI.
app = FastAPI(
    title="Agentic RAG",
    description="Retrieval-augmented generation with adaptive routing and document grading",
    version=__version__,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — confirms the server process is running."""
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Return how many document chunks are indexed."""
    return index_stats()


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """
    Index documents from a file or directory path.

    Translates domain errors into appropriate HTTP status codes.
    """
    try:
        return ingest_documents(request.path, reset=request.reset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the agentic RAG pipeline and return the answer plus metadata."""
    return query_documents(request.question)
