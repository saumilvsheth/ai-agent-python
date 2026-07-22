"""
High-level agent facade used by CLI and API layers.

RAGAgent wraps the compiled LangGraph and returns a simple QueryResult
dataclass instead of raw graph state.
"""

from dataclasses import dataclass, field

from langchain_core.documents import Document

from agentic_rag.agent.graph import build_graph


@dataclass
class QueryResult:
    """
    Normalized response from a single agent query.

    Attributes:
        answer: Final text shown to the user.
        route: "retrieve" or "direct" from the routing node (may be None).
        sources: Unique file paths of chunks used in the answer.
        rewrite_count: How many query rewrites occurred before answering.
    """

    answer: str
    route: str | None
    sources: list[str] = field(default_factory=list)
    rewrite_count: int = 0


class RAGAgent:
    """
    Thin wrapper that builds the graph once and exposes a query() method.

    Keeps graph invocation details out of CLI/API code.
    """

    def __init__(self) -> None:
        self._graph = build_graph()

    def query(self, question: str) -> QueryResult:
        """
        Run the full agentic RAG pipeline for one user question.

        Initializes empty state, invokes the graph, and maps the final state
        into a QueryResult for callers.
        """
        result = self._graph.invoke(
            {
                "messages": [],
                "question": question,
                "documents": [],
                "generation": "",
                "rewrite_count": 0,
                "route": None,
            }
        )
        sources = _extract_sources(result.get("documents", []))
        return QueryResult(
            answer=result.get("generation", ""),
            route=result.get("route"),
            sources=sources,
            rewrite_count=result.get("rewrite_count", 0),
        )


def _extract_sources(documents: list[Document]) -> list[str]:
    """
    Collect unique source file paths from document metadata.

    Preserves first-seen order for stable citation lists in CLI/API output.
    """
    seen: set[str] = set()
    sources: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources
