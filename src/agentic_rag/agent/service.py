from dataclasses import dataclass, field

from langchain_core.documents import Document

from agentic_rag.agent.graph import build_graph


@dataclass
class QueryResult:
    answer: str
    route: str | None
    sources: list[str] = field(default_factory=list)
    rewrite_count: int = 0


class RAGAgent:
    def __init__(self) -> None:
        self._graph = build_graph()

    def query(self, question: str) -> QueryResult:
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
    seen: set[str] = set()
    sources: list[str] = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources
