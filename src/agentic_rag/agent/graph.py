from typing import Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from agentic_rag.agent.prompts import (
    DIRECT_PROMPT,
    GENERATE_PROMPT,
    GRADE_PROMPT,
    NOT_RELEVANT_PROMPT,
    REWRITE_PROMPT,
    ROUTE_PROMPT,
)
from agentic_rag.agent.state import AgentState, GradeDocuments
from agentic_rag.config import settings
from agentic_rag.retrieval.vector_store import VectorStore

MAX_REWRITES = 2


class RouteDecision(BaseModel):
    route: Literal["retrieve", "direct"] = Field(
        description="Whether to retrieve from the document index or answer directly."
    )


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.require_api_key(),
        temperature=0,
    )


def build_graph() -> CompiledStateGraph:
    llm = _llm()
    vector_store = VectorStore()

    def route_question(state: AgentState) -> dict:
        structured = llm.with_structured_output(RouteDecision)
        decision = structured.invoke(ROUTE_PROMPT.invoke({"question": state["question"]}))
        return {"route": decision.route}

    def retrieve(state: AgentState) -> dict:
        docs = vector_store.as_retriever().invoke(state["question"])
        return {"documents": docs}

    def grade_documents(state: AgentState) -> dict:
        grader = llm.with_structured_output(GradeDocuments)
        filtered: list[Document] = []
        for doc in state["documents"]:
            score = grader.invoke(
                GRADE_PROMPT.invoke(
                    {
                        "question": state["question"],
                        "document": doc.page_content,
                    }
                )
            )
            if score["binary_score"].lower() == "yes":
                filtered.append(doc)
        return {"documents": filtered}

    def generate(state: AgentState) -> dict:
        context = _format_context(state["documents"])
        response = llm.invoke(
            GENERATE_PROMPT.invoke(
                {"question": state["question"], "context": context}
            )
        )
        return {
            "generation": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    def answer_direct(state: AgentState) -> dict:
        response = llm.invoke(DIRECT_PROMPT.invoke({"question": state["question"]}))
        return {
            "generation": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    def rewrite_query(state: AgentState) -> dict:
        reason = llm.invoke(
            NOT_RELEVANT_PROMPT.invoke(
                {
                    "question": state["question"],
                    "documents": _format_context(state["documents"]),
                }
            )
        )
        rewritten = llm.invoke(
            REWRITE_PROMPT.invoke(
                {"question": state["question"], "reason": reason.content}
            )
        )
        return {
            "question": rewritten.content.strip(),
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "documents": [],
        }

    def decide_after_route(state: AgentState) -> str:
        return state["route"] or "retrieve"

    def decide_after_grade(state: AgentState) -> str:
        if state["documents"]:
            return "generate"
        if state.get("rewrite_count", 0) >= MAX_REWRITES:
            return "generate"
        return "rewrite"

    graph = StateGraph(AgentState)
    graph.add_node("route_question", route_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("answer_direct", answer_direct)
    graph.add_node("rewrite_query", rewrite_query)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges(
        "route_question",
        decide_after_route,
        {"retrieve": "retrieve", "direct": "answer_direct"},
    )
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {"generate": "generate", "rewrite": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("answer_direct", END)

    return graph.compile()


def _format_context(documents: list[Document]) -> str:
    if not documents:
        return "No relevant documents found."
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)
