"""
LangGraph workflow definition for the agentic RAG pipeline.

Builds a state machine with six nodes:
  route_question → (retrieve | answer_direct)
  retrieve → grade_documents → (generate | rewrite_query → retrieve)
  generate / answer_direct → END
"""

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

# Prevent infinite rewrite loops when retrieval keeps failing.
MAX_REWRITES = 2


class RouteDecision(BaseModel):
    """
    Structured output for the routing node.

    Forces the LLM to return either "retrieve" or "direct" as JSON.
    """

    route: Literal["retrieve", "direct"] = Field(
        description="Whether to retrieve from the document index or answer directly."
    )


def _llm() -> ChatOpenAI:
    """Shared chat model instance; temperature=0 for deterministic routing/grading."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.require_api_key(),
        temperature=0,
    )


def build_graph() -> CompiledStateGraph:
    """
    Construct and compile the full agent graph.

    Returns a runnable graph that accepts AgentState and produces updated state
    with generation, documents, route metadata, etc.
    """
    llm = _llm()
    vector_store = VectorStore()

    def route_question(state: AgentState) -> dict:
        """
        Node 1: Decide if the question needs document retrieval.

        Uses structured output so the model must pick retrieve or direct.
        """
        structured = llm.with_structured_output(RouteDecision)
        decision = structured.invoke(ROUTE_PROMPT.invoke({"question": state["question"]}))
        return {"route": decision.route}

    def retrieve(state: AgentState) -> dict:
        """
        Node 2: Semantic search over the FAISS index.

        Uses the current question (original or rewritten) as the query.
        """
        docs = vector_store.as_retriever().invoke(state["question"])
        return {"documents": docs}

    def grade_documents(state: AgentState) -> dict:
        """
        Node 3: Filter retrieved chunks by relevance.

        Each chunk is scored yes/no; only "yes" chunks proceed to generation.
        """
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
        """
        Node 4: Produce a grounded answer from graded context.

        Even with no relevant docs (after max rewrites), still runs so the
        user gets an explicit "insufficient information" response.
        """
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
        """
        Alternate exit path: answer without touching the vector store.

        Used for greetings and general knowledge the router flagged as direct.
        """
        response = llm.invoke(DIRECT_PROMPT.invoke({"question": state["question"]}))
        return {
            "generation": response.content,
            "messages": [AIMessage(content=response.content)],
        }

    def rewrite_query(state: AgentState) -> dict:
        """
        Node 5: Improve the search query when grading filtered everything out.

        Two-step: explain failure, then rewrite. Clears documents and bumps
        rewrite_count before looping back to retrieve.
        """
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
        """Conditional edge: follow the route chosen by route_question."""
        return state["route"] or "retrieve"

    def decide_after_grade(state: AgentState) -> str:
        """
        Conditional edge after grading.

        - Relevant docs found → generate
        - No docs but rewrites exhausted → generate anyway (best-effort)
        - Otherwise → rewrite and retry retrieval
        """
        if state["documents"]:
            return "generate"
        if state.get("rewrite_count", 0) >= MAX_REWRITES:
            return "generate"
        return "rewrite"

    # --- Wire nodes and edges into the LangGraph state machine ---

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
    """
    Turn a list of Document chunks into a single prompt-ready context string.

    Each chunk is numbered and tagged with its source file path from metadata.
    """
    if not documents:
        return "No relevant documents found."
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)
