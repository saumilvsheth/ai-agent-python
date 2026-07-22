"""
Shared state schemas for the LangGraph agent.

AgentState is the graph's working memory passed between nodes.
GradeDocuments is the structured output shape for the relevance grader.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Mutable state threaded through every node in the RAG graph.

    Fields:
        messages: Conversation history; add_messages merges new AIMessages in.
        question: Current query text (may be rewritten during the run).
        documents: Retrieved and/or graded chunks from the vector store.
        generation: Final answer string produced by generate or answer_direct.
        rewrite_count: How many times the query was rewritten (caps at MAX_REWRITES).
        route: Decision from route_question — "retrieve" or "direct".
    """

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    generation: str
    rewrite_count: int
    route: Literal["retrieve", "direct"] | None


class GradeDocuments(TypedDict):
    """
    Structured grader output: binary yes/no relevance for one chunk.

    Used with llm.with_structured_output so the model returns parseable JSON.
    """

    binary_score: str
