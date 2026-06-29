import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    generation: str
    rewrite_count: int
    route: Literal["retrieve", "direct"] | None


class GradeDocuments(TypedDict):
    binary_score: str
