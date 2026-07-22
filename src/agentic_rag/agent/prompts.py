"""
LLM prompt templates for each step of the agentic RAG workflow.

Each prompt is a ChatPromptTemplate with system + human messages.
Variables in {braces} are filled in by the graph nodes at runtime.
"""

from langchain_core.prompts import ChatPromptTemplate

# Decides whether to search the index or answer without retrieval.
ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route user questions. "
            "Return 'retrieve' if the question needs knowledge from indexed documents "
            "(facts, policies, product details, internal docs). "
            "Return 'direct' for greetings, small talk, or general knowledge "
            "that does not require document lookup.",
        ),
        ("human", "{question}"),
    ]
)

# Produces a better semantic search query after a failed retrieval attempt.
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a concise search query optimized for "
            "semantic retrieval over a document index. Output only the rewritten query.",
        ),
        (
            "human",
            "Original question:\n{question}\n\n"
            "Reason retrieval was insufficient:\n{reason}",
        ),
    ]
)

# Scores one retrieved chunk as relevant (yes) or not (no) to the question.
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade document relevance to a user question. "
            "Return 'yes' if the document contains information useful for answering. "
            "Return 'no' otherwise.",
        ),
        (
            "human",
            "Question: {question}\n\nDocument:\n{document}",
        ),
    ]
)

# Grounded answer generation using only the filtered context chunks.
GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the context is insufficient, say you do not have enough information. "
            "Cite sources using [source: filename] when possible.",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)

# Used when routing chose "direct" — no document context needed.
DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user directly and concisely.",
        ),
        ("human", "{question}"),
    ]
)

# Explains why retrieved docs were useless; feeds into REWRITE_PROMPT.
NOT_RELEVANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Explain briefly why the retrieved documents did not help answer the question.",
        ),
        (
            "human",
            "Question: {question}\n\nDocuments:\n{documents}",
        ),
    ]
)
