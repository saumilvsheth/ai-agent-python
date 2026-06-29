from langchain_core.prompts import ChatPromptTemplate

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

DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user directly and concisely.",
        ),
        ("human", "{question}"),
    ]
)

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
