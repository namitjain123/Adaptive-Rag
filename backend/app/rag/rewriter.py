from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings


def build_question_rewriter():
    """
    Builds a chain that rewrites the user's question to be better for vectorstore retrieval.

    Why this exists:
    - Users ask short / vague questions ("agent memory", "types of memory")
    - Vector similarity search works better with more explicit semantic intent
    - Rewriting improves recall/precision without changing user meaning
    """
    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
    )

    system = (
        "You are a question re-writer. Convert the input question into a better version "
        "that is optimized for vector database retrieval. "
        "Keep the meaning the same, but make it more specific and context-rich. "
        "Return ONLY the rewritten question."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Original question:\n\n{question}\n\nRewrite:"),
        ]
    )

    return prompt | llm | StrOutputParser()


def rewrite_question(question: str) -> str:
    """
    Convenience helper (used inside graph nodes).
    """
    chain = build_question_rewriter()
    return chain.invoke({"question": question}).strip()
