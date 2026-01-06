from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from app.config import settings


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_rag_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer using ONLY the provided context. "
                       "If the context is insufficient, say you don't know."),
            ("human", "Context:\n\n{context}\n\nQuestion:\n\n{question}"),
        ]
    )

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.temperature,
    )

    return prompt | llm | StrOutputParser()
