from __future__ import annotations
from typing import Literal, List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.config import settings


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant: 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer grounded in facts: 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Answer resolves the question: 'yes' or 'no'"
    )


def _groq_llm():
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
    )


def build_retrieval_grader():
    system = (
        "You are a grader assessing relevance of a retrieved document to a user question. "
        "If it is semantically related, respond yes, else no."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ]
    )
    return prompt | _groq_llm().with_structured_output(GradeDocuments)


def build_hallucination_grader():
    system = (
        "You are a grader assessing whether an answer is grounded in the provided facts. "
        "Respond yes if grounded, else no."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Facts:\n\n{documents}\n\nAnswer:\n\n{generation}"),
        ]
    )
    return prompt | _groq_llm().with_structured_output(GradeHallucinations)


def build_answer_grader():
    system = (
        "You are a grader assessing whether an answer addresses the question. "
        "Respond yes if it answers, else no."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question:\n\n{question}\n\nAnswer:\n\n{generation}"),
        ]
    )
    return prompt | _groq_llm().with_structured_output(GradeAnswer)


def docs_to_plaintext(docs: List[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)
