from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from langchain_core.documents import Document

from app.rag.retriever import retrieve_documents
from app.rag.generator import build_rag_chain, format_docs
from app.rag.graders import (
    build_answer_grader,
    build_hallucination_grader,
    build_retrieval_grader,
)
from app.rag.rewriter import rewrite_question
from app.rag.web_search import web_search
from app.graph.router import route_question


# ----------------------------
# Graph State
# ----------------------------
class GraphState(TypedDict, total=False):
    """
    Shared state passed between nodes.

    total=False means keys are optional, so nodes can add fields progressively.
    """

    question: str                  # current question (may be rewritten)
    original_question: str         # original user question
    documents: List[Document]      # retrieved or web docs
    generation: str                # model output (final answer)
    route: Literal["vectorstore", "web_search"]
    route_reason: Optional[str]

    # Counters for safety / debugging
    retrieve_attempts: int         # how many times we've tried retrieval
    generate_attempts: int         # how many times we've tried generation


# ----------------------------
# Build reusable chains once
# ----------------------------
RAG_CHAIN = build_rag_chain()
RETRIEVAL_GRADER = build_retrieval_grader()
HALLUCINATION_GRADER = build_hallucination_grader()
ANSWER_GRADER = build_answer_grader()


# ----------------------------
# Node: Route (START -> web or vectorstore)
# ----------------------------
def route_node(state: GraphState) -> GraphState:
    """
    Decide route based on the user's original question.
    Saves route decision into state.
    """
    question = state["question"]
    #Extract the question
    rq = route_question(question)

    return {
        **state,
        "route": rq.datasource,
        "route_reason": rq.reason,
        "original_question": state.get("original_question", question),
        "retrieve_attempts": state.get("retrieve_attempts", 0),
        "generate_attempts": state.get("generate_attempts", 0),
    }


# ----------------------------
# Node: Web Search
# ----------------------------
def web_search_node(state: GraphState) -> GraphState:
    """
    Fetch web results as Documents, store in state.
    """
    question = state["question"]
    docs = web_search(question, k=3)

    return {**state, "documents": docs}
#This spreads the existing state forward.


# ----------------------------
# Node: Retrieve (Vectorstore)
# ----------------------------
def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieve documents from vectorstore.
    """
    question = state["question"]
    attempts = state.get("retrieve_attempts", 0) + 1

    docs = retrieve_documents(question, k=4)
    return {**state, "documents": docs, "retrieve_attempts": attempts}


# ----------------------------
# Node: Grade documents
# ----------------------------
def grade_documents_node(state: GraphState) -> GraphState:
    """
    Filter out irrelevant retrieved docs using an LLM grader.
    """
    question = state["question"]
    documents = state.get("documents", [])

    filtered: List[Document] = []
    for d in documents:
        score = RETRIEVAL_GRADER.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score == "yes":
            filtered.append(d)

    return {**state, "documents": filtered}


# ----------------------------
# Node: Transform / rewrite query
# ----------------------------
def rewrite_query_node(state: GraphState) -> GraphState:
    """
    Rewrite question to improve retrieval.
    Keeps original_question unchanged, updates question.
    """
    current_q = state["question"]
    better_q = rewrite_question(current_q)
    return {**state, "question": better_q}


# ----------------------------
# Node: Generate answer (RAG)
# ----------------------------
def generate_node(state: GraphState) -> GraphState:
    """
    Generate answer using RAG_CHAIN and the documents in state.
    """
    question = state["question"]
    documents = state.get("documents", [])
    attempts = state.get("generate_attempts", 0) + 1

    context = format_docs(documents) if documents else ""
    generation = RAG_CHAIN.invoke({"context": context, "question": question})

    return {**state, "generation": generation, "generate_attempts": attempts}


# ----------------------------
# Decision: after grading docs
# ----------------------------
def decide_after_doc_grading(state: GraphState) -> Literal["rewrite_query", "generate"]:
    """
    If no relevant documents remain, rewrite query and retry retrieval.
    Otherwise generate an answer.
    """
    docs = state.get("documents", [])
    if len(docs) == 0:
        return "rewrite_query"
    return "generate"


# ----------------------------
# Decision: after generation
# ----------------------------
def decide_after_generation(
    state: GraphState,
) -> Literal["useful", "retry_generate", "rewrite_query"]:
    """
    Check if generation is:
    1) grounded in documents (hallucination check)
    2) answers the question (usefulness check)

    Decisions:
    - useful         -> END
    - retry_generate -> if not grounded but docs exist, try again
    - rewrite_query  -> if not useful, rewrite query and retry retrieval
    """
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")

    # If we have no docs (web route might still have docs), treat as not grounded.
    if not documents:
        return "rewrite_query"

    # 1) Grounding check
    h = HALLUCINATION_GRADER.invoke(
        {"documents": format_docs(documents), "generation": generation}
    )
    if h.binary_score != "yes":
        # If not grounded, we can try regenerate once or twice
        if state.get("generate_attempts", 0) < 2:
            return "retry_generate"
        return "rewrite_query"

    # 2) Answer check
    a = ANSWER_GRADER.invoke({"question": question, "generation": generation})
    if a.binary_score == "yes":
        return "useful"

    # If it's grounded but not answering, rewrite and retrieve again
    return "rewrite_query"
