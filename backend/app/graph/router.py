from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import settings


# ----------------------------
# 1) Structured routing schema
# ----------------------------
class RouteQuery(BaseModel):
    """
    LLM must choose one datasource.

    - vectorstore: use our local index (agents / prompt engineering / adv attacks)
    - web_search: use Tavily for recent events / anything outside our index
    """
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Route the user question to 'vectorstore' or 'web_search'.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short reason for routing decision (optional).",
    )


# ----------------------------
# 2) Heuristic router (cheap + fast)
# ----------------------------
RECENCY_PATTERNS = [
    r"\btoday\b",
    r"\byesterday\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bthis year\b",
    r"\blatest\b",
    r"\brecent\b",
    r"\bbreaking\b",
    r"\bnews\b",
    r"\bcurrent\b",
    r"\bupdate\b",
    r"\b202[4-9]\b",
    r"\b20[3-9]\d\b",  # future-ish years
]

RECENCY_RE = re.compile("|".join(RECENCY_PATTERNS), flags=re.IGNORECASE)


def heuristic_route(question: str) -> Optional[RouteQuery]:
    """
    If we are very confident it's a recency / news query, route to web_search.
    Otherwise return None and let the LLM decide.
    """
    q = question.strip()

    # Explicit recency signals → web
    if RECENCY_RE.search(q):
        return RouteQuery(
            datasource="web_search",
            reason="Heuristic: question contains recency/news keywords (e.g., latest/today/news/year).",
        )

    # Otherwise, uncertain → let LLM decide
    return None


# ----------------------------
# 3) LLM router (Groq)
# ----------------------------
def build_llm_router():
    """
    Builds an LLM router that decides between web_search and vectorstore.

    The vectorstore contains docs about:
    - LLM agents
    - prompt engineering
    - adversarial attacks on LLMs

    Everything else → web_search
    """
    llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
    )

    system = (
        "You are an expert at routing a user question to a vectorstore or web search.\n"
        "The vectorstore contains documents about:\n"
        "- LLM agents (agent memory, planning, reflection)\n"
        "- prompt engineering\n"
        "- adversarial attacks on LLMs\n\n"
        "If the question is about these topics, choose 'vectorstore'.\n"
        "Otherwise choose 'web_search'.\n"
        "If the question is about current events, news, or anything time-sensitive, choose 'web_search'."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Structured output ensures we get a valid RouteQuery object
    return prompt | llm.with_structured_output(RouteQuery)


def route_question(question: str) -> RouteQuery:
    """
    Main routing function used by the graph.
    1) Try heuristic route first (cheap + deterministic).
    2) Fall back to LLM route (more flexible).
    3) If router fails, default to web_search (safe fallback).
    """
    # 1) Heuristic first
    heur = heuristic_route(question)
    if heur is not None:
        return heur

    # 2) LLM router
    try:
        router = build_llm_router()
        return router.invoke({"question": question})
    except ValidationError:
        # If structured output parsing fails, safest is web_search
        return RouteQuery(
            datasource="web_search",
            reason="Fallback: structured routing validation failed.",
        )
    except Exception as e:
        return RouteQuery(
            datasource="web_search",
            reason=f"Fallback: routing exception: {type(e).__name__}",
        )
