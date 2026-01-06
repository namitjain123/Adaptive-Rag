from __future__ import annotations

from typing import List, Optional, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
    Shared state that flows through the LangGraph.

    total=False means keys can be optional and added progressively by nodes.
    """

    # Input
    question: str

    # Routing
    route: Literal["vectorstore", "web_search"]

    # Retrieval / context
    documents: List[Document]
    rewritten_question: str
    #So we can show it in UI traces and debug retrieval quality—this is crucial in RAG systems.

    # Generation
    generation: str

    # Quality signals / metadata
    grounded: bool
    answered: bool
    iterations: int
    #Because we do self-correction loops. Iteration count helps prevent infinite loops and helps observability.
    error: Optional[str]
