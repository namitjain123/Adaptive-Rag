from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.graph.nodes import (
    GraphState,
    route_node,
    web_search_node,
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    decide_after_doc_grading,
    decide_after_generation,
)


def build_workflow():
    """
    Build the Adaptive RAG LangGraph.

    Routes between:
    - web_search path (for recent / outside-index questions)
    - vectorstore path with self-corrective loop
    """

    workflow = StateGraph(GraphState)

    # ----------------------------
    # Register nodes
    # ----------------------------
    workflow.add_node("router", route_node)

    workflow.add_node("web_search", web_search_node)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)

    workflow.add_node("generate", generate_node)

    # ----------------------------
    # Edges: START -> route
    # ----------------------------
    workflow.add_edge(START, "router")

    # ----------------------------
    # Conditional: route -> web_search OR retrieve
    # ----------------------------
    def route_to_next(state: GraphState) -> str:
        # State must contain route after route_node runs
        if state.get("route") == "web_search":
            return "web_search"
        return "retrieve"

    workflow.add_conditional_edges(
        "router",
        route_to_next,
        {
            "web_search": "web_search",
            "retrieve": "retrieve",
        },
    )

    # ----------------------------
    # Web path: web_search -> generate -> (grade decision) -> END or retry logic
    # ----------------------------
    workflow.add_edge("web_search", "generate")

    # After generate, we decide if answer is useful/grounded.
    # If not, we rewrite and fall back to vectorstore retrieval.
    workflow.add_conditional_edges(
        "generate",
        decide_after_generation,
        {
            "useful": END,
            "retry_generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # ----------------------------
    # Vectorstore path:
    # retrieve -> grade_documents -> (rewrite or generate)
    # rewrite_query -> retrieve  (loop)
    # ----------------------------
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_doc_grading,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")

    return workflow


# Compile graph once (module-level singleton)
_GRAPH = build_workflow().compile()


def get_graph():
    """
    Return compiled graph. Used by API layer.
    """
    return _GRAPH
