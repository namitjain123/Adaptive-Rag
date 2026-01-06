from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.graph.workflow import get_graph
import json
from sqlalchemy.orm import Session
from fastapi import Depends

from app.db.session import get_db
from app.db.models import ChatRun
import os
import traceback
from pydantic import ValidationError

router = APIRouter(prefix="", tags=["chat"])


# ----------------------------
# Request / Response schemas
# ----------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session id for client-side tracking (not stored yet).",
    )
    max_recursion: int = Field(
        default=20,
        ge=5,
        le=80,
        description="LangGraph recursion limit to avoid infinite loops.",
    )


class SourceItem(BaseModel):
    preview: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    route: Optional[str] = None
    route_reason: Optional[str] = None

    original_question: Optional[str] = None
    final_question: Optional[str] = None

    retrieve_attempts: int = 0
    generate_attempts: int = 0

    sources: List[SourceItem] = Field(default_factory=list)


# ----------------------------
# Helpers
# ----------------------------
def _doc_preview(text: str, n: int = 240) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("..." if len(text) > n else "")


def _extract_sources(state: Dict[str, Any], limit: int = 4) -> List[SourceItem]:
    docs = state.get("documents") or []
    items: List[SourceItem] = []
    for d in docs[:limit]:
        # langchain Document has page_content + metadata
        preview = _doc_preview(getattr(d, "page_content", ""))
        metadata = getattr(d, "metadata", {}) or {}
        items.append(SourceItem(preview=preview, metadata=metadata))
    return items


# ----------------------------
# Endpoints
# ----------------------------
@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """
    Main chat endpoint:
    - Runs Adaptive RAG LangGraph
    - Returns answer + route + attempts + source previews
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    graph = get_graph()

    try:
        # recursion_limit is the standard way to cap looping in LangGraph
        final_state = graph.invoke(
            {"question": question},
            config={"recursion_limit": req.max_recursion},
        
        )
        print("ROUTE_USED:", final_state.get("route"), "| REASON:", final_state.get("route_reason"))
        print("TAVILY_KEY_PRESENT:", bool(os.getenv("TAVILY_API_KEY")))
    
    except ValidationError as e:
        print("GRAPH VALIDATION ERROR:", e)
        print("ERRORS:", e.errors())
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph execution failed: ValidationError: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {type(e).__name__}")

    answer = (final_state.get("generation") or "").strip()
    if not answer:
        # Defensive: graph should always generate, but keep API safe
        raise HTTPException(status_code=500, detail="No answer generated.")

    # Store chat run in DB
        # Save to DB (one row per chat run)
    sources = _extract_sources(final_state)
    run = ChatRun(
        session_id=req.session_id,
        original_question=final_state.get("original_question") or question,
        final_question=final_state.get("question") or question,
        answer=answer,
        route=final_state.get("route"),
        route_reason=final_state.get("route_reason"),
        retrieve_attempts=int(final_state.get("retrieve_attempts", 0) or 0),
        generate_attempts=int(final_state.get("generate_attempts", 0) or 0),
        sources_json=json.dumps([s.model_dump() for s in sources]),
    )
    db.add(run)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        print("DB insert failed:", repr(e))


    return ChatResponse(
        answer=answer,
        route=final_state.get("route"),
        route_reason=final_state.get("route_reason"),
        original_question=final_state.get("original_question"),
        final_question=final_state.get("question"),
        retrieve_attempts=int(final_state.get("retrieve_attempts", 0) or 0),
        generate_attempts=int(final_state.get("generate_attempts", 0) or 0),
        sources=sources,
    )


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
