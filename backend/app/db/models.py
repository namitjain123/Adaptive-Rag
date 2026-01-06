from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatRun(Base):
    """
    Stores one completed graph run (one user question -> one system answer).

    We store sources as JSON string for simplicity in an MVP.
    In production, you might normalize sources into a separate table.
    """
    __tablename__ = "chat_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    original_question: Mapped[str] = mapped_column(Text, nullable=False)
    final_question: Mapped[str] = mapped_column(Text, nullable=False)

    answer: Mapped[str] = mapped_column(Text, nullable=False)

    route: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    route_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    retrieve_attempts: Mapped[int] = mapped_column(Integer, default=0)
    generate_attempts: Mapped[int] = mapped_column(Integer, default=0)

    sources_json: Mapped[str] = mapped_column(Text, default="[]")
