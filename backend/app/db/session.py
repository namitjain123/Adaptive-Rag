from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

# SQLite needs check_same_thread=False for FastAPI dev reload / multi-threading
connect_args = {"check_same_thread": False} if settings.db_url.startswith("sqlite") else {}

engine = create_engine(settings.db_url, echo=False, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
