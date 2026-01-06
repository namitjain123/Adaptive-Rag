from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.config import settings
from app.db.init_db import init_db
from dotenv import load_dotenv
load_dotenv()


def configure_tracing():
    if settings.enable_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGCHAIN_PROJECT", None)


configure_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure ./data exists for sqlite
    if settings.db_url.startswith("sqlite:///./data/"):
        os.makedirs("./data", exist_ok=True)

    # 🔥 CREATE TABLES HERE
    init_db()

    yield

    # optional shutdown cleanup


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/")
def root():
    return {"app": settings.app_name, "env": settings.environment}
