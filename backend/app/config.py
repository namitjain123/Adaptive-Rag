from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    app_name: str = "Adaptive RAG System"
    environment: str = "dev"

    # DB
    db_url: str = Field(default="sqlite:///./data/app.db")

    # ✅ Groq
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"  # fast + good
    temperature: float = 0.0

    # Web search
    tavily_api_key: str | None = None

        # Tracing toggle (OFF by default)
    enable_langsmith: bool = Field(default=False)

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "adaptive-rag-system"

    # ✅ Vectorstore
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "rag-chroma"

    # ✅ HuggingFace embeddings (local)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
