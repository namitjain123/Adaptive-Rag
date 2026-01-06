# Adaptive RAG System (LangGraph + FastAPI + Next.js)

A production-style **Adaptive Retrieval-Augmented Generation (RAG)** system built with **LangGraph** that dynamically routes queries between:
- **Vectorstore retrieval (ChromaDB)** for knowledge inside your index
- **Web search (Tavily)** for fresh / recency queries ("latest", "today", "news")

It also includes:
- **Self-corrective loops** (query rewriting + retry)
- **Document relevance grading**
- **Answer grounding/usefulness checks**
- **Trace panel** (route used + attempts + source previews)
- **SQLite logging** of chat runs (optional; can be disabled if DB not set)

---

## Features

### ✅ Adaptive Routing
Routes each query to:
- `vectorstore` → if likely answer exists in indexed docs
- `web_search` → if query needs fresh info / outside-index info

Routing uses:
1) **Heuristic router** (fast, deterministic)
2) **LLM router** (fallback, flexible)

---

### ✅ Self-correcting Retrieval Loop (Vectorstore path)
If retrieval returns weak/irrelevant docs:
- grade docs
- **rewrite the query**
- retry retrieval (bounded attempts)

---

### ✅ Grounding Checks (Generation path)
After generation, system checks:
- Is the answer **supported by sources**? (grounded)
- Is it **useful** for the question?

If not:
- retry generation OR rewrite query and retrieve again

---

## Tech Stack

**Backend**
- FastAPI (API)
- LangGraph (workflow)
- LangChain (LLM + tools)
- ChromaDB (vector store)
- HuggingFace embeddings (free local embeddings)
- Tavily (web search)
- SQLite + SQLAlchemy (logging)

**Frontend**
- Next.js (App Router)
- Simple chat UI + routing trace panel

---

## Project Structure
backend/
app/
api/ # FastAPI routes (/chat, /health)
graph/ # LangGraph workflow + nodes
rag/ # retriever, web_search, graders, rewriter, generator
db/ # SQLAlchemy models + session
config.py # Pydantic settings (env config)
main.py # FastAPI app entrypoint
scripts/
build_index.py # Chunk + embed + store docs in chroma_db/
data/ # local sqlite db (if enabled)
chroma_db/ # persisted Chroma store (vector index)
frontend/
app/
page.tsx
components/
ChatBox.tsx
Message.tsx
TracePanel.tsx
lib/
api.ts # frontend -> backend call



## Setup

### 1) Backend setup
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

) Environment variables

Create backend/.env based on .env.example

Example:

# LLM (Groq)
GROQ_API_KEY=your_groq_key

# Web Search (Tavily)
TAVILY_API_KEY=your_tavily_key

# Optional toggles
LANGCHAIN_TRACING_V2=false
ANONYMIZED_TELEMETRY=false
ENVIRONMENT=dev

3) Build the vector index (Chroma)
python -m scripts.build_index

4) Run backend
uvicorn app.main:app --reload --port 8000


Verify:

http://127.0.0.1:8000/health

5) Frontend setup
cd ../frontend
npm install
npm run dev


Open:

http://localhost:3000


API
POST /chat

Request:

{ "question": "What are the types of agent memory?" }


Response:

{
  "answer": "...",
  "route": "vectorstore",
  "route_reason": "...",
  "retrieve_attempts": 1,
  "generate_attempts": 1,
  "sources": [{ "preview": "...", "metadata": {...} }]
}

GET /health

Response:

{ "status": "ok" }


Adaptive RAG System (LangGraph, FastAPI, Groq, Chroma)
• Designed a stateful RAG pipeline with dynamic query routing between web search and vector databases
• Implemented self-corrective retrieval using document relevance grading and query rewriting
• Added hallucination detection and answer-usefulness validation to prevent unsupported responses
• Built retry-aware control flow with bounded loops to ensure robustness and cost control
• Exposed the system via FastAPI and built a Streamlit UI with source transparency and routing explanations