from __future__ import annotations

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document

from app.config import settings


def build_web_search_tool(k: int = 3) -> TavilySearchResults:
    """
    Tavily web search wrapper.

    We keep it in its own module so:
    - swapping search providers is easy later
    - the graph logic stays clean
    """
    # Tavily uses env var TAVILY_API_KEY automatically.
    # We keep settings.tavily_api_key so the app config is explicit/documented.
    return TavilySearchResults(k=k)


def web_search(query: str, k: int = 3) -> List[Document]:
    """
    Runs web search and returns results as Documents so the rest of the pipeline
    can treat web results and vectorstore results uniformly.
    """
    if not settings.tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set. Add it to backend/.env")

    tool = build_web_search_tool(k=k)
    results = tool.invoke({"query": query})

    # Tavily returns list of dicts (content, url, etc.)
    # Wrap each result as a Document.
    docs: List[Document] = []
    for r in results:
        content = r.get("content", "")
        url = r.get("url", "")
        title = r.get("title", "")

        page = f"{title}\n{url}\n\n{content}".strip()
        docs.append(Document(page_content=page, metadata={"source": "tavily", "url": url, "title": title}))

    return docs
