from __future__ import annotations
from typing import List

from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


from app.config import settings


def get_embeddings():
    # Runs locally (free). Downloads model first time.
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def retrieve_documents(question: str, k: int = 4) -> List[Document]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)
