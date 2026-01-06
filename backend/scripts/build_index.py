import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
os.environ.setdefault("USER_AGENT", "adaptive-rag-system/1.0")

URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

def main():
    # Optional (just removes the warning you saw)
    os.environ.setdefault("USER_AGENT", "adaptive-rag-system/1.0")

    print("Building index into:", settings.chroma_persist_dir)
    print("Collection:", settings.chroma_collection)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    docs_nested = [WebBaseLoader(url).load() for url in URLS]
    docs = [d for sub in docs_nested for d in sub]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_persist_dir,
    )

    print("✅ Index built. Chunks:", len(splits))


if __name__ == "__main__":
    main()
