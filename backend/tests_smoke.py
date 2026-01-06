from app.config import settings
from app.rag.retriever import retrieve_documents
from app.rag.graders import build_retrieval_grader
from dotenv import load_dotenv
load_dotenv()


def main():
    print("App:", settings.app_name)

    # Test embeddings + Chroma retriever wiring
    try:
        docs = retrieve_documents("agent memory", k=2)
        print("Retrieved docs:", len(docs))
        if docs:
            print("Doc preview:", docs[0].page_content[:200])
    except Exception as e:
        print("Retriever failed:", repr(e))

    # Test grader wiring (needs GROQ_API_KEY)
    try:
        grader = build_retrieval_grader()
        result = grader.invoke({"question": "agent memory", "document": "This discusses agent memory and long-term memory systems."})
        print("Grader result:", result)
    except Exception as e:
        print("Grader failed:", repr(e))

if __name__ == "__main__":
    main()
