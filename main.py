from src.ingestion import ingest
from src.graph import graph
import sys

from src.retriever import get_retriever


def run_query(question: str):
    result = graph.invoke(
        {
            "question": question,
            "generation": "",
            "documents": [],
            "relevance_score": "",
            "hallucination_score": "",
            "answer_score": "",
            "retries": 0,
        }
    )
    print(result)
    print(result["generation"])


def run_debug(question: str):

    docs = get_retriever().invoke(question)
    print(f"Retrieved {len(docs)} docs:")
    for i, doc in enumerate(docs):
        print(f"\n--- Doc {i + 1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content: {doc.page_content}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run main.py ingest | query '<question>'")
        sys.exit(1)

    command = sys.argv[1]
    if command == "query":
        run_query(sys.argv[2])
    elif command == "debug":
        run_debug(sys.argv[2])
