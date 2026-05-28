from langchain_community.document_loaders import (
    PyMuPDFLoader,
    WebBaseLoader,
    JSONLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


persist_directory = "./chroma_db"


def ingest():
    docs = [
        Document(page_content="LangChain is a framework for building LLM apps."),
        Document(page_content="Chroma is a local vector database for AI applications."),
        Document(page_content="BM25 is a keyword-based ranking algorithm."),
        Document(page_content="Hybrid search combines semantic and keyword search."),
    ]

    # loader = PyMuPDFLoader(source)
    # loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./model_cache",
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )
    print("Ingestion Complete")


if __name__ == "__main__":
    ingest()
