from pathlib import Path
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_dir = Path(__file__).resolve().parent.parent
persist_directory = base_dir / "chroma_db"
data_directory = base_dir / "data"


def ingest():
    data_path = Path(data_directory)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")

    files = list(data_path.glob("*.pdf"))
    all_docs = []

    if not files:
        raise ValueError("PDF files not found")

    for file in files:
        print(f"Ingesting {file.name}")

        loader = PyMuPDFLoader(str(file))
        docs = loader.load()
        all_docs.extend(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./model_cache",
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,  # good index quality
            "hnsw:search_ef": 100,  # good search recall
            "hnsw:M": 32,  # more connections = better recall
        },
    )
    print("Ingestion Complete")


if __name__ == "__main__":
    ingest()
