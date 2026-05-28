import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)


persist_directory = "./chroma_db"


def build_retriever():

    docs = [
        Document(page_content="LangChain is a framework for building LLM apps."),
        Document(page_content="Chroma is a local vector database for AI applications."),
        Document(page_content="BM25 is a keyword-based ranking algorithm."),
        Document(page_content="Hybrid search combines semantic and keyword search."),
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
        cache_folder="./model_cache",
    )

    rerank_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    compressor = CrossEncoderReranker(model=rerank_model, top_n=3)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    if os.path.exists(persist_directory):
        print("Loading vector indexes")
        vector_store = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )
    else:
        print("Building new vector indexes")
        vector_store = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
        )

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    bm25_retriever = BM25Retriever.from_documents(docs)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )
    return reranked_retriever
