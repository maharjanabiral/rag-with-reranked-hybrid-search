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
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
    cache_folder="./model_cache",
)
rerank_model = HuggingFaceCrossEncoder(model_name="cross-encoder/nli-deberta-v3-small")

compressor = CrossEncoderReranker(model=rerank_model, top_n=5)

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def build_retriever():

    vector_store = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    raw_docs = vector_store.get()
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
    ]

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 8

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.6, 0.4]
    )

    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )
    return reranked_retriever
