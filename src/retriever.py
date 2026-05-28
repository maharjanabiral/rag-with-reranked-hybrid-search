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
rerank_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

compressor = CrossEncoderReranker(model=rerank_model, top_n=3)


def build_retriever():

    vector_store = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )
    raw_docs = vector_store.get()
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
    ]

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    bm25_retriever = BM25Retriever.from_documents(docs)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever
    )
    return reranked_retriever


retriever = build_retriever()
